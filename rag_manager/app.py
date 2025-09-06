"""
Main application class for RAG Knowledgebase Manager.

This module contains the main RAGKnowledgebaseManager class that orchestrates
document processing and web interface components.
"""

import os
import shutil
import time
import hashlib
import json
import statistics
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from flask import Flask
from langchain_core.documents import Document
from urllib.robotparser import RobotFileParser

from .core.config import Config
from .core.models import DocumentProcessingStatus, URLProcessingStatus, EmailProcessingStatus
from .core.ollama_health import OllamaHealthChecker
from .utils.logger import setup_logging
from ingestion.document.processor import DocumentProcessor
from .managers.milvus_manager import MilvusManager
from .web.routes import WebRoutes
from .scheduler_manager import SchedulerManager  # Import SchedulerManager
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

# Logging will be set up by the application initialization

logger = logging.getLogger(__name__)


class RAGKnowledgebaseManager:
    """
    Main application class that orchestrates document processing and web interface.
    
    This class follows the development rules with proper separation of concerns,
    dependency injection, and comprehensive logging.
    """
    
    def __init__(self) -> None:
        """Initialize the RAG Knowledgebase Manager application."""
        self.config = Config()
        
        # Setup logging with configured log directory
        setup_logging(self.config.LOG_DIR)
        
        # Get the project root directory (parent of rag_manager)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_folder = os.path.join(project_root, 'templates')
        static_folder = os.path.join(project_root, 'static')
        
        self.app = Flask(__name__, 
                        template_folder=template_folder,
                        static_folder=static_folder)
        self.app.config['SECRET_KEY'] = self.config.SECRET_KEY
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.MAX_CONTENT_LENGTH

        # Initialize core components
        self.document_processor = DocumentProcessor(self.config)
        self.milvus_manager = MilvusManager(self.config)
        
        # Initialize Ollama health checker
        self.ollama_health = OllamaHealthChecker(self.config)
        
        # Initialize Milvus collections during application startup
        logger.info("Initializing Milvus collections during application startup...")
        self.milvus_manager.initialize_collections_for_startup()
        logger.info("Milvus collections initialized successfully")
        
        # Initialize PostgreSQL manager for metadata
        self._initialize_database_managers()
        
        # Initialize document manager for metadata operations
        self._initialize_document_manager()
        
        # Initialize URL manager based on feature flags
        self._initialize_url_manager()
        
        # Initialize URL orchestrator
        self._initialize_url_orchestrator()
        
        # Processing status tracking
        self.processing_status: Dict[str, DocumentProcessingStatus] = {}
        self.url_processing_status: Dict[str, URLProcessingStatus] = {}  # Changed to str for UUID keys
        self.email_processing_status: Dict[int, EmailProcessingStatus] = {}
        
        # Initialize the SchedulerManager (but don't start it yet)
        self.scheduler_manager = SchedulerManager(self.url_manager, self.config)
        
        # Configure scheduler with processing status dictionaries and background methods
        self.scheduler_manager.set_processing_status(
            self.url_processing_status, 
            self.email_processing_status
        )
        self.scheduler_manager.set_background_processors(
            self._process_url_background,  # Now we have URL processing implementation
            self._refresh_email_account_background
        )
        # Initialize email manager if enabled
        self._initialize_email_manager()

        # Scheduler thread handle
        self._scheduler_thread = None  # type: Optional[threading.Thread]

        # Crawler state: robots cache and per-domain last-request timestamps
        self._robots_cache: Dict[str, Any] = {}
        self._domain_last_request: Dict[str, float] = {}
        # Use configured user-agent for robots and fetching
        self.crawler_user_agent = self.config.CRAWL_USER_AGENT

        # Setup directories and routes
        self._setup_directories()

        # Initialize web routes
        self.web_routes = WebRoutes(self.app, self.config, self)
        
        logger.info("RAG Knowledgebase Manager application initialized")

    def _initialize_database_managers(self) -> None:
        """Initialize PostgreSQL database managers."""
        try:
            from rag_manager.managers.postgres_manager import PostgreSQLManager, PostgreSQLConfig
            from ingestion.core.database_manager import RAGDatabaseManager
            
            postgres_config = PostgreSQLConfig(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                database=self.config.POSTGRES_DB,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD
            )
            self.postgres_manager = PostgreSQLManager(postgres_config)
            self.database_manager = RAGDatabaseManager(postgres_config)
            logger.info("PostgreSQL integration initialized successfully")
            
            # Set PostgreSQL manager on MilvusManager for document retrieval
            if hasattr(self, 'milvus_manager') and self.milvus_manager:
                self.milvus_manager.set_postgres_manager(self.postgres_manager)
                logger.info("Document retrieval initialized")
            
            # Initialize retrieval system for email search
            # 
            # WHAT THIS DOES:
            # This initializes our advanced email search system that combines two retrieval methods:
            # 1. Vector Similarity Search (Milvus) - finds emails with similar semantic meaning
            # 2. Full-Text Search (PostgreSQL FTS) - finds emails with exact keyword matches
            # 
            # WHY WE NEED BOTH:
            # - Vector search is great for: "find emails about project updates" (semantic understanding)
            # - FTS search is great for: "find emails from john@company.com" (exact matches)
            # - Hybrid combines both using Reciprocal Rank Fusion (RRF) for better results
            #
            # COMPONENTS INITIALIZED:
            # - PostgresFTSRetriever: Searches email_chunks table using PostgreSQL's ts_query/ts_rank
            # - HybridRetriever: Combines vector + FTS results using RRF algorithm
            # - Integrated into EmailManager for clean email-specific architecture
            #
            # DEPENDENCIES:
            # - Requires PostgreSQL pool for FTS operations
            # - Requires email_chunks table with GIN indexes for fast text search
            # - Requires emails collection in Milvus for vector similarity search
            #
            # FAILURE HANDLING:
            # - If initialization fails, email search falls back to vector-only mode
            # - Error is logged but doesn't crash the application startup
            # NOTE: Hybrid retrieval is now initialized in _initialize_email_manager() 
            # to ensure proper initialization order
            
        except ImportError as e:
            logger.error(f"Failed to import PostgreSQL managers: {e}")
            self.postgres_manager = None
            self.database_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            self.postgres_manager = None
            self.database_manager = None

    def _initialize_document_manager(self) -> None:
        """Initialize document manager for metadata operations."""
        try:
            from ingestion.document.manager import DocumentManager
            if self.postgres_manager:
                self.document_manager = DocumentManager(self.postgres_manager)
                logger.info("Document Manager initialized successfully")
            else:
                logger.error("Cannot initialize document manager: PostgreSQL manager not available")
                self.document_manager = None
        except ImportError as e:
            logger.error(f"Failed to import Document Manager: {e}")
            self.document_manager = None

    def _initialize_url_manager(self) -> None:
        """Initialize URL manager based on feature flags."""
        logger.info("Initializing URL Source Manager")
        try:
            from ingestion.url.source_manager import URLSourceManager
            if self.postgres_manager:
                self.url_manager = URLSourceManager(self.postgres_manager, self.milvus_manager)
                logger.info("URL Source Manager initialized successfully")
            else:
                logger.error("Cannot initialize URL manager: PostgreSQL manager not available")
                self.url_manager = None
        except ImportError as e:
            logger.error(f"Failed to import URL Source Manager: {e}")
            self.url_manager = None

    def _initialize_url_orchestrator(self) -> None:
        """Initialize URL orchestrator for managing URL processing tasks."""
        try:
            logger.info("Initializing URL orchestrator")
            
            # Validate prerequisites
            if not self.url_manager:
                logger.error("Cannot initialize URL orchestrator: URL manager not available")
                logger.error(f"URL manager state: {self.url_manager}")
                logger.error(f"PostgreSQL manager state: {self.postgres_manager}")
                self.url_orchestrator = None
                return
            
            logger.info("URL manager available, proceeding with orchestrator initialization")
            
            # Initialize snapshot service
            snapshot_service = None
            try:
                from ingestion.url.utils.snapshot_service import URLSnapshotService
                snapshot_service = URLSnapshotService(self.postgres_manager, self.config, self.url_manager)
                logger.info("URL snapshot service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize snapshot service: {e}")
                # Continue without snapshot service if it fails
            
            # Initialize URL orchestrator
            logger.info("Importing URLOrchestrator")
            from ingestion.url.orchestrator import URLOrchestrator
            logger.info("Creating URLOrchestrator instance")
            self.url_orchestrator = URLOrchestrator(
                config=self.config,
                url_manager=self.url_manager,
                processor=self.document_processor,  # Use existing document processor for URL content
                milvus_manager=self.milvus_manager,  # Add MilvusManager for vector storage
                snapshot_service=snapshot_service,  # Add snapshot service for PDF generation
                postgres_manager=self.postgres_manager
            )
            logger.info("URL orchestrator initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize URL orchestrator: %s", e, exc_info=True)
            self.url_orchestrator = None

    def _initialize_email_manager(self) -> None:
        """Initialize email management system with PostgreSQL integration."""
        try:
            logger.info("Initializing PostgreSQL-based email management system")
            
            # Validate prerequisites
            if not self.postgres_manager:
                logger.error("Cannot initialize email manager: PostgreSQL manager not available")
                self.email_account_manager = None
                self.email_orchestrator = None
                return
            
            # Initialize PostgreSQL email account manager (stats from PostgreSQL, not Milvus)
            from ingestion.email.account_manager import EmailAccountManager
            self.email_account_manager = EmailAccountManager(self.postgres_manager)
            logger.info("PostgreSQL email account manager initialized with PostgreSQL-based statistics")
            
            # Ensure email vector store is ready (separate from document vector store)
            email_vector_store = self.milvus_manager.get_email_vector_store()
            logger.info("Email vector store validated for email embeddings")
            
            # Create PostgreSQL-based email message manager
            from ingestion.email.account_manager import EmailAccountManager as EmailMessageManager
            email_message_manager = EmailMessageManager(self.postgres_manager)
            logger.info("PostgreSQL email message manager initialized")
            
            # Inject dependencies into EmailProcessor
            from ingestion.email.processor import EmailProcessor
            email_processor = EmailProcessor(
                milvus=email_vector_store,  # Use dedicated email vector store
                email_manager=email_message_manager,  # PostgreSQL email manager
                embedding_model=self.milvus_manager.langchain_embeddings
            )
            logger.info("Email processor initialized with PostgreSQL backend and dedicated vector store")
            
            # Inject dependencies into EmailOrchestrator
            from ingestion.email.orchestrator import EmailOrchestrator
            self.email_orchestrator = EmailOrchestrator(
                config=self.config,
                account_manager=self.email_account_manager,
                processor=email_processor
            )
            logger.info("Email orchestrator initialized")
            
            # Set the email orchestrator in the scheduler manager
            if hasattr(self, 'scheduler_manager') and self.scheduler_manager:
                self.scheduler_manager.set_email_orchestrator(self.email_orchestrator)
            logger.info("Email orchestrator set in scheduler manager")
            
            # Initialize email retrieval system (clean architecture)
            try:
                email_vector_store = self.milvus_manager.get_email_vector_store()
                self.email_account_manager.initialize_hybrid_retrieval(email_vector_store)
                logger.info("Hybrid email retrieval system initialized successfully in EmailManager")
            except Exception as e:
                logger.error(f"Failed to initialize retrieval in EmailManager: {e}")
                logger.warning("Email search will fall back to vector-only mode")
            
        except Exception as e:
            logger.error("Failed to initialize email manager: %s", e)
            self.email_account_manager = None
            self.email_orchestrator = None

    def _setup_directories(self) -> None:
        """Create necessary directories for file management."""
        directories = [
            self.config.UPLOAD_FOLDER,
            self.config.UPLOADED_FOLDER,
            self.config.DELETED_FOLDER,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")

    def _scheduler_status(self) -> Dict[str, Any]:
        """
        Get scheduler status information.
        
        Returns:
            Dictionary containing scheduler status and statistics
        """
        return {
            "thread_active": self._scheduler_thread is not None and self._scheduler_thread.is_alive(),
            "processing_files": len(self.processing_status),
            "processing_urls": len(self.url_processing_status),
            "processing_emails": len(self.email_processing_status),
            "timestamp": datetime.now().isoformat()
        }

    def _get_robots(self, url: str) -> tuple:
        """
        Get robots.txt parser for a domain with caching.
        
        Args:
            url: URL to get robots.txt for
            
        Returns:
            Tuple of (RobotFileParser, crawl_delay)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if not domain:
                raise ValueError("invalid URL")

            if not hasattr(self, "_robots_cache"):
                self._robots_cache = {}

            if domain in self._robots_cache:
                return self._robots_cache[domain]

            robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception as read_err:
                # Treat unreadable robots as permissive
                logger.debug(f"robots.txt read failed for {robots_url}: {read_err}")
                rp = RobotFileParser()
                rp.parse([])  # empty, permissive

            # Prefer configured User-Agent
            user_agent = getattr(self, "crawler_user_agent", None) or "*"
            crawl_delay = rp.crawl_delay(user_agent)
            if crawl_delay is None:
                crawl_delay = rp.crawl_delay("*")
            if crawl_delay is None:
                # Use configured default crawl delay
                try:
                    crawl_delay = float(getattr(self.config, "CRAWL_DELAY_SECONDS", 30.0))
                except Exception:
                    crawl_delay = 30.0

            self._robots_cache[domain] = (rp, crawl_delay)
            return self._robots_cache[domain]

        except Exception as e:
            logger.debug(f"Robots.txt parsing failed for {url}: {e}")
            # Return permissive defaults
            rp = RobotFileParser()
            rp.parse([])
            return (rp, 1)

    def check_robots_allowed(self, url: str) -> bool:
        """
        Check if a URL is allowed by robots.txt.
        
        Args:
            url: The URL to check
            
        Returns:
            True if allowed, False if disallowed
        """
        try:
            rp, _ = self._get_robots(url)
            user_agent = getattr(self, "crawler_user_agent", None) or "*"
            return rp.can_fetch(user_agent, url)
        except Exception as e:
            logger.debug(f"Robots.txt check failed for {url}: {e}")
            # Return True (allowed) if check fails
            return True

    def _process_url_background(self, url_id: str) -> None:
        """
        Process a URL in a background thread using the URL orchestrator.
        
        Args:
            url_id: The UUID string ID of the URL to process
        """
        from datetime import datetime
        from .core.models import URLProcessingStatus
        
        # Validate dependencies
        if not self.url_orchestrator:
            logger.error("URL orchestrator not initialized")
            return
            
        if not self.url_manager:
            logger.error("URL manager not initialized")
            return
            
        # Initialize processing status
        try:
            # Get URL details for status tracking
            url_record = self.url_manager.get_url_by_id(url_id)
            
            if not url_record:
                # URL might be newly added from domain crawling, give it a moment
                logger.info(f"URL {url_id} not found, waiting briefly for potential new URL...")
                import time
                time.sleep(1)  # Brief pause for database consistency
                url_record = self.url_manager.get_url_by_id(url_id)
                
                if not url_record:
                    logger.error(f"URL {url_id} not found even after retry")
                    return
                
            # Create status tracker
            status = URLProcessingStatus(
                url=url_record.get('url', ''),
                title=url_record.get('title', '')
            )
            status.status = "processing"
            status.message = "Starting URL processing..."
            status.progress = 5
            status.start_time = datetime.now()
            
            self.url_processing_status[url_id] = status
            
            logger.info(f"Starting URL processing for ID {url_id}: {url_record.get('url', '')}")
            
            # Process URL using the orchestrator
            status.message = "Processing URL content..."
            status.progress = 25
            
            # Run the async process_url method
            import asyncio
            result = asyncio.run(self.url_orchestrator.process_url(url_id))
            
            # Complete successfully or with error
            if result.get("success"):
                status.status = "completed"
                status.message = "URL processing completed successfully"
                status.progress = 100
                status.end_time = datetime.now()
                logger.info(f"URL processing completed for ID {url_id}")
            else:
                status.status = "error"
                status.message = f"Processing failed: {result.get('error', 'Unknown error')}"
                status.error_details = result.get('error')
                status.progress = 100
                status.end_time = datetime.now()
                logger.error(f"URL processing failed for ID {url_id}: {result.get('error')}")
            
        except Exception as exc:
            logger.exception(f"URL processing failed for ID {url_id}")
            
            # Update status with error
            status = self.url_processing_status.get(url_id)
            if status:
                status.status = "error"
                status.message = f"Processing failed: {str(exc)}"
                status.error_details = str(exc)
                status.progress = 100
                status.end_time = datetime.now()
            
            raise
            
        finally:
            # Clean up processing status after a delay to allow UI to see completion
            def cleanup_status():
                import time
                time.sleep(5)  # Allow 5 seconds for UI to see final status
                try:
                    if url_id in self.url_processing_status:
                        del self.url_processing_status[url_id]
                        logger.debug(f"Cleaned up processing status for URL {url_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup status for URL {url_id}: {e}")
            
            # Run cleanup in background thread
            cleanup_thread = threading.Thread(target=cleanup_status, daemon=True)
            cleanup_thread.start()

    def _delete_url_background(self, url_id: str) -> None:
        """
        Delete a URL and its embeddings in a background thread.
        
        Args:
            url_id: The UUID string ID of the URL to delete
        """
        from datetime import datetime
        from .core.models import URLProcessingStatus
        
        logger.info(f"Starting URL deletion background process for ID: {url_id}")
        
        # Initialize processing status
        try:
            # Get URL details for status tracking
            if not self.url_manager:
                logger.error("URL manager not initialized")
                return
            url_record = self.url_manager.get_url_by_id(url_id)
            
            if not url_record:
                logger.error(f"URL {url_id} not found for deletion")
                return
                
            # Create status tracker
            status = URLProcessingStatus(
                url=url_record.get('url', ''),
                title=url_record.get('title', '')
            )
            status.status = "processing"
            status.message = "Starting URL deletion..."
            status.progress = 10
            status.start_time = datetime.now()
            
            self.url_processing_status[url_id] = status
            
            logger.info(f"Starting URL deletion for ID {url_id}: {url_record.get('url', '')}")
            
            # Delete from URL manager (includes documents and snapshot cleanup)
            status.message = "Deleting URL and associated data..."
            status.progress = 50
            
            result = self.url_manager.delete_url(url_id)
            
            # Additional Milvus cleanup since URL manager can't access it directly
            if result.get("success"):
                try:
                    status.message = "Cleaning up vector embeddings..."
                    status.progress = 70
                    
                    if self.milvus_manager and self.postgres_manager and url_record.get('url'):
                        url_string = url_record['url']
                        logger.info(f"Cleaning up Milvus embeddings for URL: {url_string}")
                        
                        # Find document IDs for this URL from PostgreSQL
                        with self.postgres_manager.get_connection() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "SELECT id FROM documents WHERE document_type = 'url' AND file_path LIKE %s",
                                    (f"%{url_string}%",)
                                )
                                doc_records = cur.fetchall()
                                
                                total_deleted = 0
                                for doc in doc_records:
                                    doc_id = str(doc['id'])
                                    try:
                                        delete_result = self.milvus_manager.delete_document(document_id=doc_id)
                                        if delete_result.get('success'):
                                            deleted_count = delete_result.get('deleted_count', 0)
                                            total_deleted += deleted_count
                                            logger.debug(f"Deleted {deleted_count} embeddings for document {doc_id}")
                                    except Exception as e:
                                        logger.error(f"Failed to delete embeddings for document {doc_id}: {e}")
                                
                                logger.info(f"Deleted {total_deleted} total embeddings from Milvus for URL: {url_string}")
                    else:
                        logger.warning("Milvus manager not available or URL missing for embedding cleanup")
                except Exception as e:
                    logger.error(f"Failed to clean up Milvus embeddings for URL {url_record.get('url', '')}: {e}")
                    # Don't fail the deletion if Milvus cleanup fails
            
            # Complete successfully or with error
            if result.get("success"):
                status.status = "completed"
                status.message = "URL deletion completed successfully"
                status.progress = 100
                status.end_time = datetime.now()
                logger.info(f"URL deletion completed for ID {url_id}")
            else:
                status.status = "error"
                status.message = f"Deletion failed: {result.get('message', 'Unknown error')}"
                status.error_details = result.get('message')
                status.progress = 100
                status.end_time = datetime.now()
                logger.error(f"URL deletion failed for ID {url_id}: {result.get('message')}")
            
        except Exception as exc:
            logger.exception(f"URL deletion failed for ID {url_id}")
            
            # Update status with error
            status = self.url_processing_status.get(url_id)
            if status:
                status.status = "error"
                status.message = f"Deletion failed: {str(exc)}"
                status.error_details = str(exc)
                status.progress = 100
                status.end_time = datetime.now()
            
            raise
            
        finally:
            # Clean up processing status after a delay to allow UI to see completion
            def cleanup_status():
                import time
                time.sleep(5)  # Allow 5 seconds for UI to see final status
                try:
                    if url_id in self.url_processing_status:
                        del self.url_processing_status[url_id]
                        logger.debug(f"Cleaned up processing status for URL {url_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup status for URL {url_id}: {e}")
            
            # Run cleanup in background thread
            cleanup_thread = threading.Thread(target=cleanup_status, daemon=True)
            cleanup_thread.start()

    def _refresh_email_account_background(self, account_id: int) -> None:
        """
        Fetch emails for a specific account in a background thread using PostgreSQL.
        
        Args:
            account_id: The ID of the email account to process
        """
        from datetime import datetime
        from .core.models import EmailProcessingStatus
        
        # Validate dependencies
        if not self.email_account_manager:
            logger.error("Email account manager not initialized")
            return
            
        if not self.email_orchestrator:
            logger.error("Email orchestrator not initialized")
            return
        
        # Initialize processing status
        try:
            # Get account details for status tracking
            accounts = self.email_account_manager.list_accounts(include_password=False)
            account = next((acc for acc in accounts if acc.get("id") == account_id), None)
            
            if not account:
                logger.error(f"Account {account_id} not found")
                return
                
            # Create status tracker
            status = EmailProcessingStatus(
                email_id=f"account_{account_id}",
                sender=account.get('email_address', ''),
                subject=f"Account refresh: {account.get('account_name', '')}"
            )
            status.status = "processing"
            status.message = "Starting email refresh..."
            status.progress = 5
            status.start_time = datetime.now()
            
            self.email_processing_status[account_id] = status
            
            logger.info(f"Starting email refresh for account {account_id}: {account.get('account_name', '')}")
            
            # Process emails using the orchestrator
            status.message = "Processing emails..."
            status.progress = 25
            
            self.email_orchestrator.run(account_id=account_id)
            
            # Complete successfully
            status.status = "completed"
            status.message = "Email refresh completed successfully"
            status.progress = 100
            status.end_time = datetime.now()
            
            logger.info(f"Email refresh completed for account {account_id}")
            
        except Exception as exc:
            logger.exception(f"Email refresh failed for account {account_id}")
            
            # Update status with error
            status = self.email_processing_status.get(account_id)
            if status:
                status.status = "error"
                status.message = f"Refresh failed: {str(exc)}"
                status.error_details = str(exc)
                status.progress = 100
                status.end_time = datetime.now()
            
            raise
            
        finally:
            # Clean up processing status after a delay to allow UI to see completion
            def cleanup_status():
                import time
                time.sleep(5)  # Allow 5 seconds for UI to see final status
                try:
                    if account_id in self.email_processing_status:
                        del self.email_processing_status[account_id]
                        logger.debug(f"Cleaned up processing status for account {account_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup status for account {account_id}: {e}")
            
            # Run cleanup in background thread
            cleanup_thread = threading.Thread(target=cleanup_status, daemon=True)
            cleanup_thread.start()

    def _fallback_title_from_filename(self, filename: str) -> str:
        """Generate a title from filename using basic text processing."""
        base = os.path.splitext(filename)[0]
        base = base.replace('_', ' ').replace('-', ' ').strip()
        if not base:
            return filename
        words = []
        for w in base.split():
            if len(w) > 3 and w.isupper():
                words.append(w)
            else:
                words.append(w.capitalize())
        return ' '.join(words)

    def _get_content_type_from_filename(self, filename: str) -> str:
        """Determine content type from file extension."""
        ext = os.path.splitext(filename)[1].lower()
        content_type_map = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.rtf': 'application/rtf'
        }
        return content_type_map.get(ext, 'application/octet-stream')

    def _process_document_background(self, filename: str) -> None:
        """
        Process a document in the background with progress tracking.
        
        Args:
            filename: Name of the file to process
        """
        status = self.processing_status[filename]
        status.status = "processing"
        status.start_time = datetime.now()
        status.progress = 10
        status.message = "Starting document processing..."
        
        try:
            file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
            status.message = "Loading and chunking document..."
            status.progress = 30
            
            logger.info(f"Processing document '{filename}'")
            
            # Step 1: Get existing document UUID from upload step
            if not self.postgres_manager:
                raise Exception("PostgreSQL manager not available - cannot process document")
            
            # Find existing document record by filename (created during upload)
            logger.debug(f"Finding existing document record for: {filename}")
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
                    result = cur.fetchone()
                    if not result:
                        raise Exception(f"No document record found for filename: {filename}")
                    document_id = str(result['id'])
                    logger.info(f"Found existing document with ID: {document_id}")
            
            # Step 2: Load and chunk document with proper UUID
            logger.debug(f"Starting document loading and chunking for: {filename}")
            chunks = self.document_processor.load_and_chunk(file_path, filename, document_id)
            status.chunks_count = len(chunks)
            logger.info(f"Document '{filename}' chunked into {len(chunks)} chunks")
            
            # Log chunk details for debugging
            if chunks:
                total_chars = sum(len(chunk.page_content or '') for chunk in chunks)
                avg_chars = total_chars // len(chunks)
                logger.debug(f"Chunk statistics: total_chars={total_chars}, avg_chars={avg_chars}")
            else:
                logger.warning(f"No chunks extracted from document: {filename}")
            
            # Initialize with filename-based title as fallback
            status.title = self._fallback_title_from_filename(filename)
            
            # Keyword extraction (hybrid)
            status.status = "embedding"
            status.message = "Extracting keywords..."
            status.progress = 55
            
            logger.debug(f"Starting keyword extraction for: {filename}")
            
            # Check Ollama classification service for keyword extraction
            try:
                classification_status = self.ollama_health.check_classification_service()
                if not classification_status.connected:
                    logger.warning(f"Ollama classification service unavailable for keyword extraction: {classification_status.error_message}")
                    # Provide fallback minimal keywords
                    kw_data = {'global_keywords': [], 'llm_title': None}
                    logger.info(f"Using fallback keyword extraction for {filename}")
                elif not classification_status.model_loaded:
                    logger.warning(f"Classification model '{classification_status.model_name}' not loaded for keyword extraction")
                    kw_data = {'global_keywords': [], 'llm_title': None}
                    logger.info(f"Using fallback keyword extraction for {filename}")
                else:
                    logger.info(f"Using Ollama classification service for keyword extraction (model: {classification_status.model_name})")
                    kw_data = self.document_processor.extract_keywords(chunks)
            except Exception as e:
                logger.warning(f"Keyword extraction failed for {filename}: {e}")
                kw_data = {'global_keywords': [], 'llm_title': None}
                logger.info(f"Using fallback keyword extraction for {filename}")
            
            global_keywords = kw_data['global_keywords']
            logger.info(f"Extracted {len(global_keywords)} global keywords for document: {filename}")
            
            # Prefer LLM-derived title if reasonable
            llm_title = kw_data.get('llm_title')
            if llm_title:
                logger.debug(f"LLM suggested title: '{llm_title}'")
                try:
                    # Basic quality heuristics: length & word count & not all caps
                    wc = len(llm_title.split())
                    if 2 <= wc <= 20 and 8 <= len(llm_title) <= 140 and not llm_title.isupper():
                        status.title = llm_title
                        logger.info(f"Using LLM-derived title: '{llm_title}'")
                    else:
                        logger.debug(f"LLM title rejected (wc={wc}, len={len(llm_title)}, caps={llm_title.isupper()})")
                except Exception as e:
                    logger.warning(f"Error processing LLM title: {e}")
            
            # Attach keywords list to each chunk metadata (global subset for search boost)
            for c in chunks:
                c.metadata['keywords'] = global_keywords
                c.metadata['category_type'] = 'document'  # Set category_type for uploaded documents
            logger.debug(f"Attached keywords and category_type to {len(chunks)} chunks")
            
            # Basic stats
            word_count = sum(len((c.page_content or '').split()) for c in chunks)
            lengths = [len(c.page_content or '') for c in chunks]
            avg_len = float(sum(lengths)/len(lengths)) if lengths else 0.0
            med_len = float(statistics.median(lengths)) if lengths else 0.0
            page_count = max(int(c.metadata.get('page',0)) for c in chunks) + 1 if chunks else 0
            
            logger.info(f"Document stats - Words: {word_count}, Pages: {page_count}, "
                       f"Avg chunk length: {avg_len:.1f}, Median: {med_len:.1f}")
            
            status.status = "embedding"
            status.message = "Enriching metadata via LLM..."
            status.progress = 60
            
            logger.debug(f"Starting topic enrichment for: {filename}")
            
            # Check Ollama classification service before topic enrichment
            try:
                classification_status = self.ollama_health.check_classification_service()
                if not classification_status.connected:
                    logger.warning(f"Ollama classification service unavailable: {classification_status.error_message}")
                    logger.info(f"Skipping topic enrichment for {filename} - classification service unavailable")
                elif not classification_status.model_loaded:
                    logger.warning(f"Classification model '{classification_status.model_name}' not loaded")
                    logger.info(f"Skipping topic enrichment for {filename} - model not available")
                else:
                    logger.info(f"Ollama classification service confirmed available (model: {classification_status.model_name})")
                    chunks = self.document_processor.enrich_topics(chunks)
                    
                    # Log topic statistics for analysis
                    topic_stats = self.document_processor.get_topic_statistics(chunks)
                    logger.info(f"Multi-topic enrichment completed for: {filename}")
                    logger.info(f"Topic statistics: {topic_stats['unique_topics']} unique topics, "
                               f"{topic_stats['coverage_percentage']}% coverage, "
                               f"avg {topic_stats['avg_topics_per_chunk']} topics/chunk")
                    logger.debug(f"Most common topics: {topic_stats['most_common_topics']}")
                    
                    # Log detailed topic distribution for debugging
                    if topic_stats['total_topic_assignments'] > 0:
                        logger.debug(f"Total topic assignments: {topic_stats['total_topic_assignments']}, "
                                   f"chunks without topics: {topic_stats['chunks_without_topics']}")
                        
                        # Log topic source breakdown
                        sources = {}
                        for topic, source_list in topic_stats['topic_sources'].items():
                            for source in source_list:
                                sources[source] = sources.get(source, 0) + 1
                        logger.debug(f"Topic source breakdown: {sources}")
            except Exception as e:
                logger.warning(f"Topic enrichment failed for {filename}: {e}")
                logger.info(f"Continuing document processing without topic enrichment")
            
            status.status = "storing"
            status.message = "Storing in Milvus..."
            status.progress = 80
            
            logger.info(f"Storing {len(chunks)} chunks in Milvus for document: {filename}")
            
            # Check Ollama embedding service before attempting storage
            try:
                # Quick health check for embedding service
                ollama_status = self.ollama_health.check_embedding_service()
                if not ollama_status.connected:
                    error_msg = f"Ollama embedding service unavailable: {ollama_status.error_message}"
                    logger.error(error_msg)
                    status.status = "error"
                    status.message = f"Embedding service unavailable. Please ensure Ollama is running with model '{ollama_status.model_name}'"
                    return
                elif not ollama_status.model_loaded:
                    error_msg = f"Required embedding model '{ollama_status.model_name}' not loaded in Ollama"
                    logger.error(error_msg)
                    status.status = "error"
                    status.message = f"Model '{ollama_status.model_name}' not available in Ollama"
                    return
                
                logger.info(f"Ollama embedding service confirmed available (model: {ollama_status.model_name}, response: {ollama_status.response_time_ms:.1f}ms)")
                
                # Proceed with Milvus storage using ONLY the UUID as document identifier
                # All metadata lives in PostgreSQL, Milvus only stores vectors + UUID reference
                self.milvus_manager.insert_documents(document_id, chunks)
                logger.info(f"Successfully stored document '{filename}' with ID '{document_id}' in Milvus vector database")
                
            except Exception as e:
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    logger.error(f"Ollama embedding service connection error for {filename}: {e}")
                    status.status = "error"
                    status.message = f"Failed to connect to embedding service. Please check that Ollama is running on {self.config.OLLAMA_EMBEDDING_HOST}:{self.config.OLLAMA_EMBEDDING_PORT}"
                    return
                else:
                    logger.error(f"Failed to store document '{filename}' in Milvus: {e}")
                    status.status = "error"
                    status.message = f"Storage failed: {str(e)[:100]}..."
                    return
            
            # Store document chunks in PostgreSQL for retrieval
            status.message = "Storing chunks in PostgreSQL..."
            status.progress = 85
            
            if self.postgres_manager and getattr(self, 'document_manager', None):
                logger.info(f"Storing {len(chunks)} chunks in PostgreSQL for document: {filename} via DocumentManager.persist_chunks")
                try:
                    stored = self.document_manager.persist_chunks(document_id=str(document_id), chunks=chunks)
                    logger.info(f"Persisted {stored}/{len(chunks)} chunks for document {document_id}")
                except Exception as e:
                    logger.error(f"Failed to persist document chunks via DocumentManager: {e}")
            else:
                logger.warning("PostgreSQL manager or DocumentManager not available, skipping document chunk storage")
            
            # Move file to uploaded folder
            uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            shutil.move(file_path, uploaded_path)
            logger.debug(f"Moved processed file from staging to uploaded: {uploaded_path}")
            
            # Update document metadata with processing results
            elapsed = (datetime.now() - status.start_time).total_seconds() if status.start_time else None
            try:
                if self.postgres_manager:
                    # Update the existing document record with processing results
                    update_query = """
                        UPDATE documents 
                        SET 
                            title = %s,
                            page_count = %s,
                            chunk_count = %s,
                            word_count = %s,
                            avg_chunk_chars = %s,
                            median_chunk_chars = %s,
                            top_keywords = %s,
                            processing_time_seconds = %s,
                            processing_status = %s,
                            updated_at = NOW()
                        WHERE id = %s
                    """
                    
                    with self.postgres_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute(update_query, [
                                status.title,
                                page_count,
                                len(chunks),
                                word_count,
                                avg_len,
                                med_len,
                                global_keywords,
                                elapsed,
                                'completed',
                                document_id
                            ])
                            conn.commit()
                    
                    logger.info(f"Successfully updated metadata for document ID {document_id} ({filename}) with {len(chunks)} chunks")
                else:
                    logger.warning("PostgreSQL manager not available for document metadata update")
            except Exception as e:
                logger.error(f"Failed to update document metadata for {filename}: {e}")
                raise  # Don't hide the error, let it fail properly
            
            status.status = "completed"
            status.message = f"Successfully processed {len(chunks)} chunks"
            status.progress = 100
            status.end_time = datetime.now()
            
        except Exception as e:
            # Also update database status to 'failed' on error
            try:
                # Update status using ID if we have it, otherwise skip database update
                if self.document_manager and 'document_id' in locals():
                    self.document_manager.update_processing_status(locals()['document_id'], 'failed')
                    logger.info(f"Updated document {locals()['document_id']} status to 'failed'")
                else:
                    logger.warning(f"Cannot update database status to 'failed' - document not yet created in database")
            except Exception as db_error:
                logger.warning(f"Failed to update database status to 'failed' for {filename}: {db_error}")
                
            status.status = "error"
            status.error_details = str(e)
            status.message = f"Processing failed: {str(e)}"
            status.end_time = datetime.now()
            logger.error(f"Processing failed for {filename}: {str(e)}")

    def _delete_staging_file_background(self, filename: str) -> None:
        """
        Delete a file from staging folder with database cleanup.
        
        Staging file deletion policy:
        - Removes file from staging folder (configured via UPLOAD_FOLDER)
        - Cleans up orphaned database records with 'pending' status
        - No vector embeddings to clean (staging files aren't processed yet)
        """
        # Ensure status entry exists for UI progress
        st = self.processing_status.get(filename) or DocumentProcessingStatus(filename=filename)
        self.processing_status[filename] = st
        st.status = 'processing'
        st.progress = 10
        st.message = 'Preparing deletion from staging...'
        st.start_time = datetime.now()
        
        try:
            file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
            
            # Clean up orphaned database record first
            st.message = 'Cleaning up database record...'
            st.progress = 30
            
            if self.postgres_manager:
                try:
                    with self.postgres_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM documents WHERE filename = %s AND processing_status = 'pending'", [filename])
                            deleted_rows = cur.rowcount
                            conn.commit()
                            if deleted_rows > 0:
                                logger.info(f"Cleaned up {deleted_rows} orphaned database record(s) for staging file: {filename}")
                except Exception as db_e:
                    logger.error(f"Failed to clean up database record for staging file {filename}: {db_e}")
                    # Don't raise - file deletion should still proceed
            
            # Remove file from staging
            st.message = 'Removing file from staging...'
            st.progress = 70
            
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed staging file: {file_path}")
            else:
                logger.warning(f"Staging file not found: {file_path}")
            
            st.status = 'completed'
            st.message = 'Staging file deleted successfully'
            st.progress = 100
            st.end_time = datetime.now()
            
            self._schedule_status_cleanup(filename, 5)  # 5 second delay for success
            
        except Exception as e:
            st.status = 'error'
            st.error_details = str(e)
            st.message = f'Staging deletion failed: {e}'
            st.progress = 100
            st.end_time = datetime.now()
            logger.error(f"Staging file deletion failed for {filename}: {e}")
            
            self._schedule_status_cleanup(filename, 10)  # 10 second delay for errors

    def _delete_uploaded_file_background(self, filename: str) -> None:
        """
        Delete a processed file from uploaded folder with full cleanup.
        
        Uploaded file deletion policy:
        - Removes embeddings from Milvus vector database
        - Purges metadata and chunks from PostgreSQL
        - Archives file to deleted folder (configured via DELETED_FOLDER)
        """
        # Ensure status entry exists for UI progress
        st = self.processing_status.get(filename) or DocumentProcessingStatus(filename=filename)
        self.processing_status[filename] = st
        st.status = 'processing'
        st.progress = 5
        st.message = 'Preparing uploaded file deletion...'
        st.start_time = datetime.now()
        
        try:
            file_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            
            # Find the document ID by filename
            st.message = 'Looking up document...'
            st.progress = 15
            
            document_id = None
            try:
                if self.postgres_manager:
                    with self.postgres_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("SELECT id FROM documents WHERE filename = %s", [filename])
                            result = cur.fetchone()
                            if result:
                                document_id = str(result['id'])
                                logger.info(f"Found document ID {document_id} for filename {filename}")
                            else:
                                logger.warning(f"No document found with filename {filename}")
                else:
                    logger.warning("PostgreSQL manager not available for document lookup")
            except Exception as e:
                logger.error(f"Failed to lookup document ID for {filename}: {e}")
            
            # Remove embeddings from Milvus
            st.message = 'Removing embeddings...'
            st.progress = 30
            
            try:
                logger.info(f"Starting embedding deletion for filename: {filename}")
                if document_id:
                    deletion_result = self.milvus_manager.delete_document(document_id=document_id)
                    logger.info(f"Embedding deletion result for ID {document_id}: {deletion_result}")
                else:
                    # Fallback to filename-based deletion for compatibility
                    deletion_result = self.milvus_manager.delete_document(filename=filename)
                    logger.info(f"Embedding deletion result for filename {filename}: {deletion_result}")
                
                if deletion_result.get('success'):
                    deleted_count = deletion_result.get('deleted_count', 0)
                    reported_count = deletion_result.get('reported_delete_count', 0)
                    
                    if deleted_count > 0:
                        logger.info(f"Immediately deleted {deleted_count} embeddings for {filename}")
                        st.message = f'Removed {deleted_count} embeddings'
                    elif reported_count > 0:
                        logger.info(f"Milvus marked {reported_count} embeddings for deletion - cleanup in progress")
                        st.message = f'Deletion initiated ({reported_count} embeddings marked for cleanup)'
                    else:
                        logger.info(f"Deletion command successful for {filename}")
                        st.message = 'Embedding deletion initiated'
                    st.progress = 50
                else:
                    error_msg = deletion_result.get('error', 'Unknown error')
                    logger.error(f"Embedding deletion failed for {filename}: {error_msg}")
                    raise Exception(f"Embedding deletion failed: {error_msg}")
                    
            except Exception as e:
                logger.error(f"Embedding deletion failed for {filename}: {e}", exc_info=True)
                raise Exception(f"Failed to delete embeddings: {str(e)}")
            
            # Purge document and chunks from PostgreSQL
            st.message = 'Removing metadata...'
            st.progress = 65
            
            try:
                if self.postgres_manager and document_id:
                    with self.postgres_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM documents WHERE id = %s", [document_id])
                            deleted_rows = cur.rowcount
                            conn.commit()
                            logger.info(f"Deleted document {document_id} and associated chunks from PostgreSQL (rows: {deleted_rows})")
                else:
                    error_msg = "No PostgreSQL manager available for metadata deletion"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            except Exception as e:
                logger.error(f"Metadata deletion failed for {filename}: {e}")
                raise
            
            # Archive file to deleted folder
            st.message = 'Archiving file...'
            st.progress = 80
            
            if os.path.exists(file_path):
                archive_path = os.path.join(self.config.DELETED_FOLDER, filename)
                if os.path.exists(archive_path):
                    stem, ext = os.path.splitext(filename)
                    archive_path = os.path.join(self.config.DELETED_FOLDER, f"{stem}_{int(time.time())}{ext}")
                shutil.move(file_path, archive_path)
                logger.info(f"Archived uploaded file to: {archive_path}")
            else:
                logger.warning(f"Uploaded file not found: {file_path}")
            
            st.status = 'completed'
            st.message = 'File deleted and archived successfully'
            st.progress = 100
            st.end_time = datetime.now()
            
            self._schedule_status_cleanup(filename, 5)  # 5 second delay for success
            
        except Exception as e:
            st.status = 'error'
            st.error_details = str(e)
            st.message = f'Uploaded file deletion failed: {e}'
            st.progress = 100
            st.end_time = datetime.now()
            logger.error(f"Uploaded file deletion failed for {filename}: {e}")
            
            self._schedule_status_cleanup(filename, 10)  # 10 second delay for errors

    def _schedule_status_cleanup(self, filename: str, delay_seconds: int) -> None:
        """
        Schedule cleanup of processing status after a delay.
        
        Args:
            filename: The filename to clean up status for
            delay_seconds: How long to wait before cleanup
        """
        def cleanup_status():
            time.sleep(delay_seconds)
            if filename in self.processing_status:
                del self.processing_status[filename]
                logger.info(f"Cleaned up processing status for file: {filename}")
        
        cleanup_thread = threading.Thread(target=cleanup_status, daemon=True)
        cleanup_thread.start()

    def run(self) -> None:
        """
        Run the Flask application.
        
        This method starts the web server and scheduler, making the application 
        available for processing requests and background tasks.
        """
        logger.info(f"Starting RAG Knowledgebase Manager on {self.config.FLASK_HOST}:{self.config.FLASK_PORT}")
        
        # Start the background scheduler for URL and email processing
        logger.info("Starting background scheduler for URL and email processing")
        self.scheduler_manager.start_scheduler()
        
        self.app.run(
            host=self.config.FLASK_HOST,
            port=self.config.FLASK_PORT,
            debug=self.config.FLASK_DEBUG
        )


if __name__ == "__main__":
    app = RAGKnowledgebaseManager()
    app.run()
