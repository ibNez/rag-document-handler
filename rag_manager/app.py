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

from .core.config import Config
from .core.models import DocumentProcessingStatus, URLProcessingStatus, EmailProcessingStatus
from ingestion.document.processor import DocumentProcessor
from .managers.milvus_manager import MilvusManager
from .web.routes import WebRoutes
from .scheduler_manager import SchedulerManager  # Import SchedulerManager
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/rag_document_handler.log')
    ]
)

# Suppress noisy external library logs that can be misleading
logging.getLogger('unstructured').setLevel(logging.WARNING)
logging.getLogger('pdfminer').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
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
        
        # Initialize PostgreSQL manager for metadata
        self._initialize_database_managers()
        
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
            from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
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
        except ImportError as e:
            logger.error(f"Failed to import PostgreSQL managers: {e}")
            self.postgres_manager = None
            self.database_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            self.postgres_manager = None
            self.database_manager = None

    def _initialize_url_manager(self) -> None:
        """Initialize URL manager based on feature flags."""
        logger.info("Initializing PostgreSQL URL Manager")
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            if self.postgres_manager:
                self.url_manager = PostgreSQLURLManager(self.postgres_manager)
                logger.info("PostgreSQL URL Manager initialized successfully")
            else:
                logger.error("Cannot initialize URL manager: PostgreSQL manager not available")
                self.url_manager = None
        except ImportError as e:
            logger.error(f"Failed to import PostgreSQL URL Manager: {e}")
            self.url_manager = None

    def _initialize_url_orchestrator(self) -> None:
        """Initialize URL orchestrator for managing URL processing tasks."""
        try:
            logger.info("Initializing URL orchestrator")
            
            # Validate prerequisites
            if not self.url_manager:
                logger.error("Cannot initialize URL orchestrator: URL manager not available")
                self.url_orchestrator = None
                return
            
            # Initialize URL orchestrator
            from ingestion.url.orchestrator import URLOrchestrator
            self.url_orchestrator = URLOrchestrator(
                config=self.config,
                url_manager=self.url_manager,
                processor=self.document_processor,  # Use existing document processor for URL content
                milvus_manager=self.milvus_manager  # Add MilvusManager for vector storage
            )
            logger.info("URL orchestrator initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize URL orchestrator: %s", e)
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
            
            # Initialize PostgreSQL email account manager
            from ingestion.email.manager import PostgreSQLEmailManager
            self.email_account_manager = PostgreSQLEmailManager(self.postgres_manager)
            logger.info("PostgreSQL email account manager initialized")
            
            # Ensure email vector store is ready (separate from document vector store)
            email_vector_store = self.milvus_manager.get_email_vector_store()
            logger.info("Email vector store validated for email embeddings")
            
            # Create PostgreSQL-based email message manager
            from ingestion.email.email_manager_postgresql import PostgreSQLEmailManager as EmailMessageManager
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
                logger.error(f"URL {url_id} not found")
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
            
            result = self.url_orchestrator.process_url(url_id)
            
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
            
            # Delete from URL manager (includes embeddings cleanup)
            status.message = "Deleting URL and embeddings..."
            status.progress = 50
            
            result = self.url_manager.delete_url(url_id)
            
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
            
            # Create deterministic document ID
            abs_path = str(Path(file_path).resolve())
            file_stat = os.stat(file_path)
            file_sig = f"{abs_path}|{file_stat.st_size}|{int(file_stat.st_mtime)}"
            document_id = hashlib.sha1(file_sig.encode('utf-8')).hexdigest()[:16]
            logger.info(f"Processing document '{filename}' with ID: {document_id}")
            
            # Load and chunk document
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
            kw_data = self.document_processor.extract_keywords(chunks)
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
            logger.debug(f"Attached keywords to {len(chunks)} chunks")
            
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
            self.document_processor.enrich_topics(chunks)
            logger.info(f"Topic enrichment completed for: {filename}")
            
            status.status = "storing"
            status.message = "Storing in Milvus..."
            status.progress = 80
            
            logger.info(f"Storing {len(chunks)} chunks in Milvus for document: {filename}")
            self.milvus_manager.insert_documents(filename, chunks)
            logger.info(f"Successfully stored document '{filename}' in Milvus vector database")
            
            # Move file to uploaded folder
            uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            shutil.move(file_path, uploaded_path)
            logger.debug(f"Moved processed file from staging to uploaded: {uploaded_path}")
            
            # Upsert document metadata row
            elapsed = (datetime.now() - status.start_time).total_seconds() if status.start_time else None
            try:
                if self.url_manager:
                    self.url_manager.upsert_document_metadata(
                        filename,
                        title=status.title,
                        page_count=page_count,
                        chunk_count=len(chunks),
                        word_count=word_count,
                        avg_chunk_chars=avg_len,
                        median_chunk_chars=med_len,
                        top_keywords=json.dumps(global_keywords),
                        processing_time_seconds=elapsed
                    )
                else:
                    logger.warning("URL manager not available for document metadata storage")
            except Exception as e:
                logger.warning(f"Failed to upsert document metadata for {filename}: {e}")
            
            status.status = "completed"
            status.message = f"Successfully processed {len(chunks)} chunks"
            status.progress = 100
            status.end_time = datetime.now()
            
        except Exception as e:
            status.status = "error"
            status.error_details = str(e)
            status.message = f"Processing failed: {str(e)}"
            status.end_time = datetime.now()
            logger.error(f"Processing failed for {filename}: {str(e)}")

    def _delete_file_background(self, folder: str, filename: str) -> None:
        """
        Background deletion worker with progress tracking.

        Strict removal policy:
        - Staging files: permanently removed from disk.
        - Uploaded files: embeddings removed (Milvus), metadata row purged, file moved
          to DELETED_FOLDER (only raw file retained for possible manual restore). No metadata
          history is preserved.
        """
        # Ensure status entry exists for UI progress
        st = self.processing_status.get(filename) or DocumentProcessingStatus(filename=filename)
        self.processing_status[filename] = st
        st.status = 'processing'
        st.progress = 5
        st.message = 'Preparing deletion...'
        st.start_time = datetime.now()
        
        try:
            folder_path = self.config.UPLOAD_FOLDER if folder == 'staging' else self.config.UPLOADED_FOLDER
            file_path = os.path.join(folder_path, filename)
            
            if folder == 'uploaded':
                st.message = 'Removing embeddings...'
                st.progress = 20
                try:
                    self.milvus_manager.delete_document(filename=filename)
                except Exception as e:
                    logger.warning(f"Embedding deletion failed for {filename}: {e}")
                else:
                    st.progress = 40
                
                # Purge metadata row
                try:
                    if self.url_manager:
                        self.url_manager.delete_document_metadata(filename)
                    else:
                        logger.warning("URL manager not available for document metadata deletion")
                except Exception as e:
                    logger.warning(f"Metadata deletion failed for {filename}: {e}")
                else:
                    st.progress = max(st.progress, 55)
            
            st.message = 'Archiving file...'
            st.progress = max(st.progress, 70)
            
            try:
                if os.path.exists(file_path):
                    if folder == 'uploaded':
                        archive_path = os.path.join(self.config.DELETED_FOLDER, filename)
                        if os.path.exists(archive_path):
                            stem, ext = os.path.splitext(filename)
                            archive_path = os.path.join(self.config.DELETED_FOLDER, f"{stem}_{int(time.time())}{ext}")
                        shutil.move(file_path, archive_path)
                    else:
                        os.remove(file_path)
            except Exception as e:
                logger.error(f"File archive/delete failed for {filename}: {e}")
                raise
            
            st.status = 'completed'
            st.message = 'Deletion complete'
            st.progress = 100
            st.end_time = datetime.now()
            
        except Exception as e:
            st.status = 'error'
            st.error_details = str(e)
            st.message = f'Deletion failed: {e}'
            st.progress = 100
            st.end_time = datetime.now()
        finally:
            # Keep status entry so UI can observe 'Deletion complete'.
            # A TTL cleanup will purge it later via get_status.
            pass

    def run(self) -> None:
        """
        Run the Flask application.
        
        This method starts the web server and scheduler, making the application 
        available for processing requests and background tasks.
        """
        logger.info(f"Starting RAG Document Handler on {self.config.FLASK_HOST}:{self.config.FLASK_PORT}")
        
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
