"""
Web routes module for RAG Knowledgebase Manager.

This module contains Flask route definitions and handlers, separated from the main
application logic following development rules for better organization.
"""

import os
import logging
import psycopg2.extras
import hashlib
from typing import Any, Dict
from datetime import datetime, timedelta

from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename

from ..core.config import Config
from ..core.models import DocumentProcessingStatus, URLProcessingStatus, EmailProcessingStatus
from .stats import StatsProvider

# Configure logging
logger = logging.getLogger(__name__)


class WebRoutes:
    """
    Manages Flask web routes and handlers for the RAG Knowledgebase Manager.
    
    This class follows the development rules with proper separation of concerns,
    type hints, and comprehensive logging.
    """
    
    def __init__(self, app: Flask, config: Config, rag_manager: Any) -> None:
        """
        Initialize web routes.
        
        Args:
            app: Flask application instance
            config: Application configuration
            rag_manager: Main RAG application manager instance
        """
        self.app = app
        self.config = config
        self.rag_manager = rag_manager
        
        # Initialize centralized stats provider
        self.stats_provider = StatsProvider(rag_manager)
        
        self._register_routes()
        logger.info("Web routes initialized with centralized stats provider")
    
    def _register_routes(self) -> None:
        """Register Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            """Rag Knowledgebase Management System."""
            # Get files in staging and uploaded folders
            all_staging_files = self._get_directory_files(self.config.UPLOAD_FOLDER)
            all_uploaded_files = self._get_directory_files(self.config.UPLOADED_FOLDER)
            
            # Filter staging files: only show files with 'pending' status or not in database
            staging_files = []
            for file_info in all_staging_files:
                db_status = self._get_file_database_status(file_info['name'])
                if db_status in [None, 'pending']:  # None means not in database yet
                    staging_files.append(file_info)
            
            # Filter uploaded files: only show files with 'completed' status
            uploaded_files = []
            for file_info in all_uploaded_files:
                db_status = self._get_file_database_status(file_info['name'])
                logger.info(f"File {file_info['name']}: db_status = {db_status}")
                if db_status == 'completed':
                    uploaded_files.append(file_info)
                    logger.info(f"Added {file_info['name']} to uploaded_files")
            
            logger.info(f"Total uploaded files found: {len(uploaded_files)} (out of {len(all_uploaded_files)} physical files)")

            # Gather stats and diagnostic data for the dashboard. Use StatsProvider
            # and helper methods where available and fall back to safe defaults.
            try:
                all_stats = self.stats_provider.get_all_stats() or {}
                kb_meta = all_stats.get('knowledgebase', {})
                email_meta = all_stats.get('email', {})
                url_meta = all_stats.get('url', {})
                system_meta = all_stats.get('system', {})
            except Exception as e:
                logger.warning(f"Failed to collect panel stats: {e}")
                kb_meta = {}
                email_meta = {}
                url_meta = {}
                system_meta = {}

            # Collection stats for primary KB and email collections (Milvus)
            collection_stats = None
            email_collection_stats = None
            try:
                mm = getattr(self.rag_manager, 'milvus_manager', None)
                if mm:
                    try:
                        collection_stats = mm.get_collection_stats() if hasattr(mm, 'get_collection_stats') else getattr(mm, 'collection_info', None)
                    except Exception:
                        collection_stats = None
                    try:
                        email_collection_stats = mm.get_email_collection_stats() if hasattr(mm, 'get_email_collection_stats') else None
                    except Exception:
                        email_collection_stats = None
            except Exception:
                collection_stats = None
                email_collection_stats = None

            # SQL and Milvus/OLLAMA status
            try:
                sql_status = self._get_database_status()
            except Exception:
                sql_status = {"connected": False}

            try:
                ollama_status = self.rag_manager.ollama_health.get_overall_status() if hasattr(self.rag_manager, 'ollama_health') else {"connected": False}
            except Exception:
                ollama_status = {"connected": False}

            try:
                milvus_status = None
                mm = getattr(self.rag_manager, 'milvus_manager', None)
                if mm and hasattr(mm, 'get_status'):
                    milvus_status = mm.get_status()
                elif mm and hasattr(mm, 'connection_args'):
                    milvus_status = {"connected": True, "connection": getattr(mm, 'connection_args', None)}
                else:
                    milvus_status = {"connected": False}
            except Exception:
                milvus_status = {"connected": False}

            # Enriched URL list for UI
            try:
                urls = self._get_enriched_urls()
            except Exception:
                urls = []

            # Email accounts for UI
            try:
                email_accounts = []
                if hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                    email_accounts = self.rag_manager.email_account_manager.list_accounts(include_password=False)
            except Exception:
                email_accounts = []

            return render_template(
                'index.html',
                staging_files=staging_files,
                all_staging_files=all_staging_files,
                uploaded_files=uploaded_files,
                collection_stats=collection_stats,
                email_collection_stats=email_collection_stats,
                sql_status=sql_status,
                milvus_status=milvus_status,
                ollama_status=ollama_status,
                kb_meta=kb_meta,
                url_meta=url_meta,
                email_meta=email_meta,
                system_meta=system_meta,
                processing_status=self.rag_manager.processing_status,
                url_processing_status=self.rag_manager.url_processing_status,
                urls=urls,
                email_accounts=email_accounts,
                email_processing_status=self.rag_manager.email_processing_status,
                config=self.config,
            )
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload to staging area with duplicate detection."""
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            if file and file.filename and self._allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                
                try:
                    # Simple duplicate check - if file already exists, reject upload
                    if os.path.exists(file_path):
                        logger.warning(f"File {filename} already exists in staging - rejecting upload")
                        flash(f'File "{filename}" already exists. Please delete the original before uploading again.', 'warning')
                        return redirect(url_for('index'))
                    
                    # Save file to staging folder
                    file.save(file_path)
                    logger.info(f"File uploaded to staging: {filename}")
                    
                    # Compute file hash for database storage
                    file_hash = self._compute_file_hash(file_path)
                    logger.info(f"Computed hash for {filename}: {file_hash[:16]}...")
                    
                    # === CREATE DATABASE RECORD ===
                    try:
                        # Use centralized PostgreSQLManager to ensure RealDict rows
                        from rag_manager.managers.postgres_manager import PostgreSQLManager
                        mgr = PostgreSQLManager()
                        with mgr.get_connection() as conn:
                            with conn.cursor() as cursor:
                                # === FILENAME COLLISION CHECK ===
                                cursor.execute(
                                    "SELECT processing_status FROM documents WHERE file_path = %s",
                                    (file_path,)
                                )
                                existing_record = cursor.fetchone()

                                if existing_record:
                                    # Filename collision with different content - handle gracefully
                                    existing_status = existing_record.get('processing_status') if isinstance(existing_record, dict) else None
                                    logger.warning(f"Filename collision detected for {filename} - existing status: {existing_status}")
                                    os.remove(file_path)
                                    flash(
                                        f'A file named "{filename}" already exists in the system. '
                                        f'Please delete the existing one before adding it again.',
                                        'error'
                                    )
                                    return redirect(url_for('index'))

                                # === CREATE DOCUMENT RECORD ===
                                cursor.execute("""
                                    INSERT INTO documents (title, filename, file_path, processing_status, file_hash, created_at, updated_at)
                                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                                """, (self.rag_manager._fallback_title_from_filename(filename), filename, file_path, 'pending', file_hash))
                                conn.commit()
                        logger.info(f"Stored new file record for {filename} in database")
                    except Exception as e:
                        # Log the full technical error details for debugging
                        logger.error(f"Could not store file record for {filename}: {e}")
                        
                        # Clean up the file since we couldn't store the record
                        os.remove(file_path)
                        
                        # Provide user-friendly error message for duplicate content
                        error_str = str(e)
                        if 'duplicate key value violates unique constraint "documents_file_hash_key"' in error_str:
                            flash('A file matching this file\'s content has already been processed. Please delete the processed document before proceeding with uploading the file again.', 'error')
                        else:
                            flash(f'Error storing file record: {error_str}', 'error')
                        return redirect(url_for('index'))
                    
                    flash(f'File "{filename}" uploaded successfully', 'success')
                    logger.info(f"File uploaded successfully: {filename}")
                        
                except Exception as e:
                    flash(f'Error uploading file: {str(e)}', 'error')
                    logger.error(f"Upload error for {filename}: {str(e)}")
                    
                    # Clean up file if it was created
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            logger.info(f"Cleaned up failed upload: {filename}")
                        except Exception as cleanup_error:
                            logger.warning(f"Could not clean up failed upload {filename}: {cleanup_error}")
            else:
                flash('Invalid file type', 'error')
            
            return redirect(url_for('index'))

        @self.app.route('/download/<path:filename>')
        def download_file(filename):
            """Download a processed (uploaded) document by filename."""
            try:
                import os
                uploaded_folder_abs = os.path.abspath(self.config.UPLOADED_FOLDER)
                file_path = os.path.join(uploaded_folder_abs, filename)
                logger.info(f"Download request for: {filename}")
                logger.info(f"Looking in folder: {uploaded_folder_abs}")
                logger.info(f"Full file path: {file_path}")
                logger.info(f"File exists: {os.path.exists(file_path)}")
                return send_from_directory(uploaded_folder_abs, filename, as_attachment=True)
            except FileNotFoundError:
                logger.warning(f"Download requested for non-existent file: {filename}")
                abort(404)
            except Exception as e:
                logger.error(f"Download error for {filename}: {str(e)}")
                abort(500)

        @self.app.route('/search', methods=['GET', 'POST'])
        def search():
            """Handle search requests and display results."""
            logger.info(f"Search endpoint called with method: {request.method}")
            
            if request.method == 'POST':
                query = request.form.get('query', '').strip()
                top_k = int(request.form.get('top_k', 10))
                search_type = request.form.get('search_type', 'rag')  # 'rag' or 'similarity'
                
                logger.info(f"Processing search request - Query: '{query}', Top K: {top_k}, Type: {search_type}")
                
                if query:
                    try:
                        if search_type == 'rag':
                            # RAG search with answer generation - first classify the query
                            logger.debug("Initiating RAG search with query classification")
                            
                            # Get classification first
                            classification = self.rag_manager.milvus_manager.classify_query_intent(query)
                            logger.info(f"Query classified as: {classification}")
                            
                            # Create classification object for template
                            classification_obj = {
                                'type': classification,
                                'confidence': '95%',  # LLM classification is considered high confidence
                                'reason': f'Query was classified as {classification} content by AI classifier'
                            }
                            
                            # Route based on classification
                            if classification == 'email' and hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                                # Email-specific search using retrieval
                                logger.debug("Routing to email search")
                                try:
                                    email_results = self.rag_manager.email_account_manager.search_emails_hybrid(query, top_k)
                                    context_text, email_sources = self.rag_manager.email_account_manager.format_email_context(email_results)
                                    
                                    # Generate answer using email context
                                    from langchain_ollama import ChatOllama
                                    from langchain.schema import SystemMessage, HumanMessage
                                    
                                    llm = ChatOllama(
                                        model=self.config.CHAT_MODEL,
                                        base_url=self.config.CHAT_BASE_URL,
                                        temperature=self.config.CHAT_TEMPERATURE
                                    )
                                    
                                    email_system_prompt = """You are an assistant that answers questions using email content and communication data.

                                        Instructions:
                                        1. Use the provided email context to answer the user's question.
                                        2. Focus on email-specific information: senders, recipients, subjects, dates, and content.
                                        3. Provide a structured answer in Markdown with **headings** and **clear paragraphs**.
                                        4. Support statements with inline citations using [1], [2], [3], etc.
                                        5. Include relevant email metadata (sender, date, subject) when citing emails.
                                        6. If information is incomplete, state this explicitly.

                                        Now, answer the following email-related question using the email context provided:"""
                                    
                                    system_message = SystemMessage(content=f"{email_system_prompt}\n\nEmail Context:\n{context_text}\n\n")
                                    human_message = HumanMessage(content=query)
                                    
                                    response = llm.invoke([system_message, human_message])
                                    
                                    results = {
                                        'answer': str(response.content),
                                        'sources': email_sources,
                                        'unique_sources': email_sources,
                                        'conversation_classification': classification,
                                        'analysis_info': {
                                            'system_instructions': email_system_prompt,
                                            'search_type': 'email',
                                            'conversation_classification': classification,
                                            'email_results_count': len(email_results)
                                        }
                                    }
                                    logger.info(f"Email search successful - Found {len(email_sources)} email sources")
                                    
                                except Exception as email_error:
                                    logger.error(f"Email search failed: {email_error}")
                                    # Return error response instead of falling back
                                    results = {
                                        'error': f"Email search failed: {str(email_error)}",
                                        'context': '',
                                        'sources': [],
                                        'answer': 'Email search is currently unavailable. Please try again later.'
                                    }
                            else:
                                # Standard document/web search
                                logger.debug("Routing to document RAG search")
                                results = self.rag_manager.milvus_manager.rag_search_and_answer(query, top_k)
                            
                            # Log results for debugging
                            if results.get('error'):
                                error_message = results.get('error')
                                if error_message and 'No documents found in vector database' in str(error_message):
                                    logger.warning(f"RAG search returned warning: {error_message}")
                                else:
                                    logger.error(f"RAG search returned error: {error_message}")
                                logger.debug(f"Debug info: {results.get('debug_info', {})}")
                            else:
                                logger.info(f"RAG search successful - Found {len(results.get('sources', []))} sources")
                            
                            return render_template('search.html', 
                                                 query=query, 
                                                 rag_results=results, 
                                                 conversation_classification=classification_obj,
                                                 search_type=search_type)
                        else:
                            # Similarity search only
                            logger.debug("Initiating similarity search")
                            results = self.rag_manager.milvus_manager.search_documents(query, top_k)
                            logger.info(f"Similarity search successful - Found {len(results) if results else 0} results")
                            
                            return render_template('search.html', 
                                                 query=query, 
                                                 results=results, 
                                                 search_type=search_type)
                                                 
                    except Exception as e:
                        logger.error(f"Search request failed for query '{query}': {str(e)}", exc_info=True)
                        logger.error(f"Error type: {type(e).__name__}")
                        
                        # Create error result for display
                        error_result = {
                            "answer": f"Search failed with error: {str(e)}",
                            "sources": [],
                            "context_used": False,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        
                        flash(f'Search error: {str(e)}', 'error')
                        return render_template('search.html', 
                                             query=query, 
                                             rag_results=error_result if search_type == 'rag' else None,
                                             conversation_classification=error_result.get('conversation_classification'),
                                             results=[] if search_type != 'rag' else None,
                                             search_type=search_type)
                else:
                    logger.warning("Empty search query submitted")
                    flash('Please enter a search query', 'error')
            
            logger.debug("Rendering empty search page")
            return render_template('search.html')

        @self.app.route('/status/<filename>')
        def get_status(filename):
            """Get status for a specific file - handles both processing and deletion."""
            
            # First check if there's an active deletion or processing status in memory
            if filename in self.rag_manager.processing_status:
                status = self.rag_manager.processing_status[filename]
                
                # If this is a deletion in progress, check cleanup status
                deletion_status = None
                deletion_keywords = ['delet', 'cleanup', 'embedd', 'archiv']
                message_lower = (status.message or '').lower()
                is_deletion_related = any(keyword in message_lower for keyword in deletion_keywords)
                
                if status.status in ['processing', 'completed'] and is_deletion_related:
                    try:
                        logger.info(f"Checking deletion status for {filename} (message: '{status.message}')")
                        deletion_status = self.rag_manager.milvus_manager.check_deletion_status(filename=filename)
                        logger.info(f"Deletion status check for {filename}: {deletion_status}")
                        
                        # Update status message based on cleanup progress
                        if deletion_status.get('cleanup_complete'):
                            status.message = 'Deletion complete - all embeddings cleaned up'
                            status.status = 'completed'
                            logger.info(f"Marking {filename} as deletion complete")
                        elif deletion_status.get('remaining_records', 0) > 0:
                            remaining = deletion_status.get('remaining_records', 0)
                            status.message = f'Deletion in progress - {remaining} embeddings awaiting cleanup'
                            logger.info(f"Deletion in progress for {filename}: {remaining} embeddings remaining")
                            
                    except Exception as e:
                        logger.warning(f"Could not check deletion status for {filename}: {e}", exc_info=True)
                
                response_data = {
                    'status': status.status,
                    'progress': status.progress,
                    'message': status.message,
                    'chunks_count': status.chunks_count,
                    'error_details': status.error_details,
                    'title': status.title,
                    'deletion_status': deletion_status
                }
                
                return jsonify(response_data)
            
            # No in-memory status, check database for processing status
            try:
                # Use centralized PostgreSQLManager for consistent dict rows
                from rag_manager.managers.postgres_manager import PostgreSQLManager
                mgr = PostgreSQLManager()
                with mgr.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            "SELECT processing_status, title, word_count, metadata FROM documents WHERE title = %s OR file_path LIKE %s",
                            (filename, f'%{filename}%')
                        )
                        result = cursor.fetchone()
                        if result:
                            # dict-style row expected
                            db_status = result.get('processing_status')
                            title = result.get('title')
                            word_count = result.get('word_count')
                            metadata = result.get('metadata')

                            # Get chunk count from metadata if available
                            chunk_count = 0
                            if metadata and isinstance(metadata, dict):
                                chunk_count = metadata.get('chunk_count', 0)

                            # Convert database status to response format
                            if db_status == 'completed':
                                response_data = {
                                    'status': 'completed',
                                    'progress': 100,
                                    'message': f'Successfully processed {chunk_count} chunks' if chunk_count > 0 else 'Processing completed',
                                    'chunks_count': chunk_count,
                                    'error_details': None,
                                    'title': title,
                                    'deletion_status': None
                                }
                                return jsonify(response_data)
                            elif db_status == 'failed':
                                response_data = {
                                    'status': 'error',
                                    'progress': 0,
                                    'message': 'Processing failed',
                                    'chunks_count': 0,
                                    'error_details': 'Processing failed',
                                    'title': title,
                                    'deletion_status': None
                                }
                                return jsonify(response_data)
                            elif db_status == 'pending':
                                response_data = {
                                    'status': 'pending',
                                    'progress': 0,
                                    'message': 'Waiting to be processed',
                                    'chunks_count': 0,
                                    'error_details': None,
                                    'title': title,
                                    'deletion_status': None
                                }
                                return jsonify(response_data)
            except Exception as e:
                logger.warning(f"Error checking database status for {filename}: {e}")
            
            return jsonify({'status': 'not_found'}), 404

        @self.app.route('/admin/scheduler_status')
        def scheduler_status():
            """Diagnostic endpoint for scheduler status."""
            return jsonify(self.rag_manager._scheduler_status())
            
        @self.app.route('/admin/ollama_status')
        def ollama_status():
            """Diagnostic endpoint for Ollama services status."""
            return jsonify(self.rag_manager.ollama_health.get_overall_status())

        @self.app.route('/admin/milvus_stats')
        def milvus_stats():
            """Return basic Milvus collection stats and retriever debug info."""
            try:
                mm = self.rag_manager.milvus_manager
                # Ensure vector store is initialized
                try:
                    mm._ensure_vector_store()
                except Exception:
                    pass

                from pymilvus import Collection
                collection_name = getattr(mm, 'collection_name', None)
                if not collection_name:
                    return jsonify({'error': 'No collection configured'}), 400

                try:
                    coll = Collection(collection_name)
                    num_entities = coll.num_entities
                except Exception as e:
                    num_entities = None

                retriever_debug = {}
                try:
                    retr = getattr(mm, 'document_retriever', None)
                    if retr and hasattr(retr, 'last_analysis'):
                        retriever_debug = retr.last_analysis or {}
                except Exception:
                    retriever_debug = {}

                return jsonify({
                    'collection': collection_name,
                    'num_entities': num_entities,
                    'connection': mm.connection_args,
                    'retriever_debug': retriever_debug
                })
            except Exception as e:
                logger.error(f"Failed to get Milvus stats: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/document/<path:filename>/update', methods=['POST'])
        def update_document_metadata(filename):
            """Update editable document metadata like title."""
            title = request.form.get('title','').strip()
            if '..' in filename or filename.startswith('/'):
                abort(400)
            uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            if not os.path.isfile(uploaded_path):
                flash('Document not found', 'error')
                return redirect(url_for('index'))
            try:
                if title:
                    # Get the correct file_path for the upsert
                    uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
                    self.rag_manager.document_manager.upsert_document_metadata(
                        filename, 
                        {
                            'title': title,
                            'file_path': uploaded_path
                        }
                    ) if self.rag_manager.document_manager else None
                    flash('Document title updated', 'success')
                else:
                    flash('No title provided', 'info')
            except Exception as e:
                flash(f'Update failed: {e}', 'error')
            return redirect(url_for('index'))
        
        @self.app.route('/process/<filename>')
        def process_file(filename):
            """Process a file from staging to database."""
            if filename not in self.rag_manager.processing_status:
                self.rag_manager.processing_status[filename] = DocumentProcessingStatus(filename=filename)
            
            # Update database status to 'pending' when processing starts
            try:
                if self.rag_manager.document_manager:
                    staging_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                    self.rag_manager.document_manager.upsert_document_metadata(
                        filename,
                        {
                            'processing_status': 'pending',
                            'file_path': staging_path
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to set processing status to 'pending' for {filename}: {e}")
            
            # Start processing in background thread
            import threading
            thread = threading.Thread(target=self.rag_manager._process_document_background, args=(filename,))
            thread.daemon = True
            thread.start()
            
            flash(f'Processing started for "{filename}"', 'info')
            return redirect(url_for('index'))
        
        @self.app.route('/url_status/<url_id>')
        def get_url_status(url_id):
            """Get processing status for a URL refresh."""
            logger.debug(f"get_url_status called with url_id: '{url_id}' (type: {type(url_id)})")
            
            # UUID validation - ensure it's a valid UUID string
            try:
                import uuid
                # Test if url_id is a valid UUID
                uuid.UUID(url_id)
                url_id_str = str(url_id)
                logger.debug(f"Valid UUID string for status check: '{url_id_str}'")
            except ValueError as e:
                logger.error(f"Invalid URL ID format for status check: '{url_id}' - {e}")
                return jsonify({'status': 'not_found'})
                
            logger.debug(f"Checking processing status for URL ID: '{url_id_str}'")
            status = self.rag_manager.url_processing_status.get(url_id_str)
            if status:
                logger.debug(f"Found active processing status for URL ID '{url_id_str}': {status.status}")
                return jsonify({
                    'status': status.status,
                    'progress': status.progress,
                    'message': status.message,
                    'error_details': status.error_details
                })
            
            # No in-memory status; fetch final state from DB for in-place UI update
            logger.debug(f"No active processing status, fetching URL record for ID: '{url_id_str}'")
            try:
                rec = self.rag_manager.url_manager.get_url_by_id(url_id_str)
                if rec:
                    logger.debug(f"Found URL record for status update: {rec.get('url', 'N/A')}")
                    # Compute next_refresh similar to list view
                    next_refresh = None
                    try:
                        last = rec.get('last_scraped')
                        interval = rec.get('refresh_interval_minutes')
                        if last and interval and int(interval) > 0:
                            try:
                                dt_last = datetime.fromisoformat(str(last))
                            except Exception:
                                dt_last = datetime.fromisoformat(str(last)[:19])
                            dt_next = dt_last + timedelta(minutes=int(interval))
                            next_refresh = dt_next.strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        next_refresh = None
                    
                    # Format last_scraped to human-readable format
                    last_scraped = rec.get('last_scraped')
                    if last_scraped:
                        try:
                            dt_last_scraped = datetime.fromisoformat(str(last_scraped))
                            last_scraped = dt_last_scraped.strftime('%Y-%m-%d %H:%M')
                        except Exception as e:
                            logger.exception(f"Failed to format last_scraped '{last_scraped}': {e}")

                    # Format next_refresh to human-readable format
                    if next_refresh:
                        try:
                            dt_next_refresh = datetime.fromisoformat(str(next_refresh))
                            next_refresh = dt_next_refresh.strftime('%Y-%m-%d %H:%M')
                        except Exception as e:
                            logger.exception(f"Failed to format next_refresh '{next_refresh}': {e}")

                    status_response = {
                        'status': 'not_found',
                        'last_update_status': rec.get('last_update_status'),
                        'last_scraped': last_scraped,
                        'next_refresh': next_refresh
                    }
                    logger.debug(f"Returning status response for URL ID '{url_id_str}': {status_response}")
                    return jsonify(status_response)
                else:
                    logger.warning(f"URL record not found for ID: '{url_id_str}'")
                    return jsonify({'status': 'deleted'})
            except Exception as e:
                logger.error(f"Exception while fetching URL status for ID '{url_id_str}': {e}")
                return jsonify({'status': 'not_found'})
            logger.debug(f"Returning default 'not_found' status for URL ID: '{url_id_str}'")
            return jsonify({'status': 'not_found'})

        @self.app.route('/email_status/<account_id>')
        def get_email_status(account_id: str):
            """Get processing status for an email account refresh."""
            status = self.rag_manager.email_processing_status.get(str(account_id))
            if status:
                return jsonify({
                    'status': status.status,
                    'progress': status.progress,
                    'message': status.message,
                    'error_details': status.error_details
                })
            try:
                if hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                    for acct in self.rag_manager.email_account_manager.list_accounts(include_password=False):
                        email_acct_id = acct.get('email_account_id')
                        if not email_acct_id:
                            logger.warning(f"Email account missing 'email_account_id' canonical key: {acct}")
                            email_acct_id = acct.get('id')
                        if str(email_acct_id) == str(account_id):
                            return jsonify({
                                'status': 'not_found',
                                'last_update_status': acct.get('last_update_status'),
                                'last_synced': acct.get('last_synced'),
                                'next_run': acct.get('next_run')
                            })
                return jsonify({'status': 'deleted'})
            except Exception:
                return jsonify({'status': 'not_found'})
        
        @self.app.route('/delete_file_bg/<folder>/<filename>', methods=['POST','GET'])
        def delete_file_bg(folder: str, filename: str):
            """Start background (soft) deletion for a file."""
            if folder not in ['staging', 'uploaded']:
                flash('Invalid folder', 'error')
                return redirect(url_for('index'))
            if request.method == 'GET':
                logger.info(f"GET fallback invoked for deletion of {filename} in {folder}")
            
            # Seed processing status so the card shows a bar immediately
            st = self.rag_manager.processing_status.get(filename)
            if not st:
                self.rag_manager.processing_status[filename] = DocumentProcessingStatus(filename=filename)
            
            # Start background worker - use appropriate method based on folder
            import threading
            if folder == 'staging':
                th = threading.Thread(target=self.rag_manager._delete_staging_file_background, args=(filename,))
            else:  # uploaded
                th = threading.Thread(target=self.rag_manager._delete_uploaded_file_background, args=(filename,))
            th.daemon = True
            th.start()
            flash('Deletion started in background', 'info')
            return redirect(url_for('index'))
        
        @self.app.route('/delete_embeddings/<filename>', methods=['POST'])
        def delete_embeddings(filename):
            """Delete all embeddings for a document from Milvus database."""
            try:
                result = self.rag_manager.milvus_manager.delete_document(filename=filename)
                
                if result.get("success"):
                    deleted_count = result.get("deleted_count", 0)
                    flash(f'Successfully deleted {deleted_count} embeddings for "{filename}"', 'success')
                    logger.info(f"Embeddings deleted for {filename}: {deleted_count} chunks")
                else:
                    error_msg = result.get("error", "Unknown error occurred")
                    flash(f'Failed to delete embeddings for "{filename}": {error_msg}', 'error')
                    logger.error(f"Embedding deletion failed for {filename}: {error_msg}")
                    
            except Exception as e:
                flash(f'Error deleting embeddings: {str(e)}', 'error')
                logger.error(f"Embedding deletion error for {filename}: {str(e)}")

            return redirect(url_for('index'))

        @self.app.route('/add_url', methods=['POST'])
        def add_url():
            """Add a new URL with automatic title extraction and initial processing."""
            url = request.form.get('url', '').strip()
            
            if not url:
                flash('URL is required', 'error')
                return redirect(url_for('index'))
            
            result = self.rag_manager.url_manager.add_url(url)
            if result['success']:
                extracted_title = result.get('title', 'Unknown')
                url_id = result.get('url_id')
                if not url_id:
                    logger.error("URL manager returned success but no canonical 'url_id': %s", result)
                    flash('Internal error: URL added but missing identifier; check logs', 'error')
                    return redirect(url_for('index'))
                
                # Automatically trigger initial processing with snapshot creation
                if url_id and not url_id in self.rag_manager.url_processing_status:
                    logger.info(f"Auto-triggering initial processing for new URL: {url}")
                    
                    # Create processing status
                    self.rag_manager.url_processing_status[url_id] = URLProcessingStatus(
                        url=url,
                        title=extracted_title
                    )
                    
                    # Start background processing
                    import threading
                    th = threading.Thread(target=self.rag_manager._process_url_background, args=(url_id,))
                    th.daemon = True
                    th.start()
                    
                    flash(f'URL added successfully with title: "{extracted_title}" - Processing started in background', 'success')
                else:
                    flash(f'URL added successfully with title: "{extracted_title}"', 'success')
            else:
                flash(f'Failed to add URL: {result["message"]}', 'error')
            
            return redirect(url_for('index'))
        
        @self.app.route('/delete_url/<url_id>', methods=['POST'])
        def delete_url(url_id):
            """Delete a URL."""
            logger.info(f"delete_url route called with url_id: '{url_id}' (type: {type(url_id)})")
            
            # UUID validation - ensure it's a valid UUID string
            try:
                import uuid
                # Test if url_id is a valid UUID
                uuid.UUID(url_id)
                url_id_str = str(url_id)
                logger.debug(f"Valid UUID string for deletion: '{url_id_str}'")
            except ValueError as e:
                logger.error(f"Invalid URL ID format for deletion: '{url_id}' - {e}")
                flash('Invalid URL ID format', 'error')
                return redirect(url_for('index'))
                
            # Load the URL and any crawled pages
            logger.debug(f"Looking up URL for deletion with ID: '{url_id_str}'")
            url_rec = self.rag_manager.url_manager.get_url_by_id(url_id_str)
            if not url_rec:
                logger.error(f"URL not found for deletion with ID: '{url_id_str}'")
                flash('URL not found', 'error')
                return redirect(url_for('index'))
            url = url_rec.get('url')
            logger.info(f"Deleting URL: {url} (ID: {url_id_str})")
            
            # Delete embeddings for the single URL
            try:
                doc_id = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()[:16]
                logger.debug(f"Deleting primary URL embeddings for doc_id: {doc_id}")
                self.rag_manager.milvus_manager.delete_document(document_id=doc_id)
            except Exception as e:
                logger.warning(f"Failed to delete primary URL embeddings: {e}")
            
            # If domain crawl, delete each page embeddings
            try:
                logger.debug(f"Checking for crawled pages for URL ID: '{url_id_str}'")
                pages = self.rag_manager.url_manager.get_pages_for_parent(url_id_str)
                logger.debug(f"Found {len(pages)} pages to delete for URL ID: '{url_id_str}'")
                for page_url in pages:
                    try:
                        page_doc_id = hashlib.sha1(page_url.strip().encode('utf-8')).hexdigest()[:16]
                        logger.debug(f"Deleting page embeddings for: {page_url} (doc_id: {page_doc_id})")
                        self.rag_manager.milvus_manager.delete_document(document_id=page_doc_id)
                    except Exception as de:
                        logger.warning(f"Failed to delete page embeddings for {page_url}: {de}")
                # Remove page records
                logger.debug(f"Deleting page records for URL ID: '{url_id_str}'")
                self.rag_manager.url_manager.delete_pages_for_parent(url_id_str)
            except Exception as e:
                logger.warning(f"Failed to clean up url_pages: {e}")

            # Finally remove the URL record
            logger.debug(f"Deleting URL record for ID: '{url_id_str}'")
            result = self.rag_manager.url_manager.delete_url(url_id_str)
            if result['success']:
                logger.info(f"URL deletion completed successfully for ID: '{url_id_str}'")
                flash('URL and related embeddings deleted successfully', 'success')
            else:
                logger.error(f"Failed to delete URL ID '{url_id_str}': {result['message']}")
                flash(f'Failed to delete URL: {result["message"]}', 'error')
            return redirect(url_for('index'))

        @self.app.route('/delete_url_bg/<url_id>', methods=['GET', 'POST'])
        def delete_url_bg(url_id):
            """Start background deletion of a URL and its embeddings, with progress bar."""
            logger.info(f"DELETE_URL_BG ROUTE HIT: url_id={url_id}, method={request.method}")
            
            if request.method == 'GET':
                logger.warning(f"GET request received for delete_url_bg (should be POST): url_id={url_id}")
                flash('Invalid request method. Please use the delete button.', 'error')
                return redirect(url_for('index'))
                
            try:
                logger.info(f"URL ID is UUID string: {url_id}")
                # URL ID is a UUID string, don't convert to int
                url_id_str = str(url_id)
                logger.info(f"Using UUID string for deletion: {url_id_str}")
            except Exception as e:
                logger.error(f"Failed to process URL ID: {url_id}, error: {e}")
                flash('Invalid URL ID', 'error')
                return redirect(url_for('index'))
                
            logger.info(f"Getting URL record for ID: {url_id_str}")
            url_rec = self.rag_manager.url_manager.get_url_by_id(url_id_str)
            logger.info(f"URL record retrieved: {url_rec is not None}")
            if not url_rec:
                logger.warning(f"URL not found for ID: {url_id_str}")
                flash('URL not found', 'error')
                return redirect(url_for('index'))
            
            # If already processing something (refresh or delete), don't start another
            logger.info(f"Checking if URL {url_id_str} is already being processed")
            if url_id_str in self.rag_manager.url_processing_status:
                logger.info(f"URL {url_id_str} already being processed, skipping")
                flash('An operation is already in progress for this URL', 'info')
                return redirect(url_for('index'))
            
            # Seed a status entry so the UI shows a progress bar immediately after redirect
            logger.info(f"Creating processing status for URL {url_id_str}")
            self.rag_manager.url_processing_status[url_id_str] = URLProcessingStatus(
                url=url_rec.get('url', ''),
                title=url_rec.get('title')
            )
            
            logger.info(f"Starting background deletion thread for URL {url_id_str}")
            import threading
            th = threading.Thread(target=self.rag_manager._delete_url_background, args=(url_id_str,))
            th.daemon = True
            th.start()
            logger.info(f"Background deletion thread started for URL {url_id_str}")
            flash('Deletion started in background', 'info')
            logger.info(f"DELETE_URL_BG completing successfully for URL {url_id_str}")
            return redirect(url_for('index'))

        @self.app.route('/update_url/<url_id>', methods=['POST'])
        def update_url(url_id):
            """Update URL metadata including title, description, and refresh schedule."""
            logger.info(f"update_url route called with url_id: '{url_id}' (type: {type(url_id)})")
            
            # UUID validation - ensure it's a valid UUID string
            try:
                import uuid
                # Test if url_id is a valid UUID
                uuid.UUID(url_id)
                url_id_str = str(url_id)
                logger.debug(f"Valid UUID string for update: '{url_id_str}'")
            except ValueError as e:
                logger.error(f"Invalid URL ID format for update: '{url_id}' - {e}")
                flash('Invalid URL ID format', 'error')
                return redirect(url_for('index'))
                
            title = request.form.get('title')
            description = request.form.get('description')
            refresh_raw = request.form.get('refresh_interval_minutes')
            crawl_domain_flag = 1 if request.form.get('crawl_domain') in ('on', '1', 'true', 'True') else 0
            ignore_robots_flag = 1 if request.form.get('ignore_robots') in ('on', '1', 'true', 'True') else 0
            
            # Optional retention fields
            sr_raw = request.form.get('snapshot_retention_days')
            sm_raw = request.form.get('snapshot_max_snapshots')
            
            refresh_interval_minutes = None
            if refresh_raw:
                try:
                    refresh_interval_minutes = int(refresh_raw)
                    if refresh_interval_minutes < 0:
                        refresh_interval_minutes = None
                except ValueError:
                    refresh_interval_minutes = None
            
            snapshot_retention_days = 0  # Default to 0 (unlimited)
            snapshot_max_snapshots = 0   # Default to 0 (unlimited)
            try:
                if sr_raw and sr_raw.strip() != '':
                    v = int(sr_raw)
                    if v >= 0:
                        snapshot_retention_days = v
            except ValueError:
                snapshot_retention_days = 0  # Invalid input defaults to unlimited
            try:
                if sm_raw and sm_raw.strip() != '':
                    v = int(sm_raw)
                    if v >= 0:
                        snapshot_max_snapshots = v
            except ValueError:
                snapshot_max_snapshots = 0  # Invalid input defaults to unlimited

            logger.debug(f"Updating URL metadata for ID: '{url_id_str}'")
            logger.debug(f"Form data - title: '{title}', description: '{description}', refresh: {refresh_interval_minutes}")
            
            result = self.rag_manager.url_manager.update_url_metadata(
                url_id_str, title, description, refresh_interval_minutes,
                crawl_domain_flag, ignore_robots_flag,
                snapshot_retention_days, snapshot_max_snapshots
            )
            if result.get('success'):
                logger.info(f"URL metadata updated successfully for ID: '{url_id_str}'")
                flash('URL metadata updated', 'success')
            else:
                logger.error(f"Failed to update URL metadata for ID '{url_id_str}': {result.get('message','Unknown error')}")
                flash(f"Failed to update URL: {result.get('message','Unknown error')}", 'error')
            return redirect(url_for('index'))

        @self.app.route('/ingest_url/<url_id>', methods=['POST'])
        def ingest_url(url_id):
            """Trigger immediate ingestion/refresh for a URL."""
            logger.info(f"ingest_url route called with url_id: '{url_id}' (type: {type(url_id)})")
            
            # UUID validation - ensure it's a valid UUID string
            try:
                import uuid
                # Test if url_id is a valid UUID
                uuid.UUID(url_id)
                url_id_str = str(url_id)
                logger.debug(f"Valid UUID string: '{url_id_str}'")
            except ValueError as e:
                logger.error(f"Invalid URL ID format: '{url_id}' - {e}")
                flash('Invalid URL ID format', 'error')
                return redirect(url_for('index'))
                
            logger.debug(f"Looking up URL with ID: '{url_id_str}'")
            url_rec = self.rag_manager.url_manager.get_url_by_id(url_id_str)
            if not url_rec:
                logger.error(f"URL not found for ID: '{url_id_str}'")
                flash('URL not found', 'error')
                return redirect(url_for('index'))
            
            logger.info(f"Found URL record: {url_rec.get('url', 'N/A')} (title: {url_rec.get('title', 'N/A')})")
            
            # Initialize and start background refresh with progress if not already running
            if url_id_str in self.rag_manager.url_processing_status:
                logger.warning(f"Refresh already in progress for URL ID: '{url_id_str}'")
                flash('Refresh already in progress for this URL', 'info')
                return redirect(url_for('index'))
            
            logger.debug(f"Creating processing status for URL ID: '{url_id_str}'")
            self.rag_manager.url_processing_status[url_id_str] = URLProcessingStatus(
                url=url_rec.get('url', ''),
                title=url_rec.get('title')
            )
            
            logger.info(f"Starting background refresh thread for URL ID: '{url_id_str}'")
            import threading
            th = threading.Thread(target=self.rag_manager._process_url_background, args=(url_rec['id'],))
            th.daemon = True
            th.start()
            logger.info(f"Background refresh thread started successfully for URL: {url_rec.get('url', 'N/A')}")
            flash('Refresh started in background', 'info')
            return redirect(url_for('index'))

        @self.app.route('/email_accounts', methods=['GET'])
        def list_email_accounts():
            """Return JSON list of configured email accounts."""
            try:
                if hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                    accounts = self.rag_manager.email_account_manager.list_accounts(include_password=False)
                    return jsonify(accounts)
                else:
                    return jsonify([])
            except Exception as e:
                logger.error(f"Failed to fetch email accounts: {e}")
                return jsonify({"error": "Failed to fetch email accounts"}), 500

        @self.app.route('/email_accounts', methods=['POST'])
        def add_email_account():
            """Create a new email account configuration."""
            account_name = request.form.get('account_name', '').strip()
            server_type = request.form.get('server_type', 'imap').strip().lower() or 'imap'
            server = request.form.get('server', '').strip()
            email_address = request.form.get('email_address', '').strip()
            password = request.form.get('password', '').strip()
            port_str = request.form.get('port', '').strip()
            mailbox = request.form.get('mailbox', '').strip() or None
            batch_limit_str = request.form.get('batch_limit', '').strip()
            refresh_raw = request.form.get('refresh_interval_minutes', '').strip()
            use_ssl = request.form.get('use_ssl') in ('1', 'on')

            logger.info(
                "Request to add email account '%s' (%s) on %s for %s",
                account_name, server_type, server, email_address,
            )

            if not all([account_name, server, email_address, password, port_str]):
                flash('Missing required fields', 'error')
                return redirect(url_for('index'))

            try:
                port = int(port_str)
                batch_limit = int(batch_limit_str) if batch_limit_str else None
                refresh_interval = int(refresh_raw) if refresh_raw else self.config.EMAIL_DEFAULT_REFRESH_MINUTES
            except ValueError:
                flash('Port, batch limit and refresh interval must be numbers', 'error')
                return redirect(url_for('index'))

            record = {
                'account_name': account_name,
                'server_type': server_type,
                'server': server,
                'port': port,
                'email_address': email_address,
                'password': password,
                'mailbox': mailbox,
                'batch_limit': batch_limit,
                'use_ssl': 1 if use_ssl else 0,
                'refresh_interval_minutes': refresh_interval,
            }

            try:
                if hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                    self.rag_manager.email_account_manager.create_account(record)
                    logger.info("Email account '%s' added successfully", account_name)
                    flash('Email account added successfully', 'success')
                else:
                    flash('Email account manager not available', 'error')
            except Exception as e:
                logger.exception("Failed to add email account '%s': %s", account_name, e)
                flash(f'Failed to add email account: {e}', 'error')

            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<account_id>', methods=['POST'])
        def update_email_account(account_id: str):
            """Update an existing email account configuration."""
            account_name = request.form.get('account_name', '').strip()
            server = request.form.get('server', '').strip()
            email_address = request.form.get('email_address', '').strip()
            password = request.form.get('password', '').strip()
            port_str = request.form.get('port', '').strip()
            mailbox = request.form.get('mailbox', '').strip() or None
            batch_limit_str = request.form.get('batch_limit', '').strip()
            refresh_raw = request.form.get('refresh_interval_minutes', '').strip()
            last_synced_offset_str = request.form.get('last_synced_offset', '').strip()
            use_ssl = request.form.get('use_ssl') in ('1', 'on')
            server_type = request.form.get('server_type', '').strip()

            updates: Dict[str, Any] = {}
            if account_name:
                updates['account_name'] = account_name
            if server:
                updates['server'] = server
            if server_type:
                updates['server_type'] = server_type.lower()
            if email_address:
                updates['email_address'] = email_address
            if password:
                updates['password'] = password
            if mailbox is not None:
                updates['mailbox'] = mailbox
            if batch_limit_str:
                try:
                    updates['batch_limit'] = int(batch_limit_str)
                except ValueError:
                    flash('Batch limit must be a number', 'error')
                    return redirect(url_for('index'))
            if port_str:
                try:
                    updates['port'] = int(port_str)
                except ValueError:
                    flash('Port must be a number', 'error')
                    return redirect(url_for('index'))
            if refresh_raw:
                try:
                    updates['refresh_interval_minutes'] = int(refresh_raw)
                except ValueError:
                    flash('Refresh interval must be a number', 'error')
                    return redirect(url_for('index'))
            if last_synced_offset_str:
                try:
                    updates['last_synced_offset'] = int(last_synced_offset_str)
                except ValueError:
                    flash('Last synced offset must be a number', 'error')
                    return redirect(url_for('index'))
            updates['use_ssl'] = 1 if use_ssl else 0

            if not updates:
                flash('No fields to update', 'error')
                return redirect(url_for('index'))

            try:
                if hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                    self.rag_manager.email_account_manager.update_account(account_id, updates)
                    flash('Email account updated successfully', 'success')
                else:
                    flash('Email account manager not available', 'error')
            except Exception as e:
                flash(f'Failed to update email account: {e}', 'error')

            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<account_id>/delete', methods=['POST'])
        def delete_email_account(account_id: str):
            """Remove an email account configuration."""
            try:
                if hasattr(self.rag_manager, 'email_account_manager') and self.rag_manager.email_account_manager:
                    self.rag_manager.email_account_manager.delete_account(account_id)
                    flash('Email account deleted', 'success')
                else:
                    flash('Email account manager not available', 'error')
            except Exception as e:
                flash(f'Failed to delete email account: {e}', 'error')

            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<account_id>/refresh', methods=['POST'])
        def refresh_email_account(account_id: str):
            """Trigger immediate sync for an email account."""
            if not hasattr(self.rag_manager, 'email_account_manager') or not self.rag_manager.email_account_manager:
                flash('Email ingestion not configured', 'error')
                return redirect(url_for('index'))

            import threading
            th = threading.Thread(target=self.rag_manager._refresh_email_account_background, args=(account_id,))
            th.daemon = True
            th.start()
            flash('Email refresh started in background', 'info')
            return redirect(url_for('index'))

        @self.app.route('/email/<email_id>')
        def display_email(email_id: str):
            """Display email content in a popup modal."""
            try:
                logger.debug(f"Displaying email with ID: {email_id}")
                
                # Get email content from PostgreSQL
                if not (hasattr(self.rag_manager, 'email_account_manager') and 
                       self.rag_manager.email_account_manager):
                    logger.error("Email account manager not available")
                    return jsonify({"error": "Email system not available"}), 500
                
                # Get database connection via central PostgreSQLManager
                from rag_manager.managers.postgres_manager import PostgreSQLManager
                pg_mgr = None
                if hasattr(self.rag_manager, 'database_manager') and hasattr(self.rag_manager.database_manager, 'postgresql_manager'):
                    pg_mgr = self.rag_manager.database_manager.postgresql_manager
                else:
                    pg_mgr = PostgreSQLManager()

                with pg_mgr.get_connection() as conn:
                    with conn.cursor() as cursor:
                            # Query email by email_id
                            cursor.execute("""
                                SELECT email_id, account_id, subject, sender, recipients, 
                                       email_date, content, content_type, message_id, 
                                       has_attachments, thread_id, in_reply_to, folder
                                FROM emails 
                                WHERE email_id = %s
                                LIMIT 1
                            """, (email_id,))

                            email_row = cursor.fetchone()

                            if not email_row:
                                logger.warning(f"Email not found: {email_id}")
                                return jsonify({"error": "Email not found"}), 404

                            # Extract email data (dict-style row expected)
                            email_data = {
                                'email_id': email_row.get('email_id'),
                                'account_id': email_row.get('account_id'),
                                'subject': email_row.get('subject') or "No Subject",
                                'sender': email_row.get('sender') or "Unknown Sender",
                                'recipients': email_row.get('recipients') or "",
                                'email_date': str(email_row.get('email_date')) if email_row.get('email_date') else "Unknown Date",
                                'content': email_row.get('content') or "",
                                'content_type': email_row.get('content_type') or "text/plain",
                                'message_id': email_row.get('message_id') or "",
                                'has_attachments': bool(email_row.get('has_attachments')),
                                'thread_id': email_row.get('thread_id') or "",
                                'in_reply_to': email_row.get('in_reply_to') or "",
                                'folder': email_row.get('folder') or "inbox"
                            }

                            # Get account info for context
                            cursor.execute("""
                                SELECT account_name, email_address 
                                FROM email_accounts 
                                WHERE account_id = %s
                            """, (email_data['account_id'],))

                            account_row = cursor.fetchone()
                            if account_row:
                                email_data['account_name'] = account_row.get('account_name') or account_row.get('name')
                                email_data['account_email'] = account_row.get('email_address') or account_row.get('email')

                            # Get attachments if any
                            if email_data['has_attachments']:
                                cursor.execute("""
                                    SELECT attachment_id, filename, content_type, size
                                    FROM email_attachments 
                                    WHERE email_id = %s
                                """, (email_id,))

                                attachments = []
                                for att_row in cursor.fetchall():
                                    attachments.append({
                                        'attachment_id': att_row.get('attachment_id') if isinstance(att_row, dict) else att_row[0],
                                        'filename': att_row.get('filename') if isinstance(att_row, dict) else att_row[1],
                                        'content_type': att_row.get('content_type') if isinstance(att_row, dict) else att_row[2],
                                        'size': att_row.get('size') if isinstance(att_row, dict) else att_row[3]
                                    })
                                email_data['attachments'] = attachments
                            else:
                                email_data['attachments'] = []
                
                # If we created a temporary manager above, close its pool
                try:
                    if pg_mgr and getattr(pg_mgr, 'config', None) is not None:
                        pg_mgr.close()
                except Exception:
                    pass
                
                # Format content for display
                if email_data['content_type'] == 'text/html':
                    # For HTML emails, sanitize but preserve formatting
                    import html
                    email_data['content_display'] = email_data['content']
                    email_data['is_html'] = True
                else:
                    # For plain text, escape HTML and preserve line breaks
                    import html
                    escaped_content = html.escape(email_data['content'])
                    email_data['content_display'] = escaped_content.replace('\n', '<br>')
                    email_data['is_html'] = False
                
                logger.info(f"Successfully retrieved email: {email_id}")
                return jsonify(email_data)
                
            except Exception as e:
                logger.error(f"Failed to display email {email_id}: {e}", exc_info=True)
                return jsonify({"error": f"Failed to load email: {str(e)}"}), 500

    def _get_directory_files(self, directory: str) -> list:
        """
        Get list of files in a directory with metadata enrichment.
        
        Args:
            directory: Directory path to scan
            
        Returns:
            List of file dictionaries with metadata for uploaded files
        """
        from datetime import datetime
        
        files = []
        try:
            if not os.path.exists(directory):
                return files
                
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if not os.path.isfile(file_path):
                    continue
                    
                stat = os.stat(file_path)
                entry = {
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'status': self.rag_manager.processing_status.get(filename)
                }
                
                # Enrich uploaded files with metadata from database
                if directory == self.config.UPLOADED_FOLDER:
                    try:
                        meta = self.rag_manager.document_manager.get_document_metadata(filename) or {} if self.rag_manager.document_manager else {}
                        entry['title'] = meta.get('title') or self.rag_manager._fallback_title_from_filename(filename)
                        
                        # Debug logging to see what metadata we have
                        logger.info(f"Metadata for {filename}: {meta}")
                        
                        # Get counts with multiple fallback keys and type coercion
                        def _safe_int(val):
                            if val is None:
                                return None
                            try:
                                return int(float(str(val)))
                            except (ValueError, TypeError):
                                return None
                        
                        # Try multiple possible keys for each count type
                        chunks_val = (meta.get('chunk_count') or meta.get('chunks_count') or 
                                    meta.get('chunkCount') or meta.get('chunks'))
                        page_val = (meta.get('page_count') or meta.get('pages') or 
                                  meta.get('pageCount') or meta.get('page'))
                        word_val = (meta.get('word_count') or meta.get('words') or 
                                  meta.get('wordCount') or meta.get('word'))
                        
                        entry['chunks_count'] = _safe_int(chunks_val)
                        entry['page_count'] = _safe_int(page_val)
                        entry['word_count'] = _safe_int(word_val)
                        
                        # Debug what we extracted
                        logger.info(f"Extracted counts for {filename}: chunks={entry['chunks_count']}, pages={entry['page_count']}, words={entry['word_count']}")
                        
                        # Ensure top_keywords is a list, not a string
                        top_keywords = meta.get('top_keywords')
                        
                        if isinstance(top_keywords, str):
                            # Handle JSON string or comma-separated string
                            import json
                            try:
                                parsed = json.loads(top_keywords)
                                entry['top_keywords'] = parsed
                            except (json.JSONDecodeError, ValueError):
                                # Fall back to comma-split
                                entry['top_keywords'] = [kw.strip() for kw in top_keywords.split(',') if kw.strip()]
                        elif isinstance(top_keywords, list):
                            entry['top_keywords'] = top_keywords
                        else:
                            entry['top_keywords'] = []
                            
                        # Skip files marked for deletion
                        st = entry['status']
                        if st and st.status == 'completed' and (st.message or '').lower().startswith('deletion complete'):
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to get metadata for {filename}: {e}")
                        entry['title'] = self.rag_manager._fallback_title_from_filename(filename)
                        entry['top_keywords'] = []
                        
                files.append(entry)
                
            return sorted(files, key=lambda x: x['modified'], reverse=True)
        except Exception as e:
            logger.error(f"Error reading directory {directory}: {e}")
            return []

    def _get_file_database_status(self, filename: str) -> str | None:
        """Get the processing status for a file from the database."""
        try:
            from rag_manager.managers.postgres_manager import PostgreSQLManager
            mgr = PostgreSQLManager()
            with mgr.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT processing_status FROM documents 
                        WHERE title = %s 
                           OR (document_type = 'file' AND file_path LIKE %s)
                           OR (document_type = 'url' AND file_path = %s)
                           OR (filename IS NOT NULL AND filename = %s)
                        ORDER BY 
                            CASE processing_status 
                                WHEN 'completed' THEN 1 
                                WHEN 'pending' THEN 2 
                                ELSE 3 
                            END
                        LIMIT 1
                    """, (filename, f'%/{filename}', filename, filename))
                    result = cursor.fetchone()
                    logger.debug(f"Direct DB query for {filename}: result = {result}")
                    if result:
                        status = result.get('processing_status') if isinstance(result, dict) else None
                        logger.debug(f"Found status for {filename}: {status}")
                        return status
                    else:
                        logger.debug(f"No database record found for {filename}")
                        return None
        except Exception as e:
            logger.warning(f"Error checking database status for {filename}: {e}")
            logger.debug(f"Exception details: {type(e)} {e.args}")
        return None

    def _allowed_file(self, filename: str) -> bool:
        """
        Check if file extension is allowed.
        
        Args:
            filename: Name of the file to check
            
        Returns:
            True if file extension is allowed, False otherwise
        """
        return ('.' in filename and 
                filename.rsplit('.', 1)[1].lower() in self.config.ALLOWED_EXTENSIONS)

    def _get_database_status(self) -> Dict[str, Any]:
        """Get PostgreSQL database connection status."""
        try:
            if hasattr(self.rag_manager, 'database_manager') and hasattr(self.rag_manager.database_manager, 'postgresql_manager'):
                return self.rag_manager.database_manager.postgresql_manager.get_version_info()
            else:
                # Fallback - create a temporary PostgreSQL manager to get version
                from rag_manager.managers.postgres_manager import PostgreSQLManager
                temp_pg_manager = PostgreSQLManager()
                status = temp_pg_manager.get_version_info()
                temp_pg_manager.close()
                return status
        except Exception as e:
            return {"connected": False, "error": f"PostgreSQL: {str(e)}"}

    def _get_enriched_urls(self) -> list:
        """Get URLs enriched with robots.txt information."""
        try:
            urls = self.rag_manager.url_manager.get_all_urls()
            enriched_urls = []
            for u in urls:
                try:
                    rp, crawl_delay = self.rag_manager._get_robots(u.get('url', ''))
                    try:
                        allowed = rp.can_fetch(self.config.CRAWL_USER_AGENT, u.get('url', ''))
                    except Exception:
                        allowed = True
                    try:
                        has_entries = bool(getattr(rp, 'default_entry', None) or getattr(rp, 'entries', []))
                    except Exception:
                        has_entries = False
                    u['robots_allowed'] = 1 if allowed else 0
                    u['robots_has_rules'] = 1 if (has_entries or crawl_delay is not None) else 0
                    u['robots_crawl_delay'] = crawl_delay
                except Exception:
                    u['robots_allowed'] = None
                    u['robots_has_rules'] = None
                    u['robots_crawl_delay'] = None
                # Add vector DB metrics: chunk count per URL (by source + document_id with legacy fallback)
                try:
                    src = u.get('url')
                    url_id = u.get('url_id')
                    u['chunk_count'] = self.rag_manager.milvus_manager.get_chunk_count_for_url(src, url_id) if src else 0
                except Exception:
                    u['chunk_count'] = 0
                # Placeholder metric for pages discovered (requires url_pages schema)
                u.setdefault('pages_discovered', None)
                # Expose a simple pages value for the UI (default to 1 until we track discovered pages)
                try:
                    u['pages'] = u['pages_discovered'] if u['pages_discovered'] is not None else 1
                except Exception:
                    u['pages'] = 1
                    
                    # Add child URL statistics for parent URLs
                    try:
                        url_id = u.get('url_id')
                        if url_id:
                            child_stats = self.rag_manager.url_manager.get_child_url_stats(url_id)
                            u['child_stats'] = child_stats
                            # Check if this URL has any children
                            u['has_children'] = child_stats['total_children'] > 0
                            # Calculate progress for parent URLs with children
                            if u['has_children']:
                                total = child_stats['total_children']
                                completed = child_stats['completed_children']
                                processing = child_stats['processing_children']
                                u['child_progress_percent'] = (completed / total * 100) if total > 0 else 0
                                u['is_parent_with_active_children'] = processing > 0
                    except Exception as e:
                        uid = u.get('url_id')
                        if not uid:
                            logger.warning(f"Missing canonical 'url_id' for URL record while fetching child stats: {u}")
                        logger.debug(f"Error getting child stats for URL {uid}: {e}")
                        u['child_stats'] = {'total_children': 0, 'processing_children': 0, 'completed_children': 0, 'failed_children': 0}
                        u['has_children'] = False
                        u['child_progress_percent'] = 0
                        u['is_parent_with_active_children'] = False
                # Append the enriched URL to the result list
                enriched_urls.append(u)
            return enriched_urls
        except Exception as e:
            logger.warning(f"Error building enriched URLs: {e}")
            return []
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of a file for duplicate detection.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Hexadecimal string representation of the file hash
        """
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            raise
