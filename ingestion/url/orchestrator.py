"""
URL Processing Orchestrator for RAG Document Handler
Coordinates URL processing tasks including domain crawling and PDF snapshots.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from rag_manager.data.url_data import URLDataManager

logger = logging.getLogger(__name__)


class URLOrchestrator:
    """Periodically process content from configured URLs with snapshot support."""

    def __init__(
        self,
        config,
        url_manager: Optional[URLDataManager] = None,
        processor: Optional[Any] = None,  # URL processor 
        milvus_manager: Optional[Any] = None,  # For storing chunks in vector database
        snapshot_service: Optional[Any] = None,  # For creating PDF snapshots
        postgres_manager: Optional[Any] = None,  # Explicit Postgres manager for chunk persistence
    ) -> None:
        """Initialize URL orchestrator with dependencies."""
        self.config = config
        self.url_manager = url_manager
        self.processor = processor
        self.milvus_manager = milvus_manager
        self.snapshot_service = snapshot_service
        # Prefer explicit postgres_manager if provided; otherwise try to infer from milvus_manager
        self.postgres_manager = postgres_manager or (getattr(self.milvus_manager, 'postgres_manager', None) if self.milvus_manager else None)
        
        # Initialize document data manager directly for chunk persistence
        if not self.postgres_manager:
            raise RuntimeError("URLOrchestrator requires postgres_manager to function. Cannot initialize without database connection.")
        
        from rag_manager.data.document_data import DocumentDataManager
        from rag_manager.data.url_data import URLDataManager
        
        self.document_data_manager = DocumentDataManager(self.postgres_manager, config=self.config)
        self.url_data_manager = URLDataManager(self.postgres_manager)
        self.urls: List[Dict[str, Any]] = []

        # Initialize domain crawler for discovering new URLs with robots.txt support
        self.domain_crawler = None
        if self.url_manager:
            from ingestion.url.utils.domain_crawler import DomainCrawler
            self.domain_crawler = DomainCrawler(self.url_manager, document_manager=self.document_data_manager, respect_robots=True)
            logger.info("Domain crawler initialized with robots.txt enforcement")

        logger.info("URL Orchestrator initialized with async domain crawling support")

        self.refresh_urls()

    def refresh_urls(self) -> None:
        """Reload the URL configurations."""
        if not self.url_manager:
            return
        try:
            logger.debug("Refreshing URL configurations.")
            # Get all URLs from data layer and add hierarchy information
            all_urls = self.url_data_manager.get_all_urls()
            
            # Add hierarchy information
            for url in all_urls:
                if url.get('parent_url_id'):
                    # Mark as child
                    url['is_child'] = True
                    url['hierarchy_level'] = 1  # Could be extended for deeper nesting
                else:
                    # Mark as parent
                    url['is_child'] = False
                    url['hierarchy_level'] = 0
            
            self.urls = all_urls
            logger.debug("URLs retrieved: %s", len(self.urls))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to load URLs: {exc}")

    async def process_url(self, url_id: str) -> Dict[str, Any]:
        """
        Process a single URL: extract content, create snapshots, and discover domain URLs.
        
        Args:
            url_id: UUID of the URL to process
            
        Returns:
            Dict with processing results
        """
        if not self.url_manager:
            return {"success": False, "error": "URL manager not available"}
        try:
            # Get URL details
            # PostgreSQLURLManager canonicalizes the primary key to 'url_id'
            urls = [url for url in self.urls if url.get('url_id') == url_id]
            if not urls:
                # Refresh URLs and try again
                logger.info(f"URL {url_id} not found in cache, refreshing URL list...")
                self.refresh_urls()
                urls = [url for url in self.urls if url.get('url_id') == url_id]
                if not urls:
                    logger.error(f"URL {url_id} not found even after refresh")
                    return {"success": False, "error": f"URL {url_id} not found"}
            
            url_record = urls[0]
            url_string = url_record.get('url')
            
            logger.info(f"Processing URL {url_id}: {url_string}")
            # Create a per-run trace logger to capture step-by-step ingestion events
            trace_logger = None
            try:
                traces_dir = os.path.join(getattr(self.config, 'LOG_DIR', 'logs'), 'ingestion_traces')
                os.makedirs(traces_dir, exist_ok=True)
                trace_log_path = os.path.join(traces_dir, f"{url_id}.log")
                trace_logger = logging.getLogger(f"ingest_trace_{url_id}")
                # Avoid duplicating handlers if this logger already exists
                if not trace_logger.handlers:
                    fh = logging.FileHandler(trace_log_path)
                    fh.setLevel(logging.INFO)
                    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                    fh.setFormatter(formatter)
                    trace_logger.addHandler(fh)
                trace_logger.setLevel(logging.INFO)
                trace_logger.info(f"Starting ingestion trace for URL {url_id} ({url_string}) at {datetime.utcnow().isoformat()}Z")
            except Exception:
                # Best-effort tracing: don't fail ingestion if trace setup fails
                trace_logger = None
            
            # Mark URL as being processed
            self.url_data_manager.set_refreshing(url_id, True)
            
            snapshot_result = None
            doc_id = None
            snapshot_id = None
            
            if self.snapshot_service:
                try:
                    logger.info(f"Creating snapshot for URL {url_id}")
                    
                    # Create the snapshot
                    # Returns "file_path", "pdf_file", "json_file", "created_at"
                    snapshot_result = await self.snapshot_service.create_snapshot_files(url_id, url_string)
                    
                    if snapshot_result.get("success"):
                        # Check if content has changed before storing
                        content_hash = snapshot_result.get("content_hash")
                        if trace_logger:
                            trace_logger.info(f"Snapshot created; content_hash={content_hash}")
                        
                        if self.snapshot_service.check_content_changed(url_id, content_hash):
                            # Store snapshot to document table in Postgres
                            existing_document = None  # Initialize to track if we created a new document
                            try:
                                # Extract metadata from snapshot result
                                file_path = snapshot_result["file_path"]
                                pdf_file = snapshot_result["pdf_file"]
                                json_file = snapshot_result["json_file"]
                                created_at = snapshot_result["created_at"]
                                # Extract the actual page title from the URL
                                page_title = self.url_data_manager.extract_title_from_url(str(url_string))
                            
                                filename = str(url_string)
                                
                                # Check if document already exists for this URL
                                existing_document = self.snapshot_service._find_existing_document(url_id, url_string)
                                
                                if existing_document:
                                    # Document exists - just create snapshot entry
                                    doc_id = existing_document["document_id"]
                                    logger.info(f"Found existing document {doc_id} for URL {url_id}, creating snapshot entry only")
                                else:
                                    # Document doesn't exist - create new document directly via data manager
                                    doc_id = self.document_data_manager.store_document(
                                        file_path=str(url_string),  # Store the URL as string
                                        filename=filename,
                                        title=page_title,
                                        content_type="text/html",
                                        document_type="url",
                                        parent_url_id=url_id  # Link to parent URL
                                    )
                                    logger.info(f"Created new document {doc_id} for URL {url_id}")

                                # Create url_snapshots entry linking this snapshot to the document with file paths
                                if self.postgres_manager:
                                    with self.postgres_manager.get_connection() as conn:
                                        with conn.cursor() as cursor:
                                            cursor.execute("""
                                                INSERT INTO url_snapshots (url_id, document_id, file_path, pdf_file, json_file, created_at) 
                                                VALUES (%s, %s, %s, %s, %s, %s) 
                                                RETURNING id
                                            """, (
                                                url_id, 
                                                doc_id, 
                                                file_path,  # Full path to PDF file
                                                pdf_file,  # PDF filename only
                                                json_file,  # JSON filename only
                                                created_at
                                            ))
                                            result = cursor.fetchone()
                                            snapshot_id = str(result['id'])
                                            conn.commit()
                                            logger.debug(f"Created url_snapshots entry {snapshot_id} with file_path={file_path}")

                                logger.info(f"Stored snapshot document {doc_id} with snapshot_id {snapshot_id} for URL {url_id}")
                                if trace_logger:
                                    trace_logger.info(f"Stored snapshot document_id={doc_id} snapshot_id={snapshot_id}")
                                    # Also attach a per-document trace logger that writes to the same file
                                    try:
                                        doc_trace_logger = logging.getLogger(f"ingest_trace_{doc_id}")
                                        if not doc_trace_logger.handlers:
                                            for h in trace_logger.handlers:
                                                doc_trace_logger.addHandler(h)
                                        doc_trace_logger.setLevel(trace_logger.level)
                                        doc_trace_logger.info(f"Linked document trace to URL trace: doc_id={doc_id} url_id={url_id}")
                                    except Exception:
                                        # Best-effort linking of tracegers
                                        pass
                                        
                                # Use document processor to extract and chunk content
                                if not self.processor:
                                    logger.error("Document processor is not initialized. Cannot extract and chunk URL content.")
                                    raise RuntimeError("Document processor is not initialized. Please provide a valid processor instance.")
                                chunks = self.processor.load_and_chunk_url(url_string, str(url_id))
                                logger.info(f"Extracted {len(chunks)} chunks from URL {url_id}")
                                if trace_logger:
                                    trace_logger.info(f"Chunking result count={len(chunks)}")
                                
                                # Store chunks in Milvus if available
                                if self.milvus_manager and chunks:
                                   
                                    # Use doc_id from successful snapshot creation
                                    milvus_id = doc_id
                                    if trace_logger:
                                        trace_logger.info(f"Using milvus_id={milvus_id} for insert_documents")

                                    inserted = self.milvus_manager.insert_documents(milvus_id, chunks)
                                    logger.info(f"Successfully stored {inserted} chunks from URL {url_id} in vector database (milvus_id={milvus_id})")
                                    if trace_logger:
                                        trace_logger.info(f"Milvus insert result: {inserted}")

                                elif not self.milvus_manager:
                                    logger.error(f"URL {url_id}: Extracted {len(chunks)} chunks but no MilvusManager available for storage")
                                    raise RuntimeError(f"URL {url_id}: Extracted {len(chunks)} chunks but no MilvusManager available for storage")
                                # Use doc_id from successful snapshot creation
                                doc_identifier = doc_id
                                
                                # Add snapshot_id to chunk metadata for proper temporal linking
                                if snapshot_id:
                                    for chunk in chunks:
                                        if hasattr(chunk, 'metadata'):
                                            chunk.metadata['snapshot_id'] = snapshot_id
                                            logger.debug(f"Added snapshot_id {snapshot_id} to chunk metadata")
                                
                                if trace_logger:
                                    trace_logger.info(f"Persisting {len(chunks)} chunks to Postgres for document_id={doc_identifier} snapshot_id={snapshot_id} via DocumentDataManager")

                                stored = self.document_data_manager.persist_chunks(doc_identifier, chunks, trace_logger=trace_logger)
                                if trace_logger:
                                    trace_logger.info(f"Persisted {stored}/{len(chunks)} chunks to Postgres for document_id={doc_identifier} via DocumentDataManager")
                                
                                # Update URL status to reflect successful completion and update last_crawled
                                self.url_data_manager.update_url_hash_status(
                                    url_id, content_hash, "success"
                                )
                                
                                # Mark URL as no longer being processed
                                self.url_data_manager.set_refreshing(url_id, False)
                                # Always run cleanup at the end of processing to catch any retention policy changes
                                logger.debug(f"Running final cleanup check for URL {url_id}")
                                try:
                                    if self.snapshot_service:
                                        retention_days = url_record.get('snapshot_retention_days', 0) or 0
                                        max_snapshots = url_record.get('snapshot_max_snapshots', 0) or 0
                                        final_cleanup_result = self.snapshot_service.cleanup_old_snapshot_files(url_string, retention_days, max_snapshots)
                                        if final_cleanup_result.get("deleted", 0) > 0:
                                            logger.info(f"Final cleanup removed {final_cleanup_result['deleted']} additional snapshots for URL {url_id}")
                                        elif final_cleanup_result.get("success"):
                                            logger.debug(f"Final cleanup completed for URL {url_id} - no additional files to remove")
                                except Exception as cleanup_error:
                                    logger.warning(f"Final cleanup failed for URL {url_id}: {cleanup_error}")

                            except Exception as e:
                                logger.error(f"Failed to store snapshot document for URL {url_id}: {e}")
                                
                                # Rollback: Clean up what was created during this failed run
                                try:
                                    # 1. Delete snapshot record from url_snapshots if it was created
                                    if snapshot_id and self.snapshot_service:
                                        logger.error(f"Rolling back: deleting snapshot record {snapshot_id}")
                                        self.snapshot_service._delete_snapshot_record_by_id(snapshot_id)
                                        logger.error(f"Rolled back snapshot record {snapshot_id}")
                                    
                                    # 2. Delete document chunks by snapshot_id if any were created
                                    if snapshot_id and self.snapshot_service:
                                        logger.error(f"Rolling back: deleting document chunks for snapshot {snapshot_id}")
                                        self.snapshot_service._delete_snapshot_document_chunks(snapshot_id)
                                        logger.error(f"Rolled back document chunks for snapshot {snapshot_id}")
                                    
                                    # 3. Delete embeddings from Milvus by snapshot_id if any were created
                                    if snapshot_id and self.snapshot_service:
                                        logger.error(f"Rolling back: deleting Milvus embeddings for snapshot {snapshot_id}")
                                        self.snapshot_service._delete_snapshot_milvus_embeddings(snapshot_id)
                                        logger.error(f"Rolled back Milvus embeddings for snapshot {snapshot_id}")
                                    
                                    # 4. Delete newly created document if not existing
                                    if doc_id and not existing_document and self.postgres_manager:
                                        logger.error(f"Rolling back: deleting newly created document {doc_id}")
                                        with self.postgres_manager.get_connection() as conn:
                                            with conn.cursor() as cursor:
                                                cursor.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                                                conn.commit()
                                                logger.error(f"Rolled back document {doc_id}")
                                    
                                    # 5. Delete physical snapshot files that were created
                                    if snapshot_result.get("success"):
                                        file_path = snapshot_result.get("file_path")
                                        pdf_file = snapshot_result.get("pdf_file")
                                        json_file = snapshot_result.get("json_file")
                                        
                                        if file_path and pdf_file:
                                            pdf_path = os.path.join(file_path, pdf_file)
                                            if os.path.exists(pdf_path):
                                                os.remove(pdf_path)
                                                logger.error(f"Rolled back PDF file: {pdf_path}")
                                        
                                        if file_path and json_file:
                                            json_path = os.path.join(file_path, json_file)
                                            if os.path.exists(json_path):
                                                os.remove(json_path)
                                                logger.error(f"Rolled back JSON file: {json_path}")
                                                
                                except Exception as rollback_error:
                                    logger.error(f"Rollback failed for URL {url_id}: {rollback_error}")
                                
                                # Reset variables since rollback occurred
                                doc_id = None
                                snapshot_id = None
                                
                                # Update URL status to reflect the failure
                                self.url_data_manager.update_url_hash_status(
                                    url_id, content_hash, "snapshot_failed"
                                )
                        else:
                            logger.info(f"No content changes detected for URL {url_id}, skipping all processing")
                            
                            # Update URL content hash and status
                            self.url_data_manager.update_url_hash_status(
                                url_id, content_hash, "no_changes"
                            )
                            
                            if trace_logger:
                                trace_logger.info(f"No content changes - skipping document processing")
                    else:
                        logger.warning(f"Snapshot creation failed for URL {url_id}: {snapshot_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Snapshot processing failed for URL {url_id}: {e}")
                    if trace_logger:
                        trace_logger.exception(f"Snapshot processing failed: {e}")
                    
                    
                    
                    # Continue domain crawl even if snapshot fails
            

            # Mark URL as processed
            refresh_interval = url_record.get('refresh_interval_minutes')
            self.url_data_manager.mark_scraped(url_id, refresh_interval)
            if trace_logger:
                try:
                    trace_logger.info(f"Marked URL {url_id} as scraped; refresh_interval={refresh_interval}")
                    trace_logger.info(f"Completed ingestion trace for URL {url_id} at {datetime.utcnow().isoformat()}Z")
                except Exception:
                    pass
            
            # Perform domain crawling if enabled
            domain_crawl_result = None
            if url_record.get('crawl_domain') and self.domain_crawler and url_string:
                # Ensure required services are available for domain crawling
                if not self.snapshot_service:
                    logger.error("Snapshot service required for domain crawling but not available")
                    domain_crawl_result = {"success": False, "error": "Snapshot service not available"}
                elif not self.processor:
                    logger.error("Document processor required for domain crawling but not available")
                    domain_crawl_result = {"success": False, "error": "Document processor not available"}
                elif not self.postgres_manager:
                    logger.error("Postgres manager required for domain crawling but not available")
                    domain_crawl_result = {"success": False, "error": "Postgres manager not available"}
                else:
                    try:
                        logger.info(f"Starting domain crawl for URL {url_id} (crawl_domain enabled)")
                        
                        # Get robots.txt setting for this URL
                        ignore_robots = url_record.get('ignore_robots', False)
                        
                        # Discover new URLs in the domain (async call)
                        discovery_result = await self.domain_crawler.discover_domain_urls(
                            str(url_string), ignore_robots=ignore_robots
                        )
                        
                        if discovery_result.get("success") and discovery_result.get("discovered_urls"):
                            discovered_urls = discovery_result["discovered_urls"]
                            logger.info(f"Discovered {len(discovered_urls)} URLs to process for domain crawl")
                            
                            # Process each discovered URL with the same pipeline as the main URL
                            processed_count = 0
                            failed_count = 0
                            
                            for discovered_url in discovered_urls:
                                # Initialize rollback variables
                                page_snapshot_result = None
                                page_snapshot_id = None 
                                page_doc_id = None
                                page_existing_document = None
                                
                                try:
                                    logger.info(f"Processing discovered URL: {discovered_url}")
                                    
                                    # Check if document already exists for this discovered URL in documents table
                                    existing_url_record = None
                                    existing_url_record = self.snapshot_service._find_existing_document(url_id, discovered_url)
                                    discovered_page_title = self.url_data_manager.extract_title_from_url(str(discovered_url))
                                    
                                    if existing_url_record:
                                        # Document exists - just create snapshot entry
                                        discovered_document_id = existing_url_record["document_id"]
                                        logger.info(f"Found existing document {discovered_document_id} for discovered URL {discovered_url}")
                                    else:
                                        # Document doesn't exist - create new document directly via data manager
                                        discovered_document_id = self.document_data_manager.store_document(
                                            file_path=str(discovered_url),  # Store the URL as string
                                            filename=str(discovered_url),
                                            title=discovered_page_title,
                                            content_type="text/html",
                                            document_type="url",
                                            parent_url_id=url_id  # Link to parent URL
                                        )
                                        logger.info(f"Created new document {discovered_document_id} for discovered URL {discovered_url}")
                                    
                                    # 3a.) Create snapshot for discovered URL (using parent url_id)
                                    page_snapshot_result = await self.snapshot_service.create_snapshot_files(url_id, discovered_url)
                                    
                                    if not page_snapshot_result.get("success"):
                                        logger.error(f"Failed to create snapshot for discovered URL {discovered_url}")
                                        failed_count += 1
                                        continue
                                    
                                    # 3b.) Check if snapshot has new content
                                    page_content_hash = page_snapshot_result.get("content_hash")
                                    if not self.snapshot_service.check_content_changed(url_id, page_content_hash):
                                        logger.info(f"No content changes for discovered URL {discovered_url}, skipping processing")
                                        continue
                                    
                                    # 3c.) Use the document we created/found earlier
                                    page_existing_document = existing_url_record
                                    page_doc_id = discovered_document_id
                                    
                                    # 3d.) Create url_snapshots record (using parent url_id)
                                    page_file_path = page_snapshot_result["file_path"]
                                    page_pdf_file = page_snapshot_result["pdf_file"]
                                    page_json_file = page_snapshot_result["json_file"]
                                    page_created_at = page_snapshot_result["created_at"]
                                    
                                    with self.postgres_manager.get_connection() as conn:
                                        with conn.cursor() as cursor:
                                            cursor.execute("""
                                                INSERT INTO url_snapshots (url_id, document_id, file_path, pdf_file, json_file, created_at) 
                                                VALUES (%s, %s, %s, %s, %s, %s) 
                                                RETURNING id
                                            """, (
                                                url_id,  # Use parent URL ID
                                                page_doc_id,
                                                page_file_path,
                                                page_pdf_file,
                                                page_json_file,
                                                page_created_at
                                            ))
                                            result = cursor.fetchone()
                                            page_snapshot_id = str(result['id'])
                                            conn.commit()
                                            logger.debug(f"Created url_snapshots entry {page_snapshot_id} for discovered URL")
                                    
                                    # 3e.) Create chunks for discovered URL
                                    page_chunks = self.processor.load_and_chunk_url(discovered_url, str(url_id))
                                    logger.info(f"Extracted {len(page_chunks)} chunks from discovered URL {discovered_url}")
                                    
                                    # 3f.) Store embeddings chunks to Milvus
                                    if self.milvus_manager and page_chunks:
                                        page_inserted = self.milvus_manager.insert_documents(page_doc_id, page_chunks)
                                        logger.info(f"Stored {page_inserted} chunks from discovered URL {discovered_url} in Milvus")
                                    elif not self.milvus_manager:
                                        logger.error(f"No MilvusManager available for discovered URL {discovered_url}")
                                        raise RuntimeError(f"No MilvusManager available for discovered URL {discovered_url}")
                                    
                                    # 3g.) Update chunks with snapshot_id
                                    if page_snapshot_id:
                                        for chunk in page_chunks:
                                            if hasattr(chunk, 'metadata'):
                                                chunk.metadata['snapshot_id'] = page_snapshot_id
                                                logger.debug(f"Added snapshot_id {page_snapshot_id} to chunk metadata")
                                    
                                    # 3h.) Create document_chunks in document_chunks table
                                    page_stored = self.document_data_manager.persist_chunks(page_doc_id, page_chunks)
                                    logger.info(f"Persisted {page_stored}/{len(page_chunks)} chunks for discovered URL {discovered_url}")
                                    
                                    # 3i.) Run snapshot cleanup for discovered URL (using parent url_id)
                                    page_retention_days = url_record.get('snapshot_retention_days', 0) or 0
                                    page_max_snapshots = url_record.get('snapshot_max_snapshots', 0) or 0
                                    page_cleanup_result = self.snapshot_service.cleanup_old_snapshot_files(
                                        discovered_url, page_retention_days, page_max_snapshots
                                    )
                                    if page_cleanup_result.get("deleted", 0) > 0:
                                        logger.info(f"Cleaned up {page_cleanup_result['deleted']} old snapshots for discovered URL {discovered_url}")
                                    
                                    # Update URL hash status (using parent url_id)
                                    self.url_data_manager.update_url_hash_status(
                                        url_id, page_content_hash, "snapshot_created"
                                    )
                                    
                                    processed_count += 1
                                    logger.info(f"Successfully processed discovered URL {discovered_url}")
                                    
                                except Exception as page_error:
                                    logger.error(f"Failed to process discovered URL {discovered_url}: {page_error}")
                                    
                                    # Rollback: Clean up what was created during this failed run for discovered URL
                                    try:
                                        # 1. Delete snapshot record from url_snapshots if it was created
                                        if page_snapshot_id and self.snapshot_service:
                                            logger.info(f"Rolling back: deleting snapshot record {page_snapshot_id} for discovered URL")
                                            self.snapshot_service._delete_snapshot_record_by_id(page_snapshot_id)
                                            logger.info(f"Rolled back snapshot record {page_snapshot_id} for discovered URL")
                                        
                                        # 2. Delete document chunks by snapshot_id if any were created
                                        if page_snapshot_id and self.snapshot_service:
                                            logger.info(f"Rolling back: deleting document chunks for snapshot {page_snapshot_id} for discovered URL")
                                            self.snapshot_service._delete_snapshot_document_chunks(page_snapshot_id)
                                            logger.info(f"Rolled back document chunks for snapshot {page_snapshot_id} for discovered URL")
                                        
                                        # 3. Delete embeddings from Milvus by snapshot_id if any were created
                                        if page_snapshot_id and self.snapshot_service:
                                            logger.info(f"Rolling back: deleting Milvus embeddings for snapshot {page_snapshot_id} for discovered URL")
                                            self.snapshot_service._delete_snapshot_milvus_embeddings(page_snapshot_id)
                                            logger.info(f"Rolled back Milvus embeddings for snapshot {page_snapshot_id} for discovered URL")
                                        
                                        # 4. Delete newly created document if not existing
                                        if page_doc_id and not page_existing_document and self.postgres_manager:
                                            logger.info(f"Rolling back: deleting newly created document {page_doc_id} for discovered URL")
                                            with self.postgres_manager.get_connection() as conn:
                                                with conn.cursor() as cursor:
                                                    cursor.execute("DELETE FROM documents WHERE id = %s", (page_doc_id,))
                                                    conn.commit()
                                                    logger.info(f"Rolled back document {page_doc_id} for discovered URL")
                                        
                                        # 5. Delete physical snapshot files that were created
                                        if page_snapshot_result and page_snapshot_result.get("success"):
                                            page_file_path = page_snapshot_result.get("file_path")
                                            page_pdf_file = page_snapshot_result.get("pdf_file")
                                            page_json_file = page_snapshot_result.get("json_file")
                                            
                                            if page_file_path and page_pdf_file:
                                                page_pdf_path = os.path.join(page_file_path, page_pdf_file)
                                                if os.path.exists(page_pdf_path):
                                                    os.remove(page_pdf_path)
                                                    logger.info(f"Rolled back PDF file for discovered URL: {page_pdf_path}")
                                            
                                            if page_file_path and page_json_file:
                                                page_json_path = os.path.join(page_file_path, page_json_file)
                                                if os.path.exists(page_json_path):
                                                    os.remove(page_json_path)
                                                    logger.info(f"Rolled back JSON file for discovered URL: {page_json_path}")
                                                    
                                    except Exception as page_rollback_error:
                                        logger.error(f"Rollback failed for discovered URL {discovered_url}: {page_rollback_error}")
                                    
                                    failed_count += 1
                                    continue
                            
                            # Final refresh after processing all discovered URLs
                            self.refresh_urls()
                            
                            domain_crawl_result = {
                                **discovery_result,
                                "processed_count": processed_count,
                                "failed_count": failed_count,
                                "total_discovered": len(discovered_urls)
                            }
                            
                            logger.info(f"Domain crawl completed for URL {url_id}: "
                                       f"discovered {len(discovered_urls)} URLs, "
                                       f"processed {processed_count}, failed {failed_count}")
                        else:
                            domain_crawl_result = discovery_result
                            logger.info(f"Domain crawl for URL {url_id} found no new URLs to process")
                            
                    except Exception as e:
                        logger.error(f"Domain crawling failed for URL {url_id}: {e}")
                        domain_crawl_result = {"success": False, "error": str(e)}
            
            # Mark URL as no longer being processed
            self.url_data_manager.set_refreshing(url_id, False)
            
            
            
            logger.info(f"Successfully processed URL {url_id}")
            return {
                "success": True, 
                "url_id": url_id,
                "url": url_string,
                "message": "URL processed successfully",
                "snapshot_result": snapshot_result,
                "domain_crawl_result": domain_crawl_result
            }
            
        except Exception as exc:
            logger.error(f"Failed to process URL {url_id}: {exc}")
            try:
                # Make sure to clear the refreshing flag on error
                self.url_data_manager.set_refreshing(url_id, False)
            except Exception:
                pass  # Don't fail on cleanup failure
            
            return {
                "success": False, 
                "url_id": url_id,
                "error": str(exc)
            }

    def get_processing_status(self, url_id: str) -> Dict[str, Any]:
        """
        Get the current processing status for a URL.
        
        Args:
            url_id: The ID of the URL to check
            
        Returns:
            Dictionary containing status information
        """
        if not self.url_manager:
            return {"status": "unavailable", "message": "URL manager not available"}
        
        try:
            # Get URL details
            urls = [url for url in self.urls if url.get('url_id') == url_id]
            if not urls:
                self.refresh_urls()
                urls = [url for url in self.urls if url.get('url_id') == url_id]
                if not urls:
                    return {"status": "not_found", "message": f"URL {url_id} not found"}
            
            url_record = urls[0]
            
            return {
                "status": "active" if url_record.get('refreshing') else "idle",
                "url_id": url_id,
                "url": url_record.get('url'),
                "last_scraped": url_record.get('last_crawled'),  # Fixed: use last_crawled from database
                "next_refresh": url_record.get('next_refresh'),
                "refresh_interval_minutes": url_record.get('refresh_interval_minutes')
            }
            
        except Exception as exc:
            logger.error(f"Failed to get processing status for URL {url_id}: {exc}")
            return {"status": "error", "message": str(exc)}
