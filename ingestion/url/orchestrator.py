"""
URL Processing Orchestrator for RAG Document Handler
Coordinates URL processing tasks including domain crawling and PDF snapshots.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from .source_manager import URLSourceManager

logger = logging.getLogger(__name__)


class URLOrchestrator:
    """Periodically process content from configured URLs with snapshot support."""

    def __init__(
        self,
        config,
        url_manager: Optional[URLSourceManager] = None,
        processor: Optional[Any] = None,  # URL processor will be implemented later
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
        if self.postgres_manager:
            from rag_manager.data.document_data import DocumentDataManager
            self.document_data_manager = DocumentDataManager(self.postgres_manager, config=self.config)
        else:
            self.document_data_manager = None
        self.urls: List[Dict[str, Any]] = []

        # Initialize domain crawler for discovering new URLs with robots.txt support
        self.domain_crawler = None
        if self.url_manager:
            from ingestion.url.utils.domain_crawler import DomainCrawler
            self.domain_crawler = DomainCrawler(self.url_manager, respect_robots=True)
            logger.info("Domain crawler initialized with robots.txt enforcement")

        logger.info("URL Orchestrator initialized with async domain crawling support")

        self.refresh_urls()

    def refresh_urls(self) -> None:
        """Reload the URL configurations."""
        if not self.url_manager:
            return
        try:
            logger.debug("Refreshing URL configurations.")
            self.urls = self.url_manager.get_all_urls_including_children()
            logger.debug("URLs retrieved: %s", len(self.urls))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Failed to load URLs: {exc}")

    def get_due_urls(self) -> List[Dict[str, Any]]:
        """Return URLs that are due for processing based on their refresh intervals."""
        if not self.url_manager:
            return []
        
        try:
            due_urls = self.url_manager.get_due_urls()
            logger.debug(f"Found {len(due_urls)} URLs due for processing")
            return due_urls
        except Exception as exc:
            logger.error(f"Failed to get due URLs: {exc}")
            return []

    def get_url_count(self) -> int:
        """Return the total number of configured URLs."""
        if not self.url_manager:
            return 0
        
        try:
            return self.url_manager.url_data.get_url_count()
        except Exception as exc:
            logger.error(f"Failed to get URL count: {exc}")
            return 0

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
            self.url_manager.set_refreshing(url_id, True)
            
            # Create snapshot (always enabled)
            snapshot_created = False
            snapshot_result = None
            doc_id = None
            snapshot_id = None
            
            if self.snapshot_service:
                try:
                    logger.info(f"Creating snapshot for URL {url_id} (snapshots always enabled)")
                    
                    # Create the snapshot
                    snapshot_result = await self.snapshot_service.create_snapshot(url_id, url_string)
                    
                    if snapshot_result.get("success"):
                        # Check if content has changed before storing
                        content_hash = snapshot_result.get("content_hash")
                        if trace_logger:
                            trace_logger.info(f"Snapshot created; content_hash={content_hash}")
                        
                        if self.snapshot_service.check_content_changed(url_id, content_hash):
                            # Store snapshot as document
                            store_result = self.snapshot_service.store_snapshot_document(
                                url_id, url_string, snapshot_result
                            )
                            
                            if store_result and store_result.get("document_id"):
                                doc_id = store_result["document_id"]
                                snapshot_id = store_result.get("snapshot_id")
                                logger.info(f"Stored snapshot document {doc_id} with snapshot_id {snapshot_id} for URL {url_id}")
                                snapshot_created = True
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
                                
                                # Update URL content hash
                                self.url_manager.update_url_hash_status(
                                    url_id, content_hash, "snapshot_created"
                                )
                                
                                # Clean up old snapshots based on retention policies
                                cleanup_result = self.snapshot_service.cleanup_old_snapshots(url_id)
                                if cleanup_result.get("deleted", 0) > 0:
                                    logger.info(f"Cleaned up {cleanup_result['deleted']} old snapshots for URL {url_id}")
                                    if trace_logger:
                                        trace_logger.info(f"Cleaned up {cleanup_result['deleted']} old snapshots")
                            else:
                                logger.warning(f"Failed to store snapshot document for URL {url_id}")
                        else:
                            logger.info(f"No content changes detected for URL {url_id}, skipping snapshot storage")
                    else:
                        logger.warning(f"Snapshot creation failed for URL {url_id}: {snapshot_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Snapshot processing failed for URL {url_id}: {e}")
                    if trace_logger:
                        trace_logger.exception(f"Snapshot processing failed: {e}")
                    # Continue with regular processing even if snapshot fails
            
            # Process URL content using the document processor
            if self.processor:
                try:
                    # Use document processor to extract and chunk content
                    chunks = self.processor.load_and_chunk_url(url_string, str(url_id))
                    logger.info(f"Extracted {len(chunks)} chunks from URL {url_id}")
                    if trace_logger:
                        trace_logger.info(f"Chunking result count={len(chunks)}")
                    
                    # CRITICAL: Snapshots are always required - fail fast if not created
                    if not snapshot_created or not doc_id:
                        error_msg = (f"CRITICAL FAILURE: Snapshot creation failed for URL {url_id}. "
                                   f"snapshot_created={snapshot_created}, doc_id={doc_id}. "
                                   f"Cannot proceed with chunk storage without a valid document ID. "
                                   f"This indicates an upstream issue with snapshot creation that must be fixed.")
                        logger.error(error_msg)
                        if trace_logger:
                            trace_logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    
                    # Store chunks in Milvus if available
                    if self.milvus_manager and chunks:
                        try:
                            # Use doc_id from successful snapshot creation
                            milvus_id = doc_id
                            if trace_logger:
                                trace_logger.info(f"Using milvus_id={milvus_id} for insert_documents (snapshot_created={snapshot_created})")

                            inserted = self.milvus_manager.insert_documents(milvus_id, chunks)
                            logger.info(f"Successfully stored {inserted} chunks from URL {url_id} in vector database (milvus_id={milvus_id})")
                            if trace_logger:
                                trace_logger.info(f"Milvus insert result: {inserted}")
                        except Exception as e:
                            logger.error(f"Failed to store chunks from URL {url_id} in vector database: {e}")
                            if trace_logger:
                                trace_logger.exception(f"Milvus insert failed: {e}")
                            raise
                    elif not self.milvus_manager:
                        logger.warning(f"URL {url_id}: Extracted {len(chunks)} chunks but no MilvusManager available for storage")

                    # Persist chunk rows to Postgres using DocumentDataManager directly
                    if not self.document_data_manager:
                        err = "No document_data_manager available on URLOrchestrator; cannot persist chunks"
                        logger.error(err)
                        if trace_logger:
                            trace_logger.error(err)
                        raise RuntimeError(err)

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

                    try:
                        stored = self.document_data_manager.persist_chunks(doc_identifier, chunks, trace_logger=trace_logger)
                        if trace_logger:
                            trace_logger.info(f"Persisted {stored}/{len(chunks)} chunks to Postgres for document_id={doc_identifier} via DocumentDataManager")
                    except Exception as e:
                        logger.error(f"CHUNK PERSISTENCE FAILED for URL {url_id}: {e}")
                        logger.error(f"Failed while persisting chunks with doc_identifier={doc_identifier}")
                        logger.error(f"Context: snapshot_created={snapshot_created}, doc_id={doc_id}, url_id={url_id}")
                        if "foreign key constraint" in str(e).lower():
                            logger.error(f"FOREIGN KEY VIOLATION: Document {doc_identifier} does not exist in 'documents' table!")
                            logger.error(f"This indicates snapshot creation failed but chunk persistence was still attempted")
                        if trace_logger:
                            trace_logger.exception(f"Chunk persistence failed: {e}")
                        raise  # Re-raise to surface the error immediately
                    
                except Exception as e:
                    logger.error(f"Failed to process URL content for {url_id}: {e}")
                    if trace_logger:
                        trace_logger.exception(f"Content processing failed: {e}")
                    raise
            else:
                logger.warning("No document processor available for URL content processing")
            
            # Mark URL as processed
            refresh_interval = url_record.get('refresh_interval_minutes')
            self.url_manager.mark_scraped(url_id, refresh_interval)
            if trace_logger:
                try:
                    trace_logger.info(f"Marked URL {url_id} as scraped; refresh_interval={refresh_interval}")
                    trace_logger.info(f"Completed ingestion trace for URL {url_id} at {datetime.utcnow().isoformat()}Z")
                except Exception:
                    pass
            
            # Perform domain crawling if enabled
            domain_crawl_result = None
            if url_record.get('crawl_domain') and self.domain_crawler and url_string:
                try:
                    logger.info(f"Starting domain crawl for URL {url_id} (crawl_domain enabled)")
                    
                    # Get robots.txt setting for this URL
                    ignore_robots = url_record.get('ignore_robots', False)
                    
                    # Discover new URLs in the domain (async call)
                    discovery_result = await self.domain_crawler.discover_domain_urls(
                        str(url_string), ignore_robots=ignore_robots
                    )
                    
                    if discovery_result.get("success") and discovery_result.get("discovered_urls"):
                        # Add discovered URLs to the system
                        from urllib.parse import urlparse
                        parsed_url = urlparse(str(url_string))
                        domain = str(parsed_url.netloc)
                        
                        add_result = self.domain_crawler.add_discovered_urls(
                            discovery_result["discovered_urls"], domain, url_id
                        )
                        
                        # Refresh URL cache after adding new URLs
                        logger.info(f"Refreshing URL cache after domain crawl for URL {url_id}")
                        self.refresh_urls()
                        
                        domain_crawl_result = {
                            **discovery_result,
                            **add_result
                        }
                        
                        logger.info(f"Domain crawl completed for URL {url_id}: "
                                   f"discovered {discovery_result['new_urls_found']} new URLs, "
                                   f"added {add_result['added_count']} to system")
                    else:
                        domain_crawl_result = discovery_result
                        logger.info(f"Domain crawl for URL {url_id} found no new URLs to add")
                        
                except Exception as e:
                    logger.error(f"Domain crawling failed for URL {url_id}: {e}")
                    domain_crawl_result = {"success": False, "error": str(e)}
            
            # Mark URL as no longer being processed
            self.url_manager.set_refreshing(url_id, False)
            
            logger.info(f"Successfully processed URL {url_id}")
            return {
                "success": True, 
                "url_id": url_id,
                "url": url_string,
                "message": "URL processed successfully",
                "snapshot_created": snapshot_created,
                "snapshot_result": snapshot_result,
                "domain_crawl_result": domain_crawl_result
            }
            
        except Exception as exc:
            logger.error(f"Failed to process URL {url_id}: {exc}")
            try:
                # Make sure to clear the refreshing flag on error
                self.url_manager.set_refreshing(url_id, False)
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
                "last_scraped": url_record.get('last_scraped'),
                "next_refresh": url_record.get('next_refresh'),
                "refresh_interval_minutes": url_record.get('refresh_interval_minutes')
            }
            
        except Exception as exc:
            logger.error(f"Failed to get processing status for URL {url_id}: {exc}")
            return {"status": "error", "message": str(exc)}
