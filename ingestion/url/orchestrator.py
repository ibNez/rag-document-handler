"""
URL Processing Orchestrator for RAG Document Handler
Coordinates URL processing tasks including domain crawling and PDF snapshots.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from .manager import PostgreSQLURLManager

logger = logging.getLogger(__name__)


class URLOrchestrator:
    """Periodically process content from configured URLs with snapshot support."""

    def __init__(
        self,
        config,
        url_manager: Optional[PostgreSQLURLManager] = None,
        processor: Optional[Any] = None,  # URL processor will be implemented later
        milvus_manager: Optional[Any] = None,  # For storing chunks in vector database
        snapshot_service: Optional[Any] = None,  # For creating PDF snapshots
    ) -> None:
        """Initialize URL orchestrator with dependencies."""
        self.config = config
        self.url_manager = url_manager
        self.processor = processor
        self.milvus_manager = milvus_manager
        self.snapshot_service = snapshot_service
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
            return self.url_manager.get_url_count()
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
            urls = [url for url in self.urls if url.get('id') == url_id]
            if not urls:
                # Refresh URLs and try again
                logger.info(f"URL {url_id} not found in cache, refreshing URL list...")
                self.refresh_urls()
                urls = [url for url in self.urls if url.get('id') == url_id]
                if not urls:
                    logger.error(f"URL {url_id} not found even after refresh")
                    return {"success": False, "error": f"URL {url_id} not found"}
            
            url_record = urls[0]
            url_string = url_record.get('url')
            
            logger.info(f"Processing URL {url_id}: {url_string}")
            
            # Mark URL as being processed
            self.url_manager.set_refreshing(url_id, True)
            
            # Create snapshot (always enabled)
            snapshot_created = False
            snapshot_result = None
            
            if self.snapshot_service:
                try:
                    logger.info(f"Creating snapshot for URL {url_id} (snapshots always enabled)")
                    
                    # Create the snapshot
                    snapshot_result = await self.snapshot_service.create_snapshot(url_id, url_string)
                    
                    if snapshot_result.get("success"):
                        # Check if content has changed before storing
                        content_hash = snapshot_result.get("content_hash")
                        
                        if self.snapshot_service.check_content_changed(url_id, content_hash):
                            # Store snapshot as document
                            doc_id = self.snapshot_service.store_snapshot_document(
                                url_id, url_string, snapshot_result
                            )
                            
                            if doc_id:
                                logger.info(f"Stored snapshot document {doc_id} for URL {url_id}")
                                snapshot_created = True
                                
                                # Update URL content hash
                                self.url_manager.update_url_hash_status(
                                    url_id, content_hash, "snapshot_created"
                                )
                                
                                # Clean up old snapshots based on retention policies
                                cleanup_result = self.snapshot_service.cleanup_old_snapshots(url_id)
                                if cleanup_result.get("deleted", 0) > 0:
                                    logger.info(f"Cleaned up {cleanup_result['deleted']} old snapshots for URL {url_id}")
                            else:
                                logger.warning(f"Failed to store snapshot document for URL {url_id}")
                        else:
                            logger.info(f"No content changes detected for URL {url_id}, skipping snapshot storage")
                    else:
                        logger.warning(f"Snapshot creation failed for URL {url_id}: {snapshot_result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Snapshot processing failed for URL {url_id}: {e}")
                    # Continue with regular processing even if snapshot fails
            
            # Process URL content using the document processor
            if self.processor:
                try:
                    # Use document processor to extract and chunk content
                    chunks = self.processor.load_and_chunk_url(url_string, str(url_id))
                    logger.info(f"Extracted {len(chunks)} chunks from URL {url_id}")
                    
                    # Store chunks in vector database if milvus_manager is available
                    if self.milvus_manager and chunks:
                        try:
                            # Use UUID for Milvus storage, not the URL string
                            # All metadata is managed by PostgreSQL as single source of truth
                            self.milvus_manager.insert_documents(str(url_id), chunks)
                            logger.info(f"Successfully stored {len(chunks)} chunks from URL {url_id} in vector database")
                        except Exception as e:
                            logger.error(f"Failed to store chunks from URL {url_id} in vector database: {e}")
                            raise
                    elif not self.milvus_manager:
                        logger.warning(f"URL {url_id}: Extracted {len(chunks)} chunks but no MilvusManager available for storage")
                    
                except Exception as e:
                    logger.error(f"Failed to process URL content for {url_id}: {e}")
                    raise
            else:
                logger.warning("No document processor available for URL content processing")
            
            # Mark URL as processed
            refresh_interval = url_record.get('refresh_interval_minutes')
            self.url_manager.mark_scraped(url_id, refresh_interval)
            
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

    def get_processing_status(self, url_id: int) -> Dict[str, Any]:
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
            urls = [url for url in self.urls if url.get('id') == url_id]
            if not urls:
                self.refresh_urls()
                urls = [url for url in self.urls if url.get('id') == url_id]
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
