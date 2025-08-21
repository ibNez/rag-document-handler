"""
URL Orchestrator for managing periodic URL processing and content synchronization.

This module defines the URLOrchestrator class, which is responsible for managing the periodic
synchronization of content from configured URLs. It interacts with PostgreSQL for URL
management and coordinates URL processing tasks.

Classes:
    URLOrchestrator: Handles URL processing tasks, including refreshing URL
    configurations and determining URLs due for processing.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

from .manager import PostgreSQLURLManager

logger = logging.getLogger(__name__)


class URLOrchestrator:
    """Periodically process content from configured URLs."""

    def __init__(
        self,
        config,
        url_manager: Optional[PostgreSQLURLManager] = None,
        processor: Optional[Any] = None,  # URL processor will be implemented later
        milvus_manager: Optional[Any] = None,  # For storing chunks in vector database
    ) -> None:
        """Initialize URL orchestrator with dependencies."""
        self.config = config
        self.url_manager = url_manager
        self.processor = processor
        self.milvus_manager = milvus_manager
        self.urls: List[Dict[str, Any]] = []
        self.refresh_urls()

    def refresh_urls(self) -> None:
        """Reload the URL configurations."""
        if not self.url_manager:
            return
        try:
            logger.debug("Refreshing URL configurations.")
            self.urls = self.url_manager.get_all_urls()
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

    def process_url(self, url_id: int) -> Dict[str, Any]:
        """
        Process a single URL by fetching its content and updating metadata.
        
        Args:
            url_id: The ID of the URL to process
            
        Returns:
            Dictionary containing processing results and status
        """
        if not self.url_manager:
            return {"success": False, "error": "URL manager not available"}
        
        try:
            # Get URL details
            urls = [url for url in self.urls if url.get('id') == url_id]
            if not urls:
                # Refresh URLs and try again
                self.refresh_urls()
                urls = [url for url in self.urls if url.get('id') == url_id]
                if not urls:
                    return {"success": False, "error": f"URL {url_id} not found"}
            
            url_record = urls[0]
            url_string = url_record.get('url')
            
            logger.info(f"Processing URL {url_id}: {url_string}")
            
            # Mark URL as being processed
            self.url_manager.set_refreshing(url_id, True)
            
            # Process URL content using the document processor
            if self.processor:
                try:
                    # Use document processor to extract and chunk content
                    chunks = self.processor.load_and_chunk_url(url_string, str(url_id))
                    logger.info(f"Extracted {len(chunks)} chunks from URL {url_id}")
                    
                    # Store chunks in vector database if milvus_manager is available
                    if self.milvus_manager and chunks:
                        try:
                            self.milvus_manager.insert_documents(chunks)
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
            
            # Mark URL as no longer being processed
            self.url_manager.set_refreshing(url_id, False)
            
            logger.info(f"Successfully processed URL {url_id}")
            return {
                "success": True, 
                "url_id": url_id,
                "url": url_string,
                "message": "URL processed successfully"
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
