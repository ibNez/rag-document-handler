#!/usr/bin/env python
"""
Snapshot Cleanup Scheduler
Independent cleanup process that runs regularly to enforce retention policies.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from rag_manager.managers.postgres_manager import PostgreSQLManager
from rag_manager.core.config import Config
from ingestion.url.utils.snapshot_service import URLSnapshotService

logger = logging.getLogger(__name__)


class SnapshotCleanupScheduler:
    """
    Independent cleanup scheduler that enforces retention policies for all URLs.
    
    Runs on its own schedule and can be called on-demand from other processes.
    """
    
    def __init__(self, config: Optional[Config] = None, postgres_manager: Optional[PostgreSQLManager] = None):
        """
        Initialize cleanup scheduler.
        
        Args:
            config: Configuration object
            postgres_manager: PostgreSQL manager instance
        """
        self.config = config or Config()
        self.postgres_manager = postgres_manager or PostgreSQLManager()
        self.snapshot_service = URLSnapshotService(self.postgres_manager, self.config)
        self.running = False
        self.cleanup_interval = 30  # Run every 30 seconds
        
        logger.info(f"Snapshot cleanup scheduler initialized (interval: {self.cleanup_interval}s)")
    
    async def start_scheduler(self) -> None:
        """Start the cleanup scheduler loop."""
        self.running = True
        logger.info("Starting snapshot cleanup scheduler")
        
        while self.running:
            try:
                # Run cleanup for all URLs
                cleanup_summary = await self.run_cleanup_cycle()
                
                if cleanup_summary.get('total_cleaned', 0) > 0:
                    logger.info(f"Cleanup cycle completed: {cleanup_summary}")
                else:
                    logger.debug(f"Cleanup cycle completed: no files cleaned")
                
                # Wait for next cycle
                await asyncio.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup scheduler cycle: {e}")
                # Continue running even if one cycle fails
                await asyncio.sleep(self.cleanup_interval)
    
    def stop_scheduler(self) -> None:
        """Stop the cleanup scheduler."""
        logger.info("Stopping snapshot cleanup scheduler")
        self.running = False
    
    async def run_cleanup_cycle(self) -> Dict[str, Any]:
        """
        Run a complete cleanup cycle for all URLs.
        
        Returns:
            Dictionary with cleanup summary
        """
        try:
            # Get all URLs with their retention settings
            urls_with_policies = self._get_urls_with_retention_policies()
            
            total_cleaned = 0
            processed_urls = 0
            errors = []
            
            for url_data in urls_with_policies:
                try:
                    url_id = url_data['id']
                    url = url_data['url']
                    retention_days = url_data['snapshot_retention_days']
                    max_snapshots = url_data['snapshot_max_snapshots']
                    
                    # Only run cleanup if there are actual policies set
                    if retention_days > 0 or max_snapshots > 0:
                        logger.debug(f"Running cleanup for URL {url_id} (retention: {retention_days}d, max: {max_snapshots})")
                        
                        cleanup_result = self.snapshot_service.cleanup_old_snapshot_files(url, retention_days, max_snapshots)
                        
                        if cleanup_result.get('success'):
                            cleaned_count = cleanup_result.get('deleted', 0)
                            total_cleaned += cleaned_count
                            
                            if cleaned_count > 0:
                                logger.info(f"Cleaned up {cleaned_count} snapshots for URL {url_id} ({url_data['url']})")
                        else:
                            error_msg = f"Cleanup failed for URL {url_id}: {cleanup_result.get('error')}"
                            logger.warning(error_msg)
                            errors.append(error_msg)
                    
                    processed_urls += 1
                    
                except Exception as e:
                    error_msg = f"Error cleaning URL {url_data.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            return {
                'success': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'processed_urls': processed_urls,
                'total_cleaned': total_cleaned,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Cleanup cycle failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _get_urls_with_retention_policies(self) -> List[Dict[str, Any]]:
        """
        Get all active URLs that have retention policies set.
        
        Returns:
            List of URL records with retention policies
        """
        try:
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, url, snapshot_retention_days, snapshot_max_snapshots
                        FROM urls 
                        WHERE status = 'active'
                            AND (snapshot_retention_days > 0 OR snapshot_max_snapshots > 0)
                        ORDER BY last_crawled DESC NULLS LAST
                    """)
                    
                    return [dict(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Failed to get URLs with retention policies: {e}")
            return []
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """
        Get current status of the cleanup scheduler.
        
        Returns:
            Status dictionary
        """
        urls_with_policies = self._get_urls_with_retention_policies()
        
        return {
            'running': self.running,
            'cleanup_interval': self.cleanup_interval,
            'urls_with_policies': len(urls_with_policies),
            'last_check': datetime.now(timezone.utc).isoformat()
        }


# Singleton instance for shared use
_cleanup_scheduler: Optional[SnapshotCleanupScheduler] = None


def get_cleanup_scheduler(config: Optional[Config] = None, postgres_manager: Optional[PostgreSQLManager] = None) -> SnapshotCleanupScheduler:
    """
    Get or create the global cleanup scheduler instance.
    
    Args:
        config: Configuration object
        postgres_manager: PostgreSQL manager instance
        
    Returns:
        SnapshotCleanupScheduler instance
    """
    global _cleanup_scheduler
    
    if _cleanup_scheduler is None:
        _cleanup_scheduler = SnapshotCleanupScheduler(config, postgres_manager)
    
    return _cleanup_scheduler


async def run_cleanup_scheduler():
    """
    Standalone function to run the cleanup scheduler.
    Can be called from main application or as a separate process.
    """
    scheduler = get_cleanup_scheduler()
    await scheduler.start_scheduler()


if __name__ == "__main__":
    # Allow running as standalone script
    import sys
    import os
    
    # Add project root to Python path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        try:
            await run_cleanup_scheduler()
        except KeyboardInterrupt:
            logger.info("Cleanup scheduler stopped by user")
        except Exception as e:
            logger.error(f"Cleanup scheduler failed: {e}")
    
    asyncio.run(main())
