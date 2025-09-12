#!/usr/bin/env python
"""
Snapshot Cleanup Utility
Simple utility to run cleanup operations for URL snapshots.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def run_snapshot_cleanup(snapshot_service, postgres_manager) -> Dict[str, Any]:
    """
    Run snapshot cleanup for all URLs with retention policies.
    
    Args:
        snapshot_service: URLSnapshotService instance
        postgres_manager: PostgreSQL manager instance
        
    Returns:
        Dictionary with cleanup summary
    """
    try:
        # Get all URLs with retention policies
        urls_with_policies = _get_urls_with_retention_policies(postgres_manager)
        
        total_cleaned = 0
        processed_urls = 0
        errors = []
        
        for url_data in urls_with_policies:
            try:
                url_id = url_data['url_id']
                url = url_data['url']
                retention_days = url_data['snapshot_retention_days']
                max_snapshots = url_data['snapshot_max_snapshots']
                
                logger.debug(f"Running cleanup for URL {url_id} (retention: {retention_days}d, max: {max_snapshots})")
                
                cleanup_result = snapshot_service.cleanup_old_snapshot_files(url, retention_days, max_snapshots)
                
                if cleanup_result.get('success'):
                    cleaned_count = cleanup_result.get('deleted', 0)
                    total_cleaned += cleaned_count
                    
                    if cleaned_count > 0:
                        logger.info(f"Cleaned up {cleaned_count} snapshots for URL {url_id}")
                else:
                    error_msg = f"Cleanup failed for URL {url_id}: {cleanup_result.get('error')}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                
                processed_urls += 1
                
            except Exception as e:
                error_msg = f"Error cleaning URL {url_data.get('url_id', 'unknown')}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        result = {
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'processed_urls': processed_urls,
            'total_cleaned': total_cleaned,
            'errors': errors
        }
        
        if total_cleaned > 0:
            logger.info(f"Cleanup completed: processed {processed_urls} URLs, cleaned {total_cleaned} files")
        
        return result
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


def _get_urls_with_retention_policies(postgres_manager) -> List[Dict[str, Any]]:
    """
    Get all active URLs that have retention policies set.
    
    Args:
        postgres_manager: PostgreSQL manager instance
        
    Returns:
        List of URL records with retention policies
    """
    try:
        with postgres_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id AS url_id, url, snapshot_retention_days, snapshot_max_snapshots
                    FROM urls 
                    WHERE status = 'active'
                        AND (snapshot_retention_days > 0 OR snapshot_max_snapshots > 0)
                    ORDER BY last_crawled DESC NULLS LAST
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
    except Exception as e:
        logger.error(f"Failed to get URLs with retention policies: {e}")
        raise
