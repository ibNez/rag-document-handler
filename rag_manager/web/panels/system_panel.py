"""
System Panel Statistics Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all statistics for the System panel on the status dashboard.
Centralizes connection status, processing queues, system health metrics, and capacity monitoring.
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from rag_manager.utils.disk_usage import (
    get_directory_size, 
    get_disk_usage, 
    get_postgres_database_size,
    get_postgres_table_sizes
)

logger = logging.getLogger(__name__)


class SystemPanelStats:
    """Statistics provider for the System panel."""
    
    def __init__(self, rag_manager):
        """Initialize with reference to the main RAG manager."""
        self.rag_manager = rag_manager
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all system statistics for the system panel.
        
        Returns:
            Dictionary containing system panel statistics
        """
        try:
            # Get disk usage and database stats
            disk_usage_stats = self._get_disk_usage_stats()
            database_size_stats = self._get_database_size_stats(disk_usage_stats)
            
            # Get threshold statuses for color coding
            threshold_stats = self._get_capacity_thresholds()
            
            return {
                # Connection status
                'sql_status': self._get_database_status(),
                'milvus_status': self._get_milvus_status(),
                
                # Processing queues
                'processing_files': len(self.rag_manager.processing_status),
                'processing_urls': len(self.rag_manager.url_processing_status),
                'processing_emails': len(self.rag_manager.email_processing_status),
                
                # Email ingestion monitoring
                'email_backlog': self._get_email_backlog_stats(),
                'email_ingest_age': self._get_email_ingest_age_stats(),
                
                # System health
                'scheduler_active': self._is_scheduler_active(),
                'total_processing': (
                    len(self.rag_manager.processing_status) + 
                    len(self.rag_manager.url_processing_status) + 
                    len(self.rag_manager.email_processing_status)
                ),
                
                # Disk usage and capacity metrics
                'disk_usage': disk_usage_stats,
                'database_size': database_size_stats,
                'total_usage': database_size_stats.get('total_usage', {"bytes": 0, "human": "0 B"}),
                
                # Threshold statuses for color coding
                'thresholds': threshold_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get system panel stats: {e}")
            return self._empty_stats()
    
    def _get_database_status(self) -> str:
        """Get PostgreSQL database connection status."""
        try:
            if not self.rag_manager.postgres_manager:
                return 'disconnected'
                
            # Test the connection
            with self.rag_manager.postgres_manager.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return 'connected'
            
        except Exception as e:
            logger.debug(f"Database status check failed: {e}")
            return 'error'
    
    def _get_milvus_status(self) -> str:
        """Get Milvus vector database connection status."""
        try:
            if not self.rag_manager.milvus_manager:
                return 'disconnected'
                
            status = self.rag_manager.milvus_manager.check_connection()
            
            if isinstance(status, dict):
                return 'connected' if status.get('connected', False) else 'error'
            elif isinstance(status, bool):
                return 'connected' if status else 'error'
            else:
                return 'error'
                
        except Exception as e:
            logger.debug(f"Milvus status check failed: {e}")
            return 'error'
    
    def _is_scheduler_active(self) -> bool:
        """Check if the background scheduler is active."""
        try:
            if hasattr(self.rag_manager, 'scheduler_manager') and self.rag_manager.scheduler_manager:
                # Check if scheduler thread is running
                if hasattr(self.rag_manager, '_scheduler_thread') and self.rag_manager._scheduler_thread:
                    return self.rag_manager._scheduler_thread.is_alive()
            return False
        except Exception as e:
            logger.debug(f"Scheduler status check failed: {e}")
            return False
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty system panel stats."""
        return {
            'sql_status': 'error',
            'milvus_status': 'error',
            'processing_files': 0,
            'processing_urls': 0,
            'processing_emails': 0,
            'scheduler_active': False,
            'total_processing': 0,
            'disk_usage': {
                'logs_size': {"bytes": 0, "human": "Error", "exists": False},
                'staging_size': {"bytes": 0, "human": "Error", "exists": False},
                'uploaded_size': {"bytes": 0, "human": "Error", "exists": False},
                'deleted_size': {"bytes": 0, "human": "Error", "exists": False},
                'snapshots_size': {"bytes": 0, "human": "Error", "exists": False},
                'disk_capacity': {
                    "total_human": "Unknown",
                    "free_human": "Unknown",
                    "free_percent": 0,
                    "warning_level": "unknown"
                }
            },
            'database_size': {
                'postgres_total': {"bytes": 0, "human": "Error", "error": "System error"},
                'milvus_estimate': {"bytes": 0, "human": "Error", "error": "System error"}
            }
        }

    def _get_disk_usage_stats(self) -> Dict[str, Any]:
        """Get disk usage statistics for all directories."""
        try:
            config = self.rag_manager.config
            
            # Get directory sizes - avoid double counting based on user's concern
            logs_size = get_directory_size(config.LOG_DIR)
            staging_size = get_directory_size(config.UPLOAD_FOLDER)
            uploaded_size = get_directory_size(config.UPLOADED_FOLDER)
            deleted_size = get_directory_size(config.DELETED_FOLDER)
            
            # Get snapshots size
            snapshots_size = get_directory_size(config.SNAPSHOT_DIR)
            
            # Get overall disk capacity (use current directory as reference)
            disk_capacity = get_disk_usage(".")
            
            return {
                'logs_size': logs_size,
                'staging_size': staging_size,
                'uploaded_size': uploaded_size,
                'deleted_size': deleted_size,
                'snapshots_size': snapshots_size,
                'disk_capacity': disk_capacity
            }
            
        except Exception as e:
            logger.debug(f"Error getting disk usage stats: {e}")
            return {
                'logs_size': {"bytes": 0, "human": "Error", "exists": False},
                'staging_size': {"bytes": 0, "human": "Error", "exists": False},
                'uploaded_size': {"bytes": 0, "human": "Error", "exists": False},
                'deleted_size': {"bytes": 0, "human": "Error", "exists": False},
                'snapshots_size': {"bytes": 0, "human": "Error", "exists": False},
                'disk_capacity': {
                    "total_human": "Unknown",
                    "free_human": "Unknown", 
                    "free_percent": 0,
                    "warning_level": "unknown"
                }
            }
    
    def _get_database_size_stats(self, disk_usage_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get database size statistics."""
        try:
            # PostgreSQL database size
            postgres_size = get_postgres_database_size(
                getattr(self.rag_manager, 'postgres_manager', None)
            )
            
            # Estimate Milvus size based on entity count and dimensions
            milvus_size = self._estimate_milvus_size()
            
            # Calculate total usage from all storage components
            total_usage = self._calculate_total_usage(disk_usage_stats, postgres_size, milvus_size)
            
            return {
                'postgres_total': postgres_size,
                'milvus_estimate': milvus_size,
                'total_usage': total_usage
            }
            
        except Exception as e:
            logger.debug(f"Error getting database size stats: {e}")
            return {
                'postgres_total': {"bytes": 0, "human": "Error", "error": str(e)},
                'milvus_estimate': {"bytes": 0, "human": "Error", "error": str(e)},
                'total_usage': {"bytes": 0, "human": "Error", "error": str(e)}
            }
    
    def _calculate_total_usage(self, disk_usage_stats: Dict[str, Any], postgres_size: Dict[str, Any], milvus_size: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total usage from all storage components."""
        try:
            from rag_manager.utils.disk_usage import format_bytes
            
            total_bytes = 0
            components = []
            
            # Add disk usage components
            for component_name, component_data in disk_usage_stats.items():
                if component_name == 'disk_capacity':
                    continue  # Skip capacity info, we only want usage
                    
                if isinstance(component_data, dict) and 'bytes' in component_data:
                    bytes_value = component_data.get('bytes', 0)
                    if bytes_value > 0:
                        total_bytes += bytes_value
                        components.append({
                            'name': component_name,
                            'bytes': bytes_value,
                            'human': component_data.get('human', '0 B')
                        })
            
            # Add database components
            if isinstance(postgres_size, dict) and 'bytes' in postgres_size:
                postgres_bytes = postgres_size.get('bytes', 0)
                if postgres_bytes > 0:
                    total_bytes += postgres_bytes
                    components.append({
                        'name': 'postgres_db',
                        'bytes': postgres_bytes,
                        'human': postgres_size.get('human', '0 B')
                    })
            
            if isinstance(milvus_size, dict) and 'bytes' in milvus_size:
                milvus_bytes = milvus_size.get('bytes', 0)
                if milvus_bytes > 0:
                    total_bytes += milvus_bytes
                    components.append({
                        'name': 'milvus_db',
                        'bytes': milvus_bytes,
                        'human': milvus_size.get('human', '0 B')
                    })
            
            return {
                'bytes': total_bytes,
                'human': format_bytes(total_bytes),
                'components': components,
                'error': None
            }
            
        except Exception as e:
            logger.debug(f"Error calculating total usage: {e}")
            return {"bytes": 0, "human": "Error", "error": str(e)}
    
    def _estimate_milvus_size(self) -> Dict[str, Any]:
        """Estimate Milvus collection size based on entity count and dimensions."""
        try:
            if not self.rag_manager.milvus_manager:
                return {"bytes": 0, "human": "N/A", "error": "No Milvus connection"}
            
            # Get collection stats
            collections_info = []
            total_estimated_bytes = 0
            
            # Check document collection
            doc_collection_name = self.rag_manager.config.DOCUMENT_COLLECTION
            try:
                doc_stats = self.rag_manager.milvus_manager.get_collection_stats()
                if doc_stats and 'num_entities' in doc_stats:
                    # Estimate: entities * dimensions * 4 bytes (float32) + overhead
                    vector_dim = doc_stats.get('dim', self.rag_manager.config.VECTOR_DIM)
                    entity_count = doc_stats['num_entities']
                    estimated_size = entity_count * vector_dim * 4 * 1.2  # 20% overhead
                    total_estimated_bytes += estimated_size
                    
                    collections_info.append({
                        "name": doc_collection_name,
                        "entities": entity_count,
                        "estimated_bytes": int(estimated_size)
                    })
            except Exception:
                pass
            
            # Check email collection if it exists
            # Note: Email collection requires direct pymilvus access
            try:
                from pymilvus import utility, Collection
                email_collection_name = self.rag_manager.config.EMAIL_COLLECTION
                if utility.has_collection(email_collection_name):
                    email_col = Collection(email_collection_name)
                    email_col.load()
                    entity_count = email_col.num_entities
                    if entity_count > 0:
                        # Use default vector dimension from config
                        vector_dim = self.rag_manager.config.VECTOR_DIM
                        estimated_size = entity_count * vector_dim * 4 * 1.2  # 20% overhead
                        total_estimated_bytes += estimated_size
                        
                        collections_info.append({
                            "name": email_collection_name,
                            "entities": entity_count,
                            "estimated_bytes": int(estimated_size)
                        })
            except Exception:
                pass
            
            from rag_manager.utils.disk_usage import format_bytes
            
            return {
                "bytes": int(total_estimated_bytes),
                "human": format_bytes(int(total_estimated_bytes)),
                "collections": collections_info,
                "error": None
            }
            
        except Exception as e:
            logger.debug(f"Error estimating Milvus size: {e}")
            return {"bytes": 0, "human": "Error", "error": str(e)}
    
    def _get_email_backlog_stats(self) -> Dict[str, Any]:
        """Get email account backlog statistics with severity buckets."""
        try:
            if not hasattr(self.rag_manager, 'email_account_manager') or not self.rag_manager.email_account_manager:
                return {
                    'total_backlogged': 0,
                    'severe': 0,
                    'moderate': 0,
                    'light': 0,
                    'status': 'no_manager'
                }
            
            accounts_due = self.rag_manager.email_account_manager._get_accounts_due_for_sync()
            
            if not accounts_due:
                return {
                    'total_backlogged': 0,
                    'severe': 0,
                    'moderate': 0,
                    'light': 0,
                    'status': 'all_current'
                }
            
            # Calculate time thresholds
            now = datetime.utcnow()
            severe_threshold = now - timedelta(days=7)  # More than 7 days
            moderate_threshold = now - timedelta(days=3)  # 3-7 days
            # light threshold is anything less than 3 days
            
            severe_count = 0
            moderate_count = 0
            light_count = 0
            
            for account in accounts_due:
                last_sync = account.get('last_sync_time')
                if last_sync:
                    if isinstance(last_sync, str):
                        last_sync = datetime.fromisoformat(last_sync.replace('Z', '+00:00'))
                    
                    if last_sync < severe_threshold:
                        severe_count += 1
                    elif last_sync < moderate_threshold:
                        moderate_count += 1
                    else:
                        light_count += 1
                else:
                    # No last sync time means never synced - severe
                    severe_count += 1
            
            total_backlogged = len(accounts_due)
            
            return {
                'total_backlogged': total_backlogged,
                'severe': severe_count,
                'moderate': moderate_count,
                'light': light_count,
                'status': 'has_backlog' if total_backlogged > 0 else 'all_current'
            }
            
        except Exception as e:
            logger.debug(f"Error getting email backlog stats: {e}")
            return {
                'total_backlogged': 0,
                'severe': 0,
                'moderate': 0,
                'light': 0,
                'status': 'error'
            }
    
    def _get_email_ingest_age_stats(self) -> Dict[str, Any]:
        """Get email ingestion age statistics (oldest freshness delta)."""
        try:
            if not hasattr(self.rag_manager, 'email_account_manager') or not self.rag_manager.email_account_manager:
                return {
                    'oldest_age_days': 0,
                    'oldest_account': None,
                    'status': 'no_manager',
                    'age_category': 'unknown'
                }
            
            accounts = self.rag_manager.email_account_manager.get_all_accounts()
            
            if not accounts:
                return {
                    'oldest_age_days': 0,
                    'oldest_account': None,
                    'status': 'no_accounts',
                    'age_category': 'none'
                }
            
            now = datetime.utcnow()
            oldest_age_days = 0
            oldest_account = None
            
            for account in accounts:
                last_sync = account.get('last_sync_time')
                if last_sync:
                    if isinstance(last_sync, str):
                        last_sync = datetime.fromisoformat(last_sync.replace('Z', '+00:00'))
                    
                    age_delta = now - last_sync
                    age_days = age_delta.total_seconds() / (24 * 3600)
                    
                    if age_days > oldest_age_days:
                        oldest_age_days = age_days
                        oldest_account = account.get('email', 'Unknown')
                else:
                    # No sync time means never synced - set to very high value
                    oldest_age_days = 9999
                    oldest_account = account.get('email', 'Unknown')
                    break  # Never synced is the worst case
            
            # Determine age category for styling
            if oldest_age_days == 0:
                age_category = 'current'
            elif oldest_age_days <= 1:
                age_category = 'fresh'
            elif oldest_age_days <= 3:
                age_category = 'moderate'
            elif oldest_age_days <= 7:
                age_category = 'stale'
            else:
                age_category = 'critical'
            
            return {
                'oldest_age_days': round(oldest_age_days, 1),
                'oldest_account': oldest_account,
                'status': 'has_data',
                'age_category': age_category
            }
            
        except Exception as e:
            logger.debug(f"Error getting email ingest age stats: {e}")
            return {
                'oldest_age_days': 0,
                'oldest_account': None,
                'status': 'error',
                'age_category': 'error'
            }
    
    def _get_threshold_status(self, value: float, warning_threshold: float, critical_threshold: float, 
                            is_percentage: bool = False, reverse_logic: bool = False) -> str:
        """
        Get threshold status for a metric value.
        
        Args:
            value: The metric value to check
            warning_threshold: Warning threshold 
            critical_threshold: Critical threshold
            is_percentage: Whether value is a percentage (0-100)
            reverse_logic: True for metrics where lower values are worse (e.g., disk free)
        
        Returns:
            Status string: 'success', 'warning', 'critical'
        """
        try:
            if reverse_logic:
                # For metrics like disk free % where lower is worse
                if value <= critical_threshold:
                    return 'critical'
                elif value <= warning_threshold:
                    return 'warning'
                else:
                    return 'success'
            else:
                # For metrics like pending files where higher is worse
                if value >= critical_threshold:
                    return 'critical'
                elif value >= warning_threshold:
                    return 'warning'
                else:
                    return 'success'
        except (TypeError, ValueError):
            return 'success'
    
    def _get_capacity_thresholds(self) -> Dict[str, Any]:
        """Get threshold status for all capacity metrics."""
        try:
            disk_usage_stats = self._get_disk_usage_stats()
            database_size_stats = self._get_database_size_stats(disk_usage_stats)
            
            thresholds = {}
            
            # Disk free percentage thresholds (Warning: <25%, Critical: <15%)
            disk_capacity = disk_usage_stats.get('disk_capacity', {})
            if disk_capacity.get('total_bytes', 0) > 0:
                free_percentage = (disk_capacity.get('free_bytes', 0) / disk_capacity.get('total_bytes', 1)) * 100
                thresholds['disk_free_status'] = self._get_threshold_status(
                    free_percentage, 25.0, 15.0, is_percentage=True, reverse_logic=True
                )
                thresholds['disk_free_percentage'] = round(free_percentage, 1)
            else:
                thresholds['disk_free_status'] = 'success'
                thresholds['disk_free_percentage'] = 0
            
            # Processing queue thresholds
            total_processing = (
                len(self.rag_manager.processing_status) +
                len(self.rag_manager.url_processing_status) +
                len(self.rag_manager.email_processing_status)
            )
            
            # Total processing files (Warning: >5, Critical: >15)
            thresholds['total_processing_status'] = self._get_threshold_status(
                total_processing, 5, 15
            )
            
            # Individual processing queues
            thresholds['processing_files_status'] = self._get_threshold_status(
                len(self.rag_manager.processing_status), 3, 10
            )
            thresholds['processing_urls_status'] = self._get_threshold_status(
                len(self.rag_manager.url_processing_status), 5, 20
            )
            thresholds['processing_emails_status'] = self._get_threshold_status(
                len(self.rag_manager.email_processing_status), 3, 10
            )
            
            return thresholds
            
        except Exception as e:
            logger.debug(f"Error getting capacity thresholds: {e}")
            return {
                'disk_free_status': 'success',
                'disk_free_percentage': 0,
                'total_processing_status': 'success',
                'processing_files_status': 'success',
                'processing_urls_status': 'success',
                'processing_emails_status': 'success'
            }
