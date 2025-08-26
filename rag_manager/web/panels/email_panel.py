"""
Email Panel Statistics Provider
Following DEVELOPMENT_RULES.md for all development requirements

This module handles all statistics for the Email panel on the status dashboard.
Centralizes email account statistics, processing counts, and email collection metrics.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EmailPanelStats:
    """Statistics provider for the Email panel."""
    
    def __init__(self, rag_manager):
        """Initialize with reference to the main RAG manager."""
        self.rag_manager = rag_manager
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get all email statistics for the email panel.
        
        Returns:
            Dictionary containing email panel statistics
        """
        try:
            if not self.rag_manager.email_account_manager:
                return self._empty_stats()
                
            # Get aggregated email statistics from PostgreSQL
            email_stats = self.rag_manager.email_account_manager.get_account_stats()
            
            # Calculate derived stats
            never_synced = email_stats.get('total_accounts', 0) - email_stats.get('synced_accounts', 0)
            
            # Get email collection stats from Milvus
            email_collection_stats = self._get_email_collection_stats()
            
            # Get additional email metrics
            avg_emails_per_account = self._calculate_avg_emails_per_account(email_stats)
            most_active_account = self._get_most_active_account()
            emails_with_attachments = self._get_emails_with_attachments_count()
            due_now = self._get_accounts_due_for_sync()
            
            return {
                # Account metrics
                'total_accounts': email_stats.get('total_accounts', 0),
                'active_accounts': email_stats.get('active_accounts', 0),
                'never_synced': never_synced,
                'due_now': due_now,
                
                # Email metrics
                'total_emails': email_stats.get('total_messages', 0),
                'processed_emails': email_stats.get('processed_messages', 0),
                'emails_with_attachments': emails_with_attachments,
                
                # Vector store metrics
                'embedding_dim': email_collection_stats.get('dim', ''),
                'indexed': 'Yes' if email_collection_stats.get('indexed') else 'No',
                'metric_type': email_collection_stats.get('metric_type', 'cosine'),
                
                # Derived metrics
                'avg_emails_per_account': avg_emails_per_account,
                'most_active_account': most_active_account
            }
            
        except Exception as e:
            logger.error(f"Failed to get email panel stats: {e}")
            return self._empty_stats()
    
    def _get_email_collection_stats(self) -> Dict[str, Any]:
        """Get email collection statistics from Milvus."""
        try:
            if self.rag_manager.milvus_manager:
                return self.rag_manager.milvus_manager.get_email_collection_stats() or {}
        except Exception as e:
            logger.warning(f"Failed to get email collection stats: {e}")
        return {}
    
    def _calculate_avg_emails_per_account(self, email_stats: Dict[str, Any]) -> float:
        """Calculate average emails per account."""
        total_accounts = email_stats.get('total_accounts', 0)
        total_emails = email_stats.get('total_messages', 0)
        
        if total_accounts > 0:
            return round(total_emails / total_accounts, 1)
        return 0.0
    
    def _get_most_active_account(self) -> str:
        """Get the most active email account."""
        try:
            if not self.rag_manager.email_account_manager:
                return ""
                
            # Get all accounts with email counts
            accounts = self.rag_manager.email_account_manager.list_accounts(include_password=False)
            
            if not accounts:
                return ""
                
            # Find account with most synced emails
            most_active = max(accounts, key=lambda acc: acc.get('synced_emails', 0))
            
            if most_active.get('synced_emails', 0) > 0:
                return most_active.get('account_name', most_active.get('email_address', 'Unknown'))
                
        except Exception as e:
            logger.debug(f"Error finding most active account: {e}")
        
        return ""
    
    def _get_emails_with_attachments_count(self) -> int:
        """Get count of emails with attachments."""
        try:
            if not self.rag_manager.postgres_manager:
                return 0
                
            with self.rag_manager.postgres_manager.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT COUNT(DISTINCT message_id) 
                        FROM emails 
                        WHERE attachments IS NOT NULL 
                        AND attachments != 'null' 
                        AND attachments != '[]'
                    """)
                    result = cur.fetchone()
                    return result[0] if result else 0
                    
        except Exception as e:
            logger.debug(f"Error getting emails with attachments count: {e}")
            return 0
    
    def _get_accounts_due_for_sync(self) -> int:
        """Get count of accounts due for synchronization."""
        try:
            if not self.rag_manager.email_account_manager:
                return 0
                
            accounts = self.rag_manager.email_account_manager.list_accounts(include_password=False)
            due_count = 0
            
            from datetime import datetime
            now = datetime.now()
            
            for account in accounts:
                if not account.get('refresh_interval_minutes') or account.get('refresh_interval_minutes') <= 0:
                    continue
                    
                last_synced = account.get('last_synced')
                if not last_synced:
                    # Never synced but has refresh interval = due now
                    due_count += 1
                    continue
                    
                try:
                    # Parse last synced time
                    if isinstance(last_synced, str):
                        last_synced_dt = datetime.fromisoformat(last_synced.replace('Z', '+00:00'))
                    else:
                        last_synced_dt = last_synced
                        
                    # Calculate if due
                    from datetime import timedelta
                    interval = timedelta(minutes=account['refresh_interval_minutes'])
                    next_sync = last_synced_dt + interval
                    
                    if now >= next_sync:
                        due_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error checking if account {account.get('id')} is due: {e}")
                    
            return due_count
            
        except Exception as e:
            logger.debug(f"Error getting accounts due for sync: {e}")
            return 0
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty email panel stats."""
        return {
            'total_accounts': 0,
            'active_accounts': 0,
            'never_synced': 0,
            'due_now': 0,
            'total_emails': 0,
            'processed_emails': 0,
            'emails_with_attachments': 0,
            'embedding_dim': '',
            'indexed': 'No',
            'metric_type': 'cosine',
            'avg_emails_per_account': 0.0,
            'most_active_account': ''
        }
