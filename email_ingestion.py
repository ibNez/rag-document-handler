#!/usr/bin/env python3
"""
EmailIngestion Compatibility Module
Following DEVELOPMENT_RULES.md for all development requirements

This module provides compatibility between the scheduler's expected EmailIngestion class
and the actual ingestion.email package structure.
"""

import logging
import sqlite3
import os
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from ingestion.email import EmailOrchestrator, EmailAccountManager
from ingestion.email.processor import EmailProcessor

logger = logging.getLogger(__name__)


class EmailConfig:
    """Simple config object for email system"""
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        # Email is always enabled if accounts exist


class EmailIngestion:
    """Compatibility wrapper for the ingestion.email system"""
    
    def __init__(self):
        """Initialize the email ingestion system"""
        
        # Initialize database connection
        db_path = os.path.join('databases', 'Knowledgebase.db')
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Initialize components
        self.config = EmailConfig()
        self.account_manager = EmailAccountManager(self.conn)
        
        # Note: EmailProcessor requires milvus and sqlite connections
        # We'll initialize it when needed during actual sync
        self.processor = None
        
        self.orchestrator = EmailOrchestrator(
            config=self.config,
            account_manager=self.account_manager,
            processor=self.processor
        )
        
        logger.info("EmailIngestion compatibility wrapper initialized")
    
    def sync_account(self, account_id: int) -> Dict[str, Any]:
        """
        Sync a specific email account
        
        Args:
            account_id: The ID of the email account to sync
            
        Returns:
            Dict containing sync results
        """
        try:
            logger.info(f"Starting sync for email account ID: {account_id}")
            
            # Get account details - need to get all accounts and find the one we want
            accounts = self.account_manager.list_accounts(include_password=True)
            account = None
            for acc in accounts:
                if acc['id'] == account_id:
                    account = acc
                    break
            
            if not account:
                raise ValueError(f"Email account {account_id} not found")
            
            logger.info(f"Syncing account: {account['account_name']}")
            
            # Perform the sync using orchestrator - pass the account dict
            self.orchestrator.sync_account(account)
            
            # Update last_synced timestamp
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE email_accounts 
                SET last_synced = datetime('now') 
                WHERE id = ?
            """, (account_id,))
            self.conn.commit()
            
            logger.info(f"Email sync completed for account {account_id}")
            
            return {
                'status': 'success',
                'account_id': account_id,
                'account_name': account['account_name'],
                'synced_at': datetime.now(UTC).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Email sync failed for account {account_id}: {e}")
            
            return {
                'status': 'error',
                'account_id': account_id,
                'error': str(e),
                'failed_at': datetime.now(UTC).isoformat()
            }
    
    def sync_all_accounts(self) -> Dict[str, Any]:
        """
        Sync all configured email accounts
        
        Returns:
            Dict containing overall sync results
        """
        try:
            logger.info("Starting sync for all email accounts")
            
            # Get all accounts
            accounts = self.account_manager.list_accounts(include_password=False)
            
            results = []
            total_success = 0
            total_failed = 0
            
            for account in accounts:
                account_id = account['id']
                result = self.sync_account(account_id)
                results.append(result)
                
                if result['status'] == 'success':
                    total_success += 1
                else:
                    total_failed += 1
            
            overall_result = {
                'status': 'completed',
                'total_accounts': len(accounts),
                'successful': total_success,
                'failed': total_failed,
                'results': results,
                'synced_at': datetime.now(UTC).isoformat()
            }
            
            logger.info(f"All accounts sync completed: {total_success} success, {total_failed} failed")
            
            return overall_result
            
        except Exception as e:
            logger.error(f"Sync all accounts failed: {e}")
            
            return {
                'status': 'error',
                'error': str(e),
                'failed_at': datetime.now(UTC).isoformat()
            }
    
    def get_due_accounts(self) -> list:
        """
        Get list of email accounts that are due for sync
        
        Returns:
            List of account IDs due for sync
        """
        try:
            cursor = self.conn.cursor()
            
            # Query for accounts due for sync
            cursor.execute("""
                SELECT id FROM email_accounts 
                WHERE last_synced IS NULL 
                   OR datetime(last_synced, '+' || refresh_interval_minutes || ' minutes') <= datetime('now')
            """)
            
            due_accounts = [row[0] for row in cursor.fetchall()]
            
            logger.debug(f"Found {len(due_accounts)} accounts due for sync")
            
            return due_accounts
            
        except Exception as e:
            logger.error(f"Failed to get due accounts: {e}")
            return []
    
    def __del__(self):
        """Close database connection on cleanup"""
        if hasattr(self, 'conn'):
            self.conn.close()
