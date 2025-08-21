#!/usr/bin/env python3
"""
EmailIngestion Compatibility Module
Following DEVELOPMENT_RULES.md for all development requirements

This module provides compatibility between the scheduler's expected EmailIngestion class
and the actual ingestion.email package structure.
"""

import logging
import os
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from ingestion.email import EmailOrchestrator
from ingestion.email.manager import PostgreSQLEmailManager
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig

logger = logging.getLogger(__name__)


class EmailConfig:
    """Simple config object for email system"""
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        
        # Email is always enabled if accounts exist


class EmailIngestion:
    """Compatibility wrapper for the ingestion.email system"""
    
    def __init__(self, postgres_manager):
        """Initialize the email ingestion system with dependency injection."""
        
        # Store postgres_manager as an instance attribute
        self.postgres_manager = postgres_manager
        
        # Initialize PostgreSQL email account manager
        self.email_account_manager = PostgreSQLEmailManager(postgres_manager)
        
        # Initialize PostgreSQL-based email message manager
        from ingestion.email.email_manager_postgresql import PostgreSQLEmailManager as EmailMessageManager
        email_message_manager = EmailMessageManager(postgres_manager)
        
        # Inject dependencies into EmailProcessor
        from ingestion.email.processor import EmailProcessor
        self.email_processor = EmailProcessor(
            milvus=None,  # Replace with actual vector store if needed
            email_manager=email_message_manager,
            embedding_model=None  # Replace with actual embedding model if needed
        )
        
        # Inject dependencies into EmailOrchestrator
        from ingestion.email.orchestrator import EmailOrchestrator
        self.email_orchestrator = EmailOrchestrator(
            config=None,  # Replace with actual config if needed
            account_manager=self.email_account_manager,
            processor=self.email_processor
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
            accounts = self.email_account_manager.list_accounts(include_password=True)
            account = None
            for acc in accounts:
                if acc['id'] == account_id:
                    account = acc
                    break
            
            if not account:
                raise ValueError(f"Email account {account_id} not found")
            
            logger.info(f"Syncing account: {account['account_name']}")
            
            # Perform the sync using orchestrator - pass the account dict
            self.email_orchestrator.sync_account(account)
            
            # Update last_synced timestamp using PostgreSQL
            self.email_account_manager.update_account(account_id, {
                'last_synced': datetime.now(UTC).isoformat()
            })
            
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
            accounts = self.email_account_manager.list_accounts(include_password=False)
            
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
            # Use orchestrator's method for getting due accounts
            due_accounts = self.email_orchestrator.get_due_accounts()
            due_account_ids = [account['id'] for account in due_accounts]
            
            logger.debug(f"Found {len(due_account_ids)} accounts due for sync")
            
            return due_account_ids
            
        except Exception as e:
            logger.error(f"Failed to get due accounts: {e}")
            return []
    
    def __del__(self):
        """Close database connection on cleanup"""
        if hasattr(self, 'postgres_manager'):
            self.postgres_manager.close()
