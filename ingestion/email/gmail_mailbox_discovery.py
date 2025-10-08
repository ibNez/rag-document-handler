#!/usr/bin/env python3
"""
Gmail Mailbox Discovery Utility

This utility helps discover available Gmail labels/mailboxes for account setup.
Used during the account creation process to let users select which mailboxes to sync.
"""

import logging
from typing import List, Dict, Any, Optional

from google.oauth2.credentials import Credentials
from ingestion.email.connectors.gmail_connector import GmailConnector

logger = logging.getLogger(__name__)


def get_gmail_mailboxes(access_token: str, refresh_token: str, client_id: str, client_secret: str, email_address: str) -> List[Dict[str, Any]]:
    """
    Get available Gmail mailboxes/labels for account setup.
    
    Args:
        access_token: OAuth access token
        refresh_token: OAuth refresh token  
        client_id: Gmail OAuth client ID
        client_secret: Gmail OAuth client secret
        email_address: Gmail email address
        
    Returns:
        List of mailbox dictionaries with id, name, type, and message count
    """
    try:
        # Create credentials object
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Create Gmail connector
    connector = GmailConnector(
            credentials=creds,
            user_id=email_address or "me"
        )
        
        # Test connection first
        if not connector.test_connection():
            logger.error("Failed to connect to Gmail API")
            return []
        
        # Get available mailboxes
        mailboxes = connector.get_available_mailboxes()
        
        # Filter to show most useful mailboxes first
        filtered_mailboxes = []
        
        # Priority system labels
        priority_system = ['INBOX', 'SENT', 'DRAFT', 'SPAM', 'TRASH', 'IMPORTANT']
        
        # Add priority system labels first
        for mailbox in mailboxes:
            if mailbox['type'] == 'system' and mailbox['id'] in priority_system:
                filtered_mailboxes.append(mailbox)
        
        # Add other system labels
        for mailbox in mailboxes:
            if mailbox['type'] == 'system' and mailbox['id'] not in priority_system:
                filtered_mailboxes.append(mailbox)
        
        # Add user labels
        for mailbox in mailboxes:
            if mailbox['type'] == 'user':
                filtered_mailboxes.append(mailbox)
        
        logger.info(f"📬 Retrieved {len(filtered_mailboxes)} Gmail mailboxes for {email_address}")
        return filtered_mailboxes
        
    except Exception as e:
        logger.error(f"❌ Failed to get Gmail mailboxes: {e}")
        return []


def format_mailboxes_for_storage(selected_mailbox_ids: List[str]) -> str:
    """
    Format selected mailbox IDs for storage in database.
    
    Args:
        selected_mailbox_ids: List of Gmail label IDs selected by user
        
    Returns:
        Comma-separated string of mailbox IDs for database storage
    """
    if not selected_mailbox_ids:
        return "INBOX"  # Default to INBOX if nothing selected
        
    # Remove duplicates and filter out empty strings
    unique_ids = list(set(id.strip() for id in selected_mailbox_ids if id.strip()))
    
    # Ensure INBOX is included if not already selected
    if 'INBOX' not in unique_ids:
        unique_ids.insert(0, 'INBOX')
    
    return ','.join(unique_ids)


def parse_mailboxes_from_storage(mailbox_string: str) -> List[str]:
    """
    Parse mailbox string from database back into list.
    
    Args:
        mailbox_string: Comma-separated string from database
        
    Returns:
        List of mailbox IDs
    """
    if not mailbox_string:
        return ['INBOX']
        
    return [box.strip() for box in mailbox_string.split(',') if box.strip()]


if __name__ == "__main__":
    # Example usage
    print("Gmail Mailbox Discovery Utility")
    print("This module provides functions for discovering Gmail labels/mailboxes during account setup")
