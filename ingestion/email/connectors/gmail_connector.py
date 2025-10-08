"""
Simple Gmail Connector - Clean Implementation

Based on official Google Gmail API documentation and quickstart examples.
No over-engineering, no complex history tracking, just basic email fetching.
"""

import base64
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .base import EmailConnector

logger = logging.getLogger(__name__)


class GmailConnector(EmailConnector):
    """Simple Gmail connector following Google's official examples."""
    
    def __init__(self, account: Dict[str, Any]):
        """Initialize Gmail connector with account dictionary containing OAuth credentials."""
        self.account = account
        self.user_id = account.get("email_address") or "me"
        
        # Extract OAuth credentials from account
        oauth_token = account.get('oauth_access_token')  # Database field name
        oauth_refresh_token = account.get('oauth_refresh_token')
        oauth_expires_at = account.get('oauth_token_expires_at')
        
        if not oauth_token:
            raise ValueError("Gmail account missing oauth_access_token")
            
        # Handle encrypted tokens
        from ingestion.utils.crypto import decrypt
        
        try:
            access_token = decrypt(oauth_token) if isinstance(oauth_token, str) else oauth_token
            refresh_token = decrypt(oauth_refresh_token) if oauth_refresh_token and isinstance(oauth_refresh_token, str) else oauth_refresh_token
        except Exception as e:
            logger.error(f"Failed to decrypt OAuth tokens: {e}")
            raise ValueError(f"Failed to decrypt OAuth tokens: {e}")
        
        # Parse token expiry
        token_expiry = None
        if oauth_expires_at:
            try:
                from datetime import datetime
                if isinstance(oauth_expires_at, str):
                    token_expiry = datetime.fromisoformat(oauth_expires_at.replace('Z', '+00:00'))
                else:
                    token_expiry = oauth_expires_at
            except Exception as e:
                logger.warning(f"Failed to parse token expiry: {e}")
        
        # Create credentials object
        self.credentials = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv('GMAIL_CLIENT_ID'),
            client_secret=os.getenv('GMAIL_CLIENT_SECRET'),
            expiry=token_expiry
        )
        
        self.service = build('gmail', 'v1', credentials=self.credentials)
        
    def test_connection(self) -> bool:
        """Test Gmail API connection."""
        try:
            # Simple test - get user profile
            profile = self.service.users().getProfile(userId=self.user_id).execute()
            logger.info(f"✅ Gmail connection successful: {profile.get('emailAddress')}")
            return True
        except HttpError as e:
            logger.error(f"❌ Gmail connection failed: {e}")
            return False
    
    def fetch_emails(
        self,
        since_date: Optional[datetime] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch emails from Gmail using Batch API for efficiency with proper pagination.
        
        Args:
            since_date: Fetch emails since this date (optional)
            max_results: Maximum number of emails to fetch (None = fetch all)
            
        Returns:
            List of email records
        """
        try:
            logger.info(f"Fetching Gmail emails (max: {'ALL' if max_results is None else max_results})")
            
            # Build query if date filter needed
            query = ""
            if since_date:
                date_str = since_date.strftime('%Y/%m/%d')
                query = f"after:{date_str}"
                logger.info(f"Using date filter: {query}")
            
            # Collect all message IDs through pagination
            all_messages = []
            page_token = None
            page_count = 0
            total_estimate = None
            
            while True:
                page_count += 1
                logger.debug(f"📄 Fetching page {page_count} of message IDs...")
                
                # Get list of message IDs for this page
                list_params = {
                    'userId': self.user_id,
                    'q': query,
                    'maxResults': 500,  # Max allowed by Gmail API per page
                }
                if page_token:
                    list_params['pageToken'] = page_token
                
                messages_result = self.service.users().messages().list(**list_params).execute()
                if total_estimate is None:
                    total_estimate = messages_result.get('resultSizeEstimate')
                
                page_messages = messages_result.get('messages', [])
                if not page_messages:
                    logger.debug(f"📄 Page {page_count}: No messages found")
                    break
                
                all_messages.extend(page_messages)
                logger.debug(f"📄 Page {page_count}: Found {len(page_messages)} messages (total: {len(all_messages)})")
                
                # Check if we've reached the max_results limit
                if max_results and len(all_messages) >= max_results:
                    all_messages = all_messages[:max_results]
                    logger.debug(f"📄 Reached max_results limit of {max_results}, stopping pagination")
                    break
                
                # Check for next page
                page_token = messages_result.get('nextPageToken')
                if not page_token:
                    logger.debug(f"📄 No more pages available after page {page_count}")
                    break
            
            if not all_messages:
                logger.info("No messages found in any pages")
                return []
                
            logger.info(f"✅ Found {len(all_messages)} total messages across {page_count} pages (estimate={total_estimate})")
            
            # Fetch messages individually (more reliable than batch API)
            emails = self._fetch_messages_individual(all_messages)
            
            logger.info(f"✅ Successfully fetched {len(emails)} emails")
            return emails
            
        except HttpError as e:
            logger.error(f"❌ Gmail fetch failed: {e}")
            raise

    def list_message_ids(self, since_date: Optional[datetime] = None) -> List[str]:
        """List message IDs with pagination (newest first)."""
        try:
            query = ""
            if since_date:
                date_str = since_date.strftime('%Y/%m/%d')
                query = f"after:{date_str}"

            page_token = None
            ids: List[str] = []
            page_count = 0
            while True:
                page_count += 1
                list_params = {
                    'userId': self.user_id,
                    'q': query,
                    'maxResults': 500,
                }
                if page_token:
                    list_params['pageToken'] = page_token
                resp = self.service.users().messages().list(**list_params).execute()
                msgs = resp.get('messages', [])
                if not msgs:
                    break
                ids.extend(m['id'] for m in msgs)
                page_token = resp.get('nextPageToken')
                if not page_token:
                    break
            logger.info(f"Listed {len(ids)} message IDs across {page_count} pages")
            return ids
        except HttpError as e:
            logger.error(f"Failed to list message IDs: {e}")
            raise

    def get_email_by_id(self, msg_id: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse a single email by Gmail message ID."""
        try:
            full_msg = self.service.users().messages().get(
                userId=self.user_id, id=msg_id, format='full'
            ).execute()
            return self._parse_message(full_msg)
        except HttpError as e:
            logger.warning(f"Failed to fetch message {msg_id}: {e}")
            return None
    
    # Note: Batch fetching removed for simplicity and stability.
    
    def _fetch_messages_individual(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch messages individually (reliable alternative to batch API).
        
        Args:
            messages: List of message metadata
            
        Returns:
            List of parsed email records
        """
        emails = []
        total_messages = len(messages)
        logger.info(f"📬 Fetching {total_messages} messages individually...")
        
        for i, msg in enumerate(messages, 1):
            try:
                # Get full message
                full_msg = self.service.users().messages().get(
                    userId=self.user_id,
                    id=msg['id'],
                    format='full'
                ).execute()
                
                # Parse email
                email_record = self._parse_message(full_msg)
                if email_record:
                    emails.append(email_record)
                    
                # Progress logging every 25 messages
                if i % 100 == 0 or i == total_messages:
                    percentage = (i / total_messages) * 100
                    logger.debug(f"📬 Progress: {i}/{total_messages} messages ({percentage:.1f}%) - {len(emails)} emails parsed")
                    
            except HttpError as e:
                logger.warning(f"Failed to fetch message {msg['id']}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing message {msg['id']}: {e}")
                continue
        
        logger.info(f"✅ Successfully fetched {len(emails)} emails from {total_messages} messages")
        return emails
    
    def _parse_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Gmail message into normalized email record for storage."""
        try:
            # Basic message info
            msg_id = message['id']
            thread_id = message.get('threadId', '')
            snippet = message.get('snippet', '')
            
            # Convert internal date to datetime
            internal_date = int(message.get('internalDate', 0))
            received_date = datetime.fromtimestamp(internal_date / 1000, timezone.utc)
            
            # Parse headers
            payload = message.get('payload', {})
            headers = payload.get('headers', [])
            
            # Extract key headers
            subject = ""
            from_addr = ""
            to_addrs: List[str] = []
            
            for header in headers:
                name = header.get('name', '').lower()
                value = header.get('value', '')
                
                if name == 'subject':
                    subject = value
                elif name == 'from':
                    from_addr = value
                elif name == 'to':
                    # Split recipients into list conservatively; downstream can improve parsing
                    to_addrs = [addr.strip() for addr in value.split(',') if addr.strip()]
            
            # Extract body content
            body_text = self._extract_body(payload)
            
            # Create email record
            email_record = {
                'message_id': msg_id,
                'thread_id': thread_id,
                'subject': subject or '',
                'from_addr': from_addr or '',
                'to_addrs': to_addrs or None,
                'date_utc': received_date,
                'snippet': snippet,
                'content': body_text or '',
                'headers': headers,
                'source': 'gmail'
            }
            
            logger.debug(f"Parsed email: {subject[:50]}...")
            return email_record
            
        except Exception as e:
            logger.warning(f"Failed to parse message: {e}")
            return None
    
    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract text body from message payload."""
        try:
            # Check if this part has body data
            body = payload.get('body', {})
            if body.get('data'):
                # Decode base64 body
                body_data = base64.urlsafe_b64decode(body['data']).decode('utf-8')
                return body_data
            
            # Check multipart message
            parts = payload.get('parts', [])
            for part in parts:
                mime_type = part.get('mimeType', '')
                
                # Look for text/plain or text/html
                if mime_type in ['text/plain', 'text/html']:
                    part_body = part.get('body', {})
                    if part_body.get('data'):
                        body_data = base64.urlsafe_b64decode(part_body['data']).decode('utf-8')
                        return body_data
                
                # Recursively check nested parts
                nested_body = self._extract_body(part)
                if nested_body:
                    return nested_body
            
            return ""
            
        except Exception as e:
            logger.warning(f"Failed to extract body: {e}")
            return ""
    
    # Removed legacy convenience method get_recent_emails; use orchestrator-driven sync instead.
    
    def get_available_mailboxes(self) -> List[Dict[str, Any]]:
        """Get available Gmail labels that can be used for email sync."""
        try:
            logger.info("Fetching Gmail labels...")
            
            # Get Gmail labels from the API
            results = self.service.users().labels().list(userId=self.user_id).execute()
            labels = results.get('labels', [])
            
            # Convert Gmail labels to mailbox format
            mailboxes = []
            for label in labels:
                # Gmail labels have different types
                label_type = 'system' if label['type'] == 'system' else 'user'
                
                mailbox = {
                    'id': label['id'],
                    'name': label['name'],
                    'type': label_type,
                    'selectable': True,
                    'messageCount': label.get('messagesTotal', 0)
                }
                mailboxes.append(mailbox)
            
            logger.info(f"✅ Retrieved {len(mailboxes)} Gmail labels")
            return mailboxes
            
        except HttpError as e:
            logger.error(f"❌ Failed to fetch Gmail labels: {e}")
            # Return basic default labels if API call fails
            return [
                {
                    'id': 'INBOX',
                    'name': 'INBOX',
                    'type': 'system',
                    'selectable': True,
                    'messageCount': 0
                }
            ]
 
