#!/usr/bin/env python3
"""
Diagnostic script to investigate the specific email causing message_id issues.

This script connects directly to the IMAP server and tests the problematic email
to determine if it's consistently corrupted or a transient parsing issue.
"""

import sys
import os
import ssl
import imaplib
import logging
from email import message_from_bytes
from email.header import decode_header, make_header
from pathlib import Path

# Add the project root to the path so we can import from ingestion
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion.email.connectors.imap_connector import IMAPConnector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def decode_header_value(raw_val):
    """Decode email header value."""
    if not raw_val:
        return None
    try:
        return str(make_header(decode_header(raw_val))).strip()
    except Exception:
        parts = decode_header(raw_val)
        decoded = []
        for text, enc in parts:
            if isinstance(text, bytes):
                try:
                    decoded.append(text.decode(enc or "utf-8", errors="ignore"))
                except Exception:
                    decoded.append(text.decode("utf-8", errors="ignore"))
            else:
                decoded.append(text)
        return "".join(decoded).strip()

def analyze_email(msg, sequence_num, imap_id):
    """Analyze a parsed email message and return diagnostic info."""
    info = {
        'sequence': sequence_num,
        'imap_id': imap_id,
        'message_id': None,
        'subject': None,
        'from_addr': None,
        'date': None,
        'has_message_id_header': False,
        'raw_headers': {},
        'is_corrupted': False,
        'corruption_reasons': []
    }
    
    # Check critical headers
    critical_headers = ['Message-ID', 'From', 'To', 'Subject', 'Date', 'Content-Type']
    for header_name in critical_headers:
        raw_value = msg.get(header_name)
        info['raw_headers'][header_name] = raw_value
    
    # Parse key fields
    info['subject'] = decode_header_value(msg.get("Subject"))
    info['message_id'] = (msg.get("Message-ID") or "").strip() or None
    info['from_addr'] = msg.get("From")
    info['date'] = msg.get("Date")
    
    # Check for corruption
    if msg.get("Message-ID"):
        info['has_message_id_header'] = True
    else:
        info['is_corrupted'] = True
        info['corruption_reasons'].append("Missing Message-ID header")
    
    if not info['from_addr']:
        info['is_corrupted'] = True
        info['corruption_reasons'].append("Missing From header")
        
    if not info['date']:
        info['is_corrupted'] = True
        info['corruption_reasons'].append("Missing Date header")
    
    return info

def test_specific_emails_around_issue(host, port, email_address, password, mailbox="INBOX", target_sequence=519):
    """Test emails around the known problematic sequence number."""
    logger.info("Connecting to IMAP server %s:%s", host, port)
    
    conn = imaplib.IMAP4_SSL(host, port)
    
    try:
        # Login
        status, _ = conn.login(email_address, password)
        if status != "OK":
            logger.error("Login failed")
            return
        
        # Select mailbox
        status, _ = conn.select(mailbox)
        if status != "OK":
            logger.error("Mailbox selection failed")
            return
        
        # Get all email IDs
        status, messages = conn.search(None, "ALL")
        if status != "OK":
            logger.error("Search failed")
            return
            
        email_ids = messages[0].split()
        total_emails = len(email_ids)
        
        logger.info("Found %d total emails in mailbox", total_emails)
        
        # Test emails around the target sequence
        test_range = range(max(0, target_sequence - 2), min(total_emails, target_sequence + 3))
        
        for sequence_num in test_range:
            eid = email_ids[sequence_num]
            decoded_eid = eid.decode() if isinstance(eid, bytes) else str(eid)
            
            logger.info("=" * 60)
            logger.info("Testing email sequence %d (IMAP ID %s)", sequence_num, decoded_eid)
            
            # Fetch the email multiple times to test consistency
            for attempt in range(1, 4):
                logger.info("  Attempt %d:", attempt)
                
                try:
                    status, msg_data = conn.fetch(decoded_eid, "(RFC822)")
                    if status != "OK" or not msg_data or not msg_data[0]:
                        logger.error("    Failed to fetch email")
                        continue
                    
                    # Parse the email
                    raw_data = msg_data[0][1]
                    if isinstance(raw_data, bytes):
                        msg = message_from_bytes(raw_data)
                    else:
                        logger.error("    Invalid message data type: %r", type(raw_data))
                        continue
                    
                    # Analyze the email
                    info = analyze_email(msg, sequence_num, decoded_eid)
                    
                    logger.info("    Sequence: %d, IMAP ID: %s", info['sequence'], info['imap_id'])
                    logger.info("    Subject: %r", info['subject'])
                    logger.info("    Message-ID: %r", info['message_id'])
                    logger.info("    From: %r", info['from_addr'])
                    logger.info("    Date: %r", info['date'])
                    logger.info("    Has Message-ID header: %s", info['has_message_id_header'])
                    logger.info("    Is corrupted: %s", info['is_corrupted'])
                    
                    if info['is_corrupted']:
                        logger.warning("    Corruption reasons: %s", ", ".join(info['corruption_reasons']))
                        logger.warning("    Raw headers:")
                        for header, value in info['raw_headers'].items():
                            logger.warning("      %s: %r", header, value)
                    
                except Exception as exc:
                    logger.error("    Exception during parsing: %s", exc)
            
            logger.info("")
            
    finally:
        try:
            conn.logout()
        except Exception:
            pass

def main():
    """Main diagnostic function."""
    # These should match your email account configuration
    # You can get these from your email account settings in the app
    HOST = "localhost"  # Replace with your IMAP server
    PORT = 993
    EMAIL_ADDRESS = "porcupine.pokey@local"  # Replace with your email
    PASSWORD = "password"  # Replace with your password
    MAILBOX = "INBOX"
    
    # Based on the logs, the problematic email was around sequence 519
    TARGET_SEQUENCE = 519
    
    logger.info("Starting email diagnostic test")
    logger.info("Target sequence: %d", TARGET_SEQUENCE)
    logger.info("This will test emails around sequence %d to identify corruption patterns", TARGET_SEQUENCE)
    
    test_specific_emails_around_issue(
        HOST, PORT, EMAIL_ADDRESS, PASSWORD, MAILBOX, TARGET_SEQUENCE
    )
    
    logger.info("Diagnostic test complete")

if __name__ == "__main__":
    main()
