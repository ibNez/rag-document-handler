#!/usr/bin/env python3
"""
Diagnostic script to test the specific corrupted email outside of the main application.
This script will connect to the IMAP server and test the problematic email that was
causing "record missing message_id" errors.
"""

import os
import sys
sys.path.insert(0, '/Users/tonyphilip/Code/rag-document-handler')

import psycopg2
from ingestion.utils.crypto import decrypt
from ingestion.email.connectors.imap_connector import IMAPConnector

def get_email_account():
    """Get the email account from the database using the same process as the application."""
    print("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="rag_metadata",
        user="rag_user",
        password="secure_password"
    )
    
    try:
        cur = conn.cursor()
        cur.execute("SELECT server, port, email_address, password, mailbox, use_ssl FROM email_accounts WHERE id = 1")
        row = cur.fetchone()
        
        if not row:
            raise Exception("No email account found with id=1")
        
        server, port, email_address, encrypted_password, mailbox, use_ssl = row
        print(f"Found email account: {email_address}")
        
        # Decrypt password using the same process as the application
        password = decrypt(encrypted_password)
        
        return {
            'server': server,
            'port': port,
            'email_address': email_address,
            'password': password,
            'mailbox': mailbox or 'INBOX',
            'use_ssl': use_ssl
        }
    finally:
        conn.close()

def test_specific_emails():
    """Test emails around the problematic sequence number to find the corrupted one."""
    # Set the encryption key environment variable
    os.environ['EMAIL_ENCRYPTION_KEY'] = 'EsYO9e4CWbMGwUKFwSBlKa3UHTQSgXEU5Myzf1KpzkY='
    
    account = get_email_account()
    print(f"Testing IMAP connection to {account['server']} for {account['email_address']}")
    
    connector = IMAPConnector(
        host=account['server'],
        port=account['port'],
        email_address=account['email_address'],
        password=account['password'],
        mailbox=account['mailbox'],
        batch_limit=10,  # Small batch for testing
        use_ssl=account['use_ssl']
    )
    
    # Test a range around where we expect the problem (sequence 19 based on the logs)
    print("Testing emails from offset 15...")
    try:
        emails, has_more, total = connector.fetch_smart_batch(
            email_manager=None,  # We don't need manager for this test
            start_offset=15,
            fetch_size=10
        )
        
        print(f"Successfully fetched {len(emails)} emails from offset 15")
        for i, email in enumerate(emails):
            seq = 15 + i
            print(f"  Email {seq}: message_id={email.get('message_id', 'MISSING')}, subject={email.get('subject', 'NO SUBJECT')[:50]}")
            if email.get('message_id') is None:
                print(f"    *** FOUND CORRUPTED EMAIL AT SEQUENCE {seq} ***")
                print(f"    From: {email.get('from_addr')}")
                print(f"    Date: {email.get('date_utc')}")
                print(f"    Body preview: {email.get('body_text', '')[:100]}...")
                return seq
                
    except Exception as e:
        print(f"Error testing emails: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting diagnostic script...")
    try:
        print("Calling test_specific_emails()...")
        corrupted_seq = test_specific_emails()
        if corrupted_seq:
            print(f"\nCorrupted email confirmed at sequence {corrupted_seq}")
        else:
            print("\nNo corrupted emails found in tested range")
    except Exception as e:
        print(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
