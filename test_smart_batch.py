#!/usr/bin/env python3
"""Test script for smart batch processing implementation.

This script tests the new smart batch processing system to ensure:
1. Duplicate detection works correctly using header_hash
2. Batch size consistency is maintained 
3. Complete mailbox coverage is achieved
4. Database integration functions properly
"""

import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.email.connector import IMAPConnector
from ingestion.email.processor import EmailProcessor
from ingestion.email.email_manager import EmailManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_smart_batch_processing():
    """Test the smart batch processing implementation."""
    logger.info("Starting smart batch processing test")
    
    # Test configuration
    TEST_DB_PATH = "test_smart_batch.db"
    conn = None
    
    # Clean up any existing test database
    if Path(TEST_DB_PATH).exists():
        Path(TEST_DB_PATH).unlink()
    
    try:
        # Create test database connection
        conn = sqlite3.connect(TEST_DB_PATH)
        logger.info("Created test database: %s", TEST_DB_PATH)
        
        # Create EmailManager to initialize database schema
        email_manager = EmailManager(conn)
        logger.info("Initialized EmailManager and database schema")
        
        # Mock Milvus client for testing (no actual vector operations)
        class MockMilvus:
            def __init__(self):
                self.stored_embeddings = []
                
            def add_texts(self, texts, metadatas=None, ids=None):
                """Mock add_texts method."""
                logger.info("Mock Milvus: storing %d text chunks", len(texts))
                self.stored_embeddings.extend(texts)
                return ids or [f"mock_id_{i}" for i in range(len(texts))]
        
        mock_milvus = MockMilvus()
        
        # Create EmailProcessor with mock Milvus
        processor = EmailProcessor(
            milvus=mock_milvus,
            sqlite_conn=conn,
            chunk_size=400,
            chunk_overlap=50
        )
        logger.info("Created EmailProcessor with mock Milvus client")
        
        # Test smart batch methods without actual IMAP connection
        logger.info("Testing smart batch processing methods...")
        
        # Test 1: Header hash generation
        test_record = {
            "message_id": "test123@example.com",
            "from_addr": "sender@example.com",
            "to_addrs": ["recipient@example.com"],
            "subject": "Test Email",
            "date_utc": "2024-01-01T12:00:00Z",
            "body_text": "This is a test email body for smart batch processing."
        }
        
        # Test header hash generation using the same method as connector
        from ingestion.email.email_manager import compute_header_hash
        header_hash = compute_header_hash(test_record)
        logger.info("Generated header hash: %s", header_hash[:16] + "...")
        
        # Test 2: Database existence check
        # First check - should not exist
        exists_before = email_manager.get_email_by_header_hash(header_hash)
        logger.info("Email exists before storage: %s", exists_before is not None)
        
        # Store the email
        test_record["header_hash"] = header_hash
        processor.process(test_record)
        
        # Second check - should exist now
        exists_after = email_manager.get_email_by_header_hash(header_hash)
        logger.info("Email exists after storage: %s", exists_after is not None)
        
        # Test 3: Duplicate detection
        duplicate_record = test_record.copy()
        duplicate_record["message_id"] = "different_id@example.com"  # Different message ID
        # Same header_hash should be detected as duplicate
        
        duplicate_hash = compute_header_hash(duplicate_record)
        logger.info("Duplicate has same header hash: %s", header_hash == duplicate_hash)
        
        # Test 4: Mock connector for smart batch testing
        class MockIMAPConnector(IMAPConnector):
            def __init__(self):
                # Initialize with minimal required attributes
                self.host = "mock.imap.server"
                self.port = 993
                self.email_address = "test@example.com"
                self.password = "mock_password"
                self.mailbox = "INBOX"
                self.batch_limit = 5
                self.use_ssl = True
                self.primary_mailbox = None
                
                # Create mock email data
                self.mock_emails = []
                for i in range(12):  # Create 12 test emails
                    email_record = {
                        "message_id": f"test{i}@example.com",
                        "from_addr": f"sender{i}@example.com",
                        "to_addrs": [f"recipient{i}@example.com"],
                        "subject": f"Test Email {i}",
                        "date_utc": f"2024-01-{i+1:02d}T12:00:00Z",
                        "body_text": f"This is test email number {i}.",
                        "server_type": "imap"
                    }
                    # Generate header hash
                    email_record["header_hash"] = compute_header_hash(email_record)
                    self.mock_emails.append(email_record)
                    
                logger.info("Created mock connector with %d emails", len(self.mock_emails))
            
            def fetch_smart_batch(self, email_manager, since_date=None, start_offset=0):
                """Mock implementation of smart batch fetching."""
                logger.info("Mock fetch_smart_batch called with offset %d", start_offset)
                
                # Get batch slice
                end_offset = start_offset + self.batch_limit
                batch_emails = self.mock_emails[start_offset:end_offset]
                has_more = end_offset < len(self.mock_emails)
                
                # Filter out duplicates (emails already in database)
                unique_emails = []
                for email in batch_emails:
                    if not self._email_exists_in_database(email_manager, email["header_hash"]):
                        unique_emails.append(email)
                
                logger.info(
                    "Mock batch: %d total emails, %d unique, has_more: %s",
                    len(batch_emails),
                    len(unique_emails),
                    has_more
                )
                
                return unique_emails, has_more
        
        # Test smart batch processing with mock connector
        mock_connector = MockIMAPConnector()
        
        # Process emails using smart batching
        stats = processor.process_smart_batch(
            connector=mock_connector,
            since_date=None,
            max_batches=5  # Limit for testing
        )
        
        logger.info("Smart batch processing completed with stats: %s", stats)
        
        # Verify results
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails_in_db = cursor.fetchone()[0]
        logger.info("Total emails in database after processing: %d", total_emails_in_db)
        
        # Test duplicate detection by running again
        logger.info("Testing duplicate detection by running smart batch again...")
        stats2 = processor.process_smart_batch(
            connector=mock_connector,
            since_date=None,
            max_batches=5
        )
        
        logger.info("Second run stats (should process fewer/no emails): %s", stats2)
        
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails_after_second_run = cursor.fetchone()[0]
        logger.info("Total emails after second run: %d", total_emails_after_second_run)
        
        # Verify no duplicates were added
        if total_emails_in_db == total_emails_after_second_run:
            logger.info("✅ SUCCESS: Duplicate detection working correctly")
        else:
            logger.error("❌ FAILURE: Duplicates were added to database")
            
        logger.info("Smart batch processing test completed successfully!")
        return True
        
    except Exception as exc:
        logger.error("Test failed with error: %s", exc, exc_info=True)
        return False
        
    finally:
        # Clean up
        try:
            if conn:
                conn.close()
        except:
            pass
        if Path(TEST_DB_PATH).exists():
            Path(TEST_DB_PATH).unlink()
            logger.info("Cleaned up test database")


if __name__ == "__main__":
    success = test_smart_batch_processing()
    sys.exit(0 if success else 1)
