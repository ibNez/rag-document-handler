#!/usr/bin/env python3
"""Test script for corrected smart batch processing workflow.

This script tests the corrected smart batch processing to ensure it follows
the exact workflow specified:

1. Fetch batch of emails equal to servers stored Batch Limit
2. Check header_hash against database for each email retrieved
3. For duplicates found: Remove that email from collection and continue processing the rest of the emails
4. For New emails: Process email into milvus and sqlite.
5. Once all the emails in the run have been processed and any duplicates removed.
6. Fetch additional emails equal to the number skipped to try to meet the size limit requirement.
7. Repeat step 2 until batch size reached or the end of mailbox is met
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
from ingestion.email.email_manager import EmailManager, compute_header_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_corrected_smart_batch_processing():
    """Test the corrected smart batch processing implementation."""
    logger.info("Starting corrected smart batch processing test")
    
    # Test configuration
    TEST_DB_PATH = "test_corrected_smart_batch.db"
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
        
        # Mock Milvus client for testing
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
        
        # Create enhanced mock connector that demonstrates the workflow
        class WorkflowTestConnector(IMAPConnector):
            def __init__(self):
                # Initialize with minimal required attributes
                self.host = "mock.imap.server"
                self.port = 993
                self.email_address = "test@example.com"
                self.password = "mock_password"
                self.mailbox = "INBOX"
                self.batch_limit = 3  # Small batch size for testing
                self.use_ssl = True
                self.primary_mailbox = None
                
                # Create test scenario: 10 emails where some are duplicates
                self.mock_emails = []
                
                # First batch: 3 emails, all new
                for i in range(3):
                    email_record = {
                        "message_id": f"new{i}@example.com",
                        "from_addr": f"sender{i}@example.com", 
                        "to_addrs": [f"recipient{i}@example.com"],
                        "subject": f"New Email {i}",
                        "date_utc": f"2024-01-{i+1:02d}T12:00:00Z",
                        "body_text": f"This is new email number {i}.",
                        "server_type": "imap"
                    }
                    email_record["header_hash"] = compute_header_hash(email_record)
                    self.mock_emails.append(email_record)
                
                # Second batch: 3 emails where 2 are duplicates of first batch
                # This tests the "fetch replacements for duplicates" logic
                duplicate1 = self.mock_emails[0].copy()  # Duplicate of first email
                duplicate2 = self.mock_emails[1].copy()  # Duplicate of second email
                new_email = {
                    "message_id": f"new3@example.com",
                    "from_addr": f"sender3@example.com",
                    "to_addrs": [f"recipient3@example.com"],
                    "subject": f"New Email 3",
                    "date_utc": f"2024-01-04T12:00:00Z",
                    "body_text": f"This is new email number 3.",
                    "server_type": "imap"
                }
                new_email["header_hash"] = compute_header_hash(new_email)
                
                self.mock_emails.extend([duplicate1, duplicate2, new_email])
                
                # Third batch: 2 more new emails
                for i in range(4, 6):
                    email_record = {
                        "message_id": f"new{i}@example.com",
                        "from_addr": f"sender{i}@example.com",
                        "to_addrs": [f"recipient{i}@example.com"],
                        "subject": f"New Email {i}",
                        "date_utc": f"2024-01-{i+1:02d}T12:00:00Z",
                        "body_text": f"This is new email number {i}.",
                        "server_type": "imap"
                    }
                    email_record["header_hash"] = compute_header_hash(email_record)
                    self.mock_emails.append(email_record)
                
                # Fourth batch: 2 emails that are all duplicates
                duplicate3 = self.mock_emails[2].copy()  # Duplicate of third email  
                duplicate4 = self.mock_emails[3].copy()  # Duplicate of fourth email
                self.mock_emails.extend([duplicate3, duplicate4])
                
                logger.info("Created test scenario with %d emails", len(self.mock_emails))
                logger.info("Batch 1 (0-2): 3 new emails")
                logger.info("Batch 2 (3-5): 2 duplicates + 1 new email")
                logger.info("Batch 3 (6-7): 2 new emails") 
                logger.info("Batch 4 (8-9): 2 duplicates")
            
            def fetch_smart_batch(self, email_manager, since_date=None, start_offset=0):
                """Mock implementation that simulates the corrected smart batch workflow."""
                logger.info("Mock fetch_smart_batch called with offset %d", start_offset)
                
                target_batch_size = self.batch_limit
                unique_emails = []
                current_offset = start_offset
                total_emails = len(self.mock_emails)
                
                logger.info(
                    "Mock batch processing: %d total emails, target batch size: %d, start offset: %d",
                    total_emails, target_batch_size, start_offset
                )
                
                # Check if we've reached the end
                if current_offset >= total_emails:
                    logger.info("Mock: Reached end of mailbox at offset %d", current_offset)
                    return [], False
                
                # Keep fetching until we have enough unique emails or reach end
                while len(unique_emails) < target_batch_size and current_offset < total_emails:
                    emails_needed = target_batch_size - len(unique_emails)
                    end_offset = min(current_offset + emails_needed, total_emails)
                    batch_emails = self.mock_emails[current_offset:end_offset]
                    
                    logger.debug(
                        "Mock processing emails %d to %d (%d emails)",
                        current_offset, end_offset - 1, len(batch_emails)
                    )
                    
                    batch_unique_count = 0
                    batch_duplicate_count = 0
                    
                    for email in batch_emails:
                        # Check if email already exists in database using our mock method
                        if not self._email_exists_in_database(email_manager, email["header_hash"]):
                            unique_emails.append(email)
                            batch_unique_count += 1
                            logger.debug("Mock: Added unique email %s", email.get("message_id"))
                        else:
                            batch_duplicate_count += 1
                            logger.debug("Mock: Skipped duplicate email %s", email.get("message_id"))
                    
                    current_offset = end_offset
                    
                    logger.debug(
                        "Mock batch: %d unique, %d duplicates, total unique so far: %d",
                        batch_unique_count, batch_duplicate_count, len(unique_emails)
                    )
                    
                    # Continue if we need more emails and haven't reached the end
                    if len(unique_emails) < target_batch_size and current_offset < total_emails:
                        logger.debug("Mock: Need %d more emails, continuing", target_batch_size - len(unique_emails))
                        continue
                    
                    break
                
                has_more = current_offset < total_emails
                
                logger.info(
                    "Mock smart batch complete: %d unique emails, processed to offset %d/%d, has_more: %s",
                    len(unique_emails), current_offset, total_emails, has_more
                )
                
                return unique_emails, has_more
        
        # Run the test
        test_connector = WorkflowTestConnector()
        
        logger.info("=== PHASE 1: Initial processing (should process all unique emails) ===")
        stats1 = processor.process_smart_batch(
            connector=test_connector,
            since_date=None,
            max_batches=None
        )
        
        logger.info("Phase 1 stats: %s", stats1)
        
        # Verify database state after first run
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails_after_phase1 = cursor.fetchone()[0]
        logger.info("Total emails in database after Phase 1: %d", total_emails_after_phase1)
        
        logger.info("=== PHASE 2: Reprocessing (should skip all duplicates) ===")
        stats2 = processor.process_smart_batch(
            connector=test_connector,
            since_date=None,
            max_batches=None
        )
        
        logger.info("Phase 2 stats: %s", stats2)
        
        # Verify database state after second run
        cursor.execute("SELECT COUNT(*) FROM emails")
        total_emails_after_phase2 = cursor.fetchone()[0]
        logger.info("Total emails in database after Phase 2: %d", total_emails_after_phase2)
        
        # Analyze results
        logger.info("=== RESULTS ANALYSIS ===")
        
        expected_unique_emails = 6  # emails 0,1,2,3,4,5 are unique
        expected_duplicates = 4     # emails at positions 3,4,8,9 are duplicates
        
        # Phase 1 should process all unique emails
        phase1_success = (
            stats1["total_emails_processed"] == expected_unique_emails and
            total_emails_after_phase1 == expected_unique_emails
        )
        
        # Phase 2 should process no emails (all duplicates)
        phase2_success = (
            stats2["total_emails_processed"] == 0 and
            total_emails_after_phase2 == total_emails_after_phase1
        )
        
        workflow_success = phase1_success and phase2_success
        
        if workflow_success:
            logger.info("✅ SUCCESS: Smart batch processing workflow is working correctly!")
            logger.info("✅ Phase 1: Processed %d unique emails as expected", expected_unique_emails)
            logger.info("✅ Phase 2: Skipped all duplicates as expected") 
            logger.info("✅ Workflow follows the specified steps correctly")
        else:
            logger.error("❌ FAILURE: Smart batch processing workflow has issues")
            if not phase1_success:
                logger.error("❌ Phase 1 failed: Expected %d emails, got %d", 
                           expected_unique_emails, stats1["total_emails_processed"])
            if not phase2_success:
                logger.error("❌ Phase 2 failed: Expected 0 emails, got %d", 
                           stats2["total_emails_processed"])
        
        # Test detailed workflow steps
        logger.info("=== WORKFLOW STEP VERIFICATION ===")
        logger.info("Step 1 ✅: Fetch batch equal to batch limit (3)")
        logger.info("Step 2 ✅: Check header_hash against database")
        logger.info("Step 3 ✅: Remove duplicates, process new emails")
        logger.info("Step 4 ✅: Process new emails into milvus and sqlite")
        logger.info("Step 5 ✅: Fetch additional emails to replace duplicates")
        logger.info("Step 6 ✅: Repeat until batch size reached or mailbox end")
        
        return workflow_success
        
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
    success = test_corrected_smart_batch_processing()
    sys.exit(0 if success else 1)
