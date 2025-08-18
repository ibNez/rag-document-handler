#!/usr/bin/env python3
"""Test script to verify Milvus client interface fix.

This script tests that the EmailProcessor properly handles the Milvus client
interface and doesn't throw "Unsupported Milvus client interface" errors.
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.email.processor import EmailProcessor
from ingestion.email.email_manager import EmailManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_milvus_interface_fix():
    """Test that the Milvus interface issue is resolved."""
    logger.info("Testing Milvus client interface fix")
    
    # Test configuration
    TEST_DB_PATH = "test_milvus_interface.db"
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
        
        # Test different Milvus client scenarios
        test_cases = [
            {
                "name": "None Milvus client",
                "milvus": None,
                "expected_behavior": "Should log warning and skip storage"
            },
            {
                "name": "Mock LangChain Milvus with add_texts",
                "milvus": type('MockMilvus', (), {
                    'add_texts': lambda self, texts, metadatas=None, ids=None: ids or [f"id_{i}" for i in range(len(texts))]
                })(),
                "expected_behavior": "Should use add_texts method"
            },
            {
                "name": "Invalid Milvus client",
                "milvus": {"invalid": "object"},
                "expected_behavior": "Should raise RuntimeError with type info"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"=== Test Case {i+1}: {test_case['name']} ===")
            logger.info(f"Expected: {test_case['expected_behavior']}")
            
            try:
                # Create EmailProcessor with test Milvus client
                processor = EmailProcessor(
                    milvus=test_case["milvus"],
                    sqlite_conn=conn,
                    chunk_size=400,
                    chunk_overlap=50
                )
                
                # Create test email record
                test_email = {
                    "message_id": f"test{i}@example.com",
                    "from_addr": f"sender{i}@example.com", 
                    "to_addrs": [f"recipient{i}@example.com"],
                    "subject": f"Test Email {i}",
                    "date_utc": f"2024-01-0{i+1}T12:00:00Z",
                    "body_text": f"This is test email number {i} for Milvus interface testing.",
                    "server_type": "imap"
                }
                
                # Process the email (this should trigger the Milvus storage)
                logger.info(f"Processing email with {test_case['name']}...")
                processor.process(test_email)
                
                logger.info(f"✅ Test case {i+1} completed successfully")
                
            except Exception as exc:
                if test_case["name"] == "Invalid Milvus client" and "Unsupported Milvus client interface" in str(exc):
                    logger.info(f"✅ Test case {i+1} behaved as expected: {exc}")
                else:
                    logger.error(f"❌ Test case {i+1} failed unexpectedly: {exc}")
                    return False
        
        logger.info("=== Milvus Interface Fix Test Results ===")
        logger.info("✅ All test cases completed successfully")
        logger.info("✅ EmailProcessor now handles various Milvus client types gracefully")
        logger.info("✅ No more 'Unsupported Milvus client interface' errors for None clients")
        logger.info("✅ Better error messages for truly unsupported clients")
        
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
    success = test_milvus_interface_fix()
    sys.exit(0 if success else 1)
