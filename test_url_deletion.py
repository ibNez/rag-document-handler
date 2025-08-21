#!/usr/bin/env python3
"""
Test script to verify URL deletion functionality with improved logging.
"""
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/tonyphilip/Code/rag-document-handler')

# Configure logging to show all messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from ingestion.core.postgres_manager import PostgreSQLManager
from ingestion.url.manager import PostgreSQLURLManager

def test_url_deletion():
    """Test URL deletion with improved logging."""
    # Initialize managers
    postgres_manager = PostgreSQLManager()
    url_manager = PostgreSQLURLManager(postgres_manager)
    
    # Test URL ID from the database
    test_url_id = "92c5902c-0933-4b3d-a5e0-ea3272c876d8"
    
    print(f"\n=== Testing URL Deletion for ID: {test_url_id} ===")
    
    # Step 1: Check if URL exists
    print("\n1. Checking if URL exists...")
    url_rec = url_manager.get_url_by_id(test_url_id)
    if url_rec:
        print(f"✓ URL found: {url_rec['url']} (Title: {url_rec.get('title', 'N/A')})")
    else:
        print("✗ URL not found")
        return
    
    # Step 2: Test deletion
    print("\n2. Testing URL deletion...")
    result = url_manager.delete_url(test_url_id)
    print(f"Deletion result: {result}")
    
    # Step 3: Verify deletion
    print("\n3. Verifying deletion...")
    url_rec_after = url_manager.get_url_by_id(test_url_id)
    if url_rec_after:
        print("✗ URL still exists after deletion attempt")
    else:
        print("✓ URL successfully deleted")

if __name__ == "__main__":
    test_url_deletion()
