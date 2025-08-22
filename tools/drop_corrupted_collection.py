#!/usr/bin/env python3
"""
Drop and recreate the corrupted Milvus collection.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import Collection, utility, connections
from rag_manager.core.config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_to_milvus():
    """Connect to Milvus"""
    config = Config()
    
    try:
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        logger.info(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False

def drop_corrupted_collection(collection_name="documents"):
    """Drop the corrupted collection"""
    if not utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' does not exist - nothing to drop")
        return True
    
    try:
        logger.warning(f"DROPPING corrupted collection '{collection_name}' with {Collection(collection_name).num_entities} entities")
        
        # Confirm the drop
        response = input(f"Are you sure you want to drop collection '{collection_name}'? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Drop cancelled by user")
            return False
        
        utility.drop_collection(collection_name)
        logger.info(f"Collection '{collection_name}' dropped successfully")
        
        # Verify it's gone
        if utility.has_collection(collection_name):
            logger.error("Collection still exists after drop!")
            return False
        else:
            logger.info("Collection drop confirmed")
            return True
        
    except Exception as e:
        logger.error(f"Failed to drop collection: {e}")
        return False

def main():
    if not connect_to_milvus():
        return 1
    
    print("\n=== Drop Corrupted Milvus Collection ===")
    print("WARNING: This will permanently delete all embeddings in the collection!")
    print("Only proceed if you're sure the collection is corrupted and cannot be recovered.")
    
    success = drop_corrupted_collection()
    
    if success:
        print("\nCollection dropped successfully!")
        print("The application will recreate it automatically when you upload a new document.")
    else:
        print("Collection drop failed or was cancelled")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
