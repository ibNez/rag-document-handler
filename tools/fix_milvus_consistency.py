#!/usr/bin/env python3
"""
Try to fix Milvus collection consistency issues.
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

def fix_collection_consistency(collection_name="documents"):
    """Try to fix collection consistency issues"""
    if not utility.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        return
    
    col = Collection(collection_name)
    
    logger.info(f"Initial entity count: {col.num_entities}")
    
    # Try to flush and reload
    try:
        logger.info("Flushing collection...")
        col.flush()
        logger.info("Flush completed")
        
        logger.info("Reloading collection...")
        col.load()
        logger.info("Reload completed")
        
        logger.info(f"Entity count after flush/reload: {col.num_entities}")
        
        # Try a simple query again
        logger.info("Testing query after flush/reload...")
        result = col.query(
            expr="pk >= 0",
            output_fields=["pk", "source", "document_id"],
            limit=5
        )
        logger.info(f"Query returned {len(result)} results")
        
        if result:
            for i, r in enumerate(result[:3]):
                logger.info(f"Record {i}: pk={r.get('pk')}, source='{r.get('source')}', document_id='{r.get('document_id')}'")
        
        return len(result) > 0
        
    except Exception as e:
        logger.error(f"Fix attempt failed: {e}")
        return False

def main():
    if not connect_to_milvus():
        return 1
    
    print("\n=== Milvus Collection Consistency Fix ===")
    success = fix_collection_consistency()
    
    if success:
        print("Collection consistency restored!")
    else:
        print("Could not restore collection consistency")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
