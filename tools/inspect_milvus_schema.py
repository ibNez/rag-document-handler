#!/usr/bin/env python3
"""
Check Milvus collection schema and try different query approaches.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import Collection, utility, connections
from rag_manager.core.config import Config
import json
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

def inspect_collection_schema(collection_name="documents"):
    """Inspect the collection schema and try different approaches"""
    if not utility.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        return
    
    col = Collection(collection_name)
    col.load()
    
    # Get schema
    logger.info("Collection Schema:")
    schema = col.schema
    logger.info(f"Description: {schema.description}")
    
    logger.info("Fields:")
    for field in schema.fields:
        logger.info(f"  - {field.name}: {field.dtype} (primary: {field.is_primary})")
    
    # Get collection stats
    logger.info(f"Total entities: {col.num_entities}")
    
    # Try to get any records using scan
    try:
        logger.info("Attempting to scan collection...")
        # Try to get primary keys first
        pk_field = None
        for field in schema.fields:
            if field.is_primary:
                pk_field = field.name
                break
        
        if pk_field:
            logger.info(f"Primary key field: {pk_field}")
            
            # Try a simple count query
            try:
                result = col.query(
                    expr=f"{pk_field} >= 0",
                    output_fields=[pk_field],
                    limit=5
                )
                logger.info(f"Sample primary keys: {[r[pk_field] for r in result[:5]]}")
                
                # Now try to get full records
                if result:
                    first_pk = result[0][pk_field]
                    full_record = col.query(
                        expr=f"{pk_field} == {first_pk}",
                        output_fields=["*"],
                        limit=1
                    )
                    logger.info(f"Sample full record: {full_record}")
                    
            except Exception as e:
                logger.error(f"Query by primary key failed: {e}")
    
    except Exception as e:
        logger.error(f"Collection inspection failed: {e}")

def main():
    if not connect_to_milvus():
        return 1
    
    print("\n=== Milvus Collection Schema Inspector ===")
    inspect_collection_schema()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
