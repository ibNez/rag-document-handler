#!/usr/bin/env python3
"""
Debug tool to examine documents in Milvus collections and test deletion expressions.
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

def list_collection_documents(collection_name="documents"):
    """List all documents in the collection grouped by source"""
    if not utility.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        return
    
    col = Collection(collection_name)
    col.load()
    
    total_entities = col.num_entities
    logger.info(f"Total entities in collection '{collection_name}': {total_entities}")
    
    if total_entities == 0:
        logger.info("Collection is empty")
        return
    
    try:
        # Query all documents to see what sources exist
        # Try different expressions to see what works
        queries_to_try = [
            "source != ''",
            "document_id != ''", 
            "chunk_id != ''",
            "source != 'unknown'"
        ]
        
        all_docs = []
        for expr in queries_to_try:
            logger.info(f"Trying query expression: {expr}")
            try:
                docs = col.query(
                    expr=expr,
                    output_fields=["document_id", "source", "chunk_id", "content_hash"],
                    limit=50
                )
                logger.info(f"Query '{expr}' returned {len(docs)} documents")
                if docs:
                    all_docs = docs
                    break
            except Exception as e:
                logger.warning(f"Query '{expr}' failed: {e}")
        
        if not all_docs:
            # Try querying without expression
            logger.info("Trying query without expression...")
            try:
                all_docs = col.query(
                    expr="",
                    output_fields=["document_id", "source", "chunk_id", "content_hash"],
                    limit=50
                )
                logger.info(f"Query without expression returned {len(all_docs)} documents")
            except Exception as e:
                logger.error(f"Query without expression failed: {e}")
        
        logger.info(f"Retrieved {len(all_docs)} documents from collection")
        
        # Group by source
        sources = {}
        for doc in all_docs:
            source = doc.get('source', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(doc)
        
        logger.info(f"Found {len(sources)} unique sources:")
        for source, docs in sources.items():
            logger.info(f"  Source: '{source}' - {len(docs)} chunks")
            
            # Show first few chunk IDs for each source
            chunk_ids = [doc.get('chunk_id', 'unknown') for doc in docs[:5]]
            logger.info(f"    Sample chunk IDs: {chunk_ids}")
            
            # Show document IDs
            doc_ids = list(set(doc.get('document_id', 'unknown') for doc in docs))
            logger.info(f"    Document IDs: {doc_ids}")
        
        return sources
        
    except Exception as e:
        logger.error(f"Failed to query collection: {e}")
        return None

def test_deletion_expression(collection_name="documents", filename=None, document_id=None):
    """Test what a deletion expression would match"""
    if not filename and not document_id:
        logger.error("Either filename or document_id must be provided")
        return
    
    if not utility.has_collection(collection_name):
        logger.error(f"Collection '{collection_name}' does not exist")
        return
    
    col = Collection(collection_name)
    col.load()
    
    # Build the same expression used in delete_document
    if document_id and filename:
        expr = f'document_id == "{document_id}" or source == "{filename}"'
    elif document_id:
        expr = f'document_id == "{document_id}"'
    else:
        expr = f'source == "{filename}"'
    
    logger.info(f"Testing deletion expression: {expr}")
    
    try:
        matching_docs = col.query(
            expr=expr,
            output_fields=["document_id", "source", "chunk_id", "content_hash"],
            limit=100
        )
        
        logger.info(f"Expression would match {len(matching_docs)} documents:")
        for doc in matching_docs:
            logger.info(f"  - document_id: '{doc.get('document_id')}', source: '{doc.get('source')}', chunk_id: '{doc.get('chunk_id')}'")
        
        return matching_docs
        
    except Exception as e:
        logger.error(f"Failed to test deletion expression: {e}")
        return None

def main():
    if not connect_to_milvus():
        return 1
    
    print("\n=== Milvus Document Debug Tool ===")
    
    # List all documents
    print("\n1. Listing all documents in collection:")
    sources = list_collection_documents()
    
    if sources:
        print("\n2. Testing deletion expressions:")
        # Test deletion for each source
        for source in sources.keys():
            print(f"\nTesting deletion for source: '{source}'")
            test_deletion_expression(filename=source)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
