#!/usr/bin/env python3
"""
Test deletion using the actual MilvusManager to see if langchain queries work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_manager.core.config import Config
from rag_manager.managers.milvus_manager import MilvusManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_with_milvus_manager():
    """Test querying and deletion using the actual MilvusManager"""
    config = Config()
    milvus_manager = MilvusManager(config)
    
    # Ensure vector store is initialized
    logger.info("Initializing vector store...")
    milvus_manager._ensure_vector_store()
    
    # Try to use langchain's search to see what's in there
    logger.info("Testing vector store search...")
    try:
        # Check if vector store was initialized
        if milvus_manager.vector_store is None:
            logger.error("Vector store is None after initialization!")
            return
        
        # Use a simple search to see if we can find anything
        results = milvus_manager.vector_store.similarity_search("test", k=5)
        logger.info(f"Similarity search returned {len(results)} results")
        
        for i, doc in enumerate(results):
            logger.info(f"Document {i}: source='{doc.metadata.get('source')}', "
                       f"document_id='{doc.metadata.get('document_id')}', "
                       f"chunk_id='{doc.metadata.get('chunk_id')}'")
        
        # Test deletion with actual filenames found
        if results:
            test_filename = results[0].metadata.get('source')
            if test_filename:
                logger.info(f"Testing deletion for filename: {test_filename}")
                deletion_result = milvus_manager.delete_document(filename=test_filename)
                logger.info(f"Deletion result: {deletion_result}")
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}", exc_info=True)

def main():
    print("\n=== Test with MilvusManager ===")
    test_with_milvus_manager()
    return 0

if __name__ == "__main__":
    sys.exit(main())
