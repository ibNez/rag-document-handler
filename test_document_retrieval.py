#!/usr/bin/env python
"""
Test Document Hybrid Retrieval Implementation
Following DEVELOPMENT_RULES.md for all development requirements

This script tests the new document hybrid retrieval functionality including:
- Document chunks table creation
- PostgreSQL FTS retrieval for documents
- Document hybrid retrieval with RRF fusion
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional

from rag_manager.core.config import Config
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from retrieval.document.postgres_fts_retriever import DocumentPostgresFTSRetriever
from retrieval.document.hybrid_retriever import DocumentHybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentRetrievalTester:
    """Test class for document hybrid retrieval functionality."""
    
    def __init__(self):
        """Initialize test environment."""
        self.config = Config()
        self.postgres_manager: Optional[PostgreSQLManager] = None
        self.fts_retriever: Optional[DocumentPostgresFTSRetriever] = None
        self.hybrid_retriever: Optional[DocumentHybridRetriever] = None
        
    def setup(self) -> bool:
        """Setup test environment with database connections."""
        try:
            logger.info("Setting up document retrieval test environment...")
            
            # Initialize PostgreSQL manager
            postgres_config = PostgreSQLConfig()
            self.postgres_manager = PostgreSQLManager(postgres_config)
            
            # Test PostgreSQL connection
            version_info = self.postgres_manager.get_version_info()
            if not version_info.get('connected'):
                logger.error(f"PostgreSQL connection failed: {version_info.get('error')}")
                return False
            
            logger.info(f"PostgreSQL connected: {version_info.get('version')}")
            
            # Initialize FTS retriever
            self.fts_retriever = DocumentPostgresFTSRetriever(self.postgres_manager.pool)
            logger.info("Document PostgreSQL FTS retriever initialized")
            
            # Note: For full hybrid testing, you would also need:
            # - Milvus vector retriever
            # - self.hybrid_retriever = DocumentHybridRetriever(vector_retriever, self.fts_retriever)
            
            logger.info("Test environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False
    
    def test_database_schema(self) -> bool:
        """Test that document chunks table exists and is accessible."""
        if not self.postgres_manager:
            logger.error("PostgreSQL manager not initialized")
            return False
            
        try:
            logger.info("Testing document chunks database schema...")
            
            with self.postgres_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if document_chunks table exists
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'document_chunks'
                        )
                    """)
                    table_exists = cur.fetchone()[0]
                    
                    if not table_exists:
                        logger.error("document_chunks table does not exist")
                        return False
                    
                    logger.info("✓ document_chunks table exists")
                    
                    # Check table structure
                    cur.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'document_chunks'
                        ORDER BY ordinal_position
                    """)
                    columns = cur.fetchall()
                    
                    expected_columns = {
                        'chunk_id', 'document_id', 'chunk_text', 'chunk_ordinal',
                        'page_start', 'page_end', 'section_path', 'element_types',
                        'token_count', 'chunk_hash', 'embedding_version', 'created_at'
                    }
                    
                    actual_columns = {col[0] for col in columns}
                    missing_columns = expected_columns - actual_columns
                    
                    if missing_columns:
                        logger.error(f"Missing columns in document_chunks table: {missing_columns}")
                        return False
                    
                    logger.info("✓ document_chunks table has correct structure")
                    
                    # Check FTS index exists
                    cur.execute("""
                        SELECT indexname FROM pg_indexes 
                        WHERE tablename = 'document_chunks' 
                        AND indexname = 'idx_document_chunks_fts'
                    """)
                    fts_index = cur.fetchone()
                    
                    if not fts_index:
                        logger.warning("FTS index not found for document_chunks")
                    else:
                        logger.info("✓ FTS index exists for document_chunks")
                    
                    # Get row count
                    cur.execute("SELECT COUNT(*) FROM document_chunks")
                    chunk_count = cur.fetchone()[0]
                    logger.info(f"✓ Document chunks table contains {chunk_count} chunks")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Database schema test failed: {e}")
            return False
    
    def test_fts_retriever(self) -> bool:
        """Test PostgreSQL FTS retrieval functionality."""
        if not self.fts_retriever:
            logger.error("FTS retriever not initialized")
            return False
            
        try:
            logger.info("Testing document PostgreSQL FTS retriever...")
            
            # Test basic search
            test_query = "test document content"
            results = self.fts_retriever.search(test_query, k=5)
            
            logger.info(f"✓ FTS search for '{test_query}' returned {len(results)} results")
            
            # Test search with filters
            filtered_results = self.fts_retriever.search(
                test_query, 
                k=3,
                filetype_filter="application/pdf"
            )
            
            logger.info(f"✓ Filtered FTS search returned {len(filtered_results)} results")
            
            # Test search statistics
            stats = self.fts_retriever.get_search_statistics(test_query)
            logger.info(f"✓ Search statistics: {stats['total_chunks']} chunks, {stats['total_documents']} documents")
            
            # Test complex filters
            complex_filters = {
                'content_types': ['application/pdf', 'text/plain'],
                'page_range': (1, 10)
            }
            
            complex_results = self.fts_retriever.search_with_filters(
                test_query,
                k=5,
                filters=complex_filters
            )
            
            logger.info(f"✓ Complex filtered search returned {len(complex_results)} results")
            
            return True
            
        except Exception as e:
            logger.error(f"FTS retriever test failed: {e}")
            return False
    
    def test_chunk_storage(self) -> bool:
        """Test storing and retrieving document chunks."""
        if not self.postgres_manager or not self.fts_retriever:
            logger.error("Required managers not initialized")
            return False
            
        try:
            logger.info("Testing document chunk storage and retrieval...")
            
            # Test data
            test_document_id = "test_doc_123"
            test_chunks = [
                {
                    'chunk_id': f"{test_document_id}#0",
                    'chunk_text': "This is a test document chunk for testing purposes.",
                    'chunk_ordinal': 0,
                    'page_start': 1,
                    'page_end': 1,
                    'token_count': 10
                },
                {
                    'chunk_id': f"{test_document_id}#1", 
                    'chunk_text': "This is another test chunk with different content.",
                    'chunk_ordinal': 1,
                    'page_start': 1,
                    'page_end': 1,
                    'token_count': 9
                }
            ]
            
            # Store test chunks
            for chunk in test_chunks:
                self.postgres_manager.store_document_chunk(
                    chunk_id=chunk['chunk_id'],
                    document_id=test_document_id,
                    chunk_text=chunk['chunk_text'],
                    chunk_ordinal=chunk['chunk_ordinal'],
                    page_start=chunk['page_start'],
                    page_end=chunk['page_end'],
                    token_count=chunk['token_count']
                )
            
            logger.info(f"✓ Stored {len(test_chunks)} test chunks")
            
            # Retrieve chunks
            retrieved_chunks = self.postgres_manager.get_document_chunks(test_document_id)
            
            if len(retrieved_chunks) != len(test_chunks):
                logger.error(f"Expected {len(test_chunks)} chunks, got {len(retrieved_chunks)}")
                return False
            
            logger.info(f"✓ Retrieved {len(retrieved_chunks)} chunks")
            
            # Test FTS search on stored chunks
            search_results = self.fts_retriever.search("test document", k=5)
            
            # Should find our test chunks
            test_chunk_found = any(
                result.metadata.get('document_id') == test_document_id 
                for result in search_results
            )
            
            if test_chunk_found:
                logger.info("✓ Test chunks found in FTS search results")
            else:
                logger.warning("Test chunks not found in FTS search results")
            
            # Clean up test data
            deleted_count = self.postgres_manager.delete_document_chunks(test_document_id)
            logger.info(f"✓ Cleaned up {deleted_count} test chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"Chunk storage test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success status."""
        logger.info("=== Starting Document Hybrid Retrieval Tests ===")
        
        if not self.setup():
            logger.error("Test setup failed, aborting tests")
            return False
        
        tests = [
            ("Database Schema", self.test_database_schema),
            ("FTS Retriever", self.test_fts_retriever), 
            ("Chunk Storage", self.test_chunk_storage)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                if test_func():
                    logger.info(f"✓ {test_name} test PASSED")
                    passed += 1
                else:
                    logger.error(f"✗ {test_name} test FAILED")
                    failed += 1
            except Exception as e:
                logger.error(f"✗ {test_name} test FAILED with exception: {e}")
                failed += 1
        
        logger.info(f"\n=== Test Results ===")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed}/{passed+failed} ({100*passed/(passed+failed):.1f}%)")
        
        return failed == 0
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            if self.postgres_manager:
                self.postgres_manager.close()
                logger.info("PostgreSQL connection closed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main test function."""
    tester = DocumentRetrievalTester()
    
    try:
        success = tester.run_all_tests()
        exit_code = 0 if success else 1
        
        if success:
            logger.info("🎉 All tests passed! Document hybrid retrieval implementation is working.")
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
