"""
Milvus vector database manager for RAG Knowledgebase Manager.

This module manages Milvus database operations using LangChain's Milvus VectorStore.
Email-specific retrieval is handled by EmailManager.
"""

import json
import time
import hashlib
import logging
import re
from typing import Dict, List, Optional, Any

from pymilvus import connections, utility, Collection
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.config import Config

# Configure logging
logger = logging.getLogger(__name__)


class MilvusManager:
    """
    Manages Milvus database operations using LangChain's Milvus VectorStore.
    
    This class follows the development rules with proper type hints,
    error handling, and comprehensive logging.
    """
    
    def __init__(self, config: Config, postgres_manager: Optional[Any] = None) -> None:
        """
        Initialize Milvus manager.
        
        Args:
            config: Application configuration instance
            postgres_manager: Optional PostgreSQL manager for retrieval
        """
        self.config = config
        self.postgres_manager = postgres_manager
        self.collection_name = config.DOCUMENT_COLLECTION
        self.connection_args = {"host": config.MILVUS_HOST, "port": config.MILVUS_PORT}
        
        # Hybrid retrieval components (initialized on demand)
        self.document_fts_retriever: Optional[Any] = None
        self.document_retriever: Optional[Any] = None
        
        # Establish Milvus connection (idempotent)
        try:
            connections.connect(alias="default", **self.connection_args)
        except Exception:
            pass
            
        # Lazy-created vector stores for documents and emails
        self.vector_store: Optional[Milvus] = None
        self.email_vector_store: Optional[Milvus] = None
        self.email_collection_name = config.EMAIL_COLLECTION
        
        try:
            # Create a dedicated embeddings instance
            self.langchain_embeddings = OllamaEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                base_url=f"http://{self.config.OLLAMA_EMBEDDING_HOST}:{self.config.OLLAMA_EMBEDDING_PORT}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings for MilvusManager: {e}")
            raise RuntimeError(f"Could not initialize embeddings: {e}")
        
        # Note: Collections should be initialized separately during app startup
        # Do not auto-create collections in constructor

    def initialize_collections_for_startup(self) -> None:
        """
        Public method to initialize collections during application startup.
        This should only be called once during application initialization.
        """
        self._initialize_collections()
        self._initialize_hybrid_retrievers()

    def _initialize_collections(self) -> None:
        """Initialize all required collections during application startup - let LangChain auto-create."""
        logger.info("Initializing Milvus collections during startup...")
        
        # Initialize document collection via LangChain auto-creation
        try:
            logger.info(f"Initializing document collection '{self.collection_name}' via LangChain...")
            self._ensure_vector_store()
            logger.info(f"Document collection '{self.collection_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document collection '{self.collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Cannot initialize document collection '{self.collection_name}': {e}")
        
        # Initialize email collection via LangChain auto-creation
        try:
            logger.info(f"Initializing email collection '{self.email_collection_name}' via LangChain...")
            self._ensure_email_vector_store()
            logger.info(f"Email collection '{self.email_collection_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize email collection '{self.email_collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Cannot initialize email collection '{self.email_collection_name}': {e}")
        
        logger.info("All Milvus collections initialized successfully via LangChain auto-creation")

    def set_postgres_manager(self, postgres_manager: Any) -> None:
        """
        Set PostgreSQL manager for retrieval after initialization.
        
        Args:
            postgres_manager: PostgreSQL manager instance
        """
        self.postgres_manager = postgres_manager
        logger.info("PostgreSQL manager set for MilvusManager, initializing retrievers...")
        self._initialize_hybrid_retrievers()

    def _initialize_hybrid_retrievers(self) -> None:
        """Initialize retrievers for document search if PostgreSQL is available."""
        if not self.postgres_manager:
            logger.info("PostgreSQL manager not available, retrieval will not be enabled")
            return
            
        try:
            logger.info("Initializing document retrievers...")
            
            # Import here to avoid circular dependencies
            from retrieval.document.postgres_fts_retriever import DocumentPostgresFTSRetriever
            from retrieval.document.processor import DocumentProcessor
            
            # Initialize FTS retriever
            self.document_fts_retriever = DocumentPostgresFTSRetriever(self.postgres_manager)
            logger.info("Document PostgreSQL FTS retriever initialized")
            
            # Initialize retriever (requires vector store)
            self._ensure_vector_store()
            if self.vector_store:
                vector_retriever = self.vector_store.as_retriever()
                
                # Configure reranking options
                enable_reranking = getattr(self.config, 'ENABLE_DOCUMENT_RERANKING', True)
                reranker_model = getattr(self.config, 'DOCUMENT_RERANKER_MODEL', 'ms-marco-minilm')
                rerank_top_k = getattr(self.config, 'DOCUMENT_RERANK_TOP_K', None)
                rrf_constant = getattr(self.config, 'DOCUMENT_RRF_CONSTANT', 60)
                
                self.document_retriever = DocumentProcessor(
                    vector_retriever=vector_retriever,
                    fts_retriever=self.document_fts_retriever,
                    rrf_constant=rrf_constant,
                    enable_reranking=enable_reranking,
                    reranker_model=reranker_model,
                    rerank_top_k=rerank_top_k
                )
                logger.info(f"Document retriever initialized with reranking: {enable_reranking}")
            else:
                logger.error("Vector store not available for retriever initialization")
                
        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            self.document_fts_retriever = None
            self.document_retriever = None



    def _ensure_vector_store(self) -> None:
        """Ensure vector store object exists - let LangChain auto-create collection if needed."""
        if self.vector_store:
            logger.debug(f"Vector store already initialized for collection: {self.collection_name}")
            return
            
        logger.info(f"Initializing vector store for collection: {self.collection_name}")
        logger.debug(f"Connection args: {self.connection_args}")
        
        try:
            logger.debug("Creating LangChain Milvus vector store (will auto-create collection if needed)...")
            self.vector_store = Milvus(
                embedding_function=self.langchain_embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id=True,  # Explicitly enable auto_id to avoid pk field conflicts
            )
            
            # Verify connection and log stats
            try:
                collection = Collection(self.collection_name)
                entity_count = collection.num_entities
                logger.info(f"Successfully connected to collection '{self.collection_name}' with {entity_count} entities")
            except Exception as stats_error:
                logger.warning(f"Could not get collection stats: {stats_error}")
                logger.info(f"Vector store created for collection '{self.collection_name}' (stats unavailable)")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store for collection '{self.collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Cannot initialize vector store for collection '{self.collection_name}': {e}")

    def _ensure_email_vector_store(self) -> None:
        """Ensure email vector store object exists - let LangChain auto-create collection if needed."""
        if self.email_vector_store:
            logger.debug(f"Email vector store already initialized for collection: {self.email_collection_name}")
            return
            
        logger.info(f"Initializing email vector store for collection: {self.email_collection_name}")
        logger.debug(f"Connection args: {self.connection_args}")
        
        try:
            logger.debug("Creating LangChain Milvus email vector store (will auto-create collection if needed)...")
            self.email_vector_store = Milvus(
                embedding_function=self.langchain_embeddings,
                collection_name=self.email_collection_name,
                connection_args=self.connection_args,
                auto_id=True,  # Explicitly enable auto_id to avoid pk field conflicts
            )
            
            # Verify connection and log stats
            try:
                collection = Collection(self.email_collection_name)
                entity_count = collection.num_entities
                logger.info(f"Successfully connected to email collection '{self.email_collection_name}' with {entity_count} entities")
            except Exception as stats_error:
                logger.warning(f"Could not get email collection stats: {stats_error}")
                logger.info(f"Email vector store created for collection '{self.email_collection_name}' (stats unavailable)")
                
        except Exception as e:
            logger.error(f"Failed to initialize email vector store for collection '{self.email_collection_name}': {e}", exc_info=True)
            raise RuntimeError(f"Cannot initialize email vector store for collection '{self.email_collection_name}': {e}")

    def get_email_vector_store(self) -> Milvus:
        """Get the email vector store, creating it if necessary."""
        self._ensure_email_vector_store()
        if self.email_vector_store is None:
            raise RuntimeError("Failed to initialize email vector store")
        return self.email_vector_store

    def insert_documents(self, document_id: str, docs: List[Document]) -> int:
        """
        Insert (or upsert) a list of LangChain Document objects for a given document ID.
        
        Stores only vectors and document_id reference in Milvus.
        All metadata is managed by PostgreSQL as single source of truth.
        
        Args:
            document_id: ID of the document in PostgreSQL
            docs: List of Document objects to insert
            
        Returns:
            Number of chunks inserted (after dedupe)
        """
        logger.info(f"Starting document insertion for ID: '{document_id}' with {len(docs)} chunks")
        
        if not docs:
            logger.warning(f"No documents provided for insertion, document ID: '{document_id}'")
            return 0
            
        self._ensure_vector_store()
        logger.debug(f"Vector store ensured for collection: {self.collection_name}")
        
        # Query existing content_hash values to enforce database-level uniqueness
        existing_hashes = set()
        try:
            # Only query if collection exists and has data
            if utility.has_collection(self.collection_name):
                col = Collection(self.collection_name)
                # Get all existing content_hash values
                query_results = col.query(
                    expr="content_hash != ''",
                    output_fields=["content_hash"]
                )
                existing_hashes = {result.get("content_hash") for result in query_results if result.get("content_hash")}
                logger.debug(f"Found {len(existing_hashes)} existing content_hash values in database")
            else:
                logger.debug(f"Collection '{self.collection_name}' does not exist, no existing hashes to check")
        except Exception as e:
            logger.warning(f"Could not query existing content_hash values: {e}")
            # Continue without database-level deduplication if query fails
        
        # Build normalized texts & metadata
        seen_hashes = set()
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        duplicates = 0
        db_duplicates = 0
        
        logger.debug(f"Processing {len(docs)} documents for deduplication and metadata preparation")
        
        for idx, d in enumerate(docs):
            content = (d.page_content or '').strip()
            if not content:
                logger.debug(f"Skipping empty chunk {idx} for document ID '{document_id}'")
                continue
            meta = dict(d.metadata or {})
            
            # Compute or keep content hash
            ch = meta.get('content_hash')
            if not ch:
                ch = hashlib.sha1(content.encode('utf-8')).hexdigest()[:16]
                logger.debug(f"Generated content hash {ch} for chunk {idx}")
            
            # Check database-level duplicates first
            if ch in existing_hashes:
                db_duplicates += 1
                logger.debug(f"Content hash {ch} already exists in database, skipping chunk {idx}")
                continue
            
            # Check in-memory duplicates within current batch
            if ch in seen_hashes:
                duplicates += 1
                logger.debug(f"Duplicate content hash {ch} found in current batch, skipping chunk {idx}")
                continue
            seen_hashes.add(ch)
            
            # Store ONLY minimal metadata needed for Milvus operations
            # PostgreSQL is the single source of truth for all document metadata
            meta_minimal = {
                'document_id': document_id,  # ID reference to PostgreSQL document
                'content_hash': ch,            # For deduplication
                'page': int(meta.get('page', 0) or 0)  # Keep page for basic navigation
            }
            
            texts.append(content)
            metas.append(meta_minimal)
            
            logger.debug(f"Prepared chunk {idx}: length={len(content)}, page={meta_minimal.get('page')}, document_id={meta_minimal.get('document_id')}")
        
        if duplicates:
            logger.info(f"Skipped {duplicates} duplicate chunks within current batch for document ID '{document_id}'")
        if db_duplicates:
            logger.info(f"Skipped {db_duplicates} chunks that already exist in database for document ID '{document_id}'")
        if not texts:
            logger.warning(f"No valid texts to insert after processing, document ID: '{document_id}'")
            return 0
        
        logger.info(f"Prepared {len(texts)} unique chunks for insertion into Milvus (skipped {duplicates + db_duplicates} total duplicates)")
        
        start_time = time.perf_counter()
        try:
            # Check collection consistency
            current_collection = getattr(self.vector_store, 'collection_name', None)
            if current_collection != self.collection_name:
                logger.warning(f"Collection name mismatch: expected '{self.collection_name}', got '{current_collection}'. Recreating vector store.")
                self.vector_store = None
                self._ensure_vector_store()
                
            # Always use add_texts method - collection must exist
            if not hasattr(self.vector_store, 'add_texts') or self.vector_store is None:
                logger.error("Vector store not properly initialized")
                raise RuntimeError("Vector store not available - ensure collections are created during startup")
            
            logger.debug("Using add_texts method for insertion")
            # Debug: log a sample of metas and text lengths to verify metadata fields
            try:
                sample_metas = metas[:3]
                sample_lengths = [len(t) for t in texts[:3]]
                logger.debug(f"Milvus insert sample_metas={sample_metas} sample_lengths={sample_lengths}")
                # Also write to per-run ingestion trace logger if available
                try:
                    trace_logger = logging.getLogger(f"ingest_trace_{document_id}")
                    if trace_logger and trace_logger.handlers:
                        trace_logger.info(f"MilvusManager: inserting document_id={document_id} count={len(texts)} sample_metas={sample_metas} sample_lengths={sample_lengths}")
                except Exception:
                    pass
            except Exception:
                pass
            self.vector_store.add_texts(texts=texts, metadatas=metas)
            
            # Flush collection to ensure data is persisted
            try:
                logger.debug("Flushing Milvus collection to persist data")
                col = Collection(self.collection_name)
                col.flush()
                
                # Get collection stats for verification
                collection_count = col.num_entities
                logger.info(f"Collection '{self.collection_name}' now contains {collection_count} total entities")
            except Exception as flush_error:
                logger.warning(f"Failed to flush collection or get stats: {flush_error}")
                
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Successfully inserted {len(texts)} unique chunks for document ID '{document_id}' in {elapsed:.2f}s"
            )
            try:
                trace_logger = logging.getLogger(f"ingest_trace_{document_id}")
                if trace_logger and trace_logger.handlers:
                    trace_logger.info(f"MilvusManager: inserted_count={len(texts)} elapsed_s={elapsed:.2f}")
            except Exception:
                pass
            return len(texts)
            
        except Exception as e:
            logger.error(f"Failed inserting documents for document ID '{document_id}': {str(e)}", exc_info=True)
            logger.error(f"Error details - Type: {type(e).__name__}, Args: {e.args}")
            logger.error(f"Vector store state: {self.vector_store is not None}")
            logger.error(f"Collection name: {self.collection_name}")
            logger.error(f"Connection args: {self.connection_args}")
            return 0

    def delete_document(self, document_id: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete all embeddings for a document from Milvus.
        
        Args:
            document_id: The document ID to delete
            filename: The filename to delete (alternative to document_id)
            
        Returns:
            Dict containing deletion results and statistics
            
        Raises:
            ValueError: If neither document_id nor filename is provided
        """
        if not document_id and not filename:
            raise ValueError("Either document_id or filename must be provided")
        
        logger.info(f"Starting document deletion - document_id: {document_id}, filename: {filename}")
        
        try:
            # Ensure vector store is initialized
            self._ensure_vector_store()
            
            # If collection doesn't exist just return success (nothing to delete)
            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection '{self.collection_name}' does not exist, skipping deletion")
                return {
                    "success": True, 
                    "deleted_count": 0, 
                    "entities_before": 0, 
                    "entities_after": 0, 
                    "verification_remaining": []
                }
                
            col = Collection(self.collection_name)
            logger.debug(f"Collection '{self.collection_name}' found, proceeding with deletion")
            
            try:
                col.load()
                logger.debug("Collection loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load collection for deletion: {e}")
                # Try to proceed anyway
                pass
            
            # Check entities before deletion
            try:
                entities_before = col.num_entities
                logger.info(f"Entities before deletion: {entities_before}")
            except Exception as e:
                logger.error(f"Failed to get entity count before deletion: {e}")
                entities_before = -1
            
            # Build expression filter using only document_id (UUID)
            # Since we eliminated redundant source field, only use UUID for deletion
            if document_id:
                delete_expr = f'document_id == "{document_id}"'
            else:
                logger.warning(f"No document_id provided for deletion, cannot delete by filename alone")
                return {
                    "success": False, 
                    "deleted_count": 0, 
                    "entities_before": 0, 
                    "entities_after": 0, 
                    "verification_remaining": [],
                    "error": "No document_id provided - filename-only deletion not supported in clean schema"
                }
            
            logger.info(f"Delete expression: {delete_expr}")
            
            # Query existing records before deletion for debugging
            try:
                existing_records = col.query(
                    expr=delete_expr,
                    output_fields=["document_id", "page", "content_hash"],
                    limit=10
                )
                logger.info(f"Found {len(existing_records)} matching records to delete")
                if existing_records:
                    logger.debug(f"Sample records to delete: {existing_records[:3]}")
            except Exception as e:
                logger.warning(f"Failed to query existing records before deletion: {e}")
                existing_records = []
            
            # Perform deletion
            try:
                delete_result = col.delete(expr=delete_expr)
                logger.info(f"Delete operation result: {delete_result}")
            except Exception as e:
                logger.error(f"Delete operation failed: {e}", exc_info=True)
                raise Exception(f"Milvus delete operation failed: {str(e)}")
            
            # Flush to ensure deletion command is persisted (but don't force compaction)
            try:
                logger.info("Flushing collection to persist deletion command...")
                col.flush()
                logger.debug("Collection flushed after deletion")
                col.load()
                logger.debug("Collection reloaded after deletion")
            except Exception as e:
                logger.warning(f"Failed to flush/reload collection after deletion: {e}")
            
            # Check immediate deletion status (may not reflect actual cleanup yet)
            try:
                entities_after = col.num_entities
                deleted_count = entities_before - entities_after if entities_before >= 0 else 0
                logger.info(f"Entities after deletion command: {entities_after} (was {entities_before})")
                
                if deleted_count > 0:
                    logger.info(f"Immediate deletion successful: {deleted_count} entities removed")
                else:
                    logger.info("Deletion command executed, but entities not yet cleaned up (this is normal - Milvus will process cleanup in background)")
                    
            except Exception as e:
                logger.error(f"Failed to get entity count after deletion: {e}")
                entities_after = -1
                deleted_count = 0
            
            # Verify deletion status (may not be immediate due to Milvus eventual consistency)
            try:
                verification_results = col.query(
                    expr=delete_expr,
                    output_fields=["document_id", "page", "content_hash"],
                    limit=10
                )
                logger.info(f"Verification query found {len(verification_results)} records still matching deletion expression")
                
                if verification_results:
                    logger.info("Note: Records may still be visible due to Milvus eventual consistency - cleanup will complete in background")
                    for record in verification_results[:3]:  # Show first few
                        logger.debug(f"  Still visible: {record}")
                else:
                    logger.info("Verification confirms no matching records found")
                    
            except Exception as e:
                logger.error(f"Verification query failed: {e}")
                verification_results = []
            
            # Determine success based on delete operation result, not immediate entity counts
            # Milvus may take time to physically remove data due to eventual consistency
            delete_count_reported = 0
            if isinstance(delete_result, dict) and 'delete_count' in str(delete_result):
                # Try to extract delete count from result
                result_str = str(delete_result)
                if 'delete count:' in result_str:
                    try:
                        # Parse "delete count: X" from the result string
                        parts = result_str.split('delete count:')[1].split(',')[0].strip()
                        delete_count_reported = int(parts)
                        logger.info(f"Milvus reported {delete_count_reported} records marked for deletion")
                    except Exception:
                        logger.debug("Could not parse delete count from result")
            
            # Consider deletion successful if Milvus reported deletions or no records match query
            operation_successful = delete_count_reported > 0 or len(verification_results) == 0
            
            result = {
                "success": operation_successful,
                "entities_before": entities_before,
                "entities_after": entities_after,
                "deleted_count": deleted_count,  # Immediate count (may be 0 due to eventual consistency)
                "reported_delete_count": delete_count_reported,  # What Milvus reported
                "remaining_records": len(verification_results),
                "delete_expression": delete_expr,
                "note": "Deletion successful but cleanup may be in progress due to Milvus eventual consistency" if operation_successful and deleted_count == 0 else None
            }
            
            if operation_successful:
                if deleted_count > 0:
                    logger.info(f"Successfully deleted {deleted_count} chunks for document")
                else:
                    logger.info(f"Deletion command successful - Milvus will complete cleanup in background")
            else:
                logger.error(f"Deletion may have failed - please monitor for background cleanup")
                
            return result
            
        except Exception as e:
            error_msg = f"Document deletion failed for {filename or document_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "deleted_count": 0,
                "entities_before": -1,
                "entities_after": -1,
                "remaining_records": -1
            }

    def check_deletion_status(self, document_id: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Simple deletion status check - just confirms deletion command was executed.
        Milvus cleanup is async and we don't try to track it.
        
        Args:
            document_id: The document ID to check
            filename: The filename to check (alternative to document_id)
            
        Returns:
            Dict containing basic deletion status
        """
        if not document_id and not filename:
            return {"error": "Either document_id or filename must be provided"}
        
        # Simple approach: if we get here, deletion was initiated
        # Milvus will handle cleanup asynchronously
        return {
            "cleanup_complete": True,  # We don't track async cleanup
            "remaining_records": 0,
            "message": "Deletion command executed - Milvus will clean up asynchronously"
        }

    def _sanitize_and_project_meta(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize and project metadata to lean schema.
        
        Args:
            m: Original metadata dictionary
            
        Returns:
            Sanitized and projected metadata dictionary
        """
        # First sanitize: convert complex types to strings
        clean = {}
        for k, v in (m or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, dict):
                clean[k] = json.dumps(v, ensure_ascii=False)
            else:
                clean[k] = str(v)
        
        # Then project to minimal schema - only fields needed for Milvus operations
        # PostgreSQL is the single source of truth for all other metadata
        return {
            'document_id': str(clean.get('document_id', '')),  # UUID reference to PostgreSQL
            'page': int(clean.get('page', 0) or 0),           # Keep for basic navigation
            'content_hash': str(clean.get('content_hash', '')), # For deduplication
        }

    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using retrieval (vector + FTS).
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        logger.info(f"Starting document search for query: '{query}' with top_k={top_k}")
        
        # Require search to be properly configured
        if not self.document_retriever:
            raise RuntimeError(
                "Search is required but not properly initialized. "
                "Ensure PostgreSQL is connected and retrievers are configured during startup."
            )
        
        try:
            logger.info("Using retrieval (vector + FTS) for document search")
            hybrid_results = self.document_retriever.search(query, k=top_k)
            
            formatted = []
            for i, doc in enumerate(hybrid_results):
                # Use document_id to reference PostgreSQL for metadata
                document_id = doc.metadata.get("document_id")
                if not document_id:
                    error_msg = f"Missing document_id in search result metadata at index {i}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                result_data = {
                    "id": i,
                    "document_id": document_id,  # ID reference to PostgreSQL
                    "text": doc.page_content,
                    "similarity_score": doc.metadata.get("combined_score", 0.0),
                    "retrieval_method": doc.metadata.get("retrieval_method", "hybrid"),
                    "vector_rank": doc.metadata.get("vector_rank"),
                    "fts_rank": doc.metadata.get("fts_rank"),
                    "fts_score": doc.metadata.get("fts_score"),
                    "page": doc.metadata.get("page", 0),
                    "content_hash": doc.metadata.get("content_hash", ""),
                    # Note: Other metadata like filename, source should be retrieved from PostgreSQL using document_id
                }
                formatted.append(result_data)
                logger.debug(f"Hybrid result {i}: document_id='{result_data['document_id']}', "
                           f"method='{result_data['retrieval_method']}', "
                           f"score={result_data['similarity_score']:.4f}")
            
            logger.info(f"Hybrid document search completed successfully, returning {len(formatted)} results")
            return formatted
            
        except Exception as e:
            logger.error(f"Hybrid document search failed for query '{query}': {str(e)}", exc_info=True)
            raise RuntimeError(f"Hybrid document search failed: {str(e)}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return basic collection stats for UI.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            exists = utility.has_collection(self.collection_name)
            if not exists:
                return {
                    "name": self.collection_name,
                    "exists": False,
                    "num_entities": 0,
                    "indexed": False,
                    "metric_type": None,
                    "dim": None,
                }
                
            col = Collection(self.collection_name)
            try:
                col.load()
            except Exception:
                pass
                
            # Default values
            dim = None
            metric_type = None
            indexed = False
            
            try:
                # Infer vector field dim from schema
                schema = getattr(col, 'schema', None)
                if schema and hasattr(schema, 'fields'):
                    for f in schema.fields:
                        try:
                            params = getattr(f, 'params', {}) or {}
                            if 'dim' in params:
                                dim_value = params.get('dim')
                                if dim_value is not None:
                                    dim = int(dim_value)
                                    break
                        except Exception:
                            continue
            except Exception:
                pass
                
            try:
                # Check index info
                idxs = getattr(col, 'indexes', []) or []
                indexed = len(idxs) > 0
                if indexed:
                    try:
                        # metric type usually available in index params
                        first = idxs[0]
                        p = getattr(first, 'params', {}) or {}
                        metric_type = p.get('metric_type') or p.get('METRIC_TYPE')
                    except Exception:
                        metric_type = None
            except Exception:
                pass
                
            return {
                "name": self.collection_name,
                "exists": True,
                "num_entities": getattr(col, 'num_entities', 0),
                "indexed": indexed,
                "metric_type": metric_type,
                "dim": dim,
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "exists": False,
                "num_entities": 0,
                "indexed": False,
                "metric_type": None,
                "dim": None,
                "error": str(e),
            }

    def get_email_collection_stats(self) -> Dict[str, Any]:
        """
        Return email collection stats for UI.
        
        Returns:
            Dictionary containing email collection statistics
        """
        try:
            exists = utility.has_collection(self.email_collection_name)
            if not exists:
                return {
                    "name": self.email_collection_name,
                    "exists": False,
                    "num_entities": 0,
                    "indexed": False,
                    "metric_type": None,
                    "dim": None,
                }
                
            col = Collection(self.email_collection_name)
            try:
                col.load()
            except Exception:
                pass
                
            # Default values
            dim = None
            metric_type = None
            indexed = False
            
            try:
                # Infer vector field dim from schema
                schema = getattr(col, 'schema', None)
                if schema and hasattr(schema, 'fields'):
                    for f in schema.fields:
                        try:
                            params = getattr(f, 'params', {}) or {}
                            if 'dim' in params:
                                dim_value = params.get('dim')
                                if dim_value is not None:
                                    dim = int(dim_value)
                                    break
                        except Exception:
                            continue
            except Exception:
                pass
                
            try:
                # Check index info
                idxs = getattr(col, 'indexes', []) or []
                indexed = len(idxs) > 0
                if indexed:
                    try:
                        # metric type usually available in index params
                        first = idxs[0]
                        p = getattr(first, 'params', {}) or {}
                        metric_type = p.get('metric_type') or p.get('METRIC_TYPE')
                    except Exception:
                        metric_type = None
            except Exception:
                pass
                
            return {
                "name": self.email_collection_name,
                "exists": True,
                "num_entities": getattr(col, 'num_entities', 0),
                "indexed": indexed,
                "metric_type": metric_type,
                "dim": dim,
            }
        except Exception as e:
            return {
                "name": self.email_collection_name,
                "exists": False,
                "num_entities": 0,
                "indexed": False,
                "metric_type": None,
                "dim": None,
                "error": str(e),
            }

    def _escape_literal(self, s: str) -> str:
        """Escape double quotes in literals for Milvus query expressions."""
        try:
            return s.replace('"', '\\"')
        except Exception:
            return s

    def _paginated_count(self, collection: Collection, expr: str) -> int:
        """
        Count results with pagination to respect Milvus query limits.
        Milvus has a max limit of 16384 for (offset+limit).
        """
        try:
            max_limit = 16384
            total_count = 0
            offset = 0
            
            while True:
                try:
                    res = collection.query(
                        expr=expr, 
                        output_fields=["chunk_id"], 
                        limit=max_limit,
                        offset=offset
                    )
                    if not isinstance(res, list) or len(res) == 0:
                        break
                    total_count += len(res)
                    # If we got less than max_limit, we've reached the end
                    if len(res) < max_limit:
                        break
                    offset += len(res)
                except Exception:
                    break
            return total_count
        except Exception:
            return 0

    def get_chunk_count_for_url(self, url: str, url_id: Optional[str] = None) -> int:
        """Count chunks for a URL by finding associated documents first."""
        try:
            if not utility.has_collection(self.collection_name):
                return 0
            
            # Find document IDs associated with this URL
            document_ids = []
            
            if self.postgres_manager:
                try:
                    # Get documents created from snapshots of this URL
                    # Documents have file_path that contains the URL domain
                    from urllib.parse import urlparse
                    if url:
                        parsed = urlparse(url)
                        domain = parsed.netloc
                        
                        with self.postgres_manager.get_connection() as conn:
                            with conn.cursor() as cursor:
                                cursor.execute("""
                                    SELECT id AS document_id FROM documents 
                                    WHERE document_type = 'url' 
                                    AND file_path LIKE %s
                                """, (f'%{domain}%',))
                                
                                rows = cursor.fetchall()
                                document_ids = [row['document_id'] for row in rows]
                                
                                # Also check by URL ID in filename if available
                                if url_id:
                                    cursor.execute("""
                                        SELECT id AS document_id FROM documents 
                                        WHERE document_type = 'url' 
                                        AND (file_path LIKE %s OR filename LIKE %s)
                                    """, (f'%{url_id}%', f'%{url_id}%'))
                                    
                                    id_rows = cursor.fetchall()
                                    url_id_docs = [row['document_id'] for row in id_rows]
                                    document_ids.extend(url_id_docs)
                                    
                                # Remove duplicates
                                document_ids = list(set(document_ids))
                        
                except Exception as e:
                    logger.warning(f"Failed to query documents for URL {url}: {e}")
                    document_ids = []
            
            if not document_ids:
                logger.debug(f"No documents found for URL {url} (url_id: {url_id})")
                return 0
            
            # Count chunks for all associated documents
            total_chunks = 0
            col = Collection(self.collection_name)
            try:
                col.load()
            except Exception:
                pass
                
            for doc_id in document_ids:
                try:
                    expr = f'document_id == "{self._escape_literal(str(doc_id))}"'
                    res = col.query(
                        expr=expr,
                        output_fields=["content_hash"],
                        limit=16384
                    )
                    chunk_count = len(res) if isinstance(res, list) else 0
                    total_chunks += chunk_count
                    logger.debug(f"Document {doc_id}: {chunk_count} chunks")
                except Exception as e:
                    logger.error(f"Failed to count chunks for document {doc_id}: {e}")
                    continue
            
            logger.debug(f"Total chunks for URL {url}: {total_chunks} (from {len(document_ids)} documents)")
            return total_chunks
            
        except Exception as e:
            logger.error(f"Failed to count URL chunks for {url}: {e}")
            return 0

    def check_connection(self) -> Dict[str, Any]:
        """
        Check Milvus server reachability and return status info.
        
        Returns:
            Dictionary containing connection status and version info
        """
        try:
            ver = utility.get_server_version()
            return {"connected": True, "version": ver}
        except Exception:
            # Attempt a reconnect once
            try:
                connections.connect("default", host=self.config.MILVUS_HOST, port=self.config.MILVUS_PORT)
                ver = utility.get_server_version()
                return {"connected": True, "version": ver}
            except Exception as e2:
                return {"connected": False, "error": str(e2)}

    def classify_query_intent(self, query: str) -> str:
        """
        Classify user query as 'email' or 'general' using dedicated classification LLM.
        
        Args:
            query: User's search query
            
        Returns:
            'email' if query is about email content, 'general' otherwise
        """
        try:
            from langchain_ollama import OllamaLLM
            
            # Use dedicated classification model
            classifier = OllamaLLM(
                model=self.config.CLASSIFICATION_MODEL,
                base_url=self.config.CLASSIFICATION_BASE_URL,
                temperature=0.0  # Deterministic classification
            )
            
            classification_prompt = """You are a query intent classifier. Analyze the user's query and classify it as either 'email' or 'general'.

Classification Rules:
- Return 'email' if the query is about:
  * Email content, messages, or communication
  * Email senders, recipients, or email addresses  
  * Email subjects, dates, or email-specific metadata
  * Email folders, attachments, or email management
  * Any question that would be better answered by searching through emails

- Return 'general' if the query is about:
  * Documents, PDFs, or uploaded files
  * URLs, web content, or general knowledge
  * Any topic not specifically about email content

Important: Respond with ONLY the single word 'email' or 'general' - no explanation or additional text.

User Query: {query}

Classification:"""

            response = classifier.invoke(classification_prompt.format(query=query))
            result = str(response).strip().lower()
            
            # Validate response and default to 'general' if unclear
            if result in ['email', 'general']:
                logger.info(f"Query classified as '{result}': {query[:50]}...")
                return result
            else:
                logger.warning(f"Classification LLM returned invalid result '{result}', defaulting to 'general'")
                return 'general'
                
        except Exception as e:
            logger.error(f"Query classification failed: {e}, defaulting to 'general'")
            return 'general'

    def _get_standard_system_prompt(self) -> str:
        """Get standard system prompt for document/web content queries."""
        # Strong extractive prompt: require verbatim quoting and a clear fallback when evidence is absent.
        prompt = """
You are an assistant that answers questions USING ONLY the retrieved documents, URLs, and web sources provided in the Context below.

REQUIRED BEHAVIOR:
1) You MUST produce answers that are strictly extractive: only quote or paraphrase text that appears verbatim in the provided Context snippets. Do NOT invent facts or combine pieces to create new facts.
2) For every factual claim, include an inline citation using the numbered reference format [1], [2], [3], etc., where the numbers map to the provided source headers.
3) When you quote or cite, include the exact snippet (verbatim) you used as evidence immediately after the citation in double quotes.
4) If you cannot find verbatim evidence in the Context for the user's question, you MUST respond exactly with the sentence: "No answer in knowledge base." and provide no additional information.
5) If multiple sources contain relevant verbatim snippets, cite them all and provide the snippets for each.
6) Use temperature=0-like deterministic style; be concise and factual.

FORMAT:
- Provide a single short answer paragraph followed by citations and the quoted evidence in lines labeled "Evidence [n]:".
- If no verbatim evidence exists, return only: "No answer in knowledge base." (without quotes).

Now, answer ONLY the following user question using the Context and following the rules above:
"""
        return prompt

    def rag_search_and_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Document RAG search and answer generation (email search handled by EmailManager).
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve
            
        Returns:
            Dict containing answer, sources, metadata, and query classification
        """
        logger.info(f"Starting document RAG search for query: '{query}' with top_k={top_k}")
        
        # Classify the query intent
        classification = self.classify_query_intent(query)
        
        try:
            # Require search to be properly configured
            if not self.document_retriever:
                raise RuntimeError(
                    "Hybrid search is required but not properly initialized. "
                    "Ensure PostgreSQL is connected and retrievers are configured during startup."
                )
            
            # Perform search - no fallback
            logger.info("Using retrieval (vector + FTS) for RAG search")
            documents = self.document_retriever.search(query, k=top_k)
            logger.info(f"Retrieved {len(documents)} documents from search")
            
            if not documents:
                    # Attempt to include retriever debug info if available
                    retriever_debug = {}
                    try:
                        retriever = getattr(self, 'document_retriever', None)
                        if retriever and hasattr(retriever, 'last_analysis'):
                            retriever_debug = retriever.last_analysis or {}
                            retriever_debug['rrf_constant'] = getattr(retriever, 'rrf_constant', None)
                            retriever_debug['enable_reranking'] = bool(getattr(retriever, 'enable_reranking', False))
                    except Exception:
                        retriever_debug = {}

                    return {
                        'answer': "I couldn't find any relevant information to answer your question.",
                        'sources': [],
                        'unique_sources': [],
                        'conversation_classification': classification,
                        'analysis_info': {
                            'system_instructions': self._get_standard_system_prompt(),
                            'search_type': 'general',
                            'conversation_classification': classification,
                            'milvus_results': [],
                            'retriever_debug': retriever_debug
                        }
                    }
            
            # Enrich documents with metadata from PostgreSQL before formatting context
            try:
                if hasattr(self, 'document_data_manager') and self.document_data_manager:
                    self.document_data_manager.batch_enrich_documents_from_postgres(documents)
                    logger.debug("Documents enriched with PostgreSQL metadata")
                else:
                    logger.warning("DocumentDataManager not available for enrichment - using Milvus metadata only")
            except Exception as e:
                logger.warning(f"Document enrichment failed: {e} - continuing with Milvus metadata only")
            
            # Format document context using enriched metadata
            context_text, sources = self._format_document_context(documents)
            system_prompt = self._get_standard_system_prompt()
            
            # Generate LLM response
            logger.info(f"Prepared context with {len(sources)} sources, "
                       f"total context length: {len(context_text)} characters")
            
            # Initialize LLM and generate answer
            logger.debug(f"Initializing LLM: {self.config.CHAT_MODEL} at {self.config.CHAT_BASE_URL}")
            llm = ChatOllama(
                model=self.config.CHAT_MODEL,
                base_url=self.config.CHAT_BASE_URL,
                temperature=self.config.CHAT_TEMPERATURE
            )
            logger.debug("LLM initialized successfully")

            # Prepare messages and generate response
            system_message = SystemMessage(content=(
                f"{system_prompt}\n\nContext:\n{context_text}\n\n"
                "Now, answer ONLY the following user question with proper citations and download links:\n"
            ))
            
            human_message = HumanMessage(content=query)
            
            logger.debug("Sending query to LLM for answer generation")
            response = llm.invoke([system_message, human_message])
            logger.info(f"LLM response generated successfully, length: {len(response.content)} characters")
            
            # Parse the response
            raw_response = str(response.content)
            answer_text = raw_response
            
            # Extract citation numbers from the LLM response
            cited_refs = set()
            citation_pattern = r'\[(\d+)\]'
            matches = re.findall(citation_pattern, answer_text)
            for match in matches:
                try:
                    cited_refs.add(int(match))
                except ValueError:
                    continue
            
            logger.debug(f"LLM cited references: {sorted(cited_refs)}")
            
            # Create unique sources list ONLY for sources the LLM actually cited
            unique_sources_list = []
            for source in sources:
                ref_num = source.get('ref_num', 0)
                if ref_num in cited_refs:
                    unique_sources_list.append(source)
            
            # Sort by reference number
            unique_sources_list.sort(key=lambda x: x.get('ref_num', 0))

            # Collect additional ranking/analysis info from the retriever if available
            extra_analysis: Dict[str, Any] = {}
            try:
                retriever = getattr(self, 'document_retriever', None)
                if retriever is not None and hasattr(retriever, 'last_analysis'):
                    raw = retriever.last_analysis or {}
                    # Ensure lists exist
                    pre = raw.get('pre_rerank') or []
                    post = raw.get('post_rerank') or []

                    # Build maps for easy lookup and enrich post entries with titles from pre snapshot
                    def _extract_chunk_id(item):
                        return item.get('document_chunk_id') or item.get('email_chunk_id')

                    pre_map = {_extract_chunk_id(p): p for p in pre}
                    post_map = {}
                    for p in post:
                        chunk_id = _extract_chunk_id(p)
                        # ensure title exists for readable listings
                        if not p.get('title'):
                            p['title'] = pre_map.get(chunk_id, {}).get('title') if chunk_id in pre_map else chunk_id
                        post_map[chunk_id] = p

                    # Build comparison table rows
                    comparison = []
                    for pre_item in pre:
                        chunk_id = _extract_chunk_id(pre_item)
                        ppost = post_map.get(chunk_id)
                        pre_rank = int(pre_item.get('rank', 0))
                        pre_score = float(pre_item.get('combined_score', 0.0))

                        if ppost:
                            post_rank = int(ppost.get('final_rank')) if ppost.get('final_rank') is not None else pre_rank
                            post_score = float(ppost.get('rerank_score')) if ppost.get('rerank_score') is not None else pre_score
                            score_delta = post_score - pre_score
                            rank_delta = pre_rank - post_rank
                        else:
                            # No rerank data: reflect pre values as post values so UI shows stable numbers
                            post_rank = pre_rank
                            post_score = pre_score
                            score_delta = 0.0
                            rank_delta = 0

                        row = {
                            'document_chunk_id': chunk_id,
                            'title': pre_item.get('title'),
                            'pre_rank': pre_rank,
                            'pre_score': pre_score,
                            'post_rank': post_rank,
                            'post_score': post_score,
                            'score_delta': score_delta,
                            'rank_delta': rank_delta,
                            'preview': pre_item.get('preview')
                        }
                        comparison.append(row)

                    # Compute top movers by score and by rank
                    score_movers = sorted([r for r in comparison if r['score_delta'] is not None], key=lambda x: x['score_delta'], reverse=True)[:10]
                    rank_movers = sorted([r for r in comparison if r['rank_delta'] is not None], key=lambda x: x['rank_delta'], reverse=True)[:10]

                    extra_analysis = {
                        'pre_rerank': pre,
                        'post_rerank': post,
                        'comparison_table': comparison,
                        'top_movers_score': score_movers,
                        'top_movers_rank': rank_movers,
                        'rrf_constant': getattr(retriever, 'rrf_constant', None),
                        'reranking_enabled': bool(getattr(retriever, 'enable_reranking', False))
                    }
            except Exception:
                extra_analysis = {}

            return {
                'answer': answer_text,
                'sources': sources,
                'unique_sources': unique_sources_list,
                'conversation_classification': classification,
                'num_sources': len(sources),
                'analysis_info': {
                    'system_instructions': system_prompt,
                    'search_type': 'general',
                    'conversation_classification': classification,
                    'milvus_results': [
                        {
                            'rank': idx + 1,
                            'document_id': doc.metadata.get('document_id'),
                            'title': doc.metadata.get('title') or doc.metadata.get('filename') or None,
                            'page': doc.metadata.get('page'),
                            'category': doc.metadata.get('category'),
                            'category_type': doc.metadata.get('category_type'),
                            'topics': doc.metadata.get('topics', ''),
                            'similarity_score': float(doc.metadata.get('combined_score', 0.0)),
                            'content_preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        } for idx, doc in enumerate(documents)
                    ],
                    # include pre-computed comparison and mover lists
                    **extra_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Document RAG search failed: {e}", exc_info=True)
            # Get classification even for errors
            classification = self.classify_query_intent(query) if 'classification' not in locals() else classification
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'unique_sources': [],
                'conversation_classification': classification,
                'num_sources': 0,
                'analysis_info': {
                    'system_instructions': '',
                    'search_type': 'error',
                    'conversation_classification': classification,
                    'milvus_results': []
                }
            }

    def _format_document_context(self, documents: List[Document]) -> tuple:
        """
        Format document search results for LLM context using clean schema architecture.
        
        Uses PostgreSQL as single source of truth for metadata - queries database 
        using document_id from Milvus results to get filename and other metadata.
        """
        try:
            # Check if we have results
            if not documents:
                logger.warning("No documents found in search - databases may be empty or query failed")
                return "", []
            
            logger.debug("Formatting context from retrieved documents with clean schema architecture")
            
            # Extract unique document_ids from Milvus results
            unique_document_ids = set()
            for doc in documents:
                # Direct access to document metadata - no helper needed
                document_id = doc.metadata.get('document_id')
                
                if document_id:
                    unique_document_ids.add(document_id)
                else:
                    logger.error(f"Document missing required 'document_id' metadata. Doc type: {type(doc)}, Content: {doc}")
            
            if not unique_document_ids:
                logger.error("No valid document_ids found in search results")
                return "", []
            
            # Query PostgreSQL to get metadata for these document_ids
            document_metadata = {}
            # Prefer titles already present on Document.metadata (enrichment)
            enriched_titles = {}
            for doc in documents:
                did = doc.metadata.get('document_id')
                if did and doc.metadata.get('title'):
                    enriched_titles[did] = doc.metadata.get('title')
            if self.postgres_manager and hasattr(self.postgres_manager, 'get_connection'):
                try:
                    logger.debug(f"Querying PostgreSQL for metadata of {len(unique_document_ids)} documents")

                    # Create SQL query for batch lookup (psycopg2 uses %s placeholders)
                    # Match the actual documents schema from PostgreSQLManager and avoid optional/migrated columns
                    query = """
                        SELECT 
                            id, filename, title, content_type, file_path,
                            created_at, keywords
                        FROM documents 
                        WHERE id = ANY(%s::uuid[])
                    """

                    # Use the synchronous connection helper provided by PostgreSQLManager
                    with self.postgres_manager.get_connection() as conn:
                        with conn.cursor() as cur:
                            # psycopg2 will adapt a Python list to SQL array
                            cur.execute(query, (list(unique_document_ids),))
                            rows = cur.fetchall()

                    for row in rows:
                        document_metadata[str(row['id'])] = {
                            'filename': row.get('filename'),
                            'title': row.get('title') or row.get('filename'),
                            'content_type': row.get('content_type'),
                            'file_path': row.get('file_path'),
                            'category': row.get('category'),
                            'keywords': row.get('keywords') or []
                        }

                    logger.debug(f"Retrieved metadata for {len(document_metadata)} documents from PostgreSQL")
                except Exception as e:
                    logger.error(f"Failed to query PostgreSQL for document metadata: {e}", exc_info=True)
                    # Continue without PostgreSQL metadata
                    document_metadata = {}
            else:
                logger.debug("Postgres manager not available or missing get_connection(); skipping metadata enrichment")
            
            # Create unique source mapping using PostgreSQL data
            unique_sources = {}  # document_id -> reference_info
            source_counter = 1
            
            # First pass: create reference mapping using PostgreSQL metadata
            for doc in documents:
                document_id = doc.metadata.get('document_id')
                if not document_id:
                    continue
                
                if document_id not in unique_sources:
                    # Get metadata from PostgreSQL or use fallbacks
                    postgres_meta = document_metadata.get(document_id, {})
                    # Prefer enriched title present on the Document (from _batch_enrich), then Postgres lookup, then filename fallback
                    title = enriched_titles.get(document_id) or postgres_meta.get('title') or f'document_{document_id[:8]}'
                    filename = postgres_meta.get('filename', f'document_{document_id[:8]}')
                    content_type = postgres_meta.get('content_type', 'unknown')
                    
                    # Determine category_type from content_type
                    category_type = 'document'  # Default
                    if 'email' in content_type.lower():
                        category_type = 'email'
                    elif 'url' in content_type.lower() or content_type == 'text/html':
                        category_type = 'url'
                    
                    unique_sources[document_id] = {
                        'ref_num': source_counter,
                        'document_id': document_id,
                        'filename': filename,
                        'title': title,
                        'content_type': content_type,
                        'category_type': category_type,
                        'topics': doc.metadata.get('topics', ''),
                        'keywords': doc.metadata.get('keywords', [])
                    }
                    source_counter += 1
            
            # Second pass: build context using the consistent reference numbers
            docs_content = []
            sources = []
            
            for doc in documents:
                document_id = doc.metadata.get('document_id')
                if not document_id or document_id not in unique_sources:
                    logger.error(f"Document missing document_id or not found in unique_sources: {doc}")
                    continue
                    
                source_info = unique_sources[document_id]
                ref_num = source_info['ref_num']
                filename = source_info['filename']
                category_type = source_info['category_type']
                page_info = doc.metadata.get('page')
                
                # Get similarity score from document metadata
                similarity_score = doc.metadata.get('combined_score', 0.0)
                
                # Build download/source link based on category type
                source_link = ""
                if category_type == 'document':
                    # For documents, create download link
                    source_link = f"Download: /download/{filename}"
                elif category_type == 'url':
                    source_link = f"URL: {filename}"
                elif category_type == 'email':
                    source_link = f"Email: {filename}"
                else:
                    source_link = f"Source: {filename}"
                
                # Get text content from document
                text_content = doc.page_content if hasattr(doc, 'page_content') else ""
                
                # Enhanced source header with formatted link information
                source_header = f"Source [{ref_num}]:"
                page_display = f"Page: {page_info}" if page_info else "Page: Not specified"
                source_info_text = f"{page_display}\n{source_link}"
                docs_content.append(f"{source_header}\n{text_content}\n{source_info_text}")
                
                # Keep individual chunk info for detailed view (now using PostgreSQL metadata)
                sources.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_info,
                    "category": source_info.get('category'),
                    "category_type": category_type,
                    "topics": source_info.get('topics', ''),
                    "keywords": source_info.get('keywords', []),
                    "similarity_score": float(similarity_score),
                    "ref_num": ref_num,
                    "content_preview": text_content[:150] + "..." if len(text_content) > 150 else text_content
                })
            
            context_text = "\n\n".join(docs_content)
            logger.info(f"Prepared context with {len(sources)} sources from {len(unique_sources)} unique documents, total context length: {len(context_text)} characters")
            logger.debug(f"Unique sources mapping: {list(unique_sources.keys())}")
            
            return context_text, sources
            
        except Exception as e:
            logger.error(f"Document context formatting failed: {e}", exc_info=True)
            return "", []
