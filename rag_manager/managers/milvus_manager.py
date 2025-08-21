"""
Milvus vector database manager for RAG Knowledgebase Manager.

This module manages Milvus database operations using LangChain's Milvus VectorStore,
integrated with PostgreSQL for metadata management.
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any

from pymilvus import connections, utility, Collection
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Milvus as LC_Milvus
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.config import Config
from ingestion.core.postgres_manager import PostgreSQLConfig
from ingestion.core.database_manager import RAGDatabaseManager

# Configure logging
logger = logging.getLogger(__name__)


class MilvusManager:
    """
    Manages Milvus database operations using LangChain's Milvus VectorStore.
    Integrated with PostgreSQL for metadata management.
    
    This class follows the development rules with proper type hints,
    error handling, and comprehensive logging.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize Milvus manager with PostgreSQL integration.
        
        Args:
            config: Application configuration instance
        """
        self.config = config
        self.collection_name = config.COLLECTION_NAME
        self.connection_args = {"host": config.MILVUS_HOST, "port": config.MILVUS_PORT}
        
        # Initialize PostgreSQL configuration
        self.postgres_config = PostgreSQLConfig(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        
        # Initialize database manager
        try:
            self.db_manager = RAGDatabaseManager(postgres_config=self.postgres_config)
            logger.info("PostgreSQL integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            self.db_manager = None
        
        # Establish Milvus connection (idempotent)
        try:
            connections.connect(alias="default", **self.connection_args)
        except Exception:
            pass
            
        # Lazy-created vector stores for documents and emails
        self.vector_store: Optional[LC_Milvus] = None
        self.email_vector_store: Optional[LC_Milvus] = None
        self.email_collection_name = f"{self.collection_name}_emails"
        
        try:
            # Create a dedicated embeddings instance
            self.langchain_embeddings = OllamaEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                base_url=f"http://{self.config.OLLAMA_EMBEDDING_HOST}:{self.config.OLLAMA_EMBEDDING_PORT}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings for MilvusManager: {e}")
            raise RuntimeError(f"Could not initialize embeddings: {e}")

    def _ensure_vector_store(self) -> None:
        """Ensure vector store object exists (create collection if missing)."""
        if self.vector_store:
            logger.debug(f"Vector store already initialized for collection: {self.collection_name}")
            return
            
        logger.info(f"Initializing vector store for collection: {self.collection_name}")
        logger.debug(f"Connection args: {self.connection_args}")
        
        try:
            logger.debug("Attempting to connect to existing collection...")
            self.vector_store = LC_Milvus(
                embedding_function=self.langchain_embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
            )
            
            # Verify connection
            try:
                collection = Collection(self.collection_name)
                entity_count = collection.num_entities
                logger.info(f"Connected to existing collection '{self.collection_name}' with {entity_count} entities")
            except Exception as e:
                logger.warning(f"Could not get collection stats: {e}")
                
        except Exception as e:
            logger.warning(f"Could not connect to existing collection: {e}")
            # Fallback: try from_texts with empty set to force creation
            try:
                logger.debug("Creating new collection with initialization document...")
                self.vector_store = LC_Milvus.from_texts(
                    texts=["__init__"],
                    embedding=self.langchain_embeddings,
                    metadatas=[{"source": "__init__", "document_id": "__init__", "chunk_id": "init", "page": 0}],
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as e2:
                logger.error(f"Failed to initialize Milvus vector store: Primary error: {e}, Fallback error: {e2}", exc_info=True)
                raise Exception(f"Cannot initialize vector store. Primary: {e}, Fallback: {e2}")

    def _ensure_email_vector_store(self) -> None:
        """Ensure email vector store object exists (create collection if missing)."""
        if self.email_vector_store:
            return
        try:
            self.email_vector_store = LC_Milvus(
                embedding_function=self.langchain_embeddings,
                collection_name=self.email_collection_name,
                connection_args=self.connection_args,
            )
        except Exception as e:
            # Fallback: try from_texts with empty set to force creation
            try:
                # Email-specific metadata schema
                email_metadata = {
                    "message_id": "__init__", 
                    "source": "email:__init__", 
                    "subject": "__init__",
                    "sender": "__init__",
                    "date_sent": "1970-01-01T00:00:00Z",
                    "chunk_id": "init", 
                    "page": 0,
                    "content_hash": "__init__"
                }
                self.email_vector_store = LC_Milvus.from_texts(
                    texts=["__init__"],
                    embedding=self.langchain_embeddings,
                    metadatas=[email_metadata],
                    collection_name=self.email_collection_name,
                    connection_args=self.connection_args,
                )
                logger.info(f"Created email vector store collection: {self.email_collection_name}")
            except Exception as e2:
                logger.error(f"Failed to initialize email vector store: {e}; {e2}")
                raise

    def get_email_vector_store(self) -> LC_Milvus:
        """Get the email vector store, creating it if necessary."""
        self._ensure_email_vector_store()
        if self.email_vector_store is None:
            raise RuntimeError("Failed to initialize email vector store")
        return self.email_vector_store

    def insert_documents(self, source: str, docs: List[Document]) -> int:
        """
        Insert (or upsert) a list of LangChain Document objects for a given source/URL.
        
        Ensures deterministic chunk_id & document_id fields and performs simple dedupe
        based on content_hash.
        
        Args:
            source: Source identifier (filename or URL)
            docs: List of Document objects to insert
            
        Returns:
            Number of chunks inserted (after dedupe)
        """
        logger.info(f"Starting document insertion for source: '{source}' with {len(docs)} chunks")
        
        if not docs:
            logger.warning(f"No documents provided for insertion, source: '{source}'")
            return 0
            
        self._ensure_vector_store()
        logger.debug(f"Vector store ensured for collection: {self.collection_name}")
        
        # Build normalized texts & metadata
        seen_hashes = set()
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        duplicates = 0
        
        logger.debug(f"Processing {len(docs)} documents for deduplication and metadata preparation")
        
        for idx, d in enumerate(docs):
            content = (d.page_content or '').strip()
            if not content:
                logger.debug(f"Skipping empty chunk {idx} for source '{source}'")
                continue
            meta = dict(d.metadata or {})
            
            # Compute or keep content hash
            ch = meta.get('content_hash')
            if not ch:
                ch = hashlib.sha1(content.encode('utf-8')).hexdigest()[:16]
                logger.debug(f"Generated content hash {ch} for chunk {idx}")
            
            if ch in seen_hashes:
                duplicates += 1
                logger.debug(f"Duplicate content hash {ch} found, skipping chunk {idx}")
                continue
            seen_hashes.add(ch)
            
            # Populate required projection fields
            meta.setdefault('document_id', source)
            meta.setdefault('source', source)
            meta.setdefault('page', int(meta.get('page', 0) or 0))
            meta.setdefault('chunk_id', f"{source}-{idx}")
            meta.setdefault('topic', meta.get('topic', ''))
            meta.setdefault('category', meta.get('category', ''))
            meta['content_hash'] = ch
            meta['content_length'] = len(content)
            
            processed_meta = self._sanitize_and_project_meta(meta)
            metas.append(processed_meta)
            texts.append(content)
            
            logger.debug(f"Prepared chunk {idx}: length={len(content)}, page={meta.get('page')}, chunk_id={meta.get('chunk_id')}")
        
        if duplicates:
            logger.info(f"Skipped {duplicates} duplicate chunks for source '{source}'")
        if not texts:
            logger.warning(f"No valid texts to insert after processing, source: '{source}'")
            return 0
        
        logger.info(f"Prepared {len(texts)} unique chunks for insertion into Milvus")
        
        start_time = time.perf_counter()
        try:
            # Check collection consistency
            current_collection = getattr(self.vector_store, 'collection_name', None)
            if current_collection != self.collection_name:
                logger.warning(f"Collection name mismatch: expected '{self.collection_name}', got '{current_collection}'. Recreating vector store.")
                self.vector_store = None
                self._ensure_vector_store()
                
            # Use add_texts if store already instantiated, else from_texts
            existing_count = len(getattr(self.vector_store, 'texts', []))
            logger.debug(f"Vector store has {existing_count} existing texts")
            
            if (hasattr(self.vector_store, 'add_texts') and 
                self.vector_store is not None and 
                existing_count > 0):
                logger.debug("Using add_texts method for insertion")
                self.vector_store.add_texts(texts=texts, metadatas=metas)
            else:
                logger.debug("Using from_texts method for insertion (recreating vector store)")
                self.vector_store = LC_Milvus.from_texts(
                    texts=texts,
                    embedding=self.langchain_embeddings,
                    metadatas=metas,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
                logger.info(f"Created new vector store with {len(texts)} documents")
            
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
                f"Successfully inserted {len(texts)} unique chunks for source '{source}' in {elapsed:.2f}s"
            )
            return len(texts)
            
        except Exception as e:
            logger.error(f"Failed inserting documents for source '{source}': {str(e)}", exc_info=True)
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
        
        try:
            # If collection doesn't exist just return success (nothing to delete)
            if not utility.has_collection(self.collection_name):
                return {
                    "success": True, 
                    "deleted_count": 0, 
                    "entities_before": 0, 
                    "entities_after": 0, 
                    "verification_remaining": []
                }
                
            col = Collection(self.collection_name)
            try:
                col.load()
            except Exception:
                pass
            
            # Check entities before deletion
            entities_before = col.num_entities
            logger.info(f"Entities before deletion: {entities_before}")
            
            # Build expression filter
            if document_id and filename:
                delete_expr = f'document_id == "{document_id}" or source == "{filename}"'
            elif document_id:
                delete_expr = f'document_id == "{document_id}"'
            else:
                delete_expr = f'source == "{filename}"'
            
            logger.info(f"Delete expression: {delete_expr}")
            
            # Perform deletion
            delete_result = col.delete(expr=delete_expr)
            logger.info(f"Delete operation result: {delete_result}")
            
            # Flush to ensure deletion is persisted
            col.flush()
            col.load()
            
            # Check entities after deletion
            entities_after = col.num_entities
            deleted_count = entities_before - entities_after
            
            # Verify no records remain
            verification_results = col.query(
                expr=delete_expr,
                output_fields=["document_id", "source", "chunk_id"],
                limit=10
            )
            
            success = len(verification_results) == 0
            
            result = {
                "success": success,
                "entities_before": entities_before,
                "entities_after": entities_after,
                "deleted_count": deleted_count,
                "remaining_records": len(verification_results),
                "delete_expression": delete_expr
            }
            
            if success:
                logger.info(f"Successfully deleted {deleted_count} chunks for document")
            else:
                logger.warning(f"Deletion incomplete: {len(verification_results)} records still remain")
                
            return result
            
        except Exception as e:
            error_msg = f"Document deletion failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "deleted_count": 0
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
        
        # Then project to fixed lean schema
        return {
            'document_id': str(clean.get('document_id', '')),
            'source': str(clean.get('source', '')),
            'page': int(clean.get('page', 0) or 0),
            'chunk_id': str(clean.get('chunk_id', '')),
            'topic': str(clean.get('topic', '')),
            'category': str(clean.get('category', '')),
            'content_hash': str(clean.get('content_hash', '')),
            'content_length': int(clean.get('content_length', 0) or 0),
        }

    def search_documents(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents using vector similarity.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        logger.info(f"Starting document search for query: '{query}' with top_k={top_k}")
        
        if not self.vector_store:
            logger.debug("Vector store not initialized, creating new connection...")
            try:
                self.vector_store = LC_Milvus(
                    embedding_function=self.langchain_embeddings,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
                logger.debug(f"Vector store initialized for collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to initialize vector store for search: {e}", exc_info=True)
                return []
        
        try:
            # Check collection stats before search
            try:
                collection = Collection(self.collection_name)
                entity_count = collection.num_entities
                logger.debug(f"Collection '{self.collection_name}' has {entity_count} entities")
            except Exception as e:
                logger.warning(f"Could not get collection stats before search: {e}")
            
            logger.debug(f"Performing similarity search with query: '{query}'")
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            logger.info(f"Vector search returned {len(results)} results")
            
            formatted = []
            for i, (doc, score) in enumerate(results):
                result_data = {
                    "id": i,
                    "filename": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "text": doc.page_content,
                    "score": float(score),
                    "source": doc.metadata.get("source", "unknown"),
                }
                formatted.append(result_data)
                logger.debug(f"Result {i}: source='{result_data['source']}', score={score:.4f}, text_length={len(doc.page_content)}")
            
            logger.info(f"Document search completed successfully, returning {len(formatted)} formatted results")
            return formatted
            
        except Exception as e:
            logger.error(f"Document search failed for query '{query}': {str(e)}", exc_info=True)
            logger.error(f"Vector store state: {self.vector_store is not None}")
            logger.error(f"Collection name: {self.collection_name}")
            return []

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

    def rag_search_and_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform RAG search and generate conversational answer.
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        logger.info(f"Starting RAG search for query: '{query}' with top_k={top_k}")
        
        try:
            # Step 1: Initialize vector store if needed
            logger.debug("Initializing vector store connection")
            if not self.vector_store:
                logger.info("Vector store not initialized, creating new connection")
                self.vector_store = LC_Milvus(
                    embedding_function=self.langchain_embeddings,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
                logger.info(f"Vector store initialized with collection: {self.collection_name}")
            
            # Step 2: Retrieve relevant documents
            logger.debug(f"Performing similarity search with query: '{query}'")
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            logger.info(f"Retrieved {len(results)} documents from vector search")
            
            # Log detailed results for debugging
            for i, (doc, score) in enumerate(results):
                logger.debug(f"Result {i+1}: score={score:.4f}, source='{doc.metadata.get('source', 'unknown')}', "
                           f"chunk_id='{doc.metadata.get('chunk_id', 'unknown')}', "
                           f"content_length={len(doc.page_content)}")

            # Simple keyword-boost rerank: if query words intersect chunk keywords, slightly improve score
            try:
                logger.debug("Applying keyword-based reranking")
                import re
                q_tokens = set(re.findall(r"[A-Za-z0-9_-]+", query.lower()))
                logger.debug(f"Query tokens for reranking: {q_tokens}")
                
                adjusted = []
                for doc, score in results:
                    kws = doc.metadata.get('keywords') or []
                    if isinstance(kws, str):
                        # if stored as JSON string somewhere
                        try:
                            kws = json.loads(kws)
                        except Exception:
                            kws = []
                    overlap = len(q_tokens.intersection({str(k).lower() for k in kws})) if kws else 0
                    # Assuming lower score is better (distance); subtract small bonus per overlap
                    adj_score = score - (0.05 * min(overlap, 5))
                    adjusted.append((doc, score, adj_score))
                    
                # Sort by adjusted score
                adjusted.sort(key=lambda t: t[2])
                # Trim to top_k again just in case
                results = [(d, s) for d, s, _ in adjusted[:top_k]]
                logger.debug(f"Reranking complete, using {len(results)} results")
                
            except Exception as rerank_err:
                logger.warning(f"Keyword rerank failed, continuing with original results: {rerank_err}")
            
            # Step 3: Check if we have results
            if not results:
                logger.warning("No documents found in vector search - vector database may be empty")
                return {
                    "answer": "I don't have any relevant information to answer your question. Please make sure documents are uploaded and processed.",
                    "sources": [],
                    "context_used": False,
                    "error": "No documents found in vector database"
                }
            
            # Step 4: Format context from retrieved documents
            logger.debug("Formatting context from retrieved documents")
            docs_content = []
            sources = []
            
            for doc, score in results:
                source_name = doc.metadata.get('source', 'unknown')
                docs_content.append(f"Source: {source_name}\n{doc.page_content}")
                sources.append({
                    "filename": source_name,
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "score": float(score),
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            context_text = "\n\n".join(docs_content)
            logger.info(f"Prepared context with {len(sources)} sources, total context length: {len(context_text)} characters")
            
            # Step 5: Initialize LLM and generate answer
            logger.debug(f"Initializing LLM: {self.config.CHAT_MODEL} at {self.config.CHAT_BASE_URL}")
            llm = ChatOllama(
                model=self.config.CHAT_MODEL,
                base_url=self.config.CHAT_BASE_URL,
                temperature=self.config.CHAT_TEMPERATURE
            )
            logger.debug("LLM initialized successfully")
            
            # Step 6: Prepare messages and generate response
            system_message = SystemMessage(content=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer based on the context, say that you "
                "don't know. Use clear and concise language. "
                "Cite specific sources when possible.\n\n"
                f"Context:\n{context_text}"
            ))
            
            human_message = HumanMessage(content=query)
            
            logger.debug("Sending query to LLM for answer generation")
            response = llm.invoke([system_message, human_message])
            logger.info(f"LLM response generated successfully, length: {len(response.content)} characters")
            
            result = {
                "answer": response.content,
                "sources": sources,
                "context_used": True,
                "num_sources": len(sources)
            }
            
            logger.info(f"RAG search completed successfully for query: '{query}'")
            return result
            
        except Exception as e:
            logger.error(f"RAG search and answer failed for query '{query}': {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full error details: {repr(e)}")
            
            # Return detailed error information for debugging
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_used": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "debug_info": {
                    "collection_name": self.collection_name,
                    "vector_store_initialized": self.vector_store is not None,
                    "embeddings_model": self.config.EMBEDDING_MODEL,
                    "chat_model": self.config.CHAT_MODEL,
                    "chat_base_url": self.config.CHAT_BASE_URL
                }
            }
