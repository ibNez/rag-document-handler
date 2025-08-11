#!/usr/bin/env python3
"""
RAG Document Handler - Simplified Single-Interface Application.

A comprehensive document management system for storing and retrieving document 
embeddings in a Milvus vector database for use with RAG applications.

Features:
- Document upload & management (PDF, DOCX, DOC, TXT, MD)
- Vector embeddings using Ollama/SentenceTransformers
- Milvus integration for vector storage
- Semantic search with natural language queries
- Web interface with Bootstrap UI
- Threading for responsive UI during long operations
"""

import os
import sys
import logging
import threading
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import json
import hashlib

# Flask and web components
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

# Document processing
from pypdf import PdfReader
from docx import Document as DocxDocument
import chardet
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector and ML components
from pymilvus import connections, utility, Collection
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Milvus as LC_Milvus
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# Configuration
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging following development rules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_document_handler.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Application configuration class following development rules.
    
    Centralizes all configuration settings with proper type hints
    and default values from environment variables.
    """
    
    # Milvus Database Configuration
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "384"))
    
    # Flask Configuration
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "3000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    
    # File Upload Configuration
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", "104857600"))  # 100MB
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "staging")
    UPLOADED_FOLDER: str = os.getenv("UPLOADED_FOLDER", "uploaded")
    ALLOWED_EXTENSIONS: set = field(default_factory=lambda: {"txt", "pdf", "docx", "doc", "md"})
    
    # Embedding Model Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    
    # Ollama Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))
    
    # Chat Model Configuration
    CHAT_MODEL: str = os.getenv('CHAT_MODEL', 'mistral:latest')
    CHAT_BASE_URL: str = os.getenv('CHAT_BASE_URL', f"http://{os.getenv('OLLAMA_HOST','localhost')}:{os.getenv('OLLAMA_PORT','11434')}")
    CHAT_TEMPERATURE: float = float(os.getenv('CHAT_TEMPERATURE', '0.1'))
    
    # Unstructured chunking
    UNSTRUCTURED_CHUNKING_STRATEGY: str = os.getenv('UNSTRUCTURED_CHUNKING_STRATEGY', 'basic')
    UNSTRUCTURED_MAX_CHARACTERS: int = int(os.getenv('UNSTRUCTURED_MAX_CHARACTERS', '1000'))
    UNSTRUCTURED_OVERLAP: int = int(os.getenv('UNSTRUCTURED_OVERLAP', '200'))
    UNSTRUCTURED_INCLUDE_ORIG: bool = os.getenv('UNSTRUCTURED_INCLUDE_ORIG', 'false').lower() == 'true'
    
    # Milvus flags
    MILVUS_DROP_COLLECTION: bool = os.getenv('MILVUS_DROP_COLLECTION', 'false').lower() == 'true'


@dataclass
class ProcessingStatus:
    """Status tracking for document processing operations."""
    filename: str
    status: str = "pending"  # pending, processing, chunking, embedding, storing, completed, error
    progress: int = 0  # 0-100
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunks_count: int = 0
    error_details: Optional[str] = None


class DocumentProcessor:
    """
    Handles document processing operations including text extraction,
    chunking, and embedding generation using UnstructuredLoader.
    """
    
    def __init__(self, config: Config):
        """
        Initialize document processor.
        
        Args:
            config: Application configuration instance
        """
        self.config = config
        self.embedding_provider = OllamaEmbeddings(
            model=self.config.EMBEDDING_MODEL, 
            base_url=f"http://{self.config.OLLAMA_HOST}:{self.config.OLLAMA_PORT}"
        )
        logger.info(f"DocumentProcessor initialized with {self.config.EMBEDDING_MODEL}")
    
    def load_and_chunk(self, file_path: str, filename: str, document_id: str) -> List[Document]:
        """
        Load document and chunk using UnstructuredLoader with lean metadata like the notebook.
        
        Args:
            file_path: Path to the file to process
            filename: Name of the file
            document_id: Unique document identifier
            
        Returns:
            List of Document chunks with metadata
        """
        logger.info(f"Loading and chunking document: {filename}")
        
        try:
            # Use UnstructuredLoader for all document types
            loader = UnstructuredLoader(
                file_path,
                chunking_strategy=self.config.UNSTRUCTURED_CHUNKING_STRATEGY,
                max_characters=self.config.UNSTRUCTURED_MAX_CHARACTERS,
                overlap=self.config.UNSTRUCTURED_OVERLAP,
                include_orig_elements=self.config.UNSTRUCTURED_INCLUDE_ORIG,
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} elements via UnstructuredLoader")
            
            # Process chunks with lean metadata like the notebook
            chunks: List[Document] = []
            for i, d in enumerate(documents):
                text = d.page_content or ''
                meta = d.metadata or {}
                page = meta.get('page') or meta.get('page_number') or (i + 1)
                
                # Deterministic content hash
                content_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]
                
                # Update metadata with lean schema
                meta.update({
                    'source': filename,
                    'page': page,
                    'document_id': document_id,
                    'chunk_id': f"{document_id}:{content_hash}",
                    'content_hash': content_hash,
                    'content_length': len(text),
                })
                d.metadata = meta
                chunks.append(d)
            
            logger.info(f"Created {len(chunks)} chunks with metadata")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load and chunk document {filename}: {str(e)}")
            raise

    def enrich_topics(self, chunks: List[Document]) -> None:
        """
        LLM topic classification exactly like the notebook.
        
        Args:
            chunks: List of document chunks to enrich
        """
        logger.info("Enriching chunk metadata with LLM classification (topic)...")
        
        llm = ChatOllama(
            model=self.config.CHAT_MODEL, 
            base_url=self.config.CHAT_BASE_URL, 
            temperature=self.config.CHAT_TEMPERATURE
        )
        
        prompt_tpl = (
            "You are classifying a text chunk for RAG metadata.\n"
            "Return ONLY compact JSON with keys: topic.\n"
            "topic: concise subject title (3-6 words).\n"
            "Text:\n{chunk}\n"
        )
        
        for c in chunks:
            snippet = (c.page_content or '')[:800]
            try:
                resp = llm.invoke(prompt_tpl.format(chunk=snippet)).content.strip()
                start = resp.find('{')
                end = resp.rfind('}') + 1
                if start != -1 and end > start:
                    obj = json.loads(resp[start:end])
                    if isinstance(obj, dict) and obj.get('topic'):
                        c.metadata['topic'] = obj['topic']
            except Exception:
                c.metadata.setdefault('topic', 'unknown')
        
        logger.info("LLM enrichment complete")


class MilvusManager:
    """
    Manages Milvus database operations using LangChain's Milvus VectorStore.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Milvus manager with LangChain integration.
        
        Args:
            config: Application configuration instance
        """
        self.config = config
        self.embedding_provider = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL, base_url=f"http://{self.config.OLLAMA_HOST}:{self.config.OLLAMA_PORT}")
        self.vector_store = None
        self.connected = False
        self._connect()
        logger.info("MilvusManager (LangChain) initialized")
    
    def _connect(self) -> None:
        """Establish connection to Milvus database using LangChain."""
        self.collection_name = self.config.COLLECTION_NAME
        self.connection_args = {"host": self.config.MILVUS_HOST, "port": self.config.MILVUS_PORT}
        self.langchain_embeddings = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL, base_url=f"http://{self.config.OLLAMA_HOST}:{self.config.OLLAMA_PORT}")
        connections.connect("default", host=self.config.MILVUS_HOST, port=self.config.MILVUS_PORT)
        if self.config.MILVUS_DROP_COLLECTION and utility.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection '{self.collection_name}' (flag enabled)")
            utility.drop_collection(self.collection_name)
        self.vector_store = None
        logger.info("Milvus connection ready")

    def insert_documents(self, filename: str, chunks: List[Document]) -> None:
        """
        Insert documents using same logic as notebook: dedupe, sanitize, project metadata.
        
        Args:
            filename: Name of the source file
            chunks: List of document chunks to insert
        """
        logger.info(f"Inserting {len(chunks)} chunks for {filename}")
        
        # Deduplicate by content_hash
        seen = set()
        unique = []
        for c in chunks:
            ch = (c.metadata or {}).get('content_hash')
            if not ch or ch in seen:
                continue
            seen.add(ch)
            unique.append(c)
        
        logger.info(f"Unique chunks after dedupe: {len(unique)} (from {len(chunks)})")
        
        if not unique:
            raise RuntimeError("No unique chunks to insert")
        
        # Prepare texts and metadata
        texts = [d.page_content for d in unique]
        metas = [self._sanitize_and_project_meta(d.metadata) for d in unique]
        
        # Index parameters matching notebook
        index_params = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
        
        try:
            # Create collection using from_texts (matches notebook exactly)
            self.vector_store = LC_Milvus.from_texts(
                texts=texts,
                embedding=self.langchain_embeddings,
                metadatas=metas,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                index_params=index_params,
            )
            
            # Ensure collection is flushed and loaded
            col = Collection(self.collection_name)
            try:
                col.flush()
            except Exception:
                pass
            col.load()
            
            logger.info(f"Stored {col.num_entities} chunks in Milvus collection '{self.collection_name}'")
            
            # If zero entities, retry with add_texts like the notebook
            if col.num_entities == 0 and texts:
                logger.info("Insertion resulted in 0 entities. Retrying with add_texts()...")
                try:
                    self.vector_store.add_texts(texts=texts, metadatas=metas)
                    try:
                        col.flush()
                    except Exception:
                        pass
                    col.load()
                    logger.info(f"After retry, stored {col.num_entities} chunks in Milvus collection '{self.collection_name}'")
                except Exception as e:
                    logger.error(f"Retry with add_texts failed: {repr(e)}")
                    raise
            
        except Exception as e:
            logger.error(f"Failed to insert documents for {filename}: {str(e)}")
            raise

    def delete_document(self, document_id: str = None, filename: str = None) -> Dict[str, Any]:
        """
        Delete all embeddings for a document from Milvus.
        
        Args:
            document_id: The document ID to delete
            filename: The filename to delete (alternative to document_id)
            
        Returns:
            Dict containing deletion results and statistics
        """
        if not document_id and not filename:
            raise ValueError("Either document_id or filename must be provided")
        
        try:
            # Get the collection
            col = Collection(self.collection_name)
            col.load()
            
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
        Sanitize and project metadata to lean schema exactly like notebook.
        
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
        
        # Then project to fixed lean schema (matches notebook exactly)
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
        if not self.vector_store:
            self.vector_store = LC_Milvus(
                embedding_function=self.langchain_embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
            )
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        formatted = []
        for i, (doc, score) in enumerate(results):
            formatted.append({
                "id": i,
                "filename": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "text": doc.page_content,
                "score": float(score),
                "source": doc.metadata.get("source", "unknown"),
            })
        return formatted

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic collection stats for UI."""
        try:
            exists = utility.has_collection(self.collection_name)
            if not exists:
                return {"name": self.collection_name, "exists": False, "entities": 0}
            col = Collection(self.collection_name)
            try:
                col.load()
            except Exception:
                pass
            return {
                "name": self.collection_name,
                "exists": True,
                "entities": getattr(col, 'num_entities', 0),
            }
        except Exception as e:
            return {"name": self.collection_name, "exists": False, "entities": 0, "error": str(e)}

    def rag_search_and_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Perform RAG search and generate conversational answer.
        
        Args:
            query: User's question/query
            top_k: Number of documents to retrieve
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        try:
            # Step 1: Retrieve relevant documents
            if not self.vector_store:
                self.vector_store = LC_Milvus(
                    embedding_function=self.langchain_embeddings,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
            
            # Get similar documents
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            if not results:
                return {
                    "answer": "I don't have any relevant information to answer your question. Please make sure documents are uploaded and processed.",
                    "sources": [],
                    "context_used": False
                }
            
            # Step 2: Format context from retrieved documents
            docs_content = []
            sources = []
            
            for doc, score in results:
                docs_content.append(f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}")
                sources.append({
                    "filename": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "page": doc.metadata.get("page", "unknown"),
                    "score": float(score),
                    "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            context_text = "\n\n".join(docs_content)
            
            # Step 3: Generate answer using LLM with context
            llm = ChatOllama(
                model=self.config.CHAT_MODEL,
                base_url=self.config.CHAT_BASE_URL,
                temperature=self.config.CHAT_TEMPERATURE
            )
            
            system_message = SystemMessage(content=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer based on the context, say that you "
                "don't know. Use clear and concise language. "
                "Cite specific sources when possible.\n\n"
                f"Context:\n{context_text}"
            ))
            
            human_message = HumanMessage(content=query)
            
            response = llm.invoke([system_message, human_message])
            
            return {
                "answer": response.content,
                "sources": sources,
                "context_used": True,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"RAG search and answer failed: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "context_used": False,
                "error": str(e)
            }


class RAGDocumentHandler:
    """
    Main application class that orchestrates document processing and web interface.
    """
    
    def __init__(self):
        """Initialize the RAG Document Handler application."""
        self.config = Config()
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = self.config.SECRET_KEY
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.MAX_CONTENT_LENGTH
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.milvus_manager = MilvusManager(self.config)
        
        # Processing status tracking
        self.processing_status: Dict[str, ProcessingStatus] = {}
        
        # Setup directories
        self._setup_directories()
        
        # Register routes
        self._register_routes()
        
        logger.info("RAG Document Handler application initialized")
    
    def _setup_directories(self) -> None:
        """Create necessary directories for file management."""
        directories = [
            self.config.UPLOAD_FOLDER,
            self.config.UPLOADED_FOLDER,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
    
    def _register_routes(self) -> None:
        """Register Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            """Main page showing upload interface and file management."""
            # Get files in staging and uploaded folders
            staging_files = self._get_directory_files(self.config.UPLOAD_FOLDER)
            uploaded_files = self._get_directory_files(self.config.UPLOADED_FOLDER)
            
            # Get collection statistics
            collection_stats = self.milvus_manager.get_collection_stats()
            
            return render_template('index.html',
                                 staging_files=staging_files,
                                 uploaded_files=uploaded_files,
                                 collection_stats=collection_stats,
                                 processing_status=self.processing_status)
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload to staging area."""
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            if file and self._allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                
                try:
                    file.save(file_path)
                    flash(f'File "{filename}" uploaded successfully', 'success')
                    logger.info(f"File uploaded: {filename}")
                except Exception as e:
                    flash(f'Error uploading file: {str(e)}', 'error')
                    logger.error(f"Upload error: {str(e)}")
            else:
                flash('Invalid file type', 'error')
            
            return redirect(url_for('index'))
        
        @self.app.route('/process/<filename>')
        def process_file(filename):
            """Process a file from staging to database."""
            if filename not in self.processing_status:
                self.processing_status[filename] = ProcessingStatus(filename=filename)
            
            # Start processing in background thread
            thread = threading.Thread(target=self._process_document_background, args=(filename,))
            thread.daemon = True
            thread.start()
            
            flash(f'Processing started for "{filename}"', 'info')
            return redirect(url_for('index'))
        
        @self.app.route('/search', methods=['GET', 'POST'])
        def search():
            """Search documents using RAG (Retrieval-Augmented Generation)."""
            rag_result = None
            query = ""
            
            if request.method == 'POST':
                query = request.form.get('query', '').strip()
                if query:
                    try:
                        rag_result = self.milvus_manager.rag_search_and_answer(query)
                        logger.info(f"RAG search completed for query: '{query}'")
                    except Exception as e:
                        flash(f'Search error: {str(e)}', 'error')
                        logger.error(f"RAG search error: {str(e)}")
            
            return render_template('search.html', rag_result=rag_result, query=query)
        
        @self.app.route('/status/<filename>')
        def get_status(filename):
            """Get processing status for a file."""
            status = self.processing_status.get(filename)
            if status:
                return jsonify({
                    'status': status.status,
                    'progress': status.progress,
                    'message': status.message,
                    'chunks_count': status.chunks_count,
                    'error_details': status.error_details
                })
            return jsonify({'status': 'not_found'})
        
        @self.app.route('/delete/<folder>/<filename>')
        def delete_file(folder, filename):
            """Delete a file from staging or uploaded folder."""
            if folder not in ['staging', 'uploaded']:
                flash('Invalid folder', 'error')
                return redirect(url_for('index'))
            
            folder_path = self.config.UPLOAD_FOLDER if folder == 'staging' else self.config.UPLOADED_FOLDER
            file_path = os.path.join(folder_path, filename)
            
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    flash(f'File "{filename}" deleted successfully', 'success')
                    logger.info(f"File deleted: {filename} from {folder}")
                else:
                    flash('File not found', 'error')
            except Exception as e:
                flash(f'Error deleting file: {str(e)}', 'error')
                logger.error(f"Delete error: {str(e)}")
            
            return redirect(url_for('index'))
        
        @self.app.route('/delete_embeddings/<filename>', methods=['POST'])
        def delete_embeddings(filename):
            """Delete all embeddings for a document from Milvus database."""
            try:
                result = self.milvus_manager.delete_document(filename=filename)
                
                if result.get("success"):
                    deleted_count = result.get("deleted_count", 0)
                    flash(f'Successfully deleted {deleted_count} embeddings for "{filename}"', 'success')
                    logger.info(f"Embeddings deleted for {filename}: {deleted_count} chunks")
                else:
                    error_msg = result.get("error", "Unknown error occurred")
                    flash(f'Failed to delete embeddings for "{filename}": {error_msg}', 'error')
                    logger.error(f"Embedding deletion failed for {filename}: {error_msg}")
                    
            except Exception as e:
                flash(f'Error deleting embeddings: {str(e)}', 'error')
                logger.error(f"Embedding deletion error for {filename}: {str(e)}")
            
            return redirect(url_for('index'))
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.config.ALLOWED_EXTENSIONS
    
    def _get_directory_files(self, directory: str) -> List[Dict[str, Any]]:
        """Get list of files in a directory with metadata."""
        files = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        'name': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'status': self.processing_status.get(filename)
                    })
        return sorted(files, key=lambda x: x['modified'], reverse=True)
    
    def _process_document_background(self, filename: str) -> None:
        status = self.processing_status[filename]
        status.status = "processing"; status.start_time = datetime.now(); status.progress = 10; status.message = "Starting document processing..."
        try:
            file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
            status.message = "Loading and chunking document..."; status.progress = 30
            abs_path = str(Path(file_path).resolve())
            file_stat = os.stat(file_path)
            file_sig = f"{abs_path}|{file_stat.st_size}|{int(file_stat.st_mtime)}"
            document_id = hashlib.sha1(file_sig.encode('utf-8')).hexdigest()[:16]
            chunks = self.document_processor.load_and_chunk(file_path, filename, document_id)
            status.chunks_count = len(chunks)
            status.status = "embedding"; status.message = "Enriching metadata via LLM..."; status.progress = 60
            self.document_processor.enrich_topics(chunks)
            status.status = "storing"; status.message = "Storing in Milvus..."; status.progress = 80
            self.milvus_manager.insert_documents(filename, chunks)
            uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            shutil.move(file_path, uploaded_path)
            status.status = "completed"; status.message = f"Successfully processed {len(chunks)} chunks"; status.progress = 100; status.end_time = datetime.now()
        except Exception as e:
            status.status = "error"; status.error_details = str(e); status.message = f"Processing failed: {str(e)}"; status.end_time = datetime.now(); logger.error(f"Processing failed for {filename}: {str(e)}")
    
    def run(self) -> None:
        """Start the Flask web application."""
        logger.info(f"Starting RAG Document Handler on {self.config.FLASK_HOST}:{self.config.FLASK_PORT}")
        self.app.run(
            host=self.config.FLASK_HOST,
            port=self.config.FLASK_PORT,
            debug=self.config.FLASK_DEBUG,
            threaded=True  # Enable threading for background processing
        )


def main() -> None:
    """Main entry point for the application."""
    logger.info("Starting RAG Document Handler application")
    
    # Create and run application
    app = RAGDocumentHandler()
    app.run()


if __name__ == "__main__":
    main()
