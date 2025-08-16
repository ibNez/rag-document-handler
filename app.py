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
import sqlite3
import urllib.parse
import requests
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import json
import hashlib
from email_accounts import EmailAccount, EmailAccountManager

# Web scraping
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

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
    # EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    CLASSIFICATION_MODEL: str = os.getenv("CLASSIFICATION_MODEL", "mistral")

    # Ollama Configuration
    OLLAMA_CLASSIFICATION_HOST: str = os.getenv("OLLAMA_CLASSIFICATION_HOST", "localhost")
    CLASSIFICATION_BASE_URL: str = os.getenv('CHAT_CLASSIFICATIONBASE_URL', f"http://{os.getenv('OLLAMA_CLASSIFICATION_HOST','localhost')}:{os.getenv('OLLAMA_PORT','11434')}")
    OLLAMA_CHAT_HOST: str = os.getenv("OLLAMA_CHAT_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "11434"))

    # Chat Model Configuration
    CHAT_MODEL: str = os.getenv('CHAT_MODEL', 'mistral:latest')
    CHAT_BASE_URL: str = os.getenv('CHAT_BASE_URL', f"http://{os.getenv('OLLAMA_CHAT_HOST','localhost')}:{os.getenv('OLLAMA_PORT','11434')}")
    CHAT_TEMPERATURE: float = float(os.getenv('CHAT_TEMPERATURE', '0.1'))
    
    # Unstructured chunking
    UNSTRUCTURED_CHUNKING_STRATEGY: str = os.getenv('UNSTRUCTURED_CHUNKING_STRATEGY', 'basic')
    UNSTRUCTURED_MAX_CHARACTERS: int = int(os.getenv('UNSTRUCTURED_MAX_CHARACTERS', '1000'))
    UNSTRUCTURED_OVERLAP: int = int(os.getenv('UNSTRUCTURED_OVERLAP', '200'))
    UNSTRUCTURED_INCLUDE_ORIG: bool = os.getenv('UNSTRUCTURED_INCLUDE_ORIG', 'false').lower() == 'true'
    
    # Milvus flags
    MILVUS_DROP_COLLECTION: bool = os.getenv('MILVUS_DROP_COLLECTION', 'false').lower() == 'false'

    # Crawl settings
    CRAWL_MAX_PAGES: int = int(os.getenv('CRAWL_MAX_PAGES', '50'))
    CRAWL_REQUEST_TIMEOUT: int = int(os.getenv('CRAWL_REQUEST_TIMEOUT', '10'))
    CRAWL_USER_AGENT: str = os.getenv('CRAWL_USER_AGENT', 'RAGDocHandlerBot/1.0 (+contact: you@example.com)')
    CRAWL_DELAY_SECONDS: float = float(os.getenv('CRAWL_DELAY_SECONDS', '1.0'))
    CRAWL_JITTER_SECONDS: float = float(os.getenv('CRAWL_JITTER_SECONDS', '0.3'))

    # Scheduler settings
    SCHEDULER_POLL_SECONDS_BUSY: float = float(os.getenv('SCHEDULER_POLL_SECONDS_BUSY', '10'))
    SCHEDULER_POLL_SECONDS_IDLE: float = float(os.getenv('SCHEDULER_POLL_SECONDS_IDLE', '30'))
    URL_DEFAULT_REFRESH_MINUTES: int = int(os.getenv('URL_DEFAULT_REFRESH_MINUTES', '1440'))


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


class URLManager:
    """Manages URL storage and validation using SQLite database."""

    def __init__(self, db_path: str = "knowledgebase.db"):
        """
        Initialize URL manager with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"URLManager initialized with database: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    title TEXT,
                    description TEXT,
                    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_checked TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    refresh_interval_minutes INTEGER DEFAULT 1440,
                    last_scraped TIMESTAMP,
                    crawl_domain INTEGER DEFAULT 0,
                    ignore_robots INTEGER DEFAULT 0
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS url_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_url_id INTEGER NOT NULL,
                    page_url TEXT UNIQUE NOT NULL,
                    last_content_hash TEXT,
                    last_scraped TIMESTAMP,
                    FOREIGN KEY(parent_url_id) REFERENCES urls(id)
                )
            ''')
            # Ensure new columns exist for existing databases
            try:
                cursor.execute("PRAGMA table_info(urls)")
                cols = {row[1] for row in cursor.fetchall()}
                if 'refresh_interval_minutes' not in cols:
                    cursor.execute("ALTER TABLE urls ADD COLUMN refresh_interval_minutes INTEGER")
                if 'last_scraped' not in cols:
                    cursor.execute("ALTER TABLE urls ADD COLUMN last_scraped TIMESTAMP")
                if 'crawl_domain' not in cols:
                    cursor.execute("ALTER TABLE urls ADD COLUMN crawl_domain INTEGER DEFAULT 0")
                if 'last_content_hash' not in cols:
                    cursor.execute("ALTER TABLE urls ADD COLUMN last_content_hash TEXT")
                if 'last_update_status' not in cols:
                    cursor.execute("ALTER TABLE urls ADD COLUMN last_update_status TEXT")
                if 'ignore_robots' not in cols:
                    cursor.execute("ALTER TABLE urls ADD COLUMN ignore_robots INTEGER DEFAULT 0")
            except Exception as e:
                logger.warning(f"URL table migration check/add columns warning: {e}")
            # Backfill default refresh interval for any existing rows where it's NULL
            try:
                cursor.execute(
                    "UPDATE urls SET refresh_interval_minutes = ? WHERE refresh_interval_minutes IS NULL",
                    (Config.URL_DEFAULT_REFRESH_MINUTES,)
                )
            except Exception as e:
                logger.warning(f"Failed to backfill default refresh interval: {e}")
            # Backfill crawl_domain to 0 when NULL
            try:
                cursor.execute(
                    "UPDATE urls SET crawl_domain = 0 WHERE crawl_domain IS NULL"
                )
            except Exception as e:
                logger.warning(f"Failed to backfill crawl_domain: {e}")
            # Backfill ignore_robots to 0 when NULL
            try:
                cursor.execute(
                    "UPDATE urls SET ignore_robots = 0 WHERE ignore_robots IS NULL"
                )
            except Exception as e:
                logger.warning(f"Failed to backfill ignore_robots: {e}")
            conn.commit()
            logger.info("URL database initialized")
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if the provided string is a valid URL.
        
        Args:
            url: URL string to validate
            
        Returns:
            bool: True if valid URL, False otherwise
        """
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def extract_title_from_url(self, url: str) -> str:
        """
        Extract the title from a web page by scraping the <title> tag.
        
        Args:
            url: URL to scrape for title
            
        Returns:
            str: Extracted title or fallback to domain name
        """
        try:
            # Set headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make request with timeout
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title = title_tag.string.strip()
                # Clean up common title patterns
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                title = title[:200]  # Limit length
                return title
            
            # Fallback to domain name if no title found
            parsed_url = urllib.parse.urlparse(url)
            return parsed_url.netloc
            
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch title from {url}: {str(e)}")
            # Fallback to domain name on error
            try:
                parsed_url = urllib.parse.urlparse(url)
                return parsed_url.netloc
            except Exception:
                return "Unknown Title"
        except Exception as e:
            logger.error(f"Error extracting title from {url}: {str(e)}")
            return "Unknown Title"
    
    def add_url(self, url: str, title: str = None, description: str = None) -> Dict[str, Any]:
        """
        Add a new URL to the database with automatic title extraction.
        
        Args:
            url: URL to add
            title: Optional title override (will extract from page if None)
            description: Optional description
            
        Returns:
            Dict with success status and message
        """
        # Validate URL format
        if not self.validate_url(url):
            return {"success": False, "message": "Invalid URL format"}
        
        # Extract title from URL if not provided
        if not title:
            logger.info(f"Extracting title from URL: {url}")
            title = self.extract_title_from_url(url)
            logger.info(f"Extracted title: {title}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO urls (url, title, description, refresh_interval_minutes, crawl_domain, ignore_robots) VALUES (?, ?, ?, ?, ?, ?)",
                    (url, title, description, Config.URL_DEFAULT_REFRESH_MINUTES, 0, 0)
                )
                conn.commit()
                url_id = cursor.lastrowid
                logger.info(f"Added URL: {url} with title: {title} (ID: {url_id})")
                return {"success": True, "message": "URL added successfully", "id": url_id, "title": title}
        except sqlite3.IntegrityError:
            return {"success": False, "message": "URL already exists"}
        except Exception as e:
            logger.error(f"Error adding URL: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}
    
    def get_all_urls(self) -> List[Dict[str, Any]]:
        """
        Retrieve all URLs from the database.
        
        Returns:
            List of URL dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 
                        *,
                        CASE 
                            WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 AND last_scraped IS NOT NULL
                            THEN datetime(last_scraped, '+' || refresh_interval_minutes || ' minutes')
                            ELSE NULL
                        END AS next_refresh
                    FROM urls 
                    WHERE status = 'active' 
                    ORDER BY added_date DESC
                    """
                )
                urls = [dict(row) for row in cursor.fetchall()]
                return urls
        except Exception as e:
            logger.error(f"Error retrieving URLs: {str(e)}")
            return []
    
    def delete_url(self, url_id: int) -> Dict[str, Any]:
        """
        Delete a URL from the database.
        
        Args:
            url_id: ID of the URL to delete
            
        Returns:
            Dict with success status and message
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM urls WHERE id = ?", (url_id,))
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Deleted URL with ID: {url_id}")
                    return {"success": True, "message": "URL deleted successfully"}
                else:
                    return {"success": False, "message": "URL not found"}
        except Exception as e:
            logger.error(f"Error deleting URL: {str(e)}")
            return {"success": False, "message": f"Database error: {str(e)}"}

    def get_pages_for_parent(self, parent_url_id: int) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT page_url FROM url_pages WHERE parent_url_id = ?", (parent_url_id,))
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting pages for parent {parent_url_id}: {e}")
            return []

    def delete_pages_for_parent(self, parent_url_id: int) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM url_pages WHERE parent_url_id = ?", (parent_url_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting page records for parent {parent_url_id}: {e}")
    
    def get_url_count(self) -> int:
        """Get the total number of active URLs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM urls WHERE status = 'active'")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting URL count: {str(e)}")
            return 0

    def get_url_by_id(self, url_id: int) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM urls WHERE id = ?", (url_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error getting URL by id: {e}")
            return None

    def update_url_metadata(self, url_id: int, title: Optional[str], description: Optional[str], refresh_interval_minutes: Optional[int], crawl_domain: Optional[int], ignore_robots: Optional[int]) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE urls SET title = ?, description = ?, refresh_interval_minutes = ?, crawl_domain = ?, ignore_robots = ? WHERE id = ?",
                    (title, description, refresh_interval_minutes, crawl_domain, ignore_robots, url_id)
                )
                conn.commit()
                return {"success": cursor.rowcount > 0}
        except Exception as e:
            logger.error(f"Error updating URL metadata: {e}")
            return {"success": False, "message": str(e)}

    def mark_scraped(self, url_id: int, refresh_interval_minutes: Optional[int]) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE urls SET last_scraped = CURRENT_TIMESTAMP WHERE id = ?",
                    (url_id,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error marking URL scraped: {e}")

    def update_url_hash_status(self, url_id: int, content_hash: Optional[str], status: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE urls SET last_content_hash = ?, last_update_status = ?, last_scraped = CURRENT_TIMESTAMP WHERE id = ?",
                    (content_hash, status, url_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating URL hash/status: {e}")

    def get_page_hash(self, page_url: str) -> Optional[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT last_content_hash FROM url_pages WHERE page_url = ?", (page_url,))
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting page hash: {e}")
            return None

    def set_page_hash(self, parent_url_id: int, page_url: str, content_hash: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO url_pages (parent_url_id, page_url, last_content_hash, last_scraped) VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(page_url) DO UPDATE SET last_content_hash=excluded.last_content_hash, last_scraped=CURRENT_TIMESTAMP",
                    (parent_url_id, page_url, content_hash)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error setting page hash: {e}")

    def get_due_urls(self) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM urls
                    WHERE status = 'active'
                      AND refresh_interval_minutes IS NOT NULL
                      AND refresh_interval_minutes > 0
                      AND (
                        last_scraped IS NULL OR
                        datetime(last_scraped, '+' || refresh_interval_minutes || ' minutes') <= datetime('now')
                      )
                    """
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching due URLs: {e}")
            return []


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
            base_url=f"http://{self.config.CLASSIFICATION_BASE_URL}"
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

    def load_and_chunk_url(self, url: str, url_id: str) -> List[Document]:
        """Load a URL using UnstructuredLoader(web_url=...) and create lean chunks with metadata, mirroring file chunking settings."""
        logger.info(f"Loading and chunking URL: {url}")
        try:
            loader = UnstructuredLoader(
                web_url=url,
                chunking_strategy=self.config.UNSTRUCTURED_CHUNKING_STRATEGY,
                max_characters=self.config.UNSTRUCTURED_MAX_CHARACTERS,
                overlap=self.config.UNSTRUCTURED_OVERLAP,
                include_orig_elements=self.config.UNSTRUCTURED_INCLUDE_ORIG,
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} elements from URL via UnstructuredLoader(web_url)")

            chunks: List[Document] = []
            for i, d in enumerate(documents):
                text = (d.page_content or '').strip()
                if not text:
                    continue
                content_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]
                meta = d.metadata or {}
                page = meta.get('page') or meta.get('page_number') or (i + 1)
                meta.update({
                    'source': url,
                    'page': page,
                    'document_id': url_id,
                    'chunk_id': f"{url_id}:{content_hash}",
                    'content_hash': content_hash,
                    'content_length': len(text),
                })
                d.metadata = meta
                d.page_content = text
                chunks.append(d)

            logger.info(f"Created {len(chunks)} URL chunks with metadata")
            return chunks
        except Exception as e:
            logger.error(f"Failed to load and chunk URL {url}: {e}")
            raise

    def enrich_topics(self, chunks: List[Document]) -> None:
        """
        LLM topic classification exactly like the notebook.
        
        Args:
            chunks: List of document chunks to enrich
        """
        logger.info("Enriching chunk metadata with LLM classification (topic)...")
        
        llm = ChatOllama(
            model=self.config.CLASSIFICATION_MODEL, 
            base_url=self.config.CLASSIFICATION_BASE_URL, 
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
        self.embedding_provider = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL, base_url=f"http://{self.config.OLLAMA_CLASSIFICATION_HOST}:{self.config.OLLAMA_PORT}")
        self.vector_store = None
        self.connected = False
        self._connect()
        logger.info("MilvusManager (LangChain) initialized")
    
    def _connect(self) -> None:
        """Establish connection to Milvus database using LangChain."""
        self.collection_name = self.config.COLLECTION_NAME
        self.connection_args = {"host": self.config.MILVUS_HOST, "port": self.config.MILVUS_PORT}
        self.langchain_embeddings = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL, base_url=f"http://{self.config.OLLAMA_CLASSIFICATION_HOST}:{self.config.OLLAMA_PORT}")
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
        
        # Deduplicate by content_hash (compute fallback from text if missing)
        seen = set()
        unique: List[Document] = []
        for c in chunks:
            meta = c.metadata or {}
            ch = meta.get('content_hash')
            if not ch:
                try:
                    text = (c.page_content or '').strip()
                    if not text:
                        continue
                    ch = hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]
                    meta['content_hash'] = ch
                    c.metadata = meta
                except Exception:
                    continue
            if ch in seen:
                continue
            seen.add(ch)
            unique.append(c)
        
        logger.info(f"Unique chunks after dedupe: {len(unique)} (from {len(chunks)})")
        
        if not unique:
            logger.warning(f"No unique chunks to insert for {filename}; skipping insertion")
            return
        
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
        self.url_manager = URLManager()
        self.email_account_manager = EmailAccountManager()

        # Processing status tracking
        self.processing_status: Dict[str, ProcessingStatus] = {}
        # URL refresh status tracking (keyed by url_id)
        self.url_processing_status: Dict[int, ProcessingStatus] = {}

        # Setup directories
        self._setup_directories()

        # Register routes
        self._register_routes()

        # Start background scheduler for URL refreshes
        self._start_scheduler()

        # Crawler state: robots cache and per-domain last-request timestamps
        self._robots_cache = {}
        self._domain_last_request = {}

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
            
            # Get URLs for display
            urls = self.url_manager.get_all_urls()
            # Enrich with robots status info
            enriched_urls = []
            for u in urls:
                try:
                    rp, crawl_delay = self._get_robots(u.get('url', ''))
                    try:
                        allowed = rp.can_fetch(self.config.CRAWL_USER_AGENT, u.get('url', ''))
                    except Exception:
                        allowed = True
                    try:
                        has_entries = bool(getattr(rp, 'default_entry', None) or getattr(rp, 'entries', []))
                    except Exception:
                        has_entries = False
                    u['robots_allowed'] = 1 if allowed else 0
                    u['robots_has_rules'] = 1 if (has_entries or crawl_delay is not None) else 0
                    u['robots_crawl_delay'] = crawl_delay
                except Exception:
                    u['robots_allowed'] = None
                    u['robots_has_rules'] = None
                    u['robots_crawl_delay'] = None
                enriched_urls.append(u)
            email_accounts = self.email_account_manager.list_accounts()
            
            return render_template('index.html',
                                 staging_files=staging_files,
                                 uploaded_files=uploaded_files,
                                 collection_stats=collection_stats,
                                 processing_status=self.processing_status,
                                 url_processing_status=self.url_processing_status,
                                 urls=enriched_urls,
                                 email_accounts=email_accounts)
        
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

        @self.app.route('/url_status/<int:url_id>')
        def get_url_status(url_id: int):
            """Get processing status for a URL refresh.
            When background status is gone, also return final DB state (last_update_status, last_scraped, next_refresh).
            """
            status = self.url_processing_status.get(url_id)
            if status:
                return jsonify({
                    'status': status.status,
                    'progress': status.progress,
                    'message': status.message,
                    'error_details': status.error_details
                })
            # No in-memory status; fetch final state from DB for in-place UI update
            try:
                rec = self.url_manager.get_url_by_id(url_id)
                if rec:
                    # Compute next_refresh similar to list view
                    next_refresh = None
                    try:
                        last = rec.get('last_scraped')
                        interval = rec.get('refresh_interval_minutes')
                        if last and interval and int(interval) > 0:
                            # SQLite returns 'YYYY-MM-DD HH:MM:SS'
                            from datetime import datetime, timedelta
                            try:
                                dt_last = datetime.fromisoformat(str(last))
                            except Exception:
                                # Fallback: parse only first 19 chars
                                dt_last = datetime.fromisoformat(str(last)[:19])
                            dt_next = dt_last + timedelta(minutes=int(interval))
                            next_refresh = dt_next.strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        next_refresh = None
                    return jsonify({
                        'status': 'not_found',
                        'last_update_status': rec.get('last_update_status'),
                        'last_scraped': rec.get('last_scraped'),
                        'next_refresh': next_refresh
                    })
                else:
                    # Record is gone; the URL was deleted
                    return jsonify({'status': 'deleted'})
            except Exception:
                pass
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

        @self.app.route('/delete_file_bg/<folder>/<filename>', methods=['POST'])
        def delete_file_bg(folder: str, filename: str):
            """Start background deletion for a file with a progress bar, optionally removing embeddings if from uploaded."""
            if folder not in ['staging', 'uploaded']:
                flash('Invalid folder', 'error')
                return redirect(url_for('index'))
            # Seed processing status so the card shows a bar immediately
            st = self.processing_status.get(filename)
            if not st:
                self.processing_status[filename] = ProcessingStatus(filename=filename)
            # Start background worker
            th = threading.Thread(target=self._delete_file_background, args=(folder, filename))
            th.daemon = True
            th.start()
            flash('Deletion started in background', 'info')
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
        
        @self.app.route('/add_url', methods=['POST'])
        def add_url():
            """Add a new URL with automatic title extraction."""
            url = request.form.get('url', '').strip()
            
            if not url:
                flash('URL is required', 'error')
                return redirect(url_for('index'))
            
            result = self.url_manager.add_url(url)
            if result['success']:
                extracted_title = result.get('title', 'Unknown')
                flash(f'URL added successfully with title: "{extracted_title}"', 'success')
            else:
                flash(f'Failed to add URL: {result["message"]}', 'error')
            
            return redirect(url_for('index'))
        
        @self.app.route('/delete_url/<int:url_id>', methods=['POST'])
        def delete_url(url_id):
            """Delete a URL."""
            # Load the URL and any crawled pages
            url_rec = self.url_manager.get_url_by_id(url_id)
            if not url_rec:
                flash('URL not found', 'error')
                return redirect(url_for('index'))
            url = url_rec.get('url')
            # Delete embeddings for the single URL
            try:
                doc_id = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()[:16]
                self.milvus_manager.delete_document(document_id=doc_id)
            except Exception as e:
                logger.warning(f"Failed to delete primary URL embeddings: {e}")
            # If domain crawl, delete each page embeddings
            try:
                pages = self.url_manager.get_pages_for_parent(url_id)
                for page_url in pages:
                    try:
                        page_doc_id = hashlib.sha1(page_url.strip().encode('utf-8')).hexdigest()[:16]
                        self.milvus_manager.delete_document(document_id=page_doc_id)
                    except Exception as de:
                        logger.warning(f"Failed to delete page embeddings for {page_url}: {de}")
                # Remove page records
                self.url_manager.delete_pages_for_parent(url_id)
            except Exception as e:
                logger.warning(f"Failed to clean up url_pages: {e}")

            # Finally remove the URL record
            result = self.url_manager.delete_url(url_id)
            if result['success']:
                flash('URL and related embeddings deleted successfully', 'success')
            else:
                flash(f'Failed to delete URL: {result["message"]}', 'error')
            return redirect(url_for('index'))

        @self.app.route('/email_accounts/add', methods=['POST'])
        def add_email_account():
            """Add a new email account configuration."""
            account = EmailAccount(
                name=request.form.get('name', '').strip(),
                imap_host=request.form.get('imap_host', '').strip(),
                imap_user=request.form.get('imap_user', '').strip(),
                imap_password=request.form.get('imap_password', '').strip(),
                imap_port=int(request.form.get('imap_port', '993') or 993),
                mailbox=request.form.get('mailbox', 'INBOX').strip() or 'INBOX',
            )
            try:
                self.email_account_manager.add_account(account)
                flash('Email account added', 'success')
            except Exception as e:
                flash(f'Failed to add account: {e}', 'error')
            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<name>/delete', methods=['POST'])
        def delete_email_account(name: str):
            """Delete an email account configuration."""
            self.email_account_manager.remove_account(name)
            flash('Email account removed', 'info')
            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<name>/update', methods=['POST'])
        def update_email_account(name: str):
            """Update an existing email account."""
            account = EmailAccount(
                name=name,
                imap_host=request.form.get('imap_host', '').strip(),
                imap_user=request.form.get('imap_user', '').strip(),
                imap_password=request.form.get('imap_password', '').strip(),
                imap_port=int(request.form.get('imap_port', '993') or 993),
                mailbox=request.form.get('mailbox', 'INBOX').strip() or 'INBOX',
            )
            try:
                self.email_account_manager.update_account(account)
                flash('Email account updated', 'success')
            except Exception as e:
                flash(f'Failed to update account: {e}', 'error')
            return redirect(url_for('index'))


        @self.app.route('/delete_url_bg/<int:url_id>', methods=['POST'])
        def delete_url_bg(url_id: int):
            """Start background deletion of a URL and its embeddings, with progress bar."""
            url_rec = self.url_manager.get_url_by_id(url_id)
            if not url_rec:
                flash('URL not found', 'error')
                return redirect(url_for('index'))
            # If already processing something (refresh or delete), don't start another
            if url_id in self.url_processing_status:
                flash('An operation is already in progress for this URL', 'info')
                return redirect(url_for('index'))
            # Seed a status entry so the UI shows a progress bar immediately after redirect
            self.url_processing_status[url_id] = ProcessingStatus(filename=url_rec.get('title') or url_rec.get('url'))
            th = threading.Thread(target=self._delete_url_background, args=(url_id,))
            th.daemon = True
            th.start()
            flash('Deletion started in background', 'info')
            return redirect(url_for('index'))

        @self.app.route('/update_url/<int:url_id>', methods=['POST'])
        def update_url(url_id: int):
            """Update URL metadata including title, description, and refresh schedule."""
            title = request.form.get('title')
            description = request.form.get('description')
            refresh_raw = request.form.get('refresh_interval_minutes')
            crawl_domain_flag = 1 if request.form.get('crawl_domain') in ('on', '1', 'true', 'True') else 0
            ignore_robots_flag = 1 if request.form.get('ignore_robots') in ('on', '1', 'true', 'True') else 0
            refresh_interval_minutes = None
            if refresh_raw:
                try:
                    refresh_interval_minutes = int(refresh_raw)
                    if refresh_interval_minutes < 0:
                        refresh_interval_minutes = None
                except ValueError:
                    refresh_interval_minutes = None
            result = self.url_manager.update_url_metadata(url_id, title, description, refresh_interval_minutes, crawl_domain_flag, ignore_robots_flag)
            if result.get('success'):
                flash('URL metadata updated', 'success')
            else:
                flash(f"Failed to update URL: {result.get('message','Unknown error')}", 'error')
            return redirect(url_for('index'))

        @self.app.route('/ingest_url/<int:url_id>', methods=['POST'])
        def ingest_url(url_id: int):
            """Trigger immediate ingestion/refresh for a URL."""
            url_rec = self.url_manager.get_url_by_id(url_id)
            if not url_rec:
                flash('URL not found', 'error')
                return redirect(url_for('index'))
            # Initialize and start background refresh with progress if not already running
            if url_id in self.url_processing_status:
                flash('Refresh already in progress for this URL', 'info')
                return redirect(url_for('index'))
            self.url_processing_status[url_id] = ProcessingStatus(filename=url_rec.get('title') or url_rec.get('url'))
            th = threading.Thread(target=self._process_url_background, args=(url_rec['id'],))
            th.daemon = True
            th.start()
            flash('Refresh started in background', 'info')
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

    def _delete_file_background(self, folder: str, filename: str) -> None:
        """Background deletion of a local file; when deleting from uploaded, remove embeddings first."""
        # Ensure status entry
        st = self.processing_status.get(filename) or ProcessingStatus(filename=filename)
        self.processing_status[filename] = st
        st.status = 'processing'; st.progress = 5; st.message = 'Preparing deletion...'; st.start_time = datetime.now()
        try:
            folder_path = self.config.UPLOAD_FOLDER if folder == 'staging' else self.config.UPLOADED_FOLDER
            file_path = os.path.join(folder_path, filename)
            # When in uploaded, remove embeddings by filename
            if folder == 'uploaded':
                st.message = 'Removing embeddings...'; st.progress = 25
                try:
                    self.milvus_manager.delete_document(filename=filename)
                except Exception as e:
                    logger.warning(f"Embedding deletion failed for {filename}: {e}")
            # Delete the file from disk
            st.message = 'Deleting file from disk...'; st.progress = 75
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"File delete failed for {filename}: {e}")
                raise
            st.status = 'completed'; st.message = 'Deletion complete'; st.progress = 100; st.end_time = datetime.now()
        except Exception as e:
            st.status = 'error'; st.error_details = str(e); st.message = f'Deletion failed: {e}'; st.progress = 100; st.end_time = datetime.now()
        finally:
            # Clear the status after a short delay to let UI catch last update
            try:
                # Keep brief window; UI polling will switch to normal card display after reloads or refresh
                pass
            except Exception:
                pass

    def _process_url_background(self, url_id: int) -> None:
        """Fetch, chunk, and index the content at a URL. Replace previous embeddings for that URL."""
        try:
            url_rec = self.url_manager.get_url_by_id(url_id)
            if not url_rec:
                return
            url = url_rec['url']
            # prepare URL status object
            st = self.url_processing_status.get(url_id)
            if st is None:
                st = ProcessingStatus(filename=url)
                self.url_processing_status[url_id] = st
            st.status = "processing"; st.progress = 5; st.message = "Starting URL refresh..."; st.start_time = datetime.now()
            # Deterministic document_id from URL string
            url_sig = url.strip()
            document_id = hashlib.sha1(url_sig.encode('utf-8')).hexdigest()[:16]
            logger.info(f"Refreshing URL id={url_id} url={url} doc_id={document_id}")
            # Crawl entire domain if enabled; else only single page
            if int(url_rec.get('crawl_domain') or 0) == 1:
                st.message = "Discovering domain pages..."; st.progress = 10
                pages = self._discover_domain_urls(url, max_pages=getattr(self.config, 'CRAWL_MAX_PAGES', 50))
                logger.info(f"Domain crawl enabled: discovered {len(pages)} page(s) for {url}")
                any_updated = False
                any_content = False
                total = max(1, len(pages))
                processed = 0
                for page_url in pages:
                    try:
                        if not (int(url_rec.get('ignore_robots') or 0) == 1 or self._can_fetch(page_url)):
                            logger.debug(f"Robots disallow (skip): {page_url}")
                            continue
                        netloc = urllib.parse.urlparse(page_url).netloc
                        self._respect_rate_limit(netloc)
                        st.status = "chunking"; st.message = f"Fetching & chunking {processed+1}/{total} pages..."; st.progress = max(st.progress, 15)
                        page_doc_id = hashlib.sha1(page_url.strip().encode('utf-8')).hexdigest()[:16]
                        chunks = self.document_processor.load_and_chunk_url(page_url, page_doc_id)
                        if not chunks:
                            # mark page as no content
                            self.url_manager.set_page_hash(url_id, page_url, '')
                            processed += 1
                            st.progress = min(95, 15 + int(75 * processed / total))
                            continue
                        # Compute full content hash from concatenated text
                        full_text = "\n".join(d.page_content or '' for d in chunks).strip()
                        page_hash = hashlib.sha1(full_text.encode('utf-8')).hexdigest()[:16] if full_text else ''
                        prev_hash = self.url_manager.get_page_hash(page_url)
                        if prev_hash and prev_hash == page_hash:
                            logger.info(f"No change for {page_url}; skipping reindex")
                            self.url_manager.set_page_hash(url_id, page_url, page_hash)
                            processed += 1
                            st.progress = min(95, 15 + int(75 * processed / total))
                            continue
                        st.status = "embedding"; st.message = f"Enriching metadata {processed+1}/{total}..."; st.progress = max(st.progress, 40)
                        self.document_processor.enrich_topics(chunks)
                        st.status = "storing"; st.message = f"Storing in Milvus {processed+1}/{total}..."; st.progress = max(st.progress, 60)
                        try:
                            self.milvus_manager.delete_document(document_id=page_doc_id)
                        except Exception as de:
                            logger.warning(f"Delete existing embeddings failed for {page_url}: {de}")
                        # Safety: avoid inserting if chunks list is empty after any filtering
                        if chunks:
                            self.milvus_manager.insert_documents(page_url, chunks)
                        self.url_manager.set_page_hash(url_id, page_url, page_hash)
                        any_updated = True
                        any_content = True
                        processed += 1
                        st.progress = min(95, 15 + int(75 * processed / total))
                    except Exception as pe:
                        logger.error(f"Failed processing page {page_url}: {pe}")
                # Set aggregate parent status
                if any_updated:
                    self.url_manager.update_url_hash_status(url_id, None, 'updated')
                else:
                    # if we saw any content and none updated, unchanged; else no_content
                    agg_status = 'unchanged' if any_content else 'no_content'
                    self.url_manager.update_url_hash_status(url_id, None, agg_status)
            else:
                # Load and chunk single page
                if not (int(url_rec.get('ignore_robots') or 0) == 1 or self._can_fetch(url)):
                    logger.warning(f"Robots disallow for {url}; skipping fetch")
                    st.status = "error"; st.message = "Blocked by robots.txt"; st.progress = 100; st.end_time = datetime.now()
                    return
                self._respect_rate_limit(urllib.parse.urlparse(url).netloc)
                st.status = "chunking"; st.message = "Loading and chunking page..."; st.progress = 30
                chunks = self.document_processor.load_and_chunk_url(url, document_id)
                if not chunks:
                    logger.warning(f"No chunks produced for {url}; marking no_content")
                    self.url_manager.update_url_hash_status(url_id, None, 'no_content')
                    logger.info(f"URL refresh complete (no content): {url}")
                    st.status = "completed"; st.message = "No content"; st.progress = 100; st.end_time = datetime.now()
                    return
                full_text = "\n".join(d.page_content or '' for d in chunks).strip()
                page_hash = hashlib.sha1(full_text.encode('utf-8')).hexdigest()[:16] if full_text else ''
                prev_hash = (url_rec.get('last_content_hash') or '').strip()
                if prev_hash and prev_hash == page_hash:
                    logger.info(f"No change for {url}; skipping reindex")
                    self.url_manager.update_url_hash_status(url_id, page_hash, 'unchanged')
                    st.status = "completed"; st.message = "Unchanged"; st.progress = 100; st.end_time = datetime.now()
                    return
                st.status = "embedding"; st.message = "Enriching metadata via LLM..."; st.progress = 60
                self.document_processor.enrich_topics(chunks)
                st.status = "storing"; st.message = "Storing in Milvus..."; st.progress = 80
                try:
                    self.milvus_manager.delete_document(document_id=document_id)
                except Exception as e:
                    logger.warning(f"Delete existing URL embeddings warning: {e}")
                # Safety: avoid inserting if chunks ended up empty after dedupe
                if chunks:
                    self.milvus_manager.insert_documents(url, chunks)
                self.url_manager.update_url_hash_status(url_id, page_hash, 'updated')

            # Mark scraped
            if int(url_rec.get('crawl_domain') or 0) == 1:
                # For domain crawl we already set an aggregate status above; just ensure last_scraped is updated
                try:
                    self.url_manager.mark_scraped(url_id, url_rec.get('refresh_interval_minutes'))
                except Exception:
                    pass
            logger.info(f"URL refresh complete: {url}")
            st.status = "completed"; st.message = "Refresh complete"; st.progress = 100; st.end_time = datetime.now()
        except Exception as e:
            logger.error(f"URL processing failed (id={url_id}): {e}")
            try:
                st = self.url_processing_status.get(url_id)
                if st:
                    st.status = "error"; st.message = f"Refresh failed: {e}"; st.error_details = str(e); st.progress = 100; st.end_time = datetime.now()
            except Exception:
                pass
        finally:
            # Always clear the in-memory URL status to avoid infinite UI polling/reloads
            try:
                del self.url_processing_status[url_id]
            except Exception:
                pass

    def _scheduler_loop(self):
        """Background loop to schedule URL refresh based on next_run."""
        logger.info("URL scheduler started")
        while True:
            try:
                due = self.url_manager.get_due_urls()
                if due:
                    logger.info(f"Scheduler found {len(due)} due URL(s)")
                for rec in due:
                    # Avoid starting a duplicate refresh if one is already running for this URL
                    if rec['id'] in self.url_processing_status:
                        continue
                    self.url_processing_status[rec['id']] = ProcessingStatus(filename=rec.get('title') or rec.get('url'))
                    t = threading.Thread(target=self._process_url_background, args=(rec['id'],))
                    t.daemon = True
                    t.start()
                # sleep shorter if work was done
                time.sleep(self.config.SCHEDULER_POLL_SECONDS_BUSY if due else self.config.SCHEDULER_POLL_SECONDS_IDLE)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(30)

    def _start_scheduler(self):
        th = threading.Thread(target=self._scheduler_loop)
        th.daemon = True
        th.start()

    def _discover_domain_urls(self, base_url: str, max_pages: int = 50) -> List[str]:
        """Discover a bounded set of in-domain URLs starting from base_url."""
        try:
            parsed = urllib.parse.urlparse(base_url)
            base_netloc = parsed.netloc
            headers = { 'User-Agent': self.config.CRAWL_USER_AGENT }
            to_visit: List[str] = [base_url]
            visited: set = set()
            discovered: List[str] = []
            blocked_exts = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf', '.zip', '.tar', '.gz', '.rar', '.7z', '.mp4', '.mp3', '.wav'}
            # Check if the parent URL has ignore_robots flag
            ignore_robots_parent = False
            try:
                # Resolve parent URL id via DB
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    c = conn.cursor()
                    c.execute("SELECT ignore_robots FROM urls WHERE url = ?", (base_url,))
                    row = c.fetchone()
                    if row:
                        ignore_robots_parent = int(dict(row).get('ignore_robots') or 0) == 1
            except Exception:
                pass
            while to_visit and len(discovered) < max_pages:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                try:
                    if not (ignore_robots_parent or self._can_fetch(current)):
                        logger.debug(f"Robots disallow: {current}")
                        continue
                    self._respect_rate_limit(base_netloc)
                    resp = requests.get(current, headers=headers, timeout=self.config.CRAWL_REQUEST_TIMEOUT)
                    ct = resp.headers.get('Content-Type', '')
                    if 'text/html' not in ct:
                        continue
                    discovered.append(current)
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        abs_url = urllib.parse.urljoin(current, href)
                        u = urllib.parse.urlparse(abs_url)
                        if u.scheme not in ('http', 'https'):
                            continue
                        if u.netloc != base_netloc:
                            continue
                        path_lower = u.path.lower()
                        if any(path_lower.endswith(ext) for ext in blocked_exts):
                            continue
                        # Normalize: remove params/query/fragment; strip trailing slash
                        norm = urllib.parse.urlunparse((u.scheme, u.netloc, u.path.rstrip('/'), '', '', ''))
                        if norm not in visited and norm not in to_visit and norm not in discovered:
                            to_visit.append(norm)
                except Exception:
                    continue
            return discovered
        except Exception as e:
            logger.error(f"Discovery failed for {base_url}: {e}")
            return []

    def _get_robots(self, base_url: str) -> Tuple[RobotFileParser, Optional[float]]:
        """Fetch and cache robots.txt for a domain; returns parser and optional crawl delay."""
        try:
            parsed = urllib.parse.urlparse(base_url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin in self._robots_cache:
                return self._robots_cache[origin]
            robots_url = urllib.parse.urljoin(origin, '/robots.txt')
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                pass
            # urllib's RobotFileParser doesn't expose crawl-delay; we'll do a manual fetch to parse
            crawl_delay: Optional[float] = None
            try:
                headers = { 'User-Agent': self.config.CRAWL_USER_AGENT }
                resp = requests.get(robots_url, headers=headers, timeout=self.config.CRAWL_REQUEST_TIMEOUT)
                if resp.status_code == 200 and isinstance(resp.text, str):
                    ua = None
                    for line in resp.text.splitlines():
                        s = line.strip()
                        if not s or s.startswith('#'):
                            continue
                        lower = s.lower()
                        if lower.startswith('user-agent:'):
                            ua = s.split(':',1)[1].strip()
                        elif lower.startswith('crawl-delay:'):
                            val = s.split(':',1)[1].strip()
                            try:
                                cd = float(val)
                                # apply if this section matches our UA or wildcard
                                if ua in (self.config.CRAWL_USER_AGENT, '*'):
                                    crawl_delay = cd
                            except Exception:
                                pass
            except Exception:
                pass
            self._robots_cache[origin] = (rp, crawl_delay)
            return rp, crawl_delay
        except Exception:
            # default: allow
            rp = RobotFileParser()
            rp.parse("")
            return rp, None

    def _can_fetch(self, url: str) -> bool:
        try:
            rp, _ = self._get_robots(url)
            return rp.can_fetch(self.config.CRAWL_USER_AGENT, url)
        except Exception:
            return True

    def _respect_rate_limit(self, netloc: str) -> None:
        """Sleep to respect per-domain rate limits and robots crawl-delay if set."""
        try:
            origin = f"https://{netloc}"
            _, crawl_delay = self._get_robots(origin)
            base_delay = crawl_delay if crawl_delay is not None else self.config.CRAWL_DELAY_SECONDS
            jitter = random.uniform(0, self.config.CRAWL_JITTER_SECONDS)
            delay = max(0.0, base_delay + jitter)
            now = time.time()
            last = self._domain_last_request.get(netloc, 0)
            elapsed = now - last
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._domain_last_request[netloc] = time.time()
        except Exception:
            pass

    def _delete_url_background(self, url_id: int) -> None:
        """Background deletion of a URL: remove embeddings for the URL and its crawled pages, then delete DB records."""
        st = self.url_processing_status.get(url_id) or ProcessingStatus(filename=str(url_id))
        self.url_processing_status[url_id] = st
        st.status = 'processing'; st.progress = 5; st.message = 'Preparing deletion...'; st.start_time = datetime.now()
        try:
            url_rec = self.url_manager.get_url_by_id(url_id)
            if not url_rec:
                st.status = 'completed'; st.progress = 100; st.message = 'Already deleted'; st.end_time = datetime.now()
                return
            url = url_rec.get('url')
            title = url_rec.get('title') or url
            # Delete primary URL embeddings
            st.status = 'storing'; st.message = 'Removing primary embeddings...'; st.progress = 25
            try:
                doc_id = hashlib.sha1(url.strip().encode('utf-8')).hexdigest()[:16]
                self.milvus_manager.delete_document(document_id=doc_id)
            except Exception as e:
                logger.warning(f"Failed to delete primary URL embeddings: {e}")
            # Delete crawled page embeddings if any
            pages = []
            try:
                pages = self.url_manager.get_pages_for_parent(url_id)
            except Exception as e:
                logger.warning(f"Failed to list url_pages for {url_id}: {e}")
            total = max(1, len(pages))
            for idx, page_url in enumerate(pages):
                try:
                    st.message = f"Removing page {idx+1}/{len(pages)}..."; st.progress = min(90, 25 + int(60 * (idx+1) / total))
                    page_doc_id = hashlib.sha1(page_url.strip().encode('utf-8')).hexdigest()[:16]
                    self.milvus_manager.delete_document(document_id=page_doc_id)
                except Exception as de:
                    logger.warning(f"Failed to delete page embeddings for {page_url}: {de}")
            # Remove url_pages records
            try:
                self.url_manager.delete_pages_for_parent(url_id)
            except Exception as e:
                logger.warning(f"Failed to delete url_pages rows for {url_id}: {e}")
            # Finally remove the URL record
            st.message = 'Removing URL record...'; st.progress = 95
            try:
                self.url_manager.delete_url(url_id)
            except Exception as e:
                logger.error(f"Failed to delete URL record {url_id}: {e}")
            st.status = 'completed'; st.message = 'Deletion complete'; st.progress = 100; st.end_time = datetime.now()
        except Exception as e:
            st.status = 'error'; st.error_details = str(e); st.message = f'Deletion failed: {e}'; st.progress = 100; st.end_time = datetime.now()
        finally:
            # Clear status so UI can get 'deleted' from url_status
            try:
                del self.url_processing_status[url_id]
            except Exception:
                pass
    
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
