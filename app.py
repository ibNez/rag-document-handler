#!/usr/bin/env python3
"""
RAG Knowledgebase Handler - Simplified Single-Interface Application for managing knowledge sources.

A comprehensive knowledgebase store for storing and retrieving information specific requests. A 
common place for managing knowledge sources for RAG implementations.

Features:
- Document upload & management (PDF, DOCX, DOC, TXT, MD)
- URL Locations of trusted sources
- Vector embeddings using Ollama/SentenceTransformers
- Milvus integration for vector storage
- Semantic search with natural language queries
- Web interface with Bootstrap UI
- Threading for responsive UI during long operations
"""

import os
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
from datetime import UTC, datetime
import re
import json
import hashlib

# Web scraping
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

# Flask and web components
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename

# Document processing
from langchain_unstructured import UnstructuredLoader

# Vector and ML components
from pymilvus import connections, utility, Collection
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Milvus as LC_Milvus
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# Configuration
from dotenv import load_dotenv
from ingestion.email import EmailOrchestrator, EmailAccountManager

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
    DELETED_FOLDER: str = os.getenv("DELETED_FOLDER", "deleted")
    ALLOWED_EXTENSIONS: set = field(default_factory=lambda: {"txt", "pdf", "docx", "doc", "md"})
    
    # Embedding Model Configuration
    # EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "ollama")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    CLASSIFICATION_MODEL: str = os.getenv("CLASSIFICATION_MODEL", "mistral")

    # Ollama Configuration
    OLLAMA_EMBEDDING_HOST: str = os.getenv("OLLAMA_EMBEDDING_HOST", "localhost")
    OLLAMA_EMBEDDING_PORT: int = int(os.getenv("OLLAMA_EMBEDDING_PORT", "11434"))
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

    # Email ingestion settings
    EMAIL_ENABLED: bool = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    IMAP_HOST: str = os.getenv('IMAP_HOST', '')
    IMAP_PORT: int = int(os.getenv('IMAP_PORT', '993'))
    IMAP_USERNAME: str = os.getenv('IMAP_USERNAME', '')
    IMAP_PASSWORD: str = os.getenv('IMAP_PASSWORD', '')
    IMAP_MAILBOX: str = os.getenv('IMAP_MAILBOX', 'INBOX')
    IMAP_BATCH_LIMIT: int = int(os.getenv('IMAP_BATCH_LIMIT', '50'))
    IMAP_USE_SSL: bool = os.getenv('IMAP_USE_SSL', 'true').lower() == 'true'
    EMAIL_SYNC_INTERVAL_SECONDS: int = int(os.getenv('EMAIL_SYNC_INTERVAL_SECONDS', '300'))
    EMAIL_DEFAULT_REFRESH_MINUTES: int = int(os.getenv('EMAIL_DEFAULT_REFRESH_MINUTES', '5'))


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
    title: Optional[str] = None


class URLManager:
    """Manages URL storage and validation using SQLite database."""
    
    def __init__(self, db_path: str = os.path.join("databases", "knowledgebase.db")):
        """
        Initialize URL manager with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        # Ensure parent directory exists
        try:
            parent = os.path.dirname(db_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        self.db_path = db_path
        self._init_database()
        try:
            self._log_schema_state()
        except Exception:
            pass
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
                    ignore_robots INTEGER DEFAULT 0,
                    last_content_hash TEXT,
                    last_update_status TEXT,
                    refreshing INTEGER DEFAULT 0,
                    last_refresh_started TIMESTAMP
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
            # Documents metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    title TEXT,
                    page_count INTEGER,
                    chunk_count INTEGER,
                    word_count INTEGER,
                    avg_chunk_chars REAL,
                    median_chunk_chars REAL,
                    top_keywords TEXT, -- JSON array
                    processing_time_seconds REAL,
                    ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Emails metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emails (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    thread_id TEXT,
                    subject TEXT,
                    from_addr TEXT,
                    to_addrs TEXT,
                    cc_addrs TEXT,
                    date_utc TEXT,
                    received_utc TEXT,
                    in_reply_to TEXT,
                    references_ids TEXT,
                    is_reply INTEGER,
                    is_forward INTEGER,
                    raw_size_bytes INTEGER,
                    body_text TEXT,
                    body_html TEXT,
                    language TEXT,
                    has_attachments INTEGER,
                    attachment_manifest TEXT,
                    processed INTEGER DEFAULT 0,
                    ingested_at TEXT,
                    updated_at TEXT,
                    content_hash TEXT,
                    summary TEXT,
                    keywords TEXT,
                    auto_topic TEXT,
                    manual_topic TEXT,
                    topic_confidence REAL,
                    topic_version TEXT,
                    error_state TEXT,
                    direction TEXT,
                    participants TEXT,
                    participants_hash TEXT,
                    to_primary TEXT
                )
            ''')
            # Lightweight migration: ensure required columns exist (idempotent)
            try:
                cursor.execute("PRAGMA table_info(urls)")
                existing = {row[1] for row in cursor.fetchall()}
                required = {
                    'last_content_hash': "ALTER TABLE urls ADD COLUMN last_content_hash TEXT",
                    'last_update_status': "ALTER TABLE urls ADD COLUMN last_update_status TEXT",
                    'refreshing': "ALTER TABLE urls ADD COLUMN refreshing INTEGER DEFAULT 0",
                    'last_refresh_started': "ALTER TABLE urls ADD COLUMN last_refresh_started TIMESTAMP"
                }
                for col, ddl in required.items():
                    if col not in existing:
                        try:
                            cursor.execute(ddl)
                            logger.info(f"Added missing column to urls: {col}")
                        except Exception as _e:
                            logger.warning(f"Failed adding column {col}: {_e}")
                cursor.execute("PRAGMA table_info(url_pages)")
                page_existing = {row[1] for row in cursor.fetchall()}
                if 'last_content_hash' not in page_existing:
                    try:
                        cursor.execute("ALTER TABLE url_pages ADD COLUMN last_content_hash TEXT")
                        logger.info("Added missing column to url_pages: last_content_hash")
                    except Exception as _pe:
                        logger.warning(f"Failed adding last_content_hash to url_pages: {_pe}")
            except Exception as _mig:
                logger.warning(f"URL schema migration check failed: {_mig}")
            conn.commit()
            logger.info("URL database initialized")

    def _ensure_schema(self) -> None:
        """Ensure the database file and core tables exist (self-heal after external deletion).

        This guards against scenarios where the SQLite file is removed (e.g. reset script)
        while the Flask app process keeps running. Without this, subsequent operations
        would raise 'no such table: urls'. We check cheaply and re-run _init_database if missing.
        """
        try:
            if not os.path.exists(self.db_path):
                logger.warning("Knowledgebase DB file missing; reinitializing schema")
                self._init_database()
                return
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='urls'")
                if not cur.fetchone():
                    logger.warning("'urls' table missing; reinitializing schema")
                    self._init_database()
        except Exception as e:
            logger.error(f"Schema ensure failed: {e}")

    def _log_schema_state(self) -> None:
        """Log current columns for key tables (diagnostic)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for table in ("urls", "url_pages", "documents"):
                try:
                    cur.execute(f"PRAGMA table_info({table})")
                    cols = [r[1] for r in cur.fetchall()]
                    logger.info(f"Schema {table} columns: {cols}")
                except Exception as e:
                    logger.warning(f"Could not introspect table {table}: {e}")

    def upsert_document_metadata(self, filename: str, **fields) -> None:
        """Insert or update a documents row."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Build columns
                col_names = ["filename"] + list(fields.keys())
                placeholders = ["?"] * len(col_names)
                values = [filename] + list(fields.values())
                update_assignments = ", ".join([f"{k}=excluded.{k}" for k in fields.keys()])
                sql = f"INSERT INTO documents ({', '.join(col_names)}) VALUES ({', '.join(placeholders)}) ON CONFLICT(filename) DO UPDATE SET {update_assignments}"
                cursor.execute(sql, values)
                conn.commit()
        except Exception:
            logger.exception(f"Failed to upsert metadata for {filename}")
            raise

    def get_document_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM documents WHERE filename = ?", (filename,))
                row = cursor.fetchone()
                if not row:
                    logger.info(f"Metadata lookup miss for {filename}")
                    return None
                d = dict(row)
                # Parse keywords JSON
                try:
                    if d.get('top_keywords'):
                        d['top_keywords'] = json.loads(d['top_keywords'])
                except Exception:
                    pass
                logger.info(f"Metadata lookup succeeded for {filename}")
                return d
        except Exception:
            logger.exception(f"Metadata lookup failed for {filename}")
            return None

    def delete_document_metadata(self, filename: str) -> None:
        """Remove a document metadata row permanently.

        Args:
            filename: Name of the document whose metadata should be deleted.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
                removed = cursor.rowcount
                conn.commit()
            logger.info(f"Metadata delete for {filename}; removed_row={removed > 0}")
        except Exception:
            logger.exception(f"Failed to delete metadata row for {filename}")
    
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
            # Self-heal in case DB was externally reset after startup
            self._ensure_schema()
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
            self._ensure_schema()
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
            self._ensure_schema()
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

    def set_refreshing(self, url_id: int, refreshing: bool) -> None:
        """Set or clear the refreshing flag (and start timestamp when setting)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE urls SET refreshing = ?, last_refresh_started = CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE last_refresh_started END WHERE id = ?",
                    (1 if refreshing else 0, 1 if refreshing else 0, url_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating refreshing flag: {e}")

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
                        # Ensure schema exists (handles external DB deletion while app runs)
                        self._ensure_schema()
                        with sqlite3.connect(self.db_path) as conn:
                                conn.row_factory = sqlite3.Row
                                cursor = conn.cursor()
                                cursor.execute(
                                        """
                                        SELECT * FROM urls
                                        WHERE status = 'active'
                                            AND refresh_interval_minutes IS NOT NULL
                                            AND refresh_interval_minutes > 0
                                            AND (refreshing IS NULL OR refreshing = 0)
                                            AND (
                                                last_scraped IS NULL OR
                                                datetime(last_scraped, '+' || refresh_interval_minutes || ' minutes') <= datetime('now')
                                            )
                                        """
                                )
                                rows = [dict(row) for row in cursor.fetchall()]
                                if not rows:
                                        logger.debug("Scheduler: no due URLs this cycle")
                                return rows
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
            base_url=f"http://{self.config.OLLAMA_EMBEDDING_HOST}:{self.config.OLLAMA_EMBEDDING_PORT}"
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

    def extract_keywords(self, chunks: List[Document], max_keywords: int = 30) -> Dict[str, Any]:
        """Hybrid keyword extraction: LLM global + local frequency per page.

        Returns dict with keys: global_keywords (list), per_page_keywords (dict page->list)
        """
        logger.info(
            f"Starting keyword extraction for {len(chunks)} chunks (max_keywords={max_keywords})"
        )

        # Build page texts
        pages: Dict[int, List[str]] = {}
        for c in chunks:
            p = int(c.metadata.get('page', 0))
            pages.setdefault(p, []).append(c.page_content or '')
        page_texts = {p: '\n'.join(txts) for p, txts in pages.items()}

        # Local frequency extraction
        import re
        stop = set(["the","and","of","to","a","in","for","on","is","with","by","an","be","this","that","are","as","at","from","it","or","was","were","which","can"])
        word_re = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
        per_page_keywords: Dict[int, List[str]] = {}
        global_counts: Dict[str,int] = {}
        for p, text in page_texts.items():
            counts: Dict[str,int] = {}
            for m in word_re.finditer(text.lower()):
                w = m.group(0)
                if w in stop or len(w) > 40:
                    continue
                counts[w] = counts.get(w,0)+1
            # top 8 per page
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
            kws = [k for k,_ in ranked]
            per_page_keywords[p] = kws
            logger.debug(f"Page {p} keyword count: {len(kws)}")
            for k in kws:
                global_counts[k] = global_counts.get(k,0)+1

        # Initial global set from frequency
        freq_global = [k for k,_ in sorted(global_counts.items(), key=lambda kv: kv[1], reverse=True)]

        # LLM call on first page text for semantic keywords + title candidate (reuse logic later)
        llm_keywords = []
        llm_title: Optional[str] = None
        try:
            first_page_text = page_texts.get(min(page_texts.keys()) if page_texts else 0, '')[:2000]
            if first_page_text.strip():
                llm = ChatOllama(model=self.config.CLASSIFICATION_MODEL, base_url=self.config.CLASSIFICATION_BASE_URL, temperature=0)
                prompt = (
                    "You extract document metadata. Given the text excerpt, return JSON with keys: title, keywords. "
                    "keywords: 5-10 concise topical terms (1-3 words), lowercase unless proper noun, no duplicates.\nText:\n" + first_page_text
                )
                resp = llm.invoke(prompt).content.strip()
                s = resp.find('{'); e = resp.rfind('}') + 1
                if s!=-1 and e> s:
                    import json as _json
                    obj = _json.loads(resp[s:e])
                    llm_keywords = [kw for kw in obj.get('keywords', []) if isinstance(kw,str)]
                    if isinstance(obj.get('title'), str):
                        llm_title = obj.get('title').strip() or None
        except Exception as e:
            logger.warning(f"LLM keyword extraction failed: {e}")

        merged = []
        seen = set()
        for source in [llm_keywords, freq_global]:
            for kw in source:
                if kw not in seen:
                    seen.add(kw)
                    merged.append(kw)
                if len(merged) >= max_keywords:
                    break
            if len(merged) >= max_keywords:
                break

        logger.info(
            f"Keyword extraction complete: first keywords={merged[:5]}, llm_title_returned={llm_title is not None}"
        )
        return {
            'global_keywords': merged,
            'per_page_keywords': per_page_keywords,
            'llm_title': llm_title
        }

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
        """Initialize Milvus manager."""
        self.config = config
        self.collection_name = config.COLLECTION_NAME
        self.connection_args = {"host": config.MILVUS_HOST, "port": config.MILVUS_PORT}
        # Establish connection (idempotent)
        try:
            connections.connect(alias="default", **self.connection_args)
        except Exception:
            pass
        # Lazy-created vector store; embeddings model comes from outer processor
        self.vector_store: Optional[LC_Milvus] = None
        try:
            # Create a dedicated embeddings instance (mirrors DocumentProcessor usage)
            self.langchain_embeddings = OllamaEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                base_url=f"http://{self.config.OLLAMA_EMBEDDING_HOST}:{self.config.OLLAMA_EMBEDDING_PORT}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings for MilvusManager: {e}")
            self.langchain_embeddings = None

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _ensure_vector_store(self):
        """Ensure vector store object exists (create collection if missing)."""
        if self.vector_store:
            return
        if not self.langchain_embeddings:
            raise RuntimeError("Embeddings not initialized; cannot create vector store")
        try:
            self.vector_store = LC_Milvus(
                embedding_function=self.langchain_embeddings,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
            )
        except Exception as e:
            # Fallback: try from_texts with empty set to force creation
            try:
                self.vector_store = LC_Milvus.from_texts(
                    texts=["__init__"],
                    embedding=self.langchain_embeddings,
                    metadatas=[{"source": "__init__", "document_id": "__init__", "chunk_id": "init", "page": 0}],
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
            except Exception as e2:
                logger.error(f"Failed to initialize Milvus vector store: {e}; {e2}")
                raise

    # --------------------------------------------------
    # Public insertion API
    # --------------------------------------------------
    def insert_documents(self, source: str, docs: List[Document]) -> int:
        """Insert (or upsert) a list of LangChain Document objects for a given source/URL.

        Ensures deterministic chunk_id & document_id fields and performs simple dedupe
        based on content_hash.
        Returns number of chunks inserted (after dedupe).
        """
        if not docs:
            return 0
        self._ensure_vector_store()
        # Build normalized texts & metadata
        seen_hashes = set()
        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        duplicates = 0
        for idx, d in enumerate(docs):
            content = (d.page_content or '').strip()
            if not content:
                continue
            meta = dict(d.metadata or {})
            # Compute or keep content hash
            ch = meta.get('content_hash')
            if not ch:
                ch = hashlib.sha1(content.encode('utf-8')).hexdigest()[:16]
            if ch in seen_hashes:
                duplicates += 1
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
            metas.append(self._sanitize_and_project_meta(meta))
            texts.append(content)
        if duplicates:
            logger.debug(f"Skipped {duplicates} duplicate chunks for source '{source}'")
        if not texts:
            return 0
        start_time = time.perf_counter()
        try:
            if getattr(self.vector_store, 'collection_name', None) != self.collection_name:
                # Rare mismatch; recreate
                self.vector_store = None
                self._ensure_vector_store()
            # Use add_texts if store already instantiated, else from_texts
            if hasattr(self.vector_store, 'add_texts') and len(getattr(self.vector_store, 'texts', [])) > 0:
                self.vector_store.add_texts(texts=texts, metadatas=metas)
            else:
                # Recreate with from_texts to include all docs + prior ones (fallback)
                self.vector_store = LC_Milvus.from_texts(
                    texts=texts,
                    embedding=self.langchain_embeddings,
                    metadatas=metas,
                    collection_name=self.collection_name,
                    connection_args=self.connection_args,
                )
            try:
                col = Collection(self.collection_name)
                col.flush()
            except Exception:
                pass
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"Inserted {len(texts)} unique chunks for source '{source}' in {elapsed:.2f}s"
            )
            return len(texts)
        except Exception as e:
            logger.error(f"Failed inserting documents for {source}: {e}")
            return 0

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
            # If collection doesn't exist just return success (nothing to delete)
            if not utility.has_collection(self.collection_name):
                return {"success": True, "deleted_count": 0, "entities_before": 0, "entities_after": 0, "verification_remaining": []}
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
                for f in getattr(col, 'schema', {}).fields:
                    try:
                        params = getattr(f, 'params', {}) or {}
                        if 'dim' in params:
                            dim = int(params.get('dim'))
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
        """Check Milvus server reachability and return status info."""
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

            # Simple keyword-boost rerank: if query words intersect chunk keywords, slightly improve score
            try:
                import re
                q_tokens = set(re.findall(r"[A-Za-z0-9_-]+", query.lower()))
                adjusted = []
                for doc, score in results:
                    kws = doc.metadata.get('keywords') or []
                    if isinstance(kws, str):
                        # if stored as JSON string somewhere
                        try:
                            import json as _json
                            kws = _json.loads(kws)
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
            except Exception as _rerank_err:
                logger.debug(f"Keyword rerank skipped: {_rerank_err}")
            
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


class RAGKnowledgebaseManager:
    """
    Main application class that orchestrates document processing and web interface.
    """
    
    def __init__(self):
        """Initialize the RAG Knowledgebase Manager application."""
        self.config = Config()
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = self.config.SECRET_KEY
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.MAX_CONTENT_LENGTH

        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.milvus_manager = MilvusManager(self.config)
        self.url_manager = URLManager()
        # Processing status tracking
        self.processing_status: Dict[str, ProcessingStatus] = {}
        # URL refresh status tracking (keyed by url_id)
        self.url_processing_status: Dict[int, ProcessingStatus] = {}
        # Email refresh status tracking (keyed by account id)
        self.email_processing_status: Dict[int, ProcessingStatus] = {}
        # Scheduler thread handle
        self._scheduler_thread: Optional[threading.Thread] = None

        # Crawler state: robots cache and per-domain last-request timestamps
        self._robots_cache = {}
        self._domain_last_request = {}

        # Setup directories and routes
        self._setup_directories()
        self._register_routes()

        # Track last scheduler cycle time
        self._scheduler_last_cycle: Optional[float] = None

        # Start background scheduler for URL refreshes only in reloader main (to avoid double start in debug)
        should_start = True
        if self.config.FLASK_DEBUG and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
            should_start = False
            logger.info("Deferring scheduler start until reloader main process")
        if should_start:
            try:
                self._start_scheduler()
            except Exception as e:
                logger.error(f"Failed to start scheduler: {e}")

        # Start email orchestrator if enabled
        try:
            email_conn = sqlite3.connect(self.url_manager.db_path, check_same_thread=False)
            self.email_account_manager = EmailAccountManager(email_conn)
            self.email_orchestrator = EmailOrchestrator(
                self.config, self.email_account_manager
            )
        except Exception as e:
            logger.error(f"Failed to start email orchestrator: {e}")

        logger.info("RAG Knowledgebase Manager application initialized")

    # Document metadata now stored in SQLite 'documents' table (no JSON file)
    
    def _setup_directories(self) -> None:
        """Create necessary directories for file management."""
        directories = [
            self.config.UPLOAD_FOLDER,
            self.config.UPLOADED_FOLDER,
            self.config.DELETED_FOLDER,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
    
    def _register_routes(self) -> None:
        """Register Flask routes for the web interface."""
        @self.app.route('/admin/scheduler_status')
        def scheduler_status():  # diagnostic endpoint
            return jsonify(self._scheduler_status())
        
        @self.app.route('/')
        def index():
            """Main page showing upload interface and file management."""
            # Get files in staging and uploaded folders
            staging_files = self._get_directory_files(self.config.UPLOAD_FOLDER)
            uploaded_files = self._get_directory_files(self.config.UPLOADED_FOLDER)
            
            # Get collection statistics
            collection_stats = self.milvus_manager.get_collection_stats()

            # Connection health statuses
            # SQL (SQLite)
            try:
                with sqlite3.connect(self.url_manager.db_path) as _c:
                    _cur = _c.cursor()
                    _cur.execute("SELECT sqlite_version()")
                    ver_row = _cur.fetchone()
                    ver = ver_row[0] if ver_row else None
                sql_status = {"connected": True, "version": ver}
            except Exception as _e_sql:
                sql_status = {"connected": False, "error": str(_e_sql)}

            # Milvus
            milvus_status = self.milvus_manager.check_connection()

            # Aggregate knowledgebase metadata for upstream RAG
            kb_meta = {
                'documents_total': 0,
                'avg_words_per_doc': 0,
                'avg_chunks_per_doc': 0,
                'median_chunk_chars': 0,
                'top_keywords': []
            }
            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) as c, AVG(word_count) as aw, AVG(chunk_count) as ac, AVG(median_chunk_chars) as mc FROM documents")
                    row = cur.fetchone()
                    if row:
                        kb_meta['documents_total'] = int(row['c'] or 0)
                        kb_meta['avg_words_per_doc'] = int(row['aw'] or 0)
                        kb_meta['avg_chunks_per_doc'] = int(row['ac'] or 0)
                        kb_meta['median_chunk_chars'] = int(row['mc'] or 0)
                    # Aggregate top keywords across docs (flatten and count)
                    cur.execute("SELECT top_keywords FROM documents WHERE top_keywords IS NOT NULL")
                    kw_counts = {}
                    for (kw_json,) in cur.fetchall():
                        try:
                            kws = json.loads(kw_json) if kw_json else []
                            for k in kws:
                                kw_counts[k] = kw_counts.get(k, 0) + 1
                        except Exception:
                            continue
                    kb_meta['top_keywords'] = [k for k,_ in sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
            except Exception as e:
                logger.warning(f"KB meta aggregation failed: {e}")

            # Aggregate URL management stats
            url_meta = {
                'total': 0,
                'active': 0,
                'crawl_on': 0,
                'robots_ignored': 0,
                'scraped': 0,
                'never_scraped': 0,
                'due_now': 0,
            }
            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT
                          COUNT(*) AS total,
                          SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active,
                          SUM(CASE WHEN crawl_domain = 1 THEN 1 ELSE 0 END) AS crawl_on,
                          SUM(CASE WHEN ignore_robots = 1 THEN 1 ELSE 0 END) AS robots_ignored,
                          SUM(CASE WHEN last_scraped IS NOT NULL THEN 1 ELSE 0 END) AS scraped,
                          SUM(CASE WHEN last_scraped IS NULL THEN 1 ELSE 0 END) AS never_scraped,
                          SUM(CASE
                                WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 AND (
                                      (last_scraped IS NOT NULL AND datetime(last_scraped, '+' || refresh_interval_minutes || ' minutes') <= datetime('now'))
                                   OR (last_scraped IS NULL)
                                )
                                THEN 1 ELSE 0 END) AS due_now
                        FROM urls
                        """
                    )
                    row = cur.fetchone()
                    if row:
                        url_meta = {k: int(row[k] or 0) for k in url_meta.keys()}
            except Exception as e:
                logger.warning(f"URL meta aggregation failed: {e}")
            
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

            # Get configured email accounts
            email_accounts: List[Dict[str, Any]] = []
            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    manager = EmailAccountManager(conn)
                    email_accounts = manager.list_accounts(include_password=False)
            except Exception as e:
                logger.warning(f"Failed to load email accounts: {e}")

            # Aggregate email statistics
            email_meta = {
                'total_accounts': 0,
                'active_accounts': 0,
                'total_emails': 0,
                'processed_emails': 0,
                'unprocessed_emails': 0,
                'emails_with_attachments': 0,
                'never_synced': 0,
                'due_now': 0,
                'avg_emails_per_account': 0,
                'latest_email_date': None,
                'most_active_account': None
            }
            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    
                    # Account statistics
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_accounts,
                            SUM(CASE WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 THEN 1 ELSE 0 END) as active_accounts,
                            SUM(CASE WHEN last_synced IS NULL THEN 1 ELSE 0 END) as never_synced,
                            SUM(CASE 
                                WHEN refresh_interval_minutes IS NOT NULL AND refresh_interval_minutes > 0 AND (
                                    (last_synced IS NOT NULL AND datetime(last_synced, '+' || refresh_interval_minutes || ' minutes') <= datetime('now'))
                                    OR (last_synced IS NULL)
                                )
                                THEN 1 ELSE 0 END) as due_now
                        FROM email_accounts
                    """)
                    row = cur.fetchone()
                    if row:
                        email_meta['total_accounts'] = int(row['total_accounts'] or 0)
                        email_meta['active_accounts'] = int(row['active_accounts'] or 0)
                        email_meta['never_synced'] = int(row['never_synced'] or 0)
                        email_meta['due_now'] = int(row['due_now'] or 0)
                    
                    # Email content statistics
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_emails,
                            SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) as processed_emails,
                            SUM(CASE WHEN processed = 0 THEN 1 ELSE 0 END) as unprocessed_emails,
                            SUM(CASE WHEN has_attachments = 1 THEN 1 ELSE 0 END) as emails_with_attachments,
                            MAX(date_utc) as latest_email_date
                        FROM emails
                    """)
                    row = cur.fetchone()
                    if row:
                        email_meta['total_emails'] = int(row['total_emails'] or 0)
                        email_meta['processed_emails'] = int(row['processed_emails'] or 0)
                        email_meta['unprocessed_emails'] = int(row['unprocessed_emails'] or 0)
                        email_meta['emails_with_attachments'] = int(row['emails_with_attachments'] or 0)
                        email_meta['latest_email_date'] = row['latest_email_date']
                    
                    # Calculate average emails per account
                    if email_meta['total_accounts'] > 0:
                        email_meta['avg_emails_per_account'] = round(email_meta['total_emails'] / email_meta['total_accounts'], 1)
                    
                    # Find most active account (by email count)
                    cur.execute("""
                        SELECT ea.account_name, COUNT(e.id) as email_count
                        FROM email_accounts ea
                        LEFT JOIN emails e ON ea.email_address = e.from_addr OR ea.email_address = e.to_primary
                        GROUP BY ea.id, ea.account_name
                        ORDER BY email_count DESC
                        LIMIT 1
                    """)
                    row = cur.fetchone()
                    if row and row['email_count'] > 0:
                        email_meta['most_active_account'] = row['account_name']
                        
            except Exception as e:
                logger.warning(f"Email meta aggregation failed: {e}")

            return render_template(
                'index.html',
                staging_files=staging_files,
                uploaded_files=uploaded_files,
                collection_stats=collection_stats,
                sql_status=sql_status,
                milvus_status=milvus_status,
                kb_meta=kb_meta,
                url_meta=url_meta,
                email_meta=email_meta,
                processing_status=self.processing_status,
                url_processing_status=self.url_processing_status,
                urls=enriched_urls,
                email_accounts=email_accounts,
                email_processing_status=self.email_processing_status,
                config=self.config,
            )
        
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

        @self.app.route('/download/<path:filename>')
        def download_file(filename):
            """Download a processed (uploaded) document by filename.

            Only serves files that exist in the uploaded (processed) folder. Prevents path traversal.
            """
            # Basic path traversal guard
            if '..' in filename or filename.startswith('/'):
                abort(400)
            file_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            if not os.path.isfile(file_path):
                flash('File not found or not processed yet', 'error')
                return redirect(url_for('index'))
            return send_from_directory(self.config.UPLOADED_FOLDER, filename, as_attachment=True)

        @self.app.route('/document/<path:filename>/update', methods=['POST'])
        def update_document_metadata(filename):
            """Update editable document metadata like title."""
            title = request.form.get('title','').strip()
            if '..' in filename or filename.startswith('/'):
                abort(400)
            uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            if not os.path.isfile(uploaded_path):
                flash('Document not found', 'error')
                return redirect(url_for('index'))
            try:
                if title:
                    self.url_manager.upsert_document_metadata(filename, title=title)
                    flash('Document title updated', 'success')
                else:
                    flash('No title provided', 'info')
            except Exception as e:
                flash(f'Update failed: {e}', 'error')
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
                # TTL cleanup: drop completed entries older than 90s
                try:
                    if status.status == 'completed' and status.end_time and (datetime.now() - status.end_time).total_seconds() > 90:
                        self.processing_status.pop(filename, None)
                        return jsonify({'status': 'not_found'})
                except Exception:
                    pass
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

        @self.app.route('/email_status/<int:account_id>')
        def get_email_status(account_id: int):
            """Get processing status for an email account refresh.
            When background status is gone, also return final DB state."""
            status = self.email_processing_status.get(account_id)
            if status:
                return jsonify({
                    'status': status.status,
                    'progress': status.progress,
                    'message': status.message,
                    'error_details': status.error_details
                })
            try:
                if self.email_account_manager:
                    for acct in self.email_account_manager.list_accounts(include_password=False):
                        if acct.get('id') == account_id:
                            return jsonify({
                                'status': 'not_found',
                                'last_update_status': acct.get('last_update_status'),
                                'last_synced': acct.get('last_synced'),
                                'next_run': acct.get('next_run')
                            })
                return jsonify({'status': 'deleted'})
            except Exception:
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

        @self.app.route('/delete_file_bg/<folder>/<filename>', methods=['POST','GET'])
        def delete_file_bg(folder: str, filename: str):
            """Start background (soft) deletion for a file.

            Primary method is POST from a form. GET is accepted as a safe fallback to
            prevent user-facing 405 errors (e.g. if a link is followed) and will also
            initiate deletion but logs a note. For security-sensitive deployments you
            may wish to disable GET here or add CSRF.
            """
            if folder not in ['staging', 'uploaded']:
                flash('Invalid folder', 'error')
                return redirect(url_for('index'))
            if request.method == 'GET':
                logger.info(f"GET fallback invoked for deletion of {filename} in {folder}")
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

        @self.app.route('/email_accounts', methods=['GET'])
        def list_email_accounts():
            """Return JSON list of configured email accounts."""
            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    manager = EmailAccountManager(conn)
                    accounts = manager.list_accounts(include_password=False)
                return jsonify(accounts)
            except Exception as e:
                logger.error(f"Failed to fetch email accounts: {e}")
                return jsonify({"error": "Failed to fetch email accounts"}), 500

        @self.app.route('/email_accounts', methods=['POST'])
        def add_email_account():
            """Create a new email account configuration."""
            account_name = request.form.get('account_name', '').strip()
            server_type = request.form.get('server_type', 'imap').strip().lower() or 'imap'
            server = request.form.get('server', '').strip()
            email_address = request.form.get('email_address', '').strip()
            password = request.form.get('password', '').strip()
            port_str = request.form.get('port', '').strip()
            mailbox = request.form.get('mailbox', '').strip() or None
            batch_limit_str = request.form.get('batch_limit', '').strip()
            refresh_raw = request.form.get('refresh_interval_minutes', '').strip()
            use_ssl = request.form.get('use_ssl') in ('1', 'on')

            logger.info(
                "Request to add email account '%s' (%s) on %s for %s",
                account_name,
                server_type,
                server,
                email_address,
            )

            if not all([account_name, server, email_address, password, port_str]):
                flash('Missing required fields', 'error')
                return redirect(url_for('index'))

            try:
                port = int(port_str)
                batch_limit = int(batch_limit_str) if batch_limit_str else None
                refresh_interval = int(refresh_raw) if refresh_raw else Config.EMAIL_DEFAULT_REFRESH_MINUTES
            except ValueError:
                flash('Port, batch limit and refresh interval must be numbers', 'error')
                return redirect(url_for('index'))

            record = {
                'account_name': account_name,
                'server_type': server_type,
                'server': server,
                'port': port,
                'email_address': email_address,
                'password': password,
                'mailbox': mailbox,
                'batch_limit': batch_limit,
                'use_ssl': 1 if use_ssl else 0,
                'refresh_interval_minutes': refresh_interval,
            }

            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    manager = EmailAccountManager(conn)
                    manager.create_account(record)
                logger.info("Email account '%s' added successfully", account_name)
                flash('Email account added successfully', 'success')
            except Exception as e:
                logger.exception(
                    "Failed to add email account '%s': %s", account_name, e
                )
                flash(f'Failed to add email account: {e}', 'error')

            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<int:account_id>', methods=['POST'])
        def update_email_account(account_id: int):
            """Update an existing email account configuration."""
            account_name = request.form.get('account_name', '').strip()
            server = request.form.get('server', '').strip()
            email_address = request.form.get('email_address', '').strip()
            password = request.form.get('password', '').strip()
            port_str = request.form.get('port', '').strip()
            mailbox = request.form.get('mailbox', '').strip() or None
            batch_limit_str = request.form.get('batch_limit', '').strip()
            refresh_raw = request.form.get('refresh_interval_minutes', '').strip()
            use_ssl = request.form.get('use_ssl') in ('1', 'on')
            server_type = request.form.get('server_type', '').strip()

            updates: Dict[str, Any] = {}
            if account_name:
                updates['account_name'] = account_name
            if server:
                updates['server'] = server
            if server_type:
                updates['server_type'] = server_type.lower()
            if email_address:
                updates['email_address'] = email_address
            if password:
                updates['password'] = password
            if mailbox is not None:
                updates['mailbox'] = mailbox
            if batch_limit_str:
                try:
                    updates['batch_limit'] = int(batch_limit_str)
                except ValueError:
                    flash('Batch limit must be a number', 'error')
                    return redirect(url_for('index'))
            if port_str:
                try:
                    updates['port'] = int(port_str)
                except ValueError:
                    flash('Port must be a number', 'error')
                    return redirect(url_for('index'))
            if refresh_raw:
                try:
                    updates['refresh_interval_minutes'] = int(refresh_raw)
                except ValueError:
                    flash('Refresh interval must be a number', 'error')
                    return redirect(url_for('index'))
            updates['use_ssl'] = 1 if use_ssl else 0

            if not updates:
                flash('No fields to update', 'error')
                return redirect(url_for('index'))

            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    manager = EmailAccountManager(conn)
                    manager.update_account(account_id, updates)
                flash('Email account updated successfully', 'success')
            except Exception as e:
                flash(f'Failed to update email account: {e}', 'error')

            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<int:account_id>/delete', methods=['POST'])
        def delete_email_account(account_id: int):
            """Remove an email account configuration."""
            try:
                with sqlite3.connect(self.url_manager.db_path) as conn:
                    manager = EmailAccountManager(conn)
                    manager.delete_account(account_id)
                flash('Email account deleted', 'success')
            except Exception as e:
                flash(f'Failed to delete email account: {e}', 'error')

            return redirect(url_for('index'))

        @self.app.route('/email_accounts/<int:account_id>/refresh', methods=['POST'])
        def refresh_email_account(account_id: int):
            """Trigger immediate sync for an email account."""
            if not getattr(self, 'email_account_manager', None):
                flash('Email ingestion not configured', 'error')
                return redirect(url_for('index'))
            th = threading.Thread(target=self._refresh_email_account_background, args=(account_id,))
            th.daemon = True
            th.start()
            flash('Email refresh started in background', 'info')
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
        """Return file entries for a directory with optional metadata enrichment.

        Args:
            directory: Absolute or relative path of folder to scan.

        Returns:
            List of dict entries each containing: name, size (bytes), modified (datetime),
            optional processing status, and for processed documents (uploaded folder) the
            persisted metadata (title/chunk/page/word counts, keywords).
        """
        files: List[Dict[str, Any]] = []
        if not os.path.exists(directory):
            return files
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue
            stat = os.stat(file_path)
            entry: Dict[str, Any] = {
                'name': filename,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'status': self.processing_status.get(filename)
            }
            if directory == self.config.UPLOADED_FOLDER:
                meta = self.url_manager.get_document_metadata(filename) or {}
                entry['title'] = meta.get('title') or self._fallback_title_from_filename(filename)
                entry['chunks_count'] = meta.get('chunk_count')
                entry['page_count'] = meta.get('page_count')
                entry['word_count'] = meta.get('word_count')
                entry['top_keywords'] = meta.get('top_keywords')
                st = entry['status']
                if st and st.status == 'completed' and (st.message or '').lower().startswith('deletion complete'):
                    # Skip lingering deletion status entries to avoid stale display
                    continue
            files.append(entry)
        return sorted(files, key=lambda x: x['modified'], reverse=True)

    def _fallback_title_from_filename(self, filename: str) -> str:
        base = os.path.splitext(filename)[0]
        base = base.replace('_', ' ').replace('-', ' ').strip()
        if not base:
            return filename
        words = []
        for w in base.split():
            if len(w) > 3 and w.isupper():
                words.append(w)
            else:
                words.append(w.capitalize())
        return ' '.join(words)

    def _derive_title_from_chunks(self, filename: str, chunks: List[Document]) -> str:
        try:
            candidates: List[str] = []
            for c in chunks[:5]:
                text = (c.page_content or '').strip()
                for line in text.splitlines()[:5]:
                    l = line.strip()
                    if not l:
                        continue
                    if len(l) > 120:
                        continue
                    words = l.split()
                    if not (3 <= len(words) <= 15):
                        continue
                    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
                    if len(alpha_words) / len(words) < 0.6:
                        continue
                    tc = sum(1 for w in alpha_words if w[:1].isupper()) / max(1, len(alpha_words))
                    if tc < 0.5:
                        continue
                    if l.endswith((':', ',')):
                        continue
                    candidates.append(l)
            if candidates:
                good = [c for c in candidates if len(c) >= 15]
                chosen = sorted(good or candidates, key=len)[0]
                return chosen
        except Exception:
            pass
        return self._fallback_title_from_filename(filename)
    
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
            # Derive and persist title
            try:
                title = self._derive_title_from_chunks(filename, chunks)
                status.title = title
            except Exception as e:
                logger.warning(f"Title derivation failed for {filename}: {e}")
            # Keyword extraction (hybrid)
            status.status = "embedding"; status.message = "Extracting keywords..."; status.progress = 55
            kw_data = self.document_processor.extract_keywords(chunks)
            global_keywords = kw_data['global_keywords']
            per_page_kw = kw_data['per_page_keywords']
            # Prefer LLM-derived title if reasonable
            llm_title = kw_data.get('llm_title')
            if llm_title:
                try:
                    # Basic quality heuristics: length & word count & not all caps
                    wc = len(llm_title.split())
                    if 2 <= wc <= 20 and 8 <= len(llm_title) <= 140 and not llm_title.isupper():
                        status.title = llm_title
                except Exception:
                    pass
            # Attach keywords list to each chunk metadata (global subset for search boost)
            for c in chunks:
                c.metadata['keywords'] = global_keywords
            # Basic stats
            word_count = sum(len((c.page_content or '').split()) for c in chunks)
            lengths = [len(c.page_content or '') for c in chunks]
            import statistics
            avg_len = float(sum(lengths)/len(lengths)) if lengths else 0.0
            med_len = float(statistics.median(lengths)) if lengths else 0.0
            page_count = max(int(c.metadata.get('page',0)) for c in chunks) + 1 if chunks else 0
            status.status = "embedding"; status.message = "Enriching metadata via LLM..."; status.progress = 60
            self.document_processor.enrich_topics(chunks)
            status.status = "storing"; status.message = "Storing in Milvus..."; status.progress = 80
            self.milvus_manager.insert_documents(filename, chunks)
            uploaded_path = os.path.join(self.config.UPLOADED_FOLDER, filename)
            shutil.move(file_path, uploaded_path)
            # Upsert document metadata row
            elapsed = (datetime.now() - status.start_time).total_seconds() if status.start_time else None
            try:
                self.url_manager.upsert_document_metadata(
                    filename,
                    title=status.title,
                    page_count=page_count,
                    chunk_count=len(chunks),
                    word_count=word_count,
                    avg_chunk_chars=avg_len,
                    median_chunk_chars=med_len,
                    top_keywords=json.dumps(global_keywords),
                    processing_time_seconds=elapsed
                )
            except Exception as e:
                logger.warning(f"Failed to upsert document metadata for {filename}: {e}")
            status.status = "completed"; status.message = f"Successfully processed {len(chunks)} chunks"; status.progress = 100; status.end_time = datetime.now()
        except Exception as e:
            status.status = "error"; status.error_details = str(e); status.message = f"Processing failed: {str(e)}"; status.end_time = datetime.now(); logger.error(f"Processing failed for {filename}: {str(e)}")

    def _delete_file_background(self, folder: str, filename: str) -> None:
        """Background deletion worker.

        Strict removal policy:
        - Staging files: permanently removed from disk.
        - Uploaded files: embeddings removed (Milvus), metadata row purged (SQLite), file moved
          to DELETED_FOLDER (only raw file retained for possible manual restore). No metadata
          history is preserved.
        After completion the ProcessingStatus entry is dropped so re-uploads start with fresh
        processing and regenerated metadata.
        """
        # Ensure status entry exists for UI progress
        st = self.processing_status.get(filename) or ProcessingStatus(filename=filename)
        self.processing_status[filename] = st
        st.status = 'processing'
        st.progress = 5
        st.message = 'Preparing deletion...'
        st.start_time = datetime.now()
        try:
            folder_path = self.config.UPLOAD_FOLDER if folder == 'staging' else self.config.UPLOADED_FOLDER
            file_path = os.path.join(folder_path, filename)
            if folder == 'uploaded':
                st.message = 'Removing embeddings...'
                st.progress = 20
                try:
                    self.milvus_manager.delete_document(filename=filename)
                except Exception as e:
                    logger.warning(f"Embedding deletion failed for {filename}: {e}")
                else:
                    st.progress = 40
                # Purge metadata row
                try:
                    self.url_manager.delete_document_metadata(filename)
                except Exception as e:
                    logger.warning(f"Metadata deletion failed for {filename}: {e}")
                else:
                    st.progress = max(st.progress, 55)
            st.message = 'Archiving file...'
            st.progress = max(st.progress, 70)
            try:
                if os.path.exists(file_path):
                    if folder == 'uploaded':
                        archive_path = os.path.join(self.config.DELETED_FOLDER, filename)
                        if os.path.exists(archive_path):
                            stem, ext = os.path.splitext(filename)
                            archive_path = os.path.join(self.config.DELETED_FOLDER, f"{stem}_{int(time.time())}{ext}")
                        shutil.move(file_path, archive_path)
                    else:
                        os.remove(file_path)
            except Exception as e:
                logger.error(f"File archive/delete failed for {filename}: {e}")
                raise
            st.status = 'completed'
            st.message = 'Deletion complete'
            st.progress = 100
            st.end_time = datetime.now()
        except Exception as e:
            st.status = 'error'
            st.error_details = str(e)
            st.message = f'Deletion failed: {e}'
            st.progress = 100
            st.end_time = datetime.now()
        finally:
            # Keep status entry so UI can observe 'Deletion complete'.
            # A TTL cleanup will purge it later via get_status.
            pass

    def _refresh_email_account_background(self, account_id: int) -> None:
        """Fetch emails for a specific account in a background thread."""
        orchestrator = getattr(self, "email_orchestrator", None)
        if orchestrator is None and getattr(self, "email_account_manager", None):
            try:
                cfg = getattr(self, "config", None)
                if cfg is None:
                    class _Cfg:
                        EMAIL_ENABLED = True
                        EMAIL_DEFAULT_REFRESH_MINUTES = 5

                    cfg = _Cfg()
                orchestrator = EmailOrchestrator(cfg, self.email_account_manager)
            except Exception:
                logger.exception(
                    "Failed to initialize email orchestrator for account %s", account_id
                )
                return
        if orchestrator is None:
            return
        try:
            st = self.email_processing_status.get(account_id)
            if st is None and self.email_account_manager:
                for acct in self.email_account_manager.list_accounts(
                    include_password=False
                ):
                    if acct.get("id") == account_id:
                        st = ProcessingStatus(
                            filename=acct.get("account_name")
                            or acct.get("email_address")
                        )
                        self.email_processing_status[account_id] = st
                        break
            if st:
                st.status = "processing"
                st.message = "Starting email refresh..."
                st.progress = 5
                st.start_time = datetime.now()

            orchestrator.run(account_id=account_id)

            if st:
                st.status = "completed"
                st.message = "Email refresh complete"
                st.progress = 100
                st.end_time = datetime.now()
        except Exception as exc:
            logger.exception("Email refresh error for account %s", account_id)
            st = self.email_processing_status.get(account_id)
            if st:
                st.status = "error"
                st.message = f"Refresh failed: {exc}"
                st.error_details = str(exc)
                st.progress = 100
                st.end_time = datetime.now()
            raise
        finally:
            try:
                del self.email_processing_status[account_id]
            except KeyError:
                logger.warning(
                    f"No processing status found for account {account_id} during cleanup"
                )
            except Exception:
                logger.exception(
                    "Unexpected error cleaning up status for account %s", account_id
                )


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
        """Background loop that schedules URL and email refresh tasks."""
        logger.info("Scheduler started")
        cycle = 0
        while True:
            try:
                cycle += 1
                self._scheduler_last_cycle = time.time()
                due_urls = self.url_manager.get_due_urls()
                due_accounts = []
                if getattr(self, 'email_orchestrator', None):
                    try:
                        due_accounts = self.email_orchestrator.get_due_accounts()
                    except Exception as exc:
                        logger.error(f"Failed to get due email accounts: {exc}")
                        due_accounts = []
                    try:
                        active_emails_total = (
                            self.email_orchestrator.account_manager.get_account_count()
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error(f"Failed to get email account count: {exc}")
                        active_emails_total = 0
                else:
                    active_emails_total = 0
                logger.info(
                    "Scheduler cycle %s heartbeat: urls_due=%s emails_due=%s active_urls_total=%s active_emails_total=%s",
                    cycle,
                    len(due_urls),
                    len(due_accounts),
                    self.url_manager.get_url_count(),
                    active_emails_total,
                )
                started = 0
                for rec in due_urls:
                    if rec['id'] in self.url_processing_status:
                        continue
                    self.url_processing_status[rec['id']] = ProcessingStatus(filename=rec.get('title') or rec.get('url'))
                    t = threading.Thread(target=self._process_url_background, args=(rec['id'],))
                    t.daemon = True
                    t.start()
                    started += 1
                for account in due_accounts:
                    acct_id = account.get('id')
                    if acct_id is None or acct_id in self.email_processing_status:
                        continue
                    self.email_processing_status[acct_id] = ProcessingStatus(filename=account.get('account_name') or account.get('email_address'))
                    t = threading.Thread(target=self._refresh_email_account_background, args=(acct_id,))
                    t.daemon = True
                    t.start()
                    started += 1
                sleep_for = self.config.SCHEDULER_POLL_SECONDS_BUSY if started else self.config.SCHEDULER_POLL_SECONDS_IDLE
                logger.debug(
                    f"Scheduler cycle {cycle}: started {started} task(s); sleeping {sleep_for}s"
                )
                time.sleep(sleep_for)
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(30)

    def _start_scheduler(self):
        """Start the unified scheduler thread if not already running."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.debug("Scheduler thread already running")
            return
        th = threading.Thread(target=self._scheduler_loop, name="scheduler")
        th.daemon = True
        th.start()
        self._scheduler_thread = th
        logger.info(f"Scheduler thread started (ident={th.ident})")

    def _scheduler_status(self) -> Dict[str, Any]:
        """Return current scheduler diagnostic info."""
        alive = bool(self._scheduler_thread and self._scheduler_thread.is_alive())
        try:
            due_preview = self.url_manager.get_due_urls()[:5]
        except Exception:
            due_preview = []
        last_cycle_age = None
        if self._scheduler_last_cycle:
            try:
                last_cycle_age = round(time.time() - self._scheduler_last_cycle, 2)
            except Exception:
                pass
        return {
            'running': alive,
            'thread_ident': getattr(self._scheduler_thread, 'ident', None),
            'due_count': len(due_preview),
            'due_sample': [d.get('url') for d in due_preview],
            'in_progress': list(self.url_processing_status.keys()),
            'last_cycle_age_seconds': last_cycle_age,
            'poll_busy_seconds': self.config.SCHEDULER_POLL_SECONDS_BUSY,
            'poll_idle_seconds': self.config.SCHEDULER_POLL_SECONDS_IDLE
        }

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
        logger.info(f"Starting RAG Knowledgebase Manager on {self.config.FLASK_HOST}:{self.config.FLASK_PORT}")
        self.app.run(
            host=self.config.FLASK_HOST,
            port=self.config.FLASK_PORT,
            debug=self.config.FLASK_DEBUG,
            threaded=True  # Enable threading for background processing
        )


def main() -> None:
    """Main entry point for the application."""
    logger.info("Starting RAG Knowledgebase Manager application")
    
    # Create and run application
    app = RAGKnowledgebaseManager()
    app.run()


if __name__ == "__main__":
    main()
