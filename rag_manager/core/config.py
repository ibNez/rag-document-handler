"""
Core configuration module for RAG Knowledgebase Manager.

This module contains all configuration settings and dataclasses following 
the development rules for centralized configuration management.
"""

import os
from dataclasses import dataclass, field
from typing import Set, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {"txt", "pdf", "docx", "doc", "md"})
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    CLASSIFICATION_MODEL: str = os.getenv("CLASSIFICATION_MODEL", "mistral")

    # Ollama Configuration
    OLLAMA_EMBEDDING_HOST: str = os.getenv("OLLAMA_EMBEDDING_HOST", "localhost")
    OLLAMA_CLASSIFICATION_HOST: str = os.getenv("OLLAMA_CLASSIFICATION_HOST", "localhost")
    OLLAMA_CHAT_HOST: str = os.getenv("OLLAMA_CHAT_HOST", "localhost")
    OLLAMA_EMBEDDING_PORT: int = int(os.getenv("OLLAMA_EMBEDDING_PORT", "11434"))
    OLLAMA_CLASSIFICATION_PORT: int = int(os.getenv("OLLAMA_CLASSIFICATION_PORT", "11434"))
    OLLAMA_CHAT_PORT: int = int(os.getenv("OLLAMA_CHAT_PORT", "11434"))
    
    # Chat Model Configuration
    CHAT_MODEL: str = os.getenv('CHAT_MODEL', 'mistral:latest')
    CHAT_BASE_URL: str = os.getenv(
        'CHAT_BASE_URL', 
        f"http://{os.getenv('OLLAMA_CHAT_HOST','localhost')}:{os.getenv('OLLAMA_CHAT_PORT','11434')}"
    )
    CLASSIFICATION_BASE_URL: str = os.getenv(
        'CLASSIFICATION_BASE_URL', 
        f"http://{os.getenv('OLLAMA_CLASSIFICATION_HOST','localhost')}:{os.getenv('OLLAMA_CLASSIFICATION_PORT','11434')}"
    )
    CHAT_TEMPERATURE: float = float(os.getenv('CHAT_TEMPERATURE', '0.1'))
    
    # PostgreSQL Configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rag_metadata")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "rag_user")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "secure_password")
    
    # Unstructured chunking configuration
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
    IMAP_HOST: str = os.getenv('IMAP_HOST', '')
    IMAP_PORT: int = int(os.getenv('IMAP_PORT', '993'))
    IMAP_USERNAME: str = os.getenv('IMAP_USERNAME', '')
    IMAP_PASSWORD: str = os.getenv('IMAP_PASSWORD', '')
    IMAP_MAILBOX: str = os.getenv('IMAP_MAILBOX', 'INBOX')
    IMAP_BATCH_LIMIT: int = int(os.getenv('IMAP_BATCH_LIMIT', '50'))
    IMAP_USE_SSL: bool = os.getenv('IMAP_USE_SSL', 'true').lower() == 'true'
    EMAIL_SYNC_INTERVAL_SECONDS: int = int(os.getenv('EMAIL_SYNC_INTERVAL_SECONDS', '300'))
    EMAIL_DEFAULT_REFRESH_MINUTES: int = int(os.getenv('EMAIL_DEFAULT_REFRESH_MINUTES', '5'))

    # Document Hybrid Retrieval & Reranking Configuration
    ENABLE_DOCUMENT_RERANKING: bool = os.getenv('ENABLE_DOCUMENT_RERANKING', 'true').lower() == 'true'
    DOCUMENT_RERANKER_MODEL: str = os.getenv('DOCUMENT_RERANKER_MODEL', 'ms-marco-minilm')
    DOCUMENT_RERANK_TOP_K: Optional[int] = int(os.getenv('DOCUMENT_RERANK_TOP_K', '0')) or None  # 0 means rerank all
    DOCUMENT_RRF_CONSTANT: int = int(os.getenv('DOCUMENT_RRF_CONSTANT', '60'))
    
    # Advanced Document Chunking Configuration
    DOCUMENT_CHUNKING_STRATEGY: str = os.getenv('DOCUMENT_CHUNKING_STRATEGY', 'title_aware')  # title_aware, page_aware, basic
    DOCUMENT_TARGET_TOKENS: int = int(os.getenv('DOCUMENT_TARGET_TOKENS', '850'))  # Target ~800-1,000 tokens
    DOCUMENT_OVERLAP_PERCENTAGE: float = float(os.getenv('DOCUMENT_OVERLAP_PERCENTAGE', '0.125'))  # 10-15% overlap
    DOCUMENT_MAX_TOKENS: int = int(os.getenv('DOCUMENT_MAX_TOKENS', '1200'))  # Hard limit
    DOCUMENT_MIN_TOKENS: int = int(os.getenv('DOCUMENT_MIN_TOKENS', '100'))  # Minimum viable chunk size
    DOCUMENT_PRESERVE_TABLES: bool = os.getenv('DOCUMENT_PRESERVE_TABLES', 'true').lower() == 'true'
    DOCUMENT_TRACK_SECTIONS: bool = os.getenv('DOCUMENT_TRACK_SECTIONS', 'true').lower() == 'true'
    
    # PostgreSQL Migration Feature Flags
    USE_POSTGRESQL_URL_MANAGER: bool = os.getenv('USE_POSTGRESQL_URL_MANAGER', 'false').lower() == 'true'
    POSTGRES_MIGRATION_MODE: str = os.getenv('POSTGRES_MIGRATION_MODE', 'disabled')  # disabled, testing, enabled

    # URL Snapshot settings (always enabled for consistency)
    SNAPSHOT_DIR: str = os.getenv('SNAPSHOT_DIR', os.path.join('uploaded', 'snapshots'))
    SNAPSHOT_FORMATS: List[str] = field(
        default_factory=lambda: os.getenv('SNAPSHOT_FORMATS', 'pdf,mhtml').split(',')
    )
    SNAPSHOT_TIMEOUT_SECONDS: int = int(os.getenv('SNAPSHOT_TIMEOUT_SECONDS', '60'))
    
    # Playwright browser settings for snapshots
    SNAPSHOT_VIEWPORT_WIDTH: int = int(os.getenv('SNAPSHOT_VIEWPORT_WIDTH', '1920'))
    SNAPSHOT_VIEWPORT_HEIGHT: int = int(os.getenv('SNAPSHOT_VIEWPORT_HEIGHT', '1080'))
    SNAPSHOT_PDF_FORMAT: str = os.getenv('SNAPSHOT_PDF_FORMAT', 'A4')  # A4, Letter, etc.
    SNAPSHOT_LOCALE: str = os.getenv('SNAPSHOT_LOCALE', 'en-US')
    SNAPSHOT_EMULATE_MEDIA: str = os.getenv('SNAPSHOT_EMULATE_MEDIA', 'print')  # print, screen
