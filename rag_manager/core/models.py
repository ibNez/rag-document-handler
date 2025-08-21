"""
Core data models for RAG Knowledgebase Manager.

This module contains dataclasses and models used throughout the application,
following the development rules for proper type annotations and documentation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union


@dataclass
class BaseProcessingStatus:
    """
    Base class for all processing status tracking.
    
    Attributes:
        identifier: Unique identifier for the item being processed
        status: Current processing status
        progress: Progress percentage (0-100)
        message: Human-readable status message
        start_time: When processing started
        end_time: When processing completed
        chunks_count: Number of chunks created
        error_details: Error information if processing failed
        title: Content title if extracted
    """
    identifier: str
    status: str = "pending"  # pending, processing, chunking, embedding, storing, completed, error
    progress: int = 0  # 0-100
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunks_count: int = 0
    error_details: Optional[str] = None
    title: Optional[str] = None


@dataclass
class DocumentProcessingStatus:
    """
    Status tracking for document file processing operations.
    
    Attributes:
        filename: Name of the document file being processed
        identifier: Unique identifier for the item being processed
        status: Current processing status
        progress: Progress percentage (0-100)
        message: Human-readable status message
        start_time: When processing started
        end_time: When processing completed
        chunks_count: Number of chunks created
        error_details: Error information if processing failed
        title: Document title if extracted
        file_size: Size of the file in bytes
        file_type: MIME type or file extension
        pages_count: Number of pages in the document
    """
    filename: str
    identifier: str = ""  # Will be set in __post_init__
    status: str = "pending"  # pending, processing, chunking, embedding, storing, completed, error
    progress: int = 0  # 0-100
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunks_count: int = 0
    error_details: Optional[str] = None
    title: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    pages_count: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Set identifier to filename after initialization."""
        if not self.identifier:
            self.identifier = self.filename


@dataclass
class URLProcessingStatus:
    """
    Status tracking for URL processing operations.
    
    Attributes:
        url: The URL being processed
        identifier: Unique identifier for the item being processed
        status: Current processing status
        progress: Progress percentage (0-100)
        message: Human-readable status message
        start_time: When processing started
        end_time: When processing completed
        chunks_count: Number of chunks created
        error_details: Error information if processing failed
        title: Content title if extracted
        domain: Domain of the URL
        content_type: Content type returned by the server
        response_code: HTTP response code
        content_length: Size of the content in bytes
    """
    url: str
    identifier: str = ""  # Will be set in __post_init__
    status: str = "pending"  # pending, processing, chunking, embedding, storing, completed, error
    progress: int = 0  # 0-100
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunks_count: int = 0
    error_details: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None
    content_type: Optional[str] = None
    response_code: Optional[int] = None
    content_length: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Set identifier to URL and extract domain after initialization."""
        if not self.identifier:
            self.identifier = self.url
        
        # Extract domain from URL if not provided
        if not self.domain and self.url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(self.url)
                self.domain = parsed.netloc
            except Exception:
                self.domain = "unknown"


@dataclass
class EmailProcessingStatus:
    """
    Status tracking for email processing operations.
    
    Attributes:
        email_id: Unique email identifier
        identifier: Unique identifier for the item being processed
        status: Current processing status
        progress: Progress percentage (0-100)
        message: Human-readable status message
        start_time: When processing started
        end_time: When processing completed
        chunks_count: Number of chunks created
        error_details: Error information if processing failed
        title: Content title if extracted
        sender: Email sender address
        subject: Email subject line
        received_date: When the email was received
        attachments_count: Number of attachments processed
        email_size: Size of the email in bytes
    """
    email_id: str
    identifier: str = ""  # Will be set in __post_init__
    status: str = "pending"  # pending, processing, chunking, embedding, storing, completed, error
    progress: int = 0  # 0-100
    message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chunks_count: int = 0
    error_details: Optional[str] = None
    title: Optional[str] = None
    sender: Optional[str] = None
    subject: Optional[str] = None
    received_date: Optional[datetime] = None
    attachments_count: int = 0
    email_size: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Set identifier to email_id and title to subject after initialization."""
        if not self.identifier:
            self.identifier = self.email_id
        
        # Use subject as title if title not provided
        if not self.title and self.subject:
            self.title = self.subject


# Type alias for any processing status
ProcessingStatus = Union[DocumentProcessingStatus, URLProcessingStatus, EmailProcessingStatus]


@dataclass
class ChunkMetadata:
    """
    Metadata for document chunks stored in vector database.
    
    Attributes:
        source: Original source identifier (filename, URL, email_id)
        source_type: Type of source (document, url, email)
        chunk_index: Index of this chunk in the document
        total_chunks: Total number of chunks in the document
        title: Document/content title
        page_number: Page number for document chunks
        created_at: When the chunk was created
        file_type: Original file type/format
        chunk_size: Size of the chunk in characters
    """
    source: str
    source_type: str  # "document", "url", "email"
    chunk_index: int
    total_chunks: int
    title: Optional[str] = None
    page_number: Optional[int] = None
    created_at: Optional[datetime] = None
    file_type: Optional[str] = None
    chunk_size: Optional[int] = None


@dataclass
class SearchResult:
    """
    Result from similarity search in vector database.
    
    Attributes:
        content: The chunk content
        metadata: ChunkMetadata
        score: Similarity score
        source: Original source identifier
        title: Document/content title
        page_number: Page number if applicable
    """
    content: str
    metadata: ChunkMetadata
    score: float
    source: str
    title: Optional[str] = None
    page_number: Optional[int] = None
