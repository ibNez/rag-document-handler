"""Email processing utilities for embedding and persistence.

This module defines :class:`EmailProcessor` which normalizes and stores
email records. It splits message bodies into smaller chunks using
``RecursiveCharacterTextSplitter`` and creates vector embeddings with a
configurable model (default :class:`OllamaEmbeddings`).

Embeddings are inserted into Milvus while the original email metadata is
persisted to a SQLite database.  Both the Milvus client and the SQLite
connection are provided by the caller, keeping this processor free of any
application specific initialization.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from .email_manager import EmailManager

logger = logging.getLogger(__name__)


class EmailProcessor:
    """Process email records into embeddings and metadata rows.

    Parameters
    ----------
    milvus : Any
        A connected Milvus client or vector store supporting ``add_texts`` or
        ``insert`` style APIs.
    sqlite_conn : sqlite3.Connection
        Connection object for the metadata SQLite database.
    embedding_model : Optional[Any]
        Embedding model implementing ``embed_documents``. Defaults to
        :class:`OllamaEmbeddings` with the provider's defaults.
    chunk_size : int
        Character length for each text chunk before embedding.
    chunk_overlap : int
        Overlap between consecutive chunks.
    """

    def __init__(
        self,
        milvus: Any,
        sqlite_conn: sqlite3.Connection,
        *,
        embedding_model: Optional[Any] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.milvus = milvus
        self.sqlite_conn = sqlite_conn
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
            base_url=f"http://{os.getenv('OLLAMA_EMBEDDING_HOST', 'localhost')}:{os.getenv('OLLAMA_EMBEDDING_PORT', '11434')}"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.manager = EmailManager(sqlite_conn)
        self.skipped_messages = 0
        logger.info(
            "EmailProcessor initialized with embedding model %s",
            getattr(self.embedding_model, "model", self.embedding_model.__class__.__name__),
        )

    # ------------------------------------------------------------------
    def _store_metadata(self, record: Dict[str, Any]) -> None:
        """Persist an email record using :class:`EmailManager`."""
        self.manager.upsert_email(record)
        logger.info("Stored metadata for message %s", record.get("message_id"))

    # ------------------------------------------------------------------
    def _store_embeddings(
        self, message_id: str, chunks: List[str], embeddings: Iterable[List[float]], record: Dict[str, Any]
    ) -> None:
        """Insert embeddings into Milvus using the provided client."""
        logger.debug(
            "Storing %d embeddings for message %s", len(chunks), message_id
        )
        metadatas = []
        ids = []
        for idx, _ in enumerate(chunks):
            cid = f"{message_id}:{idx}"
            ids.append(cid)
            # Map email metadata to document schema fields
            subject = record.get("subject", "")
            from_addr = record.get("from_addr", "")
            date_utc = record.get("date_utc", "")
            
            meta = {
                "document_id": message_id,  # Use message_id as document_id
                "source": f"email:{from_addr}",  # Use from_addr as source
                "page": idx,  # Use chunk index as page number
                "chunk_id": cid,
                "topic": subject,  # Use subject as topic
                "category": "email",  # Fixed category for emails
                "content_hash": f"email_{message_id}_{idx}",  # Generate content hash
                "content_length": len(chunks[idx]) if idx < len(chunks) else 0,
                # Keep original email fields for backward compatibility
                "message_id": message_id,
                "subject": subject,
                "from_addr": from_addr,
                "to_addrs": record.get("to_addrs"),
                "date_utc": date_utc,
                "server_type": record.get("server_type"),
            }
            metadatas.append(meta)

        try:
            if hasattr(self.milvus, "add_texts"):
                # Preferred: Use LangChain interface which handles schema mapping
                self.milvus.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
            elif hasattr(self.milvus, "add_embeddings"):
                # Fallback: Use embeddings if available
                self.milvus.add_embeddings(
                    embeddings=list(embeddings), ids=ids, metadatas=metadatas
                )
            elif self.milvus is None:
                logger.warning("Milvus client is None, skipping embedding storage for %s", message_id)
                return
            else:
                raise RuntimeError(f"Unsupported Milvus client interface: {type(self.milvus)}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Milvus insertion failed for %s: %s", message_id, exc)
        else:
            logger.debug("Stored embeddings for message %s", message_id)

    # ------------------------------------------------------------------
    def process(self, record: Dict[str, Any]) -> None:
        """Process a single email record."""
        message_id = record.get("message_id")
        if not message_id:
            raise ValueError("record missing message_id")
        body_text = record.get("body_text") or ""

        # persist metadata regardless of body content
        self._store_metadata(record)

        if not body_text.strip():
            logger.debug(
                "Skipping message %s due to missing body text", message_id
            )
            self.skipped_messages += 1
            return

        chunks = [c for c in self.splitter.split_text(body_text) if c.strip()]
        if not chunks:
            logger.debug(
                "Skipping message %s because splitter produced no chunks",
                message_id,
            )
            self.skipped_messages += 1
            return
        model_name = getattr(
            self.embedding_model, "model", self.embedding_model.__class__.__name__
        )
        logger.info(
            "Generating embeddings for message %s with model %s", message_id, model_name
        )
        embeddings = self.embedding_model.embed_documents(chunks)
        logger.info(
            "Generated %d embeddings for message %s", len(chunks), message_id
        )
        self._store_embeddings(message_id, chunks, embeddings, record)
        logger.debug("Finished processing message %s", message_id)

    # ------------------------------------------------------------------
    def process_smart_batch(
        self,
        connector,
        since_date: Optional[Any] = None,
        max_batches: Optional[int] = None
    ) -> Dict[str, int]:
        """Process emails using smart batching to avoid duplicates.
        
        Parameters
        ----------
        connector:
            Email connector with fetch_smart_batch method
        since_date:
            Optional datetime to filter emails from
        max_batches:
            Maximum number of batches to process (None for unlimited)
            
        Returns
        -------
        Dict[str, int]:
            Statistics about the processing session
        """
        logger.info("Starting smart batch processing")
        
        stats = {
            "total_emails_processed": 0,
            "total_batches": 0,
            "skipped_duplicates": 0,
            "successful_embeddings": 0,
            "errors": 0
        }
        
        # The connector handles offset tracking internally
        start_offset = 0
        batch_count = 0
        
        while True:
            # Check batch limit
            if max_batches and batch_count >= max_batches:
                logger.info("Reached maximum batch limit of %d", max_batches)
                break
                
            # Fetch next smart batch - connector manages offset internally
            try:
                emails, has_more = connector.fetch_smart_batch(
                    email_manager=self.manager,
                    since_date=since_date,
                    start_offset=start_offset
                )
                
                if not emails:
                    if has_more:
                        # This shouldn't happen with our new implementation,
                        # but handle it just in case
                        logger.warning("No unique emails returned but has_more=True - this may indicate an issue")
                        start_offset += connector.batch_limit or 50
                        continue
                    else:
                        # No emails and no more available - we're done
                        logger.info("No more emails to process")
                        break
                        
                batch_count += 1
                stats["total_batches"] = batch_count
                
                logger.info(
                    "Processing batch %d: %d unique emails (has_more: %s)",
                    batch_count,
                    len(emails),
                    has_more
                )
                
                # Process each email in the batch
                batch_successes = 0
                for email in emails:
                    try:
                        self.process(email)
                        stats["total_emails_processed"] += 1
                        stats["successful_embeddings"] += 1
                        batch_successes += 1
                    except Exception as exc:
                        logger.error(
                            "Error processing email %s: %s",
                            email.get("message_id", "unknown"),
                            exc
                        )
                        stats["errors"] += 1
                        
                logger.info(
                    "Batch %d complete: %d/%d emails processed successfully",
                    batch_count,
                    batch_successes,
                    len(emails)
                )
                
                # Update offset based on what the connector processed
                # The connector tells us how far it got through the mailbox
                start_offset += len(emails)
                
                # Break if no more emails available
                if not has_more:
                    logger.info("Processed all available emails")
                    break
                    
            except Exception as exc:
                logger.error("Error in smart batch processing: %s", exc)
                stats["errors"] += 1
                break
                
        logger.info(
            "Smart batch processing complete: %d batches, %d emails processed, %d errors",
            stats["total_batches"],
            stats["total_emails_processed"],
            stats["errors"]
        )
        
        return stats
