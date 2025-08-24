"""Email processing utilities for embedding and persistence.

This module defines :class:`EmailProcessor` which normalizes and stores
email records. It splits message bodies into smaller chunks using
``RecursiveCharacterTextSplitter`` and creates vector embeddings with a
configurable model (default :class:`OllamaEmbeddings`).

Embeddings are inserted into Milvus while the original email metadata is
persisted to PostgreSQL. Both the Milvus client and the PostgreSQL 
email manager are provided by the caller, keeping this processor free of any
application specific initialization.

Field Naming Convention:
All email processing uses consistent email protocol field names:
- from_addr: Email sender address  
- to_addrs: List of recipient addresses
- date_utc: Email date in UTC ISO format
- message_id: Unique email message identifier
- subject: Email subject line
"""

from __future__ import annotations

import logging
import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class EmailProcessor:
    """Process email records into embeddings and metadata rows.

    Parameters
    ----------
    milvus : Any
        A connected Milvus client or vector store supporting ``add_texts`` or
        ``insert`` style APIs.
    email_manager : Any
        PostgreSQL-based email manager for metadata persistence.
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
        email_manager: Any,  # PostgreSQL-based email manager
        *,
        embedding_model: Optional[Any] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.milvus = milvus
        self.email_manager = email_manager
        self.embedding_model = embedding_model or OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
            base_url=f"http://{os.getenv('OLLAMA_EMBEDDING_HOST', 'localhost')}:{os.getenv('OLLAMA_EMBEDDING_PORT', '11434')}"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        
        # Use the PostgreSQL-based email manager directly
        self.manager = email_manager
            
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
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        subject = record.get("subject", "")
        # Use consistent email protocol field names throughout
        from_addr = record.get("from_addr", "")
        date_utc = record.get("date_utc", "")

        # Build chunks with deterministic content hashes; dedupe within this batch 
        # (Removes empty content)
        seen_ids = set()
        prepared = []  # list of tuples (chunk_text, metadata_dict, id)
        for idx, chunk in enumerate(chunks):
            chunk = (chunk or "").strip()
            if not chunk:
                continue
            chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            if chunk_hash in seen_ids:
                continue
            seen_ids.add(chunk_hash)
            cid = f"{message_id}:{idx}"
            meta = {
                "message_id": message_id,
                "source": f"email:{from_addr}",
                "subject": subject,
                "from_addr": from_addr,  # Use email protocol field name
                "date_utc": date_utc,    # Use email protocol field name
                "chunk_id": cid,
                "page": idx,
                "content_hash": chunk_hash,
                "category_type": "email",  # Set category_type for email content
            }
            prepared.append((chunk, meta, chunk_hash))

        if not prepared:
            logger.debug("No chunks prepared for message %s", message_id)
            return

        # Unzip prepared lists
        chunks_to_add = [t for (t, _, _) in prepared]
        metadatas = [m for (_, m, _) in prepared]
        ids = [i for (_, _, i) in prepared]

        try:
            if hasattr(self.milvus, "add_texts"):
                try:
                    # Try batch insert first
                    self.milvus.add_texts(texts=chunks_to_add, metadatas=metadatas, ids=ids)
                except Exception as batch_exc:
                    # Fall back to per-chunk inserts, skipping duplicates by catching errors
                    logger.warning(
                        "Batch add_texts failed for message %s, attempting per-chunk with duplicate skip: %s",
                        message_id, batch_exc
                    )
                    inserted = 0
                    skipped_dups = 0
                    for t, m, i in prepared:
                        try:
                            self.milvus.add_texts(texts=[t], metadatas=[m], ids=[i])
                            inserted += 1
                        except Exception as ex:
                            skipped_dups += 1
                            logger.info("Duplicate or failed insert skipped for id=%s: %s", i, ex)
                            continue
                    logger.debug(
                        "Per-chunk add complete for message %s: %d inserted, %d skipped",
                        message_id, inserted, skipped_dups
                    )
            elif hasattr(self.milvus, "add_embeddings"):
                # No batch retry here; provider API may vary. Try once.
                self.milvus.add_embeddings(embeddings=list(embeddings), ids=ids, metadatas=metadatas)
            elif self.milvus is None:
                logger.warning("Milvus client is None, skipping embedding storage for %s", message_id)
                return
            else:
                raise RuntimeError(f"Unsupported Milvus client interface: {type(self.milvus)}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Milvus insertion failed for %s: %s", message_id, exc)
        else:
            logger.debug("Stored %d new embeddings for message %s", len(ids), message_id)

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
