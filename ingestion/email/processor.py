"""Email processing utilities for embedding and persistence.

This module defines :class:`EmailProcessor` which normalizes and stores
email records. It splits message bodies into smaller chunks using
``RecursiveCharacterTextSplitter`` and creates vector embeddings with a
configurable model (default :class:`OllamaEmbeddings`).

Embeddings are inserted into Milvus while the original email metadata and
chunks are persisted to PostgreSQL. Both the Milvus client and the PostgreSQL 
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from ingestion.utils.chunker import TextChunker
from rag_manager.managers.postgres_manager import PostgreSQLManager

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
        
        # Initialize text chunker for PostgreSQL chunk storage
        # Convert character-based chunks to token-based for consistency
        # Rough estimate: 1 token ≈ 4 characters for English text
        token_chunk_size = max(1, chunk_size // 4)
        token_overlap = max(0, chunk_overlap // 4)
        
        self.text_chunker = TextChunker(
            chunk_size_tokens=token_chunk_size,
            chunk_overlap_tokens=token_overlap
        )
        
        # Initialize PostgreSQL manager for chunk storage
        self.db_manager = email_manager.db_manager
        
        # Use the PostgreSQL-based email manager directly
        self.manager = email_manager
            
        self.skipped_messages = 0
        logger.info(
            "EmailProcessor initialized with embedding model %s",
            getattr(self.embedding_model, "model", self.embedding_model.__class__.__name__),
        )

    # ------------------------------------------------------------------
    def _store_metadata(self, record: Dict[str, Any]) -> str:
        """Persist an email record using :class:`EmailManager`.
        
        Returns:
            The database-generated UUID id for the stored email
        """
        # Ensure required fields are present and generate missing hashes
        if not record.get("header_hash"):
            # Generate header hash from message_id and headers
            header_data = f"{record.get('message_id', '')}{record.get('subject', '')}{record.get('from_addr', '')}"
            record["header_hash"] = hashlib.sha256(header_data.encode('utf-8')).hexdigest()
        
        if not record.get("content_hash"):
            # Generate content hash from body text
            content = record.get("content", "")
            record["content_hash"] = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Store the email and get the database ID directly from upsert
        email_db_id = self.manager.upsert_email(record)
        logger.info("Stored metadata for message %s with ID %s", record.get("message_id"), email_db_id)
        
        return email_db_id

    # ------------------------------------------------------------------
    def _store_chunks(self, record: Dict[str, Any]) -> int:
        """Store email chunks in PostgreSQL for retrieval."""
        message_id = record.get("message_id")
        content = record.get("content", "")
        
        # DEBUG_EMAIL_ID: Log _store_chunks entry state
        logger.debug(f"DEBUG_EMAIL_ID: _store_chunks called for message_id: {message_id}")
        logger.debug(f"DEBUG_EMAIL_ID: record['id'] = {record.get('id')} (type: {type(record.get('id'))})")
        logger.debug(f"DEBUG_EMAIL_ID: record keys = {list(record.keys())}")
        
        if not content.strip():
            logger.debug("No content to chunk for message %s", message_id)
            return 0
        
        try:
            # Generate chunks using the text chunker
            logger.debug(f"DEBUG_EMAIL_ID: About to call text_chunker.chunk_email")
            chunks = self.text_chunker.chunk_email(record)
            logger.debug(f"DEBUG_EMAIL_ID: text_chunker returned {len(chunks) if chunks else 0} chunks")
            
            if not chunks:
                logger.debug("No chunks generated for message %s", message_id)
                return 0
            
            # Store chunks in PostgreSQL
            stored_count = 0
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    for chunk in chunks:
                        try:
                            # DEBUG_EMAIL_ID: Log chunk details before insertion
                            logger.debug(f"DEBUG_EMAIL_ID: Processing chunk {chunk.get('chunk_index', 'unknown')}")
                            logger.debug(f"DEBUG_EMAIL_ID: chunk['email_id'] = {chunk.get('email_id')} (type: {type(chunk.get('email_id'))})")
                            logger.debug(f"DEBUG_EMAIL_ID: chunk keys = {list(chunk.keys())}")
                            
                            if chunk.get('email_id') is None:
                                logger.error(f"DEBUG_EMAIL_ID: FOUND NULL EMAIL_ID! Record 'id' = {record.get('id')}")
                                logger.error(f"DEBUG_EMAIL_ID: Original record keys = {list(record.keys())}")
                                
                            # Insert and return the generated UUID id for the chunk.
                            cur.execute("""
                                INSERT INTO email_chunks (
                                    email_id, chunk_text, chunk_index, 
                                    token_count, chunk_hash
                                )
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (email_id, chunk_index) DO UPDATE SET
                                    chunk_text = EXCLUDED.chunk_text,
                                    token_count = EXCLUDED.token_count,
                                    chunk_hash = EXCLUDED.chunk_hash,
                                    created_at = CURRENT_TIMESTAMP
                                RETURNING id
                            """, (
                                chunk['email_id'],
                                chunk['chunk_text'],
                                chunk['chunk_index'],
                                chunk['token_count'],
                                chunk['chunk_hash']
                            ))
                            fetched = cur.fetchone()
                            if fetched:
                                chunk_id = str(fetched.get('id') or '')
                                chunk['email_chunk_id'] = chunk_id
                            stored_count += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to store chunk {chunk.get('email_chunk_id')}: {e}")
                            continue
                
                conn.commit()
                logger.info(f"Stored {stored_count} chunks for message {message_id}")
                
        except Exception as e:
            logger.error(f"Failed to store chunks for message {message_id}: {e}")
            return 0
        
        return stored_count

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
                # Email-specific fields (clean schema without redundancy)
                "message_id": message_id,   # Primary email identifier
                "source": f"email:{from_addr}",
                "subject": subject,
                "from_addr": from_addr,     # Email protocol field name
                "date_utc": date_utc,       # Email protocol field name
                "chunk_id": cid,
                "page": idx,
                "content_hash": chunk_hash,
                "category_type": "email",
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
                    # Don't pass IDs when auto_id=True - let Milvus generate primary keys automatically
                    self.milvus.add_texts(texts=chunks_to_add, metadatas=metadatas)
                except Exception as batch_exc:
                    # Fall back to per-chunk inserts, skipping duplicates by catching errors
                    logger.warning(
                        "Batch add_texts failed for message %s, attempting per-chunk with duplicate skip: %s",
                        message_id, batch_exc
                    )
                    inserted = 0
                    skipped_dups = 0
                    for t, m in zip(chunks_to_add, metadatas):
                        try:
                            # Don't pass IDs in per-chunk inserts either
                            self.milvus.add_texts(texts=[t], metadatas=[m])
                            inserted += 1
                        except Exception as ex:
                            skipped_dups += 1
                            logger.info("Duplicate or failed insert skipped: %s", ex)
                            continue
                    logger.debug(
                        "Per-chunk add complete for message %s: %d inserted, %d skipped",
                        message_id, inserted, skipped_dups
                    )
            elif hasattr(self.milvus, "add_embeddings"):
                # Don't pass IDs when auto_id=True - let Milvus generate primary keys automatically
                self.milvus.add_embeddings(embeddings=list(embeddings), metadatas=metadatas)
            elif self.milvus is None:
                logger.warning("Milvus client is None, skipping embedding storage for %s", message_id)
                return
            else:
                raise RuntimeError(f"Unsupported Milvus client interface: {type(self.milvus)}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Milvus insertion failed for %s: %s", message_id, exc)
        else:
            logger.debug("Stored %d new embeddings for message %s", len(chunks_to_add), message_id)

    # ------------------------------------------------------------------
    def process(self, record: Dict[str, Any]) -> None:
        """Process a single email record."""
        message_id = record.get("message_id")
        if not message_id:
            raise ValueError("record missing message_id")
        
        # DEBUG_EMAIL_ID: Log initial record state
        logger.debug(f"DEBUG_EMAIL_ID: Starting process for message_id: {message_id}")
        logger.debug(f"DEBUG_EMAIL_ID: Initial record keys: {list(record.keys())}")
        logger.debug(f"DEBUG_EMAIL_ID: 'id' in record: {'id' in record}")
        
        # Check for content 
        body_text = record.get("content", "")

        # Persist metadata and get the database UUID id
        email_db_id = self._store_metadata(record)
        logger.debug(f"DEBUG_EMAIL_ID: _store_metadata returned: {email_db_id} (type: {type(email_db_id)})")
        
        # Add database ID to record for chunking
        record['id'] = email_db_id
        logger.debug(f"DEBUG_EMAIL_ID: Set record['id'] = {record['id']}")
        
        # Store chunks in PostgreSQL for retrieval
        logger.debug(f"DEBUG_EMAIL_ID: About to call _store_chunks with record['id'] = {record.get('id')}")
        chunk_count = self._store_chunks(record)
        logger.debug(f"DEBUG_EMAIL_ID: _store_chunks returned {chunk_count} chunks")

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
        logger.debug("Finished processing message %s (%d chunks stored)", message_id, chunk_count)

    # ------------------------------------------------------------------
    def process_smart_batch(
        self,
        connector,
        since_date: Optional[Any] = None,
        max_batches: Optional[int] = None,
        start_offset: int = 0
    ) -> Dict[str, int]:
        """Process emails using smart batching with constraint failure handling.
        
        Parameters
        ----------
        connector:
            Email connector with fetch_smart_batch method
        since_date:
            Optional datetime to filter emails from
        max_batches:
            Maximum number of batches to process (None for unlimited)
        start_offset:
            Offset to start processing from (for pagination)
            
        Returns
        -------
        Dict[str, int]:
            Statistics about the processing session including final_offset
        """
        logger.info("Starting smart batch processing from offset %d", start_offset)
        
        stats = {
            "total_emails_processed": 0,
            "total_batches": 0,
            "skipped_duplicates": 0,
            "successful_embeddings": 0,
            "errors": 0,
            "final_offset": start_offset
        }
        
        batch_count = 0
        current_offset = start_offset
        
        while True:
            # Check batch limit
            if max_batches and batch_count >= max_batches:
                logger.info("Reached maximum batch limit of %d", max_batches)
                break
                
            # Fetch and process emails with constraint failure handling
            try:
                emails_processed, new_offset, has_more = self._process_batch_with_constraint_handling(
                    connector=connector,
                    since_date=since_date,
                    start_offset=current_offset,
                    target_batch_size=connector.batch_limit or 50
                )
                
                if emails_processed == 0 and not has_more:
                    logger.info("No more emails to process")
                    break
                        
                batch_count += 1
                stats["total_batches"] = batch_count
                stats["total_emails_processed"] += emails_processed
                stats["successful_embeddings"] += emails_processed
                stats["final_offset"] = new_offset
                current_offset = new_offset
                
                logger.info(
                    "Batch %d complete: %d emails processed, offset now at %d",
                    batch_count,
                    emails_processed,
                    current_offset
                )
                
                # Break if no more emails available
                if not has_more:
                    logger.info("Processed all available emails")
                    break
                    
            except Exception as exc:
                logger.error("Error in smart batch processing: %s", exc)
                stats["errors"] += 1
                break
                
        logger.info(
            "Smart batch processing complete: %d batches, %d emails processed, final offset: %d",
            stats["total_batches"],
            stats["total_emails_processed"],
            stats["final_offset"]
        )
        
        return stats

    def _process_batch_with_constraint_handling(
        self,
        connector,
        since_date: Optional[Any],
        start_offset: int,
        target_batch_size: int
    ) -> Tuple[int, int, bool]:
        """Process a batch of emails with constraint failure handling.
        
        Returns:
            Tuple of (emails_processed, final_offset, has_more)
        """
        emails_processed = 0
        current_offset = start_offset
        batch_size_multiplier = 1  # Start with 1x batch size, increase if many duplicates
        has_more = True  # Initialize to handle edge cases
        
        while emails_processed < target_batch_size:
            # Fetch emails from connector
            fetch_size = min(target_batch_size * batch_size_multiplier, target_batch_size * 3)
            emails, has_more, total_emails = connector.fetch_smart_batch(
                email_manager=self.manager,
                since_date=since_date,
                start_offset=current_offset,
                fetch_size=fetch_size  # Fetch more to account for duplicates
            )
            
            # Update the total emails count in the email manager (if available)
            if hasattr(self.manager, 'update_total_emails_in_mailbox'):
                try:
                    # Get account info to update the total
                    accounts = self.manager.list_accounts()
                    if accounts:
                        account = accounts[0]  # Assuming we're processing one account at a time
                        self.manager.update_total_emails_in_mailbox(account['id'], total_emails)
                except Exception as e:
                    logger.warning(f"Failed to update total emails count: {e}")
            
            if not emails:
                logger.info("No more emails available at offset %d", current_offset)
                return emails_processed, current_offset, has_more
            
            # Process each email and handle constraint failures
            batch_successes = 0
            
            for email in emails:
                if emails_processed >= target_batch_size:
                    break
                    
                # Additional validation before processing
                if email is None:
                    logger.warning("Skipping None email in batch processing")
                    current_offset += 1
                    continue
                    
                if not isinstance(email, dict):
                    logger.warning("Skipping non-dict email in batch processing: %r", type(email))
                    current_offset += 1
                    continue
                    
                try:
                    # Try to process the email - this will hit constraint if duplicate
                    self.process(email)
                    batch_successes += 1
                    emails_processed += 1
                    logger.debug("Successfully processed email %s", email.get("message_id", "unknown"))
                    
                except Exception as exc:
                    # Check if this is a constraint violation (duplicate)
                    if "duplicate key value" in str(exc).lower() or "unique constraint" in str(exc).lower():
                        logger.debug("Skipped duplicate email %s", email.get("message_id", "unknown") if email else "NULL_EMAIL")
                        # Continue to next email - this one was a duplicate
                    else:
                        logger.error("Error processing email %s: %s", email.get("message_id", "unknown") if email else "NULL_EMAIL", exc)
                        # Continue processing other emails even if one fails
                
                current_offset += 1
            
            logger.debug(
                "Processed batch: %d successes, %d processed so far, offset at %d",
                batch_successes,
                emails_processed,
                current_offset
            )
            
            # If we got very few successes, increase fetch size for next iteration
            if batch_successes < len(emails) // 2:
                batch_size_multiplier = min(batch_size_multiplier * 2, 3)
                logger.debug("Many duplicates detected, increasing fetch multiplier to %d", batch_size_multiplier)
            
            if not has_more:
                break
                
        return emails_processed, current_offset, has_more
