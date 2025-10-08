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
import json
import shutil
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from ingestion.utils.chunker import TextChunker
from rag_manager.managers.postgres_manager import PostgreSQLManager
from rag_manager.core.config import Config

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
        
        # Initialize configuration
        self.config = Config()
        
        # Initialize LLM for topic extraction (using existing approach from document processor)
        self.llm = ChatOllama(
            model=self.config.CLASSIFICATION_MODEL,
            base_url=self.config.CLASSIFICATION_BASE_URL,
            temperature=0.1  # Low temperature for consistent topic extraction
        )
        
        # Ensure email attachments directory exists
        os.makedirs(self.config.EMAIL_ATTACHMENTS_DIR, exist_ok=True)
        
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
        
        # Use the EmailDataManager directly instead of the wrapper
        self.email_data_manager = email_manager.email_data_manager
        
        # Keep reference to account manager for account-related operations
        self.manager = email_manager
            
        self.skipped_messages = 0
        logger.info(
            "EmailProcessor initialized with embedding model %s",
            getattr(self.embedding_model, "model", self.embedding_model.__class__.__name__),
        )

    # ------------------------------------------------------------------
    def _store_attachments(self, record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Store email attachments to disk and return metadata.
        
        Args:
            record: Email record containing attachment data
            
        Returns:
            List of attachment metadata dictionaries, or None if no attachments
        """
        attachments_data = record.get("attachments_data", [])
        if not attachments_data:
            return None
        
        message_id = record.get("message_id", "unknown")
        stored_attachments = []
        
        for i, attachment in enumerate(attachments_data):
            try:
                original_filename = attachment.get("filename", f"attachment_{i}")
                content_type = attachment.get("content_type", "application/octet-stream")
                attachment_data = attachment.get("data")
                
                if not attachment_data:
                    logger.warning(f"No data for attachment {original_filename} in message {message_id}")
                    continue
                
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = "".join(c for c in original_filename if c.isalnum() or c in "._-")
                stored_filename = f"att_{timestamp}_{i}_{safe_filename}"
                file_path = os.path.join(self.config.EMAIL_ATTACHMENTS_DIR, stored_filename)
                
                # Write attachment to disk
                with open(file_path, "wb") as f:
                    f.write(attachment_data)
                
                # Calculate file hash
                file_hash = hashlib.sha256(attachment_data).hexdigest()
                
                # Create attachment metadata
                attachment_metadata = {
                    "original_filename": original_filename,
                    "stored_filename": stored_filename,
                    "file_path": file_path,
                    "content_type": content_type,
                    "file_size": len(attachment_data),
                    "attachment_id": f"part_{i}",
                    "sha256_hash": file_hash,
                    "created_at": datetime.now().isoformat()
                }
                
                stored_attachments.append(attachment_metadata)
                logger.info(f"Stored attachment {original_filename} as {stored_filename} for message {message_id}")
                
            except Exception as e:
                logger.error(f"Failed to store attachment {i} for message {message_id}: {e}")
                continue
        
        return stored_attachments if stored_attachments else None

    def _extract_topics(self, record: Dict[str, Any]) -> Optional[str]:
        """
        Extract topics from email content using LLM.
        
        Args:
            record: Email record with subject and content
            
        Returns:
            Comma-separated topics string, or None if extraction fails
        """
        try:
            subject = record.get("subject", "")
            content = record.get("content", "")
            from_addr = record.get("from_addr")
            
            # Skip topic extraction for very short emails
            combined_content = f"{subject} {content}".strip()
            if len(combined_content) < 50:
                logger.debug("Skipping topic extraction for very short email")
                return None
            
            # Build prompt using the same approach as document processor
            topic_prompt = f"""Extract 3-8 topics that users might search for to find this email.

Think about different ways people would search for this information:
- Specific technical concepts and terms
- Broader application domains and fields
- Industry or academic areas
- Alternative terminology and synonyms
- Practical applications and use cases

Return ONLY a comma-separated list of topics (no JSON, quotes, or extra formatting):
Example: Machine Learning, Fraud Detection, Financial Security, Neural Networks, Banking Technology

Content type: email
Subject: {subject}
From: {from_addr or 'Unknown'}
Content: {content[:1200]}

Topics:"""
            
            # Call LLM using existing approach
            response = self.llm.invoke(topic_prompt)
            topics_text = str(response.content).strip()
            
            # Clean the response (same logic as document processor)
            topics_text = topics_text.replace('"', '').replace("'", '').strip()
            
            # Handle cases where LLM returns formatted text
            if topics_text.startswith('Topics:'):
                topics_text = topics_text.replace('Topics:', '').strip()
            
            # Split and clean individual topics
            topic_list = [t.strip() for t in topics_text.split(',') if t.strip()]
            
            # Limit to prevent topic explosion and filter quality
            quality_topics = []
            for topic in topic_list[:8]:  # Max 8 topics
                # Basic quality filters
                if (len(topic) > 2 and len(topic) < 50 and 
                    not topic.lower() in ['the', 'and', 'or', 'but', 'data', 'analysis', 'content']):
                    quality_topics.append(topic)
            
            if quality_topics:
                final_topics = ', '.join(quality_topics)
                logger.debug(f"Extracted topics for email: {final_topics}")
                return final_topics
            else:
                logger.debug("No quality topics extracted")
                return None
            
        except Exception as e:
            logger.error(f"Failed to extract topics for email: {e}")
            return None

    # ------------------------------------------------------------------
    def _store_metadata(self, record: Dict[str, Any], account_id: Optional[str] = None) -> str:
        """Persist an email record using :class:`EmailManager`.
        
        Args:
            record: Email record dictionary
            account_id: Optional account ID to associate the email with
        
        Returns:
            The database-generated UUID id for the stored email
        """
        # Ensure required fields are present and generate missing hashes
        if not record.get("header_hash"):
            # Generate header hash from message_id and headers
            header_data = f"{record.get('message_id', '')}{record.get('subject', '')}{record.get('from_addr', '')}"
            record["header_hash"] = hashlib.sha256(header_data.encode('utf-8')).hexdigest()
        
        if not record.get("content_hash"):
            # Generate comprehensive content hash from multiple unique fields
            # This prevents legitimate emails from colliding while catching true duplicates
            content = record.get("content", "")
            message_id = record.get("message_id", "")
            sender = record.get("from_addr", "") or record.get("sender", "")
            subject = record.get("subject", "")
            received_date = record.get("received_date", "")
            
            # Combine multiple fields to create a unique hash
            # This approach ensures emails with identical content but different metadata
            # (like newsletters sent to different people) are treated as separate entries
            hash_data = f"{message_id}|{sender}|{subject}|{received_date}|{content}"
            record["content_hash"] = hashlib.sha256(hash_data.encode('utf-8')).hexdigest()
            
            logger.debug(f"Generated content hash for message {message_id[:20]}... using multi-field approach")
        
        # Store attachments and get metadata
        attachment_files = self._store_attachments(record)
        if attachment_files:
            record["attachment_files"] = json.dumps(attachment_files)
            logger.info(f"Stored {len(attachment_files)} attachments for message {record.get('message_id')}")
        
        # Extract topics using LLM
        topics = self._extract_topics(record)
        if topics:
            record["topics"] = topics
            logger.info(f"Extracted topics for message {record.get('message_id')}: {topics}")
        
        # Store the email and get the database ID directly from EmailDataManager
        email_db_id = self.email_data_manager.upsert_email(record, account_id)
        logger.info("Stored metadata for message %s with ID %s", record.get("message_id"), email_db_id)
        
        return email_db_id

    # ------------------------------------------------------------------
    def _store_chunks(self, record: Dict[str, Any]) -> int:
        """Store email chunks in PostgreSQL for retrieval."""
        message_id = record.get("message_id")
        content = record.get("content", "")
        content_length = len(content)
        
        logger.info(f"📝 CHUNK_START: Starting chunking for message_id={message_id}, content_length={content_length}")
        
        if not content.strip():
            logger.info(f"⏭️ CHUNK_SKIP: No content to chunk for message_id={message_id}")
            return 0
        
        try:
            # Generate chunks using the text chunker
            logger.info(f"🔨 CHUNK_GENERATE: Calling text chunker for message_id={message_id}")
            chunks = self.text_chunker.chunk_email(record)
            chunk_count = len(chunks) if chunks else 0
            logger.info(f"📑 CHUNK_GENERATED: Generated {chunk_count} chunks for message_id={message_id}")
            
            if not chunks:
                logger.info(f"⏭️ CHUNK_EMPTY: No chunks generated for message_id={message_id}")
                return 0
            
            # Store chunks in PostgreSQL
            stored_count = 0
            logger.info(f"💾 CHUNK_STORE_START: Storing {len(chunks)} chunks in PostgreSQL")
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    for i, chunk in enumerate(chunks):
                        try:
                            logger.info(f"💾 CHUNK_STORE: Storing chunk {i+1}/{len(chunks)} for message_id={message_id}")
                            
                            # Validate chunk email_id before storing
                            chunk_email_id = chunk.get('email_id')
                            if not chunk_email_id or chunk_email_id == "":
                                logger.error(f"❌ CHUNK_SKIP: Chunk has invalid email_id '{chunk_email_id}' for message_id={message_id}. Skipping chunk storage.")
                                continue
                            
                            # Double-check the email_id is a valid UUID format
                            try:
                                import uuid
                                uuid.UUID(chunk_email_id)  # This will raise ValueError if not a valid UUID
                            except ValueError:
                                logger.error(f"❌ CHUNK_SKIP: Invalid UUID format for email_id '{chunk_email_id}' for message_id={message_id}. Skipping chunk storage.")
                                continue
                                
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
        logger.info(f"🔮 EMBED_START: Storing {len(chunks)} embeddings for message_id={message_id} in Milvus")
        
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        subject = record.get("subject", "")
        # Use consistent email protocol field names throughout
        from_addr = record.get("from_addr", "")
        date_utc = record.get("date_utc", "")
        # Normalize date for Milvus schema (expects VARCHAR). Keep orchestrator logic unaffected
        try:
            from datetime import datetime as _dt
            if isinstance(date_utc, _dt):
                date_utc = date_utc.isoformat()
        except Exception:
            pass

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
                        "❌ EMBED_BATCH_FAILED: Batch add_texts failed for message_id=%s, attempting per-chunk: %s",
                        message_id, batch_exc
                    )
                    inserted = 0
                    skipped_dups = 0
                    logger.info(f"🔄 EMBED_RETRY: Attempting per-chunk insertion for {len(chunks_to_add)} chunks")
                    for i, (t, m) in enumerate(zip(chunks_to_add, metadatas)):
                        try:
                            # Don't pass IDs in per-chunk inserts either
                            self.milvus.add_texts(texts=[t], metadatas=[m])
                            inserted += 1
                            logger.info(f"✅ EMBED_CHUNK: Inserted chunk {i+1}/{len(chunks_to_add)}")
                        except Exception as ex:
                            skipped_dups += 1
                            logger.info(f"⏭️ EMBED_SKIP: Duplicate or failed insert chunk {i+1}/{len(chunks_to_add)}: {ex}")
                            continue
                    logger.info(f"📊 EMBED_RETRY_SUMMARY: message_id={message_id} - {inserted} inserted, {skipped_dups} skipped")
            elif hasattr(self.milvus, "add_embeddings"):
                # Don't pass IDs when auto_id=True - let Milvus generate primary keys automatically
                logger.info(f"🔮 EMBED_DIRECT: Using add_embeddings for {len(list(embeddings))} embeddings")
                self.milvus.add_embeddings(embeddings=list(embeddings), metadatas=metadatas)
            elif self.milvus is None:
                logger.warning("Milvus client is None, skipping embedding storage for %s", message_id)
                return
            elif hasattr(self.milvus, '__class__') and '_NoopMilvus' in str(type(self.milvus)):
                logger.debug("Using NoopMilvus, skipping embedding storage for %s", message_id)
                return
            else:
                raise RuntimeError(f"Unsupported Milvus client interface: {type(self.milvus)}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Milvus insertion failed for %s: %s", message_id, exc)
        else:
            logger.debug("Stored %d new embeddings for message %s", len(chunks_to_add), message_id)

    # ------------------------------------------------------------------
    def process(self, record: Dict[str, Any], account_id: Optional[str] = None) -> None:
        """Process a single email record."""
        message_id = record.get("message_id")
        if not message_id:
            raise ValueError("record missing message_id")
        
        # DEBUG_EMAIL_ID: Log initial record state
        logger.debug(f"DEBUG_EMAIL_ID: Starting process for message_id: {message_id}")
        logger.debug(f"DEBUG_EMAIL_ID: Initial record keys: {list(record.keys())}")
        
        # Check for content FIRST - skip empty emails but log for investigation
        content = record.get("content", "")
        subject = record.get("subject", "")
        
        # Skip emails with no useful content for RAG processing
        if not content.strip() and not subject.strip():
            logger.warning(f"Skipping empty email Message-ID: {message_id} - no content or subject for RAG processing")
            self.skipped_messages += 1
            return
        
        # If only subject but no content, use subject as content
        if not content.strip() and subject.strip():
            record["content"] = subject
            logger.debug(f"Using subject as content for Message-ID: {message_id}")
        
        # Persist metadata and get the database UUID id
        email_db_id = self._store_metadata(record, account_id)
        logger.debug(f"DEBUG_EMAIL_ID: _store_metadata returned: {email_db_id} (type: {type(email_db_id)})")
        
        # Validate email_db_id before proceeding
        if not email_db_id or email_db_id == "":
            logger.error(f"Failed to get valid email_db_id for message {message_id}. Skipping chunking and embedding.")
            return
        
        # Add database ID to record for chunking
        record['id'] = email_db_id
        logger.debug(f"DEBUG_EMAIL_ID: Set record['id'] = {record['id']}")
        
        # Store chunks in PostgreSQL for retrieval
        logger.debug(f"DEBUG_EMAIL_ID: About to call _store_chunks with record['id'] = {record.get('id')}")
        chunk_count = self._store_chunks(record)
        logger.debug(f"DEBUG_EMAIL_ID: _store_chunks returned {chunk_count} chunks")

        body_text = record.get("content", "")
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
                    target_batch_size=connector.batch_size or 50
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
        logger.info(f"📧 EMAIL_BATCH_START: Processing batch - start_offset={start_offset}, target_batch_size={target_batch_size}")
        
        emails_processed = 0
        current_offset = start_offset
        batch_size_multiplier = 1  # Start with 1x batch size, increase if many duplicates
        has_more = True  # Initialize to handle edge cases
        
        while emails_processed < target_batch_size:
            # Fetch emails from connector
            fetch_size = min(target_batch_size * batch_size_multiplier, target_batch_size * 3)
            logger.info(f"🔍 EMAIL_FETCH: Fetching emails - fetch_size={fetch_size}, current_offset={current_offset}")
            emails, has_more, total_emails = connector.fetch_smart_batch(
                email_manager=self.manager,
                since_date=since_date,
                start_offset=current_offset,
                fetch_size=fetch_size  # Fetch more to account for duplicates
            )
            
            logger.info(f"📬 EMAIL_FETCH_RESULT: Retrieved {len(emails)} emails, has_more={has_more}, total_emails={total_emails}")
            
            # Update the total emails count in the email manager (if available)
            if hasattr(self.email_data_manager, 'update_total_emails_in_mailbox'):
                try:
                    # Get account info to update the total
                    accounts = self.manager.list_accounts()
                    if accounts:
                        account = accounts[0]  # Assuming we're processing one account at a time
                        self.email_data_manager.update_total_emails_in_mailbox(account['email_account_id'], total_emails)
                        logger.info(f"📊 EMAIL_COUNT_UPDATE: Updated total emails count to {total_emails}")
                except Exception as e:
                    logger.warning(f"Failed to update total emails count: {e}")
            
            if not emails:
                logger.info(f"⏹️ EMAIL_FETCH_EMPTY: No more emails available at offset {current_offset}")
                return emails_processed, current_offset, has_more
            
            # Process each email and handle constraint failures
            batch_successes = 0
            logger.info(f"🔄 EMAIL_PROCESSING_START: Processing {len(emails)} emails from batch")
            
            for i, email in enumerate(emails):
                if emails_processed >= target_batch_size:
                    logger.info(f"🎯 EMAIL_TARGET_REACHED: Reached target batch size {target_batch_size}")
                    break
                    
                # Additional validation before processing
                if email is None:
                    logger.warning("⚠️ EMAIL_VALIDATION: Skipping None email in batch processing")
                    current_offset += 1
                    continue
                    
                if not isinstance(email, dict):
                    logger.warning(f"⚠️ EMAIL_VALIDATION: Skipping non-dict email in batch processing: {type(email)}")
                    current_offset += 1
                    continue
                    
                try:
                    # Try to process the email - this will hit constraint if duplicate
                    self.process(email)
                    batch_successes += 1
                    emails_processed += 1
                    logger.info(f"✅ EMAIL_PROCESSED: Successfully processed email {i+1}/{len(emails)} - message_id={email.get('message_id', 'unknown')}")
                    
                except Exception as exc:
                    # Check if this is a constraint violation (duplicate)
                    if "duplicate key value" in str(exc).lower() or "unique constraint" in str(exc).lower():
                        logger.info(f"⏭️ EMAIL_DUPLICATE: Skipped duplicate email {i+1}/{len(emails)} - message_id={email.get('message_id', 'unknown') if email else 'NULL_EMAIL'}")
                        # Continue to next email - this one was a duplicate
                    else:
                        logger.error(f"❌ EMAIL_ERROR: Error processing email {i+1}/{len(emails)} - message_id={email.get('message_id', 'unknown') if email else 'NULL_EMAIL'}: {exc}")
                        # Continue processing other emails even if one fails
                
                current_offset += 1
            
            logger.info(f"📊 EMAIL_BATCH_SUMMARY: Batch complete - {batch_successes} successes, {emails_processed} total processed, offset now at {current_offset}")
            
            # If we got very few successes, increase fetch size for next iteration
            if batch_successes < len(emails) // 2:
                batch_size_multiplier = min(batch_size_multiplier * 2, 3)
                logger.debug("Many duplicates detected, increasing fetch multiplier to %d", batch_size_multiplier)
            
            if not has_more:
                break
                
        return emails_processed, current_offset, has_more
