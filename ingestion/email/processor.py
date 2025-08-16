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
        self.embedding_model = embedding_model or OllamaEmbeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.manager = EmailManager(sqlite_conn)

    # ------------------------------------------------------------------
    def _store_metadata(self, record: Dict[str, Any]) -> None:
        """Persist an email record using :class:`EmailManager`."""
        self.manager.upsert_email(record)

    # ------------------------------------------------------------------
    def _store_embeddings(
        self, message_id: str, chunks: List[str], embeddings: Iterable[List[float]], record: Dict[str, Any]
    ) -> None:
        """Insert embeddings into Milvus using the provided client."""
        metadatas = []
        ids = []
        for idx, _ in enumerate(chunks):
            cid = f"{message_id}:{idx}"
            ids.append(cid)
            meta = {
                "message_id": message_id,
                "chunk_id": cid,
                "subject": record.get("subject"),
                "from_addr": record.get("from_addr"),
                "to_addrs": record.get("to_addrs"),
                "date_utc": record.get("date_utc"),
            }
            metadatas.append(meta)

        try:
            if hasattr(self.milvus, "add_embeddings"):
                self.milvus.add_embeddings(
                    embeddings=list(embeddings), ids=ids, metadatas=metadatas
                )
            elif hasattr(self.milvus, "add_texts"):
                # When add_texts is available we assume the vector store will
                # handle embedding generation. However, embeddings are already
                # computed so we call add_embeddings if present; otherwise fall
                # back to add_texts with raw chunks.
                self.milvus.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
            elif hasattr(self.milvus, "insert"):
                # pymilvus.Collection style
                entities = [ids, list(embeddings), chunks, metadatas]
                self.milvus.insert(entities)
            else:
                raise RuntimeError("Unsupported Milvus client interface")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Milvus insertion failed for %s: %s", message_id, exc)

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
            return

        chunks = [c for c in self.splitter.split_text(body_text) if c.strip()]
        if not chunks:
            return
        embeddings = self.embedding_model.embed_documents(chunks)
        self._store_embeddings(message_id, chunks, embeddings, record)
