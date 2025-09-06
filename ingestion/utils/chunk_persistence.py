"""
Shared chunk persistence utility.

This module provides a single function to persist document chunks into PostgreSQL
so multiple ingestion flows can reuse the same logic (uploads, URL orchestrator,
email ingestion, etc.).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def persist_chunks(postgres_manager, document_id: str, chunks: list, trace_logger: Optional[logging.Logger] = None) -> int:
    """
    Persist a list of LangChain-style Document chunks into PostgreSQL using the
    provided `postgres_manager` which must implement `store_document_chunk` and
    `get_connection()`.

    Returns the number of chunks successfully stored.
    """
    if not postgres_manager:
        logger.warning("persist_chunks: no postgres manager provided")
        return 0

    stored = 0
    try:
        for idx, ch in enumerate(chunks, start=1):
            try:
                chunk_text = ch.page_content or ''
                meta = ch.metadata or {}
                page_start = meta.get('page')
                page_end = meta.get('page')
                section_path = meta.get('section_path')
                element_types = meta.get('element_types', [])
                token_count = meta.get('token_count') or (len(chunk_text.split()) if chunk_text else 0)
                chunk_hash = meta.get('content_hash')

                chunk_id = postgres_manager.store_document_chunk(
                    document_id=document_id,
                    chunk_text=chunk_text,
                    chunk_ordinal=idx,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    element_types=element_types,
                    token_count=token_count,
                    chunk_hash=chunk_hash
                )

                # Annotate chunk metadata for downstream consumers
                try:
                    ch.metadata['document_chunk_id'] = str(chunk_id)
                    ch.metadata['document_id'] = str(document_id)
                except Exception:
                    pass

                stored += 1
                if trace_logger:
                    trace_logger.info(f"Stored chunk ordinal={idx} chunk_id={chunk_id} content_hash={chunk_hash}")
            except Exception as e:
                logger.exception(f"Failed to persist chunk ordinal={idx} for document {document_id}: {e}")
                if trace_logger:
                    trace_logger.exception(f"Failed to persist chunk ordinal={idx}: {e}")

        # Update document metadata with chunk count and mark completed if any chunks stored
        try:
            with postgres_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE documents
                        SET chunk_count = %s,
                            processing_status = CASE WHEN %s > 0 THEN 'completed' ELSE processing_status END,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        [stored, stored, document_id]
                    )
                    conn.commit()
            if trace_logger:
                trace_logger.info(f"Updated document {document_id} metadata: chunk_count={stored}")
        except Exception as e:
            logger.warning(f"Failed to update document metadata for {document_id}: {e}")

    except Exception as e:
        logger.exception(f"persist_chunks failed for document {document_id}: {e}")
        raise

    return stored
