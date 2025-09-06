"""
Document Manager for retrieval operations.
"""

import logging
from typing import Any, List, Dict, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentManager:
    """
    Handles document retrieval database operations for PostgreSQL and Milvus.
    """

    def __init__(self, postgres_manager: Any) -> None:
        """
        Initialize the DocumentManager.

        Args:
            postgres_manager: Manager for PostgreSQL operations.
        """
        self.postgres_manager = postgres_manager
        logger.info("Document Manager initialized for retrieval operations")

    def normalize_documents_metadata(self, docs: List[Document]) -> None:
        """
        Normalize metadata for a list of Documents in-place.

        - If metadata contains a single nested 'metadata' dict, unwrap it.
        - Map common alias fields to canonical names:
            - 'id' or 'pk' -> 'chunk_id'
            - keep 'document_id' as-is
        """
        for doc in docs:
            try:
                meta = getattr(doc, 'metadata', None)
                if meta is None:
                    raise TypeError(
                        f"Document.metadata is None for document {getattr(doc,'metadata', None)!r}; expected a dict."
                        " Ensure your retriever returns Document objects with 'metadata' populated."
                    )
                if not isinstance(meta, dict):
                    raise TypeError(
                        f"Document.metadata has unexpected type {type(meta).__name__} for doc {doc!r}; expected dict."
                        " Fix the retriever to return dict metadata or run the dependency verification script."
                    )

                nested = meta.get('metadata')
                if nested is not None and not isinstance(nested, dict):
                    raise TypeError(
                        f"Nested 'metadata' field is not a dict for doc {doc!r}; found {type(nested).__name__}."
                    )
                if isinstance(nested, dict):
                    meta = dict(nested)

                # Map common aliases into table-specific id
                # Map Milvus 'pk' to 'document_chunk_id' if not already present
                if not meta.get('document_chunk_id'):
                    if meta.get('pk'):
                        meta['document_chunk_id'] = meta.get('pk')
                    elif meta.get('id'):
                        meta['document_chunk_id'] = meta.get('id')
                    else:
                        # Only log error if we truly have no chunk identifier
                        logger.debug("Document metadata missing chunk identifier: %s", meta)
                        # Don't raise error - let enrichment handle it
                
                # Some stores use 'doc_id' or 'docid' variants
                if not meta.get('document_id') and meta.get('doc_id'):
                    meta['document_id'] = meta.get('doc_id')
                if not meta.get('document_id') and meta.get('docid'):
                    meta['document_id'] = meta.get('docid')

                doc.metadata = meta
            except Exception:
                # Best-effort normalization; don't fail the whole retrieval
                logger.debug('Failed to normalize document metadata', exc_info=True)

    def batch_enrich_documents_from_postgres(self, docs: List[Document]) -> None:
        """
        Batch-fetch canonical chunk metadata from PostgreSQL and attach to
        LangChain Document.metadata in-place. This avoids duplicating large
        blobs in Milvus while ensuring UI/rerankers have readable metadata.

        Strategy:
        - Collect content_hash and document_id values from the provided docs.
        - Query `document_chunks` for matching chunk rows (batch by ANY(%s)).
        - Query `documents` for titles for any document_ids discovered.
        - Enrich each Document.metadata with: title, chunk_id (chunk row id),
          chunk_hash, chunk_ordinal, page_start/page_end and a short preview.
        """
        if not docs:
            return

        # Use the provided postgres manager
        pg = self.postgres_manager
        if pg is None:
            logger.debug("No Postgres manager available for enrichment")
            return

        # Collect unique identifiers to look up
        content_hashes = {d.metadata.get('content_hash') for d in docs if d.metadata and d.metadata.get('content_hash')}
        content_hashes = {h for h in content_hashes if h}
        document_ids = {d.metadata.get('document_id') for d in docs if d.metadata and d.metadata.get('document_id')}
        document_ids = {did for did in document_ids if did}

        chunk_rows_by_hash = {}
        chunk_rows_by_doc = {}
        titles_by_doc = {}

        # Normalize incoming docs metadata to canonical shape before enrichment
        try:
            self.normalize_documents_metadata(docs)
        except Exception:
            logger.debug('Failed to normalize documents prior to enrichment', exc_info=True)

        try:
            with pg.get_connection() as conn:
                with conn.cursor() as cur:
                    # Fetch chunk rows by chunk_hash in one query
                    if content_hashes:
                        cur.execute(
                            "SELECT id AS document_chunk_id, document_id, chunk_text, chunk_ordinal, page_start, page_end, chunk_hash, topics FROM document_chunks WHERE chunk_hash = ANY(%s)",
                            (list(content_hashes),)
                        )
                        for row in cur.fetchall():
                            chunk_rows_by_hash[row['chunk_hash']] = dict(row)

                    # Fetch chunk rows for document_ids (if any) so we can map by doc+ordinal
                    if document_ids:
                        cur.execute(
                            "SELECT id AS document_chunk_id, document_id, chunk_text, chunk_ordinal, page_start, page_end, chunk_hash, topics FROM document_chunks WHERE document_id = ANY(%s::uuid[])",
                            (list(document_ids),)
                        )
                        for row in cur.fetchall():
                            doc_id = row['document_id']
                            chunk_rows_by_doc.setdefault(doc_id, {})
                            # map by chunk_hash when available, fallback to ordinal
                            key = row.get('chunk_hash') or f"ordinal:{row.get('chunk_ordinal')}"
                            chunk_rows_by_doc[doc_id][key] = dict(row)
                            # Also index by page_start for datasets that use page metadata
                            page_start = row.get('page_start')
                            if page_start is not None:
                                chunk_rows_by_doc[doc_id][f"page:{page_start}"] = dict(row)

                    # Fetch document titles for discovered document_ids
                    if document_ids:
                        cur.execute("SELECT id AS document_id, title FROM documents WHERE id = ANY(%s::uuid[])", (list(document_ids),))
                        for row in cur.fetchall():
                            doc_id = row.get('document_id')
                            if doc_id:
                                titles_by_doc[str(doc_id)] = row.get('title')

        except Exception as e:
            logger.debug(f"Postgres enrichment query failed: {e}")
            return

        # Apply enrichment to docs
        for doc in docs:
            try:
                meta = doc.metadata or {}
                if meta.get('title') and meta.get('document_chunk_id'):
                    continue

                ch = meta.get('content_hash')
                did = meta.get('document_id')

                chosen_row = None
                # Prefer exact chunk_hash match
                if ch and ch in chunk_rows_by_hash:
                    chosen_row = chunk_rows_by_hash.get(ch)
                # Otherwise try mapping by document id + chunk_hash, page, or ordinal
                if not chosen_row and did and did in chunk_rows_by_doc:
                    # try chunk_hash key
                    if ch and ch in chunk_rows_by_doc[did]:
                        chosen_row = chunk_rows_by_doc[did][ch]
                    else:
                        # try matching by page metadata from Milvus
                        page_meta = meta.get('page')
                        if page_meta is not None:
                            # match by page_start or chunk_ordinal stored under page:<n>
                            page_key = f"page:{page_meta}"
                            if page_key in chunk_rows_by_doc[did]:
                                chosen_row = chunk_rows_by_doc[did][page_key]
                            elif f"ordinal:{page_meta}" in chunk_rows_by_doc[did]:
                                chosen_row = chunk_rows_by_doc[did][f"ordinal:{page_meta}"]
                        # fallback to first chunk for that doc (best-effort)
                        if not chosen_row:
                            first_key = next(iter(chunk_rows_by_doc[did].keys()), None)
                            if first_key:
                                chosen_row = chunk_rows_by_doc[did][first_key]

                if chosen_row:
                    if chosen_row.get('document_chunk_id'):
                        meta['document_chunk_id'] = chosen_row.get('document_chunk_id')
                    else:
                        logger.error('Enrichment returned row missing canonical document_chunk_id: %s', chosen_row)
                        raise KeyError('Enrichment missing document_chunk_id')
                    meta['chunk_hash'] = chosen_row.get('chunk_hash')
                    meta['chunk_ordinal'] = chosen_row.get('chunk_ordinal')
                    meta['page_start'] = chosen_row.get('page_start')
                    meta['page_end'] = chosen_row.get('page_end')
                    meta['topic'] = chosen_row.get('topic')  # Add LLM-generated topic from database
                    # Prefer document title from `documents` table if available
                    if did and did in titles_by_doc and titles_by_doc[did]:
                        meta['title'] = titles_by_doc[did]
                    # Provide a short preview from chunk_text if page_content is empty or short
                    if not doc.page_content or len(doc.page_content) < 50:
                        chunk_text = chosen_row.get('chunk_text') or ''
                        doc.page_content = chunk_text
                    # Short preview used by UI
                    meta['preview'] = (doc.page_content[:240] + '...') if len(doc.page_content) > 240 else doc.page_content
                    doc.metadata = meta
            except Exception:
                # Fail safe: don't break enrichment loop
                logger.debug("Failed to enrich a document from Postgres", exc_info=True)