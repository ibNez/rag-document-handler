#!/usr/bin/env python3
"""
Inspect Milvus vector store documents for missing metadata.

Usage: python3 tools/inspect_vectors.py

This script will attempt to initialize the app's MilvusManager (uses your project Config),
sample up to N vectors via the LangChain vector store wrapper, and print a JSON summary
of missing metadata (document_id, title, chunk_id) plus a small sample of offending items.
"""
import json
import os
import sys
from typing import List

try:
    # Project imports
    from rag_manager.core.config import Config
    from rag_manager.managers.milvus_manager import MilvusManager
except Exception as e:
    print("Failed to import project modules. Run this from the repository root inside the venv.")
    print(e)
    sys.exit(2)


def safe_sample(docs, n=10):
    out = []
    for d in docs[:n]:
        meta = getattr(d, 'metadata', {}) or {}
        out.append({
            'document_chunk_id': meta.get('document_chunk_id') or None,
            'document_id': meta.get('document_id') or meta.get('doc_id') or None,
            'title': meta.get('title') or meta.get('filename') or None,
            'retrieval_method': meta.get('retrieval_method') or None,
        })
    return out


def main():
    cfg = Config()
    mm = MilvusManager(cfg)

    try:
        mm._ensure_vector_store()
    except Exception as e:
        print(json.dumps({'error': 'Could not ensure vector store', 'detail': str(e)}))
        sys.exit(3)

    vs = mm.vector_store
    if vs is None:
        print(json.dumps({'error': 'Vector store not initialized'}))
        sys.exit(4)

    # Try a few retrieval methods; prefer LangChain similarity_search if available
    docs = []
    tried = []
    try:
        if hasattr(vs, 'similarity_search'):
            tried.append('similarity_search')
            docs = vs.similarity_search('test', k=200)
    except Exception:
        docs = []

    try:
        if not docs and hasattr(vs, 'as_retriever'):
            tried.append('as_retriever')
            retr = vs.as_retriever()
            docs = retr.get_relevant_documents('test')[:200]
    except Exception:
        docs = docs or []

    # Last-resort: if MilvusManager exposes a helper
    try:
        if not docs and hasattr(mm, 'vector_store'):
            tried.append('manager_fallback')
            docs = mm.vector_store.similarity_search('test', k=50)
    except Exception:
        docs = docs or []

    total = len(docs)
    missing_docid = 0
    missing_title = 0
    missing_chunk = 0
    docid_values = set()
    samples_missing = []

    for d in docs:
        meta = getattr(d, 'metadata', {}) or {}
        docid = meta.get('document_id') or meta.get('doc_id') or None
        title = meta.get('title') or meta.get('filename') or None
        chunk_id = meta.get('document_chunk_id') or None

        if not docid:
            missing_docid += 1
        else:
            docid_values.add(docid)

        if not title:
            missing_title += 1

        if not chunk_id:
            missing_chunk += 1

        if (not docid or not title) and len(samples_missing) < 12:
            samples_missing.append({
                'meta': meta,
                'preview': (getattr(d, 'page_content', '')[:240] + '...') if getattr(d, 'page_content', None) else None
            })

    out = {
        'collection': getattr(mm, 'collection_name', None),
        'num_entities_in_collection': None,
        'sampled_documents': total,
        'missing_document_id_count': missing_docid,
        'missing_title_count': missing_title,
        'missing_chunk_id_count': missing_chunk,
        'unique_document_ids_in_sample': list(docid_values)[:20],
        'sample_missing_items': samples_missing,
        'methods_tried': tried
    }

    # Try to get total num_entities from pymilvus Collection if available
    try:
        from pymilvus import Collection
        coll = Collection(out['collection'])
        out['num_entities_in_collection'] = coll.num_entities
    except Exception:
        out['num_entities_in_collection'] = None

    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()
