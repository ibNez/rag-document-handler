#!/usr/bin/env python3
"""
Probe that inspects the most-recently imported URL and its downstream artifacts.
No writes — read-only checks against Postgres and Milvus.

Usage:
  python tools/probe_url_docs.py [url_or_id]
If no arg provided, inspects the most recent URL.
"""
import sys
import os
from urllib.parse import urlparse

try:
    from rag_manager.core.config import Config
except Exception:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from rag_manager.core.config import Config


def main(target=None):
    try:
        from rag_manager.managers.postgres_manager import PostgreSQLManager
    except Exception as e:
        print('ERROR: could not import PostgreSQLManager:', e)
        return

    cfg = Config()
    mgr = PostgreSQLManager()

    try:
        with mgr.get_connection() as conn:
            with conn.cursor() as cur:
                if target:
                    cur.execute("SELECT id, url, title, last_content_hash, last_update_status, created_at FROM urls WHERE id::text = %s OR url = %s ORDER BY created_at DESC LIMIT 1", (target, target))
                else:
                    cur.execute("SELECT id, url, title, last_content_hash, last_update_status, created_at FROM urls ORDER BY created_at DESC LIMIT 1")

                url_row = cur.fetchone()
                if not url_row:
                    print('No URL rows found')
                    return

                url_id = str(url_row['id'])
                url_str = url_row['url']

                print('URL:')
                print(' id:', url_id)
                print(' url:', url_str)
                print(' title:', url_row.get('title'))
                print(' last_content_hash:', url_row.get('last_content_hash'))
                print(' last_update_status:', url_row.get('last_update_status'))
                print(' created_at:', url_row.get('created_at'))

                # Build LIKE patterns
                domain = urlparse(url_str).hostname or ''
                last_hash = url_row.get('last_content_hash') or ''

                like_hash = f"%{last_hash}%" if last_hash else '%'
                like_domain = f"%{domain}%" if domain else '%'

                # Find candidate documents (url snapshots)
                cur.execute(
                    """
                    SELECT id, title, filename, file_path, processing_status, chunk_count, created_at
                    FROM documents
                    WHERE document_type = 'url'
                      AND (
                          filename LIKE %s OR file_path LIKE %s OR title LIKE %s
                      )
                    ORDER BY created_at DESC
                    LIMIT 50
                    """,
                    (like_hash, like_domain, like_domain)
                )

                docs = cur.fetchall()
                print('\nFound', len(docs), 'candidate documents matching hash/domain')

                for d in docs:
                    print('\nDocument:')
                    print(' id:', d['id'])
                    print(' title:', d.get('title'))
                    print(' filename:', d.get('filename'))
                    print(' file_path:', d.get('file_path'))
                    print(' processing_status:', d.get('processing_status'))
                    print(' chunk_count (documents):', d.get('chunk_count'))
                    # document_chunks
                    cur.execute('SELECT COUNT(*) as cnt FROM document_chunks WHERE document_id = %s', (d['id'],))
                    cnt = cur.fetchone()
                    print(' chunk_count (document_chunks):', cnt['cnt'] if cnt else 0)

    except Exception as e:
        print('Postgres probe failed:', repr(e))
        return

    # Milvus check for the matched documents
    try:
        from pymilvus import connections, utility, Collection
        mh = cfg.MILVUS_HOST
        mp = cfg.MILVUS_PORT
        connections.connect(host=mh, port=str(mp))
        col_name = cfg.DOCUMENT_COLLECTION
        if not utility.has_collection(col_name):
            print('\nMilvus collection not found:', col_name)
            return

        coll = Collection(col_name)
        print('\nMilvus collection exists, total entities=', coll.num_entities)

        for d in docs:
            expr = f'document_id == "{d["id"]}"'
            try:
                res = coll.query(expr, output_fields=['pk','page','content_hash','text'], limit=10)
                print('\nMilvus vectors for document id', d['id'], 'count=', len(res))
                for r in res[:3]:
                    preview = (r.get('text') or '')[:300]
                    print('  -', {k: r.get(k) for k in ('pk','page','content_hash')}, 'preview=', preview.replace('\n',' ')[:200])
            except Exception as me:
                print('  Milvus query failed for', d['id'], 'error:', me)

    except Exception as e:
        print('\nMilvus probe skipped or failed:', repr(e))


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
