import types

from retrieval.document.processor import DocumentProcessor
from types import SimpleNamespace

try:
    from langchain_core.documents import Document
except Exception:
    # Minimal fallback if langchain_core is not available in test env
    class Document:
        def __init__(self, page_content='', metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}


class DummyCursor:
    def __init__(self):
        self._last_sql = ''

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self._last_sql = sql
        self._params = params

    def fetchall(self):
        # Return different results based on SQL
        if 'chunk_hash = ANY' in self._last_sql:
            return [
                {'id': 'c_hash', 'document_id': 'd1', 'chunk_text': 'Hash match text', 'chunk_ordinal': 0, 'page_start': 1, 'page_end': 1, 'chunk_hash': 'h1'}
            ]
        if 'document_chunks WHERE document_id = ANY' in self._last_sql:
            return [
                {'id': 'c_ord', 'document_id': 'd1', 'chunk_text': 'Ordinal match text', 'chunk_ordinal': 0, 'page_start': 1, 'page_end': 1, 'chunk_hash': None}
            ]
        if 'FROM documents WHERE id = ANY' in self._last_sql:
            return [
                {'id': 'd1', 'title': 'Dummy Title'}
            ]
        return []


class DummyConn:
    def cursor(self):
        return DummyCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyPG:
    def get_connection(self):
        return DummyConn()


def test_enrichment_attaches_title_and_chunk_id():
    # Create a retriever with a fake fts_retriever that has db_manager
    fake_fts = SimpleNamespace()
    fake_fts.db_manager = DummyPG()

    # Dummy vector_retriever not used for enrichment
    retriever = DocumentProcessor(vector_retriever=None, fts_retriever=fake_fts)

    # Create documents that simulate Milvus results (one with content_hash, one with document_id+page)
    docs = []
    docs.append(Document(page_content='short', metadata={'content_hash': 'h1', 'document_id': 'd1'}))
    docs.append(Document(page_content='short', metadata={'document_id': 'd1', 'page': 1}))

    # Run enrichment
    retriever.document_manager.batch_enrich_documents_from_postgres(docs)

    # Assertions
    assert docs[0].metadata.get('title') == 'Dummy Title'
    assert docs[0].metadata.get('chunk_id') in ('c_hash', 'c_ord')
    assert 'preview' in docs[0].metadata

    assert docs[1].metadata.get('title') == 'Dummy Title'
    assert docs[1].metadata.get('chunk_id') in ('c_hash', 'c_ord')
