from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from bs4 import BeautifulSoup

from .connector import EmailConnector, IMAPConnector
from .email_manager import compute_header_hash
from .processor import EmailProcessor

CONTENT_HASH_RECIPE_VERSION = 1


def _participants_hash(participants: Sequence[str]) -> Optional[str]:
    """Return deterministic hash for a list of participants."""
    if not participants:
        return None
    norm = sorted({(p or "").lower() for p in participants if p})
    if not norm:
        return None
    return hashlib.sha256("\n".join(norm).encode("utf-8")).hexdigest()


def _content_hash(record: Dict[str, Any]) -> str:
    """Compute a content hash used for dedupe."""
    payload = {
        "v": CONTENT_HASH_RECIPE_VERSION,
        "subject": record.get("subject") or "",
        "body_text": record.get("body_text") or "",
        "date_utc": record.get("date_utc") or "",
        "participants_hash": record.get("participants_hash") or "",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an email record by populating hashes and defaults.

    This function performs light text normalisation on the message body prior
    to computing hashes. If ``body_text`` is missing but ``body_html`` is
    present, the HTML is converted to plain text. Simple email signature blocks
    are stripped by removing any content following the conventional ``"--"``
    separator.
    """
    body_text = record.get("body_text")
    body_html = record.get("body_html")
    if not body_text and body_html:
        body_text = BeautifulSoup(body_html, "html.parser").get_text()
    if body_text:
        # Remove common signature blocks marked by "--" on its own line
        parts = body_text.split("\n--\n", 1)
        record["body_text"] = parts[0].strip()
    
    participants = set()
    from_addr = record.get("from_addr")
    if from_addr:
        participants.add(from_addr.lower())
    for addr in record.get("to_addrs") or []:
        participants.add(addr.lower())
    for addr in record.get("cc_addrs") or []:
        participants.add(addr.lower())
    record["participants"] = sorted(participants)
    record["participants_hash"] = _participants_hash(record["participants"])
    record["header_hash"] = compute_header_hash(record)
    record["content_hash"] = _content_hash(record)
    record.setdefault("ingested_at", datetime.utcnow().isoformat())
    return record


def run_email_ingestion(
    connector: EmailConnector,
    processor: EmailProcessor,
    *,
    since_date: Optional[datetime] = None,
) -> int:
    """Execute the email ingestion pipeline.

    The pipeline performs the following steps:

    1. Fetch records via ``connector``.
    2. Normalize and compute hashes.
    3. Dedupe by ``header_hash`` and ``content_hash``.
    4. Chunk, embed and persist via ``processor``.

    Returns
    -------
    int
        Number of new emails processed.
    """
    records = connector.fetch_emails(since_date)
    if not records:
        return 0

    processed = 0
    for rec in (_normalize(r) for r in records):
        hh = rec.get("header_hash")
        if hh and processor.manager.get_email_by_header_hash(hh):
            continue
        ch = rec.get("content_hash")
        if ch and processor.manager.get_email_by_hash(ch):
            continue
        processor.process(rec)
        processed += 1
    return processed


def main() -> None:
    """CLI entry point for scheduled execution."""
    parser = argparse.ArgumentParser(description="Run email ingestion pipeline")
    parser.add_argument("--imap-host", required=True)
    parser.add_argument("--imap-port", type=int, default=993)
    parser.add_argument("--imap-username", required=True)
    parser.add_argument("--imap-password", required=True)
    parser.add_argument("--mailbox", default="INBOX")
    parser.add_argument("--since-days", type=int, default=1)
    parser.add_argument("--sqlite-path", default="emails.db")
    args = parser.parse_args()

    connector = IMAPConnector(
        host=args.imap_host,
        port=args.imap_port,
        username=args.imap_username,
        password=args.imap_password,
        mailbox=args.mailbox,
    )

    sqlite_conn = sqlite3.connect(args.sqlite_path)

    class _NoopMilvus:
        def add_embeddings(
            self,
            embeddings: Iterable[List[float]],
            ids: Iterable[str],
            metadatas: Iterable[Dict[str, Any]],
        ) -> None:
            pass

    milvus = _NoopMilvus()
    processor = EmailProcessor(milvus, sqlite_conn)
    since = datetime.utcnow() - timedelta(days=args.since_days)
    run_email_ingestion(connector, processor, since_date=since)


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
