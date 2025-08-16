#!/usr/bin/env python3
"""Email ingestion utilities.

This module connects to an IMAP inbox, downloads messages from a specified
mailbox, splits the message bodies and attachments into chunks, stores basic
metadata in the local SQLite database (``knowledgebase.db``) and persists
embeddings in Milvus.

The main entry point is :func:`ingest_emails` which can be called by ``app.py``
or any future ETL job.
"""

from __future__ import annotations

import email
import hashlib
import imaplib
import json
import logging
import os
import sqlite3
from email.header import decode_header
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus as LC_Milvus

from email_accounts import EmailAccountManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _decode_header(value: str | None) -> str:
    """Decode possibly encoded header values."""
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for text, charset in parts:
        if isinstance(text, bytes):
            decoded.append(text.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(text)
    return "".join(decoded)


def _extract_content(msg: email.message.Message) -> Tuple[str, List[str], List[str]]:
    """Return body text, attachment texts and attachment names."""
    body = ""
    attachment_texts: List[str] = []
    attachment_names: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            if part.get_content_maintype() == "multipart":
                continue

            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"

            if "attachment" in content_disposition:
                filename = part.get_filename()
                if filename:
                    attachment_names.append(filename)
                if part.get_content_type() in {"text/plain", "text/html"}:
                    text = payload.decode(charset, errors="replace")
                    if part.get_content_type() == "text/html":
                        text = BeautifulSoup(text, "html.parser").get_text()
                    attachment_texts.append(text)
            else:
                if part.get_content_type() == "text/plain":
                    body += payload.decode(charset, errors="replace")
                elif part.get_content_type() == "text/html":
                    html = payload.decode(charset, errors="replace")
                    body += BeautifulSoup(html, "html.parser").get_text()
    else:
        payload = msg.get_payload(decode=True) or b""
        charset = msg.get_content_charset() or "utf-8"
        if msg.get_content_type() == "text/plain":
            body = payload.decode(charset, errors="replace")
        elif msg.get_content_type() == "text/html":
            html = payload.decode(charset, errors="replace")
            body = BeautifulSoup(html, "html.parser").get_text()

    return body, attachment_texts, attachment_names


def _header_hash(msg: email.message.Message) -> str:
    """Return SHA256 hash of important headers to identify unique emails."""
    headers = [
        msg.get("Message-ID", ""),
        msg.get("Date", ""),
        msg.get("From", ""),
        msg.get("To", ""),
        msg.get("Subject", ""),
    ]
    joined = "\n".join(headers).encode("utf-8", errors="ignore")
    return hashlib.sha256(joined).hexdigest()


def ingest_emails(
    account_name: str,
    mailbox: str = "INBOX",
    limit: Optional[int] = None,
    db_path: str = "knowledgebase.db",
) -> int:
    """Ingest emails from a configured IMAP account.

    Parameters
    ----------
    account_name:
        Name of the email account stored in ``email_accounts`` table.
    mailbox:
        IMAP mailbox to read from. Defaults to ``"INBOX"``.
    limit:
        Maximum number of most recent messages to process. ``None`` means all.
    db_path:
        Path to SQLite database.

    Returns
    -------
    int
        Number of new emails processed.
    """

    account_manager = EmailAccountManager(db_path)
    account = account_manager.get_account(account_name)
    if not account:
        logger.warning("Account %s not found", account_name)
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
        base_url=os.getenv(
            "CHAT_BASE_URL",
            f"http://{os.getenv('OLLAMA_CHAT_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}",
        ),
    )
    connection_args = {
        "host": os.getenv("MILVUS_HOST", "localhost"),
        "port": os.getenv("MILVUS_PORT", "19530"),
    }
    collection_name = os.getenv("COLLECTION_NAME", "documents")

    processed = 0

    with imaplib.IMAP4_SSL(account.imap_host, account.imap_port) as imap, sqlite3.connect(
        db_path
    ) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                header_hash TEXT UNIQUE,
                message_id TEXT,
                subject TEXT,
                sender TEXT,
                recipients TEXT,
                date TEXT,
                body TEXT,
                attachments TEXT,
                account_name TEXT,
                mailbox TEXT
            )
            """
        )
        for column in ["header_hash", "account_name", "mailbox", "message_id"]:
            try:
                cur.execute(f"ALTER TABLE emails ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass
        conn.commit()

        imap.login(account.imap_user, account.imap_password)
        imap.select(mailbox)
        status, data = imap.search(None, "ALL")
        if status != "OK":
            logger.error("Failed to search mailbox %s", mailbox)
            return 0

        numbers = data[0].split()
        if limit is not None:
            numbers = numbers[-limit:]

        for num in numbers:
            status, msg_data = imap.fetch(num, "(RFC822)")
            if status != "OK":
                logger.error("Failed to fetch email %s", num.decode())
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            header_hash = _header_hash(msg)
            cur.execute("SELECT 1 FROM emails WHERE header_hash = ?", (header_hash,))
            if cur.fetchone():
                continue

            subject = _decode_header(msg.get("Subject"))
            sender = _decode_header(msg.get("From"))
            recipients = _decode_header(msg.get("To"))
            date = msg.get("Date")
            message_id = msg.get("Message-ID") or num.decode()

            body_text, attachment_texts, attachment_names = _extract_content(msg)

            cur.execute(
                """
                INSERT INTO emails (
                    header_hash, message_id, subject, sender, recipients, date, body, attachments, account_name, mailbox
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    header_hash,
                    message_id,
                    subject,
                    sender,
                    recipients,
                    date,
                    body_text,
                    json.dumps(attachment_names),
                    account_name,
                    mailbox,
                ),
            )
            conn.commit()

            texts = [body_text] + attachment_texts
            documents: List[Document] = []
            for text in texts:
                if not text:
                    continue
                for idx, chunk in enumerate(splitter.split_text(text)):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": "email",
                                "subject": subject,
                                "from": sender,
                                "to": recipients,
                                "date": date,
                                "message_id": message_id,
                                "account": account_name,
                                "mailbox": mailbox,
                                "chunk_id": f"{header_hash}_{idx}",
                            },
                        )
                    )

            if documents:
                try:
                    store = LC_Milvus(
                        embedding_function=embeddings,
                        collection_name=collection_name,
                        connection_args=connection_args,
                    )
                    store.add_documents(documents)
                except Exception:
                    LC_Milvus.from_documents(
                        documents,
                        embeddings,
                        collection_name=collection_name,
                        connection_args=connection_args,
                    )
            processed += 1

        imap.logout()

    logger.info("Processed %d new emails", processed)
    return processed


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    ingest_emails("default")
