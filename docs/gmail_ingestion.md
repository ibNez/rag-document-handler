# Gmail Email Ingestion

This guide covers how to ingest emails from Gmail into RAG Document Handler.

## Prerequisites

1. **Enable the Gmail API**
   - Visit the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a project (or reuse an existing one) and enable the **Gmail API**.
2. **Create OAuth credentials**
   - Under *APIs & Services → Credentials* create an *OAuth client ID* for a **Desktop app**.
   - Download the `credentials.json` file.
3. **Generate a user token**
   - Run Google's Python quickstart or another OAuth flow using the downloaded credentials to authorize your Gmail account.
   - The process produces a `token.json` file containing the access and refresh tokens. Store this file somewhere the application can read (e.g. `~/.config/gmail/token.json`).
4. **Required scopes**
   - The connector only needs read access: `https://www.googleapis.com/auth/gmail.readonly`.
5. **Configure RAG Document Handler**
   - Set `EMAIL_ENABLED=true` in your environment to activate email ingestion.
   - Configure `EMAIL_SYNC_INTERVAL_SECONDS` to control how often accounts are synchronized (default `300`).
   - Add a Gmail account to the `email_accounts` table (via UI or SQL) with the following fields:
     - `server_type`: `gmail`
     - `username`: your Gmail address
     - `token_file`: path to the `token.json` generated above
     - Optional: `batch_limit` to cap messages fetched per sync

## Minimal Example

```python
import sqlite3
from google.oauth2.credentials import Credentials
from ingestion.email.connector import GmailConnector
from ingestion.email.processor import EmailProcessor
from ingestion.email.ingest import run_email_ingestion

# Load OAuth token
creds = Credentials.from_authorized_user_file("token.json")
connector = GmailConnector(credentials=creds, user_id="me")

# Set up a processor (replace with real Milvus client)
sqlite_conn = sqlite3.connect("databases/knowledgebase.db")
milvus = ...  # existing Milvus connection
processor = EmailProcessor(milvus, sqlite_conn)

# Fetch and process new messages
run_email_ingestion(connector, processor)
```

## Limitations & Scheduling

- **API quotas**: The Gmail API enforces per-user and daily request quotas. Excessive polling may hit these limits.
- **Rate limits**: Gmail recommends no more than one request per user per second.
- **Token expiry**: Access tokens expire; the connector refreshes tokens when `refresh_token` is present.
- **Scheduling**: To keep your knowledge base up to date, run the `EmailOrchestrator` with `EMAIL_ENABLED=true`, or call `run_email_ingestion` from a cron job at the desired interval.
