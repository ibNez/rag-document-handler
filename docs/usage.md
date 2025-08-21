# Usage Guide

The application exposes a Flask web interface for managing documents, URLs, emails and performing RAG-powered search. The refactored architecture uses PostgreSQL for metadata and Milvus for vector embeddings.

## Starting the Application

1. Ensure required services are running:
   ```bash
   docker compose ps  # Check Milvus and PostgreSQL containers
   ```
2. Use the proper startup script:
   ```bash
   source .venv/bin/activate  # Always source environment first (DEVELOPMENT_RULES.md)
   ./start.sh                 # Use the startup script with service checks
   ```
3. Open `http://localhost:3000` in your browser.

## Architecture Overview

The refactored application uses:
- **PostgreSQL**: Document metadata, URL scheduling, email accounts and messages
- **Milvus**: Vector embeddings for semantic search
- **Modular Ingestion**: Separate processing for documents, URLs, and emails

## Features

- **Document Upload** – Upload PDF, DOCX, DOC, TXT, or MD files with enhanced processing
- **URL Management** – Store and schedule URL crawling with PostgreSQL-based metadata  
- **Email Integration** – Support for IMAP, Gmail API, and Exchange with encrypted storage
- **Semantic Search** – Query stored content via vector search with improved accuracy
- **RAG Chat** – Ask questions and receive answers synthesized from relevant documents

## Managing Email Accounts

The dashboard includes an **Email Accounts** section for configuring IMAP, Gmail API, or Exchange sources using the refactored email management system.

### Adding Accounts

1. Click **Add Email Account** to open the form.
2. Complete all required fields:
   - **Display Name** – unique label for the account
   - **Server Type** – protocol (IMAP, Gmail, Exchange)
   - **Server** – mail server hostname or API endpoint
   - **Port** – connection port number
   - **Username** – account login name
   - **Password** – account password (encrypted via `ingestion/utils/crypto.py`)
   
   Optional fields include **Mailbox** (defaults to `INBOX`), **Batch Limit**, and **Use SSL**.
3. Submit the form to save the account with encrypted credentials.

### Processing Pipeline

The refactored email system:
- **Account Management**: Handled by `ingestion/email/manager.py`
- **Message Storage**: Via `ingestion/email/email_manager_postgresql.py`
- **Content Processing**: Through `ingestion/email/processor.py`
- **Coordination**: Managed by `ingestion/email/orchestrator.py`

### Editing or Deleting

- Use the pencil icon to edit an existing account, update the fields, and select **Save changes**
- Use the trash icon to remove an account; deletion requires confirmation.

### Security & Sync Intervals

Credentials are stored in the local `knowledgebase.db` SQLite database. They
are kept in plain text, so restrict filesystem access or employ disk
encryption when using sensitive accounts.

All configured accounts are synchronized by a background job every
`EMAIL_SYNC_INTERVAL_SECONDS` seconds (default: `300`). Adjust this environment
variable before starting the application to change how frequently emails are
fetched.

## TODOs

- [ ] Command-line interface for batch operations.
- [ ] Email ingestion from configured accounts with metadata stored in `knowledgebase.db` and embeddings persisted in Milvus.
- [ ] Image ingestion using TensorFlow object classification with embeddings stored in Milvus.
- [ ] Export search results to external formats.
