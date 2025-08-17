# Usage Guide

The application exposes a Flask web interface for managing documents and performing RAG-powered search. Document embeddings are saved in Milvus while URLs and other metadata reside in a local SQLite database (`knowledgebase.db`).

## Starting the Application

1. Ensure Milvus is running:
   ```bash
   docker ps | grep milvus
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Launch the app:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser.

## Features

- **Document Upload** – upload PDF, DOCX, DOC, TXT, or MD files.
- **URL Management** – store URLs with automatic title extraction in `knowledgebase.db`.
- **Metadata Storage** – persist URL details now and email records in the future using `knowledgebase.db`.
- **Semantic Search** – query stored documents via vector search.
- **RAG Chat** – ask questions and receive answers synthesized from relevant documents.

## Managing Email Accounts

The dashboard includes an **Email Accounts** section for configuring IMAP
sources.

### Adding Accounts

1. Click **Add Email Account** to open the form.
2. Complete all required fields:
   - **Display Name** – unique label for the account.
   - **Server** – IMAP server hostname.
   - **Port** – connection port number.
   - **Username** – account login name.
   - **Password** – account password.
   Optional fields include **Mailbox** (defaults to `INBOX`), **Batch Limit**,
   and **Use SSL**.
3. Submit the form to save the account.

### Editing or Deleting

- Use the pencil icon to edit an existing account, update the fields, and
  select **Save changes**.
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
