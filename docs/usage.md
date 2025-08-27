# Usage Guide

The application exposes a Flask web interface for managing documents, URLs, emails and performing RAG-powered search. The refactored architecture uses PostgreSQL for metadata, Milvus for vector embeddings, and a modular template system with partials for improved maintainability.

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
- **Panel-Specific Statistics**: Dedicated statistics providers for each dashboard section
- **Template Partials**: Well-organized, maintainable template structure

## Enhanced Dashboard Interface

The web interface has been completely refactored using a modular partials system that provides:

### Statistics Panel
The main dashboard features a comprehensive statistics panel that displays real-time information across four key areas:

**Connections Status:**
- SQL database connectivity and version information
- Milvus vector database status and health
- Collection statistics and entity counts
- Email collection metrics and indexing status

**Document Analytics:**
- Total document count and processing statistics
- Average words per document and chunk distribution
- Median chunk character counts for optimization insights
- Embedding dimensions and indexing configuration
- Top keywords extracted from document content

**URL Management Metrics:**
- Total URLs tracked and their processing status
- Active URLs and domain crawling statistics
- Robots.txt compliance and ignored rules count
- Scraped vs. never-scraped URL breakdown
- URLs due for refresh based on scheduling intervals

**Email Account Overview:**
- Total configured accounts and sync status
- Accounts due for synchronization
- Never-synced accounts requiring attention
- Total email count and attachment statistics
- Embedding configuration and indexing status
- Most active account identification
- Latest email processing timestamps

### Modular Section Organization

The interface is organized into distinct, well-labeled sections:

**File Management:**
- **Upload Section**: Clean file upload interface with format validation
- **Staging Area**: Files awaiting processing with progress tracking
- **Processed Documents**: Completed files with metadata and actions

**URL Management:**
- **Add URL Form**: Simple URL addition with automatic title extraction
- **URL Table**: Comprehensive URL listing with status, scheduling, and robots.txt information
- **Edit Modals**: In-line editing for URL metadata, crawling settings, and snapshot configuration

**Email Management:**
- **Account Configuration**: Add, edit, and delete email accounts
- **Sync Monitoring**: Real-time sync status and progress tracking
- **Connection Testing**: Validation of email server connectivity

### Template Partials System

The interface is built using a modular partials system that provides:

**Benefits for Users:**
- **Consistent Experience**: Uniform UI patterns across all sections
- **Fast Loading**: Optimized partial rendering for better performance
- **Error Isolation**: Issues in one section don't affect others
- **Real-time Updates**: Auto-refresh capabilities for live data

**Benefits for Developers:**
- **Maintainable Code**: Each section is self-contained and easy to modify
- **Team Development**: Multiple developers can work on different sections simultaneously
- **Reusable Components**: Partials can be shared across multiple pages
- **Clear Organization**: Well-labeled components with specific purposes

## Features

- **Document Upload** – Upload PDF, DOCX, DOC, TXT, or MD files with enhanced processing
- **URL Management** – Store and schedule URL crawling with PostgreSQL-based metadata  
- **Email Integration** – Support for IMAP, Gmail API, and Exchange with encrypted storage
- **Semantic Search** – Query stored content via vector search with improved accuracy
- **RAG Chat** – Ask questions and receive answers synthesized from relevant documents
- **Email Classification** – Automatic detection and specialized handling of email-related queries

## Enhanced Query Processing

The system now features intelligent query classification that automatically determines whether your questions are about emails or general documents:

### Email Queries
Questions about emails are automatically detected and processed through a specialized email search pipeline:
- "What emails did John send about the project?"
- "Show me emails from last week about budget discussions"
- "Find emails with attachments about the quarterly report"

Email results include clickable email IDs that open detailed email viewers with:
- **Full email content** with HTML rendering support
- **Sender and recipient information**
- **Attachment details** with file sizes and types
- **Thread and reply context**
- **Technical metadata** (Message ID, headers, etc.)

### General Document Queries
Standard questions about documents and URLs use the existing RAG pipeline:
- "What is the main methodology described in the research papers?"
- "Explain the key findings from the uploaded documents"
- "What tools are mentioned in the documentation?"

### Automatic Classification
The system uses a dedicated LLM model to analyze query intent and route to the appropriate search system. No manual selection required - just ask natural language questions and the system handles the rest.

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

- [ ] Image ingestion using TensorFlow object classification with embeddings stored in Milvus.
- [ ] Export search results to external formats.
