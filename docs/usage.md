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
- Parent-child URL relationships and progress tracking
- Child URL processing statistics (active, completed, failed)
- Robots.txt compliance and ignored rules count
- Scraped vs. never-scraped URL breakdown
- URLs due for refresh based on scheduling intervals
- Automatic snapshot creation for all URLs (always enabled)

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
- **Parent-Child Tracking**: Real-time progress monitoring for domain crawling operations
- **Child URL Statistics**: Visual progress bars and detailed statistics for discovered URLs
- **Always-On Snapshots**: Automatic PDF snapshot creation for all URLs (no toggle option)

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
- **URL Management** – Store and schedule URL crawling with PostgreSQL-based metadata and parent-child relationship tracking  
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

### Email Protocol Specifics

#### IMAP Configuration
- Standard IMAP servers with SSL/TLS support
- Default port: 993 (SSL) or 143 (non-SSL)
- Supports batch processing with configurable limits
- Enhanced corruption detection for malformed emails

#### Gmail API Integration
1. **Enable Gmail API** in Google Cloud Console
2. **Create OAuth credentials** for Desktop application
3. **Generate token.json** file through OAuth flow
4. **Required scope**: `https://www.googleapis.com/auth/gmail.readonly`
5. **Configure account** with `server_type: gmail` and token file path

#### Exchange Server Integration
- Uses Exchange Web Services (EWS) via exchangelib library
- **Server**: EWS server hostname (e.g., `exchange.company.com`)
- **Authentication**: Username/password or modern auth
- **Requirements**: EWS access must be enabled for the account
- **Batch processing**: Fetches messages in descending order by date

### Email Processing Features
- **Corruption Detection**: Validates Message-ID headers and email structure
- **Offset-Aware Processing**: Tracks processed emails to resume from last position
- **Smart Batching**: Avoids duplicates with intelligent batch processing
- **Enhanced Error Handling**: Fail-fast approach with detailed logging
- **Real-time Monitoring**: Live statistics and processing status updates

## URL Snapshots

The system automatically creates PDF snapshots of all web pages for historical preservation and reference.

### Snapshot Features
- **Always Enabled**: Snapshots are automatically created for all URLs (no toggle option)
- **PDF Generation**: High-quality PDF creation using Playwright browser automation
- **Content Change Detection**: New snapshots only created when content changes
- **Systematic Organization**: Hierarchical directory structure with structured naming
- **Retention Policies**: Automatic cleanup based on age and count limits

### Directory Structure
```
snapshots/
  <domain>/
    <path_slug>/
      <timestampZ>__q-<qs8>__v-<variant>__c-<c8>.pdf
      <timestampZ>__q-<qs8>__v-<variant>__c-<c8>.json
```

### Naming Convention
- **`<domain>`**: Punycode-normalized domain (e.g., `example.com`)
- **`<path_slug>`**: URL path converted to safe slug
- **`<timestampZ>`**: UTC timestamp in ISO format (`YYYYMMDDTHHMMSSZ`)
- **`q-<qs8>`**: 8-character hash of normalized query string
- **`v-<variant>`**: Render variant tokens (format, viewport, locale)
- **`c-<c8>`**: 8-character hash of content for deduplication

### Configuration Options
- **Retention Days**: How long to keep snapshots (0 = forever)
- **Max Snapshots**: Maximum number of snapshots per URL (0 = unlimited)
- **Viewport Settings**: Configurable browser viewport size
- **PDF Format**: Configurable page format (A4, Letter, etc.)

## Hybrid Search System

The system uses an advanced hybrid retrieval approach that combines vector similarity search with PostgreSQL full-text search for optimal results.

### Search Methods
1. **Hybrid Retrieval**: Combines vector similarity and PostgreSQL FTS using Reciprocal Rank Fusion (RRF)
2. **Vector-Only Fallback**: Pure vector similarity search if hybrid fails
3. **Smart Routing**: Automatic fallback with comprehensive error handling

### Enhanced Search Results
Documents return rich metadata including:
- **Retrieval Method**: Indicates if hybrid or vector-only was used
- **Ranking Information**: Both vector and FTS rank scores
- **Page References**: Precise page-level citations
- **Content Context**: Element types and section information
- **Similarity Scores**: Combined and individual similarity metrics

### Search Quality Improvements
- **Semantic Understanding**: Vector search finds conceptually similar content
- **Keyword Precision**: FTS search finds exact terms and phrases
- **Combined Power**: RRF fusion leverages strengths of both methods
- **Enhanced Filtering**: Content type, page ranges, and temporal filtering

## Robots.txt Monitoring & Testing

The system includes comprehensive tools for monitoring and testing the robots.txt enforcement system:

### System Monitoring

Monitor robots.txt system health and performance:

```bash
# Test overall system health
python tools/robots_monitor.py --test-system

# Collect statistics for specific origins
python tools/robots_monitor.py --collect-stats --origins https://example.com https://httpbin.org

# Generate comprehensive diagnostic report
python tools/robots_monitor.py --diagnose --output diagnostics.json

# Real-time monitoring with detailed logging
python tools/robots_monitor.py --monitor --verbose
```

**Monitoring Features:**
- **Health Checks**: Validates all robots.txt system components
- **Statistics Collection**: Tracks fetch operations, cache hit rates, error rates
- **Performance Metrics**: Response times and throughput analysis
- **Event Recording**: Detailed logging of robots.txt operations
- **Diagnostic Reports**: JSON output for automated monitoring

### Performance Testing

Test robots.txt system performance and efficiency:

```bash
# Test individual components
python tools/robots_performance_test.py --cache-test --iterations 50
python tools/robots_performance_test.py --throttle-test --iterations 100
python tools/robots_performance_test.py --http-test --iterations 20
python tools/robots_performance_test.py --integration-test --iterations 30

# Comprehensive performance analysis
python tools/robots_performance_test.py --origins https://example.com https://httpbin.org

# Save detailed results to file
python tools/robots_performance_test.py --output performance_results.json
```

**Performance Metrics:**
- **Timing Analysis**: Mean, median, min, max response times
- **Throughput Measurement**: Operations per second calculation
- **Success Rates**: Operation success/failure tracking
- **Performance Scoring**: Overall system performance assessment
- **Statistical Analysis**: Comprehensive performance statistics

### Unit Testing

Run comprehensive unit tests for robots.txt components:

```bash
# Run all robots.txt tests
pytest tests/test_robots_txt_core.py -v
pytest tests/test_domain_crawler_robots.py -v
pytest tests/test_url_manager_robots.py -v

# Run specific test categories
pytest tests/test_robots_txt_core.py::TestCrawlerConfig -v
pytest tests/test_domain_crawler_robots.py::TestDomainCrawlerRobots -v
pytest tests/test_url_manager_robots.py::TestURLManagerRobots -v

# Generate coverage report
pytest tests/ --cov=ingestion.url.utils --cov-report=html
```

**Test Coverage:**
- **Core Components**: CrawlerConfig, AsyncHttpClient, OriginThrottle, RobotsCache
- **Integration Testing**: Domain crawler with robots.txt enforcement
- **Database Integration**: URL manager robots.txt support
- **Error Handling**: Edge cases and failure scenarios

For detailed robots.txt configuration and implementation details, see [System Architecture Documentation](architecture.md) and [Configuration Documentation](configuration.md).

## TODOs

- [ ] Image ingestion using TensorFlow object classification with embeddings stored in Milvus.
- [ ] Export search results to external formats.
