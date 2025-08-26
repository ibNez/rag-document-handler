# Email Processing Documentation

This document describes the enhanced email ingestion pipeline with robust error handling, corruption detection, and real-time monitoring capabilities.

## Overview

The email processing system features a modular architecture supporting IMAP, POP3, Gmail API, and Exchange protocols with encrypted credential storage, smart batch processing, automatic scheduling, and comprehensive error detection for corrupted emails.

## Enhanced Architecture Components

```
ingestion/email/
├── manager.py                     # PostgreSQL-based account management
├── email_manager_postgresql.py    # Email message storage and retrieval
├── processor.py                   # Email content processing and embedding
├── orchestrator.py                # Email processing coordination with offset tracking
└── connectors/
    ├── imap_connector.py          # Enhanced IMAP with corruption detection
    ├── gmail_connector.py         # Gmail API integration
    └── exchange_connector.py      # Exchange server integration
```

### Component Responsibilities

| Component | Purpose | Key Enhancements |
|-----------|---------|------------------|
| `manager.py` | Account credentials and configuration management | Fresh data fetching for UI |
| `email_manager_postgresql.py` | Email message metadata storage | Accurate statistics with DISTINCT counts |
| `processor.py` | Content extraction, chunking, and embedding | Enhanced validation and error handling |
| `orchestrator.py` | Coordinates processing pipeline | Offset-aware batch processing |
| `connectors/` | Protocol-specific email retrieval implementations | Corrupted email detection and validation |

## Enhanced Processing Pipeline

### 1. Account Configuration (manager.py)
- **Storage**: Email accounts stored in `email_accounts` table via PostgreSQL manager
- **Encryption**: Passwords encrypted using `ingestion/utils/crypto.py` utilities
- **Validation**: Connection tested through appropriate connector before saving
- **Scheduling**: Automatic sync intervals managed by orchestrator
- **Real-time Updates**: Fresh data fetching for edit operations via `/email_accounts` API endpoint

### 2. Enhanced Email Fetching (connectors/)
- **Protocol Support**: IMAP, POP3, Gmail API, Exchange with modular connector architecture
- **Corruption Detection**: Validates Message-ID headers and email structure
- **Offset-Aware Error Logging**: Reports exact positions where corruption occurs
- **Batch Processing**: Smart batching implemented in processor.py to avoid duplicates
- **Offset Management**: Tracks processed emails to resume from last position
- **Error Handling**: Fail-fast approach with detailed logging (no hidden fallbacks)

### 3. Enhanced Content Processing (processor.py)
- **Text Extraction**: Plain text extracted from email body with proper encoding handling
- **Validation**: Message-ID required for all emails (no fallback generation)
- **Chunking**: Text split using DocumentProcessor for optimal embedding
- **Deduplication**: Content hashes prevent duplicate processing across accounts
- **Metadata Preservation**: Subject, sender, recipients, dates maintained in PostgreSQL

### 4. Dual Storage Architecture
- **PostgreSQL**: Email metadata stored via `email_manager_postgresql.py`
- **Milvus**: Text embeddings stored with mapped schema fields
- **Consistency**: Both operations wrapped in error handling for data integrity
- **Corruption Detection**: Validates Message-ID headers before storage
- **Offset Tracking**: Maintains processing position for resuming interrupted operations

## Enhanced Database Schema

### Email Accounts Table
```sql
email_accounts:
  id (PRIMARY KEY)
  email_address (UNIQUE)
  password (ENCRYPTED)
  imap_server
  imap_port
  smtp_server
  smtp_port
  protocol
  username
  sync_interval
  last_sync
  enabled
  offset_position (NEW - tracks processing position for resuming)
```

### Enhanced Email Messages Storage
```sql
emails:
  id (PRIMARY KEY)
  message_id (UNIQUE, REQUIRED)  # No fallback generation
  subject
  sender
  recipients
  timestamp
  email_account_id (FOREIGN KEY)
  processed
  created_at
  content_hash (for deduplication)
  validation_status (NEW - corruption detection results)
```

## Enhanced Error Handling & Monitoring

### Corrupted Email Detection
The system now includes robust validation for corrupted emails with offset-aware logging:

```python
def _parse_email_with_offset(self, msg_data, offset):
    """Enhanced email parsing with corruption detection and offset logging."""
    try:
        email_msg = email.message_from_bytes(msg_data)
        
        # Validate Message-ID (required, no fallback generation)
        message_id = email_msg.get('Message-ID')
        if not message_id:
            logger.error(f"Email at offset {offset} missing required Message-ID header")
            return None
            
        # Validate basic email structure
        if not email_msg.get('From') or not email_msg.get('Date'):
            logger.error(f"Email at offset {offset} missing required headers")
            return None
            
        return email_msg
        
    except Exception as e:
        logger.error(f"Failed to parse email at offset {offset}: {str(e)}")
        return None
```

### Enhanced Error Logging Features
- **Offset Position Tracking**: Exact location of corrupted emails reported in logs
- **Detailed Validation**: Message-ID and header validation with specific error codes
- **Fail-Fast Approach**: No hidden fallbacks that mask underlying corruption issues
- **Comprehensive Monitoring**: Real-time error tracking with detailed context

### Real-Time Dashboard Monitoring
- **Auto-Refresh System**: All dashboard panels refresh every 10 seconds automatically
- **Fresh Data API**: Edit modals fetch current database values via AJAX endpoints
- **Live Statistics**: Email counts and processing status updated in real-time
- **Error Visibility**: Processing errors immediately visible in dashboard interface
- **Offset Management**: Manual offset reset capability through edit interface

## Schema Mapping

### Problem Statement
The Milvus collection uses a unified schema designed for documents, but email data has different field structures. The email processor must map email-specific fields to the document schema.

### Field Mapping Table

| Email Data | PostgreSQL Field | Milvus Field | Mapping Logic |
|------------|------------------|--------------|---------------|
| Message ID | `message_id` | `document_id` | Direct mapping |
| From Address | `from_addr` | `source` | Prefix: `email:{from_addr}` |
| Subject | `subject` | `topic` | Direct mapping |
| Chunk Index | N/A | `page` | Sequential chunk number |
| Generated ID | N/A | `chunk_id` | Format: `{message_id}:{chunk_idx}` |
| Content Type | N/A | `category` | Fixed value: "email" |
| Content Hash | N/A | `content_hash` | Format: `email_{message_id}_{chunk_idx}` |
| Chunk Length | N/A | `content_length` | Character count of text chunk |
| Text Chunk | `body_text` | `text` | Processed text content |

### Mapping Implementation

```python
def map_email_to_milvus_schema(record: Dict[str, Any], chunk_idx: int, text_chunk: str) -> Dict[str, Any]:
    """Map email record to Milvus document schema."""
    message_id = record.get("message_id")
    subject = record.get("subject", "")
    from_addr = record.get("from_addr", "")
    
    return {
        "document_id": message_id,
        "source": f"email:{from_addr}",
        "page": chunk_idx,
        "chunk_id": f"{message_id}:{chunk_idx}",
        "topic": subject,
        "category": "email",
        "content_hash": f"email_{message_id}_{chunk_idx}",
        "content_length": len(text_chunk),
        # Backward compatibility fields (stored in metadata)
        "message_id": message_id,
        "subject": subject,
        "from_addr": from_addr,
        "to_addrs": record.get("to_addrs"),
        "date_utc": record.get("date_utc"),
        "server_type": record.get("server_type"),
    }
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EMAIL_ENCRYPTION_KEY` | Fernet key for password encryption | None | Yes |
| `SCHEDULER_POLL_SECONDS_BUSY` | Polling interval when active | `10` | No |
| `SCHEDULER_POLL_SECONDS_IDLE` | Polling interval when idle | `30` | No |

### Account Configuration Fields

```python
{
    "account_name": "Friendly name for the account",
    "server_type": "imap",  # or "pop3"
    "server": "imap.gmail.com",
    "port": 993,
    "email_address": "user@example.com", 
    "password": "encrypted_password",
    "mailbox": "INBOX",
    "batch_limit": 50,
    "use_ssl": True,
    "refresh_interval_minutes": 60
}
```

## API Endpoints

### Account Management

```http
POST /api/email-accounts
Content-Type: application/json

{
    "account_name": "My Gmail",
    "server_type": "imap",
    "server": "imap.gmail.com",
    "port": 993,
    "email_address": "user@gmail.com",
    "password": "app_password",
    "use_ssl": true,
    "refresh_interval_minutes": 60
}
```

```http
GET /api/email-accounts
# Returns list of configured accounts (passwords excluded)

PUT /api/email-accounts/{id}
# Update account configuration

DELETE /api/email-accounts/{id}
# Remove account and stop processing
```

### Processing Status

```http
GET /api/email-status
# Returns processing statistics and account status

POST /api/email-refresh/{account_id}
# Trigger immediate sync for specific account
```

## Troubleshooting

### Common Issues

#### 1. Milvus Schema Mismatch Error
```
DataNotMatchException: The data doesn't match with schema fields, expect 10 list, got 4
```

**Cause**: Email processor using PyMilvus native API instead of LangChain interface
**Solution**: Ensure `add_texts` method is used for schema mapping

#### 2. Connection Refused Error
```
Error in smart batch processing: [Errno 61] Connection refused
```

**Cause**: Mail server not accessible or incorrect host/port
**Solution**: Verify server settings and network connectivity

#### 3. Authentication Failures
```
IMAP login failed: status=NO
```

**Cause**: Invalid credentials or app passwords not enabled
**Solution**: Check email account settings and enable app passwords if using 2FA

#### 4. Encryption Key Errors
```
Invalid token or key
```

**Cause**: `EMAIL_ENCRYPTION_KEY` changed or corrupted
**Solution**: Re-encrypt all passwords with new key

### Diagnostic Commands

```bash
# Test email server connectivity
telnet imap.gmail.com 993

# Check PostgreSQL email tables
psql -d rag_metadata -c "SELECT account_name, server, port, last_synced FROM email_accounts;"

# Verify Milvus collection schema
python3 -c "
from pymilvus import Collection
col = Collection('documents')
for field in col.schema.fields:
    print(f'{field.name}: {field.dtype}')
"

# Check email processing logs
tail -f app.log | grep "email.processor"
```

### Performance Optimization

#### 1. Batch Size Tuning
- **Small Batches (10-25)**: Better for real-time processing, higher overhead
- **Large Batches (50-100)**: Better throughput, higher memory usage
- **Optimal**: 25-50 emails per batch for most use cases

#### 2. Refresh Intervals
- **High Frequency (5-15 min)**: For important accounts, higher server load
- **Standard (60-120 min)**: Balanced approach for most accounts  
- **Low Frequency (1440 min)**: For archival or low-activity accounts

#### 3. Connection Management
- **SSL/TLS**: Required for security, slight performance impact
- **Connection Pooling**: Reuse connections when possible
- **Timeout Settings**: Balance between reliability and responsiveness

## Security Considerations

### Password Management
- **Encryption**: All passwords encrypted at rest using Fernet
- **Key Rotation**: Support for re-encrypting with new keys
- **Access Control**: Database credentials separate from email credentials

### Network Security
- **TLS/SSL**: All email connections encrypted in transit
- **Certificate Validation**: Verify server certificates to prevent MITM
- **Network Isolation**: Email processing isolated from web interface

### Data Privacy
- **Content Processing**: Email content processed locally, not sent to external APIs
- **Metadata Storage**: Minimal metadata stored, full emails can be excluded
- **Deletion**: Complete removal of account data when account deleted

## Monitoring and Alerts

### Key Metrics
- **Processing Rate**: Emails processed per minute
- **Error Rate**: Failed email processing percentage  
- **Connection Health**: Mail server connectivity status
- **Storage Growth**: PostgreSQL and Milvus storage usage

### Alert Conditions
- **High Error Rate**: > 10% failed email processing
- **Connection Failures**: Mail server unreachable for > 30 minutes
- **Schema Errors**: Milvus insertion failures
- **Storage Issues**: Database storage > 90% capacity

### Log Analysis
```bash
# Error patterns
grep -E "(ERROR|WARN)" app.log | grep email

# Processing statistics  
grep "Smart batch processing complete" app.log | tail -10

# Connection issues
grep "Connection refused\|timeout\|SSL" app.log | tail -20
```
