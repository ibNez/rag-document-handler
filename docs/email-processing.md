# Email Processing Documentation

This document describes the email ingestion pipeline, schema mapping, and troubleshooting procedures.

## Overview

The email processing system supports IMAP and POP3 protocols with encrypted credential storage, smart batch processing, and automatic scheduling.

## Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mail Server   │    │   PostgreSQL    │    │     Milvus      │
│   (IMAP/POP3)   │    │   (Metadata)    │    │   (Vectors)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                 ┌─────────────────────────────────┐
                 │      Email Processor            │
                 │   - Account Manager             │
                 │   - IMAP/POP3 Connector         │
                 │   - Text Chunking               │
                 │   - Schema Mapping              │
                 │   - Embedding Generation        │
                 └─────────────────────────────────┘
```

## Processing Pipeline

### 1. Account Configuration
- **Storage**: Email accounts stored in `email_accounts` table
- **Encryption**: Passwords encrypted using `EMAIL_ENCRYPTION_KEY` (Fernet)
- **Validation**: Connection tested before saving account
- **Scheduling**: Automatic sync intervals configured per account

### 2. Email Fetching
- **Protocols**: IMAP, POP3 with SSL/TLS support
- **Batch Processing**: Smart batching to avoid duplicate processing
- **Offset Management**: Tracks processed emails to resume from last position
- **Error Handling**: Retries with exponential backoff on connection issues

### 3. Content Processing
- **Text Extraction**: Plain text extracted from email body
- **Chunking**: Text split into overlapping chunks for optimal embedding
- **Deduplication**: Content hashes prevent duplicate processing
- **Metadata Extraction**: Subject, sender, recipients, dates preserved

### 4. Dual Storage
- **PostgreSQL**: Email metadata stored in `email_messages` table
- **Milvus**: Text embeddings stored with mapped schema fields
- **Consistency**: Both operations wrapped in error handling for data integrity

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
| `USE_POSTGRESQL_URL_MANAGER` | Enable PostgreSQL backend | `false` | No |
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
