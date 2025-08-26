# Configuration Documentation

This document provides comprehensive information about all configuration variables, environment settings, and deployment options for the RAG Knowledgebase Manager system.

## Environment Variables

### Core Application Settings

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `FLASK_HOST` | Flask application host | `0.0.0.0` | No | `localhost` |
| `FLASK_PORT` | Flask application port | `5000` | No | `3000` |
| `FLASK_DEBUG` | Enable Flask debug mode | `False` | No | `True` |
| `MAX_CONTENT_LENGTH` | Maximum file upload size in bytes | `16777216` | No | `104857600` (100MB) |
| `UPLOAD_FOLDER` | Directory for file uploads | `staging` | No | `uploads` |

### Database Connection Settings

#### PostgreSQL (Metadata Database)
| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `POSTGRES_HOST` | PostgreSQL server hostname | `localhost` | No | `postgres.example.com` |
| `POSTGRES_PORT` | PostgreSQL server port | `5432` | No | `5433` |
| `POSTGRES_DB` | PostgreSQL database name | `rag_metadata` | No | `knowledge_base` |
| `POSTGRES_USER` | PostgreSQL username | `postgres` | No | `rag_user` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `postgres` | No | `secure_password` |

#### Milvus (Vector Database)
| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `MILVUS_HOST` | Milvus server hostname | `localhost` | No | `milvus.example.com` |
| `MILVUS_PORT` | Milvus server port | `19530` | No | `19531` |
| `COLLECTION_NAME` | Milvus collection name | `documents` | No | `knowledge_base` |
| `VECTOR_DIM` | Embedding vector dimension | `384` | No | `768` |

### Email Processing Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `EMAIL_ENCRYPTION_KEY` | Fernet key for email password encryption | None | **Yes** | `generated_fernet_key` |
| `SCHEDULER_POLL_SECONDS_BUSY` | Polling interval when email processing is active | `10` | No | `5` |
| `SCHEDULER_POLL_SECONDS_IDLE` | Polling interval when email processing is idle | `30` | No | `60` |
| `EMAIL_SYNC_INTERVAL_SECONDS` | Background email sync interval | `300` | No | `600` |

### AI/ML Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `OLLAMA_HOST` | Ollama service hostname | `localhost` | No | `ollama.example.com` |
| `OLLAMA_PORT` | Ollama service port | `11434` | No | `11435` |
| `CLASSIFICATION_MODEL` | LLM model for query classification | `llama3.2:3b` | No | `llama3.1:7b` |
| `OLLAMA_CLASSIFICATION_HOST` | Dedicated Ollama host for classification | `localhost` | No | `classification.example.com` |
| `EMBEDDING_MODEL` | Model for text embeddings | `mxbai-embed-large` | No | `nomic-embed-text` |

### URL Processing Configuration

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `SNAPSHOT_DIR` | Directory for URL snapshot storage | `uploaded/snapshots` | No | `data/snapshots` |
| `DEFAULT_CRAWL_INTERVAL_MINUTES` | Default crawl interval for URLs | `1440` | No | `720` (12 hours) |
| `MAX_CRAWL_DEPTH` | Maximum crawl depth for URLs | `3` | No | `5` |

## Configuration Files

### Environment File (.env)

The project includes a comprehensive `.env.example` file with all available configuration options. To set up your environment:

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Customize the values** in your `.env` file according to your deployment needs.

3. **Required Configuration:** At minimum, you must set:
   ```bash
   # Generate an encryption key for email passwords
   EMAIL_ENCRYPTION_KEY=your_generated_fernet_key
   
   # Database passwords (if different from defaults)
   POSTGRES_PASSWORD=your_secure_password
   ```

4. **Generate the required encryption key:**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

The `.env.example` file contains all configuration variables with their default values and helpful comments. Key sections include:

- **Database Configuration** (PostgreSQL & Milvus)
- **Flask Web Server Settings**
- **File Upload & Processing**
- **AI/ML Service Configuration** (Ollama models)
- **Email Processing Settings**
- **URL Crawling & Snapshot Configuration**
- **Background Task Scheduling**

**Note:** The `.env` file is not tracked in version control for security reasons. Always use the `.env.example` file as your template and never commit sensitive configuration values.

### Email Account Configuration

Email accounts are configured through the web interface with the following fields:

```python
{
    "account_name": "Friendly display name",
    "server_type": "imap",  # Options: "imap", "pop3", "exchange"
    "server": "imap.gmail.com",
    "port": 993,
    "email_address": "user@example.com",
    "password": "app_password",  # Encrypted automatically
    "mailbox": "INBOX",
    "batch_limit": 50,
    "use_ssl": True,
    "refresh_interval_minutes": 60,
    "offset_position": 0  # For resuming interrupted processing
}
```

## Security Configuration

### Email Password Encryption

Generate a secure Fernet key for email password encryption:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set this as your `EMAIL_ENCRYPTION_KEY` environment variable.

### Database Security

#### PostgreSQL Security
- Use strong passwords for PostgreSQL users
- Consider creating dedicated database users with limited privileges
- Enable SSL connections in production environments

#### Milvus Security
- Configure authentication if deploying in production
- Use network isolation and firewall rules
- Consider enabling TLS for Milvus connections

## Development vs Production Configuration

### Development Mode
```bash
# Use default local settings
FLASK_DEBUG=True
POSTGRES_HOST=localhost
MILVUS_HOST=localhost
```

### Production Mode
```bash
# Use production-ready settings
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
POSTGRES_HOST=production-postgres-host
MILVUS_HOST=production-milvus-host
# Add SSL/TLS configuration
# Add authentication credentials
```

## Configuration Validation

The application validates configuration on startup:

1. **Required Variables**: Checks for mandatory environment variables
2. **Database Connectivity**: Tests PostgreSQL and Milvus connections
3. **Directory Structure**: Verifies upload and log directories exist
4. **Service Dependencies**: Confirms Ollama and other services are accessible

### Configuration Troubleshooting

#### Common Issues

**Missing EMAIL_ENCRYPTION_KEY**
```bash
Error: EMAIL_ENCRYPTION_KEY environment variable is required
Solution: Generate and set a Fernet key as shown above
```

**Database Connection Failed**
```bash
Error: Could not connect to PostgreSQL/Milvus
Solution: Verify database services are running and connection settings are correct
```

**File Upload Issues**
```bash
Error: File too large / Upload directory not writable
Solution: Check MAX_CONTENT_LENGTH and UPLOAD_FOLDER permissions
```

## Configuration Best Practices

1. **Environment Variables**: Use `.env` files for local development, proper environment management for production
2. **Security**: Never commit sensitive values to version control
3. **Validation**: Test configuration changes in development before production deployment
4. **Documentation**: Keep this configuration documentation updated with any new variables
5. **Backup**: Document your production configuration for disaster recovery

## Related Documentation

- [Installation Guide](installation.md) - Setup and deployment instructions
- [Architecture Documentation](architecture.md) - System design and component interactions
- [Database Schema](database-schema.md) - Database structure and relationships
- [Email Processing](email-processing.md) - Email system configuration details
