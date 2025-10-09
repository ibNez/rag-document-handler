# Installation

This guide covers setting up the RAG Knowledgebase Manager with the refactored architecture.

## Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Git

## Quick Setup Options

### Interactive Setup (Recommended)
```bash
git clone <repository-url>
cd rag-document-handler
./setup.sh
```

### Automated Setup
```bash
./setup.sh --all        # Install everything without prompts
./setup.sh --dev        # Development mode (skip webui container)
./setup.sh --help       # Show installation options
```

### Development Mode (Recommended for Development)
For active development where you want to run the Flask app locally:
```bash
./setup.sh --dev                # Start only infrastructure containers
source .venv/bin/activate       # Always source environment (DEVELOPMENT_RULES.md)
./start.sh                      # Use proper startup script with service checks
```

The setup script will:
- Create Python virtual environment
- Install all dependencies (including PostgreSQL drivers)
- Create directory structure
- Start Docker containers (Milvus + PostgreSQL)
- Test database connections
- Configure environment files

## Startup Process

The refactored application uses `./start.sh` which:
- Activates the virtual environment
- Checks Milvus and PostgreSQL connectivity
- Verifies Docker container status
- Starts the Flask application with proper configuration
- Displays service status and access URLs

## Docker Compose Deployment

The repository includes a `docker-compose.yml` for running the complete application stack.

### Full Stack (Production-like)
```bash
cp .env.example .env  # Configure as needed
docker compose up -d
```

### Infrastructure Only (Development)
```bash
./setup.sh --dev     # Recommended for development
# OR manually:
docker compose up postgres milvus -d
source .venv/bin/activate
python app.py
```

This starts the infrastructure containers (PostgreSQL + Milvus) and exposes:
- Web interface: [http://localhost:3000](http://localhost:3000)
- PostgreSQL: `localhost:5432`
- Milvus: `localhost:19530`

## Manual Installation

1. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Python dependencies**
   ```bash
   pip install rag-document-handler
   ```
3. **Start infrastructure services**
   ```bash
   docker compose up postgres milvus -d
   ```
4. **Start the application**
   ```bash
   python app.py
   ```

## Uninstallation

### Safe Removal
```bash
./uninstall.sh          # Interactive removal
./uninstall.sh --dry-run # Preview what would be removed
./uninstall.sh --help   # Show removal options
```

The uninstall script safely removes only project-specific resources:
- RAG Knowledgebase Manager containers and volumes
- Project virtual environment (.venv)
- Database files and logs
- Uploaded/staging files
- Temporary files

Other Docker containers and system files are preserved.

## Next Steps

- Configure environment variables via `.env` (see root `README.md`)
- Visit [Usage Guide](usage.md) to interact with the application  
- PostgreSQL database stores metadata with JSONB for flexible attributes
- Milvus vector database handles embeddings and similarity search

## Database Architecture

The application uses a dual database architecture:
- **PostgreSQL**: Document metadata, URLs, email data, analytics
- **Milvus**: Vector embeddings and similarity search

## Email Integration Configuration

The application can optionally synchronize IMAP inboxes. Define the following environment variables in your `.env` file:

| Variable | Default | Description |
| --- | --- | --- |
| `EMAIL_ENABLED` | `false` | Enable periodic email sync |
| `IMAP_HOST` | _(empty)_ | IMAP server hostname |
| `IMAP_PORT` | `993` | IMAP server port used for the connection |
| `IMAP_USERNAME` | _(empty)_ | IMAP account email address |
| `IMAP_PASSWORD` | _(empty)_ | IMAP account password |
| `IMAP_MAILBOX` | `INBOX` | Mailbox to read from |
| `IMAP_BATCH_LIMIT` | `50` | Maximum messages fetched per cycle |
| `IMAP_USE_SSL` | `true` | Use SSL/TLS for IMAP connection |
| `EMAIL_SYNC_INTERVAL_SECONDS` | `300` | Interval between sync cycles in seconds |

### Email account secret key

Passwords for stored email accounts are encrypted using a symmetric key. The
application expects a Fernet key to be supplied via the
`EMAIL_ENCRYPTION_KEY` environment variable. Generate a key and set it in your
environment (or `.env` file) before running the app:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
export EMAIL_ENCRYPTION_KEY="<output-from-command>"
```

Keep this value secret and **do not** commit it to version control. If the key
is rotated, previously stored passwords will need to be re-entered.

## Production Deployment

### Docker Production Deployment (Recommended)

For production environments, use the Docker deployment with HTTPS support:

```bash
# Start production services with HTTPS
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Stop services
docker compose -f docker-compose.prod.yml down
```

**Production Features:**
- **HTTPS by default** on port 443
- **HTTP to HTTPS redirect** on port 80
- **Production WSGI server** (Gunicorn) with multiple workers
- **Automatic SSL certificate generation**
- **Resource optimization** and security hardening

**Production URLs:**
- **Main Application**: https://localhost:443 (HTTPS)
- **HTTP Redirect**: http://localhost:80 → https://localhost:443

### Native WSGI Deployment

For environments where Docker is not available, use the production scripts:

```bash
# Start in production mode
./start_production.sh

# Check status
./status_production.sh

# Stop gracefully
./stop_production.sh
```

**Core Production Files:**
- `wsgi.py` - WSGI entry point for production servers
- `start_production.sh` - Production start script using Gunicorn
- `stop_production.sh` - Graceful shutdown script
- `status_production.sh` - Status monitoring script
- `rag-knowledgebase.service` - Systemd service unit file

### Systemd Service (Recommended for Production)

For system-level service management:

1. **Copy service file**:
```bash
sudo cp rag-knowledgebase.service /etc/systemd/system/
sudo systemctl daemon-reload
```

2. **Enable and start**:
```bash
sudo systemctl enable rag-knowledgebase
sudo systemctl start rag-knowledgebase
```

3. **Monitor**:
```bash
sudo systemctl status rag-knowledgebase
sudo journalctl -u rag-knowledgebase -f
```

