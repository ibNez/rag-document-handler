# Database Management Tools

This directory contains utility tools for managing the RAG Document Handler database schema and collections.

## Tools Overview

### 🗑️ `drop_tables.py`
Drops PostgreSQL tables to force recreation with new schema.

```bash
# Drop all main tables
python tools/drop_tables.py

# Drop specific tables
python tools/drop_tables.py emails documents

# Get help
python tools/drop_tables.py --help
```

### 🚀 `drop_collections.py` 
Drops Milvus collections to force recreation.

```bash
# Drop default collections
python tools/drop_collections.py

# Drop specific collections
python tools/drop_collections.py rag_knowledgebase custom_collection

# Get help
python tools/drop_collections.py --help
```

### 🔄 `reset_schema.py`
Comprehensive tool that drops both PostgreSQL tables and Milvus collections.

```bash
# Reset everything
python tools/reset_schema.py

# Reset only PostgreSQL
python tools/reset_schema.py --postgres-only

# Reset only Milvus
python tools/reset_schema.py --milvus-only

# Custom tables and collections
python tools/reset_schema.py --tables emails,urls --collections rag_knowledgebase

# Get help
python tools/reset_schema.py --help
```

### 📧 `drop_emails_table.py` (Legacy)
Original tool for dropping just the emails table.

## Current Issue: Content Hash Idempotency

**Problem**: URL refresh is creating duplicate chunks instead of being idempotent.

**Solution**: 
1. **Drop existing tables/collections** with tools above
2. **Restart application** - will recreate with UNIQUE constraints on `content_hash`  
3. **Test URL refresh** - should now be idempotent

## Quick Fix Workflow

```bash
# 1. Reset everything
python tools/reset_schema.py

# 2. Restart the application
python main.py

# 3. Test URL refresh functionality
```

## Schema Changes

The new schema includes:
- **UNIQUE constraint on `content_hash`** in emails table
- **Centralized table creation** in PostgreSQLManager
- **Application-level deduplication** for Milvus
- **Proper indexes** for performance

## Environment Variables

Make sure these are set in your `.env` file:

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_metadata
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=secure_password

# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```
