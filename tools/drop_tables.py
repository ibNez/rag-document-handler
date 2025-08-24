#!/usr/bin/env python3
"""
Drop PostgreSQL tables to force recreation with new schema and UNIQUE constraints.

This tool drops tables so they can be recreated with updated schema including:
- UNIQUE constraints on content_hash fields for idempotency
- Centralized table creation in PostgreSQLManager
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

def main():
    """Drop specified PostgreSQL tables completely."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get database credentials from environment
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'rag_metadata')
    user = os.getenv('POSTGRES_USER', 'rag_user')
    password = os.getenv('POSTGRES_PASSWORD', 'secure_password')
    
    # Tables to drop (in order - respecting foreign key dependencies)
    tables_to_drop = [
        'emails',
        'documents', 
        'urls',
        'email_accounts'
    ]
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage: python drop_tables.py [table_names...]")
            print("  No args: Drop all main tables")
            print("  With args: Drop only specified tables")
            print(f"  Available tables: {', '.join(tables_to_drop)}")
            return
        else:
            # Use specified tables from command line
            tables_to_drop = [table for table in sys.argv[1:] if table in tables_to_drop]
            if not tables_to_drop:
                print("❌ No valid table names provided")
                print(f"Available tables: {', '.join(['emails', 'documents', 'urls', 'email_accounts'])}")
                return
    
    try:
        # Connect to PostgreSQL
        print(f"🔌 Connecting to PostgreSQL: {user}@{host}:{port}/{database}")
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        with conn.cursor() as cur:
            print(f"🗑️  Dropping {len(tables_to_drop)} table(s)...")
            
            for table in tables_to_drop:
                print(f"   Dropping table: {table}")
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            
            conn.commit()
            print("✅ Successfully dropped all specified tables")
        
        conn.close()
        print("🎉 Tables dropped! Restart the application to recreate with new schema.")
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
