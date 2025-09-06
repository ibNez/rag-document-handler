#!/usr/bin/env python3
"""
Drop the existing emails table and let our new manager create the correct one.
"""

import os
from rag_manager.managers.postgres_manager import PostgreSQLManager
from dotenv import load_dotenv

def main():
    """Drop the emails table completely."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get database credentials from environment
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'rag_metadata')
    user = os.getenv('POSTGRES_USER', 'rag_user')
    password = os.getenv('POSTGRES_PASSWORD', 'secure_password')
    
    try:
        mgr = PostgreSQLManager()
        with mgr.get_connection() as conn:
            with conn.cursor() as cur:
                print("🗑️  Dropping emails table...")
                cur.execute("DROP TABLE IF EXISTS emails CASCADE;")
                conn.commit()
                print("✅ Successfully dropped emails table")
        print("🎉 Ready for new table creation!")
        
    except Exception as e:
        print(f"❌ PostgreSQL error or unexpected error: {e}")

if __name__ == "__main__":
    main()
