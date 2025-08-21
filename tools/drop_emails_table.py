#!/usr/bin/env python3
"""
Drop the existing emails table and let our new manager create the correct one.
"""

import os
import psycopg2
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
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        with conn.cursor() as cur:
            print("🗑️  Dropping emails table...")
            cur.execute("DROP TABLE IF EXISTS emails CASCADE;")
            conn.commit()
            print("✅ Successfully dropped emails table")
        
        conn.close()
        print("🎉 Ready for new table creation!")
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
