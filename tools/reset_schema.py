#!/usr/bin/env python3
"""
Comprehensive tool to reset the entire database schema for idempotency fixes.

This tool:
1. Drops PostgreSQL tables 
2. Drops Milvus collections
3. Allows selective reset of components

Run this before restarting the application to get fresh schema with 
UNIQUE constraints on content_hash fields.
"""

import os
import sys
import psycopg2
from dotenv import load_dotenv

def drop_postgres_tables(tables_to_drop):
    """Drop specified PostgreSQL tables."""
    # Load environment variables
    load_dotenv()
    
    # Get database credentials from environment
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'rag_metadata')
    user = os.getenv('POSTGRES_USER', 'rag_user')
    password = os.getenv('POSTGRES_PASSWORD', 'secure_password')
    
    try:
        print(f"🔌 Connecting to PostgreSQL: {user}@{host}:{port}/{database}")
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        with conn.cursor() as cur:
            print(f"🗑️  Dropping {len(tables_to_drop)} PostgreSQL table(s)...")
            
            for table in tables_to_drop:
                print(f"   Dropping table: {table}")
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            
            conn.commit()
            print("✅ Successfully dropped PostgreSQL tables")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL error: {e}")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL unexpected error: {e}")
        return False

def drop_milvus_collections(collections_to_drop):
    """Drop specified Milvus collections."""
    # Load environment variables
    load_dotenv()
    
    # Get Milvus connection details from environment
    milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port = os.getenv('MILVUS_PORT', '19530')
    
    try:
        from pymilvus import connections, utility
    except ImportError:
        print("❌ Error: pymilvus not installed. Install with: pip install pymilvus")
        return False
    
    try:
        print(f"🔌 Connecting to Milvus: {milvus_host}:{milvus_port}")
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port
        )
        
        print(f"🗑️  Dropping {len(collections_to_drop)} Milvus collection(s)...")
        
        dropped_count = 0
        for collection_name in collections_to_drop:
            try:
                if utility.has_collection(collection_name):
                    print(f"   Dropping collection: {collection_name}")
                    utility.drop_collection(collection_name)
                    dropped_count += 1
                    print(f"   ✅ Dropped: {collection_name}")
                else:
                    print(f"   ⚠️  Collection does not exist: {collection_name}")
            except Exception as e:
                print(f"   ❌ Failed to drop {collection_name}: {e}")
        
        print(f"✅ Successfully dropped {dropped_count} Milvus collection(s)")
        return True
        
    except Exception as e:
        print(f"❌ Milvus error: {e}")
        print("💡 Make sure Milvus is running and accessible")
        return False
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

def main():
    """Main function to handle command line arguments and orchestrate the reset."""
    
    # Default configuration
    postgres_tables = ['emails', 'documents', 'urls', 'email_accounts']
    milvus_collections = ['rag_knowledgebase', 'rag_knowledgebase_emails']
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage: python reset_schema.py [options]")
            print("")
            print("Options:")
            print("  --help, -h           Show this help message")
            print("  --postgres-only      Drop only PostgreSQL tables")
            print("  --milvus-only        Drop only Milvus collections") 
            print("  --tables TABLE1,TABLE2   Drop specific PostgreSQL tables")
            print("  --collections COL1,COL2  Drop specific Milvus collections")
            print("")
            print(f"Default PostgreSQL tables: {', '.join(postgres_tables)}")
            print(f"Default Milvus collections: {', '.join(milvus_collections)}")
            return
        
        postgres_only = '--postgres-only' in sys.argv
        milvus_only = '--milvus-only' in sys.argv
        
        # Parse custom tables
        for i, arg in enumerate(sys.argv):
            if arg == '--tables' and i + 1 < len(sys.argv):
                postgres_tables = sys.argv[i + 1].split(',')
            elif arg == '--collections' and i + 1 < len(sys.argv):
                milvus_collections = sys.argv[i + 1].split(',')
    else:
        postgres_only = False
        milvus_only = False
    
    print("🔄 RAG Knowledgebase Manager - Schema Reset Tool")
    print("=" * 50)
    
    success = True
    
    # Drop PostgreSQL tables
    if not milvus_only:
        print("\n📊 POSTGRESQL RESET")
        print("-" * 20)
        success &= drop_postgres_tables(postgres_tables)
    
    # Drop Milvus collections
    if not postgres_only:
        print("\n🚀 MILVUS RESET")
        print("-" * 15)
        success &= drop_milvus_collections(milvus_collections)
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 SCHEMA RESET COMPLETED SUCCESSFULLY!")
        print("")
        print("Next steps:")
        print("1. Restart the RAG Knowledgebase Manager application")
        print("2. Tables and collections will be recreated with:")
        print("   - UNIQUE constraints on content_hash")
        print("   - Proper deduplication logic")
        print("   - Fresh indexes and schema")
        print("3. Test URL refresh - should now be idempotent!")
    else:
        print("❌ SCHEMA RESET FAILED!")
        print("Check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
