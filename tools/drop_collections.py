#!/usr/bin/env python3
"""
Drop Milvus collections to force recreation with new schema.

This tool drops Milvus collections so they can be recreated with proper
deduplication handling for content_hash uniqueness.
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """Drop specified Milvus collections completely."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Milvus connection details from environment
    milvus_host = os.getenv('MILVUS_HOST', 'localhost')
    milvus_port = os.getenv('MILVUS_PORT', '19530')
    
    # Default collection names (can be overridden by command line)
    collections_to_drop = [
        'rag_knowledgebase',
        'rag_knowledgebase_emails'
    ]
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage: python drop_collections.py [collection_names...]")
            print("  No args: Drop default collections")
            print("  With args: Drop only specified collections")
            print(f"  Default collections: {', '.join(collections_to_drop)}")
            return
        else:
            # Use specified collections from command line
            collections_to_drop = sys.argv[1:]
    
    try:
        # Import Milvus after checking args to avoid import errors in help
        from pymilvus import connections, utility
    except ImportError:
        print("❌ Error: pymilvus not installed. Install with: pip install pymilvus")
        sys.exit(1)
    
    try:
        # Connect to Milvus
        print(f"🔌 Connecting to Milvus: {milvus_host}:{milvus_port}")
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port
        )
        
        print(f"🗑️  Checking and dropping {len(collections_to_drop)} collection(s)...")
        
        dropped_count = 0
        for collection_name in collections_to_drop:
            try:
                # Check if collection exists
                if utility.has_collection(collection_name):
                    print(f"   Dropping collection: {collection_name}")
                    utility.drop_collection(collection_name)
                    dropped_count += 1
                    print(f"   ✅ Dropped: {collection_name}")
                else:
                    print(f"   ⚠️  Collection does not exist: {collection_name}")
            except Exception as e:
                print(f"   ❌ Failed to drop {collection_name}: {e}")
        
        print(f"✅ Successfully dropped {dropped_count} collection(s)")
        print("🎉 Collections dropped! Restart the application to recreate with proper deduplication.")
        print("   New collections will have:")
        print("   - Proper content_hash handling")
        print("   - Application-level deduplication")
        print("   - Fresh vector index")
        
    except Exception as e:
        print(f"❌ Milvus error: {e}")
        print("💡 Make sure Milvus is running and accessible")
        sys.exit(1)
    finally:
        try:
            connections.disconnect("default")
        except:
            pass

if __name__ == "__main__":
    main()
