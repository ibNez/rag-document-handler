#!/usr/bin/env python3
"""
Cleanup Orphaned Embeddings Tool

This tool helps identify and remove orphaned embeddings in Milvus collections
that no longer have corresponding files in the uploaded folder.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_manager.core.config import Config
from rag_manager.managers.milvus_manager import MilvusManager
from pymilvus import Collection, utility, connections


def get_file_sources_from_milvus(collection_name: str, config: Config) -> set:
    """Get all unique source filenames from Milvus collection."""
    try:
        # Establish connection to Milvus
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist")
            return set()
            
        col = Collection(collection_name)
        col.load()
        
        # Query all source values
        query_results = col.query(
            expr="source != ''",
            output_fields=["source"],
            limit=10000  # Adjust if you have more than 10k unique sources
        )
        
        sources = {result.get("source") for result in query_results if result.get("source")}
        return sources
        
    except Exception as e:
        print(f"Error querying Milvus collection: {e}")
        return set()


def get_uploaded_files(uploaded_folder: str) -> set:
    """Get all filenames from the uploaded folder."""
    try:
        uploaded_path = Path(uploaded_folder)
        if not uploaded_path.exists():
            print(f"Uploaded folder does not exist: {uploaded_folder}")
            return set()
            
        files = {f.name for f in uploaded_path.iterdir() if f.is_file()}
        return files
        
    except Exception as e:
        print(f"Error reading uploaded folder: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(description="Cleanup orphaned embeddings in Milvus")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--collection", default="documents",
                       help="Milvus collection name (default: documents)")
    parser.add_argument("--confirm", action="store_true",
                       help="Confirm deletion of orphaned embeddings")
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    print(f"Checking for orphaned embeddings in collection: {args.collection}")
    print(f"Uploaded folder: {config.UPLOADED_FOLDER}")
    print()
    
    # Get sources from Milvus
    print("Querying Milvus for document sources...")
    milvus_sources = get_file_sources_from_milvus(args.collection, config)
    print(f"Found {len(milvus_sources)} unique sources in Milvus")
    
    # Get files from uploaded folder
    print("Scanning uploaded folder...")
    uploaded_files = get_uploaded_files(config.UPLOADED_FOLDER)
    print(f"Found {len(uploaded_files)} files in uploaded folder")
    print()
    
    # Find orphaned sources (in Milvus but not in uploaded folder)
    orphaned_sources = milvus_sources - uploaded_files
    
    if not orphaned_sources:
        print("✅ No orphaned embeddings found!")
        return
    
    print(f"🔍 Found {len(orphaned_sources)} orphaned sources:")
    for source in sorted(orphaned_sources):
        print(f"  - {source}")
    print()
    
    if args.dry_run:
        print("🔬 DRY RUN: Would delete embeddings for the above sources")
        print("Run without --dry-run and with --confirm to actually delete")
        return
    
    if not args.confirm:
        print("⚠️  To delete orphaned embeddings, run with --confirm flag")
        print("Example: python cleanup_orphaned_embeddings.py --confirm")
        return
    
    # Initialize Milvus manager and delete orphaned embeddings
    print("Initializing Milvus manager...")
    try:
        milvus_manager = MilvusManager(config)
        
        deleted_total = 0
        for source in orphaned_sources:
            print(f"Deleting embeddings for: {source}")
            try:
                result = milvus_manager.delete_document(filename=source)
                if result.get("success"):
                    deleted_count = result.get("deleted_count", 0)
                    deleted_total += deleted_count
                    print(f"  ✅ Deleted {deleted_count} embeddings")
                else:
                    print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print()
        print(f"🧹 Cleanup complete! Deleted {deleted_total} total embeddings from {len(orphaned_sources)} orphaned sources")
        
    except Exception as e:
        print(f"❌ Failed to initialize Milvus manager: {e}")
        return


if __name__ == "__main__":
    main()
