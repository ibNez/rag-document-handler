#!/usr/bin/env python3
"""
List Milvus Collections Tool

This tool lists all collections in Milvus and their basic stats.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_manager.core.config import Config
from pymilvus import Collection, utility, connections


def main():
    # Load config
    config = Config()
    
    try:
        # Establish connection to Milvus
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        
        # List all collections
        collections = utility.list_collections()
        
        if not collections:
            print("No collections found in Milvus")
            return
        
        print(f"Found {len(collections)} collection(s) in Milvus:")
        print()
        
        for collection_name in collections:
            try:
                col = Collection(collection_name)
                col.load()
                entity_count = col.num_entities
                
                # Get schema info
                schema = col.schema
                fields = [f.name for f in schema.fields]
                
                print(f"📁 Collection: {collection_name}")
                print(f"   Entities: {entity_count}")
                print(f"   Fields: {', '.join(fields)}")
                
                # Try to get some sample sources if this looks like a document collection
                if 'source' in fields and entity_count > 0:
                    try:
                        sample_results = col.query(
                            expr="source != ''",
                            output_fields=["source"],
                            limit=5
                        )
                        sources = [r.get("source") for r in sample_results if r.get("source")]
                        if sources:
                            print(f"   Sample sources: {', '.join(sources[:3])}")
                            if len(sources) > 3:
                                print(f"                   ... and {len(sources) - 3} more")
                    except Exception as e:
                        print(f"   Could not query sample sources: {e}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error accessing collection {collection_name}: {e}")
                print()
        
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")


if __name__ == "__main__":
    main()
