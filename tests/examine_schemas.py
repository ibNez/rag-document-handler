#!/usr/bin/env python3
"""
Script to examine both PostgreSQL and Milvus schemas to understand
the relationship between document records.
"""

import sys
import os
sys.path.append('..')

import psycopg2
from pymilvus import connections, Collection

def examine_postgres_schema():
    print("=" * 60)
    print("POSTGRESQL DOCUMENTS TABLE SCHEMA")
    print("=" * 60)
    
    try:
        from rag_manager.managers.postgres_manager import PostgreSQLManager
        pg_mgr = PostgreSQLManager()
        with pg_mgr.get_connection() as pg_conn:
            with pg_conn.cursor() as cursor:
                # Get table schema
                cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'documents' 
                ORDER BY ordinal_position
            """)
            
            print("Columns:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]} {'NULL' if row[2] == 'YES' else 'NOT NULL'} {f'DEFAULT {row[3]}' if row[3] else ''}")
            
            # Get constraints
            cursor.execute("""
                SELECT constraint_name, constraint_type, column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu USING (constraint_name)
                WHERE tc.table_name = 'documents'
            """)
            
            print("\nConstraints:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]} on {row[2]}")
            
            # Get sample data
            cursor.execute("SELECT * FROM documents LIMIT 3")
            rows = cursor.fetchall()
            
            print("\nSample Data:")
            if rows:
                # Get column names
                cursor.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'documents' 
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cursor.fetchall()]
                
                for i, row in enumerate(rows):
                    print(f"\nRow {i+1}:")
                    for col, val in zip(columns, row):
                        print(f"  {col}: {val}")
            else:
                print("  No data found")
                
        try:
            pg_mgr.close()
        except Exception:
            pass
        
    except Exception as e:
        print(f"PostgreSQL Error: {e}")

def examine_milvus_schema():
    print("\n" + "=" * 60)
    print("MILVUS DOCUMENTS COLLECTION SCHEMA")
    print("=" * 60)
    
    try:
        connections.connect("default", host="localhost", port="19530")
        collection = Collection("documents")
        collection.load()
        
        print("Collection Info:")
        print(f"  Name: {collection.name}")
        print(f"  Num entities: {collection.num_entities}")
        
        print("\nFields:")
        for field in collection.schema.fields:
            print(f"  {field.name}: {field.dtype} {'(Primary)' if field.is_primary else ''}")
            if hasattr(field, 'params'):
                print(f"    Params: {field.params}")
        
        # Get sample data
        print("\nSample Data (first 3 records):")
        results = collection.query(
            expr='chunk_id != ""',  # Fixed: chunk_id is VarChar, not Int
            output_fields=["*"],
            limit=3
        )
        
        if results:
            for i, result in enumerate(results):
                print(f"\nRecord {i+1}:")
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        else:
            print("  No data found")
            
        # Get unique document_ids to see the pattern
        print("\nUnique document_ids in collection:")
        all_docs = collection.query(
            expr='chunk_id != ""',  # Fixed: chunk_id is VarChar, not Int
            output_fields=["document_id"],
            limit=100
        )
        
        unique_doc_ids = set(doc["document_id"] for doc in all_docs)
        for doc_id in list(unique_doc_ids)[:10]:  # Show first 10
            print(f"  {doc_id}")
        
        if connections.has_connection("default"):
            connections.disconnect("default")
            
    except Exception as e:
        print(f"Milvus Error: {e}")

def main():
    examine_postgres_schema()
    examine_milvus_schema()
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("Now we can see the actual fields and determine the correct relationship!")

if __name__ == "__main__":
    main()
