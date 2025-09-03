#!/usr/bin/env python3
"""
Clean current upload state for fresh testing.
"""

import os
import logging
from ingestion.core.postgres_manager import PostgreSQLManager, PostgreSQLConfig
from rag_manager.core.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_upload_state():
    """Clean the current upload state for fresh testing."""
    try:
        config = Config()
        postgres_config = PostgreSQLConfig(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        postgres = PostgreSQLManager(postgres_config)
        
        filename = "WORLD_HISTORY_-_chap01.pdf"
        
        print("Cleaning upload state...")
        
        # Remove database record
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM documents WHERE filename = %s OR file_path LIKE %s", 
                             (filename, f"%{filename}%"))
                deleted_count = cursor.rowcount
                print(f"Deleted {deleted_count} database record(s)")
        
        # Remove file from staging
        staging_path = os.path.join(config.UPLOAD_FOLDER, filename)
        if os.path.exists(staging_path):
            os.remove(staging_path)
            print(f"Deleted file from staging: {staging_path}")
        else:
            print("No file in staging to delete")
        
        print("✅ Upload state cleaned - ready for fresh test!")
        
    except Exception as e:
        logger.exception(f"Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(clean_upload_state())
