#!/usr/bin/env python3
"""
Debug script to check sub-URLs in the database.
Following DEVELOPMENT_RULES.md for all development requirements.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.core.postgres_manager import PostgreSQLManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_sub_urls():
    """Check what URLs are in the database and their parent-child relationships."""
    
    # Initialize PostgreSQL connection
    postgres = PostgreSQLManager()
    
    try:
        with postgres.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get all URLs with their parent relationships
                cursor.execute("""
                    SELECT 
                        id,
                        url,
                        title,
                        parent_url_id,
                        status,
                        crawl_domain,
                        created_at
                    FROM urls 
                    ORDER BY parent_url_id IS NULL DESC, created_at ASC
                """)
                
                urls = cursor.fetchall()
                
                print(f"\n=== Total URLs in database: {len(urls)} ===\n")
                
                parent_urls = []
                child_urls = []
                
                for url in urls:
                    url_dict = dict(url)
                    if url_dict['parent_url_id'] is None:
                        parent_urls.append(url_dict)
                        print(f"PARENT URL: {url_dict['url']}")
                        print(f"  - ID: {url_dict['id']}")
                        print(f"  - Title: {url_dict['title']}")
                        print(f"  - Status: {url_dict['status']}")
                        print(f"  - Crawl Domain: {url_dict['crawl_domain']}")
                        print(f"  - Created: {url_dict['created_at']}")
                        print()
                    else:
                        child_urls.append(url_dict)
                
                print(f"=== CHILD URLs (Sub-URLs): {len(child_urls)} ===\n")
                for url in child_urls:
                    print(f"CHILD URL: {url['url']}")
                    print(f"  - ID: {url['id']}")
                    print(f"  - Parent ID: {url['parent_url_id']}")
                    print(f"  - Title: {url['title']}")
                    print(f"  - Status: {url['status']}")
                    print(f"  - Created: {url['created_at']}")
                    print()
                
                # Get count statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_urls,
                        COUNT(CASE WHEN parent_url_id IS NULL THEN 1 END) as parent_urls,
                        COUNT(CASE WHEN parent_url_id IS NOT NULL THEN 1 END) as child_urls,
                        COUNT(CASE WHEN status = 'active' AND parent_url_id IS NOT NULL THEN 1 END) as active_child_urls
                    FROM urls
                """)
                
                stats = cursor.fetchone()
                if stats:
                    stats_dict = dict(stats)
                    print("=== STATISTICS ===")
                    print(f"Total URLs: {stats_dict['total_urls']}")
                    print(f"Parent URLs: {stats_dict['parent_urls']}")
                    print(f"Child URLs: {stats_dict['child_urls']}")
                    print(f"Active Child URLs: {stats_dict['active_child_urls']}")
                    
    except Exception as e:
        logger.error(f"Error checking URLs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sub_urls()
