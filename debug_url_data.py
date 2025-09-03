#!/usr/bin/env python3
"""
Debug script to check URL data structure from get_all_urls().
Following DEVELOPMENT_RULES.md for all development requirements.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.url.manager import URLManager
from ingestion.core.postgres_manager import PostgreSQLManager
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_url_data():
    """Check what data structure is returned by get_all_urls()."""
    
    # Initialize URL manager
    postgres = PostgreSQLManager()
    url_manager = URLManager(postgres)
    
    try:
        urls = url_manager.get_all_urls()
        
        print(f"\n=== URLs returned by get_all_urls(): {len(urls)} ===\n")
        
        for i, url in enumerate(urls):
            print(f"URL {i+1}:")
            print(f"  URL: {url.get('url', 'N/A')}")
            print(f"  Title: {url.get('title', 'N/A')}")
            print(f"  ID: {url.get('id', 'N/A')}")
            print(f"  Status: {url.get('status', 'N/A')}")
            print(f"  ignore_robots: {url.get('ignore_robots', 'NOT FOUND')} (type: {type(url.get('ignore_robots'))})")
            print(f"  ignore_robots_txt: {url.get('ignore_robots_txt', 'NOT FOUND')} (type: {type(url.get('ignore_robots_txt'))})")
            print(f"  crawl_domain: {url.get('crawl_domain', 'NOT FOUND')} (type: {type(url.get('crawl_domain'))})")
            print(f"  refresh_interval_minutes: {url.get('refresh_interval_minutes', 'NOT FOUND')} (type: {type(url.get('refresh_interval_minutes'))})")
            print(f"  parent_url_id: {url.get('parent_url_id', 'NOT FOUND')} (type: {type(url.get('parent_url_id'))})")
            print(f"  last_scraped: {url.get('last_scraped', 'NOT FOUND')}")
            print(f"  last_crawled: {url.get('last_crawled', 'NOT FOUND')}")
            print()
            
            # Show all keys in the URL object
            print("  All keys in URL object:")
            for key in sorted(url.keys()):
                value = url[key]
                print(f"    {key}: {value} (type: {type(value)})")
            print("=" * 50)
            print()
        
        # Calculate stats like the panel does
        robots_ignored_old = sum(1 for url in urls if url.get('ignore_robots_txt', False))
        robots_ignored_new = sum(1 for url in urls if url.get('ignore_robots', 0) == 1)
        crawl_on = sum(1 for url in urls if url.get('refresh_interval_minutes', 0) > 0)
        
        print("=== CALCULATED STATS ===")
        print(f"robots_ignored (old method - ignore_robots_txt): {robots_ignored_old}")
        print(f"robots_ignored (new method - ignore_robots == 1): {robots_ignored_new}")
        print(f"crawl_on: {crawl_on}")
        
    except Exception as e:
        logger.error(f"Error checking URL data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_url_data()
