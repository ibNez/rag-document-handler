"""
Disk Usage Utilities for RAG Document Handler.

This module provides utilities for calculating disk usage metrics
for various directories and database components.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_directory_size(directory_path: str) -> Dict[str, Any]:
    """
    Get the size of a directory in bytes.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Dictionary with size in bytes and human-readable format
    """
    try:
        if not os.path.exists(directory_path):
            return {"bytes": 0, "human": "0 B", "exists": False}
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                try:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    # Skip files that can't be accessed
                    continue
        
        return {
            "bytes": total_size,
            "human": format_bytes(total_size),
            "exists": True
        }
        
    except Exception as e:
        logger.debug(f"Error calculating directory size for {directory_path}: {e}")
        return {"bytes": 0, "human": "Error", "exists": False}


def get_disk_usage(path: str) -> Dict[str, Any]:
    """
    Get disk usage statistics for the filesystem containing the given path.
    
    Args:
        path: Path to check disk usage for
        
    Returns:
        Dictionary with total, used, free space in bytes and percentages
    """
    try:
        if not os.path.exists(path):
            # Use current directory if path doesn't exist
            path = "."
            
        total, used, free = shutil.disk_usage(path)
        
        used_percent = (used / total) * 100 if total > 0 else 0
        free_percent = (free / total) * 100 if total > 0 else 0
        
        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_human": format_bytes(total),
            "used_human": format_bytes(used),
            "free_human": format_bytes(free),
            "used_percent": round(used_percent, 1),
            "free_percent": round(free_percent, 1),
            "warning_level": get_disk_warning_level(free_percent)
        }
        
    except Exception as e:
        logger.debug(f"Error getting disk usage for {path}: {e}")
        return {
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "total_human": "Unknown",
            "used_human": "Unknown", 
            "free_human": "Unknown",
            "used_percent": 0,
            "free_percent": 0,
            "warning_level": "unknown"
        }


def get_disk_warning_level(free_percent: float) -> str:
    """
    Get warning level based on free disk space percentage.
    
    Args:
        free_percent: Free space as percentage
        
    Returns:
        Warning level: 'critical', 'warning', 'ok'
    """
    if free_percent < 15:
        return "critical"
    elif free_percent < 25:
        return "warning"
    else:
        return "ok"


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Human-readable string (e.g., "1.5 GB")
    """
    if bytes_value == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    size = bytes_value
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{size} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def get_postgres_database_size(postgres_manager) -> Dict[str, Any]:
    """
    Get PostgreSQL database size using pg_database_size function.
    
    Args:
        postgres_manager: PostgreSQL manager instance
        
    Returns:
        Dictionary with database size information
    """
    try:
        if not postgres_manager:
            return {"bytes": 0, "human": "N/A", "error": "No database connection"}
        
        with postgres_manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Get total database size
                cur.execute("""
                    SELECT pg_database_size(current_database()) as db_size,
                           pg_size_pretty(pg_database_size(current_database())) as db_size_pretty
                """)
                result = cur.fetchone()
                
                if result:
                    return {
                        "bytes": result['db_size'],
                        "human": result['db_size_pretty'],
                        "error": None
                    }
                else:
                    return {"bytes": 0, "human": "Unknown", "error": "No result"}
                
    except Exception as e:
        logger.debug(f"Error getting PostgreSQL database size: {e}")
        return {"bytes": 0, "human": "Error", "error": str(e)}


def get_postgres_table_sizes(postgres_manager, limit: int = 5) -> Dict[str, Any]:
    """
    Get sizes of largest PostgreSQL tables.
    
    Args:
        postgres_manager: PostgreSQL manager instance
        limit: Number of top tables to return
        
    Returns:
        Dictionary with table size information
    """
    try:
        if not postgres_manager:
            return {"tables": [], "total_bytes": 0, "error": "No database connection"}
        
        with postgres_manager.pool.connection() as conn:
            with conn.cursor() as cur:
                # Get table sizes
                cur.execute("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size_pretty
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT %s
                """, (limit,))
                
                tables = []
                total_bytes = 0
                
                for row in cur.fetchall():
                    table_info = {
                        "schema": row.get('schemaname') if isinstance(row, dict) else row[0],
                        "name": row.get('tablename') if isinstance(row, dict) else row[1],
                        "bytes": row.get('size_bytes') if isinstance(row, dict) else row[2],
                        "human": row.get('size_pretty') if isinstance(row, dict) else row[3]
                    }
                    tables.append(table_info)
                    total_bytes += table_info['bytes']
                
                return {
                    "tables": tables,
                    "total_bytes": total_bytes,
                    "total_human": format_bytes(total_bytes),
                    "error": None
                }
                
    except Exception as e:
        logger.debug(f"Error getting PostgreSQL table sizes: {e}")
        return {"tables": [], "total_bytes": 0, "total_human": "Error", "error": str(e)}
