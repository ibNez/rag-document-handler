#!/usr/bin/env python
"""
Base Data Manager
Following DEVELOPMENT_RULES.md for all development requirements

This module provides base functionality for data access operations
across all content types (email, document, URL).
"""

import logging
from typing import Any, Optional, Dict, Tuple, Union

logger = logging.getLogger(__name__)


class BaseDataManager:
    """
    Base class for all data managers providing common database operations.
    
    This class encapsulates shared database functionality and ensures
    consistent patterns across all content-specific data managers.
    """
    
    def __init__(self, postgres_manager: Any, milvus_manager: Optional[Any] = None) -> None:
        """
        Initialize base data manager.
        
        Args:
            postgres_manager: PostgreSQL connection manager
            milvus_manager: Optional Milvus vector store manager
        """
        self.postgres_manager = postgres_manager
        self.milvus_manager = milvus_manager
        logger.info(f"{self.__class__.__name__} initialized")
    
    def get_connection(self):
        """Get PostgreSQL database connection."""
        return self.postgres_manager.get_connection()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None, fetch_one: bool = False, fetch_all: bool = False):
        """
        Execute a database query with consistent error handling.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results
            
        Returns:
            Query result or None
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())
                    
                    if fetch_one:
                        return cur.fetchone()
                    elif fetch_all:
                        return cur.fetchall()
                    
                conn.commit()
                return None
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
    
    def get_statistics_base(self, table_name: str, filters: Optional[Dict] = None) -> Dict:
        """
        Get basic statistics for any table.
        
        Args:
            table_name: Name of the table to query
            filters: Optional filters to apply
            
        Returns:
            Dictionary with basic statistics
        """
        try:
            where_clause = ""
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = %s")
                    params.append(value)
                where_clause = f"WHERE {' AND '.join(conditions)}"
            
            query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE) as today_records,
                    COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE - INTERVAL '7 days') as week_records,
                    COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE - INTERVAL '30 days') as month_records
                FROM {table_name}
                {where_clause}
            """
            
            result = self.execute_query(query, tuple(params), fetch_one=True)
            
            return {
                'total_records': result['total_records'] if result else 0,
                'today_records': result['today_records'] if result else 0,
                'week_records': result['week_records'] if result else 0,
                'month_records': result['month_records'] if result else 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics for {table_name}: {e}")
            return {
                'total_records': 0,
                'today_records': 0,
                'week_records': 0,
                'month_records': 0,
            }
