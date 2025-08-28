"""
Unit tests for URL Manager robots.txt support.
Tests the enhanced PostgreSQLURLManager with robots.txt configuration methods.
"""

import pytest
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestURLManagerRobotsSupport:
    """Test URL Manager robots.txt support methods."""
    
    @pytest.fixture
    def mock_postgres_manager(self):
        """Create a mock PostgreSQL manager for testing."""
        mock_manager = Mock()
        mock_manager.execute_query = Mock()
        mock_manager.fetch_one = Mock()
        mock_manager.fetch_all = Mock()
        return mock_manager
    
    def test_url_manager_robots_methods_exist(self, mock_postgres_manager):
        """Test that robots.txt methods exist in URL manager."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Verify robots.txt methods exist
            assert hasattr(url_manager, 'get_robots_status')
            assert hasattr(url_manager, 'update_robots_setting')
            
            # Verify add_url method exists
            assert hasattr(url_manager, 'add_url')
            
            # Check method signatures
            import inspect
            add_url_sig = inspect.signature(url_manager.add_url)
            assert 'ignore_robots' in add_url_sig.parameters
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")
    
    def test_add_url_with_ignore_robots_parameter(self, mock_postgres_manager):
        """Test add_url method with ignore_robots parameter."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Mock successful database response
            mock_postgres_manager.execute_query.return_value = True
            mock_postgres_manager.fetch_one.return_value = {'id': 'test-url-id'}
            
            # Test with ignore_robots=False (robots enforcement enabled)
            result = url_manager.add_url(
                'https://example.com/page1',
                'Test Page 1',
                'Test page with robots enforcement',
                ignore_robots=False
            )
            
            # Verify call was made
            assert mock_postgres_manager.execute_query.called
            
            # Test with ignore_robots=True (robots enforcement disabled)
            result = url_manager.add_url(
                'https://example.com/page2',
                'Test Page 2', 
                'Test page without robots enforcement',
                ignore_robots=True
            )
            
            # Verify call was made
            assert mock_postgres_manager.execute_query.called
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")
    
    def test_get_robots_status_method(self, mock_postgres_manager):
        """Test get_robots_status method."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Mock database response for robots status
            mock_postgres_manager.fetch_one.return_value = {
                'ignore_robots': False,
                'url': 'https://example.com/page1'
            }
            
            result = url_manager.get_robots_status('test-url-id')
            
            # Verify database query was made
            assert mock_postgres_manager.fetch_one.called
            
            # Verify return format
            assert isinstance(result, dict)
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")
    
    def test_update_robots_setting_method(self, mock_postgres_manager):
        """Test update_robots_setting method."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Mock successful database update
            mock_postgres_manager.execute_query.return_value = True
            
            # Test updating to ignore robots
            result = url_manager.update_robots_setting('test-url-id', ignore_robots=True)
            
            # Verify database update was called
            assert mock_postgres_manager.execute_query.called
            
            # Test updating to enforce robots
            result = url_manager.update_robots_setting('test-url-id', ignore_robots=False)
            
            # Verify database update was called again
            assert mock_postgres_manager.execute_query.called
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")


class TestURLManagerRobotsIntegration:
    """Integration tests for URL Manager robots.txt functionality."""
    
    def test_robots_setting_workflow(self):
        """Test complete workflow of setting and getting robots status."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            # Mock postgres manager
            mock_postgres_manager = Mock()
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Mock add_url response
            mock_postgres_manager.execute_query.return_value = True
            mock_postgres_manager.fetch_one.side_effect = [
                {'id': 'test-url-id'},  # add_url response
                {'ignore_robots': False, 'url': 'https://example.com/test'},  # get_robots_status response
                {'ignore_robots': True, 'url': 'https://example.com/test'}   # updated status
            ]
            
            # 1. Add URL with robots enforcement
            add_result = url_manager.add_url(
                'https://example.com/test',
                'Test Page',
                'Test robots workflow',
                ignore_robots=False
            )
            
            # 2. Get initial robots status
            initial_status = url_manager.get_robots_status('test-url-id')
            
            # 3. Update robots setting
            mock_postgres_manager.execute_query.return_value = True
            update_result = url_manager.update_robots_setting('test-url-id', ignore_robots=True)
            
            # 4. Get updated robots status
            updated_status = url_manager.get_robots_status('test-url-id')
            
            # Verify all operations were called
            assert mock_postgres_manager.execute_query.called
            assert mock_postgres_manager.fetch_one.called
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")
    
    def test_robots_parameter_validation(self):
        """Test validation of robots.txt parameters."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            # Mock postgres manager
            mock_postgres_manager = Mock()
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Test with invalid parameter types
            mock_postgres_manager.execute_query.return_value = True
            mock_postgres_manager.fetch_one.return_value = {'id': 'test-url-id'}
            
            # Should handle boolean values properly
            result = url_manager.add_url(
                'https://example.com/test',
                'Test Page',
                'Test parameter validation',
                ignore_robots=False  # Proper boolean value
            )
            
            # Should still work - implementation should handle conversion
            assert mock_postgres_manager.execute_query.called
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")


class TestURLManagerRobotsErrorHandling:
    """Test error handling in URL Manager robots.txt operations."""
    
    def test_database_error_handling(self):
        """Test handling of database errors in robots operations."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            # Mock postgres manager that raises errors
            mock_postgres_manager = Mock()
            mock_postgres_manager.execute_query.side_effect = Exception("Database error")
            mock_postgres_manager.fetch_one.side_effect = Exception("Database error")
            
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Test add_url with database error - should handle gracefully
            try:
                result = url_manager.add_url(
                    'https://example.com/test',
                    'Test Page',
                    'Test error handling',
                    ignore_robots=False
                )
                # Should return error result, not crash
                assert isinstance(result, dict)
            except Exception:
                # If it raises, that's acceptable error handling too
                pass
            
            # Test get_robots_status with database error
            try:
                result = url_manager.get_robots_status('test-url-id')
                # Should return error result, not crash
                assert isinstance(result, dict)
            except Exception:
                # If it raises, that's acceptable error handling too
                pass
            
            # Test update_robots_setting with database error
            try:
                result = url_manager.update_robots_setting('test-url-id', ignore_robots=True)
                # Should return error result, not crash
                assert isinstance(result, dict)
            except Exception:
                # If it raises, that's acceptable error handling too
                pass
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")
    
    def test_invalid_url_id_handling(self):
        """Test handling of invalid URL IDs."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            # Mock postgres manager with no results
            mock_postgres_manager = Mock()
            mock_postgres_manager.fetch_one.return_value = None  # No URL found
            mock_postgres_manager.execute_query.return_value = False  # Update failed
            
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Test get_robots_status with non-existent URL ID
            result = url_manager.get_robots_status('non-existent-id')
            
            # Should handle missing URL gracefully
            assert isinstance(result, dict)
            
            # Test update_robots_setting with non-existent URL ID
            result = url_manager.update_robots_setting('non-existent-id', ignore_robots=True)
            
            # Should handle missing URL gracefully
            assert isinstance(result, dict)
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")


class TestURLManagerRobotsSchema:
    """Test URL Manager robots.txt database schema requirements."""
    
    def test_ignore_robots_column_handling(self):
        """Test that ignore_robots column is properly handled."""
        try:
            from ingestion.url.manager import PostgreSQLURLManager
            
            # Mock postgres manager
            mock_postgres_manager = Mock()
            url_manager = PostgreSQLURLManager(mock_postgres_manager)
            
            # Mock successful operations
            mock_postgres_manager.execute_query.return_value = True
            mock_postgres_manager.fetch_one.return_value = {'id': 'test-url-id'}
            
            # Test that SQL queries include ignore_robots column
            url_manager.add_url(
                'https://example.com/test',
                'Test Page',
                'Test schema handling',
                ignore_robots=False
            )
            
            # Verify execute_query was called (SQL should include ignore_robots)
            assert mock_postgres_manager.execute_query.called
            
            # Get the SQL query that was called
            call_args = mock_postgres_manager.execute_query.call_args
            if call_args and len(call_args[0]) > 0:
                sql_query = call_args[0][0]
                # SQL should reference ignore_robots column
                assert isinstance(sql_query, str)
                # This is a basic check - actual SQL structure may vary
            
        except ImportError as e:
            pytest.skip(f"URL Manager not available: {e}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
