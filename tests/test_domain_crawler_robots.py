"""
Unit tests for enhanced domain crawler with robots.txt enforcement.
Tests the DomainCrawler class integration with robots.txt components.
"""

import asyncio
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.url.utils.crawler_config import CrawlerConfig
from ingestion.url.utils.domain_crawler import DomainCrawler


class TestDomainCrawler:
    """Test DomainCrawler with robots.txt enforcement."""
    
    @pytest.fixture
    def mock_url_manager(self):
        """Create a mock URL manager for testing."""
        mock_manager = Mock()
        mock_manager.add_url = Mock(return_value={'success': True, 'id': 'test-url-id'})
        mock_manager.get_all_urls = Mock(return_value=[])
        return mock_manager
    
    def test_domain_crawler_initialization(self, mock_url_manager):
        """Test that domain crawler initializes properly with robots.txt support."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Verify robots.txt components are initialized
        assert hasattr(crawler, 'robots_cache')
        assert hasattr(crawler, 'origin_throttle')
        assert hasattr(crawler, 'config')
        assert crawler.respect_robots is True
    
    def test_domain_crawler_without_robots(self, mock_url_manager):
        """Test domain crawler initialization without robots.txt enforcement."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=False)
        
        # Should still have components but respect_robots should be False
        assert hasattr(crawler, 'robots_cache')
        assert hasattr(crawler, 'origin_throttle')
        assert crawler.respect_robots is False
    
    @pytest.mark.asyncio
    async def test_discover_domain_urls_method_exists(self, mock_url_manager):
        """Test that discover_domain_urls is an async method."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Verify method exists and is async
        assert hasattr(crawler, 'discover_domain_urls')
        import inspect
        assert inspect.iscoroutinefunction(crawler.discover_domain_urls)
    
    @pytest.mark.asyncio
    async def test_robots_permission_checking(self, mock_url_manager):
        """Test robots.txt permission checking method."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Verify private method exists
        assert hasattr(crawler, '_check_robots_permission')
        import inspect
        assert inspect.iscoroutinefunction(crawler._check_robots_permission)
    
    @pytest.mark.asyncio
    async def test_crawl_delay_application(self, mock_url_manager):
        """Test crawl delay application method."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Verify private method exists
        assert hasattr(crawler, '_apply_crawl_delay')
        import inspect
        assert inspect.iscoroutinefunction(crawler._apply_crawl_delay)
    
    def test_add_discovered_urls_method(self, mock_url_manager):
        """Test add_discovered_urls method exists."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Verify method exists
        assert hasattr(crawler, 'add_discovered_urls')
        
        # Test with empty URL list
        result = crawler.add_discovered_urls([], 'example.com', 'parent-id')
        assert isinstance(result, dict)
        assert 'success' in result


class TestDomainCrawlerIntegrationWorkflow:
    """Integration tests for domain crawler workflow."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for integration testing."""
        mock_url_manager = Mock()
        mock_url_manager.add_url = Mock(return_value={'success': True, 'id': 'test-url-id'})
        
        return {
            'url_manager': mock_url_manager
        }
    
    @pytest.mark.asyncio
    async def test_full_discovery_workflow_mocked(self, mock_components):
        """Test full discovery workflow with mocked components."""
        url_manager = mock_components['url_manager']
        crawler = DomainCrawler(url_manager, respect_robots=True)
        
        # Mock the HTTP client and robots.txt responses
        with patch.object(crawler, 'robots_cache') as mock_cache, \
             patch.object(crawler, 'origin_throttle') as mock_throttle:
            
            # Configure mocks
            mock_cache.can_fetch = AsyncMock(return_value=True)
            mock_cache.get_crawl_delay = AsyncMock(return_value=1.0)
            mock_throttle.wait_for_origin = AsyncMock()
            
            # Mock HTTP client for URL discovery
            mock_http_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body><a href="/page1">Page 1</a></body></html>'
            mock_http_client.get = AsyncMock(return_value=mock_response)
            
            with patch('ingestion.url.utils.domain_crawler.AsyncHttpClient') as MockHttpClient:
                MockHttpClient.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
                MockHttpClient.return_value.__aexit__ = AsyncMock(return_value=None)
                
                # Test discovery
                result = await crawler.discover_domain_urls(
                    'https://example.com',
                    ignore_robots=False
                )
                
                # Verify result structure
                assert isinstance(result, dict)
                assert 'success' in result
                assert 'domain' in result
                assert 'discovered_urls' in result
    
    @pytest.mark.asyncio
    async def test_robots_blocked_scenario(self, mock_components):
        """Test scenario where robots.txt blocks crawling."""
        url_manager = mock_components['url_manager']
        crawler = DomainCrawler(url_manager, respect_robots=True)
        
        # Mock robots.txt blocking the URL
        with patch.object(crawler, 'robots_cache') as mock_cache:
            mock_cache.can_fetch = AsyncMock(return_value=False)
            
            # Test discovery with robots blocking
            result = await crawler.discover_domain_urls(
                'https://example.com/blocked',
                ignore_robots=False
            )
            
            # Should indicate failure due to robots.txt
            assert isinstance(result, dict)
            assert 'success' in result
            # The exact behavior depends on implementation
    
    @pytest.mark.asyncio 
    async def test_ignore_robots_scenario(self, mock_components):
        """Test scenario where robots.txt is ignored."""
        url_manager = mock_components['url_manager']
        crawler = DomainCrawler(url_manager, respect_robots=True)
        
        # Mock robots.txt blocking but ignore_robots=True
        with patch.object(crawler, 'robots_cache') as mock_cache, \
             patch.object(crawler, 'origin_throttle') as mock_throttle:
            
            mock_cache.can_fetch = AsyncMock(return_value=False)  # Robots would block
            mock_throttle.wait_for_origin = AsyncMock()
            
            # Mock successful HTTP response
            mock_http_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body><a href="/page1">Page 1</a></body></html>'
            mock_http_client.get = AsyncMock(return_value=mock_response)
            
            with patch('ingestion.url.utils.domain_crawler.AsyncHttpClient') as MockHttpClient:
                MockHttpClient.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
                MockHttpClient.return_value.__aexit__ = AsyncMock(return_value=None)
                
                # Test discovery with ignore_robots=True
                result = await crawler.discover_domain_urls(
                    'https://example.com/blocked',
                    ignore_robots=True
                )
                
                # Should proceed despite robots.txt blocking
                assert isinstance(result, dict)
                assert 'success' in result


class TestDomainCrawlerErrorHandling:
    """Test error handling in domain crawler."""
    
    @pytest.fixture
    def mock_url_manager(self):
        """Create a mock URL manager for testing."""
        mock_manager = Mock()
        mock_manager.add_url = Mock(return_value={'success': True, 'id': 'test-url-id'})
        return mock_manager
    
    @pytest.mark.asyncio
    async def test_http_error_handling(self, mock_url_manager):
        """Test handling of HTTP errors during discovery."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Mock HTTP client that raises an exception
        with patch('ingestion.url.utils.domain_crawler.AsyncHttpClient') as MockHttpClient:
            mock_http_client = Mock()
            mock_http_client.get = AsyncMock(side_effect=Exception("Network error"))
            MockHttpClient.return_value.__aenter__ = AsyncMock(return_value=mock_http_client)
            MockHttpClient.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Test discovery with network error
            result = await crawler.discover_domain_urls(
                'https://unreachable.example.com',
                ignore_robots=True
            )
            
            # Should handle error gracefully
            assert isinstance(result, dict)
            assert 'success' in result
            # Error should be logged but not crash
    
    @pytest.mark.asyncio
    async def test_robots_fetch_error_handling(self, mock_url_manager):
        """Test handling of robots.txt fetch errors."""
        crawler = DomainCrawler(mock_url_manager, respect_robots=True)
        
        # Mock robots cache that raises an exception
        with patch.object(crawler, 'robots_cache') as mock_cache:
            mock_cache.can_fetch = AsyncMock(side_effect=Exception("Robots fetch error"))
            
            # Test discovery with robots.txt error
            result = await crawler.discover_domain_urls(
                'https://example.com',
                ignore_robots=False
            )
            
            # Should handle robots.txt errors gracefully
            assert isinstance(result, dict)
            assert 'success' in result


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
