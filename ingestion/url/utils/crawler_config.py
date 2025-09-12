"""
Crawler configuration management for robots.txt enforcement.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CrawlerConfig:
    """
    Configuration settings for web crawling with robots.txt enforcement.
    
    All settings can be overridden via environment variables.
    """
    
    # User agent for HTTP requests
    user_agent: str = "RAG-Document-Handler/1.0 (+https://github.com/ibNez/rag-document-handler)"
    
    # Default crawl delay when robots.txt doesn't specify one
    default_crawl_delay: float = 1.0
    
    # How long to cache robots.txt files (seconds)
    robots_cache_ttl: int = 3600
    
    # Timeout for fetching robots.txt files
    robots_fetch_timeout: float = 10.0
    
    # Maximum backoff delay for 429/503 responses
    max_backoff_delay: float = 300.0
    
    # Whether to respect robots.txt by default
    respect_robots_by_default: bool = True
    
    # Minimum delay between requests to same origin
    min_request_delay: float = 0.5
    
    # Maximum retries for failed robots.txt fetches
    max_robots_retries: int = 3
    
    # Default timeout for HTTP requests
    default_http_timeout: float = 30.0
    
    @classmethod
    def from_environment(cls) -> 'CrawlerConfig':
        """
        Create configuration from environment variables.
        
        Environment variables:
        - RAG_CRAWLER_USER_AGENT: Custom user agent string
        - RAG_CRAWLER_DEFAULT_DELAY: Default crawl delay in seconds
        - RAG_CRAWLER_RESPECT_ROBOTS: Whether to respect robots.txt (true/false)
        - RAG_CRAWLER_ROBOTS_CACHE_TTL: Robots.txt cache TTL in seconds
        - RAG_CRAWLER_ROBOTS_TIMEOUT: Timeout for robots.txt fetching
        - RAG_CRAWLER_MAX_BACKOFF: Maximum backoff delay for rate limiting
        - RAG_CRAWLER_MIN_DELAY: Minimum delay between requests
        - RAG_CRAWLER_HTTP_TIMEOUT: Default HTTP request timeout
        
        Returns:
            CrawlerConfig: Configuration instance with environment overrides
        """
        return cls(
            user_agent=os.getenv(
                'RAG_CRAWLER_USER_AGENT',
                "RAG-Document-Handler/1.0 (+https://github.com/ibNez/rag-document-handler)"
            ),
            default_crawl_delay=float(os.getenv('RAG_CRAWLER_DEFAULT_DELAY', '1.0')),
            robots_cache_ttl=int(os.getenv('RAG_CRAWLER_ROBOTS_CACHE_TTL', '3600')),
            robots_fetch_timeout=float(os.getenv('RAG_CRAWLER_ROBOTS_TIMEOUT', '10.0')),
            max_backoff_delay=float(os.getenv('RAG_CRAWLER_MAX_BACKOFF', '300.0')),
            respect_robots_by_default=os.getenv('RAG_CRAWLER_RESPECT_ROBOTS', 'true').lower() == 'true',
            min_request_delay=float(os.getenv('RAG_CRAWLER_MIN_DELAY', '0.5')),
            max_robots_retries=int(os.getenv('RAG_CRAWLER_MAX_RETRIES', '3')),
            default_http_timeout=float(os.getenv('RAG_CRAWLER_HTTP_TIMEOUT', '30.0')),
        )
    
    def get_effective_delay(self, robots_delay: Optional[float]) -> float:
        """
        Calculate the effective delay to use for a request.
        
        Args:
            robots_delay: Crawl delay from robots.txt, if any
            
        Returns:
            float: The delay to use (maximum of robots delay, default delay, and minimum delay)
        """
        delays = [self.min_request_delay, self.default_crawl_delay]
        
        if robots_delay is not None:
            delays.append(robots_delay)
            
        return max(delays)
    
    def get_user_agent_header(self) -> dict[str, str]:
        """
        Get User-Agent header for HTTP requests.
        
        Returns:
            dict: Headers dictionary with User-Agent set
        """
        return {"User-Agent": self.user_agent}
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        if self.default_crawl_delay < 0:
            raise ValueError("default_crawl_delay must be non-negative")
            
        if self.robots_cache_ttl <= 0:
            raise ValueError("robots_cache_ttl must be positive")
            
        if self.robots_fetch_timeout <= 0:
            raise ValueError("robots_fetch_timeout must be positive")
            
        if self.max_backoff_delay < 0:
            raise ValueError("max_backoff_delay must be non-negative")
            
        if self.min_request_delay < 0:
            raise ValueError("min_request_delay must be non-negative")
            
        if self.max_robots_retries < 0:
            raise ValueError("max_robots_retries must be non-negative")
            
        if self.default_http_timeout <= 0:
            raise ValueError("default_http_timeout must be positive")
            
        if not self.user_agent.strip():
            raise ValueError("user_agent cannot be empty")


# Global configuration instance
_config: Optional[CrawlerConfig] = None


def get_config() -> CrawlerConfig:
    """
    Get the global crawler configuration instance.
    
    Returns:
        CrawlerConfig: The global configuration instance
    """
    global _config
    if _config is None:
        _config = CrawlerConfig.from_environment()
        _config.validate()
    return _config
