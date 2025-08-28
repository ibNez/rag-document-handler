"""
Robots.txt parsing and caching for respectful web crawling.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from .crawler_config import get_config
from .async_http_client import AsyncHttpClient

logger = logging.getLogger(__name__)


class RobotsInfo:
    """
    Container for robots.txt information for a specific origin.
    """
    
    def __init__(self, parser: RobotFileParser, crawl_delay: Optional[float], 
                 fetched_at: float, status_code: int):
        """
        Initialize robots info.
        
        Args:
            parser: The RobotFileParser instance
            crawl_delay: Crawl delay in seconds, if specified
            fetched_at: Timestamp when robots.txt was fetched
            status_code: HTTP status code from robots.txt fetch
        """
        self.parser = parser
        self.crawl_delay = crawl_delay
        self.fetched_at = fetched_at
        self.status_code = status_code
    
    def can_fetch(self, user_agent: str, url: str) -> bool:
        """
        Check if the user agent can fetch the given URL.
        
        Args:
            user_agent: The user agent string
            url: The URL to check
            
        Returns:
            bool: True if allowed, False if disallowed
        """
        try:
            return self.parser.can_fetch(user_agent, url)
        except Exception as e:
            logger.warning(f"Error checking robots.txt permission for {url}: {e}")
            # Fail open - allow access if robots.txt parsing fails
            return True
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """
        Check if this robots info has expired.
        
        Args:
            ttl_seconds: Time-to-live in seconds
            
        Returns:
            bool: True if expired
        """
        return (time.time() - self.fetched_at) > ttl_seconds


class RobotsCache:
    """
    Caches robots.txt files with TTL support and proper User-Agent handling.
    
    Manages fetching, parsing, and caching of robots.txt files for different
    origins, with support for crawl-delay extraction and cache expiration.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize robots cache.
        
        Args:
            config: Optional crawler configuration. If None, uses global config.
        """
        self.config = config or get_config()
        self._cache: Dict[str, RobotsInfo] = {}  # origin -> RobotsInfo
        self._locks: Dict[str, asyncio.Lock] = {}  # origin -> asyncio.Lock()
        self._fetch_attempts: Dict[str, int] = {}  # origin -> attempt count
        
    def _get_origin(self, url: str) -> str:
        """
        Extract origin (scheme+host+port) from URL.
        
        Args:
            url: The URL to extract origin from
            
        Returns:
            str: The origin string (e.g., "https://example.com:8080")
        """
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}"
    
    def _get_lock(self, origin: str) -> asyncio.Lock:
        """
        Get or create a lock for the given origin.
        
        Args:
            origin: The origin to get a lock for
            
        Returns:
            asyncio.Lock: The lock for this origin
        """
        if origin not in self._locks:
            self._locks[origin] = asyncio.Lock()
        return self._locks[origin]
    
    async def get_robots_info(self, client: AsyncHttpClient, origin_url: str) -> RobotsInfo:
        """
        Get robots.txt information for the given origin.
        
        Args:
            client: The HTTP client to use for fetching
            origin_url: The origin URL to get robots.txt for
            
        Returns:
            RobotsInfo: The robots information
        """
        origin = self._get_origin(origin_url)
        
        async with self._get_lock(origin):
            # Check cache first
            if origin in self._cache:
                robots_info = self._cache[origin]
                if not robots_info.is_expired(self.config.robots_cache_ttl):
                    logger.debug(f"Using cached robots.txt for {origin}")
                    return robots_info
                else:
                    logger.debug(f"Cached robots.txt expired for {origin}")
            
            # Fetch fresh robots.txt
            return await self._fetch_robots_info(client, origin)
    
    async def _fetch_robots_info(self, client: AsyncHttpClient, origin: str) -> RobotsInfo:
        """
        Fetch and parse robots.txt for the given origin.
        
        Args:
            client: The HTTP client to use
            origin: The origin to fetch robots.txt for
            
        Returns:
            RobotsInfo: The parsed robots information
        """
        attempt_count = self._fetch_attempts.get(origin, 0) + 1
        self._fetch_attempts[origin] = attempt_count
        
        logger.info(f"Fetching robots.txt for {origin} (attempt {attempt_count})")
        
        try:
            content, status_code = await client.get_robots_txt(origin)
            
            # Parse robots.txt
            parser = RobotFileParser()
            if content:
                parser.parse(content.splitlines())
            
            # Extract crawl-delay (prefer specific UA, then wildcard)
            crawl_delay_str = (
                parser.crawl_delay(self.config.user_agent) or 
                parser.crawl_delay("*")
            )
            
            # Convert string to float if present
            crawl_delay = None
            if crawl_delay_str is not None:
                try:
                    crawl_delay = float(crawl_delay_str)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid crawl-delay value '{crawl_delay_str}' for {origin}")
                    crawl_delay = None
            
            robots_info = RobotsInfo(
                parser=parser,
                crawl_delay=crawl_delay,
                fetched_at=time.time(),
                status_code=status_code
            )
            
            # Cache the result
            self._cache[origin] = robots_info
            
            logger.info(
                f"Cached robots.txt for {origin}: "
                f"status={status_code}, crawl_delay={crawl_delay}, "
                f"content_length={len(content)}"
            )
            
            return robots_info
            
        except Exception as e:
            logger.error(f"Failed to fetch robots.txt for {origin}: {e}")
            
            # Create permissive fallback
            parser = RobotFileParser()
            robots_info = RobotsInfo(
                parser=parser,
                crawl_delay=None,
                fetched_at=time.time(),
                status_code=0
            )
            
            # Cache the fallback (with shorter TTL)
            self._cache[origin] = robots_info
            
            return robots_info
    
    async def can_fetch(self, client: AsyncHttpClient, url: str, 
                       user_agent: Optional[str] = None) -> bool:
        """
        Check if the given URL can be fetched according to robots.txt.
        
        Args:
            client: The HTTP client to use for fetching robots.txt if needed
            url: The URL to check
            user_agent: Optional user agent override
            
        Returns:
            bool: True if allowed, False if disallowed
        """
        effective_user_agent = user_agent or self.config.user_agent
        origin = self._get_origin(url)
        
        try:
            robots_info = await self.get_robots_info(client, origin)
            return robots_info.can_fetch(effective_user_agent, url)
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            # Fail open - allow access if we can't check robots.txt
            return True
    
    async def get_crawl_delay(self, client: AsyncHttpClient, url: str,
                             user_agent: Optional[str] = None) -> Optional[float]:
        """
        Get the crawl delay for the given URL according to robots.txt.
        
        Args:
            client: The HTTP client to use for fetching robots.txt if needed
            url: The URL to check
            user_agent: Optional user agent override
            
        Returns:
            Optional[float]: Crawl delay in seconds, or None if not specified
        """
        origin = self._get_origin(url)
        
        try:
            robots_info = await self.get_robots_info(client, origin)
            return robots_info.crawl_delay
        except Exception as e:
            logger.error(f"Error getting crawl delay for {url}: {e}")
            return None
    
    def clear_cache(self, origin: Optional[str] = None) -> int:
        """
        Clear the robots.txt cache.
        
        Args:
            origin: Optional specific origin to clear. If None, clears all.
            
        Returns:
            int: Number of entries cleared
        """
        if origin:
            if origin in self._cache:
                del self._cache[origin]
                self._fetch_attempts.pop(origin, None)
                logger.debug(f"Cleared robots.txt cache for {origin}")
                return 1
            return 0
        else:
            count = len(self._cache)
            self._cache.clear()
            self._fetch_attempts.clear()
            logger.debug(f"Cleared entire robots.txt cache ({count} entries)")
            return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dict: Statistics about cache state
        """
        now = time.time()
        expired_count = 0
        
        for robots_info in self._cache.values():
            if robots_info.is_expired(self.config.robots_cache_ttl):
                expired_count += 1
        
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "total_fetch_attempts": sum(self._fetch_attempts.values()),
            "origins_fetched": len(self._fetch_attempts),
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            int: Number of entries removed
        """
        origins_to_remove = []
        
        for origin, robots_info in self._cache.items():
            if robots_info.is_expired(self.config.robots_cache_ttl):
                origins_to_remove.append(origin)
        
        removed_count = 0
        for origin in origins_to_remove:
            del self._cache[origin]
            self._locks.pop(origin, None)
            removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} expired robots.txt entries")
        
        return removed_count
