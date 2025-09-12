"""
Origin-based throttling for respectful web crawling.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from urllib.parse import urlparse

from .crawler_config import get_config

logger = logging.getLogger(__name__)


class OriginThrottle:
    """
    Manages request throttling per origin (scheme+host+port) with crawl-delay support.
    
    Ensures respectful crawling by enforcing minimum delays between requests
    to the same origin and handling rate limit responses with backoff.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize origin throttling.
        
        Args:
            config: Optional crawler configuration. If None, uses global config.
        """
        self.config = config or get_config()
        self._next_allowed_time: Dict[str, float] = {}  # origin -> next allowed time (monotonic)
        self._locks: Dict[str, asyncio.Lock] = {}      # origin -> asyncio.Lock()
        self._backoff_count: Dict[str, int] = {}       # origin -> consecutive backoff count
        
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
    
    async def wait_for_origin(self, url: str, crawl_delay: Optional[float] = None) -> None:
        """
        Wait for the appropriate delay before making a request to the given URL's origin.
        
        Args:
            url: The URL to make a request to
            crawl_delay: Optional crawl delay from robots.txt
        """
        origin = self._get_origin(url)
        effective_delay = self.config.get_effective_delay(crawl_delay)
        
        async with self._get_lock(origin):
            now = time.monotonic()
            next_allowed = self._next_allowed_time.get(origin, now)
            
            if now < next_allowed:
                wait_time = next_allowed - now
                logger.debug(f"Throttling {origin}: waiting {wait_time:.2f}s (delay: {effective_delay}s)")
                await asyncio.sleep(wait_time)
                now = time.monotonic()
            
            # Set next allowed time
            self._next_allowed_time[origin] = now + effective_delay
            
            # Reset backoff count on successful wait
            self._backoff_count[origin] = 0
            
        logger.debug(f"Request allowed for {origin} (next in {effective_delay}s)")
    
    async def handle_rate_limit(self, url: str, retry_after: Optional[float] = None) -> float:
        """
        Handle rate limiting response with exponential backoff.
        
        Args:
            url: The URL that was rate limited
            retry_after: Optional retry-after delay from response headers
            
        Returns:
            float: The actual delay applied (for logging/monitoring)
        """
        origin = self._get_origin(url)
        
        async with self._get_lock(origin):
            # Increment backoff count
            self._backoff_count[origin] = self._backoff_count.get(origin, 0) + 1
            backoff_count = self._backoff_count[origin]
            
            # Calculate backoff delay
            if retry_after is not None:
                # Use server-provided retry-after
                backoff_delay = min(retry_after, self.config.max_backoff_delay)
            else:
                # Exponential backoff: base_delay * 2^(backoff_count-1)
                base_delay = max(self.config.default_crawl_delay, 5.0)
                backoff_delay = min(
                    base_delay * (2 ** (backoff_count - 1)),
                    self.config.max_backoff_delay
                )
            
            logger.warning(
                f"Rate limited on {origin}: backoff #{backoff_count}, "
                f"waiting {backoff_delay:.1f}s (retry_after: {retry_after})"
            )
            
            # Apply the backoff delay
            await asyncio.sleep(backoff_delay)
            
            # Update next allowed time
            now = time.monotonic()
            self._next_allowed_time[origin] = now + self.config.get_effective_delay(None)
            
            return backoff_delay
    
    def get_next_allowed_time(self, url: str) -> float:
        """
        Get the next allowed request time for the given URL's origin.
        
        Args:
            url: The URL to check
            
        Returns:
            float: Monotonic time when next request is allowed
        """
        origin = self._get_origin(url)
        return self._next_allowed_time.get(origin, time.monotonic())
    

    def get_stats(self) -> Dict[str, Any]:
        """
        Get throttling statistics for monitoring.
        
        Returns:
            Dict: Statistics about current throttling state
        """
        now = time.monotonic()
        
        stats = {
            "origins_tracked": len(self._next_allowed_time),
            "origins_in_backoff": sum(1 for count in self._backoff_count.values() if count > 0),
            "total_backoff_events": sum(self._backoff_count.values()),
        }
        
        # Count origins that are currently throttled
        throttled_count = 0
        for origin, next_time in self._next_allowed_time.items():
            if now < next_time:
                throttled_count += 1
        
        stats["origins_currently_throttled"] = throttled_count
        
        return stats
    