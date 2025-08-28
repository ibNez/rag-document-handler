"""
Domain crawler utilities for discovering and processing URLs within a domain.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
import re
import urllib.parse
from typing import Set, List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
import requests

from .crawler_config import get_config
from .async_http_client import AsyncHttpClient
from .origin_throttle import OriginThrottle
from .robots_parser import RobotsCache

logger = logging.getLogger(__name__)


class DomainCrawler:
    """
    Discovers and processes URLs within a domain for comprehensive content indexing.
    
    Handles link extraction, domain validation, and URL normalization while
    respecting robots.txt rules and avoiding infinite loops.
    """
    
    def __init__(self, url_manager, config: Optional[Any] = None, respect_robots: bool = True):
        """
        Initialize domain crawler with URL management and robots.txt enforcement.
        
        Args:
            url_manager: The URL manager for database operations
            config: Optional crawler configuration. If None, uses global config.
            respect_robots: Whether to respect robots.txt by default
        """
        self.url_manager = url_manager
        self.config = config or get_config()
        self.respect_robots = respect_robots
        
        # Initialize robots.txt enforcement components
        self.robots_cache = RobotsCache(self.config)
        self.origin_throttle = OriginThrottle(self.config)
        
        # Common patterns to exclude from crawling
        self.exclude_patterns = [
            r'\.pdf$', r'\.doc$', r'\.docx$', r'\.xls$', r'\.xlsx$',
            r'\.zip$', r'\.tar$', r'\.gz$', r'\.rar$', r'\.7z$',
            r'\.jpg$', r'\.jpeg$', r'\.png$', r'\.gif$', r'\.svg$',
            r'\.mp3$', r'\.mp4$', r'\.avi$', r'\.mov$', r'\.wmv$',
            r'\.css$', r'\.js$', r'\.ico$', r'\.xml$', r'\.json$',
            r'\/wp-admin\/', r'\/admin\/', r'\/login', r'\/logout',
            r'\?.*=.*&.*=.*&.*=.*',  # URLs with many query parameters
            r'#',  # Fragment-only links
        ]
        
        logger.info(f"DomainCrawler initialized with robots.txt enforcement: {respect_robots}")
        
    async def _check_robots_permission(self, client: AsyncHttpClient, url: str, 
                                     ignore_robots: bool = False) -> bool:
        """
        Check if the URL can be crawled according to robots.txt.
        
        Args:
            client: The HTTP client to use for robots.txt fetching
            url: The URL to check
            ignore_robots: Whether to ignore robots.txt for this URL
            
        Returns:
            bool: True if crawling is allowed
        """
        if ignore_robots or not self.respect_robots:
            logger.debug(f"Robots.txt check bypassed for {url}")
            return True
        
        try:
            can_fetch = await self.robots_cache.can_fetch(client, url)
            logger.debug(f"Robots.txt permission for {url}: {can_fetch}")
            return can_fetch
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            # Fail open - allow crawling if robots.txt check fails
            return True
    
    async def _apply_crawl_delay(self, client: AsyncHttpClient, url: str) -> None:
        """
        Apply appropriate crawl delay for the URL's origin.
        
        Args:
            client: The HTTP client to use for robots.txt fetching
            url: The URL to apply delay for
        """
        try:
            # Get crawl delay from robots.txt
            crawl_delay = await self.robots_cache.get_crawl_delay(client, url)
            
            # Apply throttling with robots.txt delay
            await self.origin_throttle.wait_for_origin(url, crawl_delay)
            
            if crawl_delay:
                logger.debug(f"Applied robots.txt crawl delay of {crawl_delay}s for {url}")
            else:
                logger.debug(f"Applied default crawl delay for {url}")
                
        except Exception as e:
            logger.warning(f"Error applying crawl delay for {url}: {e}")
            # Fall back to default delay
            await self.origin_throttle.wait_for_origin(url, None)
        
    def extract_links_from_html(self, html_content: str, base_url: str) -> Set[str]:
        """
        Extract all valid links from HTML content.
        
        Args:
            html_content: The HTML content to parse
            base_url: The base URL for resolving relative links
            
        Returns:
            Set of absolute URLs found in the content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()
            
            # Find all anchor tags with href attributes
            for link in soup.find_all('a', href=True):
                try:
                    # Since we're finding with href=True, href should exist
                    href = link['href'] if hasattr(link, '__getitem__') else str(link.get('href', ''))
                    href = str(href).strip()
                except (AttributeError, TypeError, KeyError):
                    continue
                
                # Skip empty hrefs, javascript, mailto, tel links
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                    continue
                    
                # Convert relative URLs to absolute
                try:
                    absolute_url = urljoin(base_url, href)
                    # Remove fragment identifier
                    parsed = urlparse(absolute_url)
                    clean_url = urlunparse(parsed._replace(fragment=''))
                    
                    if clean_url and self._is_valid_url(clean_url):
                        links.add(clean_url)
                except Exception as e:
                    logger.debug(f"Failed to process link '{href}': {e}")
                    continue
                    
            logger.debug(f"Extracted {len(links)} valid links from {base_url}")
            return links
            
        except Exception as e:
            logger.error(f"Failed to extract links from HTML content: {e}")
            return set()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not excluded by patterns."""
        try:
            parsed = urlparse(url)
            
            # Must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return False
                
            # Must be HTTP or HTTPS
            if parsed.scheme not in ('http', 'https'):
                return False
                
            # Check against exclude patterns
            for pattern in self.exclude_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def filter_same_domain(self, urls: Set[str], domain: str) -> Set[str]:
        """
        Filter URLs to only include those from the same domain.
        
        Args:
            urls: Set of URLs to filter
            domain: The domain to match against
            
        Returns:
            Set of URLs that belong to the same domain
        """
        same_domain_urls = set()
        
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.netloc.lower() == domain.lower():
                    same_domain_urls.add(url)
            except Exception as e:
                logger.debug(f"Failed to parse URL '{url}': {e}")
                continue
                
        return same_domain_urls
    
    def get_existing_urls_for_domain(self, domain: str) -> Set[str]:
        """
        Get all URLs already in the database for a given domain.
        
        Args:
            domain: The domain to check
            
        Returns:
            Set of URLs already tracked for this domain
        """
        try:
            all_urls = self.url_manager.get_all_urls()
            domain_urls = set()
            
            for url_record in all_urls:
                url = url_record.get('url', '')
                try:
                    parsed = urlparse(url)
                    if parsed.netloc.lower() == domain.lower():
                        domain_urls.add(url)
                except Exception:
                    continue
                    
            return domain_urls
            
        except Exception as e:
            logger.error(f"Failed to get existing URLs for domain {domain}: {e}")
            return set()
    
    async def discover_domain_urls(self, seed_url: str, ignore_robots: bool = False) -> Dict[str, Any]:
        """
        Discover new URLs within a domain by crawling the seed URL with robots.txt enforcement.
        
        Args:
            seed_url: The initial URL to crawl for links
            ignore_robots: Whether to ignore robots.txt for this discovery session
            
        Returns:
            Dictionary with discovery results
        """
        try:
            parsed_seed = urlparse(seed_url)
            domain = parsed_seed.netloc
            
            logger.info(f"Starting domain discovery for {domain} from seed URL: {seed_url}")
            
            async with AsyncHttpClient(self.config) as client:
                # Check robots.txt permission for seed URL
                if not await self._check_robots_permission(client, seed_url, ignore_robots):
                    logger.warning(f"Robots.txt disallows crawling seed URL: {seed_url}")
                    return {
                        "success": False,
                        "message": "Seed URL blocked by robots.txt",
                        "domain": domain,
                        "seed_url": seed_url,
                        "robots_blocked": 1,
                        "discovered_urls": []
                    }
                
                # Apply crawl delay before fetching
                await self._apply_crawl_delay(client, seed_url)
                
                # Fetch the seed URL content
                logger.debug(f"Fetching content from seed URL: {seed_url}")
                response = await client.get(seed_url, timeout=30.0)
                
                # Handle rate limiting
                if client.is_rate_limited(response):
                    retry_after = client.get_retry_after(response)
                    await self.origin_throttle.handle_rate_limit(seed_url, retry_after)
                    logger.warning(f"Rate limited while fetching {seed_url}")
                    return {
                        "success": False,
                        "message": "Rate limited during seed URL fetch",
                        "domain": domain,
                        "seed_url": seed_url,
                        "rate_limited": True,
                        "discovered_urls": []
                    }
                
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for seed URL: {seed_url}")
                    return {
                        "success": False,
                        "message": f"HTTP {response.status_code} for seed URL",
                        "domain": domain,
                        "seed_url": seed_url,
                        "status_code": response.status_code,
                        "discovered_urls": []
                    }
                
                # Extract links from the content
                discovered_links = self.extract_links_from_html(response.text, seed_url)
                
                # Filter to same domain only
                domain_links = self.filter_same_domain(discovered_links, domain)
                
                # Get existing URLs to avoid duplicates
                existing_urls = self.get_existing_urls_for_domain(domain)
                
                # Find new URLs not yet in database
                new_urls = domain_links - existing_urls
                
                # Filter by robots.txt if enabled
                allowed_urls = []
                robots_blocked = []
                
                for url in new_urls:
                    if await self._check_robots_permission(client, url, ignore_robots):
                        allowed_urls.append(url)
                    else:
                        robots_blocked.append(url)
                        logger.debug(f"Robots.txt blocked: {url}")
                
                logger.info(
                    f"Domain discovery for {domain}: "
                    f"found {len(discovered_links)} total links, "
                    f"{len(domain_links)} same-domain, "
                    f"{len(existing_urls)} existing, "
                    f"{len(new_urls)} new, "
                    f"{len(robots_blocked)} robots-blocked, "
                    f"{len(allowed_urls)} allowed"
                )
                
                return {
                    "success": True,
                    "domain": domain,
                    "seed_url": seed_url,
                    "total_links_found": len(discovered_links),
                    "same_domain_links": len(domain_links),
                    "existing_urls": len(existing_urls),
                    "new_urls_found": len(new_urls),
                    "robots_blocked": len(robots_blocked),
                    "discovered_urls": allowed_urls
                }
                
        except Exception as e:
            logger.error(f"Domain discovery failed for {seed_url}: {e}")
            parsed_seed = urlparse(seed_url)  # Parse again for error case
            return {
                "success": False,
                "error": f"Domain discovery failed: {e}",
                "domain": parsed_seed.netloc,
                "seed_url": seed_url,
                "discovered_urls": []
            }
    
    def add_discovered_urls(self, urls: List[str], parent_domain: str, parent_url_id: str) -> Dict[str, Any]:
        """
        Add discovered URLs to the URL management system.
        
        Args:
            urls: List of URLs to add
            parent_domain: The domain these URLs were discovered from
            
        Returns:
            Dictionary with results of URL addition
        """
        added_count = 0
        failed_count = 0
        errors = []
        
        for url in urls:
            try:
                # Generate a descriptive title for discovered URLs
                parsed = urlparse(url)
                path_parts = [p for p in parsed.path.split('/') if p]
                
                if path_parts:
                    title = f"{parent_domain} - {' > '.join(path_parts[:3])}"
                else:
                    title = f"{parent_domain} - Home"
                
                title = title[:200]  # Limit title length
                
                # Add URL with domain crawling disabled (to avoid infinite loops)
                result = self.url_manager.add_url(
                    url=url,
                    title=title,
                    description=f"Discovered from domain crawl of {parent_domain}",
                    parent_url_id=parent_url_id
                )
                
                if result.get("success"):
                    added_count += 1
                    logger.debug(f"Added discovered URL: {url}")
                else:
                    failed_count += 1
                    error_msg = result.get("message", "Unknown error")
                    if "already exists" not in error_msg.lower():
                        errors.append(f"{url}: {error_msg}")
                        
            except Exception as e:
                failed_count += 1
                errors.append(f"{url}: {e}")
                logger.error(f"Failed to add discovered URL {url}: {e}")
        
        logger.info(f"Added {added_count} new URLs from domain crawl, {failed_count} failed")
        
        return {
            "success": True,
            "added_count": added_count,
            "failed_count": failed_count,
            "errors": errors
        }
