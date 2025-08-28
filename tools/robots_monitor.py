#!/usr/bin/env python3
"""
Robots.txt monitoring and diagnostic tools.
Provides utilities for monitoring robots.txt enforcement and diagnosing issues.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.url.utils.crawler_config import CrawlerConfig
from ingestion.url.utils.async_http_client import AsyncHttpClient
from ingestion.url.utils.origin_throttle import OriginThrottle
from ingestion.url.utils.robots_parser import RobotsCache


@dataclass
class RobotsMonitoringStats:
    """Statistics for robots.txt monitoring."""
    
    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_expired: int = 0
    cache_size: int = 0
    
    # Throttling statistics
    throttled_requests: int = 0
    total_requests: int = 0
    average_delay: float = 0.0
    
    # Robots.txt fetch statistics
    robots_fetches: int = 0
    robots_fetch_failures: int = 0
    robots_parse_errors: int = 0
    
    # Enforcement statistics
    allowed_requests: int = 0
    blocked_requests: int = 0
    ignored_robots_requests: int = 0
    
    # Timing statistics
    average_robots_fetch_time: float = 0.0
    average_permission_check_time: float = 0.0
    
    # Error statistics
    total_errors: int = 0
    network_errors: int = 0
    timeout_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100.0
    
    def get_robots_fetch_success_rate(self) -> float:
        """Calculate robots.txt fetch success rate as percentage."""
        total = self.robots_fetches + self.robots_fetch_failures
        if total == 0:
            return 0.0
        return (self.robots_fetches / total) * 100.0
    
    def get_enforcement_rate(self) -> float:
        """Calculate enforcement rate (blocked / total enforced requests)."""
        enforced_total = self.allowed_requests + self.blocked_requests
        if enforced_total == 0:
            return 0.0
        return (self.blocked_requests / enforced_total) * 100.0


class RobotsMonitor:
    """
    Monitor for robots.txt enforcement system.
    
    Tracks statistics, performance metrics, and provides diagnostic information.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        """
        Initialize robots.txt monitor.
        
        Args:
            config: Optional crawler configuration
        """
        self.config = config or CrawlerConfig()
        self.stats = RobotsMonitoringStats()
        self.start_time = datetime.now()
        self.origins_monitored: Dict[str, Dict[str, Any]] = {}
        self.recent_events: List[Dict[str, Any]] = []
        self.max_recent_events = 1000
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def record_cache_hit(self, origin: str) -> None:
        """Record a cache hit event."""
        self.stats.cache_hits += 1
        self._record_event('cache_hit', {'origin': origin})
    
    def record_cache_miss(self, origin: str) -> None:
        """Record a cache miss event."""
        self.stats.cache_misses += 1
        self._record_event('cache_miss', {'origin': origin})
    
    def record_cache_expiry(self, origin: str) -> None:
        """Record a cache expiry event."""
        self.stats.cache_expired += 1
        self._record_event('cache_expiry', {'origin': origin})
    
    def record_throttled_request(self, origin: str, delay: float) -> None:
        """Record a throttled request."""
        self.stats.throttled_requests += 1
        self.stats.total_requests += 1
        
        # Update average delay
        if self.stats.throttled_requests == 1:
            self.stats.average_delay = delay
        else:
            self.stats.average_delay = (
                (self.stats.average_delay * (self.stats.throttled_requests - 1) + delay) /
                self.stats.throttled_requests
            )
        
        self._record_event('throttled_request', {'origin': origin, 'delay': delay})
    
    def record_robots_fetch(self, origin: str, fetch_time: float, success: bool) -> None:
        """Record a robots.txt fetch attempt."""
        if success:
            self.stats.robots_fetches += 1
            
            # Update average fetch time
            if self.stats.robots_fetches == 1:
                self.stats.average_robots_fetch_time = fetch_time
            else:
                self.stats.average_robots_fetch_time = (
                    (self.stats.average_robots_fetch_time * (self.stats.robots_fetches - 1) + fetch_time) /
                    self.stats.robots_fetches
                )
        else:
            self.stats.robots_fetch_failures += 1
        
        self._record_event('robots_fetch', {
            'origin': origin,
            'fetch_time': fetch_time,
            'success': success
        })
    
    def record_permission_check(self, origin: str, url: str, allowed: bool, 
                              check_time: float, ignored: bool = False) -> None:
        """Record a robots.txt permission check."""
        if ignored:
            self.stats.ignored_robots_requests += 1
        elif allowed:
            self.stats.allowed_requests += 1
        else:
            self.stats.blocked_requests += 1
        
        # Update average permission check time
        total_checks = self.stats.allowed_requests + self.stats.blocked_requests
        if total_checks == 1:
            self.stats.average_permission_check_time = check_time
        elif total_checks > 0:
            self.stats.average_permission_check_time = (
                (self.stats.average_permission_check_time * (total_checks - 1) + check_time) /
                total_checks
            )
        
        self._record_event('permission_check', {
            'origin': origin,
            'url': url,
            'allowed': allowed,
            'check_time': check_time,
            'ignored': ignored
        })
    
    def record_error(self, error_type: str, origin: str, error_message: str) -> None:
        """Record an error event."""
        self.stats.total_errors += 1
        
        if error_type == 'network':
            self.stats.network_errors += 1
        elif error_type == 'timeout':
            self.stats.timeout_errors += 1
        
        self._record_event('error', {
            'error_type': error_type,
            'origin': origin,
            'error_message': error_message
        })
    
    def _record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an event in the recent events list."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        }
        
        self.recent_events.append(event)
        
        # Keep only recent events
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]
    
    def get_origin_stats(self, origin: str) -> Dict[str, Any]:
        """Get statistics for a specific origin."""
        if origin not in self.origins_monitored:
            self.origins_monitored[origin] = {
                'first_seen': datetime.now().isoformat(),
                'requests': 0,
                'blocked': 0,
                'allowed': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'throttled': 0,
                'errors': 0,
                'last_robots_fetch': None,
                'last_request': None
            }
        
        return self.origins_monitored[origin]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        uptime = datetime.now() - self.start_time
        
        health = {
            'status': 'healthy',
            'uptime_seconds': uptime.total_seconds(),
            'cache_hit_rate': self.stats.get_cache_hit_rate(),
            'robots_fetch_success_rate': self.stats.get_robots_fetch_success_rate(),
            'enforcement_rate': self.stats.get_enforcement_rate(),
            'error_rate': 0.0,
            'performance': {
                'average_robots_fetch_time': self.stats.average_robots_fetch_time,
                'average_permission_check_time': self.stats.average_permission_check_time,
                'average_throttle_delay': self.stats.average_delay
            }
        }
        
        # Calculate error rate
        total_operations = (self.stats.robots_fetches + self.stats.robots_fetch_failures + 
                          self.stats.allowed_requests + self.stats.blocked_requests)
        if total_operations > 0:
            health['error_rate'] = (self.stats.total_errors / total_operations) * 100.0
        
        # Determine health status
        if health['error_rate'] > 10.0:
            health['status'] = 'unhealthy'
        elif health['error_rate'] > 5.0 or health['cache_hit_rate'] < 50.0:
            health['status'] = 'degraded'
        
        return health
    
    def get_recent_events(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events, optionally filtered by type."""
        events = self.recent_events
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        
        return events[-limit:]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'statistics': self.stats.to_dict(),
            'health': self.get_system_health(),
            'origins_monitored': len(self.origins_monitored),
            'top_origins': self._get_top_origins(),
            'recent_errors': self.get_recent_events('error', 10),
            'performance_summary': {
                'cache_efficiency': self.stats.get_cache_hit_rate(),
                'robots_reliability': self.stats.get_robots_fetch_success_rate(),
                'enforcement_effectiveness': self.stats.get_enforcement_rate()
            }
        }
    
    def _get_top_origins(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top origins by request count."""
        origins = list(self.origins_monitored.items())
        origins.sort(key=lambda x: x[1]['requests'], reverse=True)
        
        return [
            {
                'origin': origin,
                'stats': stats
            }
            for origin, stats in origins[:limit]
        ]


class RobotsDiagnostics:
    """
    Diagnostic tools for robots.txt enforcement system.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        """Initialize diagnostics."""
        self.config = config or CrawlerConfig()
        self.logger = logging.getLogger(__name__)
    
    async def test_robots_fetch(self, origin: str) -> Dict[str, Any]:
        """Test fetching robots.txt for an origin."""
        start_time = time.time()
        
        result = {
            'origin': origin,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'fetch_time': 0.0,
            'status_code': 0,
            'content_length': 0,
            'content_preview': '',
            'error': None,
            'robots_info': None
        }
        
        try:
            async with AsyncHttpClient(self.config) as client:
                robots_content, status_code = await client.get_robots_txt(origin)
                
                result['success'] = True
                result['status_code'] = status_code
                result['content_length'] = len(robots_content)
                result['content_preview'] = robots_content[:500]  # First 500 chars
                
                # Try to parse robots.txt
                try:
                    robots_cache = RobotsCache(self.config)
                    robots_info = await robots_cache.get_robots_info(client, origin)
                    
                    result['robots_info'] = {
                        'crawl_delay': robots_info.crawl_delay,
                        'status_code': robots_info.status_code,
                        'fetched_at': robots_info.fetched_at
                    }
                except Exception as parse_error:
                    result['robots_info'] = {'parse_error': str(parse_error)}
                
        except Exception as e:
            result['error'] = str(e)
        
        result['fetch_time'] = time.time() - start_time
        return result
    
    async def test_permission_check(self, url: str, user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Test robots.txt permission check for a URL."""
        user_agent = user_agent or self.config.user_agent
        start_time = time.time()
        
        result = {
            'url': url,
            'user_agent': user_agent,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'allowed': False,
            'check_time': 0.0,
            'crawl_delay': None,
            'error': None
        }
        
        try:
            robots_cache = RobotsCache(self.config)
            
            async with AsyncHttpClient(self.config) as client:
                # Check permission
                allowed = await robots_cache.can_fetch(client, url, user_agent)
                result['allowed'] = allowed
                
                # Get crawl delay
                crawl_delay = await robots_cache.get_crawl_delay(client, url, user_agent)
                result['crawl_delay'] = crawl_delay
                
                result['success'] = True
                
        except Exception as e:
            result['error'] = str(e)
        
        result['check_time'] = time.time() - start_time
        return result
    
    async def test_throttling(self, origin: str, num_requests: int = 3) -> Dict[str, Any]:
        """Test origin throttling behavior."""
        throttle = OriginThrottle(self.config)
        
        result = {
            'origin': origin,
            'num_requests': num_requests,
            'timestamp': datetime.now().isoformat(),
            'requests': [],
            'total_time': 0.0,
            'average_delay': 0.0
        }
        
        start_time = time.time()
        
        for i in range(num_requests):
            request_start = time.time()
            
            try:
                await throttle.wait_for_origin(f"{origin}/test{i}")
                request_time = time.time() - request_start
                
                result['requests'].append({
                    'request_number': i + 1,
                    'delay_time': request_time,
                    'success': True
                })
                
            except Exception as e:
                result['requests'].append({
                    'request_number': i + 1,
                    'delay_time': 0.0,
                    'success': False,
                    'error': str(e)
                })
        
        result['total_time'] = time.time() - start_time
        
        # Calculate average delay
        successful_requests = [r for r in result['requests'] if r['success']]
        if successful_requests:
            result['average_delay'] = sum(r['delay_time'] for r in successful_requests) / len(successful_requests)
        
        return result
    
    async def comprehensive_test(self, origin: str) -> Dict[str, Any]:
        """Run comprehensive diagnostics for an origin."""
        test_url = f"{origin}/test"
        
        return {
            'origin': origin,
            'timestamp': datetime.now().isoformat(),
            'robots_fetch': await self.test_robots_fetch(origin),
            'permission_check': await self.test_permission_check(test_url),
            'throttling': await self.test_throttling(origin),
            'system_config': {
                'user_agent': self.config.user_agent,
                'default_crawl_delay': self.config.default_crawl_delay,
                'robots_cache_ttl': self.config.robots_cache_ttl,
                'max_backoff_delay': self.config.max_backoff_delay
            }
        }


async def main():
    """Main diagnostic function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robots.txt monitoring and diagnostics')
    parser.add_argument('--origin', help='Origin to test (e.g., https://example.com)')
    parser.add_argument('--url', help='URL to test permission for')
    parser.add_argument('--monitor', action='store_true', help='Show monitoring statistics')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test')
    
    args = parser.parse_args()
    
    diagnostics = RobotsDiagnostics()
    
    if args.comprehensive and args.origin:
        result = await diagnostics.comprehensive_test(args.origin)
        print(json.dumps(result, indent=2))
    
    elif args.origin:
        result = await diagnostics.test_robots_fetch(args.origin)
        print(json.dumps(result, indent=2))
    
    elif args.url:
        result = await diagnostics.test_permission_check(args.url)
        print(json.dumps(result, indent=2))
    
    elif args.monitor:
        monitor = RobotsMonitor()
        report = monitor.generate_report()
        print(json.dumps(report, indent=2))
    
    else:
        print("Usage examples:")
        print("  python robots_monitor.py --origin https://example.com")
        print("  python robots_monitor.py --url https://example.com/page")
        print("  python robots_monitor.py --comprehensive --origin https://example.com")
        print("  python robots_monitor.py --monitor")


if __name__ == "__main__":
    asyncio.run(main())
