#!/usr/bin/env python3
"""
Simplified performance testing tool for robots.txt enforcement system.
Tests performance without external dependencies.

Following DEVELOPMENT_RULES.md for all development requirements.
"""

import asyncio
import time
import statistics
import logging
import json
from datetime import datetime
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
class SimplePerformanceMetrics:
    """Simple performance metrics for robots.txt operations."""
    
    operation: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    
    # Timing metrics
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    median_time: float
    
    # Throughput metrics
    operations_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SimpleRobotsPerformanceTester:
    """
    Simple performance testing for robots.txt enforcement system.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        """Initialize performance tester."""
        self.config = config or CrawlerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.metrics: Dict[str, SimplePerformanceMetrics] = {}
        self.operation_times: Dict[str, List[float]] = {}
    
    async def test_robots_cache_performance(self, origins: List[str], 
                                          iterations: int = 50) -> SimplePerformanceMetrics:
        """Test robots cache performance with multiple origins."""
        operation = "robots_cache"
        self.operation_times[operation] = []
        
        robots_cache = RobotsCache(self.config)
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        async with AsyncHttpClient(self.config) as client:
            for i in range(iterations):
                for origin in origins:
                    operation_start = time.time()
                    
                    try:
                        # Test robots info retrieval
                        await robots_cache.get_robots_info(client, origin)
                        
                        # Test permission checking
                        test_url = f"{origin}/test/page{i}"
                        await robots_cache.can_fetch(client, test_url, self.config.user_agent)
                        
                        # Test crawl delay retrieval
                        await robots_cache.get_crawl_delay(client, test_url, self.config.user_agent)
                        
                        operation_time = time.time() - operation_start
                        self.operation_times[operation].append(operation_time)
                        successful += 1
                        
                    except Exception as e:
                        operation_time = time.time() - operation_start
                        self.operation_times[operation].append(operation_time)
                        failed += 1
                        self.logger.debug(f"Cache operation failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate timing statistics
        times = self.operation_times[operation]
        
        metrics = SimplePerformanceMetrics(
            operation=operation,
            total_operations=len(times),
            successful_operations=successful,
            failed_operations=failed,
            total_time=total_time,
            average_time=statistics.mean(times) if times else 0.0,
            min_time=min(times) if times else 0.0,
            max_time=max(times) if times else 0.0,
            median_time=statistics.median(times) if times else 0.0,
            operations_per_second=len(times) / total_time if total_time > 0 else 0.0
        )
        
        self.metrics[operation] = metrics
        return metrics
    
    async def test_origin_throttle_performance(self, origins: List[str], 
                                             iterations: int = 100) -> SimplePerformanceMetrics:
        """Test origin throttle performance."""
        operation = "origin_throttle"
        self.operation_times[operation] = []
        
        throttle = OriginThrottle(self.config)
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            for origin in origins:
                test_url = f"{origin}/test/page{i}"
                operation_start = time.time()
                
                try:
                    await throttle.wait_for_origin(test_url)
                    operation_time = time.time() - operation_start
                    self.operation_times[operation].append(operation_time)
                    successful += 1
                    
                except Exception as e:
                    operation_time = time.time() - operation_start
                    self.operation_times[operation].append(operation_time)
                    failed += 1
                    self.logger.debug(f"Throttle operation failed: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate timing statistics
        times = self.operation_times[operation]
        
        metrics = SimplePerformanceMetrics(
            operation=operation,
            total_operations=len(times),
            successful_operations=successful,
            failed_operations=failed,
            total_time=total_time,
            average_time=statistics.mean(times) if times else 0.0,
            min_time=min(times) if times else 0.0,
            max_time=max(times) if times else 0.0,
            median_time=statistics.median(times) if times else 0.0,
            operations_per_second=len(times) / total_time if total_time > 0 else 0.0
        )
        
        self.metrics[operation] = metrics
        return metrics
    
    async def test_http_client_performance(self, origins: List[str], 
                                         iterations: int = 20) -> SimplePerformanceMetrics:
        """Test HTTP client performance."""
        operation = "http_client"
        self.operation_times[operation] = []
        
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        async with AsyncHttpClient(self.config) as client:
            for i in range(iterations):
                for origin in origins:
                    operation_start = time.time()
                    
                    try:
                        # Test robots.txt fetching
                        await client.get_robots_txt(origin)
                        
                        operation_time = time.time() - operation_start
                        self.operation_times[operation].append(operation_time)
                        successful += 1
                        
                    except Exception as e:
                        operation_time = time.time() - operation_start
                        self.operation_times[operation].append(operation_time)
                        failed += 1
                        self.logger.debug(f"HTTP operation failed for {origin}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate timing statistics
        times = self.operation_times[operation]
        
        metrics = SimplePerformanceMetrics(
            operation=operation,
            total_operations=len(times),
            successful_operations=successful,
            failed_operations=failed,
            total_time=total_time,
            average_time=statistics.mean(times) if times else 0.0,
            min_time=min(times) if times else 0.0,
            max_time=max(times) if times else 0.0,
            median_time=statistics.median(times) if times else 0.0,
            operations_per_second=len(times) / total_time if total_time > 0 else 0.0
        )
        
        self.metrics[operation] = metrics
        return metrics
    
    async def test_integration_performance(self, origins: List[str], 
                                         iterations: int = 20) -> SimplePerformanceMetrics:
        """Test integrated workflow performance."""
        operation = "integration"
        self.operation_times[operation] = []
        
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        robots_cache = RobotsCache(self.config)
        throttle = OriginThrottle(self.config)
        
        async with AsyncHttpClient(self.config) as client:
            for i in range(iterations):
                for origin in origins:
                    test_url = f"{origin}/test/page{i}"
                    operation_start = time.time()
                    
                    try:
                        # Full workflow: throttle -> robots check -> crawl delay
                        await throttle.wait_for_origin(test_url)
                        allowed = await robots_cache.can_fetch(client, test_url, self.config.user_agent)
                        if allowed:
                            await robots_cache.get_crawl_delay(client, test_url, self.config.user_agent)
                        
                        operation_time = time.time() - operation_start
                        self.operation_times[operation].append(operation_time)
                        successful += 1
                        
                    except Exception as e:
                        operation_time = time.time() - operation_start
                        self.operation_times[operation].append(operation_time)
                        failed += 1
                        self.logger.debug(f"Integration operation failed for {test_url}: {e}")
        
        total_time = time.time() - start_time
        
        # Calculate timing statistics
        times = self.operation_times[operation]
        
        metrics = SimplePerformanceMetrics(
            operation=operation,
            total_operations=len(times),
            successful_operations=successful,
            failed_operations=failed,
            total_time=total_time,
            average_time=statistics.mean(times) if times else 0.0,
            min_time=min(times) if times else 0.0,
            max_time=max(times) if times else 0.0,
            median_time=statistics.median(times) if times else 0.0,
            operations_per_second=len(times) / total_time if total_time > 0 else 0.0
        )
        
        self.metrics[operation] = metrics
        return metrics
    
    async def run_comprehensive_performance_test(self, 
                                               test_origins: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        if test_origins is None:
            test_origins = [
                'https://httpbin.org',
                'https://example.com'
            ]
        
        print("🚀 Starting simplified performance test...")
        
        # Test robots cache performance
        print("📋 Testing robots cache performance...")
        cache_metrics = await self.test_robots_cache_performance(test_origins, iterations=10)
        
        # Test origin throttle performance
        print("⏱️ Testing origin throttle performance...")
        throttle_metrics = await self.test_origin_throttle_performance(test_origins, iterations=20)
        
        # Test HTTP client performance
        print("🌐 Testing HTTP client performance...")
        http_metrics = await self.test_http_client_performance(test_origins, iterations=5)
        
        # Test integration performance
        print("🔄 Testing integration performance...")
        integration_metrics = await self.test_integration_performance(test_origins, iterations=10)
        
        # Generate comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'origins_tested': test_origins,
                'user_agent': self.config.user_agent,
                'cache_ttl': self.config.robots_cache_ttl,
                'default_crawl_delay': self.config.default_crawl_delay
            },
            'performance_metrics': {
                'robots_cache': cache_metrics.to_dict(),
                'origin_throttle': throttle_metrics.to_dict(),
                'http_client': http_metrics.to_dict(),
                'integration': integration_metrics.to_dict()
            },
            'summary': {
                'total_operations': sum(m.total_operations for m in [cache_metrics, throttle_metrics, http_metrics, integration_metrics]),
                'overall_success_rate': self._calculate_overall_success_rate(),
                'performance_score': self._calculate_performance_score()
            }
        }
        
        print("✅ Performance testing completed!")
        return report
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all tests."""
        total_ops = sum(m.total_operations for m in self.metrics.values())
        successful_ops = sum(m.successful_operations for m in self.metrics.values())
        
        if total_ops == 0:
            return 0.0
        
        return (successful_ops / total_ops) * 100.0
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if not self.metrics:
            return 0.0
        
        # Weight different metrics
        weights = {
            'robots_cache': 0.3,
            'origin_throttle': 0.2,
            'http_client': 0.3,
            'integration': 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for operation, weight in weights.items():
            if operation in self.metrics:
                metrics = self.metrics[operation]
                
                # Score based on operations per second and success rate
                ops_score = min(metrics.operations_per_second * 10, 100)  # Cap at 100
                success_score = (metrics.successful_operations / metrics.total_operations) * 100 if metrics.total_operations > 0 else 0
                
                operation_score = (ops_score + success_score) / 2
                score += operation_score * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0


async def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple robots.txt performance testing')
    parser.add_argument('--origins', nargs='+', help='Origins to test')
    parser.add_argument('--cache-test', action='store_true', help='Test cache performance only')
    parser.add_argument('--throttle-test', action='store_true', help='Test throttle performance only')
    parser.add_argument('--http-test', action='store_true', help='Test HTTP client performance only')
    parser.add_argument('--integration-test', action='store_true', help='Test integration performance only')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations per test')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tester = SimpleRobotsPerformanceTester()
    
    test_origins = args.origins or [
        'https://httpbin.org',
        'https://example.com'
    ]
    
    results = {}
    
    if args.cache_test:
        print("Testing robots cache performance...")
        results['cache'] = (await tester.test_robots_cache_performance(test_origins, args.iterations)).to_dict()
    
    elif args.throttle_test:
        print("Testing origin throttle performance...")
        results['throttle'] = (await tester.test_origin_throttle_performance(test_origins, args.iterations)).to_dict()
    
    elif args.http_test:
        print("Testing HTTP client performance...")
        results['http'] = (await tester.test_http_client_performance(test_origins, args.iterations)).to_dict()
    
    elif args.integration_test:
        print("Testing integration performance...")
        results['integration'] = (await tester.test_integration_performance(test_origins, args.iterations)).to_dict()
    
    else:
        # Run comprehensive test
        results = await tester.run_comprehensive_performance_test(test_origins)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
