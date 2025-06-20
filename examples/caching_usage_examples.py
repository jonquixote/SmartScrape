#!/usr/bin/env python3
"""
SmartScrape Caching Usage Examples

This module demonstrates the intelligent caching capabilities of SmartScrape,
including multi-tier caching, intelligent invalidation, and performance optimization.

Author: SmartScrape Team
Date: 2024
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SmartScrape imports
from controllers.adaptive_scraper import AdaptiveScraper
from utils.config_manager import ConfigManager
from utils.cache_manager import CacheManager


class CachingUsageExamples:
    """Comprehensive examples of SmartScrape's caching capabilities."""
    
    def __init__(self):
        """Initialize the examples with caching-optimized configuration."""
        self.config = ConfigManager()
        self.cache_manager = CacheManager()
        
        # Configure caching for examples
        self.config.update({
            # Redis caching configuration
            'redis_enabled': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'redis_password': None,
            
            # Multi-tier caching settings
            'enable_memory_cache': True,
            'enable_disk_cache': True,
            'enable_redis_cache': True,
            
            # Cache TTL settings (in seconds)
            'memory_cache_ttl': 300,    # 5 minutes
            'disk_cache_ttl': 3600,     # 1 hour
            'redis_cache_ttl': 86400,   # 24 hours
            
            # Cache size limits
            'memory_cache_max_size': 100,
            'disk_cache_max_size': 1000,
            
            # Intelligent invalidation
            'enable_smart_invalidation': True,
            'content_change_threshold': 0.1,  # 10% change triggers invalidation
            'enable_ttl_extension': True,
            
            # Performance optimization
            'cache_compression': True,
            'async_cache_writes': True,
            'cache_warming_enabled': True,
            
            # Content-aware caching
            'cache_by_content_hash': True,
            'enable_semantic_cache_keys': True,
        })
        
        self.scraper = AdaptiveScraper(config=self.config)
    
    async def example_1_basic_caching(self):
        """
        Example 1: Basic caching functionality
        Demonstrates how caching automatically speeds up repeated requests.
        """
        logger.info("=== Example 1: Basic Caching ===")
        
        url = "https://example.com/products"
        user_intent = "Find product information"
        
        # First request - cache miss
        logger.info("First request (cache miss expected)...")
        start_time = time.time()
        
        result1 = await self.scraper.scrape(
            url=url,
            user_intent=user_intent,
            strategy_name="universal"
        )
        
        first_duration = time.time() - start_time
        logger.info(f"First request completed in {first_duration:.2f} seconds")
        
        # Second request - cache hit
        logger.info("Second request (cache hit expected)...")
        start_time = time.time()
        
        result2 = await self.scraper.scrape(
            url=url,
            user_intent=user_intent,
            strategy_name="universal"
        )
        
        second_duration = time.time() - start_time
        logger.info(f"Second request completed in {second_duration:.2f} seconds")
        
        # Performance comparison
        speedup = first_duration / second_duration if second_duration > 0 else float('inf')
        logger.info(f"Cache speedup: {speedup:.2f}x faster")
        
        # Verify cache statistics
        cache_stats = await self.cache_manager.get_stats()
        logger.info(f"Cache statistics: {cache_stats}")
        
        return {
            'first_duration': first_duration,
            'second_duration': second_duration,
            'speedup': speedup,
            'cache_stats': cache_stats
        }
    
    async def example_2_multi_tier_caching(self):
        """
        Example 2: Multi-tier caching demonstration
        Shows how data flows through memory, disk, and Redis caches.
        """
        logger.info("=== Example 2: Multi-Tier Caching ===")
        
        # Configure different cache tiers
        test_data = {
            'memory_only': {'url': 'https://example.com/memory', 'ttl': 300},
            'disk_fallback': {'url': 'https://example.com/disk', 'ttl': 3600},
            'redis_persistence': {'url': 'https://example.com/redis', 'ttl': 86400}
        }
        
        results = {}
        
        for cache_type, config in test_data.items():
            logger.info(f"Testing {cache_type} caching...")
            
            # Clear relevant caches
            await self.cache_manager.clear_cache(cache_type)
            
            # First request
            start_time = time.time()
            result = await self.scraper.scrape(
                url=config['url'],
                user_intent="Extract information",
                strategy_name="universal",
                cache_ttl=config['ttl']
            )
            first_duration = time.time() - start_time
            
            # Check cache population
            cache_info = await self.cache_manager.get_cache_info(config['url'])
            
            # Second request (should hit cache)
            start_time = time.time()
            result = await self.scraper.scrape(
                url=config['url'],
                user_intent="Extract information",
                strategy_name="universal",
                cache_ttl=config['ttl']
            )
            second_duration = time.time() - start_time
            
            results[cache_type] = {
                'first_duration': first_duration,
                'second_duration': second_duration,
                'cache_info': cache_info,
                'speedup': first_duration / second_duration if second_duration > 0 else float('inf')
            }
            
            logger.info(f"{cache_type}: {results[cache_type]['speedup']:.2f}x speedup")
        
        return results
    
    async def example_3_intelligent_invalidation(self):
        """
        Example 3: Intelligent cache invalidation
        Demonstrates how the system automatically invalidates stale cache entries.
        """
        logger.info("=== Example 3: Intelligent Cache Invalidation ===")
        
        url = "https://example.com/dynamic-content"
        user_intent = "Monitor content changes"
        
        # Initial scrape and cache
        logger.info("Initial scrape and cache...")
        result1 = await self.scraper.scrape(
            url=url,
            user_intent=user_intent,
            strategy_name="universal"
        )
        
        # Check cache status
        cache_key = await self.cache_manager.generate_cache_key(url, user_intent)
        cache_entry = await self.cache_manager.get_cache_entry(cache_key)
        
        logger.info(f"Cached at: {cache_entry.get('timestamp', 'Unknown')}")
        logger.info(f"Content hash: {cache_entry.get('content_hash', 'Unknown')}")
        
        # Simulate content change detection
        logger.info("Simulating content change detection...")
        
        # Force content change simulation
        await self.cache_manager.simulate_content_change(cache_key, change_percentage=0.15)
        
        # Next request should detect change and invalidate cache
        logger.info("Next request (should detect change and refresh cache)...")
        start_time = time.time()
        
        result2 = await self.scraper.scrape(
            url=url,
            user_intent=user_intent,
            strategy_name="universal"
        )
        
        duration = time.time() - start_time
        
        # Check if cache was invalidated and refreshed
        new_cache_entry = await self.cache_manager.get_cache_entry(cache_key)
        
        invalidation_stats = {
            'original_hash': cache_entry.get('content_hash'),
            'new_hash': new_cache_entry.get('content_hash'),
            'cache_invalidated': cache_entry.get('content_hash') != new_cache_entry.get('content_hash'),
            'refresh_duration': duration
        }
        
        logger.info(f"Cache invalidation stats: {invalidation_stats}")
        
        return invalidation_stats
    
    async def example_4_cache_warming(self):
        """
        Example 4: Cache warming strategies
        Shows how to pre-populate caches for better performance.
        """
        logger.info("=== Example 4: Cache Warming ===")
        
        # Define URLs to warm
        urls_to_warm = [
            "https://example.com/popular-page-1",
            "https://example.com/popular-page-2",
            "https://example.com/popular-page-3",
            "https://example.com/frequently-accessed"
        ]
        
        user_intent = "Warm cache for popular content"
        
        # Warm caches in parallel
        logger.info("Warming caches for popular URLs...")
        start_time = time.time()
        
        warming_tasks = []
        for url in urls_to_warm:
            task = asyncio.create_task(
                self.scraper.scrape(
                    url=url,
                    user_intent=user_intent,
                    strategy_name="universal",
                    cache_warming=True
                )
            )
            warming_tasks.append(task)
        
        # Wait for all warming tasks to complete
        warming_results = await asyncio.gather(*warming_tasks, return_exceptions=True)
        warming_duration = time.time() - start_time
        
        logger.info(f"Cache warming completed in {warming_duration:.2f} seconds")
        
        # Test cache hit rates
        logger.info("Testing cache hit rates...")
        test_start = time.time()
        
        test_tasks = []
        for url in urls_to_warm:
            task = asyncio.create_task(
                self.scraper.scrape(
                    url=url,
                    user_intent=user_intent,
                    strategy_name="universal"
                )
            )
            test_tasks.append(task)
        
        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)
        test_duration = time.time() - test_start
        
        # Calculate performance improvement
        cache_stats = await self.cache_manager.get_stats()
        
        warming_stats = {
            'urls_warmed': len(urls_to_warm),
            'warming_duration': warming_duration,
            'test_duration': test_duration,
            'cache_hit_rate': cache_stats.get('hit_rate', 0),
            'performance_improvement': warming_duration / test_duration if test_duration > 0 else float('inf')
        }
        
        logger.info(f"Cache warming stats: {warming_stats}")
        
        return warming_stats
    
    async def example_5_semantic_cache_keys(self):
        """
        Example 5: Semantic cache keys
        Demonstrates how similar intents can share cache entries.
        """
        logger.info("=== Example 5: Semantic Cache Keys ===")
        
        url = "https://example.com/products"
        
        # Similar user intents that should share cache
        similar_intents = [
            "Find product information",
            "Get product details",
            "Extract product data",
            "Show product specs",
            "Product information retrieval"
        ]
        
        results = {}
        
        # First request with base intent
        base_intent = similar_intents[0]
        logger.info(f"Base request with intent: '{base_intent}'")
        
        start_time = time.time()
        base_result = await self.scraper.scrape(
            url=url,
            user_intent=base_intent,
            strategy_name="universal"
        )
        base_duration = time.time() - start_time
        
        results['base'] = {
            'intent': base_intent,
            'duration': base_duration,
            'cached': False
        }
        
        # Test similar intents
        for i, intent in enumerate(similar_intents[1:], 1):
            logger.info(f"Testing similar intent {i}: '{intent}'")
            
            start_time = time.time()
            result = await self.scraper.scrape(
                url=url,
                user_intent=intent,
                strategy_name="universal"
            )
            duration = time.time() - start_time
            
            # Check if semantic cache was used
            cache_key = await self.cache_manager.generate_semantic_cache_key(url, intent)
            cache_hit = await self.cache_manager.check_semantic_cache_hit(cache_key)
            
            results[f'similar_{i}'] = {
                'intent': intent,
                'duration': duration,
                'cached': cache_hit,
                'speedup': base_duration / duration if duration > 0 else float('inf')
            }
            
            logger.info(f"Intent {i}: {'Cache HIT' if cache_hit else 'Cache MISS'}, "
                       f"{results[f'similar_{i}']['speedup']:.2f}x speedup")
        
        return results
    
    async def example_6_cache_performance_optimization(self):
        """
        Example 6: Cache performance optimization
        Shows advanced caching features like compression and async writes.
        """
        logger.info("=== Example 6: Cache Performance Optimization ===")
        
        # Test different optimization strategies
        optimization_configs = {
            'baseline': {
                'cache_compression': False,
                'async_cache_writes': False,
                'batch_cache_operations': False
            },
            'compression': {
                'cache_compression': True,
                'async_cache_writes': False,
                'batch_cache_operations': False
            },
            'async_writes': {
                'cache_compression': False,
                'async_cache_writes': True,
                'batch_cache_operations': False
            },
            'full_optimization': {
                'cache_compression': True,
                'async_cache_writes': True,
                'batch_cache_operations': True
            }
        }
        
        test_urls = [
            f"https://example.com/large-page-{i}"
            for i in range(1, 11)
        ]
        
        results = {}
        
        for config_name, config in optimization_configs.items():
            logger.info(f"Testing {config_name} configuration...")
            
            # Update cache configuration
            await self.cache_manager.update_config(config)
            
            # Clear cache for fair comparison
            await self.cache_manager.clear_all_caches()
            
            # Test write performance
            start_time = time.time()
            
            write_tasks = []
            for url in test_urls:
                task = asyncio.create_task(
                    self.scraper.scrape(
                        url=url,
                        user_intent="Performance test",
                        strategy_name="universal"
                    )
                )
                write_tasks.append(task)
            
            await asyncio.gather(*write_tasks, return_exceptions=True)
            write_duration = time.time() - start_time
            
            # Test read performance
            start_time = time.time()
            
            read_tasks = []
            for url in test_urls:
                task = asyncio.create_task(
                    self.scraper.scrape(
                        url=url,
                        user_intent="Performance test",
                        strategy_name="universal"
                    )
                )
                read_tasks.append(task)
            
            await asyncio.gather(*read_tasks, return_exceptions=True)
            read_duration = time.time() - start_time
            
            # Get cache statistics
            cache_stats = await self.cache_manager.get_detailed_stats()
            
            results[config_name] = {
                'write_duration': write_duration,
                'read_duration': read_duration,
                'cache_size_mb': cache_stats.get('total_size_mb', 0),
                'compression_ratio': cache_stats.get('compression_ratio', 1.0),
                'hit_rate': cache_stats.get('hit_rate', 0)
            }
            
            logger.info(f"{config_name} results: "
                       f"Write: {write_duration:.2f}s, "
                       f"Read: {read_duration:.2f}s, "
                       f"Size: {results[config_name]['cache_size_mb']:.2f}MB")
        
        return results
    
    async def example_7_cache_monitoring_and_analytics(self):
        """
        Example 7: Cache monitoring and analytics
        Demonstrates how to monitor cache performance and health.
        """
        logger.info("=== Example 7: Cache Monitoring and Analytics ===")
        
        # Generate some cache activity
        test_scenarios = [
            {'url': 'https://example.com/page1', 'repeat': 3},
            {'url': 'https://example.com/page2', 'repeat': 5},
            {'url': 'https://example.com/page3', 'repeat': 2},
            {'url': 'https://example.com/page4', 'repeat': 4}
        ]
        
        logger.info("Generating cache activity for monitoring...")
        
        for scenario in test_scenarios:
            for i in range(scenario['repeat']):
                await self.scraper.scrape(
                    url=scenario['url'],
                    user_intent=f"Test request {i+1}",
                    strategy_name="universal"
                )
        
        # Get comprehensive cache analytics
        analytics = await self.cache_manager.get_analytics()
        
        logger.info("Cache Analytics:")
        logger.info(f"  Total Requests: {analytics.get('total_requests', 0)}")
        logger.info(f"  Cache Hits: {analytics.get('cache_hits', 0)}")
        logger.info(f"  Cache Misses: {analytics.get('cache_misses', 0)}")
        logger.info(f"  Hit Rate: {analytics.get('hit_rate', 0):.2%}")
        logger.info(f"  Average Response Time: {analytics.get('avg_response_time', 0):.2f}ms")
        logger.info(f"  Cache Size: {analytics.get('cache_size_mb', 0):.2f}MB")
        logger.info(f"  Memory Usage: {analytics.get('memory_usage_mb', 0):.2f}MB")
        
        # Performance trends
        trends = analytics.get('performance_trends', {})
        logger.info("Performance Trends:")
        for metric, values in trends.items():
            logger.info(f"  {metric}: {values}")
        
        # Cache health indicators
        health = await self.cache_manager.get_health_indicators()
        logger.info("Cache Health Indicators:")
        for indicator, status in health.items():
            logger.info(f"  {indicator}: {status}")
        
        return {
            'analytics': analytics,
            'health_indicators': health
        }
    
    async def example_8_custom_cache_strategies(self):
        """
        Example 8: Custom cache strategies
        Shows how to implement domain-specific caching strategies.
        """
        logger.info("=== Example 8: Custom Cache Strategies ===")
        
        # Define custom cache strategies for different content types
        custom_strategies = {
            'news_content': {
                'ttl': 1800,  # 30 minutes
                'invalidation_triggers': ['content_change', 'time_based'],
                'priority': 'high'
            },
            'product_data': {
                'ttl': 3600,  # 1 hour
                'invalidation_triggers': ['price_change', 'availability_change'],
                'priority': 'medium'
            },
            'static_content': {
                'ttl': 86400,  # 24 hours
                'invalidation_triggers': ['manual'],
                'priority': 'low'
            }
        }
        
        results = {}
        
        for strategy_name, strategy_config in custom_strategies.items():
            logger.info(f"Testing {strategy_name} strategy...")
            
            # Apply custom strategy
            await self.cache_manager.apply_custom_strategy(strategy_name, strategy_config)
            
            # Test the strategy
            test_url = f"https://example.com/{strategy_name.replace('_', '-')}"
            
            # First request
            start_time = time.time()
            result1 = await self.scraper.scrape(
                url=test_url,
                user_intent=f"Test {strategy_name}",
                strategy_name="universal",
                cache_strategy=strategy_name
            )
            first_duration = time.time() - start_time
            
            # Second request (should hit cache)
            start_time = time.time()
            result2 = await self.scraper.scrape(
                url=test_url,
                user_intent=f"Test {strategy_name}",
                strategy_name="universal",
                cache_strategy=strategy_name
            )
            second_duration = time.time() - start_time
            
            # Get strategy-specific metrics
            strategy_metrics = await self.cache_manager.get_strategy_metrics(strategy_name)
            
            results[strategy_name] = {
                'first_duration': first_duration,
                'second_duration': second_duration,
                'speedup': first_duration / second_duration if second_duration > 0 else float('inf'),
                'metrics': strategy_metrics
            }
            
            logger.info(f"{strategy_name}: {results[strategy_name]['speedup']:.2f}x speedup")
        
        return results


async def run_all_examples():
    """Run all caching usage examples."""
    logger.info("Starting SmartScrape Caching Usage Examples")
    logger.info("=" * 60)
    
    examples = CachingUsageExamples()
    results = {}
    
    try:
        # Run all examples
        examples_to_run = [
            ('basic_caching', examples.example_1_basic_caching),
            ('multi_tier_caching', examples.example_2_multi_tier_caching),
            ('intelligent_invalidation', examples.example_3_intelligent_invalidation),
            ('cache_warming', examples.example_4_cache_warming),
            ('semantic_cache_keys', examples.example_5_semantic_cache_keys),
            ('performance_optimization', examples.example_6_cache_performance_optimization),
            ('monitoring_analytics', examples.example_7_cache_monitoring_and_analytics),
            ('custom_strategies', examples.example_8_custom_cache_strategies)
        ]
        
        for example_name, example_func in examples_to_run:
            try:
                logger.info(f"\nRunning {example_name}...")
                result = await example_func()
                results[example_name] = result
                logger.info(f"✅ {example_name} completed successfully")
                
                # Brief pause between examples
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ {example_name} failed: {str(e)}")
                results[example_name] = {'error': str(e)}
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("CACHING EXAMPLES SUMMARY")
        logger.info("=" * 60)
        
        successful = sum(1 for r in results.values() if 'error' not in r)
        total = len(results)
        
        logger.info(f"Completed: {successful}/{total} examples")
        logger.info(f"Success Rate: {successful/total*100:.1f}%")
        
        # Performance summary
        if 'basic_caching' in results and 'error' not in results['basic_caching']:
            basic_speedup = results['basic_caching'].get('speedup', 1)
            logger.info(f"Basic Cache Speedup: {basic_speedup:.2f}x")
        
        if 'performance_optimization' in results and 'error' not in results['performance_optimization']:
            perf_results = results['performance_optimization']
            best_config = min(perf_results.items(), 
                             key=lambda x: x[1].get('read_duration', float('inf')))
            logger.info(f"Best Performance Config: {best_config[0]}")
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in examples: {str(e)}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Run the examples
    results = asyncio.run(run_all_examples())
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"caching_examples_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    print("\nCaching usage examples completed!")
