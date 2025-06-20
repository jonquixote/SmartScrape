#!/usr/bin/env python3
"""
Cache Warmer for SmartScrape

This script pre-populates the Redis cache with extraction results for commonly
accessed URLs and strategies to improve response times.
"""

import asyncio
import json
import logging
import sys
import os
from typing import List, Dict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from controllers.extraction_coordinator import ExtractionCoordinator
from config import CACHE_TTL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CacheWarmer")

# Common URLs for cache warming (can be configured)
WARM_URLS = [
    "https://httpbin.org/json",
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://httpbin.org/html",
]

# Strategies to warm
WARM_STRATEGIES = [
    "universal_crawl4ai",
    "requests_html",
    "playwright"
]

class CacheWarmer:
    """Cache warming utility for SmartScrape"""
    
    def __init__(self):
        self.coordinator = None
        self.stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'cached': 0
        }
    
    async def initialize(self):
        """Initialize the extraction coordinator"""
        try:
            self.coordinator = ExtractionCoordinator()
            logger.info("Cache warmer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache warmer: {e}")
            raise
    
    async def warm_cache(self, urls: List[str] = None, strategies: List[str] = None):
        """Warm cache with extraction results"""
        urls = urls or WARM_URLS
        strategies = strategies or WARM_STRATEGIES
        
        logger.info(f"Starting cache warming for {len(urls)} URLs and {len(strategies)} strategies")
        
        for url in urls:
            for strategy in strategies:
                await self._warm_single_cache(url, strategy)
                # Rate limiting to avoid overwhelming the target servers
                await asyncio.sleep(0.5)
        
        self._log_stats()
    
    async def _warm_single_cache(self, url: str, strategy: str):
        """Warm cache for a single URL/strategy combination"""
        self.stats['attempted'] += 1
        
        try:
            # Check if already cached
            cached = await self.coordinator.get_cached_content(url, strategy)
            if cached:
                logger.info(f"Already cached: {url} with {strategy}")
                self.stats['cached'] += 1
                return
            
            # Extract content (this will cache it automatically)
            logger.info(f"Warming cache for {url} with {strategy}")
            
            # Use a simplified extraction call that focuses on caching
            from extraction.content_extraction import ContentExtractor
            extractor = ContentExtractor()
            
            result = await extractor.extract_content(url, strategy=strategy)
            
            if result and result.get('success'):
                # Cache the result
                await self.coordinator.cache_content(
                    url, 
                    strategy, 
                    result, 
                    ttl=CACHE_TTL.get('content', 3600)
                )
                logger.info(f"Successfully cached: {url} with {strategy}")
                self.stats['successful'] += 1
            else:
                logger.warning(f"Extraction failed for {url} with {strategy}")
                self.stats['failed'] += 1
                
        except Exception as e:
            logger.error(f"Cache warming failed for {url} with {strategy}: {e}")
            self.stats['failed'] += 1
    
    def _log_stats(self):
        """Log cache warming statistics"""
        logger.info("Cache warming completed!")
        logger.info(f"Attempted: {self.stats['attempted']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Already cached: {self.stats['cached']}")
        
        success_rate = (self.stats['successful'] / max(self.stats['attempted'], 1)) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

async def main():
    """Main cache warming function"""
    warmer = CacheWarmer()
    
    try:
        await warmer.initialize()
        
        # Parse command line arguments for custom URLs/strategies
        import argparse
        parser = argparse.ArgumentParser(description='Warm SmartScrape cache')
        parser.add_argument('--urls', nargs='+', help='URLs to warm cache for')
        parser.add_argument('--strategies', nargs='+', help='Strategies to use')
        
        args = parser.parse_args()
        
        await warmer.warm_cache(args.urls, args.strategies)
        
    except KeyboardInterrupt:
        logger.info("Cache warming interrupted by user")
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
