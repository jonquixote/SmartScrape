"""
Discovery Coordinator

Manages multi-source URL discovery using various strategies and sources.
Coordinates discovery from search engines, sitemaps, APIs, and other sources.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import re
from urllib.parse import urljoin, urlparse, parse_qs

# Import existing components
try:
    from components.aggressive_url_discovery import AggressiveUrlDiscovery
    _url_discovery = AggressiveUrlDiscovery()
    
    async def discover_urls(query: str, max_results: int = 20) -> List[str]:
        """URL discovery using aggressive discovery component."""
        try:
            return await _url_discovery.discover_urls(query, max_urls=max_results)
        except Exception:
            return []
            
except ImportError:
    # Fallback if module structure is different
    async def discover_urls(query: str, max_results: int = 20) -> List[str]:
        """Fallback URL discovery function."""
        return []

from config import get_config

logger = logging.getLogger("DiscoveryCoordinator")

class DiscoveryStrategy(Enum):
    """Different URL discovery strategies."""
    SEARCH_ENGINE = "search_engine"
    SITEMAP_CRAWL = "sitemap"
    API_ENDPOINT = "api"
    SOCIAL_MEDIA = "social"
    RSS_FEEDS = "rss"
    DIRECT_LINKS = "direct"

@dataclass
class DiscoverySource:
    """Configuration for a discovery source."""
    name: str
    strategy: DiscoveryStrategy
    endpoint: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 50
    priority: int = 1  # Higher = more important
    enabled: bool = True
    rate_limit: float = 1.0  # requests per second

@dataclass
class DiscoveryResult:
    """Result from URL discovery."""
    urls: List[str]
    source: str
    strategy: DiscoveryStrategy
    metadata: Dict[str, Any]
    quality_score: float
    confidence_score: float
    processing_time: float

class DiscoveryCoordinator:
    """
    Discovery Coordinator
    
    Manages URL discovery from multiple sources and strategies:
    - Search engines (Google, DuckDuckGo, Bing)
    - Sitemap crawling
    - API endpoints
    - Social media platforms
    - RSS feeds
    - Direct link extraction
    """
    
    def __init__(self):
        """Initialize the Discovery Coordinator."""
        self.config = get_config()
        
        # Initialize discovery sources
        self.sources = self._initialize_sources()
        
        # Track performance
        self.discovery_stats = {
            "total_discoveries": 0,
            "successful_discoveries": 0,
            "average_urls_per_discovery": 0.0,
            "source_performance": {}
        }
        
        # Rate limiting
        self.rate_limiters = {}
        
        logger.info("DiscoveryCoordinator initialized")
    
    def _initialize_sources(self) -> Dict[str, DiscoverySource]:
        """Initialize available discovery sources."""
        sources = {
            "duckduckgo": DiscoverySource(
                name="DuckDuckGo",
                strategy=DiscoveryStrategy.SEARCH_ENGINE,
                endpoint="https://lite.duckduckgo.com/lite/",
                max_results=20,
                priority=5,
                rate_limit=0.5
            ),
            "google": DiscoverySource(
                name="Google Search",
                strategy=DiscoveryStrategy.SEARCH_ENGINE,
                endpoint="https://www.google.com/search",
                max_results=30,
                priority=4,
                rate_limit=0.3,
                enabled=False  # Requires special handling
            ),
            "bing": DiscoverySource(
                name="Bing Search",
                strategy=DiscoveryStrategy.SEARCH_ENGINE,
                endpoint="https://www.bing.com/search",
                max_results=25,
                priority=3,
                rate_limit=0.5,
                enabled=False  # Requires special handling
            ),
            "sitemap_auto": DiscoverySource(
                name="Automatic Sitemap Discovery",
                strategy=DiscoveryStrategy.SITEMAP_CRAWL,
                max_results=100,
                priority=2,
                rate_limit=1.0
            ),
            "rss_feeds": DiscoverySource(
                name="RSS Feed Discovery",
                strategy=DiscoveryStrategy.RSS_FEEDS,
                max_results=50,
                priority=2,
                rate_limit=1.0
            ),
            "direct_links": DiscoverySource(
                name="Direct Link Extraction",
                strategy=DiscoveryStrategy.DIRECT_LINKS,
                max_results=200,
                priority=1,
                rate_limit=2.0
            )
        }
        return sources
    
    async def discover_urls(self, query: str, intent: Dict[str, Any], 
                          config: Any) -> Dict[str, Any]:
        """
        Coordinate URL discovery from multiple sources.
        
        Args:
            query: User query
            intent: Intent analysis from UniversalIntentAnalyzer
            config: Discovery configuration
            
        Returns:
            Discovery results with URLs, metadata, and source information
        """
        start_time = time.time()
        
        logger.info(f"🔍 Starting URL discovery for: {query}")
        
        # ENHANCED: Use semantic queries for better discovery
        queries_to_search = [query]
        if "enhanced_queries" in intent:
            queries_to_search.extend(intent["enhanced_queries"][:3])  # Limit to top 3
            logger.info(f"Using enhanced queries: {queries_to_search}")
        
        # Determine active sources based on strategy and config
        active_sources = self._select_sources(intent, config)
        
        # Run discovery tasks in parallel for each query
        discovery_tasks = []
        for search_query in queries_to_search:
            for source_name in active_sources:
                source = self.sources[source_name]
                if source.enabled:
                    task = self._discover_from_source(search_query, intent, source)
                    discovery_tasks.append((search_query, source_name, task))
        
        # Execute all discovery tasks
        discovery_results = await asyncio.gather(*[t[2] for t in discovery_tasks], return_exceptions=True)
        
        # Process results
        all_urls = []
        successful_sources = []
        source_results = {}
        
        for i, result in enumerate(discovery_results):
            source_name = active_sources[i]
            
            if isinstance(result, Exception):
                logger.warning(f"Discovery failed for {source_name}: {result}")
                continue
                
            if result and result.urls:
                all_urls.extend(result.urls)
                successful_sources.append(source_name)
                source_results[source_name] = {
                    "urls": result.urls,
                    "count": len(result.urls),
                    "quality_score": result.quality_score,
                    "processing_time": result.processing_time
                }
        
        # Deduplicate URLs while preserving order
        unique_urls = []
        seen_urls = set()
        for url in all_urls:
            normalized_url = self._normalize_url(url)
            if normalized_url not in seen_urls:
                unique_urls.append(url)
                seen_urls.add(normalized_url)
        
        # Apply limits based on configuration
        max_urls = getattr(config, 'max_urls_per_source', 50) * len(active_sources)
        if len(unique_urls) > max_urls:
            unique_urls = unique_urls[:max_urls]
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self._update_discovery_stats(len(unique_urls), len(successful_sources))
        
        logger.info(f"✅ Discovery complete: {len(unique_urls)} unique URLs from {len(successful_sources)} sources")
        
        return {
            "urls": unique_urls,
            "sources": successful_sources,
            "source_results": source_results,
            "total_discovered": len(all_urls),
            "unique_count": len(unique_urls),
            "processing_time": processing_time,
            "metadata": {
                "query": query,
                "intent_type": intent.get("content_type", "unknown"),
                "active_sources": active_sources,
                "successful_sources": successful_sources
            }
        }
    
    def _select_sources(self, intent: Dict[str, Any], config: Any) -> List[str]:
        """Select appropriate sources based on intent and configuration."""
        # Default sources ordered by priority
        default_sources = ["duckduckgo", "sitemap_auto", "rss_feeds", "direct_links"]
        
        # Customize based on intent
        content_type = intent.get("content_type", "").lower()
        
        if "news" in content_type:
            # Prioritize RSS feeds for news
            return ["rss_feeds", "duckduckgo", "sitemap_auto"]
        elif "product" in content_type or "ecommerce" in content_type:
            # Focus on structured search for products
            return ["duckduckgo", "sitemap_auto", "direct_links"]
        elif "academic" in content_type or "research" in content_type:
            # Academic sources first
            return ["duckduckgo", "direct_links", "sitemap_auto"]
        
        return default_sources
    
    async def _discover_from_source(self, query: str, intent: Dict[str, Any], 
                                  source: DiscoverySource) -> Optional[DiscoveryResult]:
        """Discover URLs from a specific source."""
        start_time = time.time()
        
        try:
            # Apply rate limiting
            await self._apply_rate_limit(source.name, source.rate_limit)
            
            urls = []
            metadata = {}
            
            if source.strategy == DiscoveryStrategy.SEARCH_ENGINE:
                urls, metadata = await self._discover_from_search_engine(query, source)
            elif source.strategy == DiscoveryStrategy.SITEMAP_CRAWL:
                urls, metadata = await self._discover_from_sitemaps(query, source)
            elif source.strategy == DiscoveryStrategy.RSS_FEEDS:
                urls, metadata = await self._discover_from_rss(query, source)
            elif source.strategy == DiscoveryStrategy.DIRECT_LINKS:
                urls, metadata = await self._discover_from_direct_links(query, source)
            
            # Limit results
            if len(urls) > source.max_results:
                urls = urls[:source.max_results]
            
            processing_time = time.time() - start_time
            
            # Calculate quality score based on source priority and result count
            quality_score = min(1.0, (source.priority / 5.0) + (len(urls) / source.max_results * 0.3))
            confidence_score = min(1.0, len(urls) / (source.max_results * 0.5))
            
            return DiscoveryResult(
                urls=urls,
                source=source.name,
                strategy=source.strategy,
                metadata=metadata,
                quality_score=quality_score,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Discovery error for {source.name}: {e}")
            return None
    
    async def _discover_from_search_engine(self, query: str, 
                                         source: DiscoverySource) -> Tuple[List[str], Dict]:
        """Discover URLs from search engines."""
        # Use existing discover_urls function
        try:
            urls = await discover_urls(query, max_results=source.max_results)
            metadata = {"search_engine": source.name, "query": query}
            return urls, metadata
        except Exception as e:
            logger.warning(f"Search engine discovery failed: {e}")
            return [], {"error": str(e)}
    
    async def _discover_from_sitemaps(self, query: str, 
                                    source: DiscoverySource) -> Tuple[List[str], Dict]:
        """Discover URLs from sitemaps."""
        # Implement sitemap discovery logic
        urls = []
        metadata = {"method": "sitemap_discovery"}
        
        # For now, return empty - would need sitemap crawler implementation
        logger.info("Sitemap discovery not yet implemented")
        return urls, metadata
    
    async def _discover_from_rss(self, query: str, 
                               source: DiscoverySource) -> Tuple[List[str], Dict]:
        """Discover URLs from RSS feeds."""
        # Implement RSS feed discovery logic
        urls = []
        metadata = {"method": "rss_discovery"}
        
        # For now, return empty - would need RSS parser implementation
        logger.info("RSS discovery not yet implemented")
        return urls, metadata
    
    async def _discover_from_direct_links(self, query: str, 
                                        source: DiscoverySource) -> Tuple[List[str], Dict]:
        """Discover URLs from direct link extraction."""
        urls = []
        metadata = {"method": "direct_links"}
        
        # Basic implementation - could be enhanced with link pattern analysis
        logger.info("Direct link discovery not yet implemented")
        return urls, metadata
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove query parameters and fragment
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return normalized.lower().rstrip('/')
    
    async def _apply_rate_limit(self, source_name: str, rate_limit: float):
        """Apply rate limiting for a source."""
        if source_name not in self.rate_limiters:
            self.rate_limiters[source_name] = 0.0
        
        current_time = time.time()
        time_since_last = current_time - self.rate_limiters[source_name]
        min_interval = 1.0 / rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.rate_limiters[source_name] = time.time()
    
    def _update_discovery_stats(self, url_count: int, source_count: int):
        """Update discovery statistics."""
        self.discovery_stats["total_discoveries"] += 1
        if url_count > 0:
            self.discovery_stats["successful_discoveries"] += 1
        
        # Update average URLs per discovery
        total = self.discovery_stats["total_discoveries"]
        current_avg = self.discovery_stats["average_urls_per_discovery"]
        self.discovery_stats["average_urls_per_discovery"] = (
            (current_avg * (total - 1) + url_count) / total
        )
    
    async def health_check(self) -> str:
        """Perform health check of discovery coordinator."""
        try:
            # Test basic discovery
            test_result = await self.discover_urls(
                query="test query",
                intent={"content_type": "general"},
                config=type('Config', (), {'max_urls_per_source': 5})()
            )
            
            if test_result and test_result.get("urls"):
                return "healthy"
            else:
                return "degraded - no URLs discovered"
                
        except Exception as e:
            return f"error: {e}"
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get current discovery statistics."""
        stats = self.discovery_stats.copy()
        stats["enabled_sources"] = [name for name, source in self.sources.items() if source.enabled]
        stats["total_sources"] = len(self.sources)
        return stats
