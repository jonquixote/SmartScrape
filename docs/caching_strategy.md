# Caching Strategy in SmartScrape

## Overview

SmartScrape implements a comprehensive caching system designed to improve performance, reduce external API calls, minimize redundant processing, and enhance the overall user experience. The system supports multiple cache backends, intelligent cache invalidation, and adaptive caching strategies based on content type and usage patterns.

## Architecture

### Components

1. **Multi-Tier Cache System** - Memory, Redis, and persistent storage layers
2. **Intelligent Cache Keys** - Content-aware key generation
3. **Cache Invalidation Engine** - Smart expiration and refresh strategies
4. **Performance Monitoring** - Cache hit/miss tracking and optimization
5. **Content-Aware Caching** - Specialized strategies for different data types

### Cache Hierarchy

```
L1: Memory Cache (Fastest, Limited Size)
    ↓ (Cache Miss)
L2: Redis Cache (Fast, Distributed)
    ↓ (Cache Miss)
L3: Persistent Storage (Slower, Long-term)
    ↓ (Cache Miss)
Original Source (Slowest, Always Fresh)
```

## Core Features

### 1. Multi-Level Caching

Intelligent cache layer selection:

- **Memory Cache**: Frequently accessed items, small data
- **Redis Cache**: Shared cache across instances, medium-term storage
- **Persistent Storage**: Long-term cache for expensive computations
- **Content Delivery Network (CDN)**: Static resources and common data

### 2. Content-Aware Strategies

Specialized caching for different content types:

- **AI Responses**: Long TTL, high reuse potential
- **Schema Definitions**: Medium TTL, moderate reuse
- **Intent Analysis**: Short TTL, context-dependent
- **Scraped Content**: Variable TTL based on content freshness requirements
- **URL Generations**: Medium TTL, query-dependent

### 3. Intelligent Invalidation

Smart cache refresh mechanisms:

- **Time-based Expiration**: TTL based on content volatility
- **Content Change Detection**: Hash-based invalidation
- **Dependency Tracking**: Cascading invalidation for related data
- **Usage-based Refresh**: Proactive refresh for popular items

## Implementation Details

### Configuration

Available in `config.py`:

```python
# Caching Configuration
CACHING_ENABLED = True
REDIS_CACHING_ENABLED = True
MEMORY_CACHE_ENABLED = True
PERSISTENT_CACHE_ENABLED = True

# Redis Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None
REDIS_TTL_DEFAULT = 3600  # 1 hour
REDIS_MAX_CONNECTIONS = 10

# Memory Cache Configuration
MEMORY_CACHE_SIZE = 1000  # Maximum items
MEMORY_CACHE_TTL = 300   # 5 minutes
MEMORY_CACHE_CLEANUP_INTERVAL = 60  # 1 minute

# Content-Specific Cache Settings
AI_RESPONSE_CACHE_TTL = 86400     # 24 hours
SCHEMA_CACHE_TTL = 7200           # 2 hours
INTENT_ANALYSIS_CACHE_TTL = 1800  # 30 minutes
SCRAPED_CONTENT_CACHE_TTL = 3600  # 1 hour
URL_GENERATION_CACHE_TTL = 7200   # 2 hours

# Performance Settings
CACHE_COMPRESSION_ENABLED = True
CACHE_ENCRYPTION_ENABLED = False
CACHE_ASYNC_WRITE = True
CACHE_BATCH_OPERATIONS = True
```

### Cache Manager

```python
from utils.cache_manager import CacheManager

# Initialize cache manager
cache_manager = CacheManager()

# Basic cache operations
await cache_manager.set("key", data, ttl=3600)
cached_data = await cache_manager.get("key")
await cache_manager.delete("key")

# Batch operations
await cache_manager.set_multiple({"key1": data1, "key2": data2})
batch_data = await cache_manager.get_multiple(["key1", "key2"])
```

### Smart Cache Keys

```python
from utils.cache_keys import CacheKeyGenerator

key_generator = CacheKeyGenerator()

# Generate content-aware cache keys
ai_key = key_generator.generate_ai_response_key(
    model="gpt-3.5-turbo",
    prompt_hash="abc123",
    parameters={"temperature": 0.7}
)

intent_key = key_generator.generate_intent_analysis_key(
    query="restaurants near me",
    context_hash="def456"
)

schema_key = key_generator.generate_schema_key(
    content_type="product",
    sample_data_hash="ghi789"
)
```

## Usage Examples

### Basic Caching

```python
from utils.cache_manager import CacheManager

cache = CacheManager()

# Cache AI response
ai_response = await ai_service.generate_response(prompt)
await cache.set_ai_response(prompt, ai_response, ttl=86400)

# Retrieve cached response
cached_response = await cache.get_ai_response(prompt)
if cached_response:
    return cached_response
```

### Component-Specific Caching

```python
# Intent Analysis Caching
from components.universal_intent_analyzer import UniversalIntentAnalyzer

analyzer = UniversalIntentAnalyzer(enable_caching=True)

# Automatic caching of analysis results
result = await analyzer.analyze_intent("search query")
# Subsequent calls with same query will use cached result

# Schema Generation Caching
from components.ai_schema_generator import AISchemaGenerator

generator = AISchemaGenerator(enable_caching=True)

# Cache generated schemas
schema = await generator.generate_schema(sample_data, content_type="product")
# Future requests for similar content will reuse cached schema
```

### Advanced Caching Patterns

```python
# Cache with dependency tracking
async def cache_with_dependencies():
    # Cache main content
    await cache.set("main_content", content, dependencies=["user_prefs", "site_config"])
    
    # If dependencies change, main content is invalidated
    await cache.invalidate_dependency("user_prefs")
    # main_content is automatically removed

# Cache with refresh strategy
async def cache_with_refresh():
    # Set cache with background refresh
    await cache.set_with_refresh(
        key="expensive_computation",
        value=result,
        ttl=3600,
        refresh_callback=recompute_expensive_result,
        refresh_threshold=0.8  # Refresh when 80% of TTL elapsed
    )
```

## Cache Strategies by Component

### 1. AI Service Caching

```python
class AIServiceCache:
    """Specialized caching for AI service responses"""
    
    async def cache_ai_response(self, prompt, response, model_info):
        # Generate semantic-aware cache key
        key = self.generate_semantic_key(prompt, model_info)
        
        # Cache with appropriate TTL based on content type
        ttl = self.determine_ttl(response.content_type)
        
        await self.cache.set(key, response, ttl=ttl)
    
    async def get_cached_response(self, prompt, model_info):
        # Check for exact match first
        exact_key = self.generate_semantic_key(prompt, model_info)
        result = await self.cache.get(exact_key)
        
        if not result:
            # Check for semantically similar cached responses
            similar_keys = await self.find_similar_prompts(prompt)
            for key in similar_keys:
                result = await self.cache.get(key)
                if result and self.is_suitable_response(result, prompt):
                    break
        
        return result
```

### 2. Content Quality Scoring Cache

```python
class ContentQualityCache:
    """Caching for content quality scores"""
    
    async def cache_quality_score(self, content_hash, quality_metrics):
        key = f"quality_score:{content_hash}"
        
        # Cache quality scores for longer periods
        await self.cache.set(key, quality_metrics, ttl=7200)
    
    async def get_cached_quality_score(self, content_hash):
        key = f"quality_score:{content_hash}"
        return await self.cache.get(key)
```

### 3. URL Generation Cache

```python
class URLGenerationCache:
    """Caching for generated URLs"""
    
    async def cache_generated_urls(self, query, intent_result, urls):
        # Cache based on query intent and context
        key = self.generate_url_cache_key(query, intent_result)
        
        # URLs may change frequently, shorter TTL
        await self.cache.set(key, urls, ttl=3600)
    
    async def get_cached_urls(self, query, intent_result):
        key = self.generate_url_cache_key(query, intent_result)
        return await self.cache.get(key)
```

## Performance Optimization

### Cache Warming

```python
class CacheWarmer:
    """Proactive cache warming for common requests"""
    
    async def warm_common_caches(self):
        common_queries = await self.get_popular_queries()
        
        for query in common_queries:
            # Pre-generate and cache common analysis
            await self.warm_intent_analysis(query)
            await self.warm_url_generation(query)
            await self.warm_schema_templates(query)
    
    async def warm_intent_analysis(self, query):
        if not await self.cache.exists(f"intent:{query}"):
            result = await self.analyzer.analyze_intent(query)
            await self.cache.set(f"intent:{query}", result)
```

### Cache Compression

```python
from utils.cache_compression import CacheCompressor

compressor = CacheCompressor()

# Compress large cache entries
async def set_large_data(key, data):
    if len(str(data)) > 1024:  # Compress data larger than 1KB
        compressed_data = await compressor.compress(data)
        await cache.set(key, compressed_data, compressed=True)
    else:
        await cache.set(key, data)

async def get_large_data(key):
    data = await cache.get(key)
    if data and hasattr(data, 'compressed') and data.compressed:
        return await compressor.decompress(data)
    return data
```

### Memory Management

```python
class MemoryCacheManager:
    """Intelligent memory cache management"""
    
    def __init__(self, max_memory_mb=100):
        self.max_memory_mb = max_memory_mb
        self.current_memory_usage = 0
        
    async def set_with_memory_check(self, key, value, ttl=None):
        value_size = self.estimate_size(value)
        
        if self.current_memory_usage + value_size > self.max_memory_mb * 1024 * 1024:
            await self.evict_lru_items(value_size)
        
        await self.cache.set(key, value, ttl=ttl)
        self.current_memory_usage += value_size
```

## Cache Invalidation Strategies

### Time-Based Invalidation

```python
class TTLManager:
    """Dynamic TTL management based on content characteristics"""
    
    def calculate_ttl(self, content_type, content_metadata):
        base_ttl = self.get_base_ttl(content_type)
        
        # Adjust based on content volatility
        if content_metadata.get('volatile', False):
            return base_ttl * 0.5
        
        # Adjust based on content popularity
        if content_metadata.get('popular', False):
            return base_ttl * 1.5
        
        # Adjust based on computation cost
        if content_metadata.get('expensive', False):
            return base_ttl * 2
        
        return base_ttl
```

### Content-Based Invalidation

```python
class ContentInvalidator:
    """Invalidate cache based on content changes"""
    
    async def check_content_freshness(self, cache_key, original_source):
        cached_item = await self.cache.get(cache_key)
        if not cached_item:
            return False
        
        # Check if source content has changed
        current_hash = await self.compute_content_hash(original_source)
        cached_hash = cached_item.get('content_hash')
        
        if current_hash != cached_hash:
            await self.cache.delete(cache_key)
            return False
        
        return True
```

## Monitoring and Analytics

### Cache Performance Metrics

```python
from monitoring.cache_monitor import CacheMonitor

monitor = CacheMonitor()

# Get cache performance metrics
metrics = await monitor.get_cache_metrics(time_range="1h")

print(f"Cache hit rate: {metrics.hit_rate:.2%}")
print(f"Memory usage: {metrics.memory_usage_mb:.1f} MB")
print(f"Redis connections: {metrics.redis_connections}")
print(f"Average response time: {metrics.avg_response_time:.2f} ms")
```

### Cache Analytics Dashboard

```python
class CacheAnalytics:
    """Comprehensive cache analytics and reporting"""
    
    async def generate_cache_report(self, time_range="24h"):
        return {
            "hit_rate_by_component": await self.get_hit_rates_by_component(),
            "cache_size_distribution": await self.get_size_distribution(),
            "ttl_effectiveness": await self.analyze_ttl_effectiveness(),
            "top_cached_items": await self.get_most_accessed_items(),
            "eviction_statistics": await self.get_eviction_stats(),
            "performance_impact": await self.measure_performance_impact()
        }
```

### Alert Configuration

```python
from monitoring.cache_alerts import CacheAlerts

alerts = CacheAlerts()

# Configure cache-related alerts
await alerts.configure_alerts(
    hit_rate_threshold=0.70,      # Alert if hit rate drops below 70%
    memory_usage_threshold=0.85,   # Alert if memory usage exceeds 85%
    redis_connection_threshold=8,  # Alert if connections exceed 8
    response_time_threshold=100    # Alert if cache responses exceed 100ms
)
```

## Best Practices

### Cache Key Design

1. **Consistent Naming**: Use standardized key patterns
2. **Hierarchical Structure**: Organize keys for easy management
3. **Avoid Collisions**: Include sufficient context in keys
4. **Version Information**: Include versioning for schema changes

### TTL Strategy

1. **Content-Appropriate TTLs**: Match expiration to content volatility
2. **Graceful Degradation**: Handle cache misses elegantly
3. **Background Refresh**: Update popular items before expiration
4. **Monitoring and Adjustment**: Continuously optimize TTL values

### Memory Management

1. **Size Limits**: Implement appropriate cache size limits
2. **Efficient Eviction**: Use LRU or intelligent eviction policies
3. **Compression**: Compress large cache entries
4. **Monitoring**: Track memory usage and performance impact

## Integration Examples

### Extraction Coordinator Integration

```python
from controllers.extraction_coordinator import ExtractionCoordinator

# Coordinator with caching enabled
coordinator = ExtractionCoordinator(enable_caching=True)

# Extraction results are automatically cached
result = await coordinator.coordinate_extraction(
    query="product search",
    cache_results=True,
    cache_ttl=3600
)

# Subsequent identical queries use cached results
cached_result = await coordinator.coordinate_extraction(
    query="product search"  # Returns cached result if available
)
```

### Strategy Integration

```python
from strategies.composite_universal_strategy import CompositeUniversalStrategy

# Strategy with intelligent caching
strategy = CompositeUniversalStrategy(
    enable_caching=True,
    cache_config={
        "ai_responses": {"ttl": 86400},
        "url_generation": {"ttl": 7200},
        "content_analysis": {"ttl": 3600}
    }
)
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```python
   # Implement aggressive cache cleanup
   await cache_manager.cleanup_expired_items()
   await cache_manager.reduce_cache_size(target_size=0.8)
   ```

2. **Low Hit Rates**
   ```python
   # Analyze cache key patterns
   analytics = await cache_monitor.analyze_miss_patterns()
   # Adjust TTL strategies based on analysis
   ```

3. **Redis Connection Issues**
   ```python
   # Implement connection pooling and retry logic
   cache_manager.configure_redis(
       max_connections=10,
       retry_on_timeout=True,
       connection_pool_enabled=True
   )
   ```

### Debug Mode

Enable detailed cache logging:

```python
import logging

# Enable cache debugging
logging.getLogger('cache_manager').setLevel(logging.DEBUG)
logging.getLogger('redis_cache').setLevel(logging.DEBUG)

# Execute with cache debugging
result = await operation_with_caching(
    enable_cache_debug=True,
    log_cache_operations=True
)
```

## Future Enhancements

### Planned Features

1. **Distributed Cache Coordination**: Multi-instance cache synchronization
2. **Machine Learning Cache Optimization**: AI-driven TTL and eviction strategies
3. **Content-Aware Compression**: Specialized compression for different data types
4. **Predictive Caching**: Pre-cache likely-to-be-requested items

### Research Areas

1. **Semantic Cache Keys**: Similarity-based cache retrieval
2. **Adaptive Cache Hierarchies**: Dynamic cache layer optimization
3. **Privacy-Preserving Caching**: Secure caching for sensitive data
4. **Edge Caching Integration**: CDN and edge cache coordination

---

*For technical support or questions about caching strategies, please refer to the main documentation or contact the development team.*
