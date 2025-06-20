"""
AI Response Cache Module

Provides caching mechanisms for AI API responses to:
1. Reduce redundant API calls
2. Improve performance
3. Reduce costs
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIResponseCache")

class AIResponseCache:
    """
    Cache for AI API responses with both in-memory and disk-based options.
    
    Features:
    - Time-to-live (TTL) for cached entries
    - Disk-based persistence for longer-term caching
    - Memory management with eviction of old entries
    - Content-based hashing for cache keys
    - Performance statistics tracking
    """
    
    def __init__(
        self, 
        cache_dir: str = ".ai_cache",
        memory_size_limit: int = 100,
        disk_enabled: bool = True,
        ttl_seconds: int = 86400,  # Default: 1 day
        generate_stats: bool = True
    ):
        """
        Initialize the AI response cache.
        
        Args:
            cache_dir: Directory for disk cache
            memory_size_limit: Maximum number of items to keep in memory
            disk_enabled: Whether to use disk caching
            ttl_seconds: Time-to-live for cache entries in seconds
            generate_stats: Whether to track and generate cache statistics
        """
        self.cache_dir = cache_dir
        self.memory_size_limit = memory_size_limit
        self.disk_enabled = disk_enabled
        self.ttl_seconds = ttl_seconds
        self.generate_stats = generate_stats
        
        # In-memory cache using OrderedDict for LRU functionality
        self.memory_cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        
        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "evictions": 0,
            "insertions": 0
        }
        
        # Create cache directory if using disk cache
        if self.disk_enabled:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a cache key based on content and optional context.
        
        Args:
            content: The query or content to generate a key for
            context: Additional context that affects the response
            
        Returns:
            Unique hash key for the content+context combination
        """
        # Combine content with relevant context
        key_content = content
        if context:
            # Sort context keys for consistent hashing
            sorted_context = {k: context[k] for k in sorted(context.keys())}
            key_content += json.dumps(sorted_context, sort_keys=True)
        
        # Generate SHA-256 hash
        return hashlib.sha256(key_content.encode('utf-8')).hexdigest()
    
    def _get_disk_path(self, key: str) -> str:
        """
        Get the file path for a disk cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            File path for the cache entry
        """
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Get a response from the cache.
        
        Args:
            content: The query or content to get a response for
            context: Additional context that affects the response
            
        Returns:
            Cached response or None if not found/expired
        """
        cache_key = self._get_cache_key(content, context)
        
        # First try memory cache
        if cache_key in self.memory_cache:
            value, timestamp = self.memory_cache[cache_key]
            
            # Check if entry is still valid
            if time.time() - timestamp <= self.ttl_seconds:
                # Move to end of OrderedDict to mark as recently used
                self.memory_cache.move_to_end(cache_key)
                
                # Update statistics
                if self.generate_stats:
                    self.stats["cache_hits"] += 1
                    self.stats["memory_hits"] += 1
                
                logger.debug(f"Memory cache hit for key: {cache_key[:8]}...")
                return value
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]
        
        # If disk cache is enabled, try it
        if self.disk_enabled:
            disk_path = self._get_disk_path(cache_key)
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, 'r') as f:
                        cached_data = json.load(f)
                    
                    timestamp = cached_data.get("timestamp", 0)
                    
                    # Check if entry is still valid
                    if time.time() - timestamp <= self.ttl_seconds:
                        value = cached_data.get("data")
                        
                        # Also add to memory cache
                        self.memory_cache[cache_key] = (value, timestamp)
                        
                        # Update statistics
                        if self.generate_stats:
                            self.stats["cache_hits"] += 1
                            self.stats["disk_hits"] += 1
                        
                        logger.debug(f"Disk cache hit for key: {cache_key[:8]}...")
                        return value
                    else:
                        # Remove expired disk cache
                        try:
                            os.remove(disk_path)
                        except Exception as e:
                            logger.warning(f"Failed to remove expired disk cache: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error reading disk cache: {str(e)}")
        
        # Update statistics
        if self.generate_stats:
            self.stats["cache_misses"] += 1
            
        logger.debug(f"Cache miss for key: {cache_key[:8]}...")
        return None
    
    def put(self, content: str, value: Any, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a response in the cache.
        
        Args:
            content: The query or content that generated the response
            value: The response to cache
            context: Additional context that affects the response
        """
        cache_key = self._get_cache_key(content, context)
        timestamp = time.time()
        
        # Update memory cache
        self.memory_cache[cache_key] = (value, timestamp)
        
        # Update statistics
        if self.generate_stats:
            self.stats["insertions"] += 1
        
        # Evict old entries if memory cache is full
        while len(self.memory_cache) > self.memory_size_limit:
            # Remove the oldest item (first item in OrderedDict)
            self.memory_cache.popitem(last=False)
            
            # Update statistics
            if self.generate_stats:
                self.stats["evictions"] += 1
        
        # Update disk cache if enabled
        if self.disk_enabled:
            disk_path = self._get_disk_path(cache_key)
            try:
                with open(disk_path, 'w') as f:
                    json.dump({
                        "data": value,
                        "timestamp": timestamp,
                        "context": context
                    }, f)
            except Exception as e:
                logger.warning(f"Error writing to disk cache: {str(e)}")
    
    def invalidate(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            content: The query or content to invalidate
            context: Additional context that affects the response
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        cache_key = self._get_cache_key(content, context)
        found = False
        
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            found = True
        
        # Remove from disk cache if enabled
        if self.disk_enabled:
            disk_path = self._get_disk_path(cache_key)
            if os.path.exists(disk_path):
                try:
                    os.remove(disk_path)
                    found = True
                except Exception as e:
                    logger.warning(f"Error removing disk cache: {str(e)}")
        
        return found
    
    def clear(self) -> None:
        """Clear all cache entries from both memory and disk."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache if enabled
        if self.disk_enabled and os.path.exists(self.cache_dir):
            try:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path) and file_path.endswith('.json'):
                        os.remove(file_path)
            except Exception as e:
                logger.warning(f"Error clearing disk cache: {str(e)}")
        
        # Reset statistics
        if self.generate_stats:
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "memory_hits": 0,
                "disk_hits": 0,
                "evictions": 0,
                "insertions": 0
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.generate_stats:
            return {"stats_tracking_disabled": True}
            
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_ratio = 0.0
        if total_requests > 0:
            hit_ratio = self.stats["cache_hits"] / total_requests
            
        memory_size = len(self.memory_cache)
        
        # Count disk entries if disk cache is enabled
        disk_size = 0
        if self.disk_enabled and os.path.exists(self.cache_dir):
            try:
                disk_size = sum(1 for _ in os.listdir(self.cache_dir) if _.endswith('.json'))
            except Exception:
                pass
                
        return {
            **self.stats,
            "hit_ratio": hit_ratio,
            "memory_size": memory_size,
            "disk_size": disk_size,
            "memory_limit": self.memory_size_limit,
            "ttl_seconds": self.ttl_seconds
        }