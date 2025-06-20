"""
AI Response Caching Module

Implements a caching system for AI API responses to reduce redundant API calls
and improve performance.
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AIResponseCache")

class AIResponseCache:
    """
    Caching system for AI API responses to reduce redundant API calls and improve performance.
    
    This class:
    - Stores AI responses in memory and optionally on disk
    - Implements TTL (time-to-live) for cache entries
    - Uses content-based hashing for cache keys
    - Provides statistics on cache performance
    """
    
    def __init__(self, 
                cache_dir: Optional[str] = None, 
                ttl_seconds: int = 3600,  # 1 hour default TTL
                max_memory_entries: int = 1000,
                persist_to_disk: bool = True):
        """
        Initialize the AI response cache.
        
        Args:
            cache_dir: Directory to store persistent cache files
            ttl_seconds: Time-to-live for cache entries in seconds
            max_memory_entries: Maximum number of entries to keep in memory
            persist_to_disk: Whether to persist cache to disk
        """
        self.memory_cache = {}  # In-memory cache
        self.ttl_seconds = ttl_seconds
        self.max_memory_entries = max_memory_entries
        self.persist_to_disk = persist_to_disk
        
        # Cache directory setup
        if persist_to_disk:
            self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), ".cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"AI response cache initialized with persistence at {self.cache_dir}")
        else:
            self.cache_dir = None
            logger.info("AI response cache initialized in memory only")
        
        # Performance tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "disk_hits": 0,
            "evictions": 0,
            "total_requests": 0,
            "created_at": time.time()
        }
    
    def _generate_key(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Generate a deterministic cache key from content.
        
        Args:
            content: Content to hash (string or dictionary)
            
        Returns:
            Hash string to use as cache key
        """
        # Convert dict to string if necessary
        if isinstance(content, dict):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)
            
        # Generate hash
        hash_obj = hashlib.md5(content_str.encode())
        return hash_obj.hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """
        Check if a cache entry is expired based on TTL.
        
        Args:
            timestamp: Entry creation timestamp
            
        Returns:
            True if entry is expired, False otherwise
        """
        return (time.time() - timestamp) > self.ttl_seconds
    
    def get(self, key: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: Cache key or content to hash for key
            
        Returns:
            Cached value or None if not found or expired
        """
        self.stats["total_requests"] += 1
        
        # Generate key if not a string
        if not isinstance(key, str):
            key = self._generate_key(key)
            
        # Check memory cache first
        if key in self.memory_cache:
            timestamp, value = self.memory_cache[key]
            
            # Check if entry is expired
            if self._is_expired(timestamp):
                del self.memory_cache[key]
                return self._check_disk_cache(key)
            
            # Valid cache hit
            self.stats["hits"] += 1
            return value
            
        # Check disk cache if enabled
        return self._check_disk_cache(key)
    
    def _check_disk_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Check disk cache for an entry.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.persist_to_disk:
            self.stats["misses"] += 1
            return None
            
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if entry is expired
                if self._is_expired(cache_data.get("timestamp", 0)):
                    os.remove(cache_file)
                    self.stats["misses"] += 1
                    return None
                    
                # Valid disk cache hit
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1
                
                # Add to memory cache for faster access next time
                self.memory_cache[key] = (cache_data["timestamp"], cache_data["value"])
                
                return cache_data["value"]
                
            except (json.JSONDecodeError, KeyError, IOError) as e:
                logger.warning(f"Error reading cache file {cache_file}: {str(e)}")
                
        self.stats["misses"] += 1
        return None
    
    def put(self, key: Union[str, Dict[str, Any]], value: Any) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key or content to hash for key
            value: Value to store
        """
        # Generate key if not a string
        if not isinstance(key, str):
            key = self._generate_key(key)
            
        # Store in memory cache
        timestamp = time.time()
        self.memory_cache[key] = (timestamp, value)
        
        # Manage memory cache size
        if len(self.memory_cache) > self.max_memory_entries:
            self._evict_oldest()
            
        # Store in disk cache if enabled
        if self.persist_to_disk:
            self._save_to_disk(key, timestamp, value)
    
    def _evict_oldest(self) -> None:
        """Evict the oldest entries from memory cache to maintain size limit."""
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            [(k, v[0]) for k, v in self.memory_cache.items()],
            key=lambda x: x[1]
        )
        
        # Remove oldest entries until we're under the limit
        entries_to_remove = max(1, len(self.memory_cache) - self.max_memory_entries + 10)  # Remove in batches
        
        for i in range(min(entries_to_remove, len(sorted_entries))):
            key_to_remove = sorted_entries[i][0]
            del self.memory_cache[key_to_remove]
            self.stats["evictions"] += 1
    
    def _save_to_disk(self, key: str, timestamp: float, value: Any) -> None:
        """
        Save a cache entry to disk.
        
        Args:
            key: Cache key
            timestamp: Entry timestamp
            value: Value to store
        """
        if not self.persist_to_disk:
            return
            
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            cache_data = {
                "timestamp": timestamp,
                "value": value
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
                
        except (TypeError, IOError) as e:
            logger.warning(f"Error writing cache file {cache_file}: {str(e)}")
    
    def invalidate(self, key: Union[str, Dict[str, Any]]) -> None:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key or content to hash for key
        """
        # Generate key if not a string
        if not isinstance(key, str):
            key = self._generate_key(key)
            
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            
        # Remove from disk cache
        if self.persist_to_disk:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except IOError as e:
                    logger.warning(f"Error removing cache file {cache_file}: {str(e)}")
    
    def clear(self) -> None:
        """Clear the entire cache (memory and disk)."""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache
        if self.persist_to_disk:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except IOError:
                        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.stats.copy()
        
        # Calculate additional stats
        stats["memory_entries"] = len(self.memory_cache)
        stats["hit_rate"] = stats["hits"] / max(1, stats["total_requests"]) * 100
        stats["disk_hit_rate"] = stats["disk_hits"] / max(1, stats["hits"]) * 100
        stats["uptime_seconds"] = time.time() - stats["created_at"]
        
        # Count disk cache entries
        if self.persist_to_disk:
            stats["disk_entries"] = sum(1 for f in os.listdir(self.cache_dir) if f.endswith(".json"))
        else:
            stats["disk_entries"] = 0
            
        return stats