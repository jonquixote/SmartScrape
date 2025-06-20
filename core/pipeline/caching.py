"""
Pipeline Caching Module.

This module provides caching functionality for pipeline execution results,
improving performance by avoiding redundant calculations.
"""

import os
import json
import time
import hashlib
import logging
import pickle
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime, timedelta

from core.pipeline.context import PipelineContext


class CacheBackend(ABC):
    """Abstract base class for cache backend implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            Optional[Any]: The cached value, or None if not in cache.
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key (str): The cache key.
            value (Any): The value to cache.
            ttl (Optional[int]): Time-to-live in seconds, or None for no expiry.
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            bool: True if deleted, False if not found.
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            bool: True if exists, False otherwise.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values from the cache."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize in-memory cache.
        
        Args:
            max_size (int): Maximum number of items to store.
        """
        self.max_size = max_size
        self._cache = {}
        self._expiry = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        
        self.logger = logging.getLogger("pipeline.cache.memory")
        self.logger.info(f"Initialized memory cache (max_size={max_size})")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            Optional[Any]: The cached value, or None if not in cache.
        """
        with self._lock:
            # Check if key exists and not expired
            if key in self._cache:
                if key in self._expiry and self._expiry[key] < time.time():
                    # Key has expired
                    del self._cache[key]
                    del self._expiry[key]
                    self._misses += 1
                    return None
                
                # Key exists and not expired
                self._hits += 1
                return self._cache[key]
            
            # Key not found
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key (str): The cache key.
            value (Any): The value to cache.
            ttl (Optional[int]): Time-to-live in seconds, or None for no expiry.
        """
        with self._lock:
            # Check if we need to evict items
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Simple LRU - remove oldest expiring items first, then remove any item
                if self._expiry:
                    oldest_key = min(self._expiry.items(), key=lambda x: x[1])[0]
                    del self._cache[oldest_key]
                    del self._expiry[oldest_key]
                else:
                    # Just remove any key
                    del_key = next(iter(self._cache))
                    del self._cache[del_key]
            
            # Add the new item
            self._cache[key] = value
            
            # Set expiry if provided
            if ttl is not None:
                self._expiry[key] = time.time() + ttl
            elif key in self._expiry:
                del self._expiry[key]
            
            self._sets += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            bool: True if deleted, False if not found.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
                self._deletes += 1
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.
        
        Args:
            key (str): The cache key.
            
        Returns:
            bool: True if exists and not expired, False otherwise.
        """
        with self._lock:
            if key in self._cache:
                if key in self._expiry and self._expiry[key] < time.time():
                    return False
                return True
            return False
    
    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) * 100 if total > 0 else 0
            
            return {
                "type": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "sets": self._sets,
                "deletes": self._deletes,
                "items_with_ttl": len(self._expiry)
            }


class FileCacheBackend(CacheBackend):
    """File-based cache backend implementation."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100):
        """
        Initialize file-based cache.
        
        Args:
            cache_dir (str): Directory to store cache files.
            max_size_mb (int): Maximum size in megabytes.
        """
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        self._metadata = {
            "entries": {},
            "stats": {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0
            }
        }
        
        self.logger = logging.getLogger("pipeline.cache.file")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            self.logger.info(f"Created cache directory: {cache_dir}")
        
        # Load metadata if it exists
        self._load_metadata()
        
        self.logger.info(f"Initialized file cache in {cache_dir} (max_size={max_size_mb}MB)")
    
    def _load_metadata(self) -> None:
        """Load cache metadata from file."""
        if os.path.exists(self._metadata_file):
            try:
                with open(self._metadata_file, "r") as f:
                    self._metadata = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {str(e)}")
    
    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {str(e)}")
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get filesystem path for a cache key.
        
        Args:
            key (str): The cache key.
            
        Returns:
            str: Path to the cache file.
        """
        # Use hash of key as filename to avoid filesystem issues
        hashed = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed}.cache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            Optional[Any]: The cached value, or None if not in cache.
        """
        with self._lock:
            # Check if key exists in metadata
            entries = self._metadata["entries"]
            if key not in entries:
                self._metadata["stats"]["misses"] += 1
                return None
            
            # Check if entry has expired
            entry = entries[key]
            if "expiry" in entry and entry["expiry"] < time.time():
                # Delete expired file
                cache_path = self._get_cache_path(key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                
                # Remove from metadata
                del entries[key]
                self._save_metadata()
                
                self._metadata["stats"]["misses"] += 1
                return None
            
            # Load cached value from file
            cache_path = self._get_cache_path(key)
            if not os.path.exists(cache_path):
                # File missing, clean up metadata
                del entries[key]
                self._save_metadata()
                
                self._metadata["stats"]["misses"] += 1
                return None
            
            try:
                with open(cache_path, "rb") as f:
                    value = pickle.load(f)
                
                self._metadata["stats"]["hits"] += 1
                return value
            except Exception as e:
                self.logger.warning(f"Failed to load cache file for key '{key}': {str(e)}")
                
                # Clean up corrupted file
                os.remove(cache_path)
                del entries[key]
                self._save_metadata()
                
                self._metadata["stats"]["misses"] += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key (str): The cache key.
            value (Any): The value to cache.
            ttl (Optional[int]): Time-to-live in seconds, or None for no expiry.
        """
        with self._lock:
            cache_path = self._get_cache_path(key)
            
            # Save value to file
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)
            except Exception as e:
                self.logger.warning(f"Failed to save cache file for key '{key}': {str(e)}")
                return
            
            # Update metadata
            entry = {
                "created": time.time(),
                "path": cache_path,
                "size": os.path.getsize(cache_path)
            }
            
            if ttl is not None:
                entry["expiry"] = time.time() + ttl
            
            self._metadata["entries"][key] = entry
            self._metadata["stats"]["sets"] += 1
            
            # Check cache size and clean up if needed
            self._cleanup_if_needed()
            
            # Save metadata
            self._save_metadata()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up old cache files if max size is exceeded."""
        # Calculate current size
        total_size_bytes = sum(
            entry.get("size", 0) for entry in self._metadata["entries"].values()
        )
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        # If size is under limit, nothing to do
        if total_size_bytes <= max_size_bytes:
            return
        
        # Sort entries by expiry (expired first) then by creation time (oldest first)
        def sort_key(item):
            key, entry = item
            expiry = entry.get("expiry", float("inf"))
            created = entry.get("created", 0)
            return (expiry, created)
        
        sorted_entries = sorted(self._metadata["entries"].items(), key=sort_key)
        
        # Remove entries until size is under limit
        current_size = total_size_bytes
        
        for key, entry in sorted_entries:
            if current_size <= max_size_bytes:
                break
                
            # Delete file
            cache_path = entry.get("path")
            if cache_path and os.path.exists(cache_path):
                try:
                    file_size = os.path.getsize(cache_path)
                    os.remove(cache_path)
                    current_size -= file_size
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file '{cache_path}': {str(e)}")
            
            # Remove from metadata
            del self._metadata["entries"][key]
            self._metadata["stats"]["deletes"] += 1
            
        self.logger.info(f"Cleaned up cache, freed {(total_size_bytes - current_size) / 1024 / 1024:.2f}MB")
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key (str): The cache key.
            
        Returns:
            bool: True if deleted, False if not found.
        """
        with self._lock:
            entries = self._metadata["entries"]
            
            if key not in entries:
                return False
            
            # Delete file
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file for key '{key}': {str(e)}")
            
            # Remove from metadata
            del entries[key]
            self._metadata["stats"]["deletes"] += 1
            self._save_metadata()
            
            return True
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is not expired.
        
        Args:
            key (str): The cache key.
            
        Returns:
            bool: True if exists and not expired, False otherwise.
        """
        with self._lock:
            entries = self._metadata["entries"]
            
            if key not in entries:
                return False
            
            # Check if entry has expired
            entry = entries[key]
            if "expiry" in entry and entry["expiry"] < time.time():
                return False
            
            # Check if file exists
            cache_path = self._get_cache_path(key)
            return os.path.exists(cache_path)
    
    def clear(self) -> None:
        """Clear all values from the cache."""
        with self._lock:
            # Delete all cache files
            for entry in self._metadata["entries"].values():
                cache_path = entry.get("path")
                if cache_path and os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file '{cache_path}': {str(e)}")
            
            # Clear metadata
            self._metadata["entries"] = {}
            self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        with self._lock:
            stats = self._metadata["stats"]
            total = stats["hits"] + stats["misses"]
            hit_rate = (stats["hits"] / total) * 100 if total > 0 else 0
            
            # Calculate current size
            total_size_bytes = sum(
                entry.get("size", 0) for entry in self._metadata["entries"].values()
            )
            
            return {
                "type": "file",
                "dir": self.cache_dir,
                "items": len(self._metadata["entries"]),
                "size_mb": total_size_bytes / 1024 / 1024,
                "max_size_mb": self.max_size_mb,
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": hit_rate,
                "sets": stats["sets"],
                "deletes": stats["deletes"]
            }


class PipelineCacheManager:
    """
    Manages caching for pipeline results.
    
    This class provides:
    - Configurable caching of pipeline results
    - Support for different cache backends
    - Automatic cache key generation based on inputs
    - Cache invalidation strategies
    - Cache statistics for monitoring
    """
    
    def __init__(self, 
                 backend: Optional[CacheBackend] = None, 
                 default_ttl: Optional[int] = None,
                 max_cache_entries: int = 1000):
        """
        Initialize pipeline cache manager.
        
        Args:
            backend (Optional[CacheBackend]): Cache backend to use, or None for in-memory.
            default_ttl (Optional[int]): Default time-to-live in seconds, or None for no expiry.
            max_cache_entries (int): Maximum number of entries for in-memory cache.
        """
        self.logger = logging.getLogger("pipeline.cache.manager")
        
        # Use provided backend or create in-memory backend
        self.backend = backend or MemoryCacheBackend(max_size=max_cache_entries)
        self.default_ttl = default_ttl
        
        # Cache configuration by pipeline
        self.pipeline_config = {}
        
        # Cache hit tracking
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info(f"Initialized pipeline cache manager with backend: {type(self.backend).__name__}")
    
    def configure_pipeline_cache(self, 
                               pipeline_name: str, 
                               enabled: bool = True,
                               ttl: Optional[int] = None,
                               key_generator: Optional[Callable] = None,
                               cache_filter: Optional[Callable] = None) -> None:
        """
        Configure caching for a specific pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            enabled (bool): Whether caching is enabled for this pipeline.
            ttl (Optional[int]): Time-to-live for this pipeline's cache entries.
            key_generator (Optional[Callable]): Custom function for generating cache keys.
            cache_filter (Optional[Callable]): Function to decide if a result should be cached.
        """
        self.pipeline_config[pipeline_name] = {
            "enabled": enabled,
            "ttl": ttl if ttl is not None else self.default_ttl,
            "key_generator": key_generator,
            "cache_filter": cache_filter
        }
        
        self.logger.info(f"Configured caching for pipeline '{pipeline_name}' (enabled={enabled})")
    
    def is_caching_enabled(self, pipeline_name: str) -> bool:
        """
        Check if caching is enabled for a pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            
        Returns:
            bool: True if caching is enabled, False otherwise.
        """
        if pipeline_name in self.pipeline_config:
            return self.pipeline_config[pipeline_name]["enabled"]
        return True  # Default to enabled
    
    def get_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Get cache configuration for a pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            
        Returns:
            Dict: Configuration for the pipeline.
        """
        if pipeline_name in self.pipeline_config:
            return self.pipeline_config[pipeline_name]
        return {
            "enabled": True,
            "ttl": self.default_ttl,
            "key_generator": None,
            "cache_filter": None
        }
    
    def _default_key_generator(self, pipeline_name: str, initial_data: Dict[str, Any]) -> str:
        """
        Generate a default cache key for pipeline inputs.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            initial_data (Dict): Initial data for the pipeline.
            
        Returns:
            str: Generated cache key.
        """
        # Create a stable JSON representation of the initial data
        try:
            # Sort keys for stable order
            json_str = json.dumps(initial_data, sort_keys=True)
            
            # Create hash of the JSON string
            key_hash = hashlib.md5(json_str.encode()).hexdigest()
            
            return f"{pipeline_name}:{key_hash}"
        except Exception as e:
            self.logger.warning(f"Failed to generate cache key: {str(e)}")
            
            # Fallback to pipeline name with timestamp
            return f"{pipeline_name}:{time.time()}"
    
    def _should_cache_result(self, 
                           pipeline_name: str, 
                           context: PipelineContext, 
                           config: Dict[str, Any]) -> bool:
        """
        Determine if a pipeline result should be cached.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            context (PipelineContext): Pipeline execution context.
            config (Dict): Cache configuration for the pipeline.
            
        Returns:
            bool: True if the result should be cached, False otherwise.
        """
        # If there's a custom filter, use it
        if config["cache_filter"] is not None:
            try:
                return config["cache_filter"](context)
            except Exception as e:
                self.logger.warning(f"Error in cache filter for '{pipeline_name}': {str(e)}")
        
        # Default behavior: cache only successful pipeline executions
        metrics = context.get_metrics()
        return not metrics["has_errors"]
    
    def get_cached_result(self, 
                        pipeline_name: str, 
                        initial_data: Dict[str, Any]) -> Optional[PipelineContext]:
        """
        Get cached result for a pipeline execution.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            initial_data (Dict): Initial data for the pipeline.
            
        Returns:
            Optional[PipelineContext]: Cached context or None if not found.
        """
        # Check if caching is enabled for this pipeline
        config = self.get_pipeline_config(pipeline_name)
        
        if not config["enabled"]:
            return None
        
        # Generate cache key
        key_generator = config["key_generator"] or self._default_key_generator
        key = key_generator(pipeline_name, initial_data)
        
        # Try to get from cache
        try:
            cached = self.backend.get(key)
            
            if cached is not None:
                self._cache_hits += 1
                self.logger.debug(f"Cache hit for pipeline '{pipeline_name}'")
                return cached
        except Exception as e:
            self.logger.warning(f"Error getting from cache: {str(e)}")
        
        self._cache_misses += 1
        self.logger.debug(f"Cache miss for pipeline '{pipeline_name}'")
        return None
    
    def cache_result(self, 
                    pipeline_name: str, 
                    initial_data: Dict[str, Any], 
                    context: PipelineContext) -> None:
        """
        Cache a pipeline execution result.
        
        Args:
            pipeline_name (str): Name of the pipeline.
            initial_data (Dict): Initial data for the pipeline.
            context (PipelineContext): Pipeline execution context.
        """
        # Check if caching is enabled for this pipeline
        config = self.get_pipeline_config(pipeline_name)
        
        if not config["enabled"]:
            return
        
        # Check if this result should be cached
        if not self._should_cache_result(pipeline_name, context, config):
            return
        
        # Generate cache key
        key_generator = config["key_generator"] or self._default_key_generator
        key = key_generator(pipeline_name, initial_data)
        
        # Cache the context
        try:
            self.backend.set(key, context, ttl=config["ttl"])
            self.logger.debug(f"Cached result for pipeline '{pipeline_name}'")
        except Exception as e:
            self.logger.warning(f"Error caching result: {str(e)}")
    
    def invalidate_pipeline_cache(self, pipeline_name: str) -> None:
        """
        Invalidate all cached results for a pipeline.
        
        Args:
            pipeline_name (str): Name of the pipeline.
        """
        # This is a simplistic implementation that clears the entire cache
        # A more sophisticated implementation would track keys by pipeline
        self.logger.info(f"Invalidating cache for pipeline '{pipeline_name}' (entire cache)")
        self.clear_cache()
    
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        try:
            self.backend.clear()
            self.logger.info("Cleared entire cache")
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        # Get backend stats
        backend_stats = self.backend.get_stats()
        
        # Add manager stats
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total) * 100 if total > 0 else 0
        
        return {
            "manager": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": hit_rate,
                "pipelines_configured": len(self.pipeline_config)
            },
            "backend": backend_stats
        }


# Global cache manager
_manager = PipelineCacheManager()


def get_cache_manager() -> PipelineCacheManager:
    """
    Get the global cache manager.
    
    Returns:
        PipelineCacheManager: The global manager instance.
    """
    return _manager