import hashlib
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Union, List, Tuple
import os
from pathlib import Path
import pickle

class AICache:
    """
    Advanced caching system for AI responses with multiple backend strategies.
    
    Features:
    - Multiple backend support (memory and disk)
    - TTL (Time To Live) support for cache entries
    - Context-aware key generation
    - Cache statistics tracking
    - Thread-safe operations
    - Configurable size limits
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the cache with the given configuration.
        
        Args:
            config: Configuration dictionary with the following options:
                - backend: "memory" or "disk" (default: "memory")
                - max_size: Maximum number of items to store (default: 1000)
                - default_ttl: Default TTL in seconds (default: 3600)
                - cache_dir: Directory for disk cache (default: ".cache")
                - context_keys: List of context keys to include in key generation
        """
        config = config or {}
        self.logger = logging.getLogger("ai_cache")
        
        # Configure backend
        self.backend_type = config.get("backend", "memory")
        self.max_size = config.get("max_size", 1000)
        self.default_ttl = config.get("default_ttl", 3600)  # 1 hour default
        self.cache_dir = config.get("cache_dir", ".cache")
        self.context_keys = config.get("context_keys", ["temperature", "max_tokens", "task_type"])
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "token_savings": 0,
            "cost_savings": 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize backend
        self.backend = self._initialize_backend()
        self.logger.info(f"Initialized {self.backend_type} cache with max size {self.max_size}")
    
    def _initialize_backend(self) -> Any:
        """
        Initialize the appropriate cache backend.
        
        Returns:
            Cache backend object
        """
        try:
            if self.backend_type == "memory":
                # Use TTLCache from cachetools for in-memory caching with TTL support
                try:
                    from cachetools import TTLCache
                    return TTLCache(maxsize=self.max_size, ttl=self.default_ttl)
                except ImportError:
                    self.logger.warning("cachetools not installed, using dict for memory cache (no TTL support)")
                    return {}
                    
            elif self.backend_type == "disk":
                # Use diskcache for persistent disk-based caching
                try:
                    import diskcache
                    cache_dir = Path(self.cache_dir) / "ai_cache"
                    os.makedirs(cache_dir, exist_ok=True)
                    return diskcache.Cache(str(cache_dir))
                except ImportError:
                    self.logger.warning("diskcache not installed, falling back to memory cache")
                    return self._initialize_backend_fallback()
            else:
                self.logger.warning(f"Unsupported backend type: {self.backend_type}, falling back to memory cache")
                return self._initialize_backend_fallback()
                
        except Exception as e:
            self.logger.error(f"Error initializing cache backend: {str(e)}")
            return self._initialize_backend_fallback()
    
    def _initialize_backend_fallback(self) -> Dict:
        """Fallback to a simple dictionary if other backends fail."""
        self.backend_type = "memory_simple"
        return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with smart key normalization.
        
        Args:
            key: The cache key
            default: Default value to return if key not found
            
        Returns:
            Cached value or default if not found
        """
        if not key:
            return default
            
        normalized_key = self._normalize_key(key)
        
        with self._lock:
            try:
                if self.backend_type == "memory_simple":
                    # Check TTL for simple memory cache
                    now = time.time()
                    if normalized_key in self.backend:
                        value, expiry = self.backend[normalized_key]
                        if expiry is None or now < expiry:
                            self.stats["hits"] += 1
                            
                            # Track token and cost savings if available
                            if isinstance(value, dict) and "_metadata" in value:
                                metadata = value.get("_metadata", {})
                                self.stats["token_savings"] += metadata.get("output_tokens", 0)
                                self.stats["cost_savings"] += metadata.get("total_cost", 0)
                                
                            return value
                        else:
                            # Expired
                            del self.backend[normalized_key]
                    
                    self.stats["misses"] += 1
                    return default
                else:
                    # Use backend's own get method
                    value = self.backend.get(normalized_key, default)
                    
                    if value is not default:
                        self.stats["hits"] += 1
                        
                        # Track token and cost savings if available
                        if isinstance(value, dict) and "_metadata" in value:
                            metadata = value.get("_metadata", {})
                            self.stats["token_savings"] += metadata.get("output_tokens", 0)
                            self.stats["cost_savings"] += metadata.get("total_cost", 0)
                    else:
                        self.stats["misses"] += 1
                        
                    return value
                    
            except Exception as e:
                self.logger.error(f"Error retrieving from cache: {str(e)}")
                self.stats["misses"] += 1
                return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional time-to-live.
        
        Args:
            key: The cache key
            value: Value to store
            ttl: Time to live in seconds (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        if not key:
            return False
            
        normalized_key = self._normalize_key(key)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            try:
                self.stats["sets"] += 1
                
                if self.backend_type == "memory_simple":
                    # Simple dictionary with manual TTL
                    if len(self.backend) >= self.max_size:
                        # Evict oldest entry (simple LRU)
                        oldest_key = min(self.backend, key=lambda k: self.backend[k][1] if self.backend[k][1] else float('inf'))
                        del self.backend[oldest_key]
                        self.stats["evictions"] += 1
                    
                    expiry = time.time() + ttl if ttl else None
                    self.backend[normalized_key] = (value, expiry)
                else:
                    # Use backend's own set method
                    if hasattr(self.backend, 'set'):
                        self.backend.set(normalized_key, value, expire=ttl)
                    else:
                        self.backend[normalized_key] = value
                        
                return True
                
            except Exception as e:
                self.logger.error(f"Error setting cache: {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: The key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not key:
            return False
            
        normalized_key = self._normalize_key(key)
        
        with self._lock:
            try:
                if self.backend_type == "memory_simple":
                    if normalized_key in self.backend:
                        del self.backend[normalized_key]
                        return True
                    return False
                else:
                    # Use backend's own delete method
                    if hasattr(self.backend, 'pop'):
                        self.backend.pop(normalized_key, None)
                        return True
                    elif hasattr(self.backend, 'delete'):
                        return self.backend.delete(normalized_key)
                    else:
                        if normalized_key in self.backend:
                            del self.backend[normalized_key]
                            return True
                        return False
                        
            except Exception as e:
                self.logger.error(f"Error deleting from cache: {str(e)}")
                return False
    
    def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if self.backend_type == "memory_simple":
                    self.backend.clear()
                else:
                    # Use backend's own clear method
                    if hasattr(self.backend, 'clear'):
                        self.backend.clear()
                    else:
                        self.backend = {}
                
                # Reset hit/miss statistics but keep cumulative savings
                self.stats["hits"] = 0
                self.stats["misses"] = 0
                self.stats["sets"] = 0
                self.stats["evictions"] = 0
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error clearing cache: {str(e)}")
                return False
    
    def generate_key(self, prompt: str, context: Dict[str, Any], model_name: str) -> str:
        """
        Generate a consistent cache key considering all relevant factors.
        
        Args:
            prompt: The prompt text
            context: Context dictionary with parameters
            model_name: Name of the AI model
            
        Returns:
            Normalized cache key
        """
        # Extract only the elements that affect the response
        key_elements = {
            "prompt": prompt,
            "model": model_name,
        }
        
        # Only include context elements that affect the output
        if context:
            relevant_context = {}
            for key in self.context_keys:
                if key in context:
                    # Handle nested dictionaries (like options)
                    if key == "options" and isinstance(context[key], dict):
                        for option_key in ["temperature", "max_tokens", "top_p"]:
                            if option_key in context[key]:
                                relevant_context[f"options.{option_key}"] = context[key][option_key]
                    else:
                        relevant_context[key] = context[key]
            
            if relevant_context:
                key_elements["context"] = relevant_context
                
        # Create a stable JSON representation and hash it
        return self._normalize_key(json.dumps(key_elements, sort_keys=True))
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalize a key for consistent caching.
        
        Args:
            key: The raw key
            
        Returns:
            Normalized (hashed) key
        """
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            stats = dict(self.stats)
            
            # Calculate additional metrics
            total_requests = stats["hits"] + stats["misses"]
            if total_requests > 0:
                stats["hit_rate"] = stats["hits"] / total_requests
            else:
                stats["hit_rate"] = 0
                
            # Add backend-specific stats
            if self.backend_type == "memory_simple":
                stats["current_size"] = len(self.backend)
                stats["max_size"] = self.max_size
            elif hasattr(self.backend, 'currsize'):
                stats["current_size"] = self.backend.currsize
                stats["max_size"] = self.backend.maxsize
            elif hasattr(self.backend, 'size'):
                stats["current_size"] = self.backend.size()
                stats["max_size"] = self.max_size
            elif hasattr(self.backend, '__len__'):
                stats["current_size"] = len(self.backend)
                stats["max_size"] = self.max_size
            
            return stats