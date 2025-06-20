import pytest
import time
import threading
import json
import hashlib
from unittest.mock import patch, MagicMock
from core.ai_cache import AICache

class TestAICache:
    """Test suite for the AICache class."""
    
    @pytest.fixture
    def memory_cache(self):
        """Create a memory-based cache for testing."""
        return AICache({"backend": "memory", "default_ttl": 60, "max_size": 100})
    
    @pytest.fixture
    def simple_memory_cache(self):
        """Create a simple memory-based cache for testing."""
        # Force the simple memory backend by mocking the import
        with patch('core.ai_cache.TTLCache', side_effect=ImportError):
            return AICache({"backend": "memory", "default_ttl": 60, "max_size": 100})
    
    @pytest.fixture
    def disk_cache(self, tmpdir):
        """Create a disk-based cache for testing using a temporary directory."""
        cache_dir = tmpdir.mkdir("cache")
        return AICache({
            "backend": "disk", 
            "default_ttl": 60, 
            "max_size": 100,
            "cache_dir": str(cache_dir)
        })
    
    def test_cache_initialization(self, memory_cache, simple_memory_cache, disk_cache):
        """Test that different cache backends initialize correctly."""
        assert memory_cache.backend_type == "memory"
        assert simple_memory_cache.backend_type == "memory_simple"
        assert disk_cache.backend_type == "disk"
        
        # Check initial statistics
        for cache in [memory_cache, simple_memory_cache, disk_cache]:
            stats = cache.get_stats()
            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["sets"] == 0
    
    def test_basic_set_get(self, memory_cache):
        """Test basic set/get operations."""
        # Set a value
        assert memory_cache.set("test_key", "test_value")
        
        # Get the value
        assert memory_cache.get("test_key") == "test_value"
        
        # Check that statistics were updated
        stats = memory_cache.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
    
    def test_basic_set_get_simple_backend(self, simple_memory_cache):
        """Test basic set/get operations with simple memory backend."""
        # Set a value
        assert simple_memory_cache.set("test_key", "test_value")
        
        # Get the value
        assert simple_memory_cache.get("test_key") == "test_value"
        
        # Check that statistics were updated
        stats = simple_memory_cache.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
    
    def test_basic_set_get_disk_backend(self, disk_cache):
        """Test basic set/get operations with disk backend."""
        # Set a value
        assert disk_cache.set("test_key", "test_value")
        
        # Get the value
        assert disk_cache.get("test_key") == "test_value"
        
        # Check that statistics were updated
        stats = disk_cache.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
    
    def test_get_nonexistent_key(self, memory_cache):
        """Test getting a non-existent key."""
        assert memory_cache.get("non_existent") is None
        
        # Provide a default value
        assert memory_cache.get("non_existent", "default") == "default"
        
        # Check that statistics were updated
        stats = memory_cache.get_stats()
        assert stats["misses"] == 2
    
    def test_delete_key(self, memory_cache):
        """Test deleting cache entries."""
        # Set a value
        memory_cache.set("delete_me", "value")
        assert memory_cache.get("delete_me") == "value"
        
        # Delete it
        assert memory_cache.delete("delete_me")
        assert memory_cache.get("delete_me") is None
        
        # Try deleting non-existent key
        assert not memory_cache.delete("never_existed")
    
    def test_clear_cache(self, memory_cache):
        """Test clearing the cache."""
        # Set multiple values
        memory_cache.set("key1", "value1")
        memory_cache.set("key2", "value2")
        
        # Verify values are there
        assert memory_cache.get("key1") == "value1"
        assert memory_cache.get("key2") == "value2"
        
        # Clear the cache
        assert memory_cache.clear()
        
        # Verify all entries are gone
        assert memory_cache.get("key1") is None
        assert memory_cache.get("key2") is None
        
        # Check statistics were reset
        stats = memory_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2  # From the two gets after clearing
        assert stats["sets"] == 0
    
    def test_key_normalization(self, memory_cache):
        """Test that keys are normalized properly."""
        # Set with one key format
        memory_cache.set("Test Key", "value")
        
        # Different key should access same value due to normalization
        assert memory_cache.get("Test Key") == "value"
        
        # Key normalization should be consistent
        key1 = memory_cache._normalize_key("test string")
        key2 = memory_cache._normalize_key("test string")
        assert key1 == key2
        
        # Verify it's an MD5 hash
        assert len(key1) == 32
        # Try to reproduce the hash manually
        expected_hash = hashlib.md5("test string".encode("utf-8")).hexdigest()
        assert key1 == expected_hash
    
    def test_ttl_expiration(self, simple_memory_cache):
        """Test TTL expiration for cache entries."""
        # Set with a very short TTL
        simple_memory_cache.set("expires_quickly", "value", ttl=1)
        
        # Should be available immediately
        assert simple_memory_cache.get("expires_quickly") == "value"
        
        # Wait for expiration
        time.sleep(1.5)
        
        # Should be expired now
        assert simple_memory_cache.get("expires_quickly") is None
        
        # Set with default TTL (longer)
        simple_memory_cache.set("expires_default", "value")
        
        # Wait a short time - should still be available
        time.sleep(1)
        assert simple_memory_cache.get("expires_default") == "value"
    
    def test_generate_key(self, memory_cache):
        """Test key generation from prompt and context."""
        prompt = "What is the capital of France?"
        context = {
            "temperature": 0.7,
            "max_tokens": 100,
            "task_type": "qa",
            "irrelevant_key": "should_be_ignored"
        }
        model_name = "gpt-3.5-turbo"
        
        # Generate key
        key = memory_cache.generate_key(prompt, context, model_name)
        
        # Should be a string of appropriate length for MD5
        assert isinstance(key, str)
        assert len(key) == 32
        
        # Same inputs should generate same key
        key2 = memory_cache.generate_key(prompt, context, model_name)
        assert key == key2
        
        # Different model should generate different key
        key3 = memory_cache.generate_key(prompt, context, "different-model")
        assert key != key3
        
        # Different temperature should generate different key
        context2 = dict(context)
        context2["temperature"] = 0.8
        key4 = memory_cache.generate_key(prompt, context2, model_name)
        assert key != key4
        
        # Irrelevant keys should be ignored - shouldn't affect the key
        context3 = dict(context)
        context3["another_irrelevant"] = "value"
        key5 = memory_cache.generate_key(prompt, context3, model_name)
        assert key == key5
    
    def test_nested_context_in_key_generation(self, memory_cache):
        """Test key generation with nested context."""
        prompt = "What is the capital of France?"
        context = {
            "options": {
                "temperature": 0.7,
                "max_tokens": 100,
                "ignored_option": "value"
            },
            "task_type": "qa"
        }
        model_name = "gpt-3.5-turbo"
        
        # Generate key
        key = memory_cache.generate_key(prompt, context, model_name)
        
        # Change ignored option - should generate same key
        context2 = dict(context)
        context2["options"] = dict(context["options"])
        context2["options"]["ignored_option"] = "different"
        key2 = memory_cache.generate_key(prompt, context2, model_name)
        assert key == key2
        
        # Change temperature - should generate different key
        context3 = dict(context)
        context3["options"] = dict(context["options"])
        context3["options"]["temperature"] = 0.8
        key3 = memory_cache.generate_key(prompt, context3, model_name)
        assert key != key3
    
    def test_cache_statistics(self, memory_cache):
        """Test cache statistics tracking."""
        # Initial stats
        stats = memory_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["hit_rate"] == 0
        
        # Add some operations
        memory_cache.set("test", "value")
        memory_cache.get("test")
        memory_cache.get("test")
        memory_cache.get("missing")
        
        # Check updated stats
        stats = memory_cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["hit_rate"] == 2/3  # 2 hits out of 3 requests
        
        # Should have current size info
        assert "current_size" in stats
        assert "max_size" in stats
    
    def test_token_and_cost_savings(self, memory_cache):
        """Test tracking of token and cost savings."""
        # Set a response with metadata
        response = {
            "content": "This is a test response",
            "_metadata": {
                "output_tokens": 100,
                "total_cost": 0.0025
            }
        }
        memory_cache.set("savings_test", response)
        
        # Get it multiple times
        for _ in range(3):
            memory_cache.get("savings_test")
        
        # Check the savings stats
        stats = memory_cache.get_stats()
        assert stats["token_savings"] == 300  # 100 tokens saved 3 times
        assert stats["cost_savings"] == 0.0075  # 0.0025 saved 3 times
    
    def test_max_size_and_eviction(self, simple_memory_cache):
        """Test max size limit and eviction."""
        # Configure a small cache
        small_cache = AICache({"backend": "memory", "max_size": 3, "default_ttl": 3600})
        
        # Force simple memory backend
        small_cache.backend_type = "memory_simple"
        small_cache.backend = {}
        
        # Fill the cache
        small_cache.set("key1", "value1")
        small_cache.set("key2", "value2")
        small_cache.set("key3", "value3")
        
        # Verify all entries are there
        assert small_cache.get("key1") == "value1"
        assert small_cache.get("key2") == "value2"
        assert small_cache.get("key3") == "value3"
        
        # Add one more entry, should evict the oldest
        small_cache.set("key4", "value4")
        
        # One of the first entries should be evicted (depends on implementation)
        # At least one should be gone, and key4 should be present
        present_count = sum(1 for k in ["key1", "key2", "key3"] if small_cache.get(k) is not None)
        assert present_count < 3
        assert small_cache.get("key4") == "value4"
        
        # Check that eviction was tracked
        stats = small_cache.get_stats()
        assert stats["evictions"] >= 1
    
    def test_thread_safety(self, memory_cache):
        """Test thread safety of cache operations."""
        # Number of operations per thread
        operations = 100
        
        # Track successful operations
        successes = {
            "set": 0,
            "get": 0
        }
        
        def worker():
            for i in range(operations):
                # Alternate between set and get
                if i % 2 == 0:
                    if memory_cache.set(f"thread_key_{i}", f"value_{i}"):
                        successes["set"] += 1
                else:
                    memory_cache.get(f"thread_key_{i-1}")
                    successes["get"] += 1
        
        # Create and start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that operations were successful
        # We're mainly checking that there were no exceptions or deadlocks
        assert successes["set"] > 0
        assert successes["get"] > 0
    
    def test_empty_key_handling(self, memory_cache):
        """Test handling of empty or None keys."""
        # Empty string key
        assert memory_cache.set("", "value") is False
        assert memory_cache.get("") is None
        assert memory_cache.delete("") is False
        
        # None key
        assert memory_cache.set(None, "value") is False
        assert memory_cache.get(None) is None
        assert memory_cache.delete(None) is False
    
    def test_error_handling(self):
        """Test error handling in cache operations."""
        # Create a cache with a mocked backend that raises exceptions
        cache = AICache({"backend": "memory"})
        cache.backend = MagicMock()
        cache.backend.get.side_effect = Exception("Test error")
        cache.backend.set.side_effect = Exception("Test error")
        
        # Operations should not raise exceptions, but return appropriate values
        assert cache.get("error_key") is None
        assert cache.set("error_key", "value") is False
        
        # Check that stats still got updated for misses
        assert cache.stats["misses"] > 0
    
    def test_fallback_to_simple_memory(self):
        """Test fallback to simple memory cache when backends fail."""
        # Test fallback when an unsupported backend is requested
        with patch('core.ai_cache.AICache._initialize_backend_fallback', return_value={}) as mock_fallback:
            cache = AICache({"backend": "unsupported"})
            mock_fallback.assert_called_once()
        
        # Test fallback when disk backend import fails
        with patch('core.ai_cache.diskcache', side_effect=ImportError), \
             patch('core.ai_cache.AICache._initialize_backend_fallback', return_value={}) as mock_fallback:
            cache = AICache({"backend": "disk"})
            mock_fallback.assert_called_once()