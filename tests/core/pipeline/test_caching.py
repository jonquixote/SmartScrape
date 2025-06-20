"""
Test for pipeline caching functionality.

This test verifies that the caching components work correctly to improve
pipeline performance by storing and retrieving results from various cache backends.
"""

import asyncio
import unittest
import time
import json
import os
import tempfile
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Import the classes we want to test
# We'll need to create the caching.py module with these classes first
from core.pipeline.caching import (
    CacheKey,
    CacheEntry,
    CacheBackend,
    MemoryCacheBackend,
    FileCacheBackend,
    CachePolicy,
    CacheManager,
    CachedPipelineStage
)
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class SimpleStage(PipelineStage):
    """Simple stage for testing caching."""
    
    def __init__(self, name, result=None, execution_time=0.1):
        """Initialize test stage."""
        super().__init__()
        self.name = name
        self.result = result
        self.execution_time = execution_time
        self.process_called = False
        self.execution_count = 0
        
    async def process(self, context):
        """Process this stage."""
        self.process_called = True
        self.execution_count += 1
        
        # Simulate work
        await asyncio.sleep(self.execution_time)
        
        # Store result in context if provided
        if self.result is not None:
            context.set(f"{self.name}_result", self.result)
            
        # Add execution timestamp to context
        context.set(f"{self.name}_executed_at", time.time())
        
        return True


class CachingTest(unittest.TestCase):
    """Test suite for pipeline caching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.context = PipelineContext({"test_input": "sample_data"})
        
        # Create a temp directory for file cache
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def _run_cached_stage(self, stage, context=None, use_cache=True):
        """Run a stage with caching and return execution time."""
        context = context or self.context
        start_time = time.time()
        result = await stage.process(context)
        return time.time() - start_time, result
    
    def test_cache_key_generation(self):
        """Test cache key generation and uniqueness."""
        # Create cache keys with different inputs
        key1 = CacheKey("test_stage", {"input": "value1"})
        key2 = CacheKey("test_stage", {"input": "value2"})
        key3 = CacheKey("test_stage", {"input": "value1"})
        key4 = CacheKey("different_stage", {"input": "value1"})
        
        # Test string representation
        self.assertIsInstance(str(key1), str)
        
        # Test equality
        self.assertEqual(key1, key3)  # Same stage and input
        self.assertNotEqual(key1, key2)  # Different input
        self.assertNotEqual(key1, key4)  # Different stage
        
        # Test hashing (for dictionary keys)
        cache_dict = {}
        cache_dict[key1] = "result1"
        cache_dict[key2] = "result2"
        cache_dict[key4] = "result4"
        
        self.assertEqual(len(cache_dict), 3)
        self.assertEqual(cache_dict[key3], "result1")  # key3 should match key1
    
    def test_memory_cache_backend(self):
        """Test the memory cache backend."""
        cache = MemoryCacheBackend(max_size=100)
        key = CacheKey("test_stage", {"input": "value"})
        entry = CacheEntry({"result": "test_result"}, 60)
        
        # Cache should start empty
        self.assertFalse(cache.has(key))
        
        # Store and retrieve
        cache.set(key, entry)
        self.assertTrue(cache.has(key))
        retrieved = cache.get(key)
        self.assertEqual(retrieved.data["result"], "test_result")
        
        # Expiration
        expired_entry = CacheEntry({"result": "expired"}, -1)  # Negative TTL = expired
        cache.set(CacheKey("expired", {}), expired_entry)
        self.assertFalse(cache.has(CacheKey("expired", {})))
        
        # Size limit
        small_cache = MemoryCacheBackend(max_size=2)
        small_cache.set(CacheKey("stage1", {}), CacheEntry({"result": 1}, 60))
        small_cache.set(CacheKey("stage2", {}), CacheEntry({"result": 2}, 60))
        # This should evict the oldest entry
        small_cache.set(CacheKey("stage3", {}), CacheEntry({"result": 3}, 60))
        
        self.assertFalse(small_cache.has(CacheKey("stage1", {})))
        self.assertTrue(small_cache.has(CacheKey("stage2", {})))
        self.assertTrue(small_cache.has(CacheKey("stage3", {})))
        
        # Clear cache
        cache.clear()
        self.assertFalse(cache.has(key))
    
    def test_file_cache_backend(self):
        """Test the file cache backend."""
        cache_dir = os.path.join(self.temp_dir, "file_cache")
        cache = FileCacheBackend(cache_dir)
        key = CacheKey("test_stage", {"input": "value"})
        entry = CacheEntry({"result": "test_result"}, 60)
        
        # Cache should start empty
        self.assertFalse(cache.has(key))
        
        # Store and retrieve
        cache.set(key, entry)
        self.assertTrue(cache.has(key))
        retrieved = cache.get(key)
        self.assertEqual(retrieved.data["result"], "test_result")
        
        # Check file was created
        self.assertTrue(os.path.exists(cache_dir))
        
        # Expiration
        expired_entry = CacheEntry({"result": "expired"}, -1)  # Negative TTL = expired
        cache.set(CacheKey("expired", {}), expired_entry)
        self.assertFalse(cache.has(CacheKey("expired", {})))
        
        # Clear cache
        cache.clear()
        self.assertFalse(cache.has(key))
    
    def test_cache_manager(self):
        """Test the cache manager."""
        # Create cache manager with memory backend
        manager = CacheManager()
        manager.register_backend("memory", MemoryCacheBackend())
        
        # Test standard operations
        key = CacheKey("test_stage", {"input": "value"})
        entry = CacheEntry({"result": "test_result"}, 60)
        
        # Cache should start empty
        self.assertFalse(manager.has(key))
        
        # Store and retrieve
        manager.set(key, entry)
        self.assertTrue(manager.has(key))
        retrieved = manager.get(key)
        self.assertEqual(retrieved.data["result"], "test_result")
        
        # Test with multiple backends
        manager.register_backend("file", FileCacheBackend(os.path.join(self.temp_dir, "file_cache")))
        
        # Default policy should use all backends
        manager.set(key, entry, policy=CachePolicy.ALL_BACKENDS)
        self.assertTrue(manager.has(key, backend_name="memory"))
        self.assertTrue(manager.has(key, backend_name="file"))
        
        # Test with specific backend
        specific_key = CacheKey("specific_stage", {"input": "specific"})
        manager.set(specific_key, entry, backend_name="memory")
        self.assertTrue(manager.has(specific_key, backend_name="memory"))
        self.assertFalse(manager.has(specific_key, backend_name="file"))
        
        # Test clear
        manager.clear(backend_name="memory")
        self.assertFalse(manager.has(key, backend_name="memory"))
        self.assertTrue(manager.has(key, backend_name="file"))
    
    def test_cached_pipeline_stage(self):
        """Test the cached pipeline stage wrapper."""
        # Create a simple stage that we'll cache
        base_stage = SimpleStage("test_stage", result="original result", execution_time=0.2)
        
        # Create cache manager
        manager = CacheManager()
        manager.register_backend("memory", MemoryCacheBackend())
        
        # Create cached stage
        cached_stage = CachedPipelineStage(
            wrapped_stage=base_stage,
            cache_manager=manager,
            cache_ttl=60,
            cache_key_fields=["test_input"]  # Use test_input from context for cache key
        )
        
        # First execution should run the stage and cache the result
        async def test_cached_execution():
            # First run - should execute the wrapped stage
            execution_time1, _ = await self._run_cached_stage(cached_stage, self.context)
            self.assertTrue(base_stage.process_called)
            self.assertEqual(base_stage.execution_count, 1)
            self.assertEqual(self.context.get("test_stage_result"), "original result")
            
            # Reset the flag for next test
            base_stage.process_called = False
            
            # Second run with same input - should use cache
            execution_time2, _ = await self._run_cached_stage(cached_stage, self.context)
            self.assertFalse(base_stage.process_called)  # Stage should not be executed
            self.assertEqual(base_stage.execution_count, 1)  # Count should not increase
            self.assertEqual(self.context.get("test_stage_result"), "original result")
            
            # Execution should be faster with cache
            self.assertLess(execution_time2, execution_time1)
            
            # Change input - should cause cache miss
            new_context = PipelineContext({"test_input": "different_data"})
            execution_time3, _ = await self._run_cached_stage(cached_stage, new_context)
            self.assertTrue(base_stage.process_called)  # Stage should be executed
            self.assertEqual(base_stage.execution_count, 2)  # Count should increase
            
            # Disable cache - should execute even with same input
            cached_stage.use_cache = False
            base_stage.process_called = False
            await self._run_cached_stage(cached_stage, self.context)
            self.assertTrue(base_stage.process_called)  # Stage should be executed
            self.assertEqual(base_stage.execution_count, 3)  # Count should increase
        
        asyncio.run(test_cached_execution())
    
    def test_cache_invalidation(self):
        """Test cache invalidation strategies."""
        # Create cache manager
        manager = CacheManager()
        manager.register_backend("memory", MemoryCacheBackend())
        
        # Create base stage
        base_stage = SimpleStage("test_stage", result="result", execution_time=0.1)
        
        # Create cached stage with TTL of 1 second
        cached_stage = CachedPipelineStage(
            wrapped_stage=base_stage,
            cache_manager=manager,
            cache_ttl=1,  # Very short TTL for testing
            cache_key_fields=["test_input"]
        )
        
        # Test expiration-based invalidation
        async def test_invalidation():
            # First run - caches the result
            await cached_stage.process(self.context)
            self.assertEqual(base_stage.execution_count, 1)
            
            # Second run - should use cache
            await cached_stage.process(self.context)
            self.assertEqual(base_stage.execution_count, 1)
            
            # Wait for cache to expire
            await asyncio.sleep(1.1)
            
            # Run again - should execute because cache expired
            await cached_stage.process(self.context)
            self.assertEqual(base_stage.execution_count, 2)
            
            # Test manual invalidation
            key = cached_stage._create_cache_key(self.context)
            manager.invalidate(key)
            
            # Run again - should execute because we invalidated
            await cached_stage.process(self.context)
            self.assertEqual(base_stage.execution_count, 3)
        
        asyncio.run(test_invalidation())
    
    def test_cache_with_complex_data(self):
        """Test caching with complex nested data structures."""
        # Create a context with complex data
        complex_context = PipelineContext({
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"}
            },
            "array": [
                {"id": 1, "name": "item1"},
                {"id": 2, "name": "item2"}
            ]
        })
        
        # Create a stage that returns complex data
        complex_data = {
            "result": {
                "status": "success",
                "items": [
                    {"id": 1, "processed": True},
                    {"id": 2, "processed": False}
                ],
                "metadata": {
                    "count": 2,
                    "timestamp": "2023-01-01T00:00:00"
                }
            }
        }
        base_stage = SimpleStage("complex_stage", result=complex_data)
        
        # Create cache manager and cached stage
        manager = CacheManager()
        manager.register_backend("file", FileCacheBackend(os.path.join(self.temp_dir, "complex_cache")))
        
        cached_stage = CachedPipelineStage(
            wrapped_stage=base_stage,
            cache_manager=manager,
            backend_name="file",  # Use file backend to test serialization/deserialization
            cache_ttl=60,
            cache_key_fields=["nested.dict.key", "array[0].id"]  # Test complex key fields
        )
        
        # Test with complex data
        async def test_complex_caching():
            # First run - caches the result
            await cached_stage.process(complex_context)
            self.assertEqual(base_stage.execution_count, 1)
            
            # Second run - should use cache
            base_stage.process_called = False
            await cached_stage.process(complex_context)
            self.assertFalse(base_stage.process_called)
            
            # Verify the cached result matches the complex data
            result = complex_context.get("complex_stage_result")
            self.assertEqual(result, complex_data)
            
            # Change a field used in the cache key
            modified_context = PipelineContext({
                "nested": {
                    "list": [1, 2, 3],
                    "dict": {"key": "different_value"}  # Changed
                },
                "array": [
                    {"id": 1, "name": "item1"},
                    {"id": 2, "name": "item2"}
                ]
            })
            
            # Should cause cache miss
            base_stage.process_called = False
            await cached_stage.process(modified_context)
            self.assertTrue(base_stage.process_called)
        
        asyncio.run(test_complex_caching())


if __name__ == "__main__":
    unittest.main()