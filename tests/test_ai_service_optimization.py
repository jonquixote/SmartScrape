#!/usr/bin/env python
"""
AI Service Optimization Test Runner

This script runs tests specifically focused on the AI service optimization components,
including rule engine, caching, batch processing, and model selection.
"""

import asyncio
import time
import json
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_service import AIService
from core.rule_engine import RegexRule, FunctionRule
from core.ai_cache import AICache
from core.batch_processor import BatchProcessor
from core.model_selector import ModelSelector
from core.content_processor import ContentProcessor

async def test_rule_engine_performance():
    """Test the performance of rule-based processing vs. AI model calls."""
    print("\n=== Testing Rule Engine Performance ===\n")
    
    # Initialize AI service
    service = AIService()
    await service.initialize({
        "models": [{"name": "test-model", "type": "mock"}],
        "default_model": "test-model"
    })
    
    # Add sample rules
    service.rule_engine.add_rule(RegexRule(
        name="email_extractor",
        pattern=r"extract email: ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        template="Email: {0}",
        group=1,
        priority=10
    ))
    
    service.rule_engine.add_rule(RegexRule(
        name="phone_extractor",
        pattern=r"extract phone: (\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4})",
        template="Phone: {0}",
        group=1,
        priority=10
    ))
    
    # Test cases
    test_cases = [
        "extract email: test.user@example.com",
        "extract phone: 555-123-4567",
        "What is the capital of France?",
        "Tell me about the history of computing"
    ]
    
    results = {
        "rule_engine_time": [],
        "ai_model_time": [],
        "rule_engine_hits": 0,
        "ai_model_hits": 0
    }
    
    # Run tests
    for prompt in test_cases:
        # Test with rule engine
        start_time = time.time()
        result_with_rules = await service.generate_response(
            prompt,
            context={"use_rule_engine": True}
        )
        rule_time = time.time() - start_time
        results["rule_engine_time"].append(rule_time)
        
        if result_with_rules.get("source") == "rule_engine":
            results["rule_engine_hits"] += 1
            print(f"✓ Rule engine processed: '{prompt}' in {rule_time:.5f}s")
        else:
            print(f"× Rule engine skipped: '{prompt}'")
            
        # Test with AI model only
        start_time = time.time()
        result_no_rules = await service.generate_response(
            prompt,
            context={"use_rule_engine": False}
        )
        ai_time = time.time() - start_time
        results["ai_model_time"].append(ai_time)
        results["ai_model_hits"] += 1
        
        print(f"  AI model processed: '{prompt}' in {ai_time:.5f}s")
        print(f"  Speedup: {ai_time / rule_time:.2f}x" if rule_time > 0 else "")
        print()
    
    # Print summary
    if results["rule_engine_hits"] > 0:
        avg_rule_time = sum([t for t in results["rule_engine_time"] if t > 0]) / results["rule_engine_hits"]
        print(f"Average rule engine processing time: {avg_rule_time:.5f}s")
        
    avg_ai_time = sum(results["ai_model_time"]) / results["ai_model_hits"]
    print(f"Average AI model processing time: {avg_ai_time:.5f}s")
    
    if results["rule_engine_hits"] > 0:
        print(f"Average speedup with rule engine: {avg_ai_time / avg_rule_time:.2f}x")
    
    print(f"Rule engine hit rate: {results['rule_engine_hits'] / len(test_cases):.0%}")

async def test_caching_performance():
    """Test the performance improvement from caching."""
    print("\n=== Testing Cache Performance ===\n")
    
    # Initialize AI service
    service = AIService()
    await service.initialize({
        "models": [{"name": "test-model", "type": "mock"}],
        "default_model": "test-model",
        "cache": {
            "backend": "memory",
            "default_ttl": 3600
        }
    })
    
    # Test cases - mix of identical and similar prompts
    test_cases = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of France?",  # Exact cache hit
        "What is the capital of Italy?",
        "What is the capital of Germany?",  # Exact cache hit
        "What is the population of Paris?",
        "What is the population of Paris?",  # Exact cache hit
        "What is the capital of France? "  # Similar but not exact
    ]
    
    results = {
        "first_request_time": [],
        "cached_request_time": [],
        "cache_hits": 0,
        "cache_misses": 0
    }
    
    # Run tests
    for prompt in test_cases:
        # First try with cache disabled to get baseline
        service.ai_cache.clear()
        
        start_time = time.time()
        _ = await service.generate_response(
            prompt,
            context={"use_cache": False}
        )
        no_cache_time = time.time() - start_time
        
        # Now try with cache enabled
        start_time = time.time()
        result = await service.generate_response(
            prompt,
            context={"use_cache": True}
        )
        first_cache_time = time.time() - start_time
        results["first_request_time"].append(first_cache_time)
        
        cache_key = service.ai_cache.generate_key(prompt, {"use_cache": True}, "test-model")
        was_in_cache = service.ai_cache.exists(cache_key)
        
        if not was_in_cache:
            results["cache_misses"] += 1
            print(f"Cache miss: '{prompt}' in {first_cache_time:.5f}s")
        
        # Try the same prompt again - should hit cache
        start_time = time.time()
        result = await service.generate_response(
            prompt,
            context={"use_cache": True}
        )
        second_cache_time = time.time() - start_time
        results["cached_request_time"].append(second_cache_time)
        
        if "cache_hit" in result.get("_metadata", {}) and result["_metadata"]["cache_hit"]:
            results["cache_hits"] += 1
            print(f"✓ Cache hit: '{prompt}' in {second_cache_time:.5f}s")
            print(f"  Without cache: {no_cache_time:.5f}s")
            print(f"  Speedup: {no_cache_time / second_cache_time:.2f}x")
        else:
            print(f"× Expected cache hit but missed: '{prompt}'")
        
        print()
    
    # Print summary
    if results["cache_hits"] > 0:
        avg_cache_time = sum(results["cached_request_time"]) / len(results["cached_request_time"])
        avg_first_time = sum(results["first_request_time"]) / len(results["first_request_time"])
        
        print(f"Average first request time: {avg_first_time:.5f}s")
        print(f"Average cached request time: {avg_cache_time:.5f}s")
        print(f"Average cache speedup: {avg_first_time / avg_cache_time:.2f}x")
        print(f"Cache hit rate: {results['cache_hits'] / len(test_cases):.0%}")

async def test_batch_processing():
    """Test the efficiency of batch processing."""
    print("\n=== Testing Batch Processing ===\n")
    
    # Initialize batch processor
    processor = BatchProcessor(
        batch_size=3,
        max_waiting_time=0.1,
        max_concurrent_batches=2
    )
    
    async def mock_process_batch(batch_items):
        """Mock batch processing function."""
        await asyncio.sleep(0.5)  # Simulate processing time
        results = {}
        for item_id, item in batch_items.items():
            results[item_id] = {
                "content": f"Processed: {item['data']}",
                "batch_size": len(batch_items),
                "_metadata": {"processed_at": datetime.now().isoformat()}
            }
        return results
    
    processor.process_batch = mock_process_batch
    await processor.start()
    
    # Test with sequential requests
    sequential_start = time.time()
    
    tasks = []
    for i in range(10):
        request_id, future = await processor.add_request(
            data=f"Request {i}",
            priority=5,
            metadata={}
        )
        tasks.append(future)
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    sequential_time = time.time() - sequential_start
    
    # Calculate metrics
    batch_sizes = [r.get("batch_size", 1) for r in results]
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)
    
    print(f"Processed {len(results)} requests in {sequential_time:.5f}s")
    print(f"Average batch size: {avg_batch_size:.2f}")
    print(f"Effective requests per second: {len(results) / sequential_time:.2f}")
    
    # Clean up
    await processor.stop()

async def test_model_selection():
    """Test the model selection logic."""
    print("\n=== Testing Model Selection ===\n")
    
    # Initialize model selector
    selector = ModelSelector()
    selector.initialize({
        "models": [
            {
                "name": "gpt-4",
                "capabilities": ["function_calling", "image_analysis", "code_generation"],
                "max_tokens": 8192,
                "cost_per_1k_tokens": 0.06,
                "relative_quality": 9,
                "relative_speed": 4
            },
            {
                "name": "gpt-3.5-turbo",
                "capabilities": ["function_calling", "code_generation"],
                "max_tokens": 4096,
                "cost_per_1k_tokens": 0.002,
                "relative_quality": 6,
                "relative_speed": 8
            },
            {
                "name": "text-embedding",
                "capabilities": ["embeddings"],
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.0001,
                "relative_quality": 5,
                "relative_speed": 9
            }
        ]
    })
    
    # Test cases
    test_cases = [
        {
            "name": "Simple question",
            "task_type": "qa",
            "content_length": 100,
            "require_capabilities": [],
            "quality_priority": 5,
            "speed_priority": 8,
            "cost_priority": 9
        },
        {
            "name": "Code generation",
            "task_type": "code_generation",
            "content_length": 500,
            "require_capabilities": ["code_generation"],
            "quality_priority": 8,
            "speed_priority": 6,
            "cost_priority": 5
        },
        {
            "name": "Image analysis",
            "task_type": "image_analysis",
            "content_length": 300,
            "require_capabilities": ["image_analysis"],
            "quality_priority": 9,
            "speed_priority": 5,
            "cost_priority": 3
        },
        {
            "name": "Simple embedding",
            "task_type": "embedding",
            "content_length": 1000,
            "require_capabilities": ["embeddings"],
            "quality_priority": 4,
            "speed_priority": 8,
            "cost_priority": 9
        }
    ]
    
    # Run tests
    for case in test_cases:
        model = selector.select_model(
            task_type=case["task_type"],
            content_length=case["content_length"],
            require_capabilities=case["require_capabilities"],
            quality_priority=case["quality_priority"],
            speed_priority=case["speed_priority"],
            cost_priority=case["cost_priority"]
        )
        
        print(f"Test: {case['name']}")
        print(f"Selected model: {model}")
        print(f"Requirements: {case['require_capabilities']}")
        print(f"Priorities - Quality: {case['quality_priority']}, " +
              f"Speed: {case['speed_priority']}, Cost: {case['cost_priority']}")
        print()

async def run_all_tests():
    """Run all optimization tests."""
    print("\n===== AI SERVICE OPTIMIZATION TESTS =====\n")
    
    print("Running tests to measure the performance improvements")
    print("from various AI service optimization techniques.")
    
    await test_rule_engine_performance()
    await test_caching_performance()
    await test_batch_processing()
    await test_model_selection()
    
    print("\n===== TEST COMPLETE =====\n")

if __name__ == "__main__":
    asyncio.run(run_all_tests())