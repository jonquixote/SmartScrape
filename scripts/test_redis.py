#!/usr/bin/env python3
"""
Test Redis configuration and caching functionality
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from controllers.extraction_coordinator import ExtractionCoordinator
from config import REDIS_CONFIG, CACHE_TTL
import json

async def test_redis_functionality():
    """Test Redis caching functionality"""
    print("Testing Redis functionality...")
    
    try:
        # Initialize coordinator
        coordinator = ExtractionCoordinator()
        
        if not coordinator.redis_client:
            print("❌ Redis client not initialized")
            return False
            
        print("✅ Redis client initialized successfully")
        
        # Test basic Redis operations
        test_key = "test:smartscrape:functionality"
        test_data = {"message": "Hello Redis!", "timestamp": "2024-test"}
        
        # Test setting data
        coordinator.redis_client.setex(
            test_key, 
            60,  # 1 minute TTL
            json.dumps(test_data)
        )
        print("✅ Successfully stored test data in Redis")
        
        # Test getting data
        retrieved = coordinator.redis_client.get(test_key)
        if retrieved:
            parsed_data = json.loads(retrieved)
            print(f"✅ Successfully retrieved test data: {parsed_data}")
        else:
            print("❌ Failed to retrieve test data")
            return False
            
        # Test caching methods
        test_url = "https://httpbin.org/json"
        test_strategy = "test_strategy"
        test_result = {
            "content": "Test content",
            "success": True,
            "metadata": {"test": True}
        }
        
        # Test cache_content method
        await coordinator.cache_content(test_url, test_strategy, test_result)
        print("✅ Successfully cached content using cache_content method")
        
        # Test get_cached_content method
        cached_result = await coordinator.get_cached_content(test_url, test_strategy)
        if cached_result:
            print(f"✅ Successfully retrieved cached content: {cached_result.get('success')}")
        else:
            print("❌ Failed to retrieve cached content")
            return False
            
        # Test Redis info
        info = coordinator.redis_client.info()
        print(f"✅ Redis version: {info.get('redis_version', 'unknown')}")
        print(f"✅ Redis uptime: {info.get('uptime_in_seconds', 0)} seconds")
        print(f"✅ Connected clients: {info.get('connected_clients', 0)}")
        
        # Cleanup test data
        coordinator.redis_client.delete(test_key)
        
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("Starting Redis functionality test...")
    print(f"Redis config: {REDIS_CONFIG}")
    print(f"Cache TTL config: {CACHE_TTL}")
    
    success = await test_redis_functionality()
    
    if success:
        print("\n🎉 All Redis tests passed!")
    else:
        print("\n💥 Redis tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
