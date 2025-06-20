#!/usr/bin/env python3
"""
Test script for Priority 7 - API Performance Enhancement
Tests rate limiting and streaming functionality
"""
import asyncio
import time
import aiohttp
import json
from typing import Dict, Any

class Priority7Tester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_rate_limit_status(self):
        """Test rate limit status endpoint"""
        print("🔍 Testing rate limit status endpoint...")
        
        try:
            async with self.session.get(f"{self.base_url}/rate-limit/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Rate limit status: {data['message']}")
                    print(f"   Available limits: {list(data['available_limits'].keys())}")
                    return True
                else:
                    print(f"❌ Rate limit status failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Rate limit status error: {e}")
            return False
    
    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        print("🔍 Testing rate limiting...")
        
        # Test scrape endpoint rate limiting
        try:
            # Make multiple requests quickly to test rate limiting
            request_data = {
                "url": "https://httpbin.org/html",
                "strategy_name": "universal_crawl4ai",
                "options": {}
            }
            
            success_count = 0
            rate_limited_count = 0
            
            for i in range(25):  # Try 25 requests (should hit rate limit)
                try:
                    async with self.session.post(
                        f"{self.base_url}/scrape",
                        json=request_data,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 202:
                            success_count += 1
                        elif response.status == 429:
                            rate_limited_count += 1
                            print(f"   Rate limited on request {i+1}")
                            break
                        else:
                            print(f"   Unexpected status {response.status} on request {i+1}")
                except Exception as e:
                    print(f"   Request {i+1} failed: {e}")
            
            print(f"✅ Rate limiting test: {success_count} successful, {rate_limited_count} rate limited")
            return rate_limited_count > 0  # Success if we hit rate limit
            
        except Exception as e:
            print(f"❌ Rate limiting test error: {e}")
            return False
    
    async def test_streaming_endpoint(self):
        """Test streaming endpoint functionality"""
        print("🔍 Testing streaming endpoint...")
        
        try:
            request_data = {
                "urls": [
                    "https://httpbin.org/html",
                    "https://example.com"
                ],
                "strategy": "universal_crawl4ai",
                "options": {}
            }
            
            async with self.session.post(
                f"{self.base_url}/extract/stream",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    stream_events = []
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = json.loads(line[6:])
                            stream_events.append(data)
                            print(f"   Stream event: {data.get('status', 'unknown')}")
                    
                    print(f"✅ Streaming test: received {len(stream_events)} events")
                    return len(stream_events) > 0
                else:
                    print(f"❌ Streaming test failed: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Streaming test error: {e}")
            return False
    
    async def test_batch_endpoint(self):
        """Test batch endpoint functionality"""
        print("🔍 Testing batch endpoint...")
        
        try:
            request_data = {
                "urls": [
                    "https://httpbin.org/html",
                    "https://example.com"
                ],
                "strategy": "universal_crawl4ai"
            }
            
            async with self.session.post(
                f"{self.base_url}/extract/batch-progress",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 202:
                    data = await response.json()
                    print(f"✅ Batch endpoint: task created with ID {data.get('task_id')}")
                    return True
                else:
                    print(f"❌ Batch endpoint failed: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Batch endpoint error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Priority 7 tests"""
        print("🚀 Starting Priority 7 - API Performance Enhancement Tests")
        print("=" * 60)
        
        tests = [
            ("Rate Limit Status", self.test_rate_limit_status),
            ("Rate Limiting", self.test_rate_limiting),
            ("Streaming Endpoint", self.test_streaming_endpoint),
            ("Batch Endpoint", self.test_batch_endpoint)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name} test...")
            try:
                result = await test_func()
                results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {status}")
            except Exception as e:
                results[test_name] = False
                print(f"   ❌ FAILED: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("🎯 PRIORITY 7 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 Priority 7 - API Performance Enhancement: COMPLETED!")
        else:
            print("⚠️  Some tests failed. Check the implementation.")
        
        return passed == total

async def main():
    """Main test runner"""
    print("Testing Priority 7 - API Performance Enhancement")
    print("Make sure the SmartScrape server is running on localhost:5000")
    print()
    
    async with Priority7Tester() as tester:
        success = await tester.run_all_tests()
        return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
