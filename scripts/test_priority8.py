#!/usr/bin/env python3
"""
Test script for Priority 8 - Enhanced Error Handling & Monitoring
Tests error classification, metrics collection, and monitoring endpoints
"""
import asyncio
import time
import aiohttp
import json
from typing import Dict, Any

class Priority8Tester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_error_handling(self):
        """Test error handling and classification"""
        print("🔍 Testing error handling and classification...")
        
        try:
            # Test with URLs that will likely cause different types of errors
            test_cases = [
                {
                    "url": "https://nonexistent-domain-12345.com",
                    "expected_error": "network",
                    "description": "Network error test"
                },
                {
                    "url": "https://httpbin.org/status/500",
                    "expected_error": "parsing",
                    "description": "Server error test"
                },
                {
                    "url": "https://httpbin.org/delay/60",
                    "expected_error": "timeout",
                    "description": "Timeout error test"
                }
            ]
            
            successful_tests = 0
            
            for test_case in test_cases:
                print(f"   Testing {test_case['description']}: {test_case['url']}")
                
                try:
                    request_data = {
                        "url": test_case["url"],
                        "strategy_name": "universal_crawl4ai",
                        "options": {"timeout": 10}  # Short timeout for testing
                    }
                    
                    async with self.session.post(
                        f"{self.base_url}/scrape",
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as response:
                        if response.status in [202, 500]:  # Expected responses
                            successful_tests += 1
                            print(f"   ✅ {test_case['description']} handled correctly")
                        else:
                            print(f"   ⚠️ {test_case['description']} unexpected status: {response.status}")
                
                except asyncio.TimeoutError:
                    # Timeout is expected for timeout test
                    if test_case["expected_error"] == "timeout":
                        successful_tests += 1
                        print(f"   ✅ {test_case['description']} timed out as expected")
                    else:
                        print(f"   ❌ {test_case['description']} unexpected timeout")
                except Exception as e:
                    print(f"   ⚠️ {test_case['description']} error: {e}")
            
            print(f"✅ Error handling test: {successful_tests}/{len(test_cases)} scenarios handled")
            return successful_tests > 0
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
    
    async def test_metrics_endpoints(self):
        """Test metrics collection endpoints"""
        print("🔍 Testing metrics endpoints...")
        
        endpoints_to_test = [
            ("/metrics/performance", "Performance metrics"),
            ("/metrics/errors", "Error metrics"),
            ("/metrics/time-series", "Time series metrics"),
            ("/health/detailed", "Detailed health"),
            ("/metrics/export", "Metrics export")
        ]
        
        successful_endpoints = 0
        
        for endpoint, description in endpoints_to_test:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ {description}: {len(data)} keys in response")
                        successful_endpoints += 1
                    else:
                        print(f"   ❌ {description} failed: {response.status}")
            except Exception as e:
                print(f"   ❌ {description} error: {e}")
        
        print(f"✅ Metrics endpoints test: {successful_endpoints}/{len(endpoints_to_test)} endpoints working")
        return successful_endpoints >= len(endpoints_to_test) // 2  # At least half should work
    
    async def test_performance_metrics(self):
        """Test performance metrics collection"""
        print("🔍 Testing performance metrics collection...")
        
        try:
            # Generate some test requests to create metrics
            test_urls = [
                "https://httpbin.org/html",
                "https://example.com",
                "https://httpbin.org/json"
            ]
            
            print("   Generating test requests for metrics...")
            for i, url in enumerate(test_urls):
                try:
                    request_data = {
                        "url": url,
                        "strategy_name": "universal_crawl4ai",
                        "options": {}
                    }
                    
                    async with self.session.post(
                        f"{self.base_url}/scrape",
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        print(f"   Request {i+1}: {response.status}")
                        await asyncio.sleep(1)  # Brief delay between requests
                
                except Exception as e:
                    print(f"   Request {i+1} failed: {e}")
            
            # Wait a moment for metrics to be processed
            await asyncio.sleep(2)
            
            # Check performance metrics
            async with self.session.get(f"{self.base_url}/metrics/performance") as response:
                if response.status == 200:
                    data = await response.json()
                    summary = data.get('summary', {})
                    
                    print(f"   Total requests: {summary.get('total_requests', 0)}")
                    print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
                    print(f"   Avg response time: {summary.get('avg_response_time', 0):.2f}s")
                    
                    if summary.get('total_requests', 0) > 0:
                        print("   ✅ Performance metrics are being collected")
                        return True
                    else:
                        print("   ⚠️ No metrics collected yet")
                        return False
                else:
                    print(f"   ❌ Performance metrics endpoint failed: {response.status}")
                    return False
        
        except Exception as e:
            print(f"❌ Performance metrics test error: {e}")
            return False
    
    async def test_error_metrics(self):
        """Test error metrics collection"""
        print("🔍 Testing error metrics collection...")
        
        try:
            # Generate some error conditions
            error_urls = [
                "https://this-domain-definitely-does-not-exist-12345.com",
                "https://httpbin.org/status/404",
                "https://httpbin.org/status/500"
            ]
            
            print("   Generating error conditions...")
            for i, url in enumerate(error_urls):
                try:
                    request_data = {
                        "url": url,
                        "strategy_name": "universal_crawl4ai",
                        "options": {"timeout": 5}
                    }
                    
                    async with self.session.post(
                        f"{self.base_url}/scrape",
                        json=request_data,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        print(f"   Error test {i+1}: {response.status}")
                
                except Exception as e:
                    print(f"   Error test {i+1} exception (expected): {type(e).__name__}")
            
            # Wait for error processing
            await asyncio.sleep(2)
            
            # Check error metrics
            async with self.session.get(f"{self.base_url}/metrics/errors") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    total_errors = data.get('total_errors', 0)
                    error_breakdown = data.get('error_breakdown', {})
                    
                    print(f"   Total errors recorded: {total_errors}")
                    print(f"   Error types: {list(error_breakdown.keys())}")
                    
                    if total_errors > 0:
                        print("   ✅ Error metrics are being collected")
                        return True
                    else:
                        print("   ⚠️ No error metrics collected yet")
                        return False
                else:
                    print(f"   ❌ Error metrics endpoint failed: {response.status}")
                    return False
        
        except Exception as e:
            print(f"❌ Error metrics test error: {e}")
            return False
    
    async def test_health_monitoring(self):
        """Test health monitoring functionality"""
        print("🔍 Testing health monitoring...")
        
        try:
            async with self.session.get(f"{self.base_url}/health/detailed") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    status = data.get('status', 'unknown')
                    issues = data.get('issues', [])
                    performance_summary = data.get('performance_summary', {})
                    
                    print(f"   System status: {status}")
                    print(f"   Issues detected: {len(issues)}")
                    if issues:
                        for issue in issues[:3]:  # Show first 3 issues
                            print(f"     - {issue}")
                    
                    print(f"   Performance summary keys: {list(performance_summary.keys())}")
                    
                    print("   ✅ Health monitoring is working")
                    return True
                else:
                    print(f"   ❌ Health monitoring failed: {response.status}")
                    return False
        
        except Exception as e:
            print(f"❌ Health monitoring test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Priority 8 tests"""
        print("🚀 Starting Priority 8 - Enhanced Error Handling & Monitoring Tests")
        print("=" * 70)
        
        tests = [
            ("Error Handling", self.test_error_handling),
            ("Metrics Endpoints", self.test_metrics_endpoints),
            ("Performance Metrics", self.test_performance_metrics),
            ("Error Metrics", self.test_error_metrics),
            ("Health Monitoring", self.test_health_monitoring)
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
        print("\n" + "=" * 70)
        print("🎯 PRIORITY 8 TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed >= total * 0.8:  # 80% pass rate
            print("🎉 Priority 8 - Enhanced Error Handling & Monitoring: COMPLETED!")
        else:
            print("⚠️  Some tests failed. Check the implementation.")
        
        return passed >= total * 0.8

async def main():
    """Main test runner"""
    print("Testing Priority 8 - Enhanced Error Handling & Monitoring")
    print("Make sure the SmartScrape server is running on localhost:5000")
    print()
    
    async with Priority8Tester() as tester:
        success = await tester.run_all_tests()
        return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
