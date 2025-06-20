#!/usr/bin/env python3
"""
Test Performance Optimizations for SmartScrape

This script tests memory monitoring and HTTP client optimization features.
"""

import asyncio
import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.memory_monitor import memory_monitor, MemoryMonitor
from utils.http_client import OptimizedHTTPClient, get_http_client
from controllers.extraction_coordinator import ExtractionCoordinator

async def test_memory_monitoring():
    """Test memory monitoring functionality"""
    print("🧠 Testing memory monitoring...")
    
    try:
        # Test memory usage reporting
        usage = memory_monitor.get_memory_usage()
        print(f"  📊 Current memory usage: {usage['rss_mb']:.1f}MB ({usage['percent']:.1f}%)")
        print(f"  🖥️  System memory usage: {usage['system_percent']:.1f}%")
        
        # Test memory summary
        summary = memory_monitor.get_memory_summary()
        print(f"  📈 Memory status: {summary['status']}")
        print(f"  📏 Usage: {summary['current_usage_mb']:.1f}MB / {summary['max_limit_mb']}MB ({summary['usage_percentage']:.1f}%)")
        
        # Test cleanup (force it to test the functionality)
        cleanup_result = memory_monitor.cleanup(force=True)
        if cleanup_result['performed']:
            print(f"  🧹 Cleanup performed: {cleanup_result['objects_collected']} objects collected")
            if cleanup_result['memory_freed_mb'] > 0:
                print(f"  💾 Memory freed: {cleanup_result['memory_freed_mb']:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory monitoring test failed: {e}")
        return False

async def test_http_client_optimization():
    """Test HTTP client optimization"""
    print("🌐 Testing HTTP client optimization...")
    
    try:
        # Test basic HTTP client functionality
        async with OptimizedHTTPClient() as client:
            # Test connection stats
            stats = client.get_connection_stats()
            print(f"  📊 Connection pool initialized: limit={stats.get('connector_limit', 'N/A')}")
            
            # Test fetch with retry
            test_url = "https://httpbin.org/json"
            result = await client.fetch_with_retry(test_url)
            
            if result['success']:
                print(f"  ✅ HTTP fetch successful: {result['status_code']} in {result['response_time']:.2f}s")
                print(f"  📏 Content length: {result['content_length']} bytes")
            else:
                print(f"  ❌ HTTP fetch failed: {result['error']}")
                return False
            
            # Test global client context manager
            async with get_http_client() as global_client:
                result2 = await global_client.fetch_with_retry("https://httpbin.org/headers")
                if result2['success']:
                    print(f"  ✅ Global client test successful: {result2['status_code']}")
                else:
                    print(f"  ❌ Global client test failed: {result2['error']}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ HTTP client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integrated_performance():
    """Test integrated performance with extraction coordinator"""
    print("🚀 Testing integrated performance optimization...")
    
    try:
        # Initialize coordinator (this should trigger memory monitoring)
        coordinator = ExtractionCoordinator()
        
        # Test extraction with memory monitoring
        test_url = "https://httpbin.org/html"
        print(f"  🔍 Testing extraction for: {test_url}")
        
        # Get memory before
        before_memory = memory_monitor.get_memory_usage()
        
        # Perform extraction (should trigger memory monitoring decorators)
        result = await coordinator.extract_with_intelligent_selection(test_url)
        
        # Get memory after
        after_memory = memory_monitor.get_memory_usage()
        
        # Check extraction success
        if result.get('success'):
            print(f"  ✅ Extraction successful with strategy: {result.get('strategy', 'unknown')}")
            print(f"  📏 Content length: {len(result.get('content', ''))}")
        else:
            print(f"  ❌ Extraction failed: {result.get('error', 'unknown')}")
            return False
        
        # Show memory impact
        memory_diff = after_memory['rss_mb'] - before_memory['rss_mb']
        print(f"  🧠 Memory impact: {memory_diff:+.1f}MB")
        
        # Test automatic cleanup
        cleanup_result = memory_monitor.auto_cleanup_if_needed()
        if cleanup_result:
            objects_collected = cleanup_result.get('objects_collected', 0)
            print(f"  🧹 Automatic cleanup triggered: {objects_collected} objects")
        else:
            print(f"  ✅ No cleanup needed, memory usage normal")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integrated performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_stress():
    """Test performance under stress conditions"""
    print("💪 Testing performance under stress...")
    
    try:
        urls = [
            "https://httpbin.org/html",
            "https://httpbin.org/json", 
            "https://httpbin.org/xml",
            "https://httpbin.org/headers",
            "https://httpbin.org/delay/1"
        ]
        
        coordinator = ExtractionCoordinator()
        
        start_time = time.time()
        successful_extractions = 0
        
        # Run multiple extractions
        for i, url in enumerate(urls):
            print(f"  🔄 Extraction {i+1}/{len(urls)}: {url}")
            
            try:
                result = await coordinator.extract_with_fallbacks(url)
                if result.get('success'):
                    successful_extractions += 1
                    print(f"    ✅ Success")
                else:
                    print(f"    ❌ Failed: {result.get('error', 'unknown')}")
            except Exception as e:
                print(f"    ❌ Exception: {e}")
        
        total_time = time.time() - start_time
        success_rate = (successful_extractions / len(urls)) * 100
        
        print(f"  📊 Stress test results:")
        print(f"    ✅ Successful: {successful_extractions}/{len(urls)} ({success_rate:.1f}%)")
        print(f"    ⏱️ Total time: {total_time:.2f}s")
        print(f"    🚀 Average per extraction: {total_time/len(urls):.2f}s")
        
        # Final memory check
        final_memory = memory_monitor.get_memory_summary()
        print(f"    🧠 Final memory status: {final_memory['status']} ({final_memory['current_usage_mb']:.1f}MB)")
        
        return successful_extractions > 0
        
    except Exception as e:
        print(f"  ❌ Stress test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("Starting performance optimization tests...\n")
    
    tests = [
        ("Memory Monitoring", test_memory_monitoring),
        ("HTTP Client Optimization", test_http_client_optimization),
        ("Integrated Performance", test_integrated_performance),
        ("Performance Stress Test", test_performance_stress)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name}...")
        try:
            success = await test_func()
            if success:
                print(f"✅ {test_name} PASSED\n")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED\n")
                failed += 1
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}\n")
            failed += 1
    
    print("📋 Performance Test Summary:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All performance optimization tests passed!")
    else:
        print(f"\n💥 {failed} performance optimization tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
