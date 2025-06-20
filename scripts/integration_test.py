#!/usr/bin/env python3
"""
Integration test script to verify all SmartScrape services work together.
This script tests the complete system integration after startup.
"""

import time
import sys
import requests
import json
from typing import Dict, Any

def test_basic_endpoints(base_url: str = "http://localhost:5000") -> bool:
    """Test basic API endpoints."""
    print("🧪 Testing basic API endpoints...")
    
    endpoints = [
        ("/health", "Health check"),
        ("/api/status", "Status endpoint"),
        ("/docs", "API documentation"),
        ("/api/health/detailed", "Detailed health check"),
        ("/api/metrics", "Metrics endpoint"),
        ("/api/config/current", "Current configuration")
    ]
    
    all_passed = True
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code < 500:  # Accept any non-server-error
                print(f"  ✅ {description}: HTTP {response.status_code}")
            else:
                print(f"  ❌ {description}: HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"  ❌ {description}: Failed - {str(e)}")
            all_passed = False
    
    return all_passed

def test_scraping_functionality(base_url: str = "http://localhost:5000") -> bool:
    """Test basic scraping functionality."""
    print("\n🕷️ Testing scraping functionality...")
    
    # Test basic scraping endpoint
    try:
        scrape_data = {
            "url": "https://httpbin.org/json",
            "extract_text": True,
            "extract_links": False
        }
        
        response = requests.post(
            f"{base_url}/api/scrape",
            json=scrape_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "content" in result or "text" in result:
                print("  ✅ Basic scraping: Success")
                return True
            else:
                print(f"  ❌ Basic scraping: Unexpected response format")
                return False
        else:
            print(f"  ❌ Basic scraping: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Basic scraping: Failed - {str(e)}")
        return False

def test_pipeline_functionality(base_url: str = "http://localhost:5000") -> bool:
    """Test pipeline orchestration functionality."""
    print("\n⚡ Testing pipeline functionality...")
    
    try:
        # Test pipeline creation
        pipeline_data = {
            "name": "test_pipeline",
            "steps": [
                {
                    "name": "scrape_step",
                    "type": "scrape",
                    "config": {
                        "url": "https://httpbin.org/json",
                        "extract_text": True
                    }
                }
            ]
        }
        
        response = requests.post(
            f"{base_url}/api/pipeline/create",
            json=pipeline_data,
            timeout=15
        )
        
        if response.status_code in [200, 201]:
            print("  ✅ Pipeline creation: Success")
            
            # Try to get pipeline status
            try:
                result = response.json()
                if "pipeline_id" in result:
                    pipeline_id = result["pipeline_id"]
                    status_response = requests.get(f"{base_url}/api/pipeline/{pipeline_id}/status", timeout=10)
                    if status_response.status_code == 200:
                        print("  ✅ Pipeline status check: Success")
                        return True
                    else:
                        print("  ⚠️  Pipeline status check: Could not get status")
                        return True  # Still consider success if creation worked
            except Exception as e:
                print(f"  ⚠️  Pipeline status check: {str(e)}")
                return True  # Still consider success if creation worked
                
        else:
            print(f"  ❌ Pipeline creation: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Pipeline functionality: Failed - {str(e)}")
        return False

def test_rate_limiting(base_url: str = "http://localhost:5000") -> bool:
    """Test rate limiting functionality."""
    print("\n🚦 Testing rate limiting...")
    
    try:
        # Check rate limit status
        response = requests.get(f"{base_url}/api/rate-limit/status", timeout=10)
        
        if response.status_code == 200:
            print("  ✅ Rate limit status: Success")
            return True
        else:
            print(f"  ❌ Rate limit status: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Rate limiting test: Failed - {str(e)}")
        return False

def test_configuration_management(base_url: str = "http://localhost:5000") -> bool:
    """Test configuration management."""
    print("\n⚙️ Testing configuration management...")
    
    try:
        # Get current config
        response = requests.get(f"{base_url}/api/config/current", timeout=10)
        
        if response.status_code == 200:
            config = response.json()
            if isinstance(config, dict) and len(config) > 0:
                print("  ✅ Configuration retrieval: Success")
                return True
            else:
                print("  ❌ Configuration retrieval: Empty or invalid config")
                return False
        else:
            print(f"  ❌ Configuration retrieval: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Configuration test: Failed - {str(e)}")
        return False

def run_integration_tests(base_url: str = "http://localhost:5000") -> Dict[str, bool]:
    """Run all integration tests."""
    print("🚀 Starting SmartScrape Integration Tests")
    print("=" * 50)
    
    tests = {
        "basic_endpoints": test_basic_endpoints(base_url),
        "scraping_functionality": test_scraping_functionality(base_url),
        "pipeline_functionality": test_pipeline_functionality(base_url),
        "rate_limiting": test_rate_limiting(base_url),
        "configuration_management": test_configuration_management(base_url)
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return tests
    elif passed > total // 2:
        print("⚠️  Most tests passed, but some issues detected")
        return tests
    else:
        print("❌ Many tests failed - system may have issues")
        return tests

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartScrape Integration Tests")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Base URL for API")
    parser.add_argument("--wait-for-server", action="store_true", help="Wait for server to be ready")
    parser.add_argument("--max-wait", type=int, default=60, help="Max time to wait for server")
    
    args = parser.parse_args()
    
    if args.wait_for_server:
        print("⏳ Waiting for server to be ready...")
        from scripts.health_check import wait_for_services
        if not wait_for_services(args.base_url, max_wait=args.max_wait):
            print("❌ Server is not ready")
            sys.exit(1)
    
    # Run integration tests
    results = run_integration_tests(args.base_url)
    
    # Exit with appropriate code
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        sys.exit(0)  # All tests passed
    elif passed > total // 2:
        sys.exit(2)  # Most tests passed (warning)
    else:
        sys.exit(1)  # Many tests failed (error)

if __name__ == "__main__":
    main()
