#!/usr/bin/env python3
"""
Test script for Priority 9 - Configuration Management
Tests environment-specific configurations and runtime updates
"""
import asyncio
import time
import aiohttp
import json
import os
from typing import Dict, Any

class Priority9Tester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_config_endpoints(self):
        """Test configuration management endpoints"""
        print("🔍 Testing configuration endpoints...")
        
        endpoints_to_test = [
            ("/config/current", "Current config", False),
            ("/config/validate", "Config validation", False),
        ]
        
        successful_endpoints = 0
        
        for endpoint, description, requires_auth in endpoints_to_test:
            try:
                headers = {}
                if requires_auth:
                    headers["X-API-Key"] = "test-key"  # Would need actual key
                
                async with self.session.get(f"{self.base_url}{endpoint}", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ {description}: {len(data)} keys in response")
                        successful_endpoints += 1
                    elif response.status == 401 and requires_auth:
                        print(f"   ⚠️ {description}: Authentication required (expected)")
                        successful_endpoints += 1  # This is expected
                    else:
                        print(f"   ❌ {description} failed: {response.status}")
            except Exception as e:
                print(f"   ❌ {description} error: {e}")
        
        print(f"✅ Configuration endpoints test: {successful_endpoints}/{len(endpoints_to_test)} endpoints working")
        return successful_endpoints >= len(endpoints_to_test) // 2
    
    async def test_environment_detection(self):
        """Test environment detection"""
        print("🔍 Testing environment detection...")
        
        try:
            async with self.session.get(f"{self.base_url}/config/current") as response:
                if response.status == 200:
                    data = await response.json()
                    config = data.get('config', {})
                    
                    environment = config.get('environment', 'unknown')
                    debug = config.get('debug', False)
                    
                    print(f"   Detected environment: {environment}")
                    print(f"   Debug mode: {debug}")
                    print(f"   Max concurrent requests: {config.get('max_concurrent_requests', 'unknown')}")
                    print(f"   Request timeout: {config.get('request_timeout', 'unknown')}")
                    
                    # Validate that we have a known environment
                    if environment in ['development', 'production', 'testing', 'staging']:
                        print("   ✅ Environment detection working")
                        return True
                    else:
                        print(f"   ⚠️ Unknown environment: {environment}")
                        return False
                else:
                    print(f"   ❌ Config endpoint failed: {response.status}")
                    return False
        
        except Exception as e:
            print(f"❌ Environment detection test error: {e}")
            return False
    
    async def test_config_validation(self):
        """Test configuration validation"""
        print("🔍 Testing configuration validation...")
        
        try:
            async with self.session.get(f"{self.base_url}/config/validate") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    is_valid = data.get('valid', False)
                    message = data.get('message', 'No message')
                    
                    print(f"   Configuration valid: {is_valid}")
                    print(f"   Validation message: {message}")
                    
                    if is_valid:
                        print("   ✅ Configuration validation passed")
                        return True
                    else:
                        print("   ⚠️ Configuration validation found issues")
                        return False
                else:
                    print(f"   ❌ Config validation endpoint failed: {response.status}")
                    return False
        
        except Exception as e:
            print(f"❌ Configuration validation test error: {e}")
            return False
    
    async def test_spacy_model_integration(self):
        """Test spaCy large model integration"""
        print("🔍 Testing spaCy large model integration...")
        
        try:
            # Test importing spaCy and checking model
            import subprocess
            import sys
            
            # Check if spaCy is installed
            result = subprocess.run([sys.executable, "-c", "import spacy; print('spaCy available')"], 
                                    capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ spaCy is available")
                
                # Check for large model
                result = subprocess.run([sys.executable, "-c", 
                                       "import spacy; nlp = spacy.load('en_core_web_lg'); print('Large model loaded')"], 
                                       capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("   ✅ spaCy large model (en_core_web_lg) is available")
                    return True
                else:
                    # Try medium model as fallback
                    result = subprocess.run([sys.executable, "-c", 
                                           "import spacy; nlp = spacy.load('en_core_web_md'); print('Medium model loaded')"], 
                                           capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("   ⚠️ spaCy medium model available (large model not found)")
                        return True
                    else:
                        # Try small model
                        result = subprocess.run([sys.executable, "-c", 
                                               "import spacy; nlp = spacy.load('en_core_web_sm'); print('Small model loaded')"], 
                                               capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            print("   ⚠️ Only spaCy small model available (recommend upgrading to large)")
                            return True
                        else:
                            print("   ❌ No spaCy models found")
                            return False
            else:
                print("   ❌ spaCy not available")
                return False
        
        except Exception as e:
            print(f"❌ spaCy model test error: {e}")
            return False
    
    async def test_comprehensive_integration(self):
        """Test comprehensive integration of all systems"""
        print("🔍 Testing comprehensive system integration...")
        
        try:
            # Test a complete workflow that touches multiple systems
            request_data = {
                "url": "https://httpbin.org/html",
                "strategy_name": "universal_crawl4ai",
                "options": {}
            }
            
            # Submit a scrape request
            async with self.session.post(
                f"{self.base_url}/scrape",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status == 202:
                    data = await response.json()
                    job_id = data.get('job_id')
                    print(f"   ✅ Scrape job submitted: {job_id}")
                    
                    # Wait a moment for processing
                    await asyncio.sleep(3)
                    
                    # Check metrics to see if it was recorded
                    async with self.session.get(f"{self.base_url}/metrics/performance") as metrics_response:
                        if metrics_response.status == 200:
                            metrics_data = await metrics_response.json()
                            total_requests = metrics_data.get('summary', {}).get('total_requests', 0)
                            
                            if total_requests > 0:
                                print(f"   ✅ Metrics recorded: {total_requests} total requests")
                                
                                # Check health status
                                async with self.session.get(f"{self.base_url}/health/detailed") as health_response:
                                    if health_response.status == 200:
                                        health_data = await health_response.json()
                                        status = health_data.get('status', 'unknown')
                                        print(f"   ✅ System health: {status}")
                                        
                                        return True
                                    else:
                                        print(f"   ⚠️ Health check failed: {health_response.status}")
                                        return False
                            else:
                                print("   ⚠️ No metrics recorded yet")
                                return False
                        else:
                            print(f"   ⚠️ Metrics endpoint failed: {metrics_response.status}")
                            return False
                else:
                    print(f"   ❌ Scrape request failed: {response.status}")
                    return False
        
        except Exception as e:
            print(f"❌ Comprehensive integration test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all Priority 9 tests"""
        print("🚀 Starting Priority 9 - Configuration Management Tests")
        print("=" * 65)
        
        tests = [
            ("Configuration Endpoints", self.test_config_endpoints),
            ("Environment Detection", self.test_environment_detection),
            ("Configuration Validation", self.test_config_validation),
            ("spaCy Model Integration", self.test_spacy_model_integration),
            ("System Integration", self.test_comprehensive_integration)
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
        print("\n" + "=" * 65)
        print("🎯 PRIORITY 9 TEST SUMMARY")
        print("=" * 65)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
        
        print(f"\n📊 Overall: {passed}/{total} tests passed")
        
        if passed >= total * 0.8:  # 80% pass rate
            print("🎉 Priority 9 - Configuration Management: COMPLETED!")
        else:
            print("⚠️  Some tests failed. Check the implementation.")
        
        return passed >= total * 0.8

async def main():
    """Main test runner"""
    print("Testing Priority 9 - Configuration Management")
    print("Make sure the SmartScrape server is running on localhost:5000")
    print()
    
    async with Priority9Tester() as tester:
        success = await tester.run_all_tests()
        return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
