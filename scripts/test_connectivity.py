#!/usr/bin/env python3
"""
Server connectivity test script for SmartScrape.
Tests if all required services are running and accessible.
"""

import asyncio
import json
import sys
import time
import requests
import redis
from celery import Celery
from typing import Dict, List, Tuple, Optional

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message: str, status: str = "INFO"):
    """Print colored status message"""
    colors = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "HEADER": Colors.CYAN + Colors.BOLD
    }
    color = colors.get(status, Colors.BLUE)
    print(f"{color}[{status}]{Colors.END} {message}")

def test_redis_connection(host: str = "localhost", port: int = 6379) -> Tuple[bool, str]:
    """Test Redis connection"""
    try:
        r = redis.Redis(host=host, port=port, decode_responses=True)
        r.ping()
        info = r.info()
        version = info.get('redis_version', 'unknown')
        return True, f"Redis {version} connected successfully"
    except Exception as e:
        return False, f"Redis connection failed: {str(e)}"

def test_celery_connection(broker_url: str = "redis://localhost:6379/0") -> Tuple[bool, str]:
    """Test Celery connection"""
    try:
        app = Celery('test', broker=broker_url)
        
        # Test broker connection
        with app.connection() as conn:
            conn.ensure_connection(max_retries=3)
        
        # Check for active workers
        inspect = app.control.inspect()
        active_workers = inspect.active()
        
        if active_workers:
            worker_count = len(active_workers)
            return True, f"Celery broker connected, {worker_count} active worker(s)"
        else:
            return False, "Celery broker connected but no active workers found"
            
    except Exception as e:
        return False, f"Celery connection failed: {str(e)}"

def test_http_endpoint(url: str, timeout: int = 10) -> Tuple[bool, str, Optional[Dict]]:
    """Test HTTP endpoint"""
    try:
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            try:
                data = response.json()
                return True, f"HTTP {response.status_code} - Response received", data
            except json.JSONDecodeError:
                return True, f"HTTP {response.status_code} - Non-JSON response", None
        else:
            return False, f"HTTP {response.status_code} - {response.reason}", None
            
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - server not running", None
    except requests.exceptions.Timeout:
        return False, f"Request timeout after {timeout}s", None
    except Exception as e:
        return False, f"HTTP request failed: {str(e)}", None

def test_smartscrape_api(base_url: str) -> List[Tuple[str, bool, str]]:
    """Test SmartScrape API endpoints"""
    endpoints = [
        ("Health Check", f"{base_url}/health"),
        ("Root Endpoint", f"{base_url}/"),
        ("API Info", f"{base_url}/api/"),
        ("Pipeline Status", f"{base_url}/api/pipeline/status"),
        ("Metrics", f"{base_url}/api/monitoring/metrics"),
        ("Configuration", f"{base_url}/api/config/current"),
    ]
    
    results = []
    for name, url in endpoints:
        success, message, data = test_http_endpoint(url)
        results.append((name, success, message))
        
        # Print additional info for some endpoints
        if success and data:
            if "health" in url and isinstance(data, dict):
                if "services" in data:
                    services = data["services"]
                    print(f"    Services: {services}")
            elif "metrics" in url and isinstance(data, dict):
                if "system" in data:
                    uptime = data.get("system", {}).get("uptime", "unknown")
                    print(f"    System uptime: {uptime}")
    
    return results

def run_comprehensive_test(host: str = "localhost", port: int = 5000) -> bool:
    """Run comprehensive connectivity test"""
    print_status("SmartScrape Connectivity Test", "HEADER")
    print()
    
    all_passed = True
    base_url = f"http://{host}:{port}"
    
    # Test 1: Redis
    print_status("Testing Redis connection...", "INFO")
    redis_success, redis_message = test_redis_connection()
    if redis_success:
        print_status(redis_message, "SUCCESS")
    else:
        print_status(redis_message, "ERROR")
        all_passed = False
    print()
    
    # Test 2: Celery
    print_status("Testing Celery connection...", "INFO")
    celery_success, celery_message = test_celery_connection()
    if celery_success:
        print_status(celery_message, "SUCCESS")
    else:
        print_status(celery_message, "WARNING")
        # Celery workers might not be critical for basic functionality
    print()
    
    # Test 3: SmartScrape Server
    print_status("Testing SmartScrape server...", "INFO")
    api_results = test_smartscrape_api(base_url)
    
    critical_endpoints = ["Health Check", "Root Endpoint"]
    for name, success, message in api_results:
        status = "SUCCESS" if success else "ERROR"
        print_status(f"{name}: {message}", status)
        
        if not success and name in critical_endpoints:
            all_passed = False
    print()
    
    # Test 4: Integration Test
    print_status("Running integration test...", "INFO")
    try:
        # Test a basic scraping operation
        test_data = {
            "url": "https://httpbin.org/json",
            "strategy": "basic"
        }
        
        response = requests.post(f"{base_url}/api/scrape", json=test_data, timeout=30)
        if response.status_code == 200:
            print_status("Integration test passed - scraping endpoint working", "SUCCESS")
        else:
            print_status(f"Integration test failed - HTTP {response.status_code}", "WARNING")
            
    except Exception as e:
        print_status(f"Integration test failed: {str(e)}", "WARNING")
    print()
    
    # Summary
    if all_passed:
        print_status("All critical services are running and accessible!", "SUCCESS")
        print_status(f"SmartScrape is ready at {base_url}", "SUCCESS")
        return True
    else:
        print_status("Some critical services are not accessible", "ERROR")
        print_status("Please check the service logs and ensure all components are started", "ERROR")
        return False

def main():
    """Main function with command line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SmartScrape service connectivity")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--redis-host", default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument("--wait", type=int, default=0, help="Wait time in seconds before testing")
    parser.add_argument("--retry", type=int, default=1, help="Number of retry attempts")
    parser.add_argument("--retry-delay", type=int, default=5, help="Delay between retries in seconds")
    
    args = parser.parse_args()
    
    # Wait if requested
    if args.wait > 0:
        print_status(f"Waiting {args.wait} seconds before testing...", "INFO")
        time.sleep(args.wait)
    
    # Run tests with retries
    for attempt in range(args.retry):
        if attempt > 0:
            print_status(f"Retry attempt {attempt + 1}/{args.retry}", "INFO")
            time.sleep(args.retry_delay)
        
        success = run_comprehensive_test(args.host, args.port)
        
        if success:
            sys.exit(0)
        elif attempt < args.retry - 1:
            print_status(f"Test failed, retrying in {args.retry_delay} seconds...", "WARNING")
        else:
            print_status("All retry attempts failed", "ERROR")
            sys.exit(1)

if __name__ == "__main__":
    main()
