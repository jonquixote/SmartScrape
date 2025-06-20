#!/usr/bin/env python3
"""
Server Health Check Script
This script verifies that all SmartScrape services are running and healthy.
"""

import sys
import time
import requests
import redis
import argparse
from typing import Dict, Any, List
import json

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_status(message: str, status: str = "INFO"):
    """Print colored status messages."""
    color_map = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
        "TESTING": Colors.PURPLE
    }
    color = color_map.get(status, Colors.NC)
    print(f"{color}[{status}]{Colors.NC} {message}")

def check_redis_health(host: str = "localhost", port: int = 6379, timeout: int = 5) -> Dict[str, Any]:
    """Check Redis server health."""
    print_status(f"Checking Redis connection at {host}:{port}...", "TESTING")
    
    try:
        r = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=timeout)
        
        # Test basic connectivity
        start_time = time.time()
        pong = r.ping()
        response_time = round((time.time() - start_time) * 1000, 2)
        
        if pong:
            # Get Redis info
            info = r.info()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "version": info.get("redis_version", "unknown"),
                "memory_usage_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        else:
            return {"status": "unhealthy", "error": "Redis ping failed"}
            
    except redis.ConnectionError as e:
        return {"status": "unhealthy", "error": f"Connection failed: {str(e)}"}
    except redis.TimeoutError as e:
        return {"status": "unhealthy", "error": f"Timeout: {str(e)}"}
    except Exception as e:
        return {"status": "unhealthy", "error": f"Unexpected error: {str(e)}"}

def check_fastapi_health(base_url: str = "http://localhost:5000", timeout: int = 10) -> Dict[str, Any]:
    """Check FastAPI server health."""
    print_status(f"Checking FastAPI server at {base_url}...", "TESTING")
    
    try:
        # Test basic connectivity
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=timeout)
        response_time = round((time.time() - start_time) * 1000, 2)
        
        if response.status_code == 200:
            health_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            
            # Test additional endpoints
            endpoints_to_test = [
                "/docs",
                "/api/status", 
                "/api/health/detailed"
            ]
            
            endpoint_status = {}
            for endpoint in endpoints_to_test:
                try:
                    test_response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    endpoint_status[endpoint] = {
                        "status_code": test_response.status_code,
                        "accessible": test_response.status_code < 500
                    }
                except Exception as e:
                    endpoint_status[endpoint] = {
                        "status_code": None,
                        "accessible": False,
                        "error": str(e)
                    }
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "health_data": health_data,
                "endpoints": endpoint_status
            }
        else:
            return {
                "status": "unhealthy", 
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
            
    except requests.ConnectionError as e:
        return {"status": "unhealthy", "error": f"Connection failed: {str(e)}"}
    except requests.Timeout as e:
        return {"status": "unhealthy", "error": f"Request timeout: {str(e)}"}
    except Exception as e:
        return {"status": "unhealthy", "error": f"Unexpected error: {str(e)}"}

def check_celery_health(redis_host: str = "localhost", redis_port: int = 6379) -> Dict[str, Any]:
    """Check Celery worker health by inspecting Redis and trying to get worker stats."""
    print_status("Checking Celery worker health...", "TESTING")
    
    try:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Check if there are any Celery-related keys in Redis
        celery_keys = r.keys("celery*") + r.keys("_kombu*")
        
        # Try to get active workers (this requires celery to be importable)
        try:
            from celery import Celery
            from core.celery_config import celery_app
            
            # Get worker stats
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            registered_tasks = inspect.registered()
            
            if active_workers:
                return {
                    "status": "healthy",
                    "workers": list(active_workers.keys()),
                    "celery_keys_count": len(celery_keys),
                    "registered_tasks": registered_tasks
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "No active Celery workers found",
                    "celery_keys_count": len(celery_keys)
                }
                
        except ImportError as e:
            return {
                "status": "warning",
                "error": f"Cannot import Celery modules: {str(e)}",
                "celery_keys_count": len(celery_keys)
            }
            
    except Exception as e:
        return {"status": "unhealthy", "error": f"Error checking Celery: {str(e)}"}

def run_comprehensive_health_check(
    fastapi_url: str = "http://localhost:5000",
    redis_host: str = "localhost", 
    redis_port: int = 6379,
    timeout: int = 10
) -> Dict[str, Any]:
    """Run comprehensive health check on all services."""
    
    print_status("Starting comprehensive health check...", "INFO")
    print("=" * 60)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "services": {},
        "overall_status": "healthy"
    }
    
    # Check Redis
    redis_result = check_redis_health(redis_host, redis_port, timeout)
    results["services"]["redis"] = redis_result
    
    if redis_result["status"] == "healthy":
        print_status(f"✅ Redis: Healthy (v{redis_result.get('version', 'unknown')}, {redis_result.get('response_time_ms', 0)}ms)", "SUCCESS")
    else:
        print_status(f"❌ Redis: {redis_result.get('error', 'Unknown error')}", "ERROR")
        results["overall_status"] = "unhealthy"
    
    # Check FastAPI
    fastapi_result = check_fastapi_health(fastapi_url, timeout)
    results["services"]["fastapi"] = fastapi_result
    
    if fastapi_result["status"] == "healthy":
        print_status(f"✅ FastAPI: Healthy ({fastapi_result.get('response_time_ms', 0)}ms)", "SUCCESS")
        
        # Show endpoint status
        for endpoint, status in fastapi_result.get("endpoints", {}).items():
            if status["accessible"]:
                print_status(f"  ✅ {endpoint}: HTTP {status['status_code']}", "SUCCESS")
            else:
                print_status(f"  ❌ {endpoint}: Failed", "WARNING")
    else:
        print_status(f"❌ FastAPI: {fastapi_result.get('error', 'Unknown error')}", "ERROR")
        results["overall_status"] = "unhealthy"
    
    # Check Celery
    celery_result = check_celery_health(redis_host, redis_port)
    results["services"]["celery"] = celery_result
    
    if celery_result["status"] == "healthy":
        worker_count = len(celery_result.get("workers", []))
        print_status(f"✅ Celery: Healthy ({worker_count} workers)", "SUCCESS")
    elif celery_result["status"] == "warning":
        print_status(f"⚠️  Celery: {celery_result.get('error', 'Warning')}", "WARNING")
    else:
        print_status(f"❌ Celery: {celery_result.get('error', 'Unknown error')}", "ERROR")
        if results["overall_status"] == "healthy":
            results["overall_status"] = "degraded"
    
    print("=" * 60)
    
    # Overall status
    if results["overall_status"] == "healthy":
        print_status("🎉 All services are healthy!", "SUCCESS")
    elif results["overall_status"] == "degraded":
        print_status("⚠️  Some services have issues but core functionality works", "WARNING")
    else:
        print_status("❌ Critical services are down", "ERROR")
    
    return results

def wait_for_services(
    fastapi_url: str = "http://localhost:5000",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    max_wait: int = 60,
    check_interval: int = 2
) -> bool:
    """Wait for services to become healthy."""
    
    print_status(f"Waiting for services to be ready (max {max_wait}s)...", "INFO")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        print_status(f"Checking services... ({int(time.time() - start_time)}s elapsed)", "TESTING")
        
        redis_healthy = check_redis_health(redis_host, redis_port, 2)["status"] == "healthy"
        fastapi_healthy = check_fastapi_health(fastapi_url, 5)["status"] == "healthy"
        
        if redis_healthy and fastapi_healthy:
            print_status("All services are ready!", "SUCCESS")
            return True
        
        if not redis_healthy:
            print_status("Waiting for Redis...", "INFO")
        if not fastapi_healthy:
            print_status("Waiting for FastAPI...", "INFO")
            
        time.sleep(check_interval)
    
    print_status(f"Services did not become ready within {max_wait} seconds", "ERROR")
    return False

def main():
    """Main function to run health checks."""
    parser = argparse.ArgumentParser(description="SmartScrape Health Check")
    parser.add_argument("--fastapi-url", default="http://localhost:5000", help="FastAPI server URL")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--wait", action="store_true", help="Wait for services to become ready")
    parser.add_argument("--max-wait", type=int, default=60, help="Maximum time to wait for services (seconds)")
    parser.add_argument("--output-json", help="Save results to JSON file")
    parser.add_argument("--exit-code", action="store_true", help="Exit with non-zero code if unhealthy")
    
    args = parser.parse_args()
    
    if args.wait:
        success = wait_for_services(
            args.fastapi_url, 
            args.redis_host, 
            args.redis_port, 
            args.max_wait
        )
        if not success:
            sys.exit(1)
    
    # Run comprehensive health check
    results = run_comprehensive_health_check(
        args.fastapi_url,
        args.redis_host, 
        args.redis_port,
        args.timeout
    )
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print_status(f"Results saved to {args.output_json}", "INFO")
    
    # Exit with appropriate code
    if args.exit_code:
        if results["overall_status"] == "healthy":
            sys.exit(0)
        elif results["overall_status"] == "degraded":
            sys.exit(2)  # Warning exit code
        else:
            sys.exit(1)  # Error exit code

if __name__ == "__main__":
    main()
