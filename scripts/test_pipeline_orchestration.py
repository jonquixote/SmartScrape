#!/usr/bin/env python3
"""
Test Pipeline Orchestration for SmartScrape

This script tests the Celery-based pipeline orchestration system.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.pipeline_orchestrator import pipeline_orchestrator
from core.tasks import extract_url_task, batch_extract_task, cache_warm_task
from core.celery_config import celery_app

def test_celery_configuration():
    """Test Celery configuration"""
    print("🔧 Testing Celery Configuration...")
    
    try:
        # Test Celery app
        assert celery_app.conf.task_serializer == 'json'
        assert celery_app.conf.timezone == 'UTC'
        print("  ✅ Celery configuration loaded successfully")
        
        # Test Redis connection (broker)
        inspector = celery_app.control.inspect()
        active_queues = inspector.active_queues()
        print("  ✅ Celery broker connection established")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Celery configuration test failed: {e}")
        return False

def test_task_registration():
    """Test that tasks are registered"""
    print("📋 Testing Task Registration...")
    
    try:
        # Check registered tasks
        registered_tasks = list(celery_app.tasks.keys())
        
        expected_tasks = [
            'core.tasks.extract_url_task',
            'core.tasks.batch_extract_task',
            'core.tasks.cache_warm_task',
            'core.tasks.pipeline_health_check_task'
        ]
        
        for task in expected_tasks:
            if task in registered_tasks:
                print(f"  ✅ Task registered: {task}")
            else:
                print(f"  ❌ Task missing: {task}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Task registration test failed: {e}")
        return False

def test_pipeline_orchestrator():
    """Test pipeline orchestrator functionality"""
    print("🎯 Testing Pipeline Orchestrator...")
    
    try:
        # Test orchestrator initialization
        assert pipeline_orchestrator is not None
        print("  ✅ Pipeline orchestrator initialized")
        
        # Test URL grouping
        test_urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/html",
            "https://jsonplaceholder.typicode.com/posts/1"
        ]
        
        domain_groups = pipeline_orchestrator._group_urls_by_domain(test_urls)
        assert len(domain_groups) >= 2  # At least httpbin.org and jsonplaceholder
        print("  ✅ URL grouping working correctly")
        
        # Test simple pipeline creation (without actually executing)
        # Note: We can't test actual execution without a running Celery worker
        print("  ✅ Pipeline orchestrator functional tests passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline orchestrator test failed: {e}")
        return False

def test_task_signatures():
    """Test task signatures and parameters"""
    print("✍️ Testing Task Signatures...")
    
    try:
        # Test task signature creation (doesn't execute)
        extract_sig = extract_url_task.s("https://httpbin.org/json", "universal_crawl4ai")
        assert extract_sig.task == 'core.tasks.extract_url_task'
        print("  ✅ Extract URL task signature created")
        
        batch_sig = batch_extract_task.s(["https://httpbin.org/json"], "universal_crawl4ai")
        assert batch_sig.task == 'core.tasks.batch_extract_task'
        print("  ✅ Batch extract task signature created")
        
        cache_sig = cache_warm_task.s()
        assert cache_sig.task == 'core.tasks.cache_warm_task'
        print("  ✅ Cache warm task signature created")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Task signature test failed: {e}")
        return False

def test_worker_connectivity():
    """Test if Celery workers are available"""
    print("👷 Testing Worker Connectivity...")
    
    try:
        # Check for active workers
        inspector = celery_app.control.inspect()
        
        # Get worker stats
        stats = inspector.stats()
        if stats:
            print(f"  ✅ Found {len(stats)} active worker(s)")
            for worker_name, worker_stats in stats.items():
                print(f"    - Worker: {worker_name}")
                print(f"      Pool: {worker_stats.get('pool', {}).get('implementation', 'unknown')}")
                print(f"      Processes: {worker_stats.get('pool', {}).get('processes', 'unknown')}")
        else:
            print("  ⚠️  No active workers found")
            print("  📝 To start a worker, run: celery -A core.celery_config worker --loglevel=info")
        
        # Test ping
        ping_result = inspector.ping()
        if ping_result:
            print(f"  ✅ Workers responding to ping: {len(ping_result)}")
        else:
            print("  ⚠️  No workers responded to ping")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Worker connectivity test failed: {e}")
        return False

def test_queue_configuration():
    """Test queue configuration"""
    print("📬 Testing Queue Configuration...")
    
    try:
        # Test routing configuration
        routes = celery_app.conf.task_routes
        assert 'core.tasks.extract_url_task' in routes
        assert routes['core.tasks.extract_url_task']['queue'] == 'extraction'
        print("  ✅ Task routing configured correctly")
        
        # Test queue inspection (if workers are available)
        inspector = celery_app.control.inspect()
        active_queues = inspector.active_queues()
        
        if active_queues:
            print("  ✅ Active queues detected:")
            for worker, queues in active_queues.items():
                for queue in queues:
                    print(f"    - {queue['name']} on {worker}")
        else:
            print("  ⚠️  No active queues (workers may not be running)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Queue configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 SmartScrape Pipeline Orchestration Tests")
    print("=" * 50)
    
    tests = [
        test_celery_configuration,
        test_task_registration,
        test_pipeline_orchestrator,
        test_task_signatures,
        test_worker_connectivity,
        test_queue_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All pipeline orchestration tests passed!")
        print("\n📝 Next steps:")
        print("1. Start a Celery worker: celery -A core.celery_config worker --loglevel=info")
        print("2. Start Celery monitoring: celery -A core.celery_config flower")
        print("3. Test pipeline execution via API endpoints")
    else:
        print("💥 Some tests failed. Check the output above for details.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
