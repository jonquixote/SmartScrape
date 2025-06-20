#!/usr/bin/env python
"""
Comprehensive test script for SmartScrape resource management and error handling.

This script runs a suite of real-world test scenarios to evaluate the performance,
reliability, and resilience of the resource management and error handling components.
It generates detailed reports on success rates, resource utilization, error distribution,
and recovery effectiveness.
"""

import os
import sys
import time
import json
import logging
import random
import argparse
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, Counter
from datetime import datetime
from urllib.parse import urlparse
from unittest.mock import patch

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Import SmartScrape components
from core.service_registry import ServiceRegistry
from core.configuration import get_resource_config
from core.session_manager import SessionManager
from core.proxy_manager import ProxyManager
from core.rate_limiter import RateLimiter
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.error_classifier import ErrorClassifier, ErrorCategory, ErrorSeverity
from strategies.core.strategy_context import StrategyContext
from strategies.bfs_strategy import BFSStrategy
from strategies.dfs_strategy import DFSStrategy
from strategies.ai_guided_strategy import AIGuidedStrategy
from controllers.adaptive_scraper import AdaptiveScraper
from extraction.extraction_pipeline import ExtractionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(script_dir, 'resource_management_test.log'))
    ]
)
logger = logging.getLogger("resource_test")

# Define test sites categories
TEST_SITES = {
    "standard": [
        "https://books.toscrape.com/",
        "https://quotes.toscrape.com/",
        "https://httpbin.org/html",
        "https://example.com/"
    ],
    "rate_limited": [
        "https://httpbin.org/status/429",  # Simulated rate limiting
        "https://httpbin.org/response-headers?Retry-After=5&status=429"
    ],
    "captcha": [
        "https://httpbin.org/response-headers?status=200&Content-Type=text/html",  # We'll patch the content for CAPTCHA simulation
    ],
    "ip_blocking": [
        "https://httpbin.org/status/403",  # Simulated IP blocking
    ],
    "high_latency": [
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/3",
    ],
    "error_prone": [
        "https://httpbin.org/status/500",
        "https://httpbin.org/status/502",
        "https://httpbin.org/status/503",
        "https://httpbin.org/status/504",
    ]
}

# Test execution metrics
test_metrics = {
    "start_time": None,
    "end_time": None,
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "retry_count": 0,
    "circuit_breaks": 0,
    "proxy_rotations": 0,
    "total_execution_time": 0,
    "errors_by_category": Counter(),
    "site_success_rates": defaultdict(lambda: {"attempts": 0, "successes": 0}),
    "resource_usage": {
        "sessions": {"created": 0, "reused": 0},
        "proxies": {"used": 0, "failures": 0},
        "rate_limiting": {"delays": 0, "total_delay_time": 0}
    }
}

# Thread-safe increment function for metrics
metrics_lock = threading.RLock()

def increment_metric(metric_path, value=1):
    """Thread-safe increment of a metric value."""
    with metrics_lock:
        parts = metric_path.split('.')
        current = test_metrics
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        if parts[-1] not in current:
            current[parts[-1]] = 0
        current[parts[-1]] += value

def update_site_metric(site_type, success=True):
    """Update site-specific success metrics."""
    with metrics_lock:
        test_metrics["site_success_rates"][site_type]["attempts"] += 1
        if success:
            test_metrics["site_success_rates"][site_type]["successes"] += 1

# -----------------------------------------------------------------------------
# Test Execution Functions
# -----------------------------------------------------------------------------

def initialize_services():
    """Initialize all resource management services."""
    # Reset the service registry
    ServiceRegistry._instance = None
    
    # Get default configuration
    config = get_resource_config()
    
    # Create a context with resource services
    context = StrategyContext.with_resource_services({"resource_services": config})
    
    logger.info("Resource management services initialized")
    return context

def create_test_strategies(context):
    """Create test strategies with the provided context."""
    strategies = {
        "bfs": BFSStrategy(context=context),
        "dfs": DFSStrategy(context=context),
        "ai_guided": AIGuidedStrategy(context=context),
        "adaptive": AdaptiveScraper(
            strategies=[
                BFSStrategy(context=context),
                DFSStrategy(context=context),
                AIGuidedStrategy(context=context)
            ],
            context=context
        )
    }
    
    logger.info(f"Created {len(strategies)} test strategies")
    return strategies

def track_session_metrics(original_get_session):
    """Wrap the session manager's get_session method to track metrics."""
    def wrapped_get_session(self, domain, force_new=False):
        if force_new:
            increment_metric("resource_usage.sessions.created")
        else:
            # Check if session exists before creating a new one
            with self._lock:
                if domain in self._sessions:
                    increment_metric("resource_usage.sessions.reused")
                else:
                    increment_metric("resource_usage.sessions.created")
        
        return original_get_session(self, domain, force_new)
    
    return wrapped_get_session

def track_proxy_metrics(original_get_proxy):
    """Wrap the proxy manager's get_proxy method to track metrics."""
    def wrapped_get_proxy(self, domain=None):
        increment_metric("resource_usage.proxies.used")
        return original_get_proxy(self, domain)
    
    return wrapped_get_proxy

def track_proxy_failure_metrics(original_mark_proxy_failed):
    """Wrap the proxy manager's mark_proxy_failed method to track metrics."""
    def wrapped_mark_proxy_failed(self, proxy_url):
        increment_metric("resource_usage.proxies.failures")
        increment_metric("proxy_rotations")
        return original_mark_proxy_failed(self, proxy_url)
    
    return wrapped_mark_proxy_failed

def track_rate_limit_metrics(original_wait_if_needed):
    """Wrap the rate limiter's wait_if_needed method to track metrics."""
    def wrapped_wait_if_needed(self, domain):
        start_time = time.time()
        result = original_wait_if_needed(self, domain)
        
        if result:  # If waited
            wait_time = time.time() - start_time
            increment_metric("resource_usage.rate_limiting.delays")
            increment_metric("resource_usage.rate_limiting.total_delay_time", wait_time)
        
        return result
    
    return wrapped_wait_if_needed

def track_retry_metrics(original_retry):
    """Wrap the retry manager's retry decorator to track metrics."""
    def wrapped_retry(domain, max_attempts=None, retry_conditions=None, backoff_factor=None, 
                      jitter=None, logger=None):
        retry_decorator = original_retry(domain, max_attempts, retry_conditions, 
                                        backoff_factor, jitter, logger)
        
        def wrapper(func):
            retried_func = retry_decorator(func)
            
            def tracked_func(*args, **kwargs):
                # The retry count will be incremented inside the retry decorator,
                # so we don't increment it here.
                try:
                    result = retried_func(*args, **kwargs)
                    increment_metric("successful_requests")
                    return result
                except Exception as e:
                    increment_metric("failed_requests")
                    raise
            
            return tracked_func
        
        return wrapper
    
    return wrapped_retry

def track_circuit_breaker_metrics(original_circuit_breaker):
    """Wrap the circuit breaker decorator to track metrics."""
    def wrapped_circuit_breaker(name, settings=None):
        circuit_decorator = original_circuit_breaker(name, settings)
        
        def wrapper(func):
            circuit_func = circuit_decorator(func)
            
            def tracked_func(*args, **kwargs):
                try:
                    return circuit_func(*args, **kwargs)
                except OpenCircuitError:
                    increment_metric("circuit_breaks")
                    raise
                except Exception:
                    # Other exceptions are counted elsewhere
                    raise
            
            return tracked_func
        
        return wrapper
    
    return wrapped_circuit_breaker

def track_error_classification(original_classify_exception):
    """Wrap the error classifier's classify_exception method to track metrics."""
    def wrapped_classify_exception(self, exception, metadata=None):
        classification = original_classify_exception(self, exception, metadata)
        
        # Track error category
        category = classification.get('category')
        if category:
            if hasattr(category, 'value'):  # If it's an enum
                category_name = category.value
            else:
                category_name = str(category)
            
            with metrics_lock:
                test_metrics["errors_by_category"][category_name] += 1
        
        return classification
    
    return wrapped_classify_exception

def patch_all_metrics_tracking(context):
    """Patch all service methods to track metrics."""
    # Get service instances
    session_manager = context.get_session_manager()
    proxy_manager = context.get_proxy_manager()
    rate_limiter = context.get_rate_limiter()
    retry_manager = context.get_retry_manager()
    circuit_breaker_manager = context.get_circuit_breaker_manager()
    error_classifier = context.get_error_classifier()
    
    # Apply patches
    patches = [
        patch.object(SessionManager, 'get_session', track_session_metrics(SessionManager.get_session)),
        patch.object(ProxyManager, 'get_proxy', track_proxy_metrics(ProxyManager.get_proxy)),
        patch.object(ProxyManager, 'mark_proxy_failed', track_proxy_failure_metrics(ProxyManager.mark_proxy_failed)),
        patch.object(RateLimiter, 'wait_if_needed', track_rate_limit_metrics(RateLimiter.wait_if_needed)),
        patch.object(RetryManager, 'retry', track_retry_metrics(RetryManager.retry)),
        patch.object(CircuitBreakerManager, 'circuit_breaker', track_circuit_breaker_metrics(CircuitBreakerManager.circuit_breaker)),
        patch.object(ErrorClassifier, 'classify_exception', track_error_classification(ErrorClassifier.classify_exception))
    ]
    
    logger.info("Applied metrics tracking patches to all services")
    return patches

def simulate_captcha_content(url):
    """Simulate CAPTCHA content in responses for certain URLs."""
    if "captcha" in url:
        return """
        <html>
            <body>
                <div class="g-recaptcha" data-sitekey="6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI"></div>
                <form>
                    <h2>Please complete the CAPTCHA to continue</h2>
                    <div>Verify you are human by solving the puzzle below</div>
                </form>
            </body>
        </html>
        """
    return None

def run_single_test(strategy_name, strategy, url, max_pages=2):
    """Run a single test using the specified strategy and URL."""
    try:
        increment_metric("total_requests")
        
        site_type = "standard"
        for category, urls in TEST_SITES.items():
            if url in urls:
                site_type = category
                break
        
        # Execute strategy
        start_time = time.time()
        result = strategy.scrape(url, max_pages=max_pages)
        execution_time = time.time() - start_time
        
        increment_metric("total_execution_time", execution_time)
        update_site_metric(site_type, success=True)
        
        # Return test result
        return {
            "success": True,
            "strategy": strategy_name,
            "url": url,
            "site_type": site_type,
            "execution_time": execution_time,
            "pages_scraped": len(result.get("pages_scraped", [])) if result else 0,
            "data_extracted": len(result.get("extracted_data", [])) if result else 0
        }
        
    except Exception as e:
        logger.warning(f"Test failed for {url} with strategy {strategy_name}: {str(e)}")
        update_site_metric(site_type, success=False)
        
        # Return failure information
        return {
            "success": False,
            "strategy": strategy_name,
            "url": url,
            "site_type": site_type,
            "error": str(e),
            "error_type": type(e).__name__
        }

def run_test_scenarios(strategies, concurrency=4):
    """Run test scenarios with all strategies and site types."""
    test_cases = []
    
    # Generate test cases for each strategy and site type
    for strategy_name, strategy in strategies.items():
        for site_type, urls in TEST_SITES.items():
            for url in urls:
                test_cases.append({
                    "strategy_name": strategy_name,
                    "strategy": strategy,
                    "url": url,
                    "site_type": site_type
                })
    
    # Shuffle test cases to avoid hitting the same site consecutively
    random.shuffle(test_cases)
    
    # Run tests concurrently
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_case = {
            executor.submit(
                run_single_test, 
                case["strategy_name"], 
                case["strategy"], 
                case["url"]
            ): case for case in test_cases
        }
        
        for future in future_to_case:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                case = future_to_case[future]
                logger.error(f"Error executing test case: {str(e)}")
                results.append({
                    "success": False,
                    "strategy": case["strategy_name"],
                    "url": case["url"],
                    "site_type": case["site_type"],
                    "error": str(e),
                    "error_type": type(e).__name__
                })
    
    return results

def simulate_captcha_responses():
    """Patch requests to simulate CAPTCHA responses."""
    original_get = requests.Session.get
    
    def captcha_injecting_get(self, url, **kwargs):
        response = original_get(self, url, **kwargs)
        
        # If this is a captcha test URL, modify the response content
        captcha_content = simulate_captcha_content(url)
        if captcha_content and response.status_code == 200:
            # This is a bit hacky but works for a test script - we're patching
            # the response object after it's created
            response._content = captcha_content.encode('utf-8')
            
            # Also add a custom header so our captcha detection can work
            response.headers['X-Contains-Captcha'] = 'true'
        
        return response
    
    return patch.object(requests.Session, 'get', captcha_injecting_get)

def generate_report(test_results):
    """Generate a comprehensive report from test results."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "execution_time": {
            "start": test_metrics["start_time"],
            "end": test_metrics["end_time"],
            "duration_seconds": (
                test_metrics["end_time"] - test_metrics["start_time"]
                if test_metrics["end_time"] and test_metrics["start_time"] else None
            )
        },
        "overall_metrics": {
            "total_tests": len(test_results),
            "successful_tests": sum(1 for r in test_results if r.get("success", False)),
            "failed_tests": sum(1 for r in test_results if not r.get("success", False)),
            "success_rate": (
                sum(1 for r in test_results if r.get("success", False)) / len(test_results)
                if test_results else 0
            )
        },
        "resource_utilization": {
            "sessions": dict(test_metrics["resource_usage"]["sessions"]),
            "proxies": dict(test_metrics["resource_usage"]["proxies"]),
            "rate_limiting": dict(test_metrics["resource_usage"]["rate_limiting"])
        },
        "request_metrics": {
            "total_requests": test_metrics["total_requests"],
            "successful_requests": test_metrics["successful_requests"],
            "failed_requests": test_metrics["failed_requests"],
            "retry_count": test_metrics["retry_count"],
            "circuit_breaks": test_metrics["circuit_breaks"],
            "proxy_rotations": test_metrics["proxy_rotations"]
        },
        "error_distribution": {
            category: count for category, count in test_metrics["errors_by_category"].items()
        },
        "site_type_performance": {
            site_type: {
                "success_rate": (
                    metrics["successes"] / metrics["attempts"] 
                    if metrics["attempts"] > 0 else 0
                ),
                "attempts": metrics["attempts"],
                "successes": metrics["successes"]
            }
            for site_type, metrics in test_metrics["site_success_rates"].items()
        },
        "strategy_performance": defaultdict(lambda: {"success": 0, "failure": 0, "avg_execution_time": 0})
    }
    
    # Calculate strategy-specific metrics
    strategy_execution_times = defaultdict(list)
    for result in test_results:
        strategy_name = result.get("strategy", "unknown")
        success = result.get("success", False)
        
        if success:
            report["strategy_performance"][strategy_name]["success"] += 1
            execution_time = result.get("execution_time", 0)
            strategy_execution_times[strategy_name].append(execution_time)
        else:
            report["strategy_performance"][strategy_name]["failure"] += 1
    
    # Calculate average execution times
    for strategy_name, times in strategy_execution_times.items():
        if times:
            report["strategy_performance"][strategy_name]["avg_execution_time"] = sum(times) / len(times)
    
    # Convert defaultdicts to regular dicts for JSON serialization
    report["strategy_performance"] = dict(report["strategy_performance"])
    for strategy, metrics in report["strategy_performance"].items():
        report["strategy_performance"][strategy] = dict(metrics)
    
    return report

def save_report(report, filename="resource_management_test_report.json"):
    """Save the test report to a file."""
    report_path = os.path.join(script_dir, filename)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to {report_path}")
    return report_path

def print_report_summary(report):
    """Print a summary of the test report to the console."""
    print("\n" + "="*80)
    print(" SMARTSCRAPE RESOURCE MANAGEMENT & ERROR HANDLING TEST SUMMARY ")
    print("="*80)
    
    print(f"\nTest executed on: {report['timestamp']}")
    print(f"Total test duration: {report['execution_time']['duration_seconds']:.2f} seconds")
    
    print("\nOVERALL RESULTS:")
    print(f"  Total tests: {report['overall_metrics']['total_tests']}")
    print(f"  Successful tests: {report['overall_metrics']['successful_tests']}")
    print(f"  Failed tests: {report['overall_metrics']['failed_tests']}")
    print(f"  Success rate: {report['overall_metrics']['success_rate']:.2%}")
    
    print("\nRESOURCE UTILIZATION:")
    print(f"  Sessions created: {report['resource_utilization']['sessions']['created']}")
    print(f"  Sessions reused: {report['resource_utilization']['sessions']['reused']}")
    print(f"  Proxies used: {report['resource_utilization']['proxies']['used']}")
    print(f"  Proxy failures: {report['resource_utilization']['proxies']['failures']}")
    print(f"  Rate limiting delays: {report['resource_utilization']['rate_limiting']['delays']}")
    print(f"  Total delay time: {report['resource_utilization']['rate_limiting']['total_delay_time']:.2f} seconds")
    
    print("\nERROR DISTRIBUTION:")
    for category, count in report['error_distribution'].items():
        print(f"  {category}: {count}")
    
    print("\nSITE TYPE PERFORMANCE:")
    for site_type, metrics in report['site_type_performance'].items():
        print(f"  {site_type}: {metrics['success_rate']:.2%} success rate ({metrics['successes']}/{metrics['attempts']})")
    
    print("\nSTRATEGY PERFORMANCE:")
    for strategy, metrics in report['strategy_performance'].items():
        total = metrics['success'] + metrics['failure']
        success_rate = metrics['success'] / total if total > 0 else 0
        print(f"  {strategy}: {success_rate:.2%} success rate, {metrics['avg_execution_time']:.2f}s avg execution time")
    
    print("\nREQUEST METRICS:")
    print(f"  Total requests: {report['request_metrics']['total_requests']}")
    print(f"  Successful requests: {report['request_metrics']['successful_requests']}")
    print(f"  Failed requests: {report['request_metrics']['failed_requests']}")
    print(f"  Retry count: {report['request_metrics']['retry_count']}")
    print(f"  Circuit breaks: {report['request_metrics']['circuit_breaks']}")
    print(f"  Proxy rotations: {report['request_metrics']['proxy_rotations']}")
    
    print("\nFull report saved to: resource_management_test_report.json")
    print("="*80)

def run_tests(concurrency=4):
    """Run all resource management and error handling tests."""
    logger.info("Starting resource management and error handling tests")
    test_metrics["start_time"] = time.time()
    
    # Initialize services
    context = initialize_services()
    
    # Create test strategies
    strategies = create_test_strategies(context)
    
    # Apply metrics tracking patches
    service_patches = patch_all_metrics_tracking(context)
    
    # Apply captcha simulation
    captcha_patch = simulate_captcha_responses()
    
    try:
        # Start patches
        for p in service_patches:
            p.start()
        captcha_patch.start()
        
        # Run test scenarios
        test_results = run_test_scenarios(strategies, concurrency)
        
        # Stop patches
        for p in service_patches:
            p.stop()
        captcha_patch.stop()
        
    finally:
        # Ensure patches are stopped even if an exception occurs
        for p in service_patches:
            if p.is_active():
                p.stop()
        if captcha_patch.is_active():
            captcha_patch.stop()
        
        # Clean up services
        ServiceRegistry._instance.shutdown_all()
        ServiceRegistry._instance = None
    
    test_metrics["end_time"] = time.time()
    
    # Generate and save report
    report = generate_report(test_results)
    save_report(report)
    print_report_summary(report)
    
    logger.info("Resource management and error handling tests completed")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SmartScrape resource management and error handling tests')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of concurrent tests to run')
    
    args = parser.parse_args()
    run_tests(concurrency=args.concurrency)