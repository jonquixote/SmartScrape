"""
Performance test for parallel scraping capabilities of SmartScrape.

This test verifies how the system handles multiple concurrent scraping jobs,
measuring performance, resource utilization, and stability under load.
"""

import os
import sys
import time
import asyncio
import logging
import argparse
import json
import psutil
import types
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import SmartScrape components
from core.service_registry import ServiceRegistry
from core.configuration import get_resource_config
from strategies.core.strategy_context import StrategyContext
from strategies.bfs_strategy import BFSStrategy
from strategies.dfs_strategy import DFSStrategy
from strategies.ai_guided_strategy import AIGuidedStrategy

# Remove problematic import
# from controllers.adaptive_scraper import AdaptiveScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('parallel_scraping_test.log')
    ]
)
logger = logging.getLogger("parallel_scraping_test")

class PerformanceMetrics:
    """Track performance metrics during parallel scraping test."""
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.cpu_usage = []
        self.memory_usage = []
        self.task_completion_times = []
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.errors = []
        
    def record_cpu_memory(self):
        """Record current CPU and memory usage."""
        process = psutil.Process(os.getpid())
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
    
    def record_task_completion(self, elapsed):
        """Record completion time of an individual task."""
        self.task_completion_times.append(elapsed)
        self.successful_tasks += 1
    
    def record_task_failure(self, error):
        """Record a task failure."""
        self.failed_tasks += 1
        self.errors.append(str(error))
    
    def complete(self):
        """Mark the end of the test and calculate final metrics."""
        self.end_time = time.time()
    
    def get_summary(self):
        """Return a summary of performance metrics."""
        total_time = self.end_time - self.start_time if self.end_time else time.time() - self.start_time
        avg_task_time = sum(self.task_completion_times) / len(self.task_completion_times) if self.task_completion_times else 0
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        max_memory = max(self.memory_usage) if self.memory_usage else 0
        
        return {
            "total_time_seconds": total_time,
            "total_tasks": self.successful_tasks + self.failed_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.successful_tasks / (self.successful_tasks + self.failed_tasks) if (self.successful_tasks + self.failed_tasks) > 0 else 0,
            "avg_task_time_seconds": avg_task_time,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "errors": self.errors[:10]  # Limit to first 10 errors
        }

def setup_service_context():
    """Set up a complete context with all resource management services."""
    # Reset the service registry
    ServiceRegistry._instance = None
    
    # Get default configuration
    config = get_resource_config()
    
    # Initialize core services that we need before creating the context
    registry = ServiceRegistry()
    
    # Import required service classes
    from core.url_service import URLService
    from core.html_service import HTMLService
    from core.session_manager import SessionManager
    from core.proxy_manager import ProxyManager
    from core.rate_limiter import RateLimiter
    from core.error_classifier import ErrorClassifier
    from core.retry_manager import RetryManager
    from core.circuit_breaker import CircuitBreakerManager
    
    # Register essential services manually first
    registry.register_service("url_service", URLService())
    registry.register_service("html_service", HTMLService())
    
    # Now create context with all resource services
    context = StrategyContext.with_resource_services({"resource_services": config})
    
    # Verify critical services are properly registered
    try:
        url_service = registry.get_service("url_service")
        logger.info("URL service is properly registered")
    except Exception as e:
        logger.error(f"URL service registration failed: {e}")
    
    return context

def setup_scraping_strategies(context, mock_browser=True):
    """Set up strategies for testing, optionally mocking browser automation."""
    strategies = {
        "bfs": BFSStrategy(context=context),
        "dfs": DFSStrategy(context=context),
        "ai_guided": AIGuidedStrategy(context=context),
    }
    
    if mock_browser:
        # Patch the strategies to avoid actual browser automation for testing
        from unittest.mock import patch
        
        def mock_scrape(self, url, max_pages=2, **kwargs):
            """Mock implementation to avoid actual browser usage."""
            domain = urlparse(url).netloc
            
            # Simulate some work
            time.sleep(0.1)
            
            # Return mock result
            return {
                "success": True,
                "strategy": self.name,
                "pages_scraped": [
                    {"url": url, "content": f"<html><body><h1>Mock Page for {domain}</h1></body></html>"},
                    {"url": f"{url}page2", "content": f"<html><body><h1>Mock Page 2 for {domain}</h1></body></html>"}
                ],
                "metadata": {
                    "mock": True,
                    "domain": domain
                }
            }
        
        # Apply the mock to all strategies
        for strategy in strategies.values():
            strategy.scrape = types.MethodType(mock_scrape, strategy)
    
    return strategies

def generate_test_urls(count=10):
    """Generate test URLs for scraping."""
    base_urls = [
        "https://books.toscrape.com/",
        "https://quotes.toscrape.com/",
        "https://example.com/",
        "https://httpbin.org/html",
        "https://news.ycombinator.com/"
    ]
    
    urls = []
    for i in range(count):
        base = base_urls[i % len(base_urls)]
        if '?' in base:
            urls.append(f"{base}&test_param={i}")
        else:
            urls.append(f"{base}?test_param={i}")
    
    return urls

def run_scraping_task(strategy, url, metrics):
    """Run a single scraping task and record metrics."""
    task_start = time.time()
    try:
        result = strategy.scrape(url, max_pages=2)
        task_elapsed = time.time() - task_start
        metrics.record_task_completion(task_elapsed)
        return {
            "url": url,
            "success": True,
            "pages_scraped": len(result.get("pages_scraped", [])),
            "time_seconds": task_elapsed
        }
    except Exception as e:
        metrics.record_task_failure(e)
        logger.error(f"Error scraping {url}: {str(e)}")
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }

async def monitor_resources(metrics, interval=1.0, stop_event=None):
    """Continuously monitor system resources during the test."""
    while not stop_event.is_set():
        metrics.record_cpu_memory()
        await asyncio.sleep(interval)

async def run_parallel_scraping_test(concurrency=4, url_count=10, use_mock=True):
    """Run a parallel scraping test with the specified concurrency level."""
    logger.info(f"Starting parallel scraping test with concurrency={concurrency}, url_count={url_count}")
    
    # Set up test environment
    service_context = setup_service_context()
    strategies = setup_scraping_strategies(service_context, mock_browser=use_mock)
    test_urls = generate_test_urls(url_count)
    
    # Initialize performance metrics
    metrics = PerformanceMetrics()
    
    # Set up resource monitoring
    stop_monitoring = asyncio.Event()
    monitoring_task = asyncio.create_task(monitor_resources(metrics, interval=0.5, stop_event=stop_monitoring))
    
    # Create tasks to run in parallel
    tasks = []
    for i, url in enumerate(test_urls):
        # Alternate between strategies
        strategy_name = ["bfs", "dfs", "ai_guided"][i % 3]
        strategy = strategies[strategy_name]
        tasks.append((strategy, url))
    
    # Run tasks in parallel using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(run_scraping_task, strategy, url, metrics): (strategy.name, url)
            for strategy, url in tasks
        }
        
        # Process results as they complete
        for future in future_to_url:
            strategy_name, url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed scraping {url} with {strategy_name} strategy")
            except Exception as e:
                logger.error(f"Task for {url} with {strategy_name} strategy raised exception: {str(e)}")
                metrics.record_task_failure(e)
    
    # Stop resource monitoring
    stop_monitoring.set()
    await monitoring_task
    
    # Calculate final metrics
    metrics.complete()
    summary = metrics.get_summary()
    
    # Log results
    logger.info(f"Parallel scraping test completed: {summary['successful_tasks']}/{summary['total_tasks']} tasks successful")
    logger.info(f"Total time: {summary['total_time_seconds']:.2f}s, Avg task time: {summary['avg_task_time_seconds']:.2f}s")
    logger.info(f"Avg CPU: {summary['avg_cpu_percent']:.1f}%, Avg Memory: {summary['avg_memory_mb']:.1f}MB")
    
    # Save detailed results
    test_results = {
        "metrics": summary,
        "task_results": results
    }
    
    with open("parallel_scraping_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Results saved to parallel_scraping_results.json")
    
    # Shutdown services
    ServiceRegistry._instance.shutdown_all()
    
    return test_results

def main():
    """Main entry point for the parallel scraping test."""
    parser = argparse.ArgumentParser(description='Run parallel scraping performance test')
    parser.add_argument('--concurrency', type=int, default=4, help='Number of concurrent scraping tasks')
    parser.add_argument('--urls', type=int, default=10, help='Number of URLs to scrape')
    parser.add_argument('--real', action='store_true', help='Use real browser automation (slow)')
    
    args = parser.parse_args()
    
    # Run the async test
    asyncio.run(run_parallel_scraping_test(
        concurrency=args.concurrency,
        url_count=args.urls,
        use_mock=not args.real
    ))

if __name__ == "__main__":
    main()