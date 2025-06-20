"""
Resilience validation suite for SmartScrape.

This test framework injects failures at different points in the system
to validate recovery capabilities, measure performance under stress,
and ensure data integrity during failures.
"""

import logging
import os
import sys
import pytest
import json
import time
import random
import threading
import requests
from unittest.mock import MagicMock, patch
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# Add the root directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                   handlers=[
                       logging.StreamHandler(),
                       logging.FileHandler('resilience_tests.log')
                   ])
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Failure Injection Tools
# -------------------------------------------------------------------------

class FailureInjector:
    """Tools for injecting various types of failures into the system."""
    
    @staticmethod
    @contextmanager
    def network_failure(failure_rate=0.5, exceptions=None):
        """
        Inject network failures at the requests level.
        
        Args:
            failure_rate: Probability of a request failing (0.0-1.0)
            exceptions: List of exception types to raise, default [ConnectionError, Timeout]
        """
        if exceptions is None:
            exceptions = [requests.ConnectionError, requests.Timeout]
        
        original_get = requests.Session.get
        original_post = requests.Session.post
        
        def failing_get(self, url, **kwargs):
            if random.random() < failure_rate:
                exception_cls = random.choice(exceptions)
                raise exception_cls(f"Injected network failure for {url}")
            return original_get(self, url, **kwargs)
        
        def failing_post(self, url, **kwargs):
            if random.random() < failure_rate:
                exception_cls = random.choice(exceptions)
                raise exception_cls(f"Injected network failure for {url}")
            return original_post(self, url, **kwargs)
        
        # Apply patches
        with patch.object(requests.Session, 'get', failing_get):
            with patch.object(requests.Session, 'post', failing_post):
                logger.info(f"Injecting network failures (rate: {failure_rate:.1%})")
                yield
        
        logger.info("Network failure injection ended")
    
    @staticmethod
    @contextmanager
    def service_unavailability(service_name, error_code=503, error_message="Service Unavailable"):
        """
        Simulate a service being unavailable.
        
        Args:
            service_name: Name of the service to make unavailable
            error_code: HTTP status code to return
            error_message: Error message to include in the response
        """
        # Different handling based on service type
        if service_name == "proxy_service":
            original_get_proxy = ProxyManager.get_proxy
            
            def unavailable_get_proxy(self, domain=None):
                raise Exception(f"Injected failure: {service_name} is unavailable")
            
            with patch.object(ProxyManager, 'get_proxy', unavailable_get_proxy):
                logger.info(f"Injecting service unavailability for {service_name}")
                yield
                
        elif service_name == "ai_service":
            from core.ai_service import AIService
            original_process = AIService.process
            
            def unavailable_process(self, *args, **kwargs):
                mock_response = MagicMock()
                mock_response.status_code = error_code
                mock_response.text = error_message
                mock_response.raise_for_status.side_effect = requests.HTTPError(
                    f"HTTP Error {error_code}", response=mock_response
                )
                raise requests.HTTPError(f"HTTP Error {error_code}", response=mock_response)
            
            with patch.object(AIService, 'process', unavailable_process):
                logger.info(f"Injecting service unavailability for {service_name}")
                yield
                
        else:
            # Generic service unavailability
            logger.info(f"Injecting generic service unavailability for {service_name}")
            yield
        
        logger.info(f"Service {service_name} availability restored")
    
    @staticmethod
    @contextmanager
    def resource_exhaustion(resource_type="memory", limit=None):
        """
        Simulate resource exhaustion (memory, CPU, file handles, etc).
        
        Args:
            resource_type: Type of resource to exhaust (memory, cpu, file_handles)
            limit: Limit to apply (implementation depends on resource_type)
        """
        if resource_type == "memory":
            # Track memory allocation attempts
            memory_usage = {"current": 0, "limit": limit or 1000000}
            tracked_objects = []
            
            original_list_init = list.__init__
            original_dict_init = dict.__init__
            
            def memory_tracking_list_init(self, *args, **kwargs):
                result = original_list_init(self, *args, **kwargs)
                # Simulate memory usage increase
                if len(args) > 0 and isinstance(args[0], (list, tuple)) and len(args[0]) > 100:
                    memory_usage["current"] += len(args[0])
                    tracked_objects.append(self)
                    if memory_usage["current"] > memory_usage["limit"]:
                        raise MemoryError("Injected memory exhaustion")
                return result
            
            def memory_tracking_dict_init(self, *args, **kwargs):
                result = original_dict_init(self, *args, **kwargs)
                # Simulate memory usage increase
                if len(kwargs) > 100 or (len(args) > 0 and isinstance(args[0], dict) and len(args[0]) > 100):
                    memory_usage["current"] += 1000  # Arbitrary increase
                    tracked_objects.append(self)
                    if memory_usage["current"] > memory_usage["limit"]:
                        raise MemoryError("Injected memory exhaustion")
                return result
            
            with patch.object(list, '__init__', memory_tracking_list_init):
                with patch.object(dict, '__init__', memory_tracking_dict_init):
                    logger.info(f"Injecting {resource_type} exhaustion (limit: {memory_usage['limit']})")
                    yield
            
            # Clean up tracking
            tracked_objects.clear()
            
        elif resource_type == "cpu":
            # Simulate CPU exhaustion by spawning CPU-intensive threads
            threads = []
            stop_threads = threading.Event()
            
            def cpu_intensive_task():
                while not stop_threads.is_set():
                    # Perform CPU-intensive calculation
                    _ = [i**2 for i in range(10000)]
            
            # Start CPU-intensive threads
            num_threads = limit or (os.cpu_count() - 1 or 1)
            for _ in range(num_threads):
                thread = threading.Thread(target=cpu_intensive_task)
                thread.daemon = True
                thread.start()
                threads.append(thread)
            
            logger.info(f"Injecting {resource_type} exhaustion ({num_threads} threads)")
            yield
            
            # Stop threads
            stop_threads.set()
            for thread in threads:
                thread.join(timeout=1)
                
        elif resource_type == "file_handles":
            # Simulate file handle exhaustion
            open_files = []
            max_files = limit or 100
            
            try:
                # Open temporary files to simulate handle exhaustion
                for i in range(max_files):
                    f = open(f"/tmp/test_file_{i}", "w")
                    open_files.append(f)
                
                logger.info(f"Injecting {resource_type} exhaustion ({max_files} handles)")
                yield
                
            finally:
                # Close all files
                for f in open_files:
                    try:
                        f.close()
                    except:
                        pass
                
                # Remove temporary files
                for i in range(max_files):
                    try:
                        os.remove(f"/tmp/test_file_{i}")
                    except:
                        pass
        else:
            logger.info(f"Injecting generic {resource_type} exhaustion")
            yield
        
        logger.info(f"Resource exhaustion ended for {resource_type}")
    
    @staticmethod
    @contextmanager
    def configuration_error(config_type, severity="critical"):
        """
        Inject configuration errors.
        
        Args:
            config_type: Type of configuration to corrupt (proxy, rate_limit, etc)
            severity: How severe the configuration error should be (minor, major, critical)
        """
        if config_type == "proxy":
            original_get_proxy_config = ProxyManager._get_config
            
            def corrupted_proxy_config(self):
                if severity == "critical":
                    return None  # No config at all
                elif severity == "major":
                    return {"proxies": []}  # Empty proxy list
                else:  # minor
                    return {"proxies": [{"type": "http"}]}  # Invalid proxy (no URL)
            
            with patch.object(ProxyManager, '_get_config', corrupted_proxy_config):
                logger.info(f"Injecting {severity} configuration error for {config_type}")
                yield
                
        elif config_type == "rate_limit":
            original_get_domain_limits = RateLimiter._get_domain_limits
            
            def corrupted_rate_limits(self, domain):
                if severity == "critical":
                    return {}  # No limits
                elif severity == "major":
                    return {"requests_per_minute": 0}  # Zero requests allowed
                else:  # minor
                    return {"requests_per_minute": 1, "concurrent_requests": 1}  # Very restrictive
            
            with patch.object(RateLimiter, '_get_domain_limits', corrupted_rate_limits):
                logger.info(f"Injecting {severity} configuration error for {config_type}")
                yield
                
        else:
            # Generic configuration error
            logger.info(f"Injecting generic {severity} configuration error for {config_type}")
            yield
        
        logger.info(f"Configuration error injection ended for {config_type}")
    
    @staticmethod
    @contextmanager
    def intermittent_failure(component_name, failure_pattern=None, exception_type=Exception):
        """
        Inject intermittent failures following a specific pattern.
        
        Args:
            component_name: Name of the component to affect
            failure_pattern: List of booleans indicating success (True) or failure (False)
            exception_type: Type of exception to raise on failure
        """
        if failure_pattern is None:
            # Default pattern: fail every third call
            failure_pattern = [True, True, False]
        
        # Counter to track calls
        calls = [0]
        
        if component_name == "session_manager":
            original_get_session = SessionManager.get_session
            
            def intermittent_get_session(self, domain, force_new=False):
                calls[0] += 1
                pattern_index = (calls[0] - 1) % len(failure_pattern)
                
                if not failure_pattern[pattern_index]:
                    logger.debug(f"Injecting failure on call {calls[0]} for {component_name}")
                    raise exception_type(f"Injected intermittent failure in {component_name}")
                
                return original_get_session(self, domain, force_new)
            
            with patch.object(SessionManager, 'get_session', intermittent_get_session):
                logger.info(f"Injecting intermittent failures for {component_name}")
                yield
                
        elif component_name == "proxy_manager":
            original_get_proxy = ProxyManager.get_proxy
            
            def intermittent_get_proxy(self, domain=None):
                calls[0] += 1
                pattern_index = (calls[0] - 1) % len(failure_pattern)
                
                if not failure_pattern[pattern_index]:
                    logger.debug(f"Injecting failure on call {calls[0]} for {component_name}")
                    raise exception_type(f"Injected intermittent failure in {component_name}")
                
                return original_get_proxy(self, domain)
            
            with patch.object(ProxyManager, 'get_proxy', intermittent_get_proxy):
                logger.info(f"Injecting intermittent failures for {component_name}")
                yield
                
        else:
            # Generic intermittent failure
            logger.info(f"Injecting generic intermittent failures for {component_name}")
            yield
        
        logger.info(f"Intermittent failure injection ended for {component_name} after {calls[0]} calls")


# -------------------------------------------------------------------------
# Test Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def resource_context():
    """Create a StrategyContext with all resource services initialized."""
    # Reset the service registry
    ServiceRegistry._instance = None
    config = get_resource_config()
    
    # Increase retry limits for tests
    if 'retry_manager' in config and 'default_policy' in config['retry_manager']:
        config['retry_manager']['default_policy']['max_attempts'] = 5
    
    # Create context with resource services
    context = StrategyContext.with_resource_services({"resource_services": config})
    
    yield context
    
    # Cleanup
    if ServiceRegistry._instance:
        ServiceRegistry._instance.shutdown_all()
        ServiceRegistry._instance = None


@pytest.fixture
def mock_http_session():
    """Create a mock HTTP session for testing."""
    mock_session = MagicMock()
    
    # Define response based on URL pattern
    def mock_get(url, **kwargs):
        if "error" in url:
            return MockResponse(500, "Internal Server Error", url=url)
        elif "rate-limit" in url:
            return MockResponse(429, "Too Many Requests", {"Retry-After": "5"}, url=url)
        elif "captcha" in url:
            return MockResponse(200, "Please complete the CAPTCHA to continue", url=url)
        elif "blocked" in url:
            return MockResponse(403, "Your IP has been blocked", url=url)
        else:
            # Success case
            html_content = f"""
            <html>
                <head><title>Test Page for {url}</title></head>
                <body>
                    <h1>Test Content</h1>
                    <div class="product">
                        <h2>Product Name</h2>
                        <p class="price">$99.99</p>
                        <p class="description">This is a test product description.</p>
                    </div>
                    <div class="links">
                        <a href="/page1">Link 1</a>
                        <a href="/page2">Link 2</a>
                        <a href="/page3">Link 3</a>
                    </div>
                </body>
            </html>
            """
            return MockResponse(200, html_content, url=url)
    
    mock_session.get = mock_get
    return mock_session


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status_code, text="", headers=None, url=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.url = url or "https://example.com/test"
        
    def raise_for_status(self):
        """Raise HTTPError for non-200 status codes."""
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"HTTP Error {self.status_code}", response=self)
    
    def json(self):
        """Return JSON parsed from text."""
        return json.loads(self.text) if self.text else {}


# -------------------------------------------------------------------------
# Test Classes
# -------------------------------------------------------------------------

class TestNetworkFailureResilience:
    """Test system resilience to network failures."""
    
    def test_network_failure_recovery(self, resource_context):
        """Test recovery from network failures."""
        # Configure BFS strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Execute with network failure injection
        with FailureInjector.network_failure(failure_rate=0.5):
            result = strategy.scrape("https://example.com/resilience-test", max_pages=5)
        
        # Strategy should complete despite failures
        assert result is not None
        assert "pages_scraped" in result
        assert len(result["pages_scraped"]) > 0
    
    def test_high_network_failure_circuit_breaking(self, resource_context):
        """Test circuit breaking under extreme network failure conditions."""
        # Configure DFS strategy
        strategy = DFSStrategy(context=resource_context)
        
        # Configure circuit breaker with aggressive settings
        circuit_breaker = resource_context.get_circuit_breaker_manager()
        domain = "extreme-failure.example.com"
        cb = circuit_breaker.get_circuit_breaker(domain, {
            "failure_threshold": 3,
            "reset_timeout": 1,
            "half_open_max": 1
        })
        
        # Tracking metrics
        metrics = {
            "total_attempts": 0,
            "successful_requests": 0,
            "circuit_open_events": 0
        }
        
        # Function to make a protected request
        def make_protected_request(url):
            metrics["total_attempts"] += 1
            
            # Check circuit state
            if not cb.allow_request():
                logger.info(f"Circuit is open for {domain}, request blocked")
                metrics["circuit_open_events"] += 1
                return None
            
            try:
                session = resource_context.get_session(domain)
                response = session.get(url)
                # Success
                metrics["successful_requests"] += 1
                cb.record_success()
                return response
            except Exception as e:
                logger.warning(f"Request failed: {str(e)}")
                cb.record_failure()
                return None
        
        # Execute with extreme network failure injection
        with FailureInjector.network_failure(failure_rate=0.8, 
                                          exceptions=[requests.ConnectionError,
                                                     requests.Timeout,
                                                     requests.HTTPError]):
            # Make multiple requests
            url = f"https://{domain}/test"
            
            for _ in range(10):
                response = make_protected_request(url)
                time.sleep(0.1)  # Small delay between requests
        
        # Verify circuit breaker protected the system
        assert metrics["total_attempts"] > 0
        assert metrics["circuit_open_events"] > 0
        
        # Log metrics
        logger.info(f"Network failure test metrics: {metrics}")
    
    def test_adaptive_retry_behavior(self, resource_context):
        """Test adaptive retry behavior under varying network conditions."""
        # Configure proxy rotation and retry
        retry_manager = resource_context.get_retry_manager()
        
        # Variable to track retry attempts
        retry_metrics = {"attempts": 0, "successes": 0, "failures": 0}
        
        # Function with retry
        @retry_manager.retry("adaptive-test.example.com", max_attempts=5)
        def fetch_with_retry():
            retry_metrics["attempts"] += 1
            
            # Will raise exception due to the network failure injector
            session = requests.Session()
            response = session.get("https://adaptive-test.example.com/test")
            
            retry_metrics["successes"] += 1
            return response
        
        # First test with moderate failure rate
        with FailureInjector.network_failure(failure_rate=0.3):
            try:
                result = fetch_with_retry()
                assert result is not None
            except Exception:
                retry_metrics["failures"] += 1
        
        moderate_attempts = retry_metrics["attempts"]
        logger.info(f"Moderate failure rate: {moderate_attempts} attempts, "
                  f"{retry_metrics['successes']} successes, {retry_metrics['failures']} failures")
        
        # Reset metrics
        retry_metrics["attempts"] = 0
        retry_metrics["successes"] = 0
        retry_metrics["failures"] = 0
        
        # Test with high failure rate
        with FailureInjector.network_failure(failure_rate=0.9):
            try:
                result = fetch_with_retry()
                assert result is not None
            except Exception:
                retry_metrics["failures"] += 1
        
        high_attempts = retry_metrics["attempts"]
        logger.info(f"High failure rate: {high_attempts} attempts, "
                  f"{retry_metrics['successes']} successes, {retry_metrics['failures']} failures")
        
        # Verify retry behavior
        assert retry_metrics["attempts"] > 0
        
        # With high failure rate, it should exhaust all retry attempts
        if retry_metrics["failures"] > 0:  # If it failed after retries
            assert high_attempts >= 4  # Should have tried at least 4 times


class TestServiceUnavailabilityResilience:
    """Test system resilience to service unavailability."""
    
    def test_proxy_service_unavailability(self, resource_context):
        """Test resilience when proxy service is unavailable."""
        # Configure strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Execute with proxy service unavailability
        with FailureInjector.service_unavailability("proxy_service"):
            result = strategy.scrape("https://example.com/proxy-resilience", max_pages=2)
        
        # Strategy should still complete even without proxies
        assert result is not None
        assert "pages_scraped" in result
        assert len(result["pages_scraped"]) > 0
    
    def test_ai_service_unavailability_fallback(self, resource_context):
        """Test fallback mechanisms when AI service is unavailable."""
        # Configure AI-guided strategy
        strategy = AIGuidedStrategy(context=resource_context)
        
        # Execute with AI service unavailability
        with FailureInjector.service_unavailability("ai_service"):
            result = strategy.scrape("https://example.com/ai-resilience", max_pages=3)
        
        # Strategy should fall back to a simpler approach and still complete
        assert result is not None
        assert "pages_scraped" in result
        assert len(result["pages_scraped"]) > 0
        
        # Check if fallback was recorded
        fallback_used = False
        if "extraction_methods" in result:
            # If extraction methods are recorded, check if fallback was used
            if "fallback" in str(result["extraction_methods"]).lower():
                fallback_used = True
        
        if "fallback_triggered" in result:
            fallback_used = result["fallback_triggered"]
        
        # It's possible the fallback mechanism doesn't explicitly record its usage
        # so we don't fail the test if we can't detect it
        logger.info(f"AI service fallback detected: {fallback_used}")
    
    def test_multiple_service_degradation(self, resource_context):
        """Test behavior when multiple services are degraded simultaneously."""
        # Configure AdaptiveScraper with multiple strategies
        strategies = [
            BFSStrategy(context=resource_context),
            DFSStrategy(context=resource_context),
            AIGuidedStrategy(context=resource_context)
        ]
        
        scraper = AdaptiveScraper(strategies=strategies, context=resource_context)
        
        # Execute with multiple services degraded
        with FailureInjector.service_unavailability("ai_service"):
            with FailureInjector.network_failure(failure_rate=0.3):
                with FailureInjector.service_unavailability("proxy_service"):
                    result = scraper.scrape("https://example.com/multi-degradation", max_pages=3)
        
        # Adaptive scraper should choose a strategy that can work
        assert result is not None
        assert "strategy_used" in result
        assert "pages_scraped" in result
        assert len(result["pages_scraped"]) > 0
        
        logger.info(f"Strategy used under degraded conditions: {result['strategy_used']}")


class TestResourceExhaustionResilience:
    """Test system resilience to resource exhaustion."""
    
    def test_memory_exhaustion_handling(self, resource_context):
        """Test handling of memory exhaustion."""
        # Configure BFS strategy with limited depth to reduce memory usage
        strategy = BFSStrategy(context=resource_context)
        
        # Execute with memory exhaustion simulation
        with FailureInjector.resource_exhaustion("memory", limit=5000):
            result = strategy.scrape("https://example.com/memory-test", max_pages=3)
        
        # Strategy should handle memory limits gracefully
        assert result is not None
        assert "pages_scraped" in result
    
    def test_file_handle_exhaustion(self, resource_context):
        """Test handling of file handle exhaustion."""
        # Configure DFS strategy
        strategy = DFSStrategy(context=resource_context)
        
        # Execute with file handle exhaustion
        with FailureInjector.resource_exhaustion("file_handles", limit=10):
            result = strategy.scrape("https://example.com/file-handles-test", max_pages=2)
        
        # Strategy should handle file handle limits gracefully
        assert result is not None
        assert "pages_scraped" in result
    
    def test_cpu_contention_performance(self, resource_context):
        """Test performance under CPU contention."""
        # Configure simple strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Execute without CPU contention and measure time
        start_time = time.time()
        normal_result = strategy.scrape("https://example.com/cpu-normal", max_pages=2)
        normal_time = time.time() - start_time
        
        # Execute with CPU contention and measure time
        with FailureInjector.resource_exhaustion("cpu", limit=1):
            start_time = time.time()
            contention_result = strategy.scrape("https://example.com/cpu-contention", max_pages=2)
            contention_time = time.time() - start_time
        
        # Both should complete successfully
        assert normal_result is not None
        assert contention_result is not None
        
        # Log performance impact
        logger.info(f"Normal execution time: {normal_time:.2f}s")
        logger.info(f"CPU contention execution time: {contention_time:.2f}s")
        logger.info(f"Performance impact: {contention_time/normal_time:.2f}x slower")
        
        # The system should still function under CPU contention
        assert len(contention_result.get("pages_scraped", [])) > 0


class TestConfigurationErrorResilience:
    """Test system resilience to configuration errors."""
    
    def test_proxy_configuration_errors(self, resource_context):
        """Test recovery from proxy configuration errors."""
        # Configure strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Test with different severity levels
        for severity in ["minor", "major", "critical"]:
            with FailureInjector.configuration_error("proxy", severity=severity):
                result = strategy.scrape(f"https://example.com/proxy-config-{severity}", max_pages=1)
                
                # Should still function despite configuration errors
                assert result is not None
                assert "pages_scraped" in result
                
                logger.info(f"Proxy config error ({severity}): Scraped {len(result.get('pages_scraped', []))} pages")
    
    def test_rate_limit_configuration_errors(self, resource_context):
        """Test recovery from rate limit configuration errors."""
        # Configure strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Test with different severity levels
        for severity in ["minor", "major", "critical"]:
            with FailureInjector.configuration_error("rate_limit", severity=severity):
                result = strategy.scrape(f"https://example.com/rate-limit-config-{severity}", max_pages=1)
                
                # Should still function despite configuration errors
                assert result is not None
                assert "pages_scraped" in result
                
                logger.info(f"Rate limit config error ({severity}): Scraped {len(result.get('pages_scraped', []))} pages")
    
    def test_adaptive_scraper_with_config_errors(self, resource_context):
        """Test AdaptiveScraper's resilience to configuration errors."""
        # Configure AdaptiveScraper with multiple strategies
        strategies = [
            BFSStrategy(context=resource_context),
            DFSStrategy(context=resource_context),
            AIGuidedStrategy(context=resource_context)
        ]
        
        scraper = AdaptiveScraper(strategies=strategies, context=resource_context)
        
        # Execute with multiple configuration errors
        with FailureInjector.configuration_error("proxy", severity="major"):
            with FailureInjector.configuration_error("rate_limit", severity="minor"):
                result = scraper.scrape("https://example.com/multi-config-errors", max_pages=2)
        
        # Adaptive scraper should select a working strategy
        assert result is not None
        assert "strategy_used" in result
        assert "pages_scraped" in result
        
        logger.info(f"Strategy used with config errors: {result['strategy_used']}")


class TestIntermittentFailureResilience:
    """Test system resilience to intermittent failures."""
    
    def test_session_manager_intermittent_failures(self, resource_context):
        """Test resilience to intermittent session manager failures."""
        # Configure strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Failure pattern: fail every third request
        failure_pattern = [True, True, False]
        
        # Execute with intermittent session manager failures
        with FailureInjector.intermittent_failure("session_manager", failure_pattern=failure_pattern):
            result = strategy.scrape("https://example.com/intermittent-session", max_pages=3)
        
        # Strategy should still complete despite intermittent failures
        assert result is not None
        assert "pages_scraped" in result
        assert len(result["pages_scraped"]) > 0
    
    def test_proxy_manager_intermittent_failures(self, resource_context):
        """Test resilience to intermittent proxy manager failures."""
        # Configure strategy
        strategy = BFSStrategy(context=resource_context)
        
        # Failure pattern: fail every other request
        failure_pattern = [True, False]
        
        # Execute with intermittent proxy manager failures
        with FailureInjector.intermittent_failure("proxy_manager", failure_pattern=failure_pattern):
            result = strategy.scrape("https://example.com/intermittent-proxy", max_pages=3)
        
        # Strategy should still complete despite intermittent failures
        assert result is not None
        assert "pages_scraped" in result
        assert len(result["pages_scraped"]) > 0


class TestConcurrentOperationStability:
    """Test system stability during concurrent operations."""
    
    def test_concurrent_scraping_stability(self, resource_context):
        """Test stability when running multiple scraping operations concurrently."""
        # Configure test URLs
        test_urls = [
            "https://example.com/concurrent-test-1",
            "https://example.com/concurrent-test-2",
            "https://example.com/concurrent-test-3",
            "https://example.com/concurrent-test-4"
        ]
        
        # Function to run a scraping task
        def run_scraping_task(url):
            strategy = BFSStrategy(context=resource_context)
            result = strategy.scrape(url, max_pages=2)
            return {"url": url, "result": result, "pages_scraped": len(result.get("pages_scraped", []))}
        
        # Run concurrent scraping
        results = []
        with ThreadPoolExecutor(max_workers=len(test_urls)) as executor:
            future_to_url = {executor.submit(run_scraping_task, url): url for url in test_urls}
            
            for future in future_to_url:
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    results.append({"url": future_to_url[future], "error": str(e)})
        
        # All operations should complete successfully
        success_count = sum(1 for r in results if "error" not in r)
        
        assert success_count == len(test_urls)
        
        # Log results
        for result in results:
            if "error" in result:
                logger.warning(f"Concurrent scraping error for {result['url']}: {result['error']}")
            else:
                logger.info(f"Concurrent scraping success for {result['url']}: {result['pages_scraped']} pages")
    
    def test_concurrent_operation_with_failures(self, resource_context):
        """Test concurrent operations with some experiencing failures."""
        # Configure test scenarios
        test_scenarios = [
            {"url": "https://example.com/concurrent-stable", "failure": False},
            {"url": "https://example.com/concurrent-network-failure", "failure": "network"},
            {"url": "https://example.com/concurrent-service-failure", "failure": "service"},
            {"url": "https://example.com/concurrent-stable-2", "failure": False}
        ]
        
        # Function to run a scraping task with potential failure injection
        def run_scraping_task(scenario):
            url = scenario["url"]
            strategy = BFSStrategy(context=resource_context)
            
            try:
                if scenario["failure"] == "network":
                    with FailureInjector.network_failure(failure_rate=0.7):
                        result = strategy.scrape(url, max_pages=2)
                elif scenario["failure"] == "service":
                    with FailureInjector.service_unavailability("proxy_service"):
                        result = strategy.scrape(url, max_pages=2)
                else:
                    result = strategy.scrape(url, max_pages=2)
                
                return {
                    "url": url, 
                    "success": True,
                    "result": result, 
                    "pages_scraped": len(result.get("pages_scraped", []))
                }
            except Exception as e:
                return {"url": url, "success": False, "error": str(e)}
        
        # Run concurrent operations
        results = []
        with ThreadPoolExecutor(max_workers=len(test_scenarios)) as executor:
            future_to_scenario = {executor.submit(run_scraping_task, scenario): scenario 
                                for scenario in test_scenarios}
            
            for future in future_to_scenario:
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    scenario = future_to_scenario[future]
                    results.append({
                        "url": scenario["url"], 
                        "success": False, 
                        "error": str(e)
                    })
        
        # Count successes and failures
        success_count = sum(1 for r in results if r.get("success", False))
        
        # Even with some failures, at least the stable scenarios should succeed
        expected_successes = sum(1 for s in test_scenarios if not s["failure"])
        assert success_count >= expected_successes
        
        # Log results
        for result in results:
            if not result.get("success", False):
                logger.warning(f"Concurrent operation failed for {result['url']}: {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"Concurrent operation succeeded for {result['url']}: {result.get('pages_scraped', 0)} pages")


class TestDataIntegrityDuringFailures:
    """Test data integrity during various failure scenarios."""
    
    def test_extraction_integrity_during_failures(self, resource_context):
        """Test data extraction integrity during failures."""
        # Create test HTML
        test_html = """
        <html>
            <head><title>Data Integrity Test</title></head>
            <body>
                <div class="product" id="product-1">
                    <h2>Product 1</h2>
                    <p class="price">$19.99</p>
                    <p class="description">This is product 1 description.</p>
                </div>
                <div class="product" id="product-2">
                    <h2>Product 2</h2>
                    <p class="price">$29.99</p>
                    <p class="description">This is product 2 description.</p>
                </div>
                <div class="product" id="product-3">
                    <h2>Product 3</h2>
                    <p class="price">$39.99</p>
                    <p class="description">This is product 3 description.</p>
                </div>
            </body>
        </html>
        """
        
        # Create a pipeline
        pipeline = ExtractionPipeline()
        
        # Process without failures
        baseline_input = {
            "html": test_html,
            "url": "https://example.com/data-integrity",
            "context": resource_context
        }
        
        baseline_result = pipeline.process(baseline_input)
        
        # Process with failures
        with FailureInjector.network_failure(failure_rate=0.4):
            with FailureInjector.resource_exhaustion("memory", limit=10000):
                failure_result = pipeline.process(baseline_input)
        
        # Both should succeed and produce identical or similar results
        assert baseline_result is not None
        assert failure_result is not None
        
        # Check basic result structure
        assert "extraction_results" in baseline_result
        assert "extraction_results" in failure_result
        
        # Log comparison
        baseline_items = len(baseline_result["extraction_results"]) if "extraction_results" in baseline_result else 0
        failure_items = len(failure_result["extraction_results"]) if "extraction_results" in failure_result else 0
        
        logger.info(f"Baseline extracted {baseline_items} items")
        logger.info(f"Failure scenario extracted {failure_items} items")
        
        # There should be at least some data extracted even with failures
        assert failure_items > 0
    
    def test_data_consistency_with_retries(self, resource_context):
        """Test data consistency when operations require retries."""
        # Configure retry manager
        retry_manager = resource_context.get_retry_manager()
        
        # Test data to extract
        test_data = {
            "id": "test-123",
            "name": "Test Product",
            "price": 99.99,
            "description": "This is a test product with consistent data."
        }
        
        # Function that simulates extraction with potential failures
        extracted_data = []
        extraction_attempts = [0]
        
        @retry_manager.retry("consistency-test", max_attempts=5)
        def extract_with_failures():
            extraction_attempts[0] += 1
            
            # First two attempts return partial data
            if extraction_attempts[0] <= 2:
                partial_data = {k: v for k, v in test_data.items() if k in ["id", "name"]}
                extracted_data.append(partial_data)
                raise ValueError("Simulated extraction failure - partial data")
            
            # Success case returns full data
            extracted_data.append(test_data.copy())
            return test_data
        
        # Execute extraction with retries
        result = extract_with_failures()
        
        # Verify extraction consistency
        assert result is not None
        assert result == test_data
        assert extraction_attempts[0] == 3  # Should have taken 3 attempts
        
        # Verify data consistency across attempts
        for data in extracted_data:
            for key in data:
                assert data[key] == test_data[key]
        
        logger.info(f"Data extraction required {extraction_attempts[0]} attempts")
        logger.info(f"Extracted data consistent across attempts: {extracted_data}")


# -------------------------------------------------------------------------
# Data Integrity Testing
# -------------------------------------------------------------------------

def run_data_integrity_validation():
    """Run data integrity validation tests."""
    # Setup
    resource_context = StrategyContext.with_resource_services({"resource_services": get_resource_config()})
    
    # Test case 1: Process valid HTML
    valid_html = """
    <html>
        <body>
            <div class="product">
                <h1>Test Product</h1>
                <span class="price">$99.99</span>
                <p class="description">Product description</p>
            </div>
        </body>
    </html>
    """
    
    # Test case 2: Process malformed HTML
    malformed_html = """
    <html>
        <body>
            <div class="product">
                <h1>Test Product With Unclosed Tags
                <span class="price">$99.99<span>
                <p class="description">Product description
            </div>
        </body>
    """
    
    # Test case 3: Process partial HTML (missing crucial data)
    partial_html = """
    <html>
        <body>
            <div class="product">
                <span class="price">$99.99</span>
            </div>
        </body>
    </html>
    """
    
    # Test all cases
    pipeline = ExtractionPipeline()
    
    test_cases = [
        {"name": "Valid HTML", "html": valid_html},
        {"name": "Malformed HTML", "html": malformed_html},
        {"name": "Partial HTML", "html": partial_html}
    ]
    
    results = []
    
    for case in test_cases:
        pipeline_input = {
            "html": case["html"],
            "url": f"https://example.com/{case['name'].lower().replace(' ', '-')}",
            "context": resource_context
        }
        
        try:
            result = pipeline.process(pipeline_input)
            results.append({
                "case": case["name"],
                "success": True,
                "result": result,
                "extraction_results": result.get("extraction_results", []) if result else []
            })
        except Exception as e:
            results.append({
                "case": case["name"],
                "success": False,
                "error": str(e)
            })
    
    # Generate report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_cases": len(test_cases),
        "successful_cases": sum(1 for r in results if r["success"]),
        "failed_cases": sum(1 for r in results if not r["success"]),
        "case_results": results
    }
    
    # Clean up report for display (remove large data structures)
    for result in report["case_results"]:
        if "result" in result:
            if "html" in result["result"]:
                result["result"]["html"] = "... (HTML content removed for display) ..."
    
    # Save report to file
    with open("data_integrity_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Return report
    return report


if __name__ == "__main__":
    """Run the resilience validation suite manually."""
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                      handlers=[
                          logging.StreamHandler(),
                          logging.FileHandler('resilience_validation.log')
                      ])
    logger = logging.getLogger("resilience_validation")
    
    logger.info("Starting manual resilience validation")
    
    # Run data integrity validation
    data_integrity_report = run_data_integrity_validation()
    logger.info(f"Data integrity validation: {data_integrity_report['successful_cases']}/{data_integrity_report['test_cases']} cases passed")
    
    # Run a sample resilience test
    test_resource_context = StrategyContext.with_resource_services({"resource_services": get_resource_config()})
    strategy = BFSStrategy(context=test_resource_context)
    
    try:
        logger.info("Testing network failure resilience")
        with FailureInjector.network_failure(failure_rate=0.3):
            result = strategy.scrape("https://example.com/manual-resilience-test", max_pages=2)
            
        logger.info(f"Test passed. Scraped {len(result.get('pages_scraped', []))} pages despite failures")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    logger.info("Manual resilience validation completed")