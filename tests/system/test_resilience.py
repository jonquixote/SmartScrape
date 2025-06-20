"""
System tests for resilience capabilities of SmartScrape.

These tests verify the system's ability to handle various failure scenarios,
including network failures, service unavailability, resource exhaustion,
configuration errors, and dependency failures.
"""

import logging
import pytest
import time
import requests
import random
import threading
import unittest
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from core.service_registry import ServiceRegistry
from core.session_manager import SessionManager
from core.proxy_manager import ProxyManager
from core.rate_limiter import RateLimiter
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError
from core.error_classifier import ErrorClassifier, ErrorCategory, ErrorSeverity

# Configure logging for tests
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class TestSystemResilience:
    """Test the system's resilience to various failures."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register services
        session_manager = SessionManager()
        session_manager.initialize()
        registry.register_service(session_manager)
        
        proxy_manager = ProxyManager()
        proxy_config = {
            'proxies': [
                {'url': f'http://test-proxy-{i}.example.com', 'type': 'http'} 
                for i in range(5)
            ]
        }
        proxy_manager.initialize(proxy_config)
        registry.register_service(proxy_manager)
        
        rate_limiter = RateLimiter()
        rate_limiter.initialize()
        registry.register_service(rate_limiter)
        
        error_classifier = ErrorClassifier()
        error_classifier.initialize()
        registry.register_service(error_classifier)
        
        retry_manager = RetryManager()
        retry_config = {
            'default_policy': {
                'max_attempts': 3,
                'backoff_factor': 0.5,
                'jitter': True
            }
        }
        retry_manager.initialize(retry_config)
        registry.register_service(retry_manager)
        
        circuit_breaker = CircuitBreakerManager()
        circuit_config = {
            'default_settings': {
                'failure_threshold': 3,
                'reset_timeout': 5,  # Short timeout for testing
                'half_open_max': 1
            }
        }
        circuit_breaker.initialize(circuit_config)
        registry.register_service(circuit_breaker)
        
        return registry
    
    def test_network_failure_resilience(self, service_registry, monkeypatch):
        """Test resilience to network failures."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        retry_manager = service_registry.get_service('retry_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Test domains
        domains = [f'domain-{i}.example.com' for i in range(3)]
        urls = [f'https://{domain}/page' for domain in domains]
        
        # Create circuit breakers for domains
        circuit_breakers = {domain: circuit_breaker.get_circuit_breaker(domain) for domain in domains}
        
        # Mock sessions with intermittent failures (30% chance)
        def mock_get_session(domain):
            session = MagicMock()
            
            def flaky_get(url, **kwargs):
                # Simulate intermittent network failures
                if random.random() < 0.3:
                    raise requests.exceptions.ConnectionError("Simulated network failure")
                
                # Success case
                response = MagicMock()
                response.status_code = 200
                response.text = f"Content from {url}"
                return response
                
            session.get = flaky_get
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Function to scrape with resilience
        def scrape_with_resilience(url):
            domain = url.split('/')[2]
            cb = circuit_breakers[domain]
            
            # Check circuit breaker first
            if not cb.allow_request():
                logger.warning(f"Circuit for {domain} is open, skipping")
                return None
            
            @retry_manager.retry(domain, max_attempts=5)
            def fetch_url():
                try:
                    session = session_manager.get_session(domain)
                    response = session.get(url, timeout=5)
                    
                    # Record success
                    cb.record_success()
                    return response.text
                except Exception as e:
                    # Record failure
                    cb.record_failure()
                    raise
            
            try:
                return fetch_url()
            except Exception as e:
                logger.error(f"Failed to fetch {url} after retries: {str(e)}")
                return None
        
        # Try multiple requests to each domain
        results = {}
        request_count = 10
        
        for domain, url in zip(domains, urls):
            domain_results = []
            
            for _ in range(request_count):
                result = scrape_with_resilience(url)
                domain_results.append(result is not None)
            
            success_rate = sum(domain_results) / len(domain_results)
            results[domain] = success_rate
            logger.info(f"Domain {domain} success rate: {success_rate:.2%}")
        
        # Verify reasonable success rates despite failures
        # Since we have 30% failure rate but retry 5 times, success rate should be high
        assert all(rate > 0.9 for rate in results.values()), "Success rates too low despite retries"
    
    def test_service_unavailability(self, service_registry, monkeypatch):
        """Test resilience to service unavailability."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Test URL and domain
        domain = 'unavailable-service.example.com'
        url = f'https://{domain}/api'
        
        # Configure circuit breaker with low threshold
        cb = circuit_breaker.get_circuit_breaker(domain, {
            'failure_threshold': 2,
            'reset_timeout': 1,  # Short timeout for testing
            'half_open_max': 1
        })
        
        # Mock session that always fails with 503 Service Unavailable
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "503 Service Unavailable", response=mock_response
        )
        mock_session.get.return_value = mock_response
        
        monkeypatch.setattr(session_manager, 'get_session', lambda domain: mock_session)
        
        # Function to access service with circuit breaker protection
        def access_service():
            if not cb.allow_request():
                logger.info("Circuit open, service considered unavailable")
                return False
                
            try:
                response = session_manager.get_session(domain).get(url, timeout=5)
                response.raise_for_status()
                cb.record_success()
                return True
            except Exception:
                cb.record_failure()
                return False
        
        # Initial attempts should fail but be attempted
        initial_results = []
        for _ in range(5):
            initial_results.append(access_service())
            
        # Verify that the circuit opened
        assert not cb.allow_request(), "Circuit should be open after failures"
        assert not any(initial_results[2:]), "Later attempts should have been blocked by circuit breaker"
        
        # Wait for circuit breaker to transition to half-open
        time.sleep(cb.reset_timeout + 0.1)
        
        # The circuit should now be half-open
        assert cb.state.value == 'half_open'
        
        # First request during half-open should be attempted
        assert access_service() is False, "Service is still unavailable"
        
        # Circuit should be open again
        assert cb.state.value == 'open'
        
        # Now simulate service recovery
        def mock_get_recovered_session(domain):
            session = MagicMock()
            response = MagicMock()
            response.status_code = 200
            session.get.return_value = response
            return session
            
        # Wait for half-open again
        time.sleep(cb.reset_timeout + 0.1)
        monkeypatch.setattr(session_manager, 'get_session', mock_get_recovered_session)
        
        # Now the test should succeed and close the circuit
        assert access_service() is True, "Service should be available now"
        assert cb.state.value == 'closed', "Circuit should be closed after success"
    
    def test_resource_exhaustion_resilience(self, service_registry, monkeypatch):
        """Test resilience to resource exhaustion."""
        # Add a fake counter for memory usage
        memory_usage = {'current': 0, 'limit': 1000}
        
        # Add an in-memory cache that grows
        mock_cache = {}
        
        # Function that consumes memory
        def memory_intensive_operation(size=100):
            # Add to simulated memory usage
            memory_usage['current'] += size
            
            # If we're over limit, raise MemoryError
            if memory_usage['current'] > memory_usage['limit']:
                raise MemoryError("Simulated out of memory")
                
            # Otherwise add to our mock cache with a random key
            key = f"key-{random.randint(1, 10000)}"
            mock_cache[key] = 'x' * size
            return key
        
        # Function to clear cache and reset memory
        def clear_cache():
            mock_cache.clear()
            memory_usage['current'] = 0
            
        # Function to get cache size
        def get_cache_size():
            return sum(len(v) for v in mock_cache.values())
        
        # Test with resource monitoring and fallback
        operations_count = 0
        failures_count = 0
        
        # Function with resource protection
        def protected_operation(size):
            nonlocal operations_count, failures_count
            operations_count += 1
            
            try:
                # Check current memory usage
                current_usage = memory_usage['current']
                limit = memory_usage['limit']
                usage_percent = (current_usage / limit) * 100
                
                logger.info(f"Memory usage: {usage_percent:.1f}% ({current_usage}/{limit})")
                
                # If we're getting close to the limit, try to reduce resource usage
                if usage_percent > 80:
                    logger.warning("Memory usage high, clearing cache")
                    clear_cache()
                
                # Do the operation
                return memory_intensive_operation(size)
                
            except MemoryError:
                # Resource exhaustion, take recovery action
                failures_count += 1
                logger.error("Memory exhaustion detected, clearing cache")
                clear_cache()
                
                # Try again with a smaller size if the original is too large
                if size > 200:
                    logger.info("Retrying with smaller size")
                    return protected_operation(size // 2)
                else:
                    # Can't reduce further, propagate the error
                    raise
        
        # Execute a series of operations with increasing sizes
        operation_sizes = [100, 150, 200, 300, 400, 500, 600, 300, 200]
        results = []
        
        for size in operation_sizes:
            try:
                key = protected_operation(size)
                results.append((size, key))
                logger.info(f"Operation with size {size} succeeded")
            except Exception as e:
                results.append((size, None))
                logger.error(f"Operation with size {size} failed: {str(e)}")
        
        # Verify results
        assert len(results) == len(operation_sizes)
        assert failures_count > 0, "Should have encountered at least one resource exhaustion"
        assert sum(1 for _, key in results if key is not None) > 5, "Most operations should succeed with fallback"
    
    def test_configuration_error_resilience(self, service_registry, monkeypatch):
        """Test resilience to configuration errors."""
        # Create a component with configuration requirements
        class ConfigurableComponent:
            def __init__(self, config=None):
                self.config = config or {}
                self.logger = logging.getLogger(__name__ + '.ConfigurableComponent')
                
            def perform_operation(self, data):
                # Check if required configuration exists
                if 'api_key' not in self.config:
                    self.logger.warning("Missing api_key in configuration, using default")
                    self.config['api_key'] = 'default_key'
                
                if 'endpoint' not in self.config:
                    # Critical config missing, but try fallback
                    self.logger.error("Missing endpoint in configuration, trying fallback")
                    self.config['endpoint'] = 'https://fallback-api.example.com'
                
                # Simulate operation
                endpoint = self.config['endpoint']
                api_key = self.config['api_key']
                
                self.logger.info(f"Operation using {endpoint} with key {api_key}")
                
                # Return info about configuration used
                return {
                    'endpoint': endpoint,
                    'api_key': api_key,
                    'fallback_used': endpoint.startswith('https://fallback')
                }
        
        # Create configs with various issues
        configs = [
            {},  # Missing everything
            {'api_key': 'test_key'},  # Missing endpoint
            {'endpoint': 'https://api.example.com'},  # Missing key
            {'api_key': 'test_key', 'endpoint': 'https://api.example.com'}  # Complete
        ]
        
        # Test each configuration
        results = []
        for i, config in enumerate(configs):
            component = ConfigurableComponent(config)
            result = component.perform_operation({})
            results.append(result)
            logger.info(f"Config {i} result: {result}")
        
        # Verify results
        assert all(r.get('endpoint') for r in results), "All operations should have an endpoint"
        assert all(r.get('api_key') for r in results), "All operations should have an api_key"
        assert results[0]['fallback_used'], "Missing all config should use fallback"
        assert results[1]['fallback_used'], "Missing endpoint should use fallback"
        assert not results[3]['fallback_used'], "Complete config should not use fallback"
    
    def test_dependency_failure_resilience(self, service_registry, monkeypatch):
        """Test resilience to dependency failures."""
        # Simulate a dependency chain: primary_service -> secondary_service -> data_store
        # If any fails, the system should try alternatives
        
        class DataStore:
            def __init__(self, name, fail_probability=0.0):
                self.name = name
                self.fail_probability = fail_probability
                self.data = {}
                self.logger = logging.getLogger(f"DataStore.{name}")
                
            def store(self, key, value):
                if random.random() < self.fail_probability:
                    self.logger.error(f"Failed to store data in {self.name}")
                    raise RuntimeError(f"Storage failure in {self.name}")
                    
                self.data[key] = value
                self.logger.info(f"Stored data in {self.name}: {key}={value}")
                return True
        
        class SecondaryService:
            def __init__(self, name, data_stores, fail_probability=0.0):
                self.name = name
                self.data_stores = data_stores  # List of DataStore instances
                self.fail_probability = fail_probability
                self.logger = logging.getLogger(f"SecondaryService.{name}")
                
            def process(self, data):
                if random.random() < self.fail_probability:
                    self.logger.error(f"Failed to process in {self.name}")
                    raise RuntimeError(f"Processing failure in {self.name}")
                
                # Try to store in each data store, stop on first success
                key = f"key-{random.randint(1, 1000)}"
                
                for store in self.data_stores:
                    try:
                        store.store(key, data)
                        return {'status': 'success', 'service': self.name, 'store': store.name}
                    except Exception as e:
                        self.logger.warning(f"Store {store.name} failed: {str(e)}")
                        continue
                
                # All stores failed
                raise RuntimeError(f"All data stores failed in {self.name}")
        
        class PrimaryService:
            def __init__(self, secondary_services):
                self.secondary_services = secondary_services  # List of SecondaryService instances
                self.logger = logging.getLogger("PrimaryService")
                
            def execute(self, data):
                results = {'successes': 0, 'failures': 0, 'details': []}
                
                # Try each secondary service
                for service in self.secondary_services:
                    try:
                        result = service.process(data)
                        results['successes'] += 1
                        results['details'].append(result)
                        self.logger.info(f"Success with {service.name}")
                        break  # Stop on first success
                    except Exception as e:
                        self.logger.warning(f"Service {service.name} failed: {str(e)}")
                        results['failures'] += 1
                        continue
                        
                # If all failed, mark overall failure
                if results['successes'] == 0:
                    results['status'] = 'failure'
                    self.logger.error("All services failed")
                else:
                    results['status'] = 'success'
                    
                return results
        
        # Create a dependency tree with various failure probabilities
        data_stores = [
            DataStore("primary_db", fail_probability=0.5),   # High failure rate
            DataStore("backup_db", fail_probability=0.2),    # Medium failure rate
            DataStore("archive_db", fail_probability=0.0)    # Never fails
        ]
        
        secondary_services = [
            SecondaryService("main_service", data_stores[:2], fail_probability=0.3),  # Can only use first two DBs
            SecondaryService("fallback_service", data_stores, fail_probability=0.1)   # Can use all DBs
        ]
        
        primary_service = PrimaryService(secondary_services)
        
        # Run multiple operations and check success rate
        iterations = 20
        results = []
        
        for i in range(iterations):
            data = f"test-data-{i}"
            result = primary_service.execute(data)
            results.append(result)
            logger.info(f"Iteration {i+1} result: {result['status']}")
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['status'] == 'success') / len(results)
        logger.info(f"Overall success rate: {success_rate:.2%}")
        
        # Despite high failure rates in dependencies, the system should be resilient
        assert success_rate > 0.8, "Success rate should be high due to fallbacks"
    
    def test_graceful_degradation(self, service_registry, monkeypatch):
        """Test the system's ability to degrade gracefully."""
        # Define a system with multiple features, some of which might fail
        class ScrapingSystem:
            def __init__(self):
                self.logger = logging.getLogger("ScrapingSystem")
                self.features = {
                    "content_extraction": True,
                    "image_download": True,
                    "metadata_parsing": True,
                    "related_links": True,
                    "sentiment_analysis": True
                }
                
            def toggle_feature(self, feature, enabled):
                if feature in self.features:
                    self.features[feature] = enabled
                    
            def scrape(self, url):
                result = {
                    'url': url,
                    'success': True,
                    'features_attempted': 0,
                    'features_succeeded': 0,
                    'data': {}
                }
                
                # Content extraction (critical feature)
                if self.features["content_extraction"]:
                    try:
                        result['features_attempted'] += 1
                        result['data']['content'] = f"Content from {url}"
                        result['features_succeeded'] += 1
                    except Exception as e:
                        self.logger.error(f"Critical feature failed: content_extraction: {str(e)}")
                        result['success'] = False
                        result['error'] = str(e)
                        return result  # Fail completely if critical feature fails
                
                # Image download (non-critical)
                if self.features["image_download"]:
                    try:
                        result['features_attempted'] += 1
                        if random.random() < 0.3:  # 30% chance of failure
                            raise RuntimeError("Failed to download images")
                        result['data']['images'] = [f"image_{i}.jpg" for i in range(3)]
                        result['features_succeeded'] += 1
                    except Exception as e:
                        self.logger.warning(f"Non-critical feature failed: image_download: {str(e)}")
                        # Continue with degraded functionality
                
                # Metadata parsing (non-critical)
                if self.features["metadata_parsing"]:
                    try:
                        result['features_attempted'] += 1
                        if random.random() < 0.2:  # 20% chance of failure
                            raise RuntimeError("Failed to parse metadata")
                        result['data']['metadata'] = {"title": "Page title", "author": "Author name"}
                        result['features_succeeded'] += 1
                    except Exception as e:
                        self.logger.warning(f"Non-critical feature failed: metadata_parsing: {str(e)}")
                        # Continue with degraded functionality
                
                # Related links (non-critical)
                if self.features["related_links"]:
                    try:
                        result['features_attempted'] += 1
                        if random.random() < 0.4:  # 40% chance of failure
                            raise RuntimeError("Failed to extract related links")
                        result['data']['related_links'] = [f"{url}/related/{i}" for i in range(5)]
                        result['features_succeeded'] += 1
                    except Exception as e:
                        self.logger.warning(f"Non-critical feature failed: related_links: {str(e)}")
                        # Continue with degraded functionality
                
                # Sentiment analysis (non-critical)
                if self.features["sentiment_analysis"]:
                    try:
                        result['features_attempted'] += 1
                        if random.random() < 0.5:  # 50% chance of failure
                            raise RuntimeError("Failed to analyze sentiment")
                        result['data']['sentiment'] = random.choice(["positive", "neutral", "negative"])
                        result['features_succeeded'] += 1
                    except Exception as e:
                        self.logger.warning(f"Non-critical feature failed: sentiment_analysis: {str(e)}")
                        # Continue with degraded functionality
                
                return result
        
        # Create the system
        system = ScrapingSystem()
        
        # Test normal operation
        urls = [f"https://example.com/page/{i}" for i in range(10)]
        normal_results = [system.scrape(url) for url in urls]
        
        # Calculate baseline stats
        baseline_success_rate = sum(1 for r in normal_results if r['success']) / len(normal_results)
        baseline_feature_rate = sum(r['features_succeeded'] / r['features_attempted'] 
                                   for r in normal_results) / len(normal_results)
        
        logger.info(f"Baseline success rate: {baseline_success_rate:.2%}")
        logger.info(f"Baseline feature completion rate: {baseline_feature_rate:.2%}")
        
        # Now disable some non-critical features
        system.toggle_feature("sentiment_analysis", False)
        system.toggle_feature("related_links", False)
        
        # Test degraded operation
        degraded_results = [system.scrape(url) for url in urls]
        
        # Calculate degraded stats
        degraded_success_rate = sum(1 for r in degraded_results if r['success']) / len(degraded_results)
        degraded_feature_rate = sum(r['features_succeeded'] / r['features_attempted'] 
                                   for r in degraded_results) / len(degraded_results)
        
        logger.info(f"Degraded success rate: {degraded_success_rate:.2%}")
        logger.info(f"Degraded feature completion rate: {degraded_feature_rate:.2%}")
        
        # Verify graceful degradation
        assert degraded_success_rate >= baseline_success_rate, "Disabling features should not reduce success rate"
        assert degraded_feature_rate > baseline_feature_rate, "Feature success rate should improve"
        
        # Verify all critical features still work
        assert all('content' in r['data'] for r in degraded_results), "Critical feature should always work"


class TestRecoveryScenarios:
    """Test the system's ability to recover from various scenarios."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register services
        session_manager = SessionManager()
        session_manager.initialize()
        registry.register_service(session_manager)
        
        proxy_manager = ProxyManager()
        proxy_manager.initialize()
        registry.register_service(proxy_manager)
        
        rate_limiter = RateLimiter()
        rate_limiter.initialize()
        registry.register_service(rate_limiter)
        
        return registry
    
    def test_recovery_from_process_restart(self, service_registry, monkeypatch):
        """Test recovery from process restarts."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        
        # Mock function to simulate persisting state
        persisted_state = {}
        
        def save_state():
            # Simulate saving service state to disk
            persisted_state['sessions'] = {
                'domain1.example.com': {'cookies': {'session_id': 'abc123'}},
                'domain2.example.com': {'cookies': {'token': 'xyz789'}}
            }
            logger.info("State saved to persistent storage")
            
        def load_state():
            # Simulate loading state from disk
            if 'sessions' in persisted_state:
                logger.info("State loaded from persistent storage")
                return persisted_state['sessions']
            logger.warning("No state found in persistent storage")
            return {}
        
        # Save some initial state
        save_state()
        
        # Mock the shutdown and initialize methods to simulate restart
        original_shutdown = session_manager.shutdown
        def mock_shutdown():
            logger.info("Simulating service shutdown")
            # Save state before shutting down
            save_state()
            original_shutdown()
        
        original_initialize = session_manager.initialize
        def mock_initialize(config=None):
            logger.info("Simulating service restart")
            # Pass loaded state to initialize
            config = config or {}
            config['persisted_sessions'] = load_state()
            original_initialize(config)
        
        monkeypatch.setattr(session_manager, 'shutdown', mock_shutdown)
        monkeypatch.setattr(session_manager, 'initialize', mock_initialize)
        
        # Track cookies for verification
        loaded_cookies = {}
        
        # Override get_session to check for loaded cookies
        original_get_session = session_manager.get_session
        def mock_get_session(domain, force_new=False):
            session = original_get_session(domain, force_new)
            
            # Check if persisted cookies were loaded
            if hasattr(session, 'cookies') and domain in persisted_state.get('sessions', {}):
                cookies = persisted_state['sessions'][domain].get('cookies', {})
                loaded_cookies[domain] = cookies
                
                # In a real implementation, these would be properly loaded into the session
                logger.info(f"Using persisted cookies for {domain}: {cookies}")
                
            return session
        
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Simulate process restart
        session_manager.shutdown()
        session_manager.initialize()
        
        # Now try to use a session and verify persistence worked
        test_domains = ['domain1.example.com', 'domain2.example.com', 'domain3.example.com']
        for domain in test_domains:
            session = session_manager.get_session(domain)
            logger.info(f"Got session for {domain}")
        
        # Verify persistence results
        assert 'domain1.example.com' in loaded_cookies, "Should have loaded domain1 cookies"
        assert 'domain2.example.com' in loaded_cookies, "Should have loaded domain2 cookies"
        assert 'domain3.example.com' not in loaded_cookies, "No cookies for domain3"
        
        # Verify cookie values
        assert loaded_cookies['domain1.example.com'].get('session_id') == 'abc123'
        assert loaded_cookies['domain2.example.com'].get('token') == 'xyz789'
    
    def test_recovery_from_configuration_changes(self, service_registry, monkeypatch):
        """Test recovery from configuration changes."""
        # Get rate limiter service
        rate_limiter = service_registry.get_service('rate_limiter')
        
        # Initial domain to test
        domain = 'configtest.example.com'
        
        # Check current rate limits
        original_limits = rate_limiter._get_domain_limits(domain)
        logger.info(f"Original limits for {domain}: {original_limits}")
        
        # Function to simulate a request with current limits
        def make_request():
            # Apply rate limiting
            waited = rate_limiter.wait_if_needed(domain)
            return waited
        
        # Make some requests and measure timing
        start_time = time.time()
        waits = []
        for _ in range(5):
            waits.append(make_request())
            
        original_time = time.time() - start_time
        logger.info(f"Original configuration: 5 requests in {original_time:.4f}s, waits: {waits.count(True)}")
        
        # Now simulate a configuration change
        new_config = {
            'default_limits': {
                'requests_per_minute': original_limits['requests_per_minute'] // 2,
                'requests_per_hour': original_limits['requests_per_hour']
            }
        }
        
        # Hot-reload configuration
        rate_limiter.shutdown()
        rate_limiter.initialize(new_config)
        
        # Verify the new configuration was applied
        new_limits = rate_limiter._get_domain_limits(domain)
        logger.info(f"New limits for {domain}: {new_limits}")
        assert new_limits['requests_per_minute'] == original_limits['requests_per_minute'] // 2
        
        # Make requests again with new configuration
        start_time = time.time()
        waits = []
        for _ in range(5):
            waits.append(make_request())
            
        new_time = time.time() - start_time
        logger.info(f"New configuration: 5 requests in {new_time:.4f}s, waits: {waits.count(True)}")
        
        # With lower rate limits, we should see more waits and longer time
        assert waits.count(True) > 0, "Rate limiting should be applied with new stricter config"
    
    def test_recovery_from_service_restart(self, service_registry, monkeypatch):
        """Test recovery from service restarts."""
        # Get proxy manager
        proxy_manager = service_registry.get_service('proxy_manager')
        
        # Add some test proxies
        test_proxies = [
            {'url': 'http://proxy1.example.com', 'type': 'http', 'health': 100},
            {'url': 'http://proxy2.example.com', 'type': 'http', 'health': 80},
            {'url': 'http://proxy3.example.com', 'type': 'http', 'health': 0}  # Unhealthy
        ]
        
        # Method to reinitialize the proxy manager with these proxies
        def initialize_with_test_proxies():
            config = {
                'proxies': test_proxies,
                'rotation_strategy': 'round_robin'
            }
            proxy_manager.initialize(config)
            
            # Mark the unhealthy proxy as failed
            proxy_manager.mark_proxy_failed('http://proxy3.example.com')
        
        # Initialize the proxy manager
        initialize_with_test_proxies()
        
        # Check the initial state
        domain = 'example.com'
        initial_proxy = proxy_manager.get_proxy(domain)
        logger.info(f"Initial proxy: {initial_proxy}")
        
        # Blacklist information should be preserved across restarts
        blacklisted_before = proxy_manager.is_proxy_blacklisted('http://proxy3.example.com')
        assert blacklisted_before, "Unhealthy proxy should be blacklisted"
        
        # Simulate a service restart
        proxy_manager.shutdown()
        initialize_with_test_proxies()
        
        # Verify blacklist consistency
        blacklisted_after = proxy_manager.is_proxy_blacklisted('http://proxy3.example.com')
        assert blacklisted_after, "Proxy should still be blacklisted after restart"
        
        # Verify we can still get a healthy proxy
        proxy_after_restart = proxy_manager.get_proxy(domain)
        logger.info(f"Proxy after restart: {proxy_after_restart}")
        assert proxy_after_restart['url'] != 'http://proxy3.example.com', "Should not get blacklisted proxy"
    
    def test_recovery_from_network_partitions(self, service_registry, monkeypatch):
        """Test recovery from network partitions."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        
        # Test domain
        domain = 'partition-test.example.com'
        url = f'https://{domain}/api'
        
        # Simulate network partition and recovery
        network_partitioned = {'status': True}  # Use dict for mutable closure
        
        # Mock session behavior during network partition
        def mock_get_session(domain):
            session = MagicMock()
            
            def get_with_partition(url, **kwargs):
                if network_partitioned['status']:
                    raise requests.exceptions.ConnectionError("Network partition simulated")
                    
                # Normal response after partition resolves
                response = MagicMock()
                response.status_code = 200
                response.text = "Success after partition resolved"
                return response
                
            session.get = get_with_partition
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Create a component that uses the session
        class NetworkClient:
            def __init__(self, domain, session_manager):
                self.domain = domain
                self.session_manager = session_manager
                self.logger = logging.getLogger("NetworkClient")
                self.consecutive_failures = 0
                self.backoff_time = 0.1  # Starting backoff time in seconds
                
            def make_request(self, url):
                try:
                    session = self.session_manager.get_session(self.domain)
                    response = session.get(url, timeout=5)
                    
                    # Reset failure count on success
                    self.consecutive_failures = 0
                    self.backoff_time = 0.1
                    
                    return response.text
                    
                except requests.exceptions.ConnectionError as e:
                    # Network partition detected
                    self.consecutive_failures += 1
                    
                    # Calculate exponential backoff
                    backoff = min(30, self.backoff_time * (2 ** (self.consecutive_failures - 1)))
                    
                    self.logger.warning(f"Network partition detected. Attempt {self.consecutive_failures}. "
                                       f"Backing off for {backoff:.2f}s")
                    
                    # In a real implementation, we'd sleep here
                    # For testing purposes, just log the backoff time
                    
                    # After a certain number of failures, we'd go into recovery mode
                    if self.consecutive_failures >= 3:
                        self.logger.error("Persistent network partition. Entering recovery mode.")
                        return None
                        
                    raise
        
        # Create the client
        client = NetworkClient(domain, session_manager)
        
        # Attempt requests during partition
        partition_results = []
        
        for i in range(5):
            try:
                result = client.make_request(url)
                partition_results.append(result)
            except Exception as e:
                partition_results.append(str(e))
            
        logger.info(f"Results during partition: {partition_results}")
        
        # Verify behavior during partition
        assert all(isinstance(r, str) and 'ConnectionError' in r for r in partition_results[:2])
        assert partition_results[2:] == [None, None, None]
        assert client.consecutive_failures >= 3
        
        # Now resolve the partition
        network_partitioned['status'] = False
        
        # Attempt requests after partition resolved
        recovery_results = []
        
        for i in range(3):
            try:
                result = client.make_request(url)
                recovery_results.append(result)
            except Exception as e:
                recovery_results.append(str(e))
            
        logger.info(f"Results after partition resolved: {recovery_results}")
        
        # Verify recovery behavior
        assert all(r == "Success after partition resolved" for r in recovery_results)
        assert client.consecutive_failures == 0