"""
End-to-end tests for SmartScrape resource management flow.

Tests the complete request flow through SessionManager, ProxyManager, RateLimiter,
RetryManager, and CircuitBreaker components, as well as performance benchmarks
for these components.
"""

import logging
import pytest
import time
import requests
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, PropertyMock

from core.service_registry import ServiceRegistry
from core.session_manager import SessionManager
from core.proxy_manager import ProxyManager
from core.rate_limiter import RateLimiter
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreakerManager, OpenCircuitError

# Configure logging for tests
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class TestResourceManagementFlow:
    """Test end-to-end flow through all resource management components."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register all necessary services
        session_manager = SessionManager()
        session_manager.initialize()
        registry.register_service(session_manager)
        
        proxy_manager = ProxyManager()
        proxy_config = {
            'proxies': [
                {'url': f'http://test-proxy-{i}.example.com', 'type': 'http'} 
                for i in range(10)
            ]
        }
        proxy_manager.initialize(proxy_config)
        registry.register_service(proxy_manager)
        
        rate_limiter = RateLimiter()
        rate_config = {
            'default_limits': {
                'requests_per_minute': 30,
                'requests_per_hour': 300,
                'concurrent_requests': 5
            },
            'domain_limits': {
                'httpbin.org': {
                    'requests_per_minute': 10,
                    'requests_per_hour': 60
                }
            }
        }
        rate_limiter.initialize(rate_config)
        registry.register_service(rate_limiter)
        
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
                'failure_threshold': 5,
                'reset_timeout': 30,
                'half_open_max': 1
            }
        }
        circuit_breaker.initialize(circuit_config)
        registry.register_service(circuit_breaker)
        
        return registry
    
    def test_complete_request_flow(self, service_registry, monkeypatch):
        """Test a complete request flow through all resource management components."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        proxy_manager = service_registry.get_service('proxy_manager')
        rate_limiter = service_registry.get_service('rate_limiter')
        retry_manager = service_registry.get_service('retry_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Test domain and URL
        domain = 'httpbin.org'
        url = f'https://{domain}/get'
        
        # Mock the session to avoid actual network requests
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true, "origin": "test-proxy"}'
        mock_response.json.return_value = {"success": True, "origin": "test-proxy"}
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        
        monkeypatch.setattr(session_manager, 'get_session', lambda *args, **kwargs: mock_session)
        
        # Set up monitoring for component interactions
        flow_metrics = {
            'rate_limiter_called': False,
            'proxy_manager_called': False,
            'circuit_breaker_check': False,
            'circuit_success_recorded': False,
            'retry_attempts': 0
        }
        
        # Monitor rate limiter interaction
        original_wait = rate_limiter.wait_if_needed
        def monitor_rate_limiter(domain):
            flow_metrics['rate_limiter_called'] = True
            return original_wait(domain)
        monkeypatch.setattr(rate_limiter, 'wait_if_needed', monitor_rate_limiter)
        
        # Monitor proxy manager interaction
        original_get_proxy = proxy_manager.get_proxy
        def monitor_proxy_manager(domain):
            flow_metrics['proxy_manager_called'] = True
            return original_get_proxy(domain)
        monkeypatch.setattr(proxy_manager, 'get_proxy', monitor_proxy_manager)
        
        # Monitor circuit breaker interaction
        original_cb_get = circuit_breaker.get_circuit_breaker
        cb_instance = original_cb_get(domain)
        
        original_allow = cb_instance.allow_request
        def monitor_cb_allow():
            flow_metrics['circuit_breaker_check'] = True
            return original_allow()
        monkeypatch.setattr(cb_instance, 'allow_request', monitor_cb_allow)
        
        original_success = cb_instance.record_success
        def monitor_cb_success():
            flow_metrics['circuit_success_recorded'] = True
            return original_success()
        monkeypatch.setattr(cb_instance, 'record_success', monitor_cb_success)
        
        # Monitor retry manager
        original_retry_decorator = retry_manager.retry
        def monitor_retry_decorator(*args, **kwargs):
            decorator = original_retry_decorator(*args, **kwargs)
            
            def instrumented_decorator(func):
                wrapped = decorator(func)
                
                def wrapper(*args, **kwargs):
                    try:
                        return wrapped(*args, **kwargs)
                    except Exception as e:
                        flow_metrics['retry_attempts'] += 1
                        raise
                
                return wrapper
            
            return instrumented_decorator
        
        monkeypatch.setattr(retry_manager, 'retry', monitor_retry_decorator)
        
        # Perform the request with all components involved
        def make_request():
            # Get circuit breaker for domain
            cb = circuit_breaker.get_circuit_breaker(domain)
            
            # Check if circuit is closed (allowing requests)
            if not cb.allow_request():
                logger.warning(f"Circuit is open for {domain}, skipping request")
                return None
            
            # Apply rate limiting
            rate_limiter.wait_if_needed(domain)
            
            # Get proxy for the request
            proxy = proxy_manager.get_proxy(domain)
            proxy_url = proxy['url']
            
            @retry_manager.retry(domain)
            def fetch_with_retry():
                # Get session
                session = session_manager.get_session(domain)
                
                # Make the request with proxy
                response = session.get(url, proxies={
                    'http': proxy_url,
                    'https': proxy_url
                }, timeout=5)
                
                # Verify successful response
                response.raise_for_status()
                
                # Record success in circuit breaker
                cb.record_success()
                
                return response
            
            # Execute the request with retry handling
            return fetch_with_retry()
        
        # Make the request
        response = make_request()
        
        # Verify response
        assert response is not None
        assert response.status_code == 200
        
        # Verify all components were used
        assert flow_metrics['rate_limiter_called'], "Rate limiter should be called"
        assert flow_metrics['proxy_manager_called'], "Proxy manager should be called"
        assert flow_metrics['circuit_breaker_check'], "Circuit breaker should be checked"
        assert flow_metrics['circuit_success_recorded'], "Circuit success should be recorded"
        assert flow_metrics['retry_attempts'] == 0, "No retries should be needed for successful request"
        
        # Test with a failure scenario that triggers retry
        def simulate_temporary_failure(*args, **kwargs):
            if getattr(simulate_temporary_failure, 'calls', 0) < 2:
                # First two calls fail
                simulate_temporary_failure.calls = getattr(simulate_temporary_failure, 'calls', 0) + 1
                raise requests.exceptions.ConnectionError("Simulated connection error")
            # Third call succeeds
            return mock_response
        
        simulate_temporary_failure.calls = 0
        mock_session.get.side_effect = simulate_temporary_failure
        
        # Reset metrics
        flow_metrics['retry_attempts'] = 0
        flow_metrics['circuit_success_recorded'] = False
        
        # Make request with retries
        response = make_request()
        
        # Verify response and metrics
        assert response is not None
        assert response.status_code == 200
        assert flow_metrics['retry_attempts'] == 2, "Should have retried twice"
        assert flow_metrics['circuit_success_recorded'], "Circuit success should be recorded after retries"
    
    def test_rate_limited_sites(self, service_registry, monkeypatch):
        """Test system behavior against rate-limited sites."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        rate_limiter = service_registry.get_service('rate_limiter')
        
        # Test domain and URL
        domain = 'rate-limited.example.com'
        url = f'https://{domain}/api/data'
        
        # Configure mock session with rate limiting behavior
        request_count = {'count': 0}
        
        def mock_get(*args, **kwargs):
            request_count['count'] += 1
            
            # Return 429 every 3rd request
            if request_count['count'] % 3 == 0:
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.text = "Too Many Requests"
                mock_response.headers = {
                    'Retry-After': '5',  # Suggests 5 second wait
                    'X-RateLimit-Limit': '2',
                    'X-RateLimit-Remaining': '0',
                    'X-RateLimit-Reset': str(int(time.time()) + 5)
                }
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                    "429 Too Many Requests", response=mock_response
                )
                return mock_response
            else:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = '{"success": true, "data": "test"}'
                mock_response.json.return_value = {"success": True, "data": "test"}
                return mock_response
                
        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        
        monkeypatch.setattr(session_manager, 'get_session', lambda *args, **kwargs: mock_session)
        
        # Track rate limiter behavior
        rate_limit_waits = {'count': 0}
        reported_rate_limits = {'count': 0}
        
        original_wait = rate_limiter.wait_if_needed
        def mock_wait(domain):
            rate_limit_waits['count'] += 1
            # Apply normal wait logic but with shorter times for testing
            result = original_wait(domain)
            return result
            
        original_report = rate_limiter.report_rate_limited
        def mock_report(domain):
            reported_rate_limits['count'] += 1
            # Call original but track calls
            return original_report(domain)
            
        monkeypatch.setattr(rate_limiter, 'wait_if_needed', mock_wait)
        monkeypatch.setattr(rate_limiter, 'report_rate_limited', mock_report)
        
        # Make multiple requests to test rate limiting behavior
        results = []
        
        for i in range(10):
            try:
                # Apply rate limit wait
                rate_limiter.wait_if_needed(domain)
                
                # Make request
                response = mock_session.get(url)
                
                # Process successful response
                if response.status_code == 200:
                    results.append({"status": "success", "data": response.json()})
                
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response.status_code == 429:
                    # Report rate limit to adjust future behavior
                    rate_limiter.report_rate_limited(domain)
                    results.append({"status": "rate_limited"})
                else:
                    results.append({"status": "error", "message": str(e)})
        
        # Analyze results
        successes = sum(1 for r in results if r['status'] == 'success')
        rate_limits = sum(1 for r in results if r['status'] == 'rate_limited')
        
        logger.info(f"Rate limiting test results:")
        logger.info(f"Successful requests: {successes}")
        logger.info(f"Rate limited requests: {rate_limits}")
        logger.info(f"Rate limiter waits: {rate_limit_waits['count']}")
        logger.info(f"Rate limit reports: {reported_rate_limits['count']}")
        
        # Verify rate limiting behavior
        assert rate_limits > 0, "Should encounter some rate limits"
        assert reported_rate_limits['count'] == rate_limits, "Each rate limit should be reported"
        assert rate_limit_waits['count'] == 10, "Rate limiter should be consulted before each request"
        
        # Check if rate limits were properly adapted
        domain_limits = rate_limiter._limits.get(domain, {})
        assert domain_limits, "Domain-specific rate limits should be set after encountering 429s"
        
        # The original domain shouldn't be in rate_limiter._limits, so this means limits were added
        logger.info(f"Adjusted rate limits: {domain_limits}")
    
    def test_proxy_rotation(self, service_registry, monkeypatch):
        """Test proxy rotation for different scenarios."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        proxy_manager = service_registry.get_service('proxy_manager')
        
        # Test domain
        domain = 'proxy-test.example.com'
        url = f'https://{domain}/api/data'
        
        # Mock session behavior
        def mock_get(request_url, proxies=None, **kwargs):
            proxy_url = proxies.get('http', 'default') if proxies else 'default'
            
            # If proxy URL contains "fail", simulate failure
            if 'fail' in proxy_url:
                raise requests.exceptions.ProxyError(f"Proxy error: {proxy_url}")
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f'{{"success": true, "proxy": "{proxy_url}"}}'
            mock_response.json.return_value = {"success": True, "proxy": proxy_url}
            return mock_response
            
        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        
        monkeypatch.setattr(session_manager, 'get_session', lambda *args, **kwargs: mock_session)
        
        # Configure proxy list with some failing proxies
        proxies = []
        for i in range(10):
            # Every third proxy will fail
            if i % 3 == 0:
                proxies.append({
                    'url': f'http://test-proxy-fail-{i}.example.com', 
                    'type': 'http'
                })
            else:
                proxies.append({
                    'url': f'http://test-proxy-{i}.example.com', 
                    'type': 'http'
                })
        
        proxy_config = {'proxies': proxies}
        proxy_manager.initialize(proxy_config)
        
        # Track proxy selection
        proxy_usage = {}
        proxy_failures = set()
        
        # Helper function to make request with proxy rotation
        def make_request_with_proxy():
            max_attempts = 3
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    # Get proxy for the request
                    proxy = proxy_manager.get_proxy(domain)
                    proxy_url = proxy['url']
                    
                    # Track usage
                    proxy_usage[proxy_url] = proxy_usage.get(proxy_url, 0) + 1
                    
                    # Make request with proxy
                    response = mock_session.get(url, proxies={
                        'http': proxy_url,
                        'https': proxy_url
                    }, timeout=5)
                    
                    return {
                        "success": True,
                        "proxy": proxy_url,
                        "data": response.json()
                    }
                    
                except requests.exceptions.ProxyError as e:
                    # Track failed proxy
                    proxy_failures.add(proxy_url)
                    logger.warning(f"Proxy {proxy_url} failed: {str(e)}")
                    
                    # Mark proxy as failed
                    proxy_manager.mark_proxy_failed(proxy_url)
                    
                    # Try again
                    attempt += 1
                
                except Exception as e:
                    logger.error(f"Request error: {str(e)}")
                    attempt += 1
            
            return {"success": False, "error": "Max attempts reached"}
        
        # Make multiple requests
        results = []
        for _ in range(20):
            result = make_request_with_proxy()
            results.append(result)
        
        # Analyze results
        successes = sum(1 for r in results if r['success'])
        
        logger.info(f"Proxy rotation test results:")
        logger.info(f"Successful requests: {successes}/{len(results)}")
        logger.info(f"Unique proxies used: {len(proxy_usage)}")
        logger.info(f"Proxy failures encountered: {len(proxy_failures)}")
        logger.info(f"Proxy usage distribution: {proxy_usage}")
        
        # Verify proxy rotation behavior
        assert successes == len(results), "All requests should eventually succeed with proxy rotation"
        assert len(proxy_usage) > 1, "Multiple proxies should be used"
        assert len(proxy_failures) > 0, "Some proxy failures should be detected"
        
        # Check that failed proxies aren't reused
        blacklisted_proxies = proxy_manager._blacklisted_proxies
        for failed_proxy in proxy_failures:
            assert failed_proxy in blacklisted_proxies, f"Failed proxy {failed_proxy} should be blacklisted"
    
    def test_circuit_breaker_protection(self, service_registry, monkeypatch):
        """Test circuit breaker protection for failing services."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Test domains - one stable, one flaky, one completely down
        domains = ['stable.example.com', 'flaky.example.com', 'down.example.com']
        
        # Mock session behavior specific to each domain
        def mock_get_session(domain, force_new=False):
            session = MagicMock()
            
            def domain_aware_get(url, **kwargs):
                if 'stable' in domain:
                    # Always succeeds
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.text = '{"success": true}'
                    return mock_response
                    
                elif 'flaky' in domain:
                    # Fails 80% of the time
                    if random.random() < 0.8:
                        raise requests.exceptions.ConnectionError("Simulated connection error")
                    
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.text = '{"success": true}'
                    return mock_response
                    
                else:  # 'down' domain
                    # Always fails
                    raise requests.exceptions.ConnectionError("Service unavailable")
                    
            session.get = domain_aware_get
            return session
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Configure circuit breakers with smaller thresholds for testing
        circuit_config = {
            'default_settings': {
                'failure_threshold': 3,  # Open circuit after 3 failures
                'reset_timeout': 5,      # Try again after 5 seconds
                'half_open_max': 1
            }
        }
        circuit_breaker.initialize(circuit_config)
        
        # Track circuit breaker state transitions
        circuit_states = {domain: [] for domain in domains}
        
        # Helper function to make request with circuit breaker protection
        def make_protected_request(domain):
            url = f"https://{domain}/api"
            
            # Get circuit breaker for domain
            cb = circuit_breaker.get_circuit_breaker(domain)
            
            # Record current state
            current_state = cb.state
            circuit_states[domain].append(current_state.value)
            
            # Check if circuit allows request
            if not cb.allow_request():
                logger.warning(f"Circuit open for {domain}, request blocked")
                return {
                    "success": False,
                    "circuit_state": current_state.value,
                    "error": "Circuit open"
                }
            
            try:
                # Get session and make request
                session = session_manager.get_session(domain)
                response = session.get(url)
                
                # Record success
                cb.record_success()
                
                return {
                    "success": True,
                    "circuit_state": current_state.value,
                    "status_code": response.status_code
                }
                
            except Exception as e:
                # Record failure
                cb.record_failure()
                
                return {
                    "success": False,
                    "circuit_state": current_state.value,
                    "error": str(e)
                }
        
        # Make multiple requests to each domain
        domain_results = {}
        
        for domain in domains:
            results = []
            for _ in range(10):
                result = make_protected_request(domain)
                results.append(result)
                time.sleep(0.1)  # Small delay between requests
            
            domain_results[domain] = results
        
        # Wait for reset timeout to test half-open state
        logger.info("Waiting for circuit reset timeout...")
        time.sleep(6)  # Just over the reset_timeout
        
        # Make additional requests to test recovery
        for domain in domains:
            for _ in range(5):
                result = make_protected_request(domain)
                domain_results[domain].append(result)
                time.sleep(0.1)
        
        # Analyze results for each domain
        for domain in domains:
            results = domain_results[domain]
            successes = sum(1 for r in results if r['success'])
            circuit_open_count = sum(1 for r in results if r.get('error') == "Circuit open")
            
            logger.info(f"Circuit breaker test results for {domain}:")
            logger.info(f"Total requests: {len(results)}")
            logger.info(f"Successful requests: {successes}")
            logger.info(f"Requests blocked by open circuit: {circuit_open_count}")
            logger.info(f"Circuit state transitions: {circuit_states[domain]}")
        
        # Verify circuit breaker behavior for each domain type
        
        # Stable domain should always succeed with closed circuit
        stable_results = domain_results['stable.example.com']
        assert all(r['success'] for r in stable_results), "Stable domain should have all successful requests"
        assert all(s == 'closed' for s in circuit_states['stable.example.com']), "Circuit should remain closed for stable domain"
        
        # Down domain should quickly open circuit and block requests
        down_results = domain_results['down.example.com']
        assert any(r.get('error') == "Circuit open" for r in down_results), "Circuit should open for down domain"
        assert 'open' in circuit_states['down.example.com'], "Circuit should transition to open for down domain"
        
        # Flaky domain should show mixed behavior with circuit opening and testing half-open state
        flaky_results = domain_results['flaky.example.com']
        assert any(r.get('error') == "Circuit open" for r in flaky_results), "Circuit should sometimes open for flaky domain"
        assert 'open' in circuit_states['flaky.example.com'], "Circuit should sometimes be open for flaky domain"
        
        # After waiting, half-open state should be seen
        assert 'half_open' in circuit_states['down.example.com'] or 'half_open' in circuit_states['flaky.example.com'], \
            "At least one domain should show half-open state after reset timeout"


class TestResourcePerformance:
    """Test performance metrics of resource management components."""
    
    @pytest.fixture
    def service_registry(self):
        """Set up service registry with all required services."""
        registry = ServiceRegistry()
        
        # Initialize and register all necessary services
        session_manager = SessionManager()
        session_manager.initialize()
        registry.register_service(session_manager)
        
        proxy_manager = ProxyManager()
        proxy_config = {
            'proxies': [
                {'url': f'http://test-proxy-{i}.example.com', 'type': 'http'} 
                for i in range(20)  # More proxies for performance testing
            ]
        }
        proxy_manager.initialize(proxy_config)
        registry.register_service(proxy_manager)
        
        rate_limiter = RateLimiter()
        rate_config = {
            'default_limits': {
                'requests_per_minute': 120,  # Higher limits for performance testing
                'requests_per_hour': 1200,
                'concurrent_requests': 10
            }
        }
        rate_limiter.initialize(rate_config)
        registry.register_service(rate_limiter)
        
        retry_manager = RetryManager()
        retry_config = {
            'default_policy': {
                'max_attempts': 3,
                'backoff_factor': 0.1,  # Smaller backoff for testing
                'jitter': True
            }
        }
        retry_manager.initialize(retry_config)
        registry.register_service(retry_manager)
        
        circuit_breaker = CircuitBreakerManager()
        circuit_config = {
            'default_settings': {
                'failure_threshold': 5,
                'reset_timeout': 10,
                'half_open_max': 1
            }
        }
        circuit_breaker.initialize(circuit_config)
        registry.register_service(circuit_breaker)
        
        return registry
    
    def test_session_reuse_efficiency(self, service_registry, monkeypatch):
        """Test efficiency of session reuse vs creating new sessions."""
        # Get session manager
        session_manager = service_registry.get_service('session_manager')
        
        # Mock session creation to be measurable
        creation_time = 0.05  # Simulated time to create a new session
        reuse_overhead = 0.001  # Small overhead for session reuse
        
        sessions = {}
        
        def mock_get_session(domain, force_new=False):
            # Simulate session creation time
            if domain not in sessions or force_new:
                time.sleep(creation_time)  # Simulate expensive creation
                sessions[domain] = MagicMock()
            else:
                time.sleep(reuse_overhead)  # Small overhead for reuse
            
            return sessions[domain]
            
        monkeypatch.setattr(session_manager, 'get_session', mock_get_session)
        
        # Test domains
        domains = ['example1.com', 'example2.com', 'example3.com']
        
        # Benchmark always creating new sessions
        start_time = time.time()
        for _ in range(30):  # 10 requests per domain
            for domain in domains:
                session_manager.get_session(domain, force_new=True)
        new_session_time = time.time() - start_time
        
        # Reset for next test
        sessions.clear()
        
        # Benchmark session reuse
        start_time = time.time()
        for _ in range(30):  # 10 requests per domain
            for domain in domains:
                session_manager.get_session(domain)
        reuse_session_time = time.time() - start_time
        
        # Calculate improvement ratio
        improvement_ratio = new_session_time / reuse_session_time if reuse_session_time > 0 else float('inf')
        
        logger.info(f"Session reuse performance test results:")
        logger.info(f"Time with new sessions: {new_session_time:.4f}s")
        logger.info(f"Time with session reuse: {reuse_session_time:.4f}s")
        logger.info(f"Performance improvement: {improvement_ratio:.2f}x")
        
        # Assert significant improvement with session reuse
        assert improvement_ratio > 5, "Session reuse should be at least 5x faster than creating new sessions"
    
    def test_retry_impact_on_throughput(self, service_registry, monkeypatch):
        """Test impact of retries on overall throughput."""
        # Get services
        session_manager = service_registry.get_service('session_manager')
        retry_manager = service_registry.get_service('retry_manager')
        
        # Test domain
        domain = 'retry-test.example.com'
        url = f'https://{domain}/api/data'
        
        # Configure mock session with different failure rates
        def create_session_with_failure_rate(failure_rate):
            session = MagicMock()
            
            def get_with_failures(*args, **kwargs):
                if random.random() < failure_rate:
                    raise requests.exceptions.ConnectionError("Simulated connection error")
                
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.text = '{"success": true}'
                return mock_response
                
            session.get = get_with_failures
            return session
            
        # Configure different retry policies
        retry_policies = [
            {'max_attempts': 1, 'backoff_factor': 0},  # No retries
            {'max_attempts': 3, 'backoff_factor': 0.1},  # Default retries
            {'max_attempts': 5, 'backoff_factor': 0.2}   # Aggressive retries
        ]
        
        # Test failure rates
        failure_rates = [0.1, 0.3, 0.5, 0.7]
        
        results = {}
        
        for failure_rate in failure_rates:
            policy_results = {}
            
            for policy in retry_policies:
                # Configure mock session for this test
                mock_session = create_session_with_failure_rate(failure_rate)
                monkeypatch.setattr(session_manager, 'get_session', lambda *args, **kwargs: mock_session)
                
                # Track metrics
                attempts = {'count': 0}
                successes = {'count': 0}
                
                # Create test function
                @retry_manager.retry(domain, **policy)
                def test_function():
                    attempts['count'] += 1
                    response = session_manager.get_session(domain).get(url)
                    successes['count'] += 1
                    return response
                
                # Measure throughput
                start_time = time.time()
                
                # Make requests
                test_results = []
                for _ in range(50):
                    try:
                        response = test_function()
                        test_results.append(True)
                    except Exception:
                        test_results.append(False)
                
                elapsed_time = time.time() - start_time
                
                # Calculate metrics
                success_rate = sum(test_results) / len(test_results)
                throughput = len(test_results) / elapsed_time
                efficiency = successes['count'] / attempts['count'] if attempts['count'] > 0 else 0
                
                policy_results[f"max_attempts={policy['max_attempts']}"] = {
                    'success_rate': success_rate,
                    'throughput': throughput,
                    'efficiency': efficiency,
                    'total_time': elapsed_time,
                    'total_attempts': attempts['count'],
                    'successful_operations': successes['count']
                }
                
            results[f"failure_rate={failure_rate}"] = policy_results
        
        # Log results
        logger.info("Retry impact on throughput test results:")
        
        for failure_scenario, policy_results in results.items():
            logger.info(f"\n{failure_scenario}")
            
            for policy_name, metrics in policy_results.items():
                logger.info(f"  {policy_name}:")
                logger.info(f"    Success rate: {metrics['success_rate']:.2%}")
                logger.info(f"    Throughput: {metrics['throughput']:.2f} req/s")
                logger.info(f"    Efficiency: {metrics['efficiency']:.2%}")
                logger.info(f"    Total time: {metrics['total_time']:.2f}s")
        
        # Analyze tradeoffs
        for failure_rate in failure_rates:
            scenario_results = results[f"failure_rate={failure_rate}"]
            
            if failure_rate < 0.5:
                # For lower failure rates, more retries should improve success rate
                no_retry_success = scenario_results['max_attempts=1']['success_rate']
                default_retry_success = scenario_results['max_attempts=3']['success_rate']
                assert default_retry_success > no_retry_success, \
                    f"With {failure_rate:.0%} failure rate, retries should improve success rate"
            
            # Check throughput tradeoff 
            no_retry_throughput = scenario_results['max_attempts=1']['throughput']
            aggressive_throughput = scenario_results['max_attempts=5']['throughput']
            
            logger.info(f"Failure rate {failure_rate:.0%}: No-retry throughput: {no_retry_throughput:.2f} req/s, "
                       f"Aggressive retry throughput: {aggressive_throughput:.2f} req/s")
    
    def test_proxy_rotation_performance(self, service_registry, monkeypatch):
        """Test performance impact of proxy rotation strategies."""
        # Get proxy manager
        proxy_manager = service_registry.get_service('proxy_manager')
        
        # Configure proxy list
        proxies = []
        for i in range(20):
            # Every 5th proxy will be slow
            is_slow = (i % 5 == 0)
            proxies.append({
                'url': f'http://test-proxy-{i}.example.com',
                'type': 'http',
                'is_slow': is_slow
            })
        
        proxy_config = {'proxies': proxies}
        proxy_manager.initialize(proxy_config)
        
        # Mock proxy selection to simulate performance characteristics
        def get_proxy_performance(proxy):
            # Simulate network latency based on proxy characteristics
            if proxy.get('is_slow', False):
                return 0.1  # 100ms for slow proxies
            return 0.02  # 20ms for fast proxies
        
        # Test domains
        domains = ['example1.com', 'example2.com', 'example3.com']
        
        # Test different rotation strategies
        strategies = [
            "random",       # Pure random selection
            "round_robin",  # Cycle through proxies
            "performance"   # Prefer faster proxies
        ]
        
        results = {}
        
        for strategy in strategies:
            # Configure proxy manager to use this strategy
            proxy_manager._rotation_strategy = strategy
            
            # Track metrics
            proxy_usage = {}
            total_latency = 0
            
            # Simulate requests with this strategy
            start_time = time.time()
            
            for _ in range(100):  # 100 requests
                for domain in domains:
                    # Get proxy according to strategy
                    proxy = proxy_manager.get_proxy(domain)
                    proxy_url = proxy['url']
                    
                    # Track usage
                    proxy_usage[proxy_url] = proxy_usage.get(proxy_url, 0) + 1
                    
                    # Simulate latency
                    latency = get_proxy_performance(proxy)
                    total_latency += latency
                    time.sleep(latency)  # Actually wait to measure real impact
            
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            slow_proxy_usage = sum(proxy_usage.get(f'http://test-proxy-{i}.example.com', 0) 
                                 for i in range(20) if i % 5 == 0)
            slow_proxy_percentage = slow_proxy_usage / 300 if slow_proxy_usage > 0 else 0
            
            unique_proxies = len(proxy_usage)
            throughput = 300 / elapsed_time if elapsed_time > 0 else 0
            
            results[strategy] = {
                'total_time': elapsed_time,
                'throughput': throughput,
                'unique_proxies_used': unique_proxies,
                'slow_proxy_percentage': slow_proxy_percentage,
                'proxy_usage': proxy_usage
            }
        
        # Log results
        logger.info("Proxy rotation performance test results:")
        
        for strategy, metrics in results.items():
            logger.info(f"\nStrategy: {strategy}")
            logger.info(f"  Total time: {metrics['total_time']:.2f}s")
            logger.info(f"  Throughput: {metrics['throughput']:.2f} req/s")
            logger.info(f"  Unique proxies used: {metrics['unique_proxies_used']}")
            logger.info(f"  Slow proxy usage: {metrics['slow_proxy_percentage']:.2%}")
        
        # Compare strategies
        random_throughput = results['random']['throughput']
        performance_throughput = results['performance']['throughput']
        
        # Performance-based selection should be more efficient
        logger.info(f"Performance improvement ratio: {performance_throughput / random_throughput:.2f}x")
    
    def test_circuit_breaker_overhead(self, service_registry, monkeypatch):
        """Test overhead of circuit breaker protection."""
        # Get circuit breaker manager
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Function to test without circuit breaker
        def request_without_protection(domain, should_fail=False):
            if should_fail:
                raise Exception("Simulated failure")
            return "Success"
        
        # Function to test with circuit breaker
        def request_with_protection(domain, should_fail=False):
            # Get circuit breaker
            cb = circuit_breaker.get_circuit_breaker(domain)
            
            # Check circuit state
            if not cb.allow_request():
                raise OpenCircuitError(f"Circuit {domain} is open")
            
            try:
                if should_fail:
                    raise Exception("Simulated failure")
                
                # Record success
                cb.record_success()
                return "Success"
            except Exception as e:
                # Record failure
                cb.record_failure()
                raise
        
        # Test domains
        domains = ['overhead-test-1.com', 'overhead-test-2.com', 'overhead-test-3.com']
        
        # Measure baseline performance without circuit breaker
        start_time = time.time()
        for _ in range(1000):
            for domain in domains:
                try:
                    request_without_protection(domain)
                except Exception:
                    pass
        baseline_time = time.time() - start_time
        
        # Measure performance with circuit breaker
        start_time = time.time()
        for _ in range(1000):
            for domain in domains:
                try:
                    request_with_protection(domain)
                except OpenCircuitError:
                    pass
                except Exception:
                    pass
        protected_time = time.time() - start_time
        
        # Calculate overhead
        overhead_ratio = protected_time / baseline_time if baseline_time > 0 else float('inf')
        overhead_percentage = (overhead_ratio - 1) * 100
        
        logger.info(f"Circuit breaker overhead test results:")
        logger.info(f"Time without circuit breaker: {baseline_time:.4f}s")
        logger.info(f"Time with circuit breaker: {protected_time:.4f}s")
        logger.info(f"Overhead ratio: {overhead_ratio:.2f}x ({overhead_percentage:.2f}%)")
        
        # Test with failures to see actual protection value
        failure_percentage = 0.2  # 20% of requests will fail
        
        # Measure baseline performance with failures but no circuit breaker
        start_time = time.time()
        for i in range(1000):
            for domain in domains:
                should_fail = (random.random() < failure_percentage)
                try:
                    request_without_protection(domain, should_fail)
                except Exception:
                    pass
        baseline_failure_time = time.time() - start_time
        
        # Measure performance with failures and circuit breaker
        start_time = time.time()
        for i in range(1000):
            for domain in domains:
                should_fail = (random.random() < failure_percentage)
                try:
                    request_with_protection(domain, should_fail)
                except OpenCircuitError:
                    pass
                except Exception:
                    pass
        protected_failure_time = time.time() - start_time
        
        # Calculate overhead with failures
        failure_overhead_ratio = protected_failure_time / baseline_failure_time if baseline_failure_time > 0 else float('inf')
        
        logger.info(f"Circuit breaker overhead with {failure_percentage:.0%} failures:")
        logger.info(f"Time without circuit breaker: {baseline_failure_time:.4f}s")
        logger.info(f"Time with circuit breaker: {protected_failure_time:.4f}s")
        logger.info(f"Overhead ratio: {failure_overhead_ratio:.2f}x")
        
        # Verify reasonable overhead
        assert overhead_ratio < 1.5, "Circuit breaker overhead should be less than 50%"
    
    def test_system_throughput(self, service_registry, monkeypatch):
        """Test overall system throughput with all components enabled."""
        # Get all services
        session_manager = service_registry.get_service('session_manager')
        proxy_manager = service_registry.get_service('proxy_manager')
        rate_limiter = service_registry.get_service('rate_limiter')
        retry_manager = service_registry.get_service('retry_manager')
        circuit_breaker = service_registry.get_service('circuit_breaker_manager')
        
        # Mock session behavior
        def mock_get(url, proxies=None, **kwargs):
            # Extract domain from URL
            domain = url.split('://', 1)[1].split('/', 1)[0] if '://' in url else 'unknown'
            
            # Simulate different sites
            if 'throttled' in domain:
                time.sleep(0.1)  # Simulate slower response
            elif 'flaky' in domain:
                if random.random() < 0.3:
                    raise requests.exceptions.ConnectionError("Random connection error")
                
            # Normal response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f'{{"success": true, "domain": "{domain}"}}'
            return mock_response
            
        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        
        monkeypatch.setattr(session_manager, 'get_session', lambda *args, **kwargs: mock_session)
        
        # Function that uses all components
        def make_resilient_request(url):
            # Extract domain
            domain = url.split('://', 1)[1].split('/', 1)[0] if '://' in url else 'unknown'
            
            # Get circuit breaker
            cb = circuit_breaker.get_circuit_breaker(domain)
            
            # Check circuit state
            if not cb.allow_request():
                return {
                    "success": False,
                    "error": "Circuit open"
                }
            
            # Apply rate limiting
            rate_limiter.wait_if_needed(domain)
            
            # Get proxy
            proxy = proxy_manager.get_proxy(domain)
            proxy_url = proxy['url']
            
            @retry_manager.retry(domain)
            def execute_request():
                # Get session
                session = session_manager.get_session(domain)
                
                # Make request
                response = session.get(url, proxies={
                    'http': proxy_url,
                    'https': proxy_url
                })
                
                # Check for errors
                response.raise_for_status()
                
                # Record success
                cb.record_success()
                
                return {
                    "success": True,
                    "status_code": response.status_code
                }
            
            try:
                return execute_request()
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Test URLs across different domains
        test_urls = [
            f"https://normal{i}.example.com/api/data" for i in range(5)
        ] + [
            f"https://throttled{i}.example.com/api/data" for i in range(3)
        ] + [
            f"https://flaky{i}.example.com/api/data" for i in range(2)
        ]
        
        # Measure throughput with all protections enabled
        start_time = time.time()
        results = []
        
        for _ in range(3):  # Multiple passes
            for url in test_urls:
                result = make_resilient_request(url)
                results.append(result)
        
        protected_time = time.time() - start_time
        protected_throughput = len(results) / protected_time if protected_time > 0 else 0
        
        # Calculate success rate
        success_rate = sum(1 for r in results if r['success']) / len(results)
        
        logger.info(f"System throughput test results:")
        logger.info(f"Total requests: {len(results)}")
        logger.info(f"Successful requests: {sum(1 for r in results if r['success'])}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Total time: {protected_time:.2f}s")
        logger.info(f"Throughput with protections: {protected_throughput:.2f} req/s")
        
        # Measure throughput without protections (direct requests)
        start_time = time.time()
        direct_results = []
        
        for _ in range(3):  # Same number of passes
            for url in test_urls:
                try:
                    response = mock_session.get(url)
                    direct_results.append({"success": True, "status_code": response.status_code})
                except Exception as e:
                    direct_results.append({"success": False, "error": str(e)})
        
        direct_time = time.time() - start_time
        direct_throughput = len(direct_results) / direct_time if direct_time > 0 else 0
        
        # Calculate direct success rate
        direct_success_rate = sum(1 for r in direct_results if r['success']) / len(direct_results)
        
        logger.info(f"Direct throughput: {direct_throughput:.2f} req/s")
        logger.info(f"Direct success rate: {direct_success_rate:.2%}")
        
        # Calculate overhead
        overhead_ratio = direct_throughput / protected_throughput if protected_throughput > 0 else float('inf')
        
        logger.info(f"Throughput overhead: {overhead_ratio:.2f}x")
        
        # Verify results
        assert success_rate > direct_success_rate, "Protected requests should have higher success rate"
        assert overhead_ratio < 5, "Throughput overhead should be reasonable (less than 5x)"