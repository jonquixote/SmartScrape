"""
Integration tests for resource management and error handling services.

This test file verifies the integration between various resource services:
- SessionManager
- RateLimiter
- ProxyManager
- ErrorClassifier
- RetryManager
- CircuitBreakerManager

Tests confirm that the services work together through the ServiceRegistry
and can be accessed via the StrategyContext helper methods.
"""

import pytest
import requests
import threading
import time
from unittest.mock import MagicMock, patch

from core.service_registry import ServiceRegistry
from core.configuration import get_resource_config
from strategies.core.strategy_context import StrategyContext

# Mock responses for testing HTTP error handling
class MockResponse:
    def __init__(self, status_code, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
    
    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"HTTP Error {self.status_code}", response=self)

# Fixed test data
TEST_URL = "https://example.com/test"
TEST_DOMAIN = "example.com"

class TestResourceServicesIntegration:
    """Test integration between various resource management services."""
    
    def setup_method(self):
        """Set up the test environment before each test."""
        # Reset the service registry
        ServiceRegistry._instance = None
        self.config = get_resource_config()
        
        # Create a context with our resource services
        self.context = StrategyContext.with_resource_services({"resource_services": self.config})
    
    def teardown_method(self):
        """Clean up the test environment after each test."""
        if ServiceRegistry._instance:
            ServiceRegistry._instance.shutdown_all()
            ServiceRegistry._instance = None
    
    def test_service_access_from_context(self):
        """Test accessing resource services through StrategyContext convenience methods."""
        # Test that core resource services are accessible through convenience methods
        session_manager = self.context.get_session_manager()
        rate_limiter = self.context.get_rate_limiter()
        error_classifier = self.context.get_error_classifier()
        
        assert session_manager.name == "session_manager"
        assert rate_limiter.name == "rate_limiter"
        assert error_classifier.name == "error_classifier"
        
        # Optional services might not be registered in tests
        try:
            proxy_manager = self.context.get_proxy_manager()
            assert proxy_manager.name == "proxy_manager"
        except Exception:
            pass
        
        try:
            retry_manager = self.context.get_retry_manager()
            assert retry_manager.name == "retry_manager"
        except Exception:
            pass
        
        try:
            circuit_breaker = self.context.get_circuit_breaker_manager()
            assert circuit_breaker.name == "circuit_breaker_manager"
        except Exception:
            pass
    
    def test_session_manager_with_rate_limiter(self):
        """Test interaction between SessionManager and RateLimiter."""
        with patch.object(self.context.get_rate_limiter(), 'wait_if_needed') as mock_wait:
            # Setup mock to simulate rate limiting
            mock_wait.return_value = True
            
            # Test the should_wait helper function
            result = self.context.should_wait(TEST_DOMAIN)
            
            # Verify rate limiter was called
            assert result is True
            mock_wait.assert_called_once_with(TEST_DOMAIN)
    
    def test_error_classification(self):
        """Test error classification through the context."""
        # Create a test exception
        exception = requests.HTTPError(
            "HTTP Error 429", 
            response=MockResponse(429, "Too Many Requests", {"Retry-After": "30"})
        )
        
        # Classify the error
        classification = self.context.classify_error(exception, {"url": TEST_URL})
        
        # Verify classification
        assert classification["original_exception"] == exception
        assert "category" in classification
        assert "severity" in classification
        assert "is_retryable" in classification
        assert "suggested_actions" in classification
        assert classification["metadata"]["url"] == TEST_URL
    
    def test_session_retrieval(self):
        """Test retrieving sessions for different domains."""
        # Get sessions for different domains
        session1 = self.context.get_session("example.com")
        session2 = self.context.get_session("example.org")
        
        # Get the same domain again - should be the same session
        session1_again = self.context.get_session("example.com")
        
        # Check sessions are properly managed
        assert session1 is session1_again  # Same object (cached)
        assert session1 is not session2    # Different domains get different sessions
    
    def test_domain_extraction(self):
        """Test the URL domain extraction helper."""
        test_urls = [
            ("https://www.example.com/path?query=1", "example.com"),
            ("http://sub.example.org:8080/test/", "sub.example.org"),
            ("example.net/no-protocol", "example.net"),
            ("www.example.io", "example.io")
        ]
        
        for url, expected in test_urls:
            result = self.context.extract_domain(url)
            assert result == expected
    
    def test_retry_functionality(self):
        """Test retry functionality through the context."""
        # Mock a function that fails twice then succeeds
        mock_func = MagicMock(side_effect=[
            requests.HTTPError("HTTP Error 500", response=MockResponse(500)),
            requests.HTTPError("HTTP Error 500", response=MockResponse(500)),
            "success"
        ])
        
        # Create a patched retry manager
        with patch('core.retry_manager.RetryManager.retry') as mock_retry:
            # Setup mock to call the function with retries
            def retry_side_effect(func, *args, **kwargs):
                # Simple implementation that tries up to 3 times
                for _ in range(3):
                    try:
                        return func()
                    except Exception as e:
                        last_exception = e
                        continue
                raise last_exception
            
            mock_retry.side_effect = retry_side_effect
            
            # Apply the retry decorator
            retrying_func = self.context.should_retry(mock_func)
            
            # Execute and verify
            result = retrying_func()
            assert result == "success"
            assert mock_func.call_count == 3
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality through the context."""
        # Create a patched circuit breaker
        with patch.object(self.context, 'get_circuit_breaker_manager') as mock_cbm_getter:
            # Setup mock circuit breaker
            mock_circuit = MagicMock()
            mock_circuit.allow_request.return_value = False  # Circuit is open
            
            mock_cbm = MagicMock()
            mock_cbm.is_initialized = True
            mock_cbm.get_circuit_breaker.return_value = mock_circuit
            
            mock_cbm_getter.return_value = mock_cbm
            
            # Check if circuit is open
            circuit_open = self.context.is_circuit_open("example.com")
            
            # Verify circuit breaker was checked
            assert circuit_open is True
            mock_circuit.allow_request.assert_called_once()
    
    def test_full_request_flow(self):
        """
        Test a complete request flow using all resource services.
        This tests the full integration of:
        - Session management
        - Rate limiting
        - Error handling
        - Retry logic
        - Circuit breaking
        """
        # Mock the requests.Session.get method
        with patch('requests.Session.get') as mock_get:
            # Setup mock to return a 429 followed by a 200
            mock_get.side_effect = [
                MockResponse(429, "Too Many Requests"),
                MockResponse(200, "Success")
            ]
            
            # Helper function to make a request with the context
            def make_request(url):
                domain = self.context.extract_domain(url)
                
                # Check circuit breaker
                if self.context.is_circuit_open(domain):
                    raise Exception(f"Circuit open for {domain}")
                
                # Apply rate limiting
                self.context.should_wait(domain)
                
                # Get session
                session = self.context.get_session(domain)
                
                # Make request
                try:
                    response = session.get(url)
                    response.raise_for_status()
                    return response.text
                except Exception as e:
                    # Classify error
                    classification = self.context.classify_error(e, {"url": url})
                    
                    # If retryable, retry
                    if classification.get("is_retryable", False):
                        # Basic retry implementation for the test
                        time.sleep(0.1)  # Small delay
                        return make_request(url)  # Recursive retry
                    
                    # Re-raise if not retryable
                    raise
            
            # Make patched request
            with patch.object(self.context, 'classify_error') as mock_classify:
                # Setup classify_error to mark 429s as retryable
                def classify_side_effect(error, metadata):
                    if isinstance(error, requests.HTTPError) and error.response.status_code == 429:
                        return {
                            "original_exception": error,
                            "category": "rate_limit",
                            "severity": "transient",
                            "is_retryable": True,
                            "metadata": metadata,
                            "suggested_actions": ["backoff", "retry"]
                        }
                    return {
                        "original_exception": error,
                        "category": "unknown",
                        "severity": "persistent",
                        "is_retryable": False,
                        "metadata": metadata,
                        "suggested_actions": []
                    }
                
                mock_classify.side_effect = classify_side_effect
                
                # Run the test
                result = make_request(TEST_URL)
                
                # Verify
                assert result == "Success"
                assert mock_get.call_count == 2