"""
Error Handling Examples for SmartScrape

This module provides practical examples of how to use the error handling
components in SmartScrape, including ErrorClassifier, RetryManager, CircuitBreaker,
and fallback mechanisms.
"""

import time
import random
import logging
import requests
from enum import Enum
import functools
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('error_handling_examples')


# -----------------------------------------------------------------------------
# ErrorClassifier Examples
# -----------------------------------------------------------------------------

def error_classifier_basic_usage():
    """Demonstrate basic ErrorClassifier usage with different exceptions."""
    from core.service_registry import ServiceRegistry
    
    # Get the ErrorClassifier from the service registry
    registry = ServiceRegistry()
    error_classifier = registry.get_service("error_classifier")
    
    # Create sample exceptions to classify
    exceptions = [
        # Network error
        requests.exceptions.ConnectionError("Connection refused"),
        
        # Timeout error
        requests.exceptions.Timeout("Request timed out after 30 seconds"),
        
        # HTTP error (create a response-like object with status_code)
        requests.exceptions.HTTPError("404 Client Error: Not Found"),
        
        # Generic exception
        ValueError("Invalid parameter")
    ]
    
    # Add metadata for context
    metadata = {
        'url': 'https://example.com/page',
        'domain': 'example.com',
        'attempt': 2
    }
    
    # Classify each exception
    for i, exception in enumerate(exceptions):
        logger.info(f"Classifying exception {i+1}: {str(exception)}")
        
        # Add HTTP response to HTTP error for proper classification
        if isinstance(exception, requests.exceptions.HTTPError):
            # Create a mock response with status code
            mock_response = requests.Response()
            mock_response.status_code = 404
            exception.response = mock_response
        
        # Classify the exception
        classification = error_classifier.classify_exception(exception, metadata)
        
        # Log the classification details
        logger.info(f"Classification:")
        logger.info(f"  Category: {classification['category']}")
        logger.info(f"  Severity: {classification['severity']}")
        logger.info(f"  Retryable: {classification['is_retryable']}")
        logger.info(f"  Suggested actions: {', '.join(classification['suggested_actions'])}")
    
    # Cleanup
    logger.info("ErrorClassifier basic example completed")


def error_classifier_with_http_errors():
    """Demonstrate ErrorClassifier with different HTTP status codes."""
    from core.service_registry import ServiceRegistry
    
    # Get the ErrorClassifier from the service registry
    registry = ServiceRegistry()
    error_classifier = registry.get_service("error_classifier")
    
    # HTTP status codes to classify
    status_codes = [400, 401, 403, 404, 429, 500, 502, 503]
    
    # Create metadata
    metadata = {
        'url': 'https://api.example.com/data',
        'domain': 'api.example.com'
    }
    
    # Classify each status code
    for status_code in status_codes:
        # Create a response with the status code
        response = requests.Response()
        response.status_code = status_code
        
        # Create an HTTPError with the response
        exception = requests.exceptions.HTTPError(
            f"{status_code} Error",
            response=response
        )
        
        logger.info(f"Classifying HTTP {status_code} error")
        
        # Classify the exception
        classification = error_classifier.classify_exception(exception, metadata)
        
        # Log the classification details
        logger.info(f"Classification for HTTP {status_code}:")
        logger.info(f"  Category: {classification['category']}")
        logger.info(f"  Severity: {classification['severity']}")
        logger.info(f"  Retryable: {classification['is_retryable']}")
        logger.info(f"  Suggested actions: {', '.join(classification['suggested_actions'])}")
    
    # Cleanup
    logger.info("HTTP error classification example completed")


def error_classifier_with_content_errors():
    """Demonstrate ErrorClassifier with content-based error detection."""
    from core.service_registry import ServiceRegistry
    
    # Get the ErrorClassifier from the service registry
    registry = ServiceRegistry()
    error_classifier = registry.get_service("error_classifier")
    
    # Create error content samples
    error_contents = [
        # CAPTCHA
        {
            'content': 'Please complete this CAPTCHA to verify you are human.',
            'description': 'CAPTCHA page'
        },
        # Access denied
        {
            'content': 'Access Denied. You don\'t have permission to access this resource.',
            'description': 'Access denied page'
        },
        # Rate limit
        {
            'content': 'Too many requests. Please try again later.',
            'description': 'Rate limit message'
        },
        # Normal content
        {
            'content': 'Welcome to Example.com. This is a normal page with no errors.',
            'description': 'Normal page'
        }
    ]
    
    # Create metadata
    metadata = {
        'url': 'https://example.com/page',
        'domain': 'example.com'
    }
    
    # Check each content sample
    for sample in error_contents:
        logger.info(f"Checking content: {sample['description']}")
        
        # Create a mock response with the content
        response = requests.Response()
        response.status_code = 200  # Status code is OK, but content indicates an error
        response._content = sample['content'].encode('utf-8')
        
        # Check for CAPTCHA
        has_captcha = error_classifier._check_for_captcha(response)
        logger.info(f"  CAPTCHA detected: {has_captcha}")
        
        # Check for access denied
        has_access_denied = any(pattern.search(sample['content']) 
                               for pattern in error_classifier._access_denied_patterns)
        logger.info(f"  Access denied detected: {has_access_denied}")
    
    # Cleanup
    logger.info("Content error detection example completed")


# -----------------------------------------------------------------------------
# RetryManager Examples
# -----------------------------------------------------------------------------

def basic_retry_function():
    """Demonstrate a basic retry function without using the library."""
    def fetch_with_retry(url, max_retries=3, backoff_factor=1.0):
        """Fetch a URL with retry logic."""
        retries = 0
        
        while retries <= max_retries:
            try:
                logger.info(f"Attempt {retries + 1} for {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                logger.info(f"Request succeeded on attempt {retries + 1}")
                return response
            
            except (requests.exceptions.RequestException, Exception) as e:
                retries += 1
                logger.error(f"Attempt {retries} failed: {str(e)}")
                
                if retries > max_retries:
                    logger.error(f"Max retries ({max_retries}) exceeded")
                    raise
                
                # Calculate backoff with jitter
                backoff = backoff_factor * (2 ** (retries - 1))
                jitter = random.uniform(0, 0.1 * backoff)
                sleep_time = backoff + jitter
                
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
    
    # Test URLs - some succeed, some fail
    urls = [
        "https://httpbin.org/status/200",  # Should succeed
        "https://httpbin.org/status/500",  # Server error (retryable)
        "https://httpbin.org/status/404",  # Not found (shouldn't retry but will in this example)
        "https://nonexistent-domain-123456.com/"  # DNS error
    ]
    
    # Try each URL
    for url in urls:
        logger.info(f"\nTrying URL: {url}")
        try:
            response = fetch_with_retry(url)
            logger.info(f"Final result: Success - Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Final result: Failed - {str(e)}")
    
    logger.info("Basic retry function example completed")


def retry_manager_decorator_example():
    """Demonstrate RetryManager with decorator pattern."""
    from core.service_registry import ServiceRegistry
    from core.retry_manager import retry
    
    # Create a retry-decorated function
    @retry(max_attempts=3, backoff_factor=2, jitter=True)
    def fetch_url(url):
        """Fetch a URL with automatic retry."""
        logger.info(f"Fetching {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response
    
    # Test URLs
    urls = [
        "https://httpbin.org/status/200",  # Should succeed
        "https://httpbin.org/status/500",  # Server error (retryable)
        "https://httpbin.org/status/404"   # Not found (not retryable)
    ]
    
    # Try each URL
    for url in urls:
        logger.info(f"\nTrying URL: {url}")
        try:
            response = fetch_url(url)
            logger.info(f"Final result: Success - Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Final result: Failed - {str(e)}")
    
    logger.info("Retry decorator example completed")


def retry_manager_with_conditional_retry():
    """Demonstrate RetryManager with conditional retry logic."""
    from core.service_registry import ServiceRegistry
    from core.retry_manager import retry
    
    # Get the ErrorClassifier from the service registry
    registry = ServiceRegistry()
    error_classifier = registry.get_service("error_classifier")
    
    # Define a retry condition function
    def is_retryable_error(exception):
        """Determine if an exception is retryable using the ErrorClassifier."""
        if not exception:
            return False
        
        # Use the error classifier to categorize the exception
        classification = error_classifier.classify_exception(exception)
        
        # Only retry if the classifier says it's retryable
        return classification['is_retryable']
    
    # Create a conditionally retrying function
    @retry(max_attempts=3, retry_on=is_retryable_error)
    def fetch_with_smart_retry(url):
        """Fetch a URL with intelligent retry based on error classification."""
        logger.info(f"Fetching {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response
    
    # Test URLs
    urls = [
        "https://httpbin.org/status/200",  # Should succeed
        "https://httpbin.org/status/500",  # Server error (classified as retryable)
        "https://httpbin.org/status/429",  # Rate limit (classified as retryable)
        "https://httpbin.org/status/404",  # Not found (classified as not retryable)
        "https://httpbin.org/status/403"   # Forbidden (may or may not be retryable)
    ]
    
    # Try each URL
    for url in urls:
        logger.info(f"\nTrying URL: {url}")
        try:
            response = fetch_with_smart_retry(url)
            logger.info(f"Final result: Success - Status code {response.status_code}")
        except Exception as e:
            logger.error(f"Final result: Failed - {str(e)}")
    
    logger.info("Conditional retry example completed")


# -----------------------------------------------------------------------------
# CircuitBreaker Examples
# -----------------------------------------------------------------------------

def circuit_breaker_basic_example():
    """Demonstrate basic CircuitBreaker usage."""
    from core.service_registry import ServiceRegistry
    from core.circuit_breaker import OpenCircuitError
    
    # Get the CircuitBreakerManager from the service registry
    registry = ServiceRegistry()
    circuit_breaker_manager = registry.get_service("circuit_breaker_manager")
    
    # Create a circuit breaker for a service
    service_name = "example_service"
    circuit = circuit_breaker_manager.get_circuit_breaker(service_name, {
        'failure_threshold': 3,  # Open after 3 failures
        'reset_timeout': 5,      # Try again after 5 seconds
        'half_open_max': 1       # Allow 1 test request when half-open
    })
    
    # Function that simulates a service call
    def call_service(fail=False):
        """Simulate a service call that might fail."""
        if fail:
            raise Exception("Service failed")
        return "Service response"
    
    # Make several requests to the service
    for i in range(10):
        logger.info(f"\nRequest {i+1} to {service_name}")
        
        # Check if circuit allows the request
        if not circuit.allow_request():
            logger.warning(f"Circuit is {circuit.state.value}, request blocked")
            
            # Wait a bit to let the circuit transition to HALF_OPEN
            if i > 5:
                logger.info("Waiting for circuit reset timeout...")
                time.sleep(6)  # Just over the reset_timeout
                
                # Check state again after waiting
                logger.info(f"Circuit state after waiting: {circuit.state.value}")
                
                if circuit.allow_request():
                    logger.info("Circuit now allowing test request")
                else:
                    logger.warning("Circuit still not allowing requests")
            
            continue
        
        # Determine if this request should fail
        # Fail requests 3-5 to trigger circuit opening
        should_fail = 3 <= i <= 5
        
        try:
            # Attempt to call the service
            result = call_service(fail=should_fail)
            
            # Record success
            circuit.record_success()
            logger.info(f"Request succeeded: {result}")
            logger.info(f"Circuit state: {circuit.state.value}")
            
        except Exception as e:
            # Record failure
            circuit.record_failure()
            logger.error(f"Request failed: {str(e)}")
            logger.info(f"Circuit state: {circuit.state.value}")
    
    # Cleanup
    logger.info("CircuitBreaker basic example completed")


def circuit_breaker_decorator_example():
    """Demonstrate CircuitBreaker with decorator pattern."""
    from core.service_registry import ServiceRegistry
    from core.circuit_breaker import circuit_breaker, OpenCircuitError
    
    # Get the CircuitBreakerManager from the service registry
    registry = ServiceRegistry()
    circuit_breaker_manager = registry.get_service("circuit_breaker_manager")
    
    # Create a circuit-protected function
    @circuit_breaker("api_service", failure_threshold=3, reset_timeout=5)
    def call_api(endpoint, fail=False):
        """Call an API endpoint with circuit breaker protection."""
        logger.info(f"Calling API endpoint: {endpoint}")
        
        if fail:
            logger.error("API call failed")
            raise Exception("API error")
        
        logger.info("API call succeeded")
        return {"status": "success", "data": "example response"}
    
    # Make several calls to the API
    endpoints = ["users", "products", "orders"]
    
    for i in range(10):
        endpoint = random.choice(endpoints)
        logger.info(f"\nRequest {i+1} to endpoint {endpoint}")
        
        # Determine if this request should fail
        # Fail requests 3-5 to trigger circuit opening
        should_fail = 3 <= i <= 5
        
        try:
            # Attempt to call the API
            result = call_api(endpoint, fail=should_fail)
            logger.info(f"Call result: {result}")
            
        except OpenCircuitError as e:
            # Circuit is open
            logger.warning(f"Circuit open, request blocked: {str(e)}")
            
            # In a real app, we might use cached data or a fallback here
            if i > 7:
                # After a few blocked requests, wait for reset timeout
                logger.info("Waiting for circuit reset timeout...")
                time.sleep(6)
            
        except Exception as e:
            # Other error occurred
            logger.error(f"Call failed with error: {str(e)}")
    
    # Cleanup
    logger.info("CircuitBreaker decorator example completed")


def circuit_breaker_with_fallback():
    """Demonstrate CircuitBreaker with fallback mechanisms."""
    from core.service_registry import ServiceRegistry
    from core.circuit_breaker import OpenCircuitError
    
    # Get the CircuitBreakerManager from the service registry
    registry = ServiceRegistry()
    circuit_breaker_manager = registry.get_service("circuit_breaker_manager")
    
    # Create a circuit breaker for a service
    service_name = "data_service"
    circuit = circuit_breaker_manager.get_circuit_breaker(service_name, {
        'failure_threshold': 2,  # Open after 2 failures
        'reset_timeout': 5       # Try again after 5 seconds
    })
    
    # Simulated cache for fallback data
    cache = {
        "users": {"cached": True, "data": [{"id": 1, "name": "Cached User"}]},
        "products": {"cached": True, "data": [{"id": 1, "name": "Cached Product"}]}
    }
    
    # Function that fetches data with circuit breaking and fallback
    def get_data(resource_type, use_fallback=True):
        """Get data with circuit breaking and fallback to cache."""
        logger.info(f"Requesting {resource_type} data")
        
        # Check if circuit is closed
        if not circuit.allow_request():
            logger.warning(f"Circuit for {service_name} is open")
            
            if use_fallback and resource_type in cache:
                logger.info(f"Using cached data for {resource_type}")
                return cache[resource_type]
            else:
                raise OpenCircuitError(f"Circuit open for {service_name} and no fallback available")
        
        try:
            # Simulate service call
            # Randomly fail some requests
            if random.random() < 0.4:
                logger.error("Service call failed")
                circuit.record_failure()
                raise Exception("Data service error")
            
            # Simulate successful response
            logger.info("Service call succeeded")
            circuit.record_success()
            return {
                "fresh": True,
                "data": [
                    {"id": 1, "name": f"Fresh {resource_type.capitalize()} 1"},
                    {"id": 2, "name": f"Fresh {resource_type.capitalize()} 2"}
                ]
            }
            
        except Exception as e:
            # Service call failed
            if use_fallback and resource_type in cache:
                logger.info(f"Using cached data for {resource_type} after error")
                return cache[resource_type]
            else:
                raise
    
    # Make several requests to get data
    resource_types = ["users", "products", "orders"]
    
    for i in range(10):
        resource_type = resource_types[i % len(resource_types)]
        logger.info(f"\nRequest {i+1} for {resource_type}")
        
        try:
            # Get data with fallback
            result = get_data(resource_type)
            
            # Log the result source
            if "cached" in result:
                logger.info(f"Returned cached data: {result}")
            else:
                logger.info(f"Returned fresh data: {result}")
            
        except OpenCircuitError as e:
            logger.warning(f"Request blocked by circuit breaker: {str(e)}")
            
            # Wait for circuit reset if we're later in the test
            if i > 6:
                logger.info("Waiting for circuit reset timeout...")
                time.sleep(6)
                
        except Exception as e:
            logger.error(f"Request failed with error: {str(e)}")
    
    # Cleanup
    logger.info("CircuitBreaker with fallback example completed")


# -----------------------------------------------------------------------------
# Fallback Mechanism Examples
# -----------------------------------------------------------------------------

def strategy_fallback_example():
    """Demonstrate strategy fallback pattern."""
    # Define base strategy classes
    class BaseStrategy:
        def execute(self, context, url):
            raise NotImplementedError("Subclasses must implement execute method")
    
    class AIGuidedStrategy(BaseStrategy):
        def execute(self, context, url):
            # Simulate a strategy that might fail
            if random.random() < 0.6:
                logger.error("AI-guided strategy failed")
                raise Exception("AI service error")
            
            logger.info("AI-guided strategy succeeded")
            return {"method": "ai", "data": {"title": "AI Extracted Title", "content": "AI content"}}
    
    class DOMBasedStrategy(BaseStrategy):
        def execute(self, context, url):
            # Simpler strategy that rarely fails
            if random.random() < 0.2:
                logger.error("DOM-based strategy failed")
                raise Exception("DOM parsing error")
            
            logger.info("DOM-based strategy succeeded")
            return {"method": "dom", "data": {"title": "DOM Extracted Title", "content": "DOM content"}}
    
    class RegexStrategy(BaseStrategy):
        def execute(self, context, url):
            # Fallback strategy that almost never fails
            if random.random() < 0.05:
                logger.error("Regex strategy failed")
                raise Exception("Regex error")
            
            logger.info("Regex strategy succeeded")
            return {"method": "regex", "data": {"title": "Basic Title", "content": "Basic content"}}
    
    # Multi-stage strategy with fallbacks
    class MultiStrategy(BaseStrategy):
        def __init__(self):
            self.ai_strategy = AIGuidedStrategy()
            self.dom_strategy = DOMBasedStrategy()
            self.regex_strategy = RegexStrategy()
        
        def execute(self, context, url):
            logger.info(f"Executing multi-strategy on {url}")
            
            # Try strategies in order
            strategies = [
                ("AI-guided", self.ai_strategy),
                ("DOM-based", self.dom_strategy),
                ("Regex", self.regex_strategy)
            ]
            
            last_error = None
            for name, strategy in strategies:
                try:
                    logger.info(f"Trying {name} strategy")
                    result = strategy.execute(context, url)
                    logger.info(f"{name} strategy succeeded")
                    
                    # Add metadata about fallback level
                    result['fallback_level'] = name
                    return result
                    
                except Exception as e:
                    logger.error(f"{name} strategy failed: {str(e)}")
                    last_error = e
                    # Continue to next strategy
            
            # If we get here, all strategies failed
            raise Exception(f"All strategies failed for {url}") from last_error
    
    # Create a simple context mock
    class ContextMock:
        pass
    
    context = ContextMock()
    urls = [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3",
        "https://example.com/article4",
        "https://example.com/article5"
    ]
    
    # Create the multi-strategy
    strategy = MultiStrategy()
    
    # Try the strategy on several URLs
    for i, url in enumerate(urls):
        logger.info(f"\nProcessing URL {i+1}: {url}")
        try:
            result = strategy.execute(context, url)
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(f"All strategies failed: {str(e)}")
    
    logger.info("Strategy fallback example completed")


def extraction_fallback_example():
    """Demonstrate extraction fallback pattern."""
    # Define base extractor class
    class BaseExtractor:
        def extract(self, html, context):
            raise NotImplementedError("Subclasses must implement extract method")
    
    # Different extractor implementations
    class SchemaExtractor(BaseExtractor):
        def extract(self, html, context):
            # Simulate schema extraction (JSON-LD, microdata)
            if random.random() < 0.5:
                logger.error("Schema extraction failed")
                raise Exception("No schema data found")
            
            logger.info("Schema extraction succeeded")
            return {
                "method": "schema",
                "title": "Schema Extracted Title",
                "description": "Schema extracted description",
                "price": "$99.99",
                "rating": 4.5
            }
    
    class AIExtractor(BaseExtractor):
        def extract(self, html, context):
            # Simulate AI-based extraction
            if random.random() < 0.4:
                logger.error("AI extraction failed")
                raise Exception("AI service error")
            
            logger.info("AI extraction succeeded")
            return {
                "method": "ai",
                "title": "AI Extracted Title",
                "description": "AI extracted description",
                "price": "$99.99"
                # No rating in AI extraction
            }
    
    class CSSExtractor(BaseExtractor):
        def extract(self, html, context):
            # Simulate CSS selector extraction
            if random.random() < 0.3:
                logger.error("CSS extraction failed")
                raise Exception("Selector not found")
            
            logger.info("CSS extraction succeeded")
            return {
                "method": "css",
                "title": "CSS Extracted Title",
                "description": "CSS extracted description"
                # No price or rating in CSS extraction
            }
    
    class RegexExtractor(BaseExtractor):
        def extract(self, html, context):
            # Simulate regex extraction (simplest, most reliable)
            if random.random() < 0.1:
                logger.error("Regex extraction failed")
                raise Exception("Regex match failed")
            
            logger.info("Regex extraction succeeded")
            return {
                "method": "regex",
                "title": "Basic Title"
                # Only title available with regex
            }
    
    # Resilient extractor with fallbacks
    class ResilientExtractor(BaseExtractor):
        def __init__(self):
            self.schema_extractor = SchemaExtractor()
            self.ai_extractor = AIExtractor()
            self.css_extractor = CSSExtractor()
            self.regex_extractor = RegexExtractor()
        
        def extract(self, html, context):
            logger.info("Starting extraction with fallback chain")
            
            # Try extractors in order
            extractors = [
                ("Schema", self.schema_extractor),
                ("AI", self.ai_extractor),
                ("CSS", self.css_extractor),
                ("Regex", self.regex_extractor)
            ]
            
            last_error = None
            for name, extractor in extractors:
                try:
                    logger.info(f"Trying {name} extractor")
                    result = extractor.extract(html, context)
                    logger.info(f"{name} extractor succeeded")
                    
                    # Add metadata about extraction method
                    result['extraction_method'] = name
                    result['fallback_level'] = extractors.index((name, extractor))
                    return result
                    
                except Exception as e:
                    logger.error(f"{name} extractor failed: {str(e)}")
                    last_error = e
                    # Continue to next extractor
            
            # If we get here, all extractors failed
            raise Exception("All extraction methods failed") from last_error
    
    # Create a simple context mock
    class ContextMock:
        pass
    
    context = ContextMock()
    
    # Sample HTML content (just a placeholder)
    html = "<html><body><h1>Product Title</h1><p>Description</p></body></html>"
    
    # Create the resilient extractor
    extractor = ResilientExtractor()
    
    # Try extraction multiple times
    for i in range(5):
        logger.info(f"\nExtraction attempt {i+1}")
        try:
            result = extractor.extract(html, context)
            logger.info(f"Extraction result: {result}")
            
            # Log available fields
            fields = [field for field in ['title', 'description', 'price', 'rating'] 
                     if field in result]
            logger.info(f"Available fields: {', '.join(fields)}")
            logger.info(f"Extraction method: {result.get('extraction_method')}")
            logger.info(f"Fallback level: {result.get('fallback_level')}")
            
        except Exception as e:
            logger.error(f"All extraction methods failed: {str(e)}")
    
    logger.info("Extraction fallback example completed")


# -----------------------------------------------------------------------------
# Integrated Examples
# -----------------------------------------------------------------------------

def integrated_error_handling():
    """Demonstrate integrated error handling with all components."""
    from core.service_registry import ServiceRegistry
    
    # Get services from registry
    registry = ServiceRegistry()
    error_classifier = registry.get_service("error_classifier")
    circuit_breaker_manager = registry.get_service("circuit_breaker_manager")
    
    # Create a retry decorator that uses error classification
    def retry_with_classification(max_attempts=3, backoff_factor=1.0):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        # Classify the error
                        classification = error_classifier.classify_exception(e)
                        
                        # Log the attempt and classification
                        logger.error(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}\n"
                            f"Classification: {classification['category']}, "
                            f"Severity: {classification['severity']}, "
                            f"Retryable: {classification['is_retryable']}"
                        )
                        
                        # If not retryable, stop trying
                        if not classification['is_retryable']:
                            logger.warning("Error not retryable, stopping retry attempts")
                            break
                        
                        # If last attempt, stop
                        if attempt >= max_attempts - 1:
                            logger.warning("Maximum retry attempts reached")
                            break
                        
                        # Calculate backoff
                        backoff = backoff_factor * (2 ** attempt)
                        jitter = random.uniform(0, 0.1 * backoff)
                        sleep_time = backoff + jitter
                        
                        logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                
                # If we get here, all attempts failed
                raise last_exception
                
            return wrapper
        return decorator
    
    # Function that combines retry and circuit breaker
    def resilient_request(url, circuit_name=None):
        """Make a resilient HTTP request with retry and circuit breaking."""
        # If no circuit name provided, use the domain
        if circuit_name is None:
            circuit_name = urlparse(url).netloc
        
        # Get or create circuit breaker
        circuit = circuit_breaker_manager.get_circuit_breaker(circuit_name)
        
        # Check if circuit allows request
        if not circuit.allow_request():
            logger.warning(f"Circuit {circuit_name} is open, request blocked")
            raise OpenCircuitError(f"Circuit open for {circuit_name}")
        
        # Create a function with retry
        @retry_with_classification(max_attempts=3, backoff_factor=1.0)
        def make_request():
            logger.info(f"Making request to {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        
        try:
            # Make the request with retry
            response = make_request()
            
            # Record success
            circuit.record_success()
            
            return response
            
        except Exception as e:
            # Record failure
            circuit.record_failure()
            
            # Re-raise the exception
            raise
    
    # Cache for fallback data
    cache = {
        "https://httpbin.org/get": {"cached": True, "data": {"message": "Cached response"}}
    }
    
    # Function that uses fallback on failure
    def get_with_fallback(url):
        """Get data with fallback to cache on failure."""
        try:
            # Try to get live data
            response = resilient_request(url)
            logger.info(f"Request to {url} succeeded")
            
            # Return live data
            return {
                "source": "live",
                "data": response.json() if response.headers.get('content-type') == 'application/json' else {"text": response.text}
            }
            
        except OpenCircuitError as e:
            # Circuit is open, use cache if available
            logger.warning(f"Circuit open: {str(e)}")
            
            if url in cache:
                logger.info(f"Using cached data for {url}")
                return cache[url]
            else:
                raise Exception(f"Circuit open for {url} and no cache available")
                
        except Exception as e:
            # Other error, use cache if available
            logger.error(f"Request failed: {str(e)}")
            
            if url in cache:
                logger.info(f"Using cached data for {url}")
                return cache[url]
            else:
                # Classify the error
                classification = error_classifier.classify_exception(e)
                
                # If it's a fatal error, raise it
                if classification['severity'] == 'fatal':
                    raise
                
                # Otherwise, return a degraded response
                return {
                    "source": "degraded",
                    "error": str(e),
                    "data": {"message": "Error occurred, degraded response"}
                }
    
    # Test URLs
    urls = [
        "https://httpbin.org/status/200",    # Success
        "https://httpbin.org/status/429",    # Rate limit
        "https://httpbin.org/status/500",    # Server error
        "https://httpbin.org/status/404",    # Not found
        "https://nonexistent-domain-123456.com/"  # DNS error
    ]
    
    # Mix in some successful requests to httpbin.org/get
    urls.extend(["https://httpbin.org/get"] * 3)
    
    # Shuffle the URLs to make the example more realistic
    random.shuffle(urls)
    
    # Process each URL
    for i, url in enumerate(urls):
        logger.info(f"\nRequest {i+1} to {url}")
        
        try:
            result = get_with_fallback(url)
            logger.info(f"Result source: {result.get('source', 'unknown')}")
            logger.info(f"Result data: {result.get('data', {})}")
            
        except Exception as e:
            logger.error(f"Request failed with unrecoverable error: {str(e)}")
    
    # Cleanup
    logger.info("Integrated error handling example completed")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main():
    """Run examples."""
    logger.info("Starting error handling examples")
    
    examples = [
        # Error Classifier examples
        error_classifier_basic_usage,
        error_classifier_with_http_errors,
        error_classifier_with_content_errors,
        
        # RetryManager examples
        basic_retry_function,
        retry_manager_decorator_example,
        retry_manager_with_conditional_retry,
        
        # CircuitBreaker examples
        circuit_breaker_basic_example,
        circuit_breaker_decorator_example,
        circuit_breaker_with_fallback,
        
        # Fallback examples
        strategy_fallback_example,
        extraction_fallback_example,
        
        # Integrated example
        integrated_error_handling
    ]
    
    # Select which examples to run
    examples_to_run = [
        error_classifier_basic_usage,
        retry_manager_decorator_example,
        circuit_breaker_basic_example,
        integrated_error_handling
    ]
    
    # Run selected examples
    for example in examples_to_run:
        logger.info(f"\n{'=' * 80}\nRunning example: {example.__name__}\n{'=' * 80}")
        try:
            example()
        except Exception as e:
            logger.error(f"Example {example.__name__} failed: {str(e)}")
    
    logger.info("All examples completed")


if __name__ == "__main__":
    main()