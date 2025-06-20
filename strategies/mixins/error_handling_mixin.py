"""
Error Handling Mixin for strategies.

This mixin provides error handling capabilities to strategies, including
error classification, retry mechanism, and circuit breaker pattern.
"""

import logging
import random
import time
import threading
import requests
from typing import Dict, Any, Optional, List, Callable
from urllib.parse import urlparse
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of errors that can occur during scraping."""
    NETWORK = "network"
    HTTP = "http"
    CONTENT = "content"
    PARSING = "parsing"
    RATE_LIMIT = "rate_limit"
    CAPTCHA = "captcha"
    PROXY = "proxy"
    AUTHENTICATION = "authentication"
    INFRASTRUCTURE = "infrastructure"
    AI_SERVICE = "ai_service"
    UNKNOWN = "unknown"

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    TRANSIENT = "transient"  # Temporary, likely to resolve on retry
    PERSISTENT = "persistent"  # Persistent, might resolve with different approach
    FATAL = "fatal"  # Fatal, unlikely to resolve without intervention

class CircuitOpenError(Exception):
    """Exception raised when a request is blocked by an open circuit."""
    pass

class ErrorHandlingMixin:
    """
    Mixin that provides error handling capabilities to strategies.
    
    This mixin is designed to be used with BaseStrategy and its subclasses.
    It provides:
    - Error classification (categories, severity)
    - Retry mechanism (backoff, conditional)
    - Circuit breaker pattern (prevent repeated calls to failing services)
    - Error tracking and metrics
    """
    
    def _initialize_error_handling(self):
        """Initialize error handling state."""
        # Error tracking
        self._errors = []
        self._error_lock = threading.RLock()
        
        # Retry tracking
        self._retry_counts = {}
        self._retry_lock = threading.RLock()
        
        # Circuit breaker
        self._circuit_breakers = {}
        self._circuit_lock = threading.RLock()
        
        # Configure default settings
        self._error_config = {
            'max_retries': 3,
            'backoff_factor': 2.0,
            'circuit_breaker_threshold': 5,
            'circuit_breaker_reset_seconds': 60,
            'jitter_factor': 0.1
        }
    
    def _classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify an exception and return classification details.
        
        Args:
            exception: The exception to classify
            context: Additional context about the operation that failed
            
        Returns:
            Dictionary with error classification details
        """
        context = context or {}
        
        # Default classification
        classification = {
            'original_exception': exception,
            'error_message': str(exception),
            'error_type': type(exception).__name__,
            'category': ErrorCategory.UNKNOWN.value,
            'severity': ErrorSeverity.PERSISTENT.value,
            'is_retryable': False,
            'context': context,
            'timestamp': time.time(),
            'suggested_actions': []
        }
        
        # Classify based on exception type
        if isinstance(exception, requests.exceptions.Timeout):
            classification['category'] = ErrorCategory.NETWORK.value
            classification['severity'] = ErrorSeverity.TRANSIENT.value
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'increase_timeout']
            
        elif isinstance(exception, requests.exceptions.ConnectionError):
            classification['category'] = ErrorCategory.NETWORK.value
            classification['severity'] = ErrorSeverity.TRANSIENT.value
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'check_connectivity']
            
        elif isinstance(exception, requests.exceptions.HTTPError):
            return self._classify_http_error(exception, context)
            
        elif isinstance(exception, requests.exceptions.ProxyError):
            classification['category'] = ErrorCategory.PROXY.value
            classification['severity'] = ErrorSeverity.TRANSIENT.value
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['rotate_proxy', 'retry']
            
        elif isinstance(exception, requests.exceptions.TooManyRedirects):
            classification['category'] = ErrorCategory.HTTP.value
            classification['severity'] = ErrorSeverity.PERSISTENT.value
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_url']
            
        elif isinstance(exception, requests.exceptions.InvalidURL):
            classification['category'] = ErrorCategory.HTTP.value
            classification['severity'] = ErrorSeverity.FATAL.value
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['fix_url']
            
        elif isinstance(exception, json.JSONDecodeError):
            classification['category'] = ErrorCategory.PARSING.value
            classification['severity'] = ErrorSeverity.PERSISTENT.value
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_response_format']
            
        elif 'captcha' in str(exception).lower():
            classification['category'] = ErrorCategory.CAPTCHA.value
            classification['severity'] = ErrorSeverity.PERSISTENT.value
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['solve_captcha', 'rotate_proxy']
            
        elif 'permission' in str(exception).lower() or 'denied' in str(exception).lower():
            classification['category'] = ErrorCategory.AUTHENTICATION.value
            classification['severity'] = ErrorSeverity.PERSISTENT.value
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_permissions']
            
        # More specific exception types based on operation context
        operation = context.get('operation', '')
        
        if 'ai' in operation.lower() or 'ai_service' in str(exception).lower():
            classification['category'] = ErrorCategory.AI_SERVICE.value
            if 'rate' in str(exception).lower() or 'quota' in str(exception).lower():
                classification['severity'] = ErrorSeverity.TRANSIENT.value
                classification['is_retryable'] = True
                classification['suggested_actions'] = ['backoff', 'reduce_tokens']
            else:
                classification['severity'] = ErrorSeverity.PERSISTENT.value
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['check_ai_service']
        
        # Track this error
        self._track_error(classification)
        
        return classification
    
    def _classify_http_error(self, exception: requests.exceptions.HTTPError, 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify an HTTP error based on status code and response content.
        
        Args:
            exception: The HTTPError exception
            context: Additional context about the operation that failed
            
        Returns:
            Dictionary with error classification details
        """
        context = context or {}
        response = getattr(exception, 'response', None)
        
        # Default classification
        classification = {
            'original_exception': exception,
            'error_message': str(exception),
            'error_type': 'HTTPError',
            'category': ErrorCategory.HTTP.value,
            'severity': ErrorSeverity.PERSISTENT.value,
            'is_retryable': False,
            'context': context,
            'timestamp': time.time(),
            'suggested_actions': []
        }
        
        if not response:
            return classification
            
        status_code = response.status_code
        
        # Add status code to context
        context['status_code'] = status_code
        
        # Classify based on status code
        if 400 <= status_code < 500:
            # 4xx - Client errors
            if status_code == 401:
                classification['category'] = ErrorCategory.AUTHENTICATION.value
                classification['severity'] = ErrorSeverity.PERSISTENT.value
                classification['suggested_actions'] = ['authenticate', 'check_credentials']
                
            elif status_code == 403:
                # Check if it's a CAPTCHA
                try:
                    if 'captcha' in response.text.lower():
                        classification['category'] = ErrorCategory.CAPTCHA.value
                        classification['suggested_actions'] = ['solve_captcha', 'rotate_proxy']
                    else:
                        classification['category'] = ErrorCategory.HTTP.value
                        classification['suggested_actions'] = ['rotate_proxy', 'check_headers']
                except:
                    # If we can't check the content, assume it's a general HTTP error
                    classification['category'] = ErrorCategory.HTTP.value
                    classification['suggested_actions'] = ['rotate_proxy', 'check_headers']
                    
            elif status_code == 404:
                classification['category'] = ErrorCategory.HTTP.value
                classification['severity'] = ErrorSeverity.FATAL.value
                classification['suggested_actions'] = ['verify_url']
                
            elif status_code == 429:
                classification['category'] = ErrorCategory.RATE_LIMIT.value
                classification['severity'] = ErrorSeverity.TRANSIENT.value
                classification['is_retryable'] = True
                classification['suggested_actions'] = ['backoff', 'reduce_rate', 'rotate_proxy']
                
        elif 500 <= status_code < 600:
            # 5xx - Server errors
            classification['category'] = ErrorCategory.HTTP.value
            classification['severity'] = ErrorSeverity.TRANSIENT.value
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'backoff']
            
        # Add additional metadata if available
        if response:
            try:
                classification['headers'] = dict(response.headers)
            except:
                pass
        
        # Track this error
        self._track_error(classification)
        
        return classification
    
    def _track_error(self, error_info: Dict[str, Any]) -> None:
        """
        Track an error for metrics and analysis.
        
        Args:
            error_info: Error classification details
        """
        with self._error_lock:
            # Add to error list, limiting to most recent 100 errors
            self._errors.append(error_info)
            if len(self._errors) > 100:
                self._errors = self._errors[-100:]
    
    def _should_retry(self, operation: str, attempt: int, error_info: Dict[str, Any]) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            operation: Name of the operation that failed
            attempt: Current attempt number (starts at 1)
            error_info: Error classification details
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        # Never retry if we've hit max retries
        max_retries = self._error_config.get('max_retries', 3)
        if attempt >= max_retries:
            return False
        
        # Check if the error is retryable
        is_retryable = error_info.get('is_retryable', False)
        if not is_retryable:
            return False
        
        # Record retry attempt
        with self._retry_lock:
            retry_key = f"{operation}_{error_info.get('context', {}).get('url', 'unknown')}"
            self._retry_counts[retry_key] = self._retry_counts.get(retry_key, 0) + 1
        
        return True
    
    def _get_backoff_time(self, attempt: int) -> float:
        """
        Calculate backoff time for a retry attempt.
        
        Args:
            attempt: Current attempt number (starts at 1)
            
        Returns:
            Backoff time in seconds
        """
        backoff_factor = self._error_config.get('backoff_factor', 2.0)
        jitter_factor = self._error_config.get('jitter_factor', 0.1)
        
        # Calculate base backoff time (exponential)
        base_backoff = backoff_factor ** (attempt - 1)
        
        # Add jitter to prevent thundering herd
        jitter = base_backoff * jitter_factor * (2 * (random.random() - 0.5))
        
        return base_backoff + jitter
    
    def _check_circuit_breaker(self, domain: str) -> bool:
        """
        Check if a circuit breaker allows a request.
        
        Args:
            domain: The domain or service to check
            
        Returns:
            True if the circuit is closed (requests allowed), False if open (blocked)
        """
        with self._circuit_lock:
            circuit = self._circuit_breakers.get(domain)
            
            # If no circuit exists, create one
            if not circuit:
                self._circuit_breakers[domain] = {
                    'state': 'closed',
                    'failure_count': 0,
                    'last_failure_time': 0,
                    'last_success_time': 0
                }
                return True
                
            # If circuit is open, check if reset timeout has passed
            if circuit['state'] == 'open':
                reset_seconds = self._error_config.get('circuit_breaker_reset_seconds', 60)
                time_since_failure = time.time() - circuit['last_failure_time']
                
                if time_since_failure > reset_seconds:
                    # Reset to half-open
                    circuit['state'] = 'half-open'
                    logger.info(f"Circuit breaker for {domain} changed from open to half-open")
                else:
                    # Circuit still open, block request
                    return False
            
            # If half-open, allow one request
            if circuit['state'] == 'half-open':
                return True
                
            # If closed, allow request
            return True
    
    def _record_success(self, domain: str) -> None:
        """
        Record a successful request for circuit breaker.
        
        Args:
            domain: The domain or service to record success for
        """
        with self._circuit_lock:
            circuit = self._circuit_breakers.get(domain)
            
            if not circuit:
                # Create circuit if it doesn't exist
                self._circuit_breakers[domain] = {
                    'state': 'closed',
                    'failure_count': 0,
                    'last_failure_time': 0,
                    'last_success_time': time.time()
                }
                return
                
            # Update success time
            circuit['last_success_time'] = time.time()
                
            # If half-open, close the circuit
            if circuit['state'] == 'half-open':
                circuit['state'] = 'closed'
                circuit['failure_count'] = 0
                logger.info(f"Circuit breaker for {domain} changed from half-open to closed")
    
    def _record_failure(self, domain: str) -> None:
        """
        Record a failed request for circuit breaker.
        
        Args:
            domain: The domain or service to record failure for
        """
        with self._circuit_lock:
            circuit = self._circuit_breakers.get(domain)
            
            if not circuit:
                # Create circuit if it doesn't exist
                self._circuit_breakers[domain] = {
                    'state': 'closed',
                    'failure_count': 1,
                    'last_failure_time': time.time(),
                    'last_success_time': 0
                }
                return
                
            # Update failure time
            circuit['last_failure_time'] = time.time()
            
            # If half-open, open the circuit again
            if circuit['state'] == 'half-open':
                circuit['state'] = 'open'
                logger.warning(f"Circuit breaker for {domain} changed from half-open to open")
                return
                
            # If closed, increment failure count
            if circuit['state'] == 'closed':
                circuit['failure_count'] += 1
                
                # If failure count exceeds threshold, open the circuit
                threshold = self._error_config.get('circuit_breaker_threshold', 5)
                if circuit['failure_count'] >= threshold:
                    circuit['state'] = 'open'
                    logger.warning(f"Circuit breaker for {domain} changed from closed to open")
    
    def _make_request(self, url: str, method: str = 'GET', **kwargs) -> requests.Response:
        """
        Make an HTTP request with full error handling.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            CircuitOpenError: If circuit breaker is open
            requests.exceptions.RequestException: For request failures
        """
        # Get domain for rate limiting and circuit breaking
        domain = urlparse(url).netloc
        
        # Get proxy if configured
        proxy = kwargs.pop('proxy', self._get_proxy(domain))
        
        # Handle rate limiting
        self._handle_rate_limiting(domain)
        
        # Check circuit breaker
        if not self._check_circuit_breaker(domain):
            logger.warning(f"Circuit breaker open for {domain}, blocking request to {url}")
            raise CircuitOpenError(f"Circuit breaker open for {domain}")
        
        # Get session for this domain
        session = self._get_session(domain)
        
        # Set up request arguments
        request_kwargs = {
            'timeout': kwargs.pop('timeout', 30),
            'proxies': proxy,
            'verify': kwargs.pop('verify', True),
            'allow_redirects': kwargs.pop('allow_redirects', True)
        }
        
        # Add any additional kwargs
        request_kwargs.update(kwargs)
        
        # Track time for metrics
        start_time = time.time()
        
        try:
            # Make the request
            if method.upper() == 'GET':
                response = session.get(url, **request_kwargs)
            elif method.upper() == 'POST':
                response = session.post(url, **request_kwargs)
            elif method.upper() == 'PUT':
                response = session.put(url, **request_kwargs)
            elif method.upper() == 'DELETE':
                response = session.delete(url, **request_kwargs)
            elif method.upper() == 'HEAD':
                response = session.head(url, **request_kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Record success for circuit breaker
            self._record_success(domain)
            
            # Update rate limit based on response
            self._update_rate_limit(domain, response.status_code)
            
            # Record request metrics
            self._record_request_metrics(url, start_time, True)
            
            # Update proxy health
            if proxy and 'id' in proxy:
                self._update_proxy_health(domain, proxy['id'], True)
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.RequestException as e:
            # Record failure for circuit breaker
            self._record_failure(domain)
            
            # Record request metrics
            self._record_request_metrics(url, start_time, False)
            
            # Update proxy health
            if proxy and 'id' in proxy:
                self._update_proxy_health(domain, proxy['id'], False)
            
            # Classify and log the error
            error_context = {
                'url': url,
                'domain': domain,
                'method': method
            }
            error_info = self._classify_error(e, error_context)
            
            # Re-raise the exception
            raise
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get statistics about errors.
        
        Returns:
            Dictionary with error statistics
        """
        with self._error_lock, self._circuit_lock, self._retry_lock:
            # Count errors by category and severity
            categories = {}
            severities = {}
            
            for error in self._errors:
                category = error.get('category', 'unknown')
                severity = error.get('severity', 'unknown')
                
                categories[category] = categories.get(category, 0) + 1
                severities[severity] = severities.get(severity, 0) + 1
            
            # Get circuit breaker stats
            circuit_stats = {
                'total': len(self._circuit_breakers),
                'open': len([c for c in self._circuit_breakers.values() if c.get('state') == 'open']),
                'half_open': len([c for c in self._circuit_breakers.values() if c.get('state') == 'half-open']),
                'closed': len([c for c in self._circuit_breakers.values() if c.get('state') == 'closed'])
            }
            
            # Return all stats
            return {
                'errors': {
                    'total': len(self._errors),
                    'categories': categories,
                    'severities': severities
                },
                'circuit_breakers': circuit_stats,
                'retries': {
                    'total': sum(self._retry_counts.values()),
                    'operations': len(self._retry_counts)
                }
            }