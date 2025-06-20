import logging
import re
import requests
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Pattern
import threading
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of errors that can occur during scraping."""
    NETWORK = "network"           # Connection issues
    HTTP = "http"                 # Server responses
    CONTENT = "content"           # Content problems
    PARSING = "parsing"           # Content processing issues
    AUTHENTICATION = "authentication"  # Auth failures
    PERMISSION = "permission"     # Access denied issues
    RESOURCE = "resource"         # Resource limitations
    TIMEOUT = "timeout"           # Time-related failures
    VALIDATION = "validation"     # Input/output validation
    RATE_LIMIT = "rate_limit"     # Rate limiting
    CAPTCHA = "captcha"           # CAPTCHA challenges
    PROXY = "proxy"               # Proxy-related issues
    UNKNOWN = "unknown"           # Unclassified errors

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    INFO = "info"                 # Non-critical information
    WARNING = "warning"           # Potential issues
    ERROR = "error"               # Recoverable failures
    CRITICAL = "critical"         # Non-recoverable failures
    FATAL = "fatal"               # System-threatening issues

class ErrorClassifier(BaseService):
    """Service for classifying and analyzing errors to suggest remediation."""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._patterns = {}
        self._lock = threading.RLock()
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the error classifier with configuration."""
        if self._initialized:
            return
            
        self._config = config or {}
        
        # Initialize pattern databases
        self._init_captcha_patterns()
        self._init_ip_blocking_patterns()
        self._init_rate_limiting_patterns()
        self._init_error_message_patterns()
        self._init_server_error_patterns()
        self._init_auth_required_patterns()
        self._init_bot_protection_patterns()
        self._init_cloudflare_patterns()
        
        self._initialized = True
        logger.info("Error classifier initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return
        
        with self._lock:
            self._patterns.clear()
        
        self._initialized = False
        logger.info("Error classifier shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "error_classifier"
    
    def _init_captcha_patterns(self) -> None:
        """Initialize CAPTCHA detection patterns."""
        self._patterns['captcha'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('captcha_patterns', [
                r'captcha',
                r'robot check',
                r'human verification',
                r'are you a robot',
                r'prove you\'re human',
                r'security check',
                r'verify you are human',
                r'bot check',
                r'recaptcha',
                r'hcaptcha',
                r'solve this puzzle',
                r'complete the security check',
                r'please verify you are not a robot'
            ])
        ]
        
    def _init_ip_blocking_patterns(self) -> None:
        """Initialize IP blocking detection patterns."""
        self._patterns['ip_blocking'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('ip_blocking_patterns', [
                r'access denied',
                r'ip address has been blocked',
                r'your IP has been temporarily blocked',
                r'your access to this site has been limited',
                r'your IP address is blacklisted',
                r'suspicious activity',
                r'unusual traffic',
                r'automated requests',
                r'too many requests from your ip',
                r'banned',
                r'blocked for security reasons',
                r'access from your country is restricted'
            ])
        ]
        
    def _init_rate_limiting_patterns(self) -> None:
        """Initialize rate limiting detection patterns."""
        self._patterns['rate_limiting'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('rate_limiting_patterns', [
                r'rate limit(ed|ing)?',
                r'too many requests',
                r'request limit exceeded',
                r'slow down',
                r'try again later',
                r'quota exceeded',
                r'usage limit',
                r'too frequent',
                r'throttled',
                r'please wait before trying again',
                r'request frequency',
                r'too many attempts'
            ])
        ]
        
    def _init_error_message_patterns(self) -> None:
        """Initialize common error message patterns."""
        self._patterns['error_message'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('error_message_patterns', [
                r'error occurred',
                r'something went wrong',
                r'unexpected error',
                r'server error',
                r'service unavailable',
                r'internal error',
                r'error processing request',
                r'failed to process',
                r'error code',
                r'could not complete',
                r'service temporarily unavailable'
            ])
        ]
        
    def _init_server_error_patterns(self) -> None:
        """Initialize server error signature patterns."""
        self._patterns['server_error'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('server_error_patterns', [
                r'5\d\d error',
                r'internal server error',
                r'service temporarily unavailable',
                r'500 internal server error',
                r'502 bad gateway',
                r'503 service unavailable',
                r'504 gateway timeout',
                r'system overload',
                r'maintenance mode',
                r'database error',
                r'exception occurred',
                r'stack trace'
            ])
        ]
        
    def _init_auth_required_patterns(self) -> None:
        """Initialize authentication required patterns."""
        self._patterns['auth_required'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('auth_required_patterns', [
                r'login required',
                r'authentication required',
                r'please log in',
                r'sign in to continue',
                r'you need to be logged in',
                r'please authenticate',
                r'access denied',
                r'unauthorized',
                r'account required',
                r'membership required',
                r'please signin'
            ])
        ]
        
    def _init_bot_protection_patterns(self) -> None:
        """Initialize bot protection detection patterns."""
        self._patterns['bot_protection'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('bot_protection_patterns', [
                r'bot protection',
                r'automated access is prohibited',
                r'scraping is prohibited',
                r'bot detected',
                r'ddos protection',
                r'anti-bot',
                r'automated requests are blocked',
                r'non-human traffic',
                r'automated traffic detected',
                r'security challenge',
                r'browser check',
                r'javascript required'
            ])
        ]
        
    def _init_cloudflare_patterns(self) -> None:
        """Initialize Cloudflare protection patterns."""
        self._patterns['cloudflare'] = [
            re.compile(p, re.IGNORECASE) for p in self._config.get('cloudflare_patterns', [
                r'cloudflare',
                r'checking your browser',
                r'just a moment',
                r'ray id',
                r'challenge-platform',
                r'jschl_vc',
                r'waiting for cloudflare',
                r'cf-browser-verification',
                r'attention required',
                r'cf-ray',
                r'cf-chl'
            ])
        ]
    
    def classify_exception(self, exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify any exception and return classification details.
        
        Args:
            exception: The exception to classify
            context: Additional context information
            
        Returns:
            Dictionary with error classification details
        """
        context = context or {}
        
        # Initialize classification
        classification = {
            'original_exception': exception,
            'error_message': str(exception),
            'error_type': type(exception).__name__,
            'category': ErrorCategory.UNKNOWN,
            'severity': ErrorSeverity.ERROR,
            'is_retryable': False,
            'context': context,
            'suggested_actions': []
        }
        
        # Network-related errors
        if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)):
            classification['category'] = ErrorCategory.NETWORK
            classification['severity'] = ErrorSeverity.WARNING
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'check_connectivity', 'rotate_proxy']
            
        elif isinstance(exception, requests.exceptions.Timeout):
            classification['category'] = ErrorCategory.TIMEOUT
            classification['severity'] = ErrorSeverity.WARNING
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'increase_timeout', 'backoff']
            
        elif isinstance(exception, requests.exceptions.TooManyRedirects):
            classification['category'] = ErrorCategory.HTTP
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_url', 'update_referrer']
            
        elif isinstance(exception, requests.exceptions.RequestException):
            classification['category'] = ErrorCategory.HTTP
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'check_request_validity']
            
        elif isinstance(exception, requests.exceptions.HTTPError):
            # For HTTP errors, delegate to the more specific classifier
            response = getattr(exception, 'response', None)
            if response:
                return self.classify_http_error(response, context)
            
        elif isinstance(exception, ValueError) and "JSON" in str(exception):
            classification['category'] = ErrorCategory.PARSING
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_content_type', 'validate_endpoint']
            
        elif isinstance(exception, (SyntaxError, TypeError)):
            classification['category'] = ErrorCategory.PARSING
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_data_format', 'validate_schema']
            
        # Add metadata from the exception
        classification.update(self._extract_exception_metadata(exception))
            
        return classification
    
    def classify_http_error(self, response: Union[requests.Response, Exception], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify HTTP errors based on status code and response content.
        
        Args:
            response: HTTP response or exception with response attribute
            context: Additional context information
            
        Returns:
            Dictionary with error classification details
        """
        context = context or {}
        
        # Handle case where an exception with response attribute is passed
        if isinstance(response, Exception):
            response = getattr(response, 'response', None)
            if not response:
                return self.classify_exception(response, context)
        
        # Initialize classification
        classification = {
            'error_message': f"HTTP {response.status_code}: {response.reason}" if hasattr(response, 'reason') else f"HTTP {response.status_code}",
            'error_type': 'HTTPError',
            'category': ErrorCategory.HTTP,
            'severity': ErrorSeverity.ERROR,
            'is_retryable': False,
            'context': context,
            'suggested_actions': [],
            'response': response
        }
        
        # Add response metadata
        classification['status_code'] = response.status_code
        classification['url'] = response.url
        classification['headers'] = dict(response.headers) if hasattr(response, 'headers') else {}
        
        # Classify based on status code ranges
        status_code = response.status_code
        
        # 2xx Success (but still classified as an error for some reason)
        if 200 <= status_code < 300:
            classification['category'] = ErrorCategory.CONTENT
            classification['severity'] = ErrorSeverity.INFO
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_content_validity', 'validate_assumptions']
            
        # 3xx Redirection
        elif 300 <= status_code < 400:
            classification['category'] = ErrorCategory.HTTP
            classification['severity'] = ErrorSeverity.INFO
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['follow_redirect', 'update_url']
            
        # 4xx Client Errors
        elif 400 <= status_code < 500:
            classification = self._classify_4xx_error(response, classification, context)
            
        # 5xx Server Errors
        elif 500 <= status_code < 600:
            classification['category'] = ErrorCategory.HTTP
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['retry', 'backoff', 'reduce_concurrency']
            
            # Specific 5xx errors
            if status_code == 502:  # Bad Gateway
                classification['suggested_actions'].append('check_service_availability')
            elif status_code == 503:  # Service Unavailable
                classification['suggested_actions'].append('increase_backoff')
                
        # Analyze content for additional insights
        if hasattr(response, 'text') and response.text:
            content_classification = self.classify_content_error(response.text, context)
            
            # If content analysis found more specific issues, update classification
            if content_classification.get('category') != ErrorCategory.UNKNOWN:
                # Keep HTTP status info but use content classification for category and actions
                classification['category'] = content_classification['category']
                classification['content_analysis'] = content_classification
                
                # Merge suggested actions
                classification['suggested_actions'] = list(set(
                    classification['suggested_actions'] + content_classification['suggested_actions']
                ))
                
        return classification
    
    def _classify_4xx_error(self, response, classification, context):
        """Classify 4xx HTTP status errors more specifically."""
        status_code = response.status_code
        
        # 400 Bad Request
        if status_code == 400:
            classification['category'] = ErrorCategory.VALIDATION
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['check_request_format', 'validate_parameters']
            
        # 401 Unauthorized
        elif status_code == 401:
            classification['category'] = ErrorCategory.AUTHENTICATION
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['refresh_authentication', 'check_credentials']
            
        # 403 Forbidden
        elif status_code == 403:
            # Check for specific cases
            if self.detect_captcha(response):
                classification['category'] = ErrorCategory.CAPTCHA
                classification['severity'] = ErrorSeverity.ERROR
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['solve_captcha', 'rotate_proxy', 'use_browser_session']
            elif self.detect_ip_blocking(response):
                classification['category'] = ErrorCategory.PERMISSION
                classification['severity'] = ErrorSeverity.CRITICAL
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['rotate_proxy', 'change_user_agent', 'backoff']
            elif self.detect_cloudflare_protection(response):
                classification['category'] = ErrorCategory.PERMISSION
                classification['severity'] = ErrorSeverity.CRITICAL
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['use_browser_session', 'handle_js_challenge', 'rotate_proxy']
            else:
                classification['category'] = ErrorCategory.PERMISSION
                classification['severity'] = ErrorSeverity.ERROR
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['check_permissions', 'rotate_proxy', 'modify_headers']
                
        # 404 Not Found
        elif status_code == 404:
            classification['category'] = ErrorCategory.HTTP
            classification['severity'] = ErrorSeverity.ERROR
            classification['is_retryable'] = False
            classification['suggested_actions'] = ['verify_url', 'check_resource_exists']
            
        # 429 Too Many Requests
        elif status_code == 429:
            classification['category'] = ErrorCategory.RATE_LIMIT
            classification['severity'] = ErrorSeverity.WARNING
            classification['is_retryable'] = True
            classification['suggested_actions'] = ['backoff', 'reduce_rate', 'respect_retry_after', 'rotate_proxy']
            
            # Extract Retry-After header if available
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    retry_seconds = int(retry_after)
                    classification['retry_after'] = retry_seconds
                except ValueError:
                    # It might be an HTTP date format
                    classification['retry_after'] = retry_after
                    
        return classification
    
    def classify_content_error(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze content to detect error patterns even when HTTP status is 200.
        
        Args:
            content: The HTML or text content to analyze
            context: Additional context information
            
        Returns:
            Dictionary with error classification details
        """
        context = context or {}
        
        # Initialize classification with default "no error detected"
        classification = {
            'category': ErrorCategory.UNKNOWN,
            'severity': ErrorSeverity.INFO,
            'is_retryable': True,
            'context': context,
            'suggested_actions': [],
            'detected_patterns': []
        }
        
        # Use detect_error_patterns to find any error indicators
        error_patterns = self.detect_error_patterns(content, context)
        
        if error_patterns:
            # Update classification with detected patterns
            classification['detected_patterns'] = error_patterns
            
            # Determine the primary error category from patterns
            if any(p.get('type') == 'captcha' for p in error_patterns):
                classification['category'] = ErrorCategory.CAPTCHA
                classification['severity'] = ErrorSeverity.ERROR
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['solve_captcha', 'rotate_proxy']
                
            elif any(p.get('type') == 'ip_blocking' for p in error_patterns):
                classification['category'] = ErrorCategory.PERMISSION
                classification['severity'] = ErrorSeverity.CRITICAL
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['rotate_proxy', 'change_user_agent']
                
            elif any(p.get('type') == 'rate_limiting' for p in error_patterns):
                classification['category'] = ErrorCategory.RATE_LIMIT
                classification['severity'] = ErrorSeverity.WARNING
                classification['is_retryable'] = True
                classification['suggested_actions'] = ['backoff', 'reduce_rate']
                
            elif any(p.get('type') == 'auth_required' for p in error_patterns):
                classification['category'] = ErrorCategory.AUTHENTICATION
                classification['severity'] = ErrorSeverity.ERROR
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['authenticate', 'check_credentials']
                
            elif any(p.get('type') == 'cloudflare' for p in error_patterns):
                classification['category'] = ErrorCategory.PERMISSION
                classification['severity'] = ErrorSeverity.CRITICAL
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['use_browser_session', 'handle_js_challenge']
                
            elif any(p.get('type') == 'bot_protection' for p in error_patterns):
                classification['category'] = ErrorCategory.PERMISSION
                classification['severity'] = ErrorSeverity.CRITICAL
                classification['is_retryable'] = False
                classification['suggested_actions'] = ['use_browser_session', 'stealth_mode']
                
            elif any(p.get('type') == 'server_error' for p in error_patterns):
                classification['category'] = ErrorCategory.HTTP
                classification['severity'] = ErrorSeverity.ERROR
                classification['is_retryable'] = True
                classification['suggested_actions'] = ['retry', 'backoff']
                
            else:
                # Generic error detected but not specifically categorized
                classification['category'] = ErrorCategory.CONTENT
                classification['severity'] = ErrorSeverity.WARNING
                classification['is_retryable'] = True
                classification['suggested_actions'] = ['analyze_content', 'check_selectors']
            
        # Check if content is empty or extremely small, which could indicate an error
        if len(content.strip()) < 100:
            classification['category'] = ErrorCategory.CONTENT
            classification['severity'] = ErrorSeverity.WARNING
            classification['suggested_actions'].append('verify_content_extraction')
            classification['detected_patterns'].append({'type': 'empty_content', 'match': 'Content too short'})
            
        return classification
    
    def detect_error_patterns(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Detect error message patterns in text content.
        
        Args:
            text: The text content to analyze
            context: Additional context information
            
        Returns:
            List of detected patterns with type and match info
        """
        context = context or {}
        detected_patterns = []
        
        # Make a clean version of the text for pattern matching
        clean_text = text.lower().strip()
        
        # Check all pattern types
        for pattern_type, patterns in self._patterns.items():
            for pattern in patterns:
                matches = pattern.findall(clean_text)
                if matches:
                    detected_patterns.append({
                        'type': pattern_type,
                        'pattern': pattern.pattern,
                        'match': matches[0] if matches else None
                    })
        
        return detected_patterns
    
    def detect_captcha(self, content: Union[str, requests.Response]) -> bool:
        """
        Identify CAPTCHA challenges in content or response.
        
        Args:
            content: HTML content or Response object
            
        Returns:
            True if CAPTCHA is detected, False otherwise
        """
        # Extract text content from response if needed
        if isinstance(content, requests.Response):
            if not hasattr(content, 'text'):
                return False
            content = content.text
            
        # Quick check for captcha-related keywords
        if not content or not any(kw in content.lower() for kw in ['captcha', 'robot', 'human']):
            return False
            
        # Use pattern detection for comprehensive check
        for pattern in self._patterns.get('captcha', []):
            if pattern.search(content.lower()):
                return True
                
        # Check for common CAPTCHA services
        captcha_indicators = [
            'recaptcha', 'g-recaptcha', 'hcaptcha', 'h-captcha',
            'captcha-challenge', 'captcha-script', 'captcha.js'
        ]
        
        return any(indicator in content.lower() for indicator in captcha_indicators)
    
    def detect_login_required(self, content: Union[str, requests.Response], response=None) -> bool:
        """
        Identify authentication walls in content.
        
        Args:
            content: HTML content or Response object
            response: Optional Response object if content is a string
            
        Returns:
            True if login is required, False otherwise
        """
        # Handle response object
        if isinstance(content, requests.Response):
            response = content
            content = response.text
            
        # Check status code if response is provided
        if response and response.status_code == 401:
            return True
            
        # Check for login-related keywords
        return any(pattern.search(content.lower()) for pattern in self._patterns.get('auth_required', []))
    
    def detect_ip_blocking(self, content: Union[str, requests.Response], response=None) -> bool:
        """
        Identify IP blocking or restrictions.
        
        Args:
            content: HTML content or Response object
            response: Optional Response object if content is a string
            
        Returns:
            True if IP blocking is detected, False otherwise
        """
        # Handle response object
        if isinstance(content, requests.Response):
            response = content
            content = response.text
            
        # Check for IP blocking patterns
        if any(pattern.search(content.lower()) for pattern in self._patterns.get('ip_blocking', [])):
            return True
            
        # Check for common IP blocking status + header combinations
        if response and response.status_code in [403, 429]:
            if any(h.lower() in [k.lower() for k in response.headers] for h in ['X-Banned', 'X-Block', 'CF-Ban']):
                return True
                
        return False
    
    def detect_cloudflare_protection(self, response: requests.Response) -> bool:
        """
        Identify Cloudflare protection on a response.
        
        Args:
            response: The response to check
            
        Returns:
            True if Cloudflare protection is detected, False otherwise
        """
        # No response to check
        if not response:
            return False
            
        # Check for Cloudflare headers
        headers = response.headers
        if any(h in headers for h in ['CF-RAY', 'CF-Cache-Status', 'cf-request-id']):
            return True
            
        # Check content for Cloudflare patterns
        if hasattr(response, 'text') and response.text:
            for pattern in self._patterns.get('cloudflare', []):
                if pattern.search(response.text.lower()):
                    return True
                    
        return False
    
    def detect_bot_protection(self, content: Union[str, requests.Response], response=None) -> bool:
        """
        Identify bot protection mechanisms.
        
        Args:
            content: HTML content or Response object
            response: Optional Response object if content is a string
            
        Returns:
            True if bot protection is detected, False otherwise
        """
        # Handle response object
        if isinstance(content, requests.Response):
            response = content
            content = response.text
            
        # Check for bot protection patterns
        if any(pattern.search(content.lower()) for pattern in self._patterns.get('bot_protection', [])):
            return True
            
        # Check for JavaScript challenges (often used in bot protection)
        js_challenge_indicators = [
            'please enable javascript',
            'javascript is required',
            'browser check',
            'checking your browser'
        ]
        
        return any(indicator in content.lower() for indicator in js_challenge_indicators)
    
    def suggest_retry_strategy(self, error_type: ErrorCategory, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest a retry strategy based on error type and context.
        
        Args:
            error_type: The type of error
            context: Additional context information
            
        Returns:
            Dictionary with retry strategy parameters
        """
        context = context or {}
        retry_strategy = {
            'should_retry': False,
            'max_retries': 3,
            'backoff_factor': 1.0,
            'status_forcelist': [500, 502, 503, 504],
            'backoff_jitter': True
        }
        
        # Customize retry strategy based on error type
        if error_type == ErrorCategory.NETWORK:
            retry_strategy['should_retry'] = True
            retry_strategy['max_retries'] = 5
            retry_strategy['backoff_factor'] = 2.0
            
        elif error_type == ErrorCategory.HTTP:
            retry_strategy['should_retry'] = True
            retry_strategy['max_retries'] = 3
            retry_strategy['backoff_factor'] = 1.5
            
        elif error_type == ErrorCategory.TIMEOUT:
            retry_strategy['should_retry'] = True
            retry_strategy['max_retries'] = 3
            retry_strategy['backoff_factor'] = 2.0
            
        elif error_type == ErrorCategory.RATE_LIMIT:
            retry_strategy['should_retry'] = True
            retry_strategy['max_retries'] = 5
            retry_strategy['backoff_factor'] = 5.0
            retry_strategy['status_forcelist'] = [429]
            
            # Use context if available
            if 'retry_after' in context:
                retry_strategy['retry_after'] = context['retry_after']
                
        else:
            # Default for other error types
            retry_strategy['should_retry'] = False
            
        return retry_strategy
    
    def suggest_proxy_action(self, error_type: ErrorCategory, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest proxy-related actions based on error type.
        
        Args:
            error_type: The type of error
            context: Additional context information
            
        Returns:
            Dictionary with proxy action recommendations
        """
        context = context or {}
        proxy_action = {
            'rotate_proxy': False,
            'use_premium_proxy': False,
            'use_residential': False,
            'change_geo': False,
            'blacklist_current': False
        }
        
        # Customize proxy action based on error type
        if error_type in [ErrorCategory.PERMISSION, ErrorCategory.RATE_LIMIT]:
            proxy_action['rotate_proxy'] = True
            proxy_action['blacklist_current'] = True
            
        elif error_type == ErrorCategory.CAPTCHA:
            proxy_action['rotate_proxy'] = True
            proxy_action['use_residential'] = True
            
        elif error_type == ErrorCategory.NETWORK:
            proxy_action['rotate_proxy'] = True
            proxy_action['blacklist_current'] = True if context.get('failures', 0) > 2 else False
            
        # Consider domain-specific recommendations
        domain = context.get('domain', '')
        if domain:
            if any(site in domain for site in ['google', 'amazon', 'facebook']):
                proxy_action['use_residential'] = True
                
        return proxy_action
    
    def suggest_session_action(self, error_type: ErrorCategory, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest session-related actions based on error type.
        
        Args:
            error_type: The type of error
            context: Additional context information
            
        Returns:
            Dictionary with session action recommendations
        """
        context = context or {}
        session_action = {
            'rotate_user_agent': False,
            'clear_cookies': False,
            'use_browser_session': False,
            'refresh_auth': False,
            'need_cookies': False
        }
        
        # Customize session action based on error type
        if error_type == ErrorCategory.AUTHENTICATION:
            session_action['refresh_auth'] = True
            session_action['clear_cookies'] = True
            
        elif error_type == ErrorCategory.PERMISSION:
            session_action['rotate_user_agent'] = True
            session_action['clear_cookies'] = True
            
        elif error_type == ErrorCategory.CAPTCHA:
            session_action['use_browser_session'] = True
            session_action['rotate_user_agent'] = True
            
        elif error_type in [ErrorCategory.HTTP, ErrorCategory.CONTENT]:
            session_action['rotate_user_agent'] = True
            if context.get('status_code') == 403:
                session_action['clear_cookies'] = True
                
        return session_action
    
    def suggest_throttling(self, error_type: ErrorCategory, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Suggest rate adjustments based on error type.
        
        Args:
            error_type: The type of error
            context: Additional context information
            
        Returns:
            Dictionary with throttling recommendations
        """
        context = context or {}
        throttling = {
            'reduce_rate': False,
            'reduce_concurrency': False,
            'increase_delay': 0,
            'adaptive_rate': False
        }
        
        # Customize throttling based on error type
        if error_type == ErrorCategory.RATE_LIMIT:
            throttling['reduce_rate'] = True
            throttling['increase_delay'] = 5.0
            throttling['adaptive_rate'] = True
            
            # Use retry-after header if available
            if 'retry_after' in context:
                retry_after = context['retry_after']
                if isinstance(retry_after, (int, float)):
                    throttling['increase_delay'] = max(throttling['increase_delay'], retry_after)
                    
        elif error_type == ErrorCategory.PERMISSION:
            throttling['reduce_rate'] = True
            throttling['increase_delay'] = 2.0
            
        elif error_type == ErrorCategory.HTTP and context.get('status_code', 0) >= 500:
            throttling['reduce_concurrency'] = True
            throttling['increase_delay'] = 1.0
            
        return throttling
    
    def get_recovery_actions(self, error_type: ErrorCategory, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive recovery actions for an error.
        
        Args:
            error_type: The type of error
            context: Additional context information
            
        Returns:
            Dictionary with all recovery action recommendations
        """
        context = context or {}
        
        # Combine all recommendations
        recovery = {
            'retry': self.suggest_retry_strategy(error_type, context),
            'proxy': self.suggest_proxy_action(error_type, context),
            'session': self.suggest_session_action(error_type, context),
            'throttling': self.suggest_throttling(error_type, context)
        }
        
        # Add high-level actions list
        recovery['actions'] = []
        
        # Build action list from all recommendations
        if recovery['retry']['should_retry']:
            recovery['actions'].append('retry')
            
        if recovery['proxy']['rotate_proxy']:
            recovery['actions'].append('rotate_proxy')
            
        if recovery['session']['rotate_user_agent']:
            recovery['actions'].append('rotate_user_agent')
            
        if recovery['session']['use_browser_session']:
            recovery['actions'].append('use_browser_session')
            
        if recovery['throttling']['reduce_rate']:
            recovery['actions'].append('reduce_rate')
            
        # Add error-specific actions
        if error_type == ErrorCategory.CAPTCHA:
            recovery['actions'].append('solve_captcha')
            
        elif error_type == ErrorCategory.AUTHENTICATION:
            recovery['actions'].append('refresh_authentication')
            
        return recovery
    
    def _extract_exception_metadata(self, exception: Exception) -> Dict[str, Any]:
        """Extract metadata from an exception."""
        metadata = {}
        
        # Extract URL from the exception if present
        if hasattr(exception, 'url'):
            metadata['url'] = exception.url
            
            # Extract domain from URL
            try:
                parsed_url = urlparse(exception.url)
                metadata['domain'] = parsed_url.netloc
            except:
                pass
                
        # Extract response from the exception if present
        if hasattr(exception, 'response'):
            metadata['status_code'] = getattr(exception.response, 'status_code', None)
            metadata['headers'] = dict(getattr(exception.response, 'headers', {}))
            
        return metadata