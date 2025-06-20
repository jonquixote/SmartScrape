import pytest
import requests
import requests.exceptions
from unittest.mock import MagicMock, patch
import json
import re

from core.error_classifier import ErrorClassifier, ErrorCategory, ErrorSeverity

class TestErrorClassifier:
    """Tests for the ErrorClassifier service."""
    
    def setup_method(self):
        """Set up the test environment."""
        self.error_classifier = ErrorClassifier()
        self.error_classifier.initialize()
    
    def teardown_method(self):
        """Clean up the test environment."""
        self.error_classifier.shutdown()
    
    def test_initialization(self):
        """Test that the error classifier initializes correctly."""
        assert self.error_classifier.is_initialized
        assert self.error_classifier.name == "error_classifier"
        
        # Test pattern databases are initialized
        assert 'captcha' in self.error_classifier._patterns
        assert 'ip_blocking' in self.error_classifier._patterns
        assert 'rate_limiting' in self.error_classifier._patterns
        assert 'error_message' in self.error_classifier._patterns
        assert 'server_error' in self.error_classifier._patterns
        
        # Test patterns are compiled regex objects
        assert all(isinstance(p, re.Pattern) for p in self.error_classifier._patterns['captcha'])
    
    def test_classify_network_exception(self):
        """Test classification of network-related exceptions."""
        # Test ConnectionError
        exception = requests.exceptions.ConnectionError("Failed to connect")
        classification = self.error_classifier.classify_exception(exception)
        
        assert classification['category'] == ErrorCategory.NETWORK
        assert classification['severity'] == ErrorSeverity.WARNING
        assert classification['is_retryable'] == True
        assert 'retry' in classification['suggested_actions']
        assert 'check_connectivity' in classification['suggested_actions']
        
        # Test Timeout
        exception = requests.exceptions.Timeout("Request timed out")
        classification = self.error_classifier.classify_exception(exception)
        
        assert classification['category'] == ErrorCategory.TIMEOUT
        assert classification['severity'] == ErrorSeverity.WARNING
        assert classification['is_retryable'] == True
        assert 'increase_timeout' in classification['suggested_actions']
    
    def test_classify_http_exception(self):
        """Test classification of HTTP exceptions."""
        # Mock a response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.url = "https://example.com/missing"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.text = "<html><body><h1>Not Found</h1></body></html>"
        
        # Create an HTTPError with the mock response
        exception = requests.exceptions.HTTPError("404 Client Error: Not Found", response=mock_response)
        
        # Test classification
        classification = self.error_classifier.classify_exception(exception)
        
        assert classification['category'] == ErrorCategory.HTTP
        assert classification['status_code'] == 404
        assert classification['is_retryable'] == False
        assert 'verify_url' in classification['suggested_actions']
    
    def test_classify_parsing_exception(self):
        """Test classification of parsing exceptions."""
        # JSON parsing error
        exception = ValueError("JSON decode error")
        classification = self.error_classifier.classify_exception(exception)
        
        assert classification['category'] == ErrorCategory.UNKNOWN
        
        # More specific JSON error
        exception = ValueError("Expecting property name: line 1 column 2 (char 1)")
        classification = self.error_classifier.classify_exception(exception, {'context': 'JSON parsing'})
        
        assert classification['category'] == ErrorCategory.UNKNOWN
        
        # Syntax error
        exception = SyntaxError("invalid syntax")
        classification = self.error_classifier.classify_exception(exception)
        
        assert classification['category'] == ErrorCategory.PARSING
        assert classification['severity'] == ErrorSeverity.ERROR
        assert 'check_data_format' in classification['suggested_actions']
    
    def test_classify_http_error_codes(self):
        """Test classification of different HTTP status codes."""
        # Test 400 Bad Request
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.reason = "Bad Request"
        mock_response.url = "https://example.com/api"
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.text = json.dumps({"error": "Invalid parameters"})
        
        classification = self.error_classifier.classify_http_error(mock_response)
        
        assert classification['category'] == ErrorCategory.VALIDATION
        assert 'check_request_format' in classification['suggested_actions']
        
        # Test 401 Unauthorized
        mock_response.status_code = 401
        mock_response.reason = "Unauthorized"
        mock_response.text = "Authentication required"
        
        classification = self.error_classifier.classify_http_error(mock_response)
        
        assert classification['category'] == ErrorCategory.AUTHENTICATION
        assert 'refresh_authentication' in classification['suggested_actions']
        
        # Test 429 Too Many Requests
        mock_response.status_code = 429
        mock_response.reason = "Too Many Requests"
        mock_response.headers = {'Retry-After': '60', 'Content-Type': 'text/html'}
        mock_response.text = "Rate limit exceeded. Please try again later."
        
        classification = self.error_classifier.classify_http_error(mock_response)
        
        assert classification['category'] == ErrorCategory.RATE_LIMIT
        assert 'backoff' in classification['suggested_actions']
        assert 'reduce_rate' in classification['suggested_actions']
        assert classification['retry_after'] == 60
        
        # Test 500 Server Error
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.text = "An error occurred on the server."
        
        classification = self.error_classifier.classify_http_error(mock_response)
        
        assert classification['category'] == ErrorCategory.HTTP
        assert classification['is_retryable'] == True
        assert 'retry' in classification['suggested_actions']
    
    def test_detect_captcha(self):
        """Test CAPTCHA detection."""
        # Test positive detection
        captcha_content = """
        <html>
        <body>
            <h1>Security Check</h1>
            <div>
                Please complete the CAPTCHA below to continue.
                <div class="g-recaptcha" data-sitekey="..."></div>
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_captcha(captcha_content) == True
        
        # Test negative detection
        normal_content = """
        <html>
        <body>
            <h1>Welcome to our site</h1>
            <div>
                This is a normal page without any CAPTCHA.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_captcha(normal_content) == False
    
    def test_detect_login_required(self):
        """Test login detection."""
        # Test positive detection
        login_content = """
        <html>
        <body>
            <h1>Login Required</h1>
            <div>
                Please sign in to continue accessing this content.
                <form>
                    <input type="text" placeholder="Username">
                    <input type="password" placeholder="Password">
                    <button>Login</button>
                </form>
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_login_required(login_content) == True
        
        # Test negative detection
        normal_content = """
        <html>
        <body>
            <h1>Welcome to our site</h1>
            <div>
                This is public content that doesn't require login.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_login_required(normal_content) == False
    
    def test_detect_ip_blocking(self):
        """Test IP blocking detection."""
        # Test positive detection
        blocked_content = """
        <html>
        <body>
            <h1>Access Denied</h1>
            <div>
                Your IP address has been temporarily blocked due to suspicious activity.
                Please try again later.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_ip_blocking(blocked_content) == True
        
        # Test negative detection
        normal_content = """
        <html>
        <body>
            <h1>Welcome to our site</h1>
            <div>
                This is a normal page without any blocks.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_ip_blocking(normal_content) == False
    
    def test_detect_cloudflare_protection(self):
        """Test Cloudflare protection detection."""
        # Create a mock response with Cloudflare headers
        mock_response = MagicMock()
        mock_response.headers = {
            'CF-RAY': '6a8c0df86885e886-LAX',
            'CF-Cache-Status': 'DYNAMIC'
        }
        mock_response.text = """
        <html>
        <body>
            <h1>Just a moment...</h1>
            <div>
                Checking your browser before accessing the website.
                <div id="challenge-running"></div>
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_cloudflare_protection(mock_response) == True
        
        # Test negative detection
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.text = """
        <html>
        <body>
            <h1>Welcome to our site</h1>
            <div>
                This is a normal page without Cloudflare protection.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_cloudflare_protection(mock_response) == False
    
    def test_detect_bot_protection(self):
        """Test bot protection detection."""
        # Test positive detection
        bot_protection_content = """
        <html>
        <body>
            <h1>Bot Protection Active</h1>
            <div>
                We detected automated requests coming from your device.
                Please verify you are not a bot by completing the challenge.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_bot_protection(bot_protection_content) == True
        
        # Test negative detection
        normal_content = """
        <html>
        <body>
            <h1>Welcome to our site</h1>
            <div>
                This is a normal page without bot protection.
            </div>
        </body>
        </html>
        """
        
        assert self.error_classifier.detect_bot_protection(normal_content) == False
    
    def test_detect_error_patterns(self):
        """Test error pattern detection."""
        # Test mixed error content
        mixed_error_content = """
        <html>
        <body>
            <h1>Error: Rate Limit Exceeded</h1>
            <div>
                You have made too many requests. Please try again later.
                If you believe this is an error, please contact support.
            </div>
        </body>
        </html>
        """
        
        patterns = self.error_classifier.detect_error_patterns(mixed_error_content)
        
        # Should detect rate limiting patterns
        assert any(p['type'] == 'rate_limiting' for p in patterns)
        
        # Test server error content
        server_error_content = """
        <html>
        <body>
            <h1>500 Internal Server Error</h1>
            <div>
                Something went wrong on our end. We're working to fix it.
                Please try again later.
            </div>
        </body>
        </html>
        """
        
        patterns = self.error_classifier.detect_error_patterns(server_error_content)
        
        # Should detect server error patterns
        assert any(p['type'] == 'server_error' for p in patterns)
    
    def test_classify_content_error(self):
        """Test content error classification."""
        # Test CAPTCHA in content
        captcha_content = """
        <html>
        <body>
            <h1>Please verify you're human</h1>
            <div>
                Complete the CAPTCHA below to continue.
                <div class="g-recaptcha" data-sitekey="..."></div>
            </div>
        </body>
        </html>
        """
        
        classification = self.error_classifier.classify_content_error(captcha_content)
        
        assert classification['category'] == ErrorCategory.CAPTCHA
        assert 'solve_captcha' in classification['suggested_actions']
        
        # Test rate limiting in content
        rate_limit_content = """
        <html>
        <body>
            <h1>Rate Limit Exceeded</h1>
            <div>
                You have exceeded your rate limit. Please slow down and try again later.
            </div>
        </body>
        </html>
        """
        
        classification = self.error_classifier.classify_content_error(rate_limit_content)
        
        assert classification['category'] == ErrorCategory.RATE_LIMIT
        assert 'backoff' in classification['suggested_actions']
        
        # Test empty content
        empty_content = "   "
        
        classification = self.error_classifier.classify_content_error(empty_content)
        
        assert classification['category'] == ErrorCategory.CONTENT
        assert 'verify_content_extraction' in classification['suggested_actions']
    
    def test_suggest_retry_strategy(self):
        """Test retry strategy suggestions."""
        # Test network errors
        retry_strategy = self.error_classifier.suggest_retry_strategy(ErrorCategory.NETWORK)
        
        assert retry_strategy['should_retry'] == True
        assert retry_strategy['max_retries'] == 5
        assert retry_strategy['backoff_factor'] > 1.0
        
        # Test rate limit errors with retry-after
        context = {'retry_after': 30}
        retry_strategy = self.error_classifier.suggest_retry_strategy(ErrorCategory.RATE_LIMIT, context)
        
        assert retry_strategy['should_retry'] == True
        assert retry_strategy['retry_after'] == 30
        assert 429 in retry_strategy['status_forcelist']
        
        # Test non-retryable errors
        retry_strategy = self.error_classifier.suggest_retry_strategy(ErrorCategory.AUTHENTICATION)
        
        assert retry_strategy['should_retry'] == False
    
    def test_suggest_proxy_action(self):
        """Test proxy action suggestions."""
        # Test permission errors
        proxy_action = self.error_classifier.suggest_proxy_action(ErrorCategory.PERMISSION)
        
        assert proxy_action['rotate_proxy'] == True
        assert proxy_action['blacklist_current'] == True
        
        # Test with domain context for known sites
        context = {'domain': 'google.com'}
        proxy_action = self.error_classifier.suggest_proxy_action(ErrorCategory.PERMISSION, context)
        
        assert proxy_action['rotate_proxy'] == True
        assert proxy_action['use_residential'] == True
        
        # Test CAPTCHA errors
        proxy_action = self.error_classifier.suggest_proxy_action(ErrorCategory.CAPTCHA)
        
        assert proxy_action['rotate_proxy'] == True
        assert proxy_action['use_residential'] == True
    
    def test_suggest_session_action(self):
        """Test session action suggestions."""
        # Test authentication errors
        session_action = self.error_classifier.suggest_session_action(ErrorCategory.AUTHENTICATION)
        
        assert session_action['refresh_auth'] == True
        assert session_action['clear_cookies'] == True
        
        # Test CAPTCHA errors
        session_action = self.error_classifier.suggest_session_action(ErrorCategory.CAPTCHA)
        
        assert session_action['use_browser_session'] == True
        assert session_action['rotate_user_agent'] == True
        
        # Test HTTP errors with status code context
        context = {'status_code': 403}
        session_action = self.error_classifier.suggest_session_action(ErrorCategory.HTTP, context)
        
        assert session_action['rotate_user_agent'] == True
        assert session_action['clear_cookies'] == True
    
    def test_suggest_throttling(self):
        """Test throttling suggestions."""
        # Test rate limit errors
        throttling = self.error_classifier.suggest_throttling(ErrorCategory.RATE_LIMIT)
        
        assert throttling['reduce_rate'] == True
        assert throttling['increase_delay'] > 0
        assert throttling['adaptive_rate'] == True
        
        # Test with retry-after context
        context = {'retry_after': 60}
        throttling = self.error_classifier.suggest_throttling(ErrorCategory.RATE_LIMIT, context)
        
        assert throttling['increase_delay'] >= 60
        
        # Test server errors
        context = {'status_code': 503}
        throttling = self.error_classifier.suggest_throttling(ErrorCategory.HTTP, context)
        
        assert throttling['reduce_concurrency'] == True
    
    def test_get_recovery_actions(self):
        """Test comprehensive recovery action generation."""
        # Test rate limit recovery
        recovery = self.error_classifier.get_recovery_actions(ErrorCategory.RATE_LIMIT)
        
        assert 'retry' in recovery
        assert 'proxy' in recovery
        assert 'session' in recovery
        assert 'throttling' in recovery
        assert 'actions' in recovery
        
        assert 'retry' in recovery['actions']
        assert 'reduce_rate' in recovery['actions']
        
        # Test CAPTCHA recovery
        recovery = self.error_classifier.get_recovery_actions(ErrorCategory.CAPTCHA)
        
        assert 'solve_captcha' in recovery['actions']
        assert recovery['proxy']['use_residential'] == True
        assert recovery['session']['use_browser_session'] == True
        
        # Test authentication recovery
        recovery = self.error_classifier.get_recovery_actions(ErrorCategory.AUTHENTICATION)
        
        assert 'refresh_authentication' in recovery['actions']
        assert recovery['session']['refresh_auth'] == True
    
    def test_custom_patterns(self):
        """Test using custom patterns provided in configuration."""
        # Create a new classifier with custom patterns
        custom_classifier = ErrorClassifier()
        custom_config = {
            'captcha_patterns': [
                r'custom captcha pattern',
                r'security verification required'
            ],
            'ip_blocking_patterns': [
                r'custom ip block message',
                r'access from your location is restricted'
            ]
        }
        custom_classifier.initialize(custom_config)
        
        # Test custom captcha patterns
        captcha_content = "Security verification required to continue"
        assert custom_classifier.detect_captcha(captcha_content) == True
        
        # Test custom IP blocking patterns
        blocked_content = "Access from your location is restricted due to security policy"
        assert custom_classifier.detect_ip_blocking(blocked_content) == True
        
        # Cleanup
        custom_classifier.shutdown()
    
    def test_extract_exception_metadata(self):
        """Test extraction of metadata from exceptions."""
        # Create an exception with URL and response
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {'X-RateLimit-Remaining': '0'}
        
        exception = requests.exceptions.HTTPError("403 Forbidden")
        exception.response = mock_response
        exception.url = "https://api.example.com/resource"
        
        metadata = self.error_classifier._extract_exception_metadata(exception)
        
        assert metadata['url'] == "https://api.example.com/resource"
        assert metadata['domain'] == "api.example.com"
        assert metadata['status_code'] == 403
        assert 'headers' in metadata
    
    def test_error_category_enums(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.HTTP.value == "http"
        assert ErrorCategory.CONTENT.value == "content"
        assert ErrorCategory.PARSING.value == "parsing"
        assert ErrorCategory.AUTHENTICATION.value == "authentication"
        assert ErrorCategory.PERMISSION.value == "permission"
        assert ErrorCategory.RESOURCE.value == "resource"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.CAPTCHA.value == "captcha"
        assert ErrorCategory.PROXY.value == "proxy"
        assert ErrorCategory.UNKNOWN.value == "unknown"
    
    def test_error_severity_enums(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.FATAL.value == "fatal"