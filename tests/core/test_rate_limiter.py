import pytest
import time
import threading
import responses
from unittest.mock import MagicMock, patch
import requests
from http import HTTPStatus

from core.service_registry import ServiceRegistry
from core.rate_limiter import (
    RateLimiter, 
    TokenBucket, 
    FixedWindowCounter, 
    SlidingWindowCounter,
    ConcurrencyLimiter,
    DomainPolicy
)


@pytest.fixture
def rate_limiter():
    """Fixture to get a clean RateLimiter instance with basic config."""
    limiter = RateLimiter()
    limiter.initialize({
        'default_rpm': 60,
        'default_concurrent_requests': 5,
        'backoff_factor': 2.0,
        'default_policy_type': 'token_bucket',
        'domain_policies': {
            'example.com': {
                'rpm': 30,
                'max_concurrent': 3,
                'policy_type': 'token_bucket'
            },
            'api.example.com': {
                'rpm': 10,
                'max_concurrent': 2,
                'policy_type': 'sliding_window'
            },
            'highvolume.com': {
                'rpm': 120,
                'max_concurrent': 10,
                'policy_type': 'fixed_window'
            }
        },
        'domain_groups': {
            'example_group': {
                'domains': ['sub1.example.com', 'sub2.example.com'],
                'policy': {
                    'rpm': 20,
                    'max_concurrent': 3
                }
            }
        }
    })
    
    yield limiter
    
    # Clean up
    limiter.shutdown()


class TestTokenBucket:
    """Test the TokenBucket rate limiting algorithm."""

    def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, fill_rate=1)
        assert bucket.capacity == 10
        assert bucket.fill_rate == 1
        assert bucket.tokens == 10

    def test_get_token(self):
        """Test getting tokens from the bucket."""
        bucket = TokenBucket(capacity=10, fill_rate=1)
        
        # Should be able to get tokens up to capacity
        for _ in range(10):
            assert bucket.get_token() is True
            
        # Bucket should be empty now
        assert bucket.get_token() is False
        
        # Should refill over time
        with patch('time.time', return_value=time.time() + 5):
            bucket._refill()
            # Should have 5 tokens now
            assert bucket.tokens == 5
            assert bucket.get_token() is True
            assert bucket.tokens == 4

    def test_get_wait_time(self):
        """Test calculating wait time for tokens to become available."""
        bucket = TokenBucket(capacity=10, fill_rate=1)
        
        # Consume all tokens
        for _ in range(10):
            bucket.get_token()
            
        # Wait time for 1 token should be 1 second
        assert bucket.get_wait_time() == 1.0
        
        # Wait time for 5 tokens should be 5 seconds
        assert bucket.get_wait_time(5) == 5.0


class TestFixedWindowCounter:
    """Test the FixedWindowCounter rate limiting algorithm."""

    def test_initialization(self):
        """Test fixed window counter initialization."""
        counter = FixedWindowCounter(window_size=60, max_requests=10)
        assert counter.window_size == 60
        assert counter.max_requests == 10
        assert counter.request_count == 0

    def test_allow_request(self):
        """Test allowing requests within the window limit."""
        counter = FixedWindowCounter(window_size=60, max_requests=3)
        
        # Should allow max_requests
        assert counter.allow_request() is True
        assert counter.allow_request() is True
        assert counter.allow_request() is True
        
        # Should deny requests over max_requests
        assert counter.allow_request() is False
        
        # Window should reset after window_size
        with patch('time.time', return_value=time.time() + 61):
            assert counter.allow_request() is True

    def test_get_wait_time(self):
        """Test calculating wait time until next window."""
        counter = FixedWindowCounter(window_size=60, max_requests=3)
        
        # Fill up the window
        counter.allow_request()
        counter.allow_request()
        counter.allow_request()
        
        # Check wait time
        with patch('time.time', return_value=counter.current_window_start + 10):
            # 50 seconds left in the window
            wait_time = counter.get_wait_time()
            assert 49 < wait_time <= 50


class TestSlidingWindowCounter:
    """Test the SlidingWindowCounter rate limiting algorithm."""

    def test_initialization(self):
        """Test sliding window counter initialization."""
        counter = SlidingWindowCounter(window_size=60, max_requests=10)
        assert counter.window_size == 60
        assert counter.max_requests == 10
        assert len(counter.request_timestamps) == 0

    def test_allow_request(self):
        """Test allowing requests within the sliding window."""
        counter = SlidingWindowCounter(window_size=60, max_requests=3)
        
        # Should allow max_requests
        assert counter.allow_request() is True
        assert counter.allow_request() is True
        assert counter.allow_request() is True
        
        # Should deny requests over max_requests
        assert counter.allow_request() is False
        
        # Oldest request should fall out of the window after window_size
        with patch('time.time', return_value=time.time() + 61):
            assert counter.allow_request() is True

    def test_get_wait_time(self):
        """Test calculating wait time until a request can be made."""
        counter = SlidingWindowCounter(window_size=60, max_requests=3)
        
        # Set known timestamps
        start_time = time.time()
        counter.request_timestamps.append(start_time)
        counter.request_timestamps.append(start_time + 10)
        counter.request_timestamps.append(start_time + 20)
        
        # Check wait time at 30 seconds after start
        with patch('time.time', return_value=start_time + 30):
            # Wait time should be 30 seconds (until first request falls out)
            wait_time = counter.get_wait_time()
            assert 29 < wait_time <= 30


class TestConcurrencyLimiter:
    """Test the ConcurrencyLimiter for controlling concurrent requests."""

    def test_initialization(self):
        """Test concurrency limiter initialization."""
        limiter = ConcurrencyLimiter(max_concurrent=5)
        assert limiter.max_concurrent == 5
        assert limiter.active_requests == 0

    def test_acquire_release(self):
        """Test acquiring and releasing concurrency slots."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        
        # Should be able to acquire up to max_concurrent
        assert limiter.acquire() is True
        assert limiter.active_requests == 1
        
        assert limiter.acquire() is True
        assert limiter.active_requests == 2
        
        # Should deny requests over max_concurrent
        assert limiter.acquire() is False
        assert limiter.active_requests == 2
        
        # Should be able to acquire after release
        limiter.release()
        assert limiter.active_requests == 1
        
        assert limiter.acquire() is True
        assert limiter.active_requests == 2

    def test_set_max_concurrent(self):
        """Test dynamically changing max concurrent requests."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        
        # Fill the limiter
        limiter.acquire()
        limiter.acquire()
        
        # Increase capacity
        limiter.set_max_concurrent(3)
        
        # Should be able to acquire one more
        assert limiter.acquire() is True
        assert limiter.active_requests == 3


class TestDomainPolicy:
    """Test the DomainPolicy for rate limit policy management."""

    def test_initialization(self):
        """Test domain policy initialization with different policy types."""
        # Test token bucket policy
        policy = DomainPolicy(rpm=60, max_concurrent=5, policy_type='token_bucket')
        assert policy.rpm == 60
        assert policy.max_concurrent == 5
        assert policy.policy_type == 'token_bucket'
        assert isinstance(policy.rate_limiter, TokenBucket)
        
        # Test fixed window policy
        policy = DomainPolicy(rpm=60, max_concurrent=5, policy_type='fixed_window')
        assert isinstance(policy.rate_limiter, FixedWindowCounter)
        
        # Test sliding window policy
        policy = DomainPolicy(rpm=60, max_concurrent=5, policy_type='sliding_window')
        assert isinstance(policy.rate_limiter, SlidingWindowCounter)

    def test_adjust_rate(self):
        """Test adjusting the rate limiter rate."""
        policy = DomainPolicy(rpm=60, max_concurrent=5)
        
        # Test increasing
        policy.adjust_rate(1.5)
        assert policy.rpm == 90  # 60 * 1.5
        
        # Test decreasing
        policy.adjust_rate(0.5)
        assert policy.rpm == 45  # 90 * 0.5
        
        # Test minimum
        policy.adjust_rate(0.01)
        assert policy.rpm == 1  # Minimum is 1
        
        # Test maximum
        policy.adjust_rate(2000)
        assert policy.rpm == 1000  # Maximum is 1000

    def test_allow_request(self):
        """Test allowing requests based on combined rate and concurrency limits."""
        policy = DomainPolicy(rpm=60, max_concurrent=2)
        
        # Mock the rate limiter to always allow requests
        policy.rate_limiter.get_token = MagicMock(return_value=True)
        
        # Should allow requests up to concurrency limit
        assert policy.allow_request() is True
        assert policy.allow_request() is True
        assert policy.allow_request() is False  # Concurrency limit reached
        
        # Release concurrency
        policy.release_concurrency()
        
        # Should allow one more request
        assert policy.allow_request() is True


class TestRateLimiter:
    """Test the RateLimiter service."""

    def test_initialization(self, rate_limiter):
        """Test rate limiter initialization from config."""
        assert rate_limiter._initialized is True
        assert rate_limiter.name == "rate_limiter"
        
        # Check if domain policies were created from config
        with rate_limiter._lock:
            assert "example.com" in rate_limiter._domain_policies
            assert "api.example.com" in rate_limiter._domain_policies
            assert "highvolume.com" in rate_limiter._domain_policies
            
            # Check specific policy settings
            example_policy = rate_limiter._domain_policies["example.com"]
            assert example_policy.rpm == 30
            assert example_policy.max_concurrent == 3
            assert example_policy.policy_type == 'token_bucket'

    def test_service_registry_integration(self):
        """Test RateLimiter integration with ServiceRegistry."""
        registry = ServiceRegistry()
        registry.register_service_class(RateLimiter)
        
        # Get the service through the registry
        rate_limiter = registry.get_service("rate_limiter")
        assert rate_limiter.is_initialized
        assert rate_limiter.name == "rate_limiter"
        
        # Clean up
        registry.shutdown_all()

    def test_get_domain_from_url(self, rate_limiter):
        """Test domain extraction from various URL formats."""
        # Normal URLs
        assert rate_limiter._get_domain_from_url("https://example.com/path") == "example.com"
        assert rate_limiter._get_domain_from_url("http://sub.example.com/path?query=test") == "sub.example.com"
        
        # URLs with www prefix
        assert rate_limiter._get_domain_from_url("https://www.example.com") == "example.com"
        
        # URLs without scheme
        assert rate_limiter._get_domain_from_url("example.com/path") == "example.com"
        
        # IP addresses
        assert rate_limiter._get_domain_from_url("http://127.0.0.1:8080/path") == "127.0.0.1:8080"
        
        # Domain caching
        url = "https://cachethis.example.com"
        rate_limiter._get_domain_from_url(url)  # Cache the domain
        assert url in rate_limiter._domain_cache
        assert rate_limiter._domain_cache[url] == "cachethis.example.com"

    def test_get_or_create_policy(self, rate_limiter):
        """Test getting existing or creating new policies for domains."""
        # Get existing policy
        policy = rate_limiter._get_or_create_policy("example.com")
        assert policy.rpm == 30  # From config
        
        # Get parent domain's policy
        policy = rate_limiter._get_or_create_policy("new.example.com")
        assert policy.rpm == 30  # Should inherit from example.com
        
        # Create brand new policy
        policy = rate_limiter._get_or_create_policy("newdomain.com")
        assert policy.rpm == 60  # Should use default rpm

    def test_wait_if_needed(self, rate_limiter):
        """Test waiting for rate limits."""
        # Patch the allow_request method to simulate rate limiting
        with patch.object(DomainPolicy, 'allow_request') as mock_allow:
            # First call returns False, second call returns True
            mock_allow.side_effect = [False, True]
            
            # Also patch get_wait_time to return a small value
            with patch.object(DomainPolicy, 'get_wait_time', return_value=0.1):
                waited, time_waited = rate_limiter.wait_if_needed("https://example.com")
                
                # Should have waited
                assert waited is True
                assert time_waited > 0

    @responses.activate
    def test_detect_rate_limit_from_response(self, rate_limiter):
        """Test detecting rate limiting from HTTP responses."""
        # Set up a response with 429 status code
        responses.add(responses.GET, 'https://api.example.com/too-many', status=429)
        response = requests.get('https://api.example.com/too-many')
        assert rate_limiter.detect_rate_limit_from_response(response) is True
        
        # Set up a response with rate limit headers
        responses.add(
            responses.GET, 
            'https://api.example.com/limit-headers',
            status=200,
            headers={
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Limit': '100'
            }
        )
        response = requests.get('https://api.example.com/limit-headers')
        assert rate_limiter.detect_rate_limit_from_response(response) is True
        
        # Set up a response with rate limit text
        responses.add(
            responses.GET, 
            'https://api.example.com/limit-text',
            status=200,
            body="You have exceeded the rate limit. Please slow down."
        )
        response = requests.get('https://api.example.com/limit-text')
        assert rate_limiter.detect_rate_limit_from_response(response) is True
        
        # Set up a normal response
        responses.add(
            responses.GET, 
            'https://api.example.com/normal',
            status=200,
            body="Normal response"
        )
        response = requests.get('https://api.example.com/normal')
        assert rate_limiter.detect_rate_limit_from_response(response) is False

    def test_parse_retry_after(self, rate_limiter):
        """Test parsing Retry-After header values."""
        # Test seconds value
        assert rate_limiter.parse_retry_after("30") == 30.0
        
        # Test HTTP date format
        future_time = time.time() + 60
        future_date = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(future_time))
        retry_time = rate_limiter.parse_retry_after(future_date)
        assert 55 <= retry_time <= 65  # Allow some margin for test time differences
        
        # Test invalid format
        assert rate_limiter.parse_retry_after("invalid") == 60.0  # Default value

    def test_extract_rate_limit_headers(self, rate_limiter):
        """Test extracting rate limit information from response headers."""
        # Create a mock response with rate limit headers
        mock_response = MagicMock()
        mock_response.headers = {
            'X-RateLimit-Limit': '100',
            'X-RateLimit-Remaining': '75',
            'X-RateLimit-Reset': '1600000000',
            'Retry-After': '30'
        }
        
        rate_info = rate_limiter.extract_rate_limit_headers(mock_response)
        
        assert rate_info['limit'] == 100
        assert rate_info['remaining'] == 75
        assert 'reset' in rate_info
        assert rate_info['retry-after'] == 30.0

    def test_register_response(self, rate_limiter):
        """Test processing a response for rate limiting."""
        # Create a domain policy
        domain = "response-test.com"
        policy = rate_limiter._get_or_create_policy(domain)
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Register a successful response
        rate_limited = rate_limiter.register_response(f"https://{domain}/path", mock_response)
        assert rate_limited is False
        
        # Check that the policy recorded success
        assert list(policy.success_history)[-1] is True
        
        # Create a rate limited response
        mock_response.status_code = 429
        
        # Register a rate limited response
        rate_limited = rate_limiter.register_response(f"https://{domain}/path", mock_response)
        assert rate_limited is True
        
        # Check that the policy recorded failure
        assert list(policy.success_history)[-1] is False
        assert policy.rate_limited_count == 1

    def test_adaptive_rate_limiting(self, rate_limiter):
        """Test adaptive rate limiting based on response success/failure."""
        # Create a domain policy with known settings
        domain = "adaptive-test.com"
        policy = rate_limiter._get_or_create_policy(domain)
        policy.rpm = 100
        
        # Register several successful responses
        for _ in range(10):
            mock_response = MagicMock()
            mock_response.status_code = 200
            rate_limiter.register_response(f"https://{domain}/path", mock_response)
            
        # Force time to advance for adaptive adjustment
        policy.last_adjustment_time = 0
        
        # Register one more to trigger rate increase
        mock_response = MagicMock()
        mock_response.status_code = 200
        rate_limiter.register_response(f"https://{domain}/path", mock_response)
        
        # Rate should have increased (up to 10%)
        assert policy.rpm > 100
        
        # Now register a rate limited response
        mock_response.status_code = 429
        rate_limiter.register_response(f"https://{domain}/path", mock_response)
        
        # Rate should have decreased significantly
        assert policy.rpm < 100

    def test_domain_grouping(self, rate_limiter):
        """Test domain grouping for shared rate limits."""
        # Group domains should have policy from config
        policy1 = rate_limiter._get_or_create_policy("sub1.example.com")
        policy2 = rate_limiter._get_or_create_policy("sub2.example.com")
        
        assert policy1.rpm == 20
        assert policy2.rpm == 20
        
        # Apply a new group rate limit
        rate_limiter.apply_group_rate_limit("example_group", 15)
        
        # Both domains should have the updated rate
        assert policy1.rpm == 15
        assert policy2.rpm == 15
        
        # Adding a new domain to the group
        rate_limiter.add_domain_group("example_group", ["sub3.example.com"])
        policy3 = rate_limiter._get_or_create_policy("sub3.example.com")
        
        # New domain should have the group rate
        assert policy3.rpm == 15

    def test_detect_related_domains(self, rate_limiter):
        """Test finding related domains."""
        # Create several domain policies with parent relation
        domains = [
            "test1.example.org",
            "test2.example.org",
            "sub.test2.example.org",
            "unrelated.com"
        ]
        
        for domain in domains:
            rate_limiter._get_or_create_policy(domain)
            
        # Find domains related to test1.example.org
        related = rate_limiter.detect_related_domains("https://test1.example.org")
        
        # Should find test2.example.org and sub.test2.example.org as related
        assert "test2.example.org" in related
        assert "sub.test2.example.org" in related
        assert "unrelated.com" not in related

    def test_calculate_backoff(self, rate_limiter):
        """Test exponential backoff calculation."""
        # Test with default factor
        backoff1 = rate_limiter.calculate_backoff(1)
        backoff2 = rate_limiter.calculate_backoff(2)
        backoff3 = rate_limiter.calculate_backoff(3)
        
        # Ensure exponential growth
        assert backoff1 < backoff2 < backoff3
        
        # Test with custom factor
        custom_backoff = rate_limiter.calculate_backoff(2, factor=3.0)
        assert custom_backoff > backoff2  # Should be higher with larger factor

    def test_get_metrics(self, rate_limiter):
        """Test getting metrics about rate limiting."""
        metrics = rate_limiter.get_metrics()
        
        # Check structure
        assert 'domains' in metrics
        assert 'groups' in metrics
        assert 'total_domains' in metrics
        assert 'total_groups' in metrics
        
        # Check domain data
        assert 'example.com' in metrics['domains']
        domain_metrics = metrics['domains']['example.com']
        assert 'rpm' in domain_metrics
        assert 'max_concurrent' in domain_metrics
        
        # Check group data
        assert 'example_group' in metrics['groups']
        group_metrics = metrics['groups']['example_group']
        assert 'domain_count' in group_metrics
        assert 'policy' in group_metrics


class TestThreadSafety:
    """Test thread safety of the rate limiter."""

    def test_concurrent_access(self, rate_limiter):
        """Test concurrent access to the rate limiter from multiple threads."""
        # Use a domain that's not pre-configured
        domain = "https://thread-test.com"
        request_count = 0
        error_count = 0
        thread_count = 10
        requests_per_thread = 5
        
        # Track successes in thread-safe way
        lock = threading.Lock()
        
        def worker():
            nonlocal request_count, error_count
            for _ in range(requests_per_thread):
                try:
                    rate_limiter.wait_if_needed(domain)
                    
                    # Simulate request processing
                    time.sleep(0.01)
                    
                    # Create mock response
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    
                    # Register response
                    rate_limiter.register_response(domain, mock_response)
                    
                    # Count success
                    with lock:
                        request_count += 1
                except Exception as e:
                    with lock:
                        error_count += 1
        
        # Start multiple threads
        threads = []
        for _ in range(thread_count):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
            
        # Wait for all threads to finish
        for thread in threads:
            thread.join()
            
        # All requests should have succeeded
        assert request_count == thread_count * requests_per_thread
        assert error_count == 0
        
        # Get the policy for the domain
        policy = rate_limiter._get_or_create_policy(rate_limiter._get_domain_from_url(domain))
        
        # All requests should be tracked in success history
        assert len(policy.success_history) == thread_count * requests_per_thread


if __name__ == "__main__":
    pytest.main(["-v", __file__])