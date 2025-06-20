import logging
import time
import threading
import re
import random
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from urllib.parse import urlparse
import json
from email.utils import parsedate_to_datetime

from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class TokenBucket:
    """
    Implementation of the Token Bucket algorithm for rate limiting.
    
    The token bucket algorithm works by having a bucket that is filled with tokens at a 
    steady rate. Each request removes one token. If the bucket is empty, requests are
    blocked until more tokens are added.
    """
    
    def __init__(self, capacity: float, fill_rate: float):
        """
        Initialize a token bucket.
        
        Args:
            capacity: The maximum number of tokens the bucket can hold
            fill_rate: The rate at which tokens are added to the bucket per second
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.RLock()
        
    def get_token(self, count: float = 1.0) -> bool:
        """
        Attempt to get 'count' tokens from the bucket.
        
        Args:
            count: The number of tokens to consume (default 1.0)
            
        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        with self.lock:
            self._refill()
            if self.tokens >= count:
                self.tokens -= count
                return True
            return False
    
    def get_wait_time(self, count: float = 1.0) -> float:
        """
        Calculate the time to wait for 'count' tokens to become available.
        
        Args:
            count: The number of tokens needed (default 1.0)
            
        Returns:
            The time in seconds to wait for tokens to become available
        """
        with self.lock:
            self._refill()
            if self.tokens >= count:
                return 0.0
            additional_tokens_needed = count - self.tokens
            return additional_tokens_needed / self.fill_rate
    
    def _refill(self) -> None:
        """Refill the bucket based on elapsed time since last update."""
        now = time.time()
        elapsed = now - self.last_update
        
        # Calculate new tokens and add to bucket (up to capacity)
        new_tokens = elapsed * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now


class FixedWindowCounter:
    """
    Implementation of a Fixed Window Counter for rate limiting.
    
    The fixed window counter tracks requests within a time window (e.g., 1 minute).
    Once the window expires, the counter resets.
    """
    
    def __init__(self, window_size: float, max_requests: int):
        """
        Initialize a fixed window counter.
        
        Args:
            window_size: The size of the window in seconds
            max_requests: The maximum number of requests allowed in the window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.current_window_start = time.time()
        self.request_count = 0
        self.lock = threading.RLock()
        
    def allow_request(self) -> bool:
        """
        Check if a request is allowed within the current window.
        
        Returns:
            True if the request is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Check if we need to reset the window
            if now - self.current_window_start > self.window_size:
                self.current_window_start = now
                self.request_count = 0
            
            # Check if we're under the limit
            if self.request_count < self.max_requests:
                self.request_count += 1
                return True
            
            return False
    
    def get_wait_time(self) -> float:
        """
        Calculate the time to wait until the next window starts.
        
        Returns:
            The time in seconds to wait until the window resets
        """
        with self.lock:
            now = time.time()
            window_end = self.current_window_start + self.window_size
            
            if now < window_end and self.request_count >= self.max_requests:
                return window_end - now
            
            return 0.0
    
    def reset(self) -> None:
        """Reset the window counter."""
        with self.lock:
            self.current_window_start = time.time()
            self.request_count = 0


class SlidingWindowCounter:
    """
    Implementation of a Sliding Window Counter for rate limiting.
    
    The sliding window counter keeps track of requests over time and provides
    a smoother rate limiting experience compared to a fixed window.
    """
    
    def __init__(self, window_size: float, max_requests: int):
        """
        Initialize a sliding window counter.
        
        Args:
            window_size: The size of the window in seconds
            max_requests: The maximum number of requests allowed in the window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.request_timestamps = deque()
        self.lock = threading.RLock()
        
    def allow_request(self) -> bool:
        """
        Check if a request is allowed within the current sliding window.
        
        Returns:
            True if the request is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Remove timestamps outside the window
            while self.request_timestamps and self.request_timestamps[0] < now - self.window_size:
                self.request_timestamps.popleft()
            
            # Check if we're under the limit
            if len(self.request_timestamps) < self.max_requests:
                self.request_timestamps.append(now)
                return True
            
            return False
    
    def get_wait_time(self) -> float:
        """
        Calculate the time to wait until the oldest request falls out of the window.
        
        Returns:
            The time in seconds to wait until a request can be made
        """
        with self.lock:
            now = time.time()
            
            # Remove timestamps outside the window
            while self.request_timestamps and self.request_timestamps[0] < now - self.window_size:
                self.request_timestamps.popleft()
            
            if len(self.request_timestamps) < self.max_requests:
                return 0.0
            
            # Calculate when the oldest request will fall out of the window
            oldest = self.request_timestamps[0]
            return (oldest + self.window_size) - now
    
    def reset(self) -> None:
        """Reset the window counter."""
        with self.lock:
            self.request_timestamps.clear()


class ConcurrencyLimiter:
    """
    Limits the number of concurrent requests to a domain.
    """
    
    def __init__(self, max_concurrent: int):
        """
        Initialize a concurrency limiter.
        
        Args:
            max_concurrent: The maximum number of concurrent requests allowed
        """
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.lock = threading.RLock()
        
    def acquire(self) -> bool:
        """
        Acquire a slot for a concurrent request.
        
        Returns:
            True if a slot was acquired, False otherwise
        """
        with self.lock:
            if self.active_requests < self.max_concurrent:
                self.active_requests += 1
                return True
            return False
    
    def release(self) -> None:
        """Release a slot after a request completes."""
        with self.lock:
            if self.active_requests > 0:
                self.active_requests -= 1
    
    def set_max_concurrent(self, max_concurrent: int) -> None:
        """Update the maximum number of concurrent requests."""
        with self.lock:
            self.max_concurrent = max_concurrent


class DomainPolicy:
    """
    Represents the rate limiting policy for a domain or group of domains.
    """
    
    def __init__(self, rpm: int = 60, max_concurrent: int = 5, 
                backoff_factor: float = 2.0, policy_type: str = 'token_bucket'):
        """
        Initialize a domain policy.
        
        Args:
            rpm: Requests per minute allowed
            max_concurrent: Maximum concurrent requests
            backoff_factor: Factor to adjust backoff on failure
            policy_type: Type of rate limiting policy ('token_bucket', 'fixed_window', 'sliding_window')
        """
        self.rpm = rpm
        self.max_concurrent = max_concurrent
        self.backoff_factor = backoff_factor
        self.policy_type = policy_type
        
        # Initialize appropriate rate limiter based on policy type
        if policy_type == 'token_bucket':
            # Convert RPM to tokens per second for the token bucket
            self.rate_limiter = TokenBucket(capacity=rpm/4, fill_rate=rpm/60.0)
        elif policy_type == 'fixed_window':
            self.rate_limiter = FixedWindowCounter(window_size=60.0, max_requests=rpm)
        elif policy_type == 'sliding_window':
            self.rate_limiter = SlidingWindowCounter(window_size=60.0, max_requests=rpm)
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
            
        # Initialize concurrency limiter
        self.concurrency_limiter = ConcurrencyLimiter(max_concurrent)
        
        # Tracking for adaptive rate limiting
        self.success_history = deque(maxlen=100)  # Store recent success/failure
        self.rate_limited_count = 0
        self.last_rate_limited = 0
        self.last_adjustment_time = time.time()
        self.is_adaptive = True  # Whether to use adaptive rate limiting
        
    def allow_request(self) -> bool:
        """
        Check if a request is allowed by both rate and concurrency limiters.
        
        Returns:
            True if the request is allowed, False otherwise
        """
        if self.policy_type == 'token_bucket':
            rate_allowed = self.rate_limiter.get_token()
        else:
            rate_allowed = self.rate_limiter.allow_request()
            
        concurrency_allowed = self.concurrency_limiter.acquire()
        
        # Only count the request against concurrency if both allow
        if rate_allowed and concurrency_allowed:
            return True
        elif concurrency_allowed:
            # Release the concurrency slot if rate limiting blocked the request
            self.concurrency_limiter.release()
            
        return False
    
    def get_wait_time(self) -> float:
        """
        Calculate the time to wait for the next request.
        
        Returns:
            The time in seconds to wait
        """
        if self.policy_type == 'token_bucket':
            return self.rate_limiter.get_wait_time()
        else:
            return self.rate_limiter.get_wait_time()
    
    def release_concurrency(self) -> None:
        """Release a concurrency slot after a request completes."""
        self.concurrency_limiter.release()
    
    def record_success(self) -> None:
        """Record a successful request for adaptive rate limiting."""
        self.success_history.append(True)
        
        # Potentially increase rate if we've been successful for a while
        if self.is_adaptive and time.time() - self.last_adjustment_time > 60:
            success_rate = sum(self.success_history) / len(self.success_history) if self.success_history else 0
            if success_rate > 0.95 and len(self.success_history) >= 10:
                self.adjust_rate(1.1)  # Increase by 10%
                self.last_adjustment_time = time.time()
    
    def record_failure(self, is_rate_limited: bool = False) -> None:
        """
        Record a failed request for adaptive rate limiting.
        
        Args:
            is_rate_limited: Whether the failure was due to rate limiting
        """
        self.success_history.append(False)
        
        if is_rate_limited:
            self.rate_limited_count += 1
            self.last_rate_limited = time.time()
            
            # Immediately reduce rate on rate limiting
            if self.is_adaptive:
                self.adjust_rate(0.5)  # Reduce by 50%
                self.last_adjustment_time = time.time()
        elif self.is_adaptive and time.time() - self.last_adjustment_time > 30:
            # For other failures, more gradual reduction if we've had several
            success_rate = sum(self.success_history) / len(self.success_history) if self.success_history else 0
            if success_rate < 0.7 and len(self.success_history) >= 5:
                self.adjust_rate(0.8)  # Reduce by 20%
                self.last_adjustment_time = time.time()
    
    def adjust_rate(self, factor: float) -> None:
        """
        Adjust the rate by a factor.
        
        Args:
            factor: Factor to multiply the current RPM by (>1 to increase, <1 to decrease)
        """
        old_rpm = self.rpm
        self.rpm = max(1, min(1000, int(self.rpm * factor)))  # Clamp between 1 and 1000 RPM
        
        # Update the underlying rate limiter
        if self.policy_type == 'token_bucket':
            self.rate_limiter = TokenBucket(capacity=self.rpm/4, fill_rate=self.rpm/60.0)
        elif self.policy_type == 'fixed_window':
            self.rate_limiter = FixedWindowCounter(window_size=60.0, max_requests=self.rpm)
        elif self.policy_type == 'sliding_window':
            self.rate_limiter = SlidingWindowCounter(window_size=60.0, max_requests=self.rpm)
            
        logger.info(f"Adjusted rate from {old_rpm} to {self.rpm} RPM (factor: {factor:.2f})")
    
    def set_rate(self, rpm: int) -> None:
        """
        Set the requests per minute directly.
        
        Args:
            rpm: New requests per minute value
        """
        self.adjust_rate(rpm / self.rpm if self.rpm > 0 else 1.0)
    
    def set_max_concurrent(self, max_concurrent: int) -> None:
        """
        Set the maximum concurrent requests.
        
        Args:
            max_concurrent: New maximum concurrent requests value
        """
        self.max_concurrent = max_concurrent
        self.concurrency_limiter.set_max_concurrent(max_concurrent)


class RateLimiter(BaseService):
    """
    Service for managing and enforcing rate limits across domains.
    """
    
    def __init__(self):
        """Initialize the rate limiter."""
        self._initialized = False
        self._config = None
        self._domain_policies = {}  # Domain -> DomainPolicy
        self._domain_groups = {}    # Group name -> set of domains
        self._domain_to_groups = defaultdict(set)  # Domain -> set of groups it belongs to
        self._lock = threading.RLock()
        
        # Default configuration values (will be overridden in initialize())
        self._default_rpm = 60
        self._default_concurrent = 5
        self._default_backoff_factor = 2.0
        self._default_policy_type = 'token_bucket'
        
        # Rate limit detection patterns
        self._rate_limit_patterns = []
        
        # Cache for domain extraction
        self._domain_cache = {}
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the rate limiter with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        if self._initialized:
            return
            
        self._config = config or {}
        
        # Set default values from config
        self._default_rpm = self._config.get('default_rpm', 60)
        self._default_concurrent = self._config.get('default_concurrent_requests', 5)
        self._default_backoff_factor = self._config.get('backoff_factor', 2.0)
        self._default_policy_type = self._config.get('default_policy_type', 'token_bucket')
        
        # Compile rate limit detection patterns
        self._rate_limit_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self._config.get('rate_limit_patterns', [
                r'rate limit|too many requests|429|exceed|throttl',
                r'slow down|limit exceeded|too fast',
                r'access denied.*automated'
            ])
        ]
        
        # Load domain-specific policies from config
        domain_policies = self._config.get('domain_policies', {})
        for domain, policy_config in domain_policies.items():
            self._create_domain_policy(domain, policy_config)
            
        # Load domain groups from config
        domain_groups = self._config.get('domain_groups', {})
        for group_name, group_config in domain_groups.items():
            domains = group_config.get('domains', [])
            if domains:
                self.add_domain_group(group_name, domains, group_config.get('policy', {}))
        
        self._initialized = True
        logger.info("Rate limiter initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return
            
        with self._lock:
            self._domain_policies.clear()
            self._domain_groups.clear()
            self._domain_to_groups.clear()
            self._domain_cache.clear()
        
        self._initialized = False
        logger.info("Rate limiter shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "rate_limiter"
        
    def _get_domain_from_url(self, url: str) -> str:
        """
        Extract domain from URL for rate limiting.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        # Check cache first
        if url in self._domain_cache:
            return self._domain_cache[url]
            
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Handle case where URL has no scheme
            if not domain and not parsed_url.scheme:
                domain = parsed_url.path.split('/')[0].lower()
                
            # Remove 'www.' prefix for consistent domain matching
            if domain.startswith('www.'):
                domain = domain[4:]
                
            # Cache the result
            self._domain_cache[url] = domain
            
            return domain or url.lower()  # Fallback to full URL if domain extraction fails
        except Exception as e:
            logger.warning(f"Error extracting domain from {url}: {str(e)}")
            return url.lower()
    
    def _get_or_create_policy(self, domain: str) -> DomainPolicy:
        """
        Get an existing policy for a domain or create a new one.
        
        Args:
            domain: Domain to get policy for
            
        Returns:
            DomainPolicy object
        """
        with self._lock:
            # Check if we have a direct policy for this domain
            if domain in self._domain_policies:
                return self._domain_policies[domain]
                
            # Check for parent domain match 
            parts = domain.split('.')
            for i in range(1, len(parts)):
                parent = '.'.join(parts[i:])
                if parent in self._domain_policies:
                    # Create a policy for this domain based on the parent
                    parent_policy = self._domain_policies[parent]
                    policy = DomainPolicy(
                        rpm=parent_policy.rpm,
                        max_concurrent=parent_policy.max_concurrent,
                        backoff_factor=parent_policy.backoff_factor,
                        policy_type=parent_policy.policy_type
                    )
                    self._domain_policies[domain] = policy
                    return policy
            
            # No existing policy, create with defaults
            policy = DomainPolicy(
                rpm=self._default_rpm,
                max_concurrent=self._default_concurrent,
                backoff_factor=self._default_backoff_factor,
                policy_type=self._default_policy_type
            )
            self._domain_policies[domain] = policy
            
            # Check if this domain belongs to any groups and adjust accordingly
            for group in self._domain_to_groups.get(domain, []):
                if group in self._domain_groups:
                    group_policy = self._domain_groups[group].get('policy', {})
                    if 'rpm' in group_policy:
                        policy.set_rate(group_policy['rpm'])
                    if 'max_concurrent' in group_policy:
                        policy.set_max_concurrent(group_policy['max_concurrent'])
            
            return policy
            
    def _create_domain_policy(self, domain: str, policy_config: Dict[str, Any]) -> DomainPolicy:
        """
        Create a domain policy from configuration.
        
        Args:
            domain: Domain to create policy for
            policy_config: Policy configuration dictionary
            
        Returns:
            DomainPolicy object
        """
        with self._lock:
            rpm = policy_config.get('rpm', self._default_rpm)
            max_concurrent = policy_config.get('max_concurrent', self._default_concurrent)
            backoff_factor = policy_config.get('backoff_factor', self._default_backoff_factor)
            policy_type = policy_config.get('policy_type', self._default_policy_type)
            
            policy = DomainPolicy(
                rpm=rpm,
                max_concurrent=max_concurrent,
                backoff_factor=backoff_factor,
                policy_type=policy_type
            )
            policy.is_adaptive = policy_config.get('is_adaptive', True)
            
            self._domain_policies[domain] = policy
            return policy
    
    def wait_if_needed(self, url: str) -> Tuple[bool, float]:
        """
        Wait if necessary to comply with rate limits.
        
        Args:
            url: URL to check rate limit for
            
        Returns:
            Tuple of (waited, time_waited_in_seconds)
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        
        start_time = time.time()
        
        # Check if we can make a request, otherwise wait
        while not policy.allow_request():
            # Calculate wait time with jitter to avoid thundering herd
            wait_time = policy.get_wait_time()
            jitter = random.uniform(0, 0.1 * wait_time) if wait_time > 0 else 0
            total_wait = wait_time + jitter
            
            if total_wait > 0:
                logger.debug(f"Rate limiting: Waiting {total_wait:.2f}s for {domain}")
                time.sleep(total_wait)
            else:
                # If wait time is 0 but we still can't make a request, it's likely due to concurrency
                # limits. Add a small delay to prevent busy waiting.
                time.sleep(0.01)
        
        time_waited = time.time() - start_time
        return (time_waited > 0, time_waited)
    
    def register_request(self, url: str) -> None:
        """
        Register that a request is being made.
        
        This should be called after wait_if_needed() and before making the actual request.
        """
        # The wait_if_needed() method already registered the request via policy.allow_request()
        # which acquired the necessary tokens/slots. This method is a no-op to maintain a
        # consistent interface.
        pass
    
    def register_response(self, url: str, response: Any) -> bool:
        """
        Process a response to detect rate limiting and adjust rates if needed.
        
        Args:
            url: URL the response is for
            response: Response object (requests.Response or similar)
            
        Returns:
            True if rate limiting was detected, False otherwise
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        
        # Always release the concurrency slot
        policy.release_concurrency()
        
        # Check if rate limiting was detected
        is_rate_limited = self.detect_rate_limit_from_response(response)
        
        if is_rate_limited:
            policy.record_failure(is_rate_limited=True)
            return True
        else:
            policy.record_success()
            return False
    
    def detect_rate_limit_from_response(self, response: Any) -> bool:
        """
        Detect if a response indicates rate limiting.
        
        Args:
            response: Response object
            
        Returns:
            True if rate limiting was detected, False otherwise
        """
        # Check status code
        status_code = getattr(response, 'status_code', None)
        if status_code == 429:
            return True
            
        # Check headers for rate limit information
        headers = getattr(response, 'headers', {})
        for header in headers:
            header_lower = header.lower()
            if 'rate' in header_lower and 'limit' in header_lower:
                # Check if we're close to the limit
                if 'remaining' in header_lower:
                    try:
                        remaining = int(headers[header])
                        if remaining <= 1:  # Almost out of requests
                            return True
                    except (ValueError, TypeError):
                        pass
                        
        # Check for rate limit patterns in the response text
        try:
            content = getattr(response, 'text', '')
            if content:
                for pattern in self._rate_limit_patterns:
                    if pattern.search(content):
                        return True
        except:
            pass
            
        return False
    
    def is_rate_limited(self, url: str) -> bool:
        """
        Check if a domain is currently rate limited.
        
        Args:
            url: URL to check
            
        Returns:
            True if the domain is rate limited, False otherwise
        """
        domain = self._get_domain_from_url(url)
        
        with self._lock:
            if domain not in self._domain_policies:
                return False
                
            policy = self._domain_policies[domain]
            
            # Check if we've seen rate limiting recently
            if policy.last_rate_limited > 0:
                time_since_rate_limit = time.time() - policy.last_rate_limited
                if time_since_rate_limit < 60:  # Within the last minute
                    return True
            
            return False
    
    def adjust_rate(self, url: str, adjustment_factor: float) -> None:
        """
        Adjust the rate for a domain.
        
        Args:
            url: URL to adjust rate for
            adjustment_factor: Factor to multiply the current RPM by
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        policy.adjust_rate(adjustment_factor)
    
    def set_rate(self, url: str, rpm: int) -> None:
        """
        Set the requests per minute for a domain.
        
        Args:
            url: URL to set rate for
            rpm: Requests per minute
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        policy.set_rate(rpm)
    
    def set_max_concurrent(self, url: str, max_concurrent: int) -> None:
        """
        Set the maximum concurrent requests for a domain.
        
        Args:
            url: URL to set concurrent limit for
            max_concurrent: Maximum concurrent requests
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        policy.set_max_concurrent(max_concurrent)
    
    def reset_rate(self, url: str) -> None:
        """
        Reset the rate for a domain to default.
        
        Args:
            url: URL to reset rate for
        """
        domain = self._get_domain_from_url(url)
        
        with self._lock:
            if domain in self._domain_policies:
                del self._domain_policies[domain]
    
    def get_optimal_rate(self, url: str, success_history: List[bool] = None) -> int:
        """
        Estimate the optimal rate for a domain based on history.
        
        Args:
            url: URL to estimate rate for
            success_history: Optional list of boolean success/failure values
            
        Returns:
            Estimated optimal requests per minute
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        
        # Use provided history or policy's internal history
        history = success_history or list(policy.success_history)
        
        if not history:
            return policy.rpm  # No history, use current rate
            
        # Calculate success rate
        success_rate = sum(1 for x in history if x) / len(history) if history else 0
        
        # Estimate optimal rate based on success rate
        if success_rate > 0.95:
            # Very high success rate, can increase
            return min(1000, int(policy.rpm * 1.2))
        elif success_rate < 0.7:
            # Low success rate, should decrease
            return max(1, int(policy.rpm * 0.7))
        else:
            # Otherwise, keep current rate
            return policy.rpm
    
    def extract_rate_limit_headers(self, response: Any) -> Dict[str, Any]:
        """
        Extract rate limit information from response headers.
        
        Args:
            response: Response object
            
        Returns:
            Dictionary of rate limit information
        """
        rate_info = {}
        
        headers = getattr(response, 'headers', {})
        
        # Common rate limit headers
        limit_headers = {
            'limit': ['x-ratelimit-limit', 'rate-limit', 'x-rate-limit-limit'],
            'remaining': ['x-ratelimit-remaining', 'rate-limit-remaining', 'x-rate-limit-remaining'],
            'reset': ['x-ratelimit-reset', 'rate-limit-reset', 'x-rate-limit-reset'],
            'retry-after': ['retry-after']
        }
        
        for key, header_options in limit_headers.items():
            for header in header_options:
                if header in headers:
                    try:
                        if key == 'reset':
                            # Handle reset as timestamp or seconds
                            value = headers[header]
                            if len(value) > 10:  # Looks like a timestamp string
                                try:
                                    reset_time = parsedate_to_datetime(value)
                                    rate_info[key] = (reset_time - datetime.now(reset_time.tzinfo)).total_seconds()
                                except:
                                    rate_info[key] = float(value)
                            else:
                                rate_info[key] = float(value)
                        elif key == 'retry-after':
                            rate_info[key] = self.parse_retry_after(headers[header])
                        else:
                            rate_info[key] = int(headers[header])
                        break
                    except (ValueError, TypeError):
                        pass
        
        return rate_info
    
    def parse_retry_after(self, header_value: str) -> float:
        """
        Parse a Retry-After header value.
        
        Args:
            header_value: Value of the Retry-After header
            
        Returns:
            Number of seconds to wait
        """
        try:
            # First try to parse as a seconds value
            return float(header_value)
        except (ValueError, TypeError):
            try:
                # Try to parse as a date
                retry_date = parsedate_to_datetime(header_value)
                return max(0, (retry_date - datetime.now(retry_date.tzinfo)).total_seconds())
            except:
                # If all else fails, return a default value
                return 60.0
    
    def calculate_backoff(self, attempts: int, factor: float = None) -> float:
        """
        Calculate exponential backoff time.
        
        Args:
            attempts: Number of attempts so far
            factor: Backoff factor (default: use the service default)
            
        Returns:
            Time to wait in seconds
        """
        factor = factor or self._default_backoff_factor
        
        # Calculate backoff with jitter
        backoff = factor ** attempts
        jitter = random.uniform(0, 0.1 * backoff)
        
        return backoff + jitter
    
    def estimate_optimal_delay(self, url: str) -> float:
        """
        Estimate the optimal delay between requests for a domain.
        
        Args:
            url: URL to estimate for
            
        Returns:
            Estimated optimal delay in seconds
        """
        domain = self._get_domain_from_url(url)
        policy = self._get_or_create_policy(domain)
        
        # Base delay on current RPM (seconds per request)
        base_delay = 60.0 / policy.rpm
        
        # Adjust based on recent rate limiting
        if policy.rate_limited_count > 0:
            # More aggressive delay if we've seen rate limiting
            recency_factor = 1.0
            time_since_last_rate_limit = time.time() - policy.last_rate_limited
            
            if time_since_last_rate_limit < 60:
                recency_factor = 2.0  # Very recent
            elif time_since_last_rate_limit < 300:
                recency_factor = 1.5  # Within last 5 minutes
            elif time_since_last_rate_limit < 1800:
                recency_factor = 1.25  # Within last 30 minutes
                
            return base_delay * recency_factor
        
        # Default behavior
        return base_delay
    
    def add_domain_group(self, group_name: str, domains: List[str], 
                         policy_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a group of domains that share rate limit policy.
        
        Args:
            group_name: Name of the group
            domains: List of domains in the group
            policy_config: Optional policy configuration for the group
        """
        with self._lock:
            # Create or update the group
            if group_name not in self._domain_groups:
                self._domain_groups[group_name] = {
                    'domains': set(domains),
                    'policy': policy_config or {}
                }
            else:
                self._domain_groups[group_name]['domains'].update(domains)
                if policy_config:
                    self._domain_groups[group_name]['policy'].update(policy_config)
                    
            # Update domain-to-group mappings
            for domain in domains:
                self._domain_to_groups[domain].add(group_name)
                
                # Update existing policies if the domain already has one
                if domain in self._domain_policies and policy_config:
                    policy = self._domain_policies[domain]
                    if 'rpm' in policy_config:
                        policy.set_rate(policy_config['rpm'])
                    if 'max_concurrent' in policy_config:
                        policy.set_max_concurrent(policy_config['max_concurrent'])
    
    def apply_group_rate_limit(self, group_name: str, rpm: int) -> None:
        """
        Apply a rate limit to all domains in a group.
        
        Args:
            group_name: Name of the group
            rpm: Requests per minute to apply
        """
        with self._lock:
            if group_name not in self._domain_groups:
                return
                
            # Update the group policy
            self._domain_groups[group_name]['policy']['rpm'] = rpm
            
            # Apply to all domains in the group
            for domain in self._domain_groups[group_name]['domains']:
                if domain in self._domain_policies:
                    self._domain_policies[domain].set_rate(rpm)
    
    def detect_related_domains(self, url: str) -> Set[str]:
        """
        Find domains that may be related to the given URL based on patterns.
        
        Args:
            url: URL to find related domains for
            
        Returns:
            Set of related domain names
        """
        domain = self._get_domain_from_url(url)
        related = set()
        
        # Find domains with similar patterns
        parts = domain.split('.')
        
        if len(parts) > 1:
            # Check for subdomains of the same parent domain
            parent_domain = '.'.join(parts[-2:])  # e.g., example.com
            
            with self._lock:
                for other_domain in self._domain_policies.keys():
                    if other_domain != domain and other_domain.endswith(parent_domain):
                        related.add(other_domain)
        
        return related
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about rate limiting.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            metrics = {
                'domains': {},
                'groups': {},
                'total_domains': len(self._domain_policies),
                'total_groups': len(self._domain_groups)
            }
            
            # Collect domain metrics
            for domain, policy in self._domain_policies.items():
                metrics['domains'][domain] = {
                    'rpm': policy.rpm,
                    'max_concurrent': policy.max_concurrent,
                    'rate_limited_count': policy.rate_limited_count,
                    'success_rate': sum(policy.success_history) / len(policy.success_history) if policy.success_history else None,
                    'groups': list(self._domain_to_groups.get(domain, []))
                }
                
            # Collect group metrics
            for group_name, group_data in self._domain_groups.items():
                metrics['groups'][group_name] = {
                    'domain_count': len(group_data['domains']),
                    'policy': group_data['policy']
                }
                
            return metrics