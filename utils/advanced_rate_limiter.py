"""
Advanced Rate Limiter for SmartScrape

This module provides rate limiting functionality with support for different
algorithms, user-based limiting, and integration with the API endpoints.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class LimitType(Enum):
    """Rate limit types"""
    GLOBAL = "global"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: int
    limit_type: LimitType
    burst_allowance: int = 0  # Extra requests allowed in burst
    
class TokenBucket:
    """Token bucket rate limiting algorithm"""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = max_tokens
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket"""
        async with self._lock:
            now = time.time()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class SlidingWindowRateLimiter:
    """Sliding window rate limiting algorithm"""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self._locks = defaultdict(asyncio.Lock)
    
    async def acquire(self, key: str) -> bool:
        """Try to acquire a rate limit token"""
        async with self._locks[key]:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests outside the window
            while self.requests[key] and self.requests[key][0] < window_start:
                self.requests[key].popleft()
            
            # Check if we can make another request
            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(now)
                return True
            
            return False
    
    async def time_until_available(self, key: str) -> float:
        """Get time until next request is allowed"""
        async with self._locks[key]:
            if not self.requests[key]:
                return 0.0
            
            now = time.time()
            oldest_request = self.requests[key][0]
            time_until_available = oldest_request + self.window_seconds - now
            return max(0.0, time_until_available)

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms and strategies"""
    
    def __init__(self):
        self.sliding_window_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Default configurations
        self._setup_default_configs()
    
    def _setup_default_configs(self):
        """Setup default rate limit configurations"""
        self.configs.update({
            'api_global': RateLimitConfig(
                max_requests=1000,
                window_seconds=3600,  # 1 hour
                limit_type=LimitType.GLOBAL
            ),
            'api_per_ip': RateLimitConfig(
                max_requests=100,
                window_seconds=3600,  # 1 hour per IP
                limit_type=LimitType.PER_IP
            ),
            'extract_endpoint': RateLimitConfig(
                max_requests=50,
                window_seconds=300,  # 5 minutes
                limit_type=LimitType.PER_ENDPOINT,
                burst_allowance=10
            ),
            'pipeline_endpoint': RateLimitConfig(
                max_requests=10,
                window_seconds=300,  # 5 minutes
                limit_type=LimitType.PER_ENDPOINT
            ),
            'stream_endpoint': RateLimitConfig(
                max_requests=5,
                window_seconds=300,  # 5 minutes
                limit_type=LimitType.PER_ENDPOINT
            )
        })
    
    def add_config(self, name: str, config: RateLimitConfig):
        """Add a new rate limit configuration"""
        self.configs[name] = config
        logger.info(f"Added rate limit config '{name}': {config.max_requests} requests per {config.window_seconds}s")
    
    def _get_limiter_key(self, config_name: str, identifier: str) -> str:
        """Generate a unique key for the limiter"""
        return f"{config_name}:{identifier}"
    
    async def check_rate_limit(self, config_name: str, identifier: str) -> tuple[bool, Optional[Dict]]:
        """
        Check if request is within rate limits
        
        Args:
            config_name: Name of the rate limit configuration
            identifier: Unique identifier (IP, user ID, etc.)
            
        Returns:
            Tuple of (allowed: bool, metadata: Dict)
        """
        if config_name not in self.configs:
            logger.warning(f"Rate limit config '{config_name}' not found, allowing request")
            return True, None
        
        config = self.configs[config_name]
        limiter_key = self._get_limiter_key(config_name, identifier)
        
        # Create limiter if it doesn't exist
        if limiter_key not in self.sliding_window_limiters:
            self.sliding_window_limiters[limiter_key] = SlidingWindowRateLimiter(
                max_requests=config.max_requests + config.burst_allowance,
                window_seconds=config.window_seconds
            )
        
        limiter = self.sliding_window_limiters[limiter_key]
        
        # Check rate limit
        allowed = await limiter.acquire(limiter_key)
        
        metadata = {
            'limit': config.max_requests,
            'window_seconds': config.window_seconds,
            'limit_type': config.limit_type.value,
            'identifier': identifier
        }
        
        if not allowed:
            wait_time = await limiter.time_until_available(limiter_key)
            metadata['retry_after'] = wait_time
            metadata['message'] = f"Rate limit exceeded. Try again in {wait_time:.1f} seconds"
        
        return allowed, metadata
    
    async def check_multiple_limits(self, checks: list[tuple[str, str]]) -> tuple[bool, list[Dict]]:
        """
        Check multiple rate limits simultaneously
        
        Args:
            checks: List of (config_name, identifier) tuples
            
        Returns:
            Tuple of (all_allowed: bool, metadata_list: List[Dict])
        """
        results = []
        all_allowed = True
        
        for config_name, identifier in checks:
            allowed, metadata = await self.check_rate_limit(config_name, identifier)
            results.append(metadata)
            if not allowed:
                all_allowed = False
        
        return all_allowed, results
    
    def get_rate_limit_headers(self, config_name: str, identifier: str) -> Dict[str, str]:
        """Get rate limit headers for HTTP responses"""
        if config_name not in self.configs:
            return {}
        
        config = self.configs[config_name]
        limiter_key = self._get_limiter_key(config_name, identifier)
        
        if limiter_key in self.sliding_window_limiters:
            limiter = self.sliding_window_limiters[limiter_key]
            remaining = max(0, config.max_requests - len(limiter.requests[limiter_key]))
        else:
            remaining = config.max_requests
        
        return {
            'X-RateLimit-Limit': str(config.max_requests),
            'X-RateLimit-Remaining': str(remaining),
            'X-RateLimit-Window': str(config.window_seconds),
            'X-RateLimit-Type': config.limit_type.value
        }
    
    async def cleanup_expired_limiters(self):
        """Clean up expired limiters to prevent memory leaks"""
        current_time = time.time()
        expired_keys = []
        
        for key, limiter in self.sliding_window_limiters.items():
            # Check if limiter has been inactive
            if not limiter.requests[key] or (
                current_time - limiter.requests[key][-1] > limiter.window_seconds * 2
            ):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.sliding_window_limiters[key]
            if key in self.token_buckets:
                del self.token_buckets[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired rate limiters")

# Global rate limiter instance
advanced_rate_limiter = AdvancedRateLimiter()

# Utility functions for FastAPI integration
def get_client_ip(request) -> str:
    """Extract client IP address from request"""
    # Check for forwarded headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"

def generate_user_key(request, user_id: Optional[str] = None) -> str:
    """Generate a unique key for user-based rate limiting"""
    if user_id:
        return f"user:{user_id}"
    
    # Fallback to IP-based identification
    ip = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")
    
    # Create a hash for anonymized tracking
    identifier = f"{ip}:{user_agent}"
    return f"anon:{hashlib.md5(identifier.encode()).hexdigest()[:12]}"
