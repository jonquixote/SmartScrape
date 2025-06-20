"""
Rate limiting module for SmartScrape API.
Provides custom rate limiting capabilities when the slowapi package is not available.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Dict, Any, Callable
import os

import config

# Simple in-memory rate limiting storage
# Format: {"ip_address": {"count": 10, "window": 12345}}
rate_limits = {}

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement rate limiting based on client IP address.
    Falls back to this if slowapi is not available.
    """
    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        exclude_paths: list = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting in development mode or for excluded paths
        if config.ENVIRONMENT == "development" or any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
            
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        current_minute = int(time.time()) // 60  # Current minute window
        
        # Initialize or reset counter for new window
        if client_ip not in rate_limits or rate_limits[client_ip]["window"] < current_minute:
            rate_limits[client_ip] = {"count": 0, "window": current_minute}
        
        # Increment counter
        rate_limits[client_ip]["count"] += 1
        
        # Check if over limit
        if rate_limits[client_ip]["count"] > self.requests_per_minute:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests", 
                    "detail": f"Rate limit of {self.requests_per_minute} requests per minute exceeded",
                    "retry_after": 60 - (int(time.time()) % 60)  # Seconds until next window
                },
                headers={"Retry-After": str(60 - (int(time.time()) % 60))},
            )
        
        # Process request if within limits
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - rate_limits[client_ip]["count"])
        reset_time = (current_minute + 1) * 60
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response

def fallback_rate_limit_factory(requests_per_minute: int = 100):
    """
    Factory function to create a rate limiter function that can be used as a decorator.
    This is a fallback if slowapi is not available.
    """
    def rate_limit_decorator(func: Callable) -> Callable:
        """Decorator to apply rate limiting to individual endpoints."""
        async def wrapper(*args, **kwargs) -> Any:
            # Extract request from args
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                for _, value in kwargs.items():
                    if isinstance(value, Request):
                        request = value
                        break
            
            if request and hasattr(request, "client") and request.client:
                # Get client IP
                client_ip = request.client.host
                current_minute = int(time.time()) // 60  # Current minute window
                
                # Initialize or reset counter for new window
                if client_ip not in rate_limits or rate_limits[client_ip]["window"] < current_minute:
                    rate_limits[client_ip] = {"count": 0, "window": current_minute}
                
                # Increment counter
                rate_limits[client_ip]["count"] += 1
                
                # Check if over limit
                if rate_limits[client_ip]["count"] > requests_per_minute:
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit of {requests_per_minute} requests per minute exceeded",
                        headers={"Retry-After": str(60 - (int(time.time()) % 60))},
                    )
            
            # Call the original function if within limits
            return await func(*args, **kwargs)
        
        return wrapper
    
    return rate_limit_decorator
