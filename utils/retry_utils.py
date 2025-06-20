"""
Resilience and performance utilities for SmartScrape.

This module provides common retry patterns, connection helpers, and 
performance optimization utilities that can be used across the project.
"""

import logging
from functools import wraps
from typing import Callable, Type, TypeVar, Any, Optional, List, Dict, Union, Tuple

# Tenacity imports for retry patterns
from tenacity import (
    retry as tenacity_retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    RetryError,
    wait_fixed,
    before_sleep_log
)

from core.retry_manager import RetryManager

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')

# Common exceptions to retry on
NETWORK_EXCEPTIONS = (
    ConnectionError,       # Base class for connection-related errors
    TimeoutError,          # Timeout errors
    IOError,               # I/O errors
    OSError,               # OS errors (includes connection issues)
)

# Add the missing functions that are being imported in http_utils.py
def retry_on_network_errors(max_attempts: int = 3) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on network errors.
    
    This is an alias for with_exponential_backoff to maintain backward compatibility.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with retry logic
    """
    return with_exponential_backoff(max_attempts=max_attempts)

def retry_on_http_errors(max_attempts: int = 3) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on HTTP errors.
    
    This is an alias for with_http_retry to maintain backward compatibility.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with HTTP-specific retry logic
    """
    return with_http_retry(max_attempts=max_attempts)

def with_exponential_backoff(
    max_attempts: int = 3, 
    min_wait: float = 1.0, 
    max_wait: float = 10.0,
    exception_types: Tuple[Type[Exception], ...] = NETWORK_EXCEPTIONS
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds exponential backoff retry logic to a function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exception_types: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    return tenacity_retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exception_types),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )

def with_fixed_backoff(
    max_attempts: int = 3, 
    wait_time: float = 2.0,
    exception_types: Tuple[Type[Exception], ...] = NETWORK_EXCEPTIONS
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds fixed backoff retry logic to a function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        wait_time: Fixed wait time between retries (seconds)
        exception_types: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    return tenacity_retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_fixed(wait_time),
        retry=retry_if_exception_type(exception_types),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )

# Specialized retry patterns for specific use cases
def with_http_retry(max_attempts: int = 3) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Specialized retry for HTTP operations, including httpx-specific exceptions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with HTTP-specific retry logic
    """
    try:
        import httpx
        http_exceptions = (
            httpx.HTTPError,          # Base class for httpx errors
            httpx.NetworkError,       # Network-related errors
            httpx.TimeoutException,   # Timeout errors
            httpx.ConnectError,       # Connection errors
            httpx.ReadError,          # Read errors
            httpx.WriteError,         # Write errors
            httpx.ProtocolError       # Protocol errors
        )
    except ImportError:
        # If httpx is not available, use standard network exceptions
        http_exceptions = NETWORK_EXCEPTIONS
    
    return with_exponential_backoff(
        max_attempts=max_attempts,
        exception_types=http_exceptions + NETWORK_EXCEPTIONS
    )

def with_browser_retry(max_attempts: int = 2) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Specialized retry for browser automation operations with Playwright.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with browser-specific retry logic
    """
    try:
        from playwright.async_api import Error as PlaywrightError
        browser_exceptions = (PlaywrightError,)
    except ImportError:
        # If playwright is not available, use standard exceptions
        browser_exceptions = (Exception,)
    
    return with_exponential_backoff(
        max_attempts=max_attempts,
        min_wait=2.0,
        max_wait=15.0,
        exception_types=browser_exceptions + NETWORK_EXCEPTIONS
    )

def with_file_operation_retry(max_attempts: int = 3) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Specialized retry for file operations.
    
    Args:
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Decorated function with file operation retry logic
    """
    file_exceptions = (
        IOError,       # I/O errors
        OSError,       # OS errors
        PermissionError # Permission errors
    )
    
    return with_exponential_backoff(
        max_attempts=max_attempts,
        min_wait=0.5,
        max_wait=5.0,
        exception_types=file_exceptions
    )

# Async versions of the retry decorators
async def with_async_retry(
    func: Callable[..., Any],
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exception_types: Tuple[Type[Exception], ...] = NETWORK_EXCEPTIONS
) -> Any:
    """
    Execute an async function with retry logic.
    
    Args:
        func: Async function to execute
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        exception_types: Tuple of exception types to retry on
        
    Returns:
        Result of the async function
    """
    import asyncio
    attempt = 0
    last_exception = None
    
    while attempt < max_attempts:
        try:
            return await func()
        except exception_types as e:
            attempt += 1
            last_exception = e
            
            if attempt >= max_attempts:
                break
                
            # Calculate wait time with exponential backoff
            wait_time = min(max_wait, min_wait * (2 ** (attempt - 1)))
            
            logger.warning(
                f"Retrying {func.__name__} after exception {str(e)}. "
                f"Attempt {attempt}/{max_attempts}, waiting {wait_time:.2f}s"
            )
            
            await asyncio.sleep(wait_time)
    
    # If we've exhausted all attempts, raise the last exception
    if last_exception:
        raise last_exception
    
    # This shouldn't happen, but just in case
    raise Exception(f"Failed to execute {func.__name__} after {max_attempts} attempts")

# Create a utility function to access the retry decorator from RetryManager
def retry_with_manager(**kwargs):
    """
    Retry decorator from RetryManager instance.
    
    This is a utility function that provides access to RetryManager's retry decorator
    so it can be used without manually creating a RetryManager instance.
    
    Args:
        **kwargs: Arguments to pass to RetryManager.retry
        
    Returns:
        Retry decorator from RetryManager
    """
    retry_manager = RetryManager()
    
    # Map parameter names that might be used in different parts of the codebase
    if 'max_retries' in kwargs:
        kwargs['max_attempts'] = kwargs.pop('max_retries')
    
    if 'retry_delay' in kwargs:
        kwargs['base_delay'] = kwargs.pop('retry_delay')
    
    # Map 'exceptions' to 'retry_on' which is what RetryManager.retry uses
    if 'exceptions' in kwargs:
        kwargs['retry_on'] = kwargs.pop('exceptions')
    
    return retry_manager.retry(**kwargs)