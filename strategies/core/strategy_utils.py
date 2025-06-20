"""
Strategy execution utilities for enhancing strategy performance and reliability.

This module provides decorators and utility functions for:
- Retrying operations that may fail
- Enforcing timeout limits
- Measuring performance
- Logging execution details
- Validating service dependencies
"""

import time
import logging
import threading
import traceback
import functools
import inspect
import os
import psutil
from typing import Callable, Dict, Any, Optional, List, TypeVar, Union, Set, Tuple, Type

logger = logging.getLogger(__name__)

# Type variable for better type hinting with decorators
T = TypeVar('T')

def retry_on_failure(max_attempts: int = 3, 
                    delay_seconds: float = 1.0, 
                    backoff_factor: float = 2.0, 
                    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
                    log_level: int = logging.INFO) -> Callable:
    """
    Decorator that retries a function if it raises an exception.
    
    Args:
        max_attempts: Maximum number of attempts before giving up
        delay_seconds: Initial delay between attempts (in seconds)
        backoff_factor: Factor by which to increase delay for each attempt
        exceptions: Exception(s) to catch and retry on
        log_level: Logging level for retry messages
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get function name and args for logging
            func_name = func.__qualname__
            
            # Try to get strategy name from the instance for better logging
            strategy_name = "unknown"
            if args and hasattr(args[0], 'name'):
                strategy_name = args[0].name
            
            attempt = 1
            last_exception = None
            
            while attempt <= max_attempts:
                try:
                    if attempt > 1:
                        logger.log(log_level, f"Retry attempt {attempt}/{max_attempts} for {strategy_name}.{func_name}")
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 1:
                        logger.log(log_level, f"Succeeded on attempt {attempt}/{max_attempts} for {strategy_name}.{func_name}")
                    
                    return result
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        # Calculate delay with exponential backoff
                        current_delay = delay_seconds * (backoff_factor ** (attempt - 1))
                        logger.log(log_level, f"Attempt {attempt}/{max_attempts} failed for {strategy_name}.{func_name}: {str(e)}. Retrying in {current_delay:.2f}s")
                        time.sleep(current_delay)
                    else:
                        logger.log(log_level, f"All {max_attempts} attempts failed for {strategy_name}.{func_name}. Giving up.")
                
                attempt += 1
            
            # If we get here, all attempts failed
            if last_exception:
                raise last_exception
            
            # This should never happen, but just in case
            raise RuntimeError(f"All retry attempts failed for {func_name} but no exception was captured")
        
        return wrapper
    
    return decorator


def with_timeout(timeout_seconds: float, cleanup_callback: Optional[Callable] = None) -> Callable:
    """
    Decorator that enforces a timeout on a function.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        cleanup_callback: Optional callback to clean up resources if timeout occurs
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get function name for logging
            func_name = func.__qualname__
            
            # Try to get strategy name from the instance for better logging
            strategy_name = "unknown"
            if args and hasattr(args[0], 'name'):
                strategy_name = args[0].name
            
            # Variables to be shared between threads
            result = [None]
            exception = [None]
            execution_completed = [False]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                    execution_completed[0] = True
                except Exception as e:
                    exception[0] = e
                    execution_completed[0] = True
            
            # Create and start thread
            thread = threading.Thread(target=target)
            thread.daemon = True  # Thread will be killed when main thread exits
            
            start_time = time.time()
            thread.start()
            thread.join(timeout_seconds)
            
            if not execution_completed[0]:
                elapsed = time.time() - start_time
                error_msg = f"Timeout of {timeout_seconds}s exceeded for {strategy_name}.{func_name} (elapsed: {elapsed:.2f}s)"
                logger.error(error_msg)
                
                # Call cleanup callback if provided
                if cleanup_callback:
                    try:
                        logger.info(f"Executing cleanup callback for {strategy_name}.{func_name}")
                        cleanup_callback(*args, **kwargs)
                    except Exception as cleanup_exception:
                        logger.error(f"Error in cleanup callback: {cleanup_exception}")
                
                raise TimeoutError(error_msg)
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    
    return decorator


def measure_performance(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that measures and logs the performance of a function.
    
    This decorator:
    1. Tracks execution time
    2. Records memory usage
    3. Logs performance metrics
    4. Adds measurements to strategy metrics if available
    
    Args:
        func: The function to measure
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        # Get function name for logging
        func_name = func.__qualname__
        
        # Try to get strategy name from the instance for better logging
        strategy_name = "unknown"
        if args and hasattr(args[0], 'name'):
            strategy_name = args[0].name
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Time the execution
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            exception = e
            raise
        finally:
            # Calculate metrics even if an exception occurred
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get final memory usage and calculate delta
            end_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            memory_delta = end_memory - start_memory
            
            # Determine appropriate log level based on execution time
            if execution_time < 0.1:
                log_level = logging.DEBUG
            elif execution_time < 1.0:
                log_level = logging.INFO
            elif execution_time < 5.0:
                log_level = logging.WARNING
            else:
                log_level = logging.ERROR
            
            status = "succeeded" if success else "failed"
            log_message = (
                f"Performance: {strategy_name}.{func_name} {status} in {execution_time:.3f}s "
                f"(memory: {memory_delta:+.2f}MB, total: {end_memory:.2f}MB)"
            )
            
            logger.log(log_level, log_message)
            
            # Add to strategy metrics if available
            if args and hasattr(args[0], 'add_metric'):
                try:
                    args[0].add_metric(f"{func_name}_execution_time", execution_time)
                    args[0].add_metric(f"{func_name}_memory_delta", memory_delta)
                except Exception as metrics_exception:
                    logger.warning(f"Could not add metrics for {strategy_name}.{func_name}: {metrics_exception}")
        
        return result
    
    return wrapper


def log_execution(level: int = logging.INFO, 
                 log_args: bool = True, 
                 log_result: bool = False,
                 sensitive_params: Optional[Set[str]] = None) -> Callable:
    """
    Decorator that logs the execution of a function.
    
    Args:
        level: Logging level
        log_args: Whether to log function arguments
        log_result: Whether to log function return value
        sensitive_params: Set of parameter names that should be masked for security
        
    Returns:
        Decorator function
    """
    if sensitive_params is None:
        sensitive_params = {'password', 'token', 'secret', 'key', 'auth', 'credential'}
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get function name for logging
            func_name = func.__qualname__
            
            # Try to get strategy name from the instance for better logging
            strategy_name = "unknown"
            if args and hasattr(args[0], 'name'):
                strategy_name = args[0].name
            
            # Filter sensitive parameters
            safe_kwargs = {}
            if log_args:
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Process positional args (skip 'self' if it's a method)
                start_idx = 1 if param_names and param_names[0] == 'self' else 0
                for i, value in enumerate(args[start_idx:], start_idx):
                    if i < len(param_names):
                        param_name = param_names[i]
                        if param_name in sensitive_params:
                            safe_kwargs[param_name] = "***MASKED***"
                        else:
                            safe_kwargs[param_name] = _truncate_value(value)
                
                # Process keyword args
                for k, v in kwargs.items():
                    if k in sensitive_params:
                        safe_kwargs[k] = "***MASKED***"
                    else:
                        safe_kwargs[k] = _truncate_value(v)
            
            # Log the start of execution
            if log_args:
                logger.log(level, f"Executing {strategy_name}.{func_name} with args: {safe_kwargs}")
            else:
                logger.log(level, f"Executing {strategy_name}.{func_name}")
            
            # Execute the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                execution_time = time.time() - start_time
                
                # Log the exception
                logger.log(level, f"Error in {strategy_name}.{func_name} after {execution_time:.3f}s: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                
                raise
            
            # Log the result
            execution_time = time.time() - start_time
            if log_result and success:
                safe_result = _truncate_value(result)
                logger.log(level, f"Completed {strategy_name}.{func_name} in {execution_time:.3f}s with result: {safe_result}")
            else:
                logger.log(level, f"Completed {strategy_name}.{func_name} in {execution_time:.3f}s")
            
            return result
        
        return wrapper
    
    return decorator


def require_service(*service_names: str) -> Callable:
    """
    Decorator that validates required services are available in the strategy context.
    
    Args:
        service_names: Names of services required by the function
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            # Validate that the instance has a context attribute
            if not hasattr(self, 'context') or self.context is None:
                raise ValueError(f"Strategy '{getattr(self, 'name', self.__class__.__name__)}' requires a context to access services")
            
            # Validate each required service
            missing_services = []
            for service_name in service_names:
                try:
                    # Try to access the service to check if it's available
                    self.context.get_service(service_name)
                except Exception as e:
                    missing_services.append(f"{service_name}: {str(e)}")
            
            if missing_services:
                raise ValueError(f"Required services missing: {', '.join(missing_services)}")
            
            # If we get here, all services are available
            return func(self, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Helper functions

def _truncate_value(value: Any, max_length: int = 500) -> Any:
    """Truncate a value for logging purposes."""
    if isinstance(value, (str, bytes)):
        str_value = str(value)
        if len(str_value) > max_length:
            return str_value[:max_length] + "... [truncated]"
    elif isinstance(value, (list, tuple)):
        if len(value) > 10:
            return f"{value[:10]} + {len(value) - 10} more items"
    elif isinstance(value, dict):
        if len(value) > 10:
            return f"{dict(list(value.items())[:10])} + {len(value) - 10} more items"
    
    return value