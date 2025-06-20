import logging
import random
import threading
import time
import asyncio
import functools
from concurrent.futures import TimeoutError
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Callable, Type, TypeVar, cast
from enum import Enum, auto

from core.service_interface import BaseService
from core.error_classifier import ErrorCategory, ErrorSeverity, ErrorClassifier

logger = logging.getLogger(__name__)

# Type variables for function decorators
T = TypeVar('T')
R = TypeVar('R')

class RetryOutcome(Enum):
    """Possible outcomes of a retry operation."""
    SUCCESS = auto()
    FAILED_RETRIES_EXHAUSTED = auto()
    FAILED_NON_RETRYABLE = auto()
    FAILED_BUDGET_EXHAUSTED = auto()
    FAILED_TIMEOUT = auto()

class RetryManager(BaseService):
    """Service for managing operation retries with sophisticated retry policies."""
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self._default_max_attempts = 3
        self._default_backoff_factor = 1.5
        self._default_retry_codes = {429, 500, 502, 503, 504}
        self._default_retry_exceptions = (
            TimeoutError,
            ConnectionError,
            ConnectionResetError,
            ConnectionRefusedError,
            ConnectionAbortedError,
        )
        self._jitter_factor = 0.1
        
        # Tracking structures
        self._attempt_counters = {}
        self._retry_budgets = {}
        self._budget_refill_times = {}
        self._locks = {}
        self._global_lock = threading.RLock()
        
        # Error classifier reference
        self._error_classifier = None
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the retry manager with configuration."""
        if self._initialized:
            return
            
        self._config = config or {}
        
        # Load config values with defaults
        self._default_max_attempts = self._config.get('default_max_attempts', 3)
        self._default_backoff_factor = self._config.get('default_backoff_factor', 1.5)
        self._default_retry_codes = set(self._config.get('default_retry_codes', [429, 500, 502, 503, 504]))
        self._jitter_factor = self._config.get('jitter_factor', 0.1)
        
        # Initialize retry budgets
        default_budget = self._config.get('default_retry_budget', 50)
        domains = self._config.get('domains', [])
        for domain in domains:
            domain_budget = domain.get('retry_budget', default_budget)
            self._retry_budgets[domain['name']] = domain_budget
            
        # Mark as initialized
        self._initialized = True
        logger.info("Retry manager initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if not self._initialized:
            return
            
        with self._global_lock:
            self._attempt_counters.clear()
            self._retry_budgets.clear()
            self._budget_refill_times.clear()
            self._locks.clear()
        
        self._initialized = False
        logger.info("Retry manager shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "retry_manager"
    
    def _get_lock(self, key: str) -> threading.RLock:
        """Get or create a lock for a specific key."""
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.RLock()
            return self._locks[key]
    
    def should_retry(self, attempt: int, error: Optional[Exception] = None, 
                    context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Determine if a retry should be attempted based on the error and context.
        
        Args:
            attempt: Current attempt number (0-based, where 0 is the first attempt)
            error: The exception that was raised, if any
            context: Additional context such as response status code, domain, etc.
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        context = context or {}
        max_attempts = context.get('max_attempts', self._default_max_attempts)
        
        # Check if we've exceeded max attempts
        if attempt >= max_attempts:
            logger.debug(f"Retry declined: Max attempts ({max_attempts}) reached")
            return False
        
        # If no error, always retry
        if error is None:
            return True
        
        # If domain specified, check the retry budget
        domain = context.get('domain')
        if domain and self.is_budget_exhausted(domain):
            logger.warning(f"Retry declined: Budget exhausted for domain {domain}")
            return False
        
        # Use error classifier if available
        if self._error_classifier is None:
            # Try to get error classifier from service registry
            try:
                from core.service_registry import ServiceRegistry
                self._error_classifier = ServiceRegistry().get_service("error_classifier")
            except Exception as e:
                logger.debug(f"Could not get error_classifier service: {str(e)}")
        
        # If we have an error classifier, use it
        if self._error_classifier is not None:
            try:
                classification = self._error_classifier.classify_exception(error, context)
                return classification.get('is_retryable', False)
            except Exception as e:
                logger.warning(f"Error using error classifier: {str(e)}")
        
        # Fallback to basic classification
        if isinstance(error, self._default_retry_exceptions):
            return True
            
        # Check HTTP status codes
        status_code = context.get('status_code')
        if status_code and status_code in self._default_retry_codes:
            return True
            
        # By default, don't retry
        return False
    
    def get_retry_delay(self, attempt: int, error: Optional[Exception] = None, 
                      context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate the delay before the next retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            error: The exception that was raised, if any
            context: Additional context such as domain, strategy, etc.
            
        Returns:
            Delay in seconds
        """
        context = context or {}
        base_delay = context.get('base_delay', 1.0)
        backoff_factor = context.get('backoff_factor', self._default_backoff_factor)
        delay_strategy = context.get('delay_strategy', 'exponential')
        max_delay = context.get('max_delay', 60.0)
        
        # Calculate delay based on the selected strategy
        if delay_strategy == 'constant':
            delay = base_delay
        elif delay_strategy == 'linear':
            delay = base_delay * (attempt + 1)
        elif delay_strategy == 'fibonacci':
            if attempt <= 1:
                delay = base_delay * (attempt + 1)
            else:
                a, b = base_delay, base_delay * 2
                for _ in range(2, attempt + 1):
                    a, b = b, a + b
                delay = b
        else:  # Default to exponential
            delay = base_delay * (backoff_factor ** attempt)
        
        # Apply jitter to avoid thundering herd
        delay = self.add_jitter(delay)
        
        # Check for retry-after in context
        retry_after = context.get('retry_after')
        if retry_after is not None:
            delay = max(delay, float(retry_after))
        
        # Apply maximum delay limit
        delay = min(delay, max_delay)
        
        return delay
    
    def register_attempt(self, operation_id: str, success: bool, 
                        error: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Track an operation attempt's outcome.
        
        Args:
            operation_id: Unique identifier for the operation
            success: Whether the attempt was successful
            error: The exception that was raised, if any
            
        Returns:
            Dict with attempt information
        """
        lock = self._get_lock(operation_id)
        with lock:
            if operation_id not in self._attempt_counters:
                self._attempt_counters[operation_id] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'failed_attempts': 0,
                    'last_attempt_time': time.time(),
                    'last_error': None,
                    'last_success_time': None
                }
            
            counter = self._attempt_counters[operation_id]
            counter['total_attempts'] += 1
            
            if success:
                counter['successful_attempts'] += 1
                counter['last_success_time'] = time.time()
            else:
                counter['failed_attempts'] += 1
                counter['last_error'] = error
            
            counter['last_attempt_time'] = time.time()
            
            return dict(counter)  # Return a copy
    
    def get_attempt_count(self, operation_id: str) -> Dict[str, Any]:
        """
        Get the current attempt counts for an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            
        Returns:
            Dict with attempt counts or empty dict if not found
        """
        with self._global_lock:
            return dict(self._attempt_counters.get(operation_id, {
                'total_attempts': 0,
                'successful_attempts': 0,
                'failed_attempts': 0,
                'last_attempt_time': None,
                'last_error': None,
                'last_success_time': None
            }))
    
    def reset_attempts(self, operation_id: str) -> None:
        """
        Reset the attempt counter for an operation.
        
        Args:
            operation_id: Unique identifier for the operation
        """
        with self._global_lock:
            if operation_id in self._attempt_counters:
                del self._attempt_counters[operation_id]
    
    def get_retry_budget(self, domain: str) -> int:
        """
        Get the remaining retry budget for a domain.
        
        Args:
            domain: The domain to check
            
        Returns:
            Number of retries remaining
        """
        with self._global_lock:
            # Auto-refill budget if needed
            self._check_budget_refill(domain)
            return self._retry_budgets.get(domain, self._config.get('default_retry_budget', 50))
    
    def consume_retry_budget(self, domain: str, amount: int = 1) -> int:
        """
        Consume some of the retry budget for a domain.
        
        Args:
            domain: The domain to consume budget for
            amount: The amount to consume
            
        Returns:
            Updated budget after consumption
        """
        with self._global_lock:
            # Initialize budget if not set
            if domain not in self._retry_budgets:
                self._retry_budgets[domain] = self._config.get('default_retry_budget', 50)
            
            # Auto-refill budget if needed
            self._check_budget_refill(domain)
            
            # Consume budget
            current = self._retry_budgets[domain]
            self._retry_budgets[domain] = max(0, current - amount)
            
            return self._retry_budgets[domain]
    
    def refill_retry_budget(self, domain: str, amount: Optional[int] = None) -> int:
        """
        Refill the retry budget for a domain.
        
        Args:
            domain: The domain to refill budget for
            amount: The amount to refill, or None to reset to default
            
        Returns:
            Updated budget after refill
        """
        with self._global_lock:
            if amount is None:
                # Reset to default
                default_budget = self._config.get('default_retry_budget', 50)
                
                # Check for domain-specific default
                domains = self._config.get('domains', [])
                for domain_config in domains:
                    if domain_config.get('name') == domain:
                        default_budget = domain_config.get('retry_budget', default_budget)
                        break
                
                self._retry_budgets[domain] = default_budget
            else:
                # Add to current budget
                current = self._retry_budgets.get(domain, 0)
                self._retry_budgets[domain] = current + amount
            
            # Update refill time
            self._budget_refill_times[domain] = time.time()
            
            return self._retry_budgets[domain]
    
    def is_budget_exhausted(self, domain: str) -> bool:
        """
        Check if the retry budget for a domain is exhausted.
        
        Args:
            domain: The domain to check
            
        Returns:
            True if the budget is exhausted (0), False otherwise
        """
        with self._global_lock:
            # Auto-refill budget if needed
            self._check_budget_refill(domain)
            return self._retry_budgets.get(domain, 1) <= 0
    
    def _check_budget_refill(self, domain: str) -> None:
        """Check if it's time to auto-refill the budget for a domain."""
        # Get refill period from config
        refill_period = self._config.get('retry_budget_refill_period', 3600)  # Default: 1 hour
        refill_amount = self._config.get('retry_budget_refill_amount', 10)
        
        # Domain-specific settings
        domains = self._config.get('domains', [])
        for domain_config in domains:
            if domain_config.get('name') == domain:
                refill_period = domain_config.get('retry_budget_refill_period', refill_period)
                refill_amount = domain_config.get('retry_budget_refill_amount', refill_amount)
                break
        
        # Check if it's time to refill
        now = time.time()
        last_refill = self._budget_refill_times.get(domain, 0)
        
        if now - last_refill > refill_period:
            # Calculate how many refills we've missed
            periods_elapsed = int((now - last_refill) / refill_period)
            amount_to_add = periods_elapsed * refill_amount
            
            # Add to current budget
            current = self._retry_budgets.get(domain, 0)
            max_budget = self._config.get('default_retry_budget', 50)
            
            # Apply domain-specific max if available
            for domain_config in domains:
                if domain_config.get('name') == domain:
                    max_budget = domain_config.get('retry_budget', max_budget)
                    break
            
            # Update budget and refill time
            self._retry_budgets[domain] = min(current + amount_to_add, max_budget)
            self._budget_refill_times[domain] = now - (now - last_refill) % refill_period
    
    # Retry decorator utilities
    def retry(self, max_attempts: int = None, delay_strategy: str = 'exponential',
              base_delay: float = 1.0, backoff_factor: float = None,
              retry_on: Union[List[Type[Exception]], Type[Exception]] = None,
              retry_if: Callable[[Exception], bool] = None,
              retry_unless: Callable[[Exception], bool] = None):
        """
        Decorator for retrying functions with exponential backoff.
        
        Args:
            max_attempts: Maximum number of attempts (None uses default)
            delay_strategy: Strategy for calculating delay ('constant', 'linear', 'exponential', 'fibonacci')
            base_delay: Base delay in seconds
            backoff_factor: Factor by which the delay increases (None uses default)
            retry_on: Exception types to retry on
            retry_if: Function that returns True if retry should happen
            retry_unless: Function that returns True if retry should NOT happen
            
        Returns:
            Decorated function
        """
        max_attempts = max_attempts or self._default_max_attempts
        backoff_factor = backoff_factor or self._default_backoff_factor
        
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                operation_id = f"{func.__module__}.{func.__qualname__}"
                
                # Reset attempt counter for this operation
                self.reset_attempts(operation_id)
                
                attempt = 0
                while True:
                    try:
                        result = func(*args, **kwargs)
                        self.register_attempt(operation_id, True)
                        return result
                    except Exception as e:
                        self.register_attempt(operation_id, False, e)
                        
                        # Check if we should retry
                        context = {
                            'max_attempts': max_attempts,
                            'delay_strategy': delay_strategy,
                            'base_delay': base_delay,
                            'backoff_factor': backoff_factor,
                            'operation_id': operation_id
                        }
                        
                        # Add domain to context if provided in kwargs
                        if 'domain' in kwargs:
                            context['domain'] = kwargs['domain']
                        
                        # Check custom retry conditions
                        should_retry_custom = True
                        if retry_on and not isinstance(e, retry_on):
                            should_retry_custom = False
                        if retry_if and not retry_if(e):
                            should_retry_custom = False
                        if retry_unless and retry_unless(e):
                            should_retry_custom = False
                        
                        # Check if we should retry using the manager's logic
                        should_retry_manager = self.should_retry(attempt, e, context)
                        
                        if should_retry_custom and should_retry_manager and attempt < max_attempts - 1:
                            # Consume budget if domain is specified
                            if 'domain' in context:
                                self.consume_retry_budget(context['domain'])
                            
                            # Calculate delay
                            delay = self.get_retry_delay(attempt, e, context)
                            
                            logger.info(f"Retry {attempt+1}/{max_attempts} for {operation_id} in {delay:.2f}s: {str(e)}")
                            time.sleep(delay)
                            attempt += 1
                        else:
                            logger.warning(f"Retry giving up after {attempt+1} attempts: {str(e)}")
                            raise
            
            return wrapper
        
        return decorator
    
    def retry_async(self, max_attempts: int = None, delay_strategy: str = 'exponential',
                   base_delay: float = 1.0, backoff_factor: float = None,
                   retry_on: Union[List[Type[Exception]], Type[Exception]] = None,
                   retry_if: Callable[[Exception], bool] = None,
                   retry_unless: Callable[[Exception], bool] = None):
        """
        Decorator for retrying async functions with exponential backoff.
        
        Args:
            max_attempts: Maximum number of attempts (None uses default)
            delay_strategy: Strategy for calculating delay ('constant', 'linear', 'exponential', 'fibonacci')
            base_delay: Base delay in seconds
            backoff_factor: Factor by which the delay increases (None uses default)
            retry_on: Exception types to retry on
            retry_if: Function that returns True if retry should happen
            retry_unless: Function that returns True if retry should NOT happen
            
        Returns:
            Decorated async function
        """
        max_attempts = max_attempts or self._default_max_attempts
        backoff_factor = backoff_factor or self._default_backoff_factor
        
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                operation_id = f"{func.__module__}.{func.__qualname__}"
                
                # Reset attempt counter for this operation
                self.reset_attempts(operation_id)
                
                attempt = 0
                while True:
                    try:
                        result = await func(*args, **kwargs)
                        self.register_attempt(operation_id, True)
                        return result
                    except Exception as e:
                        self.register_attempt(operation_id, False, e)
                        
                        # Check if we should retry
                        context = {
                            'max_attempts': max_attempts,
                            'delay_strategy': delay_strategy,
                            'base_delay': base_delay,
                            'backoff_factor': backoff_factor,
                            'operation_id': operation_id
                        }
                        
                        # Add domain to context if provided in kwargs
                        if 'domain' in kwargs:
                            context['domain'] = kwargs['domain']
                        
                        # Check custom retry conditions
                        should_retry_custom = True
                        if retry_on and not isinstance(e, retry_on):
                            should_retry_custom = False
                        if retry_if and not retry_if(e):
                            should_retry_custom = False
                        if retry_unless and retry_unless(e):
                            should_retry_custom = False
                        
                        # Check if we should retry using the manager's logic
                        should_retry_manager = self.should_retry(attempt, e, context)
                        
                        if should_retry_custom and should_retry_manager and attempt < max_attempts - 1:
                            # Consume budget if domain is specified
                            if 'domain' in context:
                                self.consume_retry_budget(context['domain'])
                            
                            # Calculate delay
                            delay = self.get_retry_delay(attempt, e, context)
                            
                            logger.info(f"Retry {attempt+1}/{max_attempts} for {operation_id} in {delay:.2f}s: {str(e)}")
                            await asyncio.sleep(delay)
                            attempt += 1
                        else:
                            logger.warning(f"Retry giving up after {attempt+1} attempts: {str(e)}")
                            raise
            
            return wrapper
        
        return decorator
    
    def conditional_retry(self, condition: Callable[[Exception, Dict[str, Any]], bool], 
                         max_attempts: int = None, 
                         delay_func: Callable[[int, Exception, Dict[str, Any]], float] = None):
        """
        Advanced decorator for retrying with custom condition and delay functions.
        
        Args:
            condition: Function that takes (exception, context) and returns whether to retry
            max_attempts: Maximum number of attempts
            delay_func: Function that takes (attempt, exception, context) and returns delay
            
        Returns:
            Decorated function
        """
        max_attempts = max_attempts or self._default_max_attempts
        
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                operation_id = f"{func.__module__}.{func.__qualname__}"
                self.reset_attempts(operation_id)
                
                attempt = 0
                while True:
                    try:
                        result = func(*args, **kwargs)
                        self.register_attempt(operation_id, True)
                        return result
                    except Exception as e:
                        self.register_attempt(operation_id, False, e)
                        
                        # Build context
                        context = {
                            'max_attempts': max_attempts,
                            'operation_id': operation_id,
                            'attempt': attempt,
                            'args': args,
                            'kwargs': kwargs
                        }
                        
                        # Check if we should retry
                        if condition(e, context) and attempt < max_attempts - 1:
                            # Calculate delay
                            if delay_func:
                                delay = delay_func(attempt, e, context)
                            else:
                                delay = self.get_retry_delay(attempt, e, context)
                            
                            logger.info(f"Conditional retry {attempt+1}/{max_attempts} for {operation_id} in {delay:.2f}s")
                            time.sleep(delay)
                            attempt += 1
                        else:
                            logger.warning(f"Conditional retry giving up after {attempt+1} attempts: {str(e)}")
                            raise
            
            return wrapper
        
        return decorator
    
    def retry_with_timeout(self, timeout: float, max_attempts: int = None,
                         delay_strategy: str = 'exponential',
                         base_delay: float = 1.0):
        """
        Decorator for retrying functions with a total timeout.
        
        Args:
            timeout: Maximum total time in seconds to spend on retries
            max_attempts: Maximum number of attempts
            delay_strategy: Strategy for calculating delay
            base_delay: Base delay in seconds
            
        Returns:
            Decorated function
        """
        max_attempts = max_attempts or self._default_max_attempts
        
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                operation_id = f"{func.__module__}.{func.__qualname__}"
                self.reset_attempts(operation_id)
                
                attempt = 0
                start_time = time.time()
                
                while True:
                    try:
                        result = func(*args, **kwargs)
                        self.register_attempt(operation_id, True)
                        return result
                    except Exception as e:
                        self.register_attempt(operation_id, False, e)
                        
                        # Check if we've exceeded the timeout
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            logger.warning(f"Retry timeout after {elapsed:.2f}s ({attempt+1} attempts): {str(e)}")
                            raise TimeoutError(f"Retry operation timed out after {elapsed:.2f}s") from e
                        
                        # Check if we should retry
                        context = {
                            'max_attempts': max_attempts,
                            'delay_strategy': delay_strategy,
                            'base_delay': base_delay,
                            'operation_id': operation_id,
                            'elapsed': elapsed,
                            'timeout': timeout
                        }
                        
                        if self.should_retry(attempt, e, context) and attempt < max_attempts - 1:
                            # Calculate delay, ensuring we don't exceed timeout
                            delay = self.get_retry_delay(attempt, e, context)
                            remaining = timeout - elapsed
                            delay = min(delay, remaining)
                            
                            if delay <= 0:
                                logger.warning(f"Retry timeout imminent after {elapsed:.2f}s: {str(e)}")
                                raise TimeoutError(f"Retry operation timed out after {elapsed:.2f}s") from e
                            
                            logger.info(f"Retry {attempt+1}/{max_attempts} for {operation_id} in {delay:.2f}s ({elapsed:.2f}s elapsed of {timeout:.2f}s)")
                            time.sleep(delay)
                            attempt += 1
                        else:
                            logger.warning(f"Retry giving up after {attempt+1} attempts: {str(e)}")
                            raise
            
            return wrapper
        
        return decorator
    
    # Retry delay strategies
    def constant_delay(self, base_delay: float = 1.0) -> float:
        """Return a constant delay."""
        return self.add_jitter(base_delay)
    
    def linear_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Return a linearly increasing delay."""
        return self.add_jitter(base_delay * (attempt + 1))
    
    def exponential_delay(self, attempt: int, base_delay: float = 1.0, factor: float = 2.0) -> float:
        """Return an exponentially increasing delay."""
        return self.add_jitter(base_delay * (factor ** attempt))
    
    def fibonacci_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Return a delay based on the Fibonacci sequence."""
        if attempt <= 1:
            return self.add_jitter(base_delay * (attempt + 1))
        
        a, b = base_delay, base_delay * 2
        for _ in range(2, attempt + 1):
            a, b = b, a + b
        
        return self.add_jitter(b)
    
    def add_jitter(self, delay: float, factor: float = None) -> float:
        """Add random jitter to a delay to prevent thundering herd."""
        factor = factor or self._jitter_factor
        jitter = random.uniform(-factor * delay, factor * delay)
        return max(0, delay + jitter)
    
    # Retry condition strategies
    def retry_on_exception(self, exception_types: Union[Type[Exception], Tuple[Type[Exception], ...]]):
        """Create a predicate for retrying on specific exception types."""
        def predicate(e: Exception) -> bool:
            return isinstance(e, exception_types)
        return predicate
    
    def retry_on_result(self, result_predicate: Callable[[Any], bool]):
        """Create a decorator for retrying based on return value."""
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                operation_id = f"{func.__module__}.{func.__qualname__}"
                self.reset_attempts(operation_id)
                
                attempt = 0
                max_attempts = self._default_max_attempts
                
                while True:
                    result = func(*args, **kwargs)
                    
                    # Check if we should retry based on result
                    if not result_predicate(result):
                        self.register_attempt(operation_id, True)
                        return result
                    
                    # Register attempt and check if we should continue
                    self.register_attempt(operation_id, False)
                    
                    if attempt < max_attempts - 1:
                        delay = self.get_retry_delay(attempt, None, {'operation_id': operation_id})
                        logger.info(f"Retrying {operation_id} due to result condition in {delay:.2f}s")
                        time.sleep(delay)
                        attempt += 1
                    else:
                        logger.warning(f"Retry on result giving up after {attempt+1} attempts")
                        return result
            
            return wrapper
        
        return decorator
    
    def retry_on_http_status(self, status_codes: Union[int, List[int], Set[int]]):
        """Create a predicate for retrying on specific HTTP status codes."""
        if isinstance(status_codes, int):
            status_codes = {status_codes}
        elif isinstance(status_codes, list):
            status_codes = set(status_codes)
        
        def predicate(e: Exception, context: Dict[str, Any]) -> bool:
            # Try to extract status code from exception or context
            status_code = None
            
            # Check context first
            if 'status_code' in context:
                status_code = context['status_code']
            
            # Then try exception
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            
            return status_code in status_codes
        
        return predicate
    
    def retry_until_success(self, max_attempts: int = None):
        """Create a decorator that retries until success or max attempts reached."""
        max_attempts = max_attempts or self._default_max_attempts
        
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                operation_id = f"{func.__module__}.{func.__qualname__}"
                self.reset_attempts(operation_id)
                
                attempt = 0
                while True:
                    try:
                        result = func(*args, **kwargs)
                        self.register_attempt(operation_id, True)
                        logger.info(f"Success for {operation_id} after {attempt+1} attempts")
                        return result
                    except Exception as e:
                        self.register_attempt(operation_id, False, e)
                        
                        if attempt < max_attempts - 1:
                            delay = self.get_retry_delay(attempt, e, {'operation_id': operation_id})
                            logger.info(f"Retry {attempt+1}/{max_attempts} for {operation_id} in {delay:.2f}s: {str(e)}")
                            time.sleep(delay)
                            attempt += 1
                        else:
                            logger.warning(f"Retry giving up after {attempt+1} attempts: {str(e)}")
                            raise
            
            return wrapper
        
        return decorator
    
    def retry_with_fallback(self, fallback_func: Callable[..., R]):
        """Create a decorator that falls back to another function after retries fail."""
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            # Apply the retry decorator first
            retrying_func = self.retry()(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> R:
                try:
                    return retrying_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Retries failed, using fallback: {str(e)}")
                    return fallback_func(*args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def execute_with_retry(self, operation: Callable[..., R], *args, 
                          max_attempts: int = None,
                          delay_strategy: str = 'exponential',
                          base_delay: float = 1.0,
                          operation_id: str = None,
                          domain: str = None,
                          **kwargs) -> Tuple[R, RetryOutcome]:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: The function to execute
            *args: Arguments to pass to the function
            max_attempts: Maximum number of attempts
            delay_strategy: Strategy for calculating delay
            base_delay: Base delay in seconds
            operation_id: Unique identifier for this operation
            domain: Domain for budget tracking
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (result, outcome)
        """
        max_attempts = max_attempts or self._default_max_attempts
        if operation_id is None:
            operation_id = f"{operation.__module__}.{operation.__qualname__}"
        
        # Reset attempt counter
        self.reset_attempts(operation_id)
        
        attempt = 0
        while True:
            try:
                result = operation(*args, **kwargs)
                self.register_attempt(operation_id, True)
                return result, RetryOutcome.SUCCESS
            except Exception as e:
                self.register_attempt(operation_id, False, e)
                
                # Build context
                context = {
                    'max_attempts': max_attempts,
                    'delay_strategy': delay_strategy,
                    'base_delay': base_delay,
                    'operation_id': operation_id
                }
                
                if domain:
                    context['domain'] = domain
                    
                    # Check budget
                    if self.is_budget_exhausted(domain):
                        logger.warning(f"Retry budget exhausted for {domain}")
                        return None, RetryOutcome.FAILED_BUDGET_EXHAUSTED
                
                # Check if retryable
                if not self.should_retry(attempt, e, context):
                    logger.warning(f"Non-retryable error: {str(e)}")
                    return None, RetryOutcome.FAILED_NON_RETRYABLE
                
                if attempt < max_attempts - 1:
                    # Consume budget
                    if domain:
                        self.consume_retry_budget(domain)
                    
                    # Calculate delay
                    delay = self.get_retry_delay(attempt, e, context)
                    
                    logger.info(f"Retry {attempt+1}/{max_attempts} for {operation_id} in {delay:.2f}s: {str(e)}")
                    time.sleep(delay)
                    attempt += 1
                else:
                    logger.warning(f"Retry giving up after {attempt+1} attempts: {str(e)}")
                    return None, RetryOutcome.FAILED_RETRIES_EXHAUSTED