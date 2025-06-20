"""
Circuit Breaker Implementation

This module implements the Circuit Breaker pattern to prevent cascading failures
by temporarily blocking execution of operations that are likely to fail.

It provides a flexible and configurable circuit breaker implementation with
support for different failure detection mechanisms, fallbacks, and timeouts.
"""

import time
import logging
import threading
import functools
import asyncio
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, Type, TypeVar

from core.service_interface import BaseService

# For type hints
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])


class CircuitState(Enum):
    """Enum representing the possible states of a circuit breaker."""
    CLOSED = "closed"      # Circuit is closed, operations are allowed
    OPEN = "open"          # Circuit is open, operations are blocked
    HALF_OPEN = "half-open"  # Circuit is testing if operations can resume


class CircuitOpenError(Exception):
    """Exception raised when an operation is attempted while the circuit is open."""
    
    def __init__(self, circuit_name: str):
        """Initialize with circuit name."""
        self.circuit_name = circuit_name
        super().__init__(f"Circuit '{circuit_name}' is open due to previous failures")


# Alias for backward compatibility
OpenCircuitError = CircuitOpenError


class TimeoutError(Exception):
    """Exception raised when an operation times out."""
    
    def __init__(self, circuit_name: str, timeout: float):
        """Initialize with circuit name and timeout details."""
        self.circuit_name = circuit_name
        self.timeout = timeout
        super().__init__(f"Operation on circuit '{circuit_name}' timed out after {timeout} seconds")


class CircuitBreaker:
    """
    Circuit breaker implementation for managing failure scenarios.
    
    The circuit breaker prevents cascading failures by tracking the success/failure
    of operations and temporarily blocking execution when failure thresholds are met.
    
    Attributes:
        name (str): Name of this circuit breaker
        state (CircuitState): Current state of the circuit
        failure_threshold (int): Number of failures before opening the circuit
        reset_timeout (int): Seconds to wait before attempting to close the circuit
        half_open_max_calls (int): Maximum number of calls to allow in half-open state
        excluded_exceptions (set): Set of exception types that should not count as failures
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 reset_timeout: int = 60,
                 half_open_max_calls: int = 1,
                 reset_failure_count_timeout: int = 600,
                 excluded_exceptions: Optional[Set[Type[Exception]]] = None):
        """
        Initialize a new circuit breaker.
        
        Args:
            name (str): Name of this circuit breaker
            failure_threshold (int): Number of failures before opening circuit
            reset_timeout (int): Seconds to wait before testing if system has recovered
            half_open_max_calls (int): Number of test calls to allow in half-open state
            reset_failure_count_timeout (int): Seconds after which to reset failure count in closed state
            excluded_exceptions (set): Set of exception types that should not count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset_failure_count_timeout = reset_failure_count_timeout
        self.excluded_exceptions = excluded_exceptions or set()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None
        self._last_state_change = time.time()
        self._half_open_calls = 0
        
        self._lock = threading.RLock()
        self._state_change_listeners: Set[Callable[[str, CircuitState, CircuitState], None]] = set()
        self._success_listeners: Set[Callable[[str], None]] = set()
        self._failure_listeners: Set[Callable[[str, Exception], None]] = set()
        self._execution_listeners: Set[Callable[[str, bool], None]] = set()
        
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        self.logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        with self._lock:
            current_time = time.time()
            
            # Auto-transition from OPEN to HALF_OPEN after timeout
            if (self._state == CircuitState.OPEN and 
                self._last_failure_time is not None and
                current_time - self._last_failure_time > self.reset_timeout):
                self._transition_to_half_open()
            
            # In CLOSED state, reset failure count after timeout
            elif (self._state == CircuitState.CLOSED and 
                  self._failure_count > 0 and
                  self._last_failure_time is not None and
                  current_time - self._last_failure_time > self.reset_failure_count_timeout):
                self._failure_count = 0
                self.logger.debug(f"Circuit '{self.name}' reset failure count due to timeout")
                
            return self._state
    
    def get_state(self) -> CircuitState:
        """
        Get the current state of the circuit breaker.
        
        Returns:
            CircuitState: The current state
        """
        return self.state
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The return value from the function
            
        Raises:
            CircuitOpenError: If the circuit is open
            Any exception raised by the function
        """
        if not self.allow_request():
            self._notify_execution(False)
            raise CircuitOpenError(self.name)
        
        try:
            result = func(*args, **kwargs)
            self.success()
            self._notify_execution(True)
            return result
        except Exception as e:
            if not any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions):
                self.failure(e)
            self._notify_execution(False)
            raise
    
    async def execute_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Execute an async function with circuit breaker protection.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The return value from the function
            
        Raises:
            CircuitOpenError: If the circuit is open
            Any exception raised by the function
        """
        if not self.allow_request():
            self._notify_execution(False)
            raise CircuitOpenError(self.name)
        
        try:
            result = await func(*args, **kwargs)
            self.success()
            self._notify_execution(True)
            return result
        except Exception as e:
            if not any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions):
                self.failure(e)
            self._notify_execution(False)
            raise
    
    def execute_with_fallback(self, 
                             func: Callable[..., T], 
                             fallback: Callable[..., T], 
                             *args: Any, 
                             **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker protection and fallback.
        
        Args:
            func: The function to execute
            fallback: Fallback function to call if the circuit is open or the primary function fails
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The return value from the function or fallback
        """
        try:
            return self.execute(func, *args, **kwargs)
        except Exception as e:
            self.logger.info(f"Circuit '{self.name}' using fallback due to: {str(e)}")
            return fallback(*args, **kwargs)
    
    async def execute_with_fallback_async(self, 
                                        func: Callable[..., Any], 
                                        fallback: Callable[..., Any], 
                                        *args: Any, 
                                        **kwargs: Any) -> Any:
        """
        Execute an async function with circuit breaker protection and fallback.
        
        Args:
            func: The async function to execute
            fallback: Async fallback function to call if the circuit is open or the primary function fails
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The return value from the function or fallback
        """
        try:
            return await self.execute_async(func, *args, **kwargs)
        except Exception as e:
            self.logger.info(f"Circuit '{self.name}' using async fallback due to: {str(e)}")
            return await fallback(*args, **kwargs)
    
    async def execute_with_timeout(self, 
                                 func: Callable[..., Any], 
                                 timeout: float, 
                                 *args: Any, 
                                 **kwargs: Any) -> Any:
        """
        Execute an async function with circuit breaker protection and timeout.
        
        Args:
            func: The async function to execute
            timeout: Maximum time in seconds to wait for the function to complete
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The return value from the function
            
        Raises:
            CircuitOpenError: If the circuit is open
            TimeoutError: If the function execution exceeds the timeout
            Any exception raised by the function
        """
        if not self.allow_request():
            self._notify_execution(False)
            raise CircuitOpenError(self.name)
        
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            self.success()
            self._notify_execution(True)
            return result
        except asyncio.TimeoutError:
            error = TimeoutError(self.name, timeout)
            self.failure(error)
            self._notify_execution(False)
            raise error
        except Exception as e:
            if not any(isinstance(e, exc_type) for exc_type in self.excluded_exceptions):
                self.failure(e)
            self._notify_execution(False)
            raise
    
    def success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._success_count += 1
            self._last_success_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                
                if self._half_open_calls >= self.half_open_max_calls:
                    self.close()
        
        # Notify success listeners outside the lock to avoid deadlocks
        self._notify_success()
    
    def failure(self, exception: Optional[Exception] = None) -> None:
        """
        Record a failed operation.
        
        Args:
            exception: The exception that caused the failure
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self.open()
            elif self._state == CircuitState.HALF_OPEN:
                self.open()
        
        # Notify failure listeners outside the lock to avoid deadlocks
        self._notify_failure(exception or Exception("Unknown failure"))
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._last_success_time = None
            self._half_open_calls = 0
            
            if old_state != CircuitState.CLOSED:
                self._last_state_change = time.time()
                self._notify_state_change(old_state, CircuitState.CLOSED)
                
            self.logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")
    
    def open(self) -> None:
        """Manually open the circuit."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                old_state = self._state
                self._state = CircuitState.OPEN
                self._last_state_change = time.time()
                self._half_open_calls = 0
                
                self.logger.warning(f"Circuit breaker '{self.name}' opened after {self._failure_count} failures")
                self._notify_state_change(old_state, CircuitState.OPEN)
    
    def close(self) -> None:
        """Manually close the circuit."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                old_state = self._state
                self._state = CircuitState.CLOSED
                self._last_state_change = time.time()
                self._failure_count = 0
                self._half_open_calls = 0
                
                self.logger.info(f"Circuit breaker '{self.name}' closed")
                self._notify_state_change(old_state, CircuitState.CLOSED)
    
    def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN state to test if system has recovered."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._last_state_change = time.time()
        self._half_open_calls = 0
        
        self.logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN state")
        self._notify_state_change(old_state, CircuitState.HALF_OPEN)
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed based on circuit state.
        
        Returns:
            bool: True if the request should be allowed, False otherwise
        """
        with self._lock:
            current_state = self.state  # This will auto-transition if needed
            
            if current_state == CircuitState.CLOSED:
                return True
            elif current_state == CircuitState.OPEN:
                return False
            elif current_state == CircuitState.HALF_OPEN:
                # Only allow a limited number of test requests in half-open state
                return self._half_open_calls < self.half_open_max_calls
            
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about this circuit breaker.
        
        Returns:
            Dict[str, Any]: Dictionary of circuit breaker metrics
        """
        with self._lock:
            total_calls = self._success_count + self._failure_count
            failure_rate = (self._failure_count / total_calls * 100) if total_calls > 0 else 0
            
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": total_calls,
                "failure_rate": failure_rate,
                "last_failure_time": self._last_failure_time,
                "last_success_time": self._last_success_time, 
                "last_state_change": self._last_state_change,
                "time_in_current_state": time.time() - self._last_state_change
            }
    
    # Event listener management
    
    def add_state_change_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """
        Add a listener for state change events.
        
        Args:
            listener: Callback function that accepts (name, old_state, new_state)
        """
        with self._lock:
            self._state_change_listeners.add(listener)
    
    def remove_state_change_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """
        Remove a state change listener.
        
        Args:
            listener: The listener to remove
        """
        with self._lock:
            if listener in self._state_change_listeners:
                self._state_change_listeners.remove(listener)
    
    def add_success_listener(self, listener: Callable[[str], None]) -> None:
        """
        Add a listener for success events.
        
        Args:
            listener: Callback function that accepts (name)
        """
        with self._lock:
            self._success_listeners.add(listener)
    
    def remove_success_listener(self, listener: Callable[[str], None]) -> None:
        """
        Remove a success listener.
        
        Args:
            listener: The listener to remove
        """
        with self._lock:
            if listener in self._success_listeners:
                self._success_listeners.remove(listener)
    
    def add_failure_listener(self, listener: Callable[[str, Exception], None]) -> None:
        """
        Add a listener for failure events.
        
        Args:
            listener: Callback function that accepts (name, exception)
        """
        with self._lock:
            self._failure_listeners.add(listener)
    
    def remove_failure_listener(self, listener: Callable[[str, Exception], None]) -> None:
        """
        Remove a failure listener.
        
        Args:
            listener: The listener to remove
        """
        with self._lock:
            if listener in self._failure_listeners:
                self._failure_listeners.remove(listener)
    
    def add_execution_listener(self, listener: Callable[[str, bool], None]) -> None:
        """
        Add a listener for execution events.
        
        Args:
            listener: Callback function that accepts (name, success)
        """
        with self._lock:
            self._execution_listeners.add(listener)
    
    def remove_execution_listener(self, listener: Callable[[str, bool], None]) -> None:
        """
        Remove an execution listener.
        
        Args:
            listener: The listener to remove
        """
        with self._lock:
            if listener in self._execution_listeners:
                self._execution_listeners.remove(listener)
    
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """
        Notify all state change listeners.
        
        Args:
            old_state: Previous circuit state
            new_state: New circuit state
        """
        # Make a copy to avoid issues if listeners modify the set
        listeners = list(self._state_change_listeners)
        
        for listener in listeners:
            try:
                listener(self.name, old_state, new_state)
            except Exception as e:
                self.logger.error(f"Error in circuit breaker state change listener: {str(e)}")
    
    def _notify_success(self) -> None:
        """Notify all success listeners."""
        # Make a copy to avoid issues if listeners modify the set
        listeners = list(self._success_listeners)
        
        for listener in listeners:
            try:
                listener(self.name)
            except Exception as e:
                self.logger.error(f"Error in circuit breaker success listener: {str(e)}")
    
    def _notify_failure(self, exception: Exception) -> None:
        """
        Notify all failure listeners.
        
        Args:
            exception: The exception that caused the failure
        """
        # Make a copy to avoid issues if listeners modify the set
        listeners = list(self._failure_listeners)
        
        for listener in listeners:
            try:
                listener(self.name, exception)
            except Exception as e:
                self.logger.error(f"Error in circuit breaker failure listener: {str(e)}")
    
    def _notify_execution(self, success: bool) -> None:
        """
        Notify all execution listeners.
        
        Args:
            success: Whether the execution was successful
        """
        # Make a copy to avoid issues if listeners modify the set
        listeners = list(self._execution_listeners)
        
        for listener in listeners:
            try:
                listener(self.name, success)
            except Exception as e:
                self.logger.error(f"Error in circuit breaker execution listener: {str(e)}")


class CircuitBreakerManager(BaseService):
    """
    Manager for circuit breakers.
    
    This class provides a centralized way to manage multiple circuit breakers
    in an application.
    """
    
    def __init__(self):
        """Initialize the circuit breaker manager."""
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self._default_config = {
            "failure_threshold": 5,
            "reset_timeout": 60,
            "half_open_max_calls": 1,
            "reset_failure_count_timeout": 600
        }
        self.logger = logging.getLogger("circuit_breaker.manager")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the circuit breaker manager with configuration.
        
        Args:
            config: Configuration dictionary with default settings and predefined circuits
        """
        if config:
            # Update default configuration
            if "default_settings" in config:
                self._default_config.update(config["default_settings"])
            
            # Create predefined circuit breakers
            for name, settings in config.get("circuit_breakers", {}).items():
                self.get_circuit_breaker(name, settings)
                
        self.logger.info("Circuit breaker manager initialized")
    
    def shutdown(self) -> None:
        """Clean up resources."""
        with self._lock:
            self._circuit_breakers.clear()
            
        self.logger.info("Circuit breaker manager shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "circuit_breaker_manager"
    
    def get_circuit_breaker(self, name: str, config: Optional[Dict[str, Any]] = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker with the given name.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker (optional)
            
        Returns:
            CircuitBreaker: The retrieved or created circuit breaker
        """
        with self._lock:
            if name in self._circuit_breakers:
                return self._circuit_breakers[name]
            
            # Merge configuration with defaults
            merged_config = dict(self._default_config)
            if config:
                merged_config.update(config)
                
            # Create circuit breaker
            circuit_breaker = CircuitBreaker(
                name=name,
                failure_threshold=merged_config.get("failure_threshold", 5),
                reset_timeout=merged_config.get("reset_timeout", 60),
                half_open_max_calls=merged_config.get("half_open_max_calls", 1),
                reset_failure_count_timeout=merged_config.get("reset_failure_count_timeout", 600),
                excluded_exceptions=merged_config.get("excluded_exceptions", set())
            )
            
            self._circuit_breakers[name] = circuit_breaker
            self.logger.info(f"Created circuit breaker '{name}'")
            
            return circuit_breaker
    
    def register_circuit_breaker(self, name: str, circuit_breaker: CircuitBreaker) -> None:
        """
        Register an existing circuit breaker.
        
        Args:
            name: Name to register the circuit breaker under
            circuit_breaker: The circuit breaker to register
        """
        with self._lock:
            self._circuit_breakers[name] = circuit_breaker
            self.logger.info(f"Registered circuit breaker '{name}'")
    
    def remove_circuit_breaker(self, name: str) -> None:
        """
        Remove a circuit breaker.
        
        Args:
            name: Name of the circuit breaker to remove
        """
        with self._lock:
            if name in self._circuit_breakers:
                del self._circuit_breakers[name]
                self.logger.info(f"Removed circuit breaker '{name}'")
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for circuit_breaker in self._circuit_breakers.values():
                circuit_breaker.reset()
                
        self.logger.info("Reset all circuit breakers")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all circuit breakers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping circuit breaker names to metrics
        """
        with self._lock:
            return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items()}
    
    def get_unhealthy_circuits(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all circuit breakers that are in OPEN or HALF_OPEN state.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping unhealthy circuit breaker names to metrics
        """
        with self._lock:
            return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items() 
                   if cb.state != CircuitState.CLOSED}


# Circuit breaker decorator functions

def circuit_protected(circuit_name: str, manager: Optional[CircuitBreakerManager] = None, **config):
    """
    Decorator to protect a function with a circuit breaker.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        manager: CircuitBreakerManager instance (optional, uses global instance if not provided)
        **config: Additional configuration for the circuit breaker
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create manager
            nonlocal manager
            if manager is None:
                manager = get_manager()
                
            # Get or create circuit breaker
            circuit_breaker = manager.get_circuit_breaker(circuit_name, config)
            
            # Execute with circuit breaker protection
            return circuit_breaker.execute(func, *args, **kwargs)
            
        return wrapper  # type: ignore
    return decorator


def circuit_protected_async(circuit_name: str, manager: Optional[CircuitBreakerManager] = None, **config):
    """
    Decorator to protect an async function with a circuit breaker.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        manager: CircuitBreakerManager instance (optional, uses global instance if not provided)
        **config: Additional configuration for the circuit breaker
        
    Returns:
        Async decorator function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create manager
            nonlocal manager
            if manager is None:
                manager = get_manager()
                
            # Get or create circuit breaker
            circuit_breaker = manager.get_circuit_breaker(circuit_name, config)
            
            # Execute with circuit breaker protection
            return await circuit_breaker.execute_async(func, *args, **kwargs)
            
        return wrapper  # type: ignore
    return decorator


def fallback_protected(circuit_name: str, fallback_func: Callable, manager: Optional[CircuitBreakerManager] = None, **config):
    """
    Decorator to protect a function with a circuit breaker and fallback.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        fallback_func: Fallback function to call if the primary function fails
        manager: CircuitBreakerManager instance (optional, uses global instance if not provided)
        **config: Additional configuration for the circuit breaker
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create manager
            nonlocal manager
            if manager is None:
                manager = get_manager()
                
            # Get or create circuit breaker
            circuit_breaker = manager.get_circuit_breaker(circuit_name, config)
            
            # Execute with circuit breaker protection and fallback
            return circuit_breaker.execute_with_fallback(func, fallback_func, *args, **kwargs)
            
        return wrapper  # type: ignore
    return decorator


def fallback_protected_async(circuit_name: str, fallback_func: Callable, manager: Optional[CircuitBreakerManager] = None, **config):
    """
    Decorator to protect an async function with a circuit breaker and fallback.
    
    Args:
        circuit_name: Name of the circuit breaker to use
        fallback_func: Async fallback function to call if the primary function fails
        manager: CircuitBreakerManager instance (optional, uses global instance if not provided)
        **config: Additional configuration for the circuit breaker
        
    Returns:
        Async decorator function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create manager
            nonlocal manager
            if manager is None:
                manager = get_manager()
                
            # Get or create circuit breaker
            circuit_breaker = manager.get_circuit_breaker(circuit_name, config)
            
            # Execute with circuit breaker protection and fallback
            return await circuit_breaker.execute_with_fallback_async(func, fallback_func, *args, **kwargs)
            
        return wrapper  # type: ignore
    return decorator


def timeout_protected(timeout: float, circuit_name: str, manager: Optional[CircuitBreakerManager] = None, **config):
    """
    Decorator to protect an async function with a circuit breaker and timeout.
    
    Args:
        timeout: Maximum time in seconds to wait for the function to complete
        circuit_name: Name of the circuit breaker to use
        manager: CircuitBreakerManager instance (optional, uses global instance if not provided)
        **config: Additional configuration for the circuit breaker
        
    Returns:
        Async decorator function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get or create manager
            nonlocal manager
            if manager is None:
                manager = get_manager()
                
            # Get or create circuit breaker
            circuit_breaker = manager.get_circuit_breaker(circuit_name, config)
            
            # Execute with circuit breaker protection and timeout
            return await circuit_breaker.execute_with_timeout(func, timeout, *args, **kwargs)
            
        return wrapper  # type: ignore
    return decorator


# Global manager instance
_manager = CircuitBreakerManager()

def get_manager() -> CircuitBreakerManager:
    """
    Get the global circuit breaker manager.
    
    Returns:
        CircuitBreakerManager: The global manager instance
    """
    return _manager

# Compatibility with pipeline-specific circuit breaker
def get_pipeline_circuit_breaker_registry():
    """
    Get the global circuit breaker registry for backward compatibility.
    
    Returns:
        The global registry from the pipeline module
    """
    from core.pipeline.circuit_breaker import get_circuit_breaker_registry
    return get_circuit_breaker_registry()