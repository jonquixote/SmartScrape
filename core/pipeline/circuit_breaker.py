"""
Circuit Breaker Module for Pipeline Architecture.

This module implements the Circuit Breaker pattern to prevent cascading failures
by temporarily disabling pipeline stages that are consistently failing.
"""

import time
import logging
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime, timedelta

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class CircuitState(Enum):
    """Enum representing the possible states of a circuit breaker."""
    CLOSED = "closed"      # Circuit is closed, operations are allowed
    OPEN = "open"          # Circuit is open, operations are blocked
    HALF_OPEN = "half-open"  # Circuit is testing if operations can resume


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
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 reset_timeout: int = 60,
                 half_open_max_calls: int = 1):
        """
        Initialize a new circuit breaker.
        
        Args:
            name (str): Name of this circuit breaker
            failure_threshold (int): Number of failures before opening circuit
            reset_timeout (int): Seconds to wait before testing if system has recovered
            half_open_max_calls (int): Number of test calls to allow in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_state_change = time.time()
        self._half_open_calls = 0
        
        self._lock = threading.RLock()
        self._listeners = set()
        
        self.logger = logging.getLogger(f"pipeline.circuit_breaker.{name}")
        self.logger.info(f"Circuit breaker '{name}' initialized in CLOSED state")
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        with self._lock:
            # Auto-transition from OPEN to HALF_OPEN after timeout
            if (self._state == CircuitState.OPEN and 
                self._last_failure_time is not None and
                time.time() - self._last_failure_time > self.reset_timeout):
                self._transition_to_half_open()
            
            return self._state
    
    def on_success(self) -> None:
        """Register a successful operation with the circuit breaker."""
        with self._lock:
            self._success_count += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                
                if self._half_open_calls >= self.half_open_max_calls:
                    self.close()
    
    def on_failure(self) -> None:
        """Register a failed operation with the circuit breaker."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self.open()
            elif self._state == CircuitState.HALF_OPEN:
                self.open()
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
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
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": total_calls,
                "failure_rate": failure_rate,
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
                "time_in_current_state": time.time() - self._last_state_change
            }
    
    def add_state_change_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """
        Add a listener for state change events.
        
        Args:
            listener: Callback function that accepts (name, old_state, new_state)
        """
        with self._lock:
            self._listeners.add(listener)
    
    def remove_state_change_listener(self, listener: Callable[[str, CircuitState, CircuitState], None]) -> None:
        """
        Remove a state change listener.
        
        Args:
            listener: The listener to remove
        """
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
    
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState) -> None:
        """
        Notify all listeners of a state change.
        
        Args:
            old_state: Previous circuit state
            new_state: New circuit state
        """
        for listener in list(self._listeners):
            try:
                listener(self.name, old_state, new_state)
            except Exception as e:
                self.logger.error(f"Error in circuit breaker listener: {str(e)}")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    This registry provides a centralized way to manage and monitor all
    circuit breakers in the system.
    """
    
    def __init__(self):
        """Initialize the circuit breaker registry."""
        self._circuit_breakers = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger("pipeline.circuit_breaker.registry")
    
    def get_or_create(self, 
                     name: str, 
                     failure_threshold: int = 5,
                     reset_timeout: int = 60,
                     half_open_max_calls: int = 1) -> CircuitBreaker:
        """
        Get an existing circuit breaker or create a new one.
        
        Args:
            name: Name of the circuit breaker
            failure_threshold: Number of failures before opening
            reset_timeout: Seconds to wait before half-open
            half_open_max_calls: Max calls in half-open state
            
        Returns:
            CircuitBreaker: The retrieved or created circuit breaker
        """
        with self._lock:
            if name in self._circuit_breakers:
                return self._circuit_breakers[name]
            
            # Create a new circuit breaker
            circuit_breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                half_open_max_calls=half_open_max_calls
            )
            
            self._circuit_breakers[name] = circuit_breaker
            return circuit_breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get an existing circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            Optional[CircuitBreaker]: The circuit breaker or None if not found
        """
        with self._lock:
            return self._circuit_breakers.get(name)
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._circuit_breakers.values():
                cb.reset()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all circuit breakers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping names to metrics
        """
        with self._lock:
            return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items()}


class CircuitBreakerStage(PipelineStage):
    """
    A pipeline stage that uses a circuit breaker to protect downstream stages.
    
    This stage wraps another stage and prevents its execution if the circuit
    is open due to previous failures.
    """
    
    def __init__(self, 
                 wrapped_stage: PipelineStage,
                 circuit_breaker: Optional[CircuitBreaker] = None,
                 failure_threshold: int = 5,
                 reset_timeout: int = 60,
                 circuit_name: Optional[str] = None):
        """
        Initialize a circuit breaker stage.
        
        Args:
            wrapped_stage: The stage to protect with circuit breaker
            circuit_breaker: Existing circuit breaker to use, or None to create
            failure_threshold: Number of failures before opening (if creating)
            reset_timeout: Seconds to wait before half-open (if creating)
            circuit_name: Name for the circuit breaker (if creating)
        """
        super().__init__()
        self.wrapped_stage = wrapped_stage
        self.name = f"circuit_breaker_{wrapped_stage.name}"
        
        # Use existing circuit breaker or create a new one
        if circuit_breaker is not None:
            self.circuit_breaker = circuit_breaker
        else:
            name = circuit_name or f"circuit_{wrapped_stage.name}"
            self.circuit_breaker = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout
            )
        
        self.logger = logging.getLogger(f"pipeline.circuit_breaker_stage.{self.name}")
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the stage, respecting the circuit breaker state.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        if not self.circuit_breaker.allow_request():
            self.logger.warning(
                f"Circuit breaker '{self.circuit_breaker.name}' is {self.circuit_breaker.state.value}, "
                f"skipping stage '{self.wrapped_stage.name}'"
            )
            context.add_error(
                self.name, 
                f"Circuit breaker '{self.circuit_breaker.name}' is open due to previous failures"
            )
            return False
        
        try:
            success = await self.wrapped_stage.process(context)
            
            if success:
                self.circuit_breaker.on_success()
            else:
                self.circuit_breaker.on_failure()
                
            return success
        except Exception as e:
            self.circuit_breaker.on_failure()
            raise
    
    async def initialize(self) -> None:
        """Initialize the wrapped stage."""
        await self.wrapped_stage.initialize()
    
    async def cleanup(self) -> None:
        """Clean up the wrapped stage."""
        await self.wrapped_stage.cleanup()
    
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate input for the wrapped stage.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Even if circuit is open, still validate input to keep context consistent
        return await self.wrapped_stage.validate_input(context)
    
    async def validate_output(self, context: PipelineContext) -> bool:
        """
        Validate output from the wrapped stage.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        return await self.wrapped_stage.validate_output(context)
    
    async def handle_error(self, context: PipelineContext, error: Exception) -> bool:
        """
        Handle an error that occurred during processing.
        
        Args:
            context: The pipeline context
            error: The exception that occurred
            
        Returns:
            bool: True if error was handled, False otherwise
        """
        # Record failure in circuit breaker
        self.circuit_breaker.on_failure()
        
        # Delegate error handling to wrapped stage
        return await self.wrapped_stage.handle_error(context, error)


# Global registry
_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """
    Get the global circuit breaker registry.
    
    Returns:
        CircuitBreakerRegistry: The global registry instance
    """
    return _registry