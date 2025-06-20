"""
Circuit Breaker Stage Module.

This module provides a CircuitBreakerStage that implements the Circuit Breaker pattern
to prevent cascading failures in a distributed system by limiting calls to failing components.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.circuit_breaker import CircuitBreaker
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class CircuitBreakerStage(PipelineStage):
    """
    A pipeline stage that implements the Circuit Breaker pattern.
    
    This stage wraps another stage with a circuit breaker that tracks failures
    and prevents calls to failing components to avoid cascading failures.
    
    Features:
    - Configurable failure threshold and reset timeout
    - Half-open state for testing recovery
    - Global circuit breaker registry (optional)
    - Fallback mechanism when circuit is open
    - Detailed metrics for monitoring circuit state
    - Integration with PipelineContext for state tracking
    
    Configuration:
    - wrapped_stage: The stage to protect with the circuit breaker
    - failure_threshold: Number of failures before opening the circuit
    - reset_timeout: Seconds to wait before attempting to half-open the circuit
    - use_registry: Whether to use the global circuit breaker registry
    - half_open_max_calls: Number of test calls to allow in half-open state
    - fallback_enabled: Whether to use fallback behavior when circuit is open
    - fallback_data: Data to use for fallback when circuit is open
    """
    
    def __init__(self, 
                 wrapped_stage: PipelineStage,
                 name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the circuit breaker stage.
        
        Args:
            wrapped_stage: The stage to protect with the circuit breaker
            name: Optional name for the stage (defaults to wrapped stage name)
            config: Optional configuration dictionary
        """
        stage_name = name or f"circuit_breaker_{wrapped_stage.name}"
        super().__init__(stage_name, config or {})
        
        self.wrapped_stage = wrapped_stage
        
        # Extract configuration
        self.failure_threshold = self.config.get("failure_threshold", 5)
        self.reset_timeout = self.config.get("reset_timeout", 60)
        self.use_registry = self.config.get("use_registry", True)
        self.half_open_max_calls = self.config.get("half_open_max_calls", 1)
        self.fallback_enabled = self.config.get("fallback_enabled", False)
        self.fallback_data = self.config.get("fallback_data", {})
        
        # Create the circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=self.name,
            failure_threshold=self.failure_threshold,
            reset_timeout=self.reset_timeout,
            use_registry=self.use_registry,
            half_open_max_calls=self.half_open_max_calls
        )
        
        self.logger = logging.getLogger(f"pipeline.circuit_breaker.{self.name}")
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the stage, respecting circuit breaker state.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        # Check if circuit is closed (or half-open and allowing test requests)
        if not self.circuit_breaker.allow_request():
            # Circuit is open, handle according to configuration
            self.logger.warning(f"Circuit {self.name} is open, request blocked")
            
            # Add circuit state to context
            context.set("circuit_breaker_open", True)
            context.set("circuit_breaker_name", self.name)
            context.set("circuit_breaker_failure_count", self.circuit_breaker.failure_count)
            context.set("circuit_breaker_last_failure_time", self.circuit_breaker.last_failure_time)
            
            # Check if fallback is enabled
            if self.fallback_enabled and self.fallback_data:
                self.logger.info(f"Using fallback data for circuit {self.name}")
                
                # Apply fallback data to context
                for key, value in self.fallback_data.items():
                    context.set(key, value)
                
                # Mark fallback as used
                context.set("circuit_breaker_fallback_used", True)
                return True
            
            # No fallback, return failure
            error_msg = f"Circuit breaker {self.name} is open"
            context.add_error(self.name, error_msg)
            return False
        
        # Circuit is closed or half-open, validate input for the wrapped stage
        if not await self.wrapped_stage.validate_input(context):
            self.logger.warning(f"Input validation failed for wrapped stage {self.wrapped_stage.name}")
            context.add_error(self.name, f"Input validation failed for wrapped stage {self.wrapped_stage.name}")
            return False
        
        # Process with circuit breaker
        try:
            # Execute the wrapped stage
            result = await self.wrapped_stage.process(context)
            
            # Record success or failure based on result
            if result:
                # Success, reset failure count
                self.circuit_breaker.record_success()
            else:
                # Failure, increment failure count
                self.circuit_breaker.record_failure()
                
                # Get error details from context
                errors = context.get_all_errors()
                error_msg = f"Wrapped stage failed: {errors}"
                self.logger.warning(error_msg)
                
                # Add circuit breaker state to context
                context.set("circuit_breaker_failure_count", self.circuit_breaker.failure_count)
                context.set("circuit_breaker_state", self.circuit_breaker.state)
            
            return result
        
        except Exception as e:
            # Exception during processing, record failure
            self.circuit_breaker.record_failure()
            
            # Add circuit breaker state to context
            context.set("circuit_breaker_failure_count", self.circuit_breaker.failure_count)
            context.set("circuit_breaker_state", self.circuit_breaker.state)
            
            # Log the error
            error_msg = f"Error in wrapped stage: {str(e)}"
            self.logger.error(error_msg)
            context.add_error(self.name, error_msg)
            
            # Let the wrapped stage handle the error
            return await self.wrapped_stage.handle_error(context, e)
    
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
        return await self.wrapped_stage.handle_error(context, error)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about this stage.
        
        Returns:
            Dict[str, Any]: Dictionary of stage metrics
        """
        metrics = super().get_metadata()
        
        # Add circuit breaker specific metrics
        circuit_metrics = {
            "state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "failure_threshold": self.circuit_breaker.failure_threshold,
            "reset_timeout": self.circuit_breaker.reset_timeout,
            "last_failure_time": self.circuit_breaker.last_failure_time,
            "last_success_time": self.circuit_breaker.last_success_time,
            "open_time": self.circuit_breaker.open_time,
            "half_open_call_count": self.circuit_breaker.half_open_call_count,
            "half_open_max_calls": self.circuit_breaker.half_open_max_calls
        }
        
        metrics["circuit_breaker"] = circuit_metrics
        
        # Try to get wrapped stage metrics if available
        if hasattr(self.wrapped_stage, "get_metrics"):
            metrics["wrapped_stage"] = self.wrapped_stage.get_metrics()
            
        return metrics
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for circuit breaker stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        return {
            "type": "object",
            "properties": {
                "failure_threshold": {"type": "integer", "minimum": 1},
                "reset_timeout": {"type": "integer", "minimum": 1},
                "use_registry": {"type": "boolean"},
                "half_open_max_calls": {"type": "integer", "minimum": 1},
                "fallback_enabled": {"type": "boolean"},
                "fallback_data": {"type": "object"}
            }
        }