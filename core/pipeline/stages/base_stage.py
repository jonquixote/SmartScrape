"""
Enhanced base pipeline stage with integrated resource management and error handling.

This module provides a BaseStage class that integrates with the resource management
and error handling components created in Batch 5, including RetryManager, CircuitBreaker,
ErrorClassifier, and various monitoring capabilities.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Type

from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage
from core.service_registry import ServiceRegistry
from core.error_classifier import ErrorClassifier, ErrorCategory, ErrorSeverity
from core.retry_manager import RetryManager
from core.circuit_breaker import CircuitBreaker, OpenCircuitError

class BaseStage(PipelineStage):
    """
    Enhanced base stage with integrated error handling and resource management.
    
    This class extends the basic PipelineStage with additional functionality for:
    - Error classification and handling
    - Automatic retries for transient errors
    - Circuit breaker protection for external services
    - Resource usage tracking and monitoring
    - Standardized error response formatting
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the base stage with enhanced capabilities.
        
        Args:
            name: Optional stage name (defaults to class name if not provided)
            config: Optional stage configuration
        """
        super().__init__(name, config)
        
        # Set up logging
        self._logger = logging.getLogger(f"pipeline.stage.{self.name}")
        
        # Service access cache
        self._services = {}
        
        # Performance metrics
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._retry_count = 0
        self._start_time = 0
        self._end_time = 0
        self._last_duration = 0
        
        # Resource tracking
        self._resource_metrics = {
            "cpu_time": 0,
            "memory_usage": 0,
            "network_requests": 0,
            "io_operations": 0
        }
        
        # Error tracking
        self._last_error = None
        self._error_history = []
        
        # Retry configuration
        self._retry_enabled = self.config.get("enable_retry", True)
        self._max_retries = self.config.get("max_retries", 3)
        self._retry_backoff_factor = self.config.get("retry_backoff_factor", 2.0)
        self._retry_jitter = self.config.get("retry_jitter", 0.1)
        self._retryable_errors = self.config.get("retryable_errors", ["network", "rate_limit"])
        
        # Circuit breaker configuration
        self._circuit_breaker_enabled = self.config.get("enable_circuit_breaker", True)
        self._circuit_breaker_name = self.config.get("circuit_breaker_name")
        
        # Resource limits
        self._timeout = self.config.get("timeout", 60)
        self._memory_limit = self.config.get("memory_limit", 0)  # 0 means no limit
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the stage with integrated error handling and resource tracking.
        
        This method wraps the concrete _process implementation with retry logic,
        error handling, circuit breaker protection, and resource tracking.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Track metrics
        self._execution_count += 1
        context.start_stage(self.name)
        self._start_time = time.time()
        
        # Apply circuit breaker if enabled
        if self._circuit_breaker_enabled:
            circuit = self._get_circuit_breaker(
                self._circuit_breaker_name or f"{self.name}_breaker"
            )
            
            if circuit and not circuit.allow_request():
                self._logger.warning(f"Circuit breaker {circuit.name} is open, skipping stage")
                context.add_error(self.name, f"Circuit breaker is open: {circuit.name}")
                
                # Record metrics for circuit breaker trip
                context.end_stage(False)
                self._failure_count += 1
                self._end_time = time.time()
                self._last_duration = self._end_time - self._start_time
                self._register_metrics({"circuit_breaker_trips": 1})
                
                return False
        
        # Set up timeout if configured
        if self._timeout > 0:
            try:
                return await asyncio.wait_for(
                    self._process_with_retry(context),
                    timeout=self._timeout
                )
            except asyncio.TimeoutError:
                self._logger.error(f"Stage timed out after {self._timeout}s")
                context.add_error(self.name, f"Stage execution timed out after {self._timeout}s")
                
                # Record metrics for timeout
                context.end_stage(False)
                self._failure_count += 1
                self._end_time = time.time()
                self._last_duration = self._end_time - self._start_time
                self._register_metrics({"timeouts": 1})
                
                return False
        else:
            # No timeout
            return await self._process_with_retry(context)
    
    async def _process_with_retry(self, context: PipelineContext) -> bool:
        """
        Process the stage with retry logic for transient failures.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        retry_attempt = 0
        retry_delay = 1.0
        
        while True:
            try:
                # Attempt to process
                success = await self._process(context)
                
                # Record success and metrics
                if success:
                    self._success_count += 1
                    
                    # Report success to circuit breaker if enabled
                    if self._circuit_breaker_enabled:
                        circuit = self._get_circuit_breaker(
                            self._circuit_breaker_name or f"{self.name}_breaker"
                        )
                        if circuit:
                            circuit.record_success()
                else:
                    self._failure_count += 1
                
                # Calculate duration and record metrics
                self._end_time = time.time()
                self._last_duration = self._end_time - self._start_time
                self._register_metrics({})
                
                # End the stage tracking
                context.end_stage(success)
                
                return success
                
            except OpenCircuitError as e:
                # Circuit breaker is open - don't retry
                self._logger.warning(f"Circuit breaker open: {str(e)}")
                context.add_error(self.name, f"Circuit breaker open: {str(e)}")
                
                self._record_failure(context, e)
                return False
                
            except Exception as e:
                # Handle the error
                retry_attempt += 1
                
                # Classify the error
                error_info = self._classify_error(e, context)
                
                # Record the error for analysis
                self._record_error(error_info)
                
                # Check if we should retry
                if self._retry_enabled and retry_attempt <= self._max_retries:
                    if self._should_retry(error_info):
                        # Apply backoff with jitter
                        jitter = 1.0 + (self._retry_jitter * (2 * (time.time() % 1) - 1))
                        delay = retry_delay * jitter
                        
                        self._logger.info(
                            f"Retrying ({retry_attempt}/{self._max_retries}) after {delay:.2f}s due to: {str(e)}"
                        )
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        
                        # Increase backoff for next attempt
                        retry_delay *= self._retry_backoff_factor
                        
                        # Record retry metrics
                        self._retry_count += 1
                        self._register_metrics({"retries": self._retry_count})
                        
                        # Continue to next attempt
                        continue
                
                # No more retries or not retryable - report failure
                self._record_failure(context, e)
                return False
    
    async def _process(self, context: PipelineContext) -> bool:
        """
        Concrete implementation of stage processing logic.
        
        This method should be overridden by subclasses to provide the actual
        processing implementation.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing was successful, False otherwise
            
        Raises:
            NotImplementedError: If the method is not overridden
        """
        raise NotImplementedError("Subclasses must implement _process()")
    
    def _get_service(self, service_name: str) -> Any:
        """
        Get a service from the service registry.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            The requested service
            
        Raises:
            Exception: If the service is not found
        """
        # Check cache first
        if service_name in self._services:
            return self._services[service_name]
        
        # Get the service from the registry
        service = ServiceRegistry().get_service(service_name)
        
        # Cache it for future use
        if service:
            self._services[service_name] = service
        
        return service
    
    def _classify_error(self, error: Exception, context: Optional[PipelineContext] = None) -> Dict[str, Any]:
        """
        Classify an error using the ErrorClassifier service if available.
        
        Args:
            error: The exception to classify
            context: Optional pipeline context with additional metadata
            
        Returns:
            Dict[str, Any]: Error classification information
        """
        # Basic classification info
        error_info = {
            "error": error,
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
            "category": "unknown",
            "severity": "persistent",
            "is_retryable": False
        }
        
        # Try to use ErrorClassifier service
        try:
            error_classifier = self._get_service("error_classifier")
            
            # Prepare metadata from context
            metadata = {}
            if context:
                # Add relevant context data that might help with classification
                metadata = {
                    "stage_name": self.name,
                    "pipeline_id": context.get("pipeline_id"),
                    "request_id": context.get("request_id")
                }
                
                # Add additional context metadata if available
                if context.get("url"):
                    metadata["url"] = context.get("url")
                if context.get("domain"):
                    metadata["domain"] = context.get("domain")
            
            # Classify the error
            classification = error_classifier.classify_exception(error, metadata)
            
            # Update error info with classification
            error_info.update({
                "category": classification.get("category", "unknown"),
                "severity": classification.get("severity", "persistent"),
                "is_retryable": classification.get("is_retryable", False),
                "suggested_actions": classification.get("suggested_actions", [])
            })
            
        except Exception as e:
            # Fallback to basic classification
            self._logger.warning(f"Error using ErrorClassifier: {str(e)}")
            
            # Simple classification based on exception type
            if "timeout" in str(error).lower():
                error_info["category"] = "network"
                error_info["severity"] = "transient"
                error_info["is_retryable"] = True
            elif "connection" in str(error).lower():
                error_info["category"] = "network"
                error_info["severity"] = "transient"
                error_info["is_retryable"] = True
            elif "429" in str(error) or "too many requests" in str(error).lower():
                error_info["category"] = "rate_limit"
                error_info["severity"] = "transient"
                error_info["is_retryable"] = True
        
        return error_info
    
    def _should_retry(self, error_info: Dict[str, Any]) -> bool:
        """
        Determine if an operation should be retried based on the error.
        
        Args:
            error_info: Error classification information
            
        Returns:
            bool: True if the operation should be retried, False otherwise
        """
        # Check if error is marked as retryable by classifier
        if error_info.get("is_retryable", False):
            return True
        
        # Check against configured retryable error categories
        category = error_info.get("category", "unknown")
        if category in self._retryable_errors:
            return True
        
        # Check severity
        severity = error_info.get("severity", "persistent")
        if severity == "transient":
            return True
        
        return False
    
    def _get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker instance for a specific service.
        
        Args:
            name: Name of the circuit breaker
            
        Returns:
            Optional[CircuitBreaker]: The circuit breaker instance or None if unavailable
        """
        try:
            circuit_breaker_manager = self._get_service("circuit_breaker_manager")
            return circuit_breaker_manager.get_circuit_breaker(name)
        except Exception as e:
            self._logger.warning(f"Error getting circuit breaker: {str(e)}")
            return None
    
    def _record_error(self, error_info: Dict[str, Any]) -> None:
        """
        Record an error for analysis and monitoring.
        
        Args:
            error_info: Error classification information
        """
        self._last_error = error_info
        
        # Keep a limited history of errors
        max_history = 10
        self._error_history.append(error_info)
        if len(self._error_history) > max_history:
            self._error_history.pop(0)
    
    def _record_failure(self, context: PipelineContext, error: Exception) -> None:
        """
        Record a stage failure in the context and metrics.
        
        Args:
            context: The pipeline context
            error: The exception that caused the failure
        """
        # Record the failure in metrics
        self._failure_count += 1
        
        # End the stage tracking with failure
        context.end_stage(False)
        
        # Add the error to context
        context.add_error(self.name, str(error))
        
        # Calculate duration and record metrics
        self._end_time = time.time()
        self._last_duration = self._end_time - self._start_time
        self._register_metrics({"errors": 1})
        
        # Update circuit breaker if enabled
        if self._circuit_breaker_enabled:
            circuit = self._get_circuit_breaker(
                self._circuit_breaker_name or f"{self.name}_breaker"
            )
            if circuit:
                circuit.record_failure()
    
    def _register_metrics(self, additional_metrics: Dict[str, Any]) -> None:
        """
        Register performance and resource metrics for monitoring.
        
        Args:
            additional_metrics: Additional metrics to include
        """
        # Compile all metrics
        metrics = {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "retry_count": self._retry_count,
            "last_duration": self._last_duration,
            "success_rate": (self._success_count / self._execution_count) if self._execution_count > 0 else 0,
            **self._resource_metrics,
            **additional_metrics
        }
        
        # Try to register with monitoring system
        try:
            monitoring = self._get_service("monitoring")
            if monitoring:
                monitoring.register_stage_metrics(self.name, metrics)
        except Exception as e:
            # Just log locally if monitoring service is unavailable
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(f"Stage metrics: {metrics}")
    
    async def handle_error(self, context: PipelineContext, error: Exception) -> bool:
        """
        Handle an error that occurred during stage processing.
        
        This method provides a standard way for subclasses to handle errors.
        
        Args:
            context: The pipeline context
            error: The exception that occurred
            
        Returns:
            bool: True if the error was handled, False otherwise
        """
        # Classify the error
        error_info = self._classify_error(error, context)
        
        # Record the error
        self._record_error(error_info)
        
        # Log the error
        self._logger.error(
            f"Error in stage {self.name}: {error_info.get('message')} "
            f"[{error_info.get('category')}/{error_info.get('severity')}]"
        )
        
        # Add the error to context with classification
        context.add_error(self.name, {
            "message": str(error),
            "type": type(error).__name__,
            "category": error_info.get("category", "unknown"),
            "severity": error_info.get("severity", "persistent"),
            "suggested_actions": error_info.get("suggested_actions", [])
        })
        
        # Record the failure in metrics
        self._record_failure(context, error)
        
        return False
    
    async def initialize(self) -> None:
        """Initialize the stage, setting up required resources."""
        await super().initialize()
        self._logger.debug(f"Initializing stage {self.name}")
        
        # Reset metrics for new initialization
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._retry_count = 0
        self._error_history = []
        
        # Reset resource metrics
        self._resource_metrics = {
            "cpu_time": 0,
            "memory_usage": 0,
            "network_requests": 0,
            "io_operations": 0
        }
    
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        await super().cleanup()
        self._logger.debug(f"Cleaning up stage {self.name}")
        
        # Clear service cache
        self._services.clear()
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for base stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        schema = super().get_config_schema() or {
            "type": "object",
            "properties": {}
        }
        
        # Add base stage properties
        base_properties = {
            "enable_retry": {"type": "boolean"},
            "max_retries": {"type": "integer", "minimum": 0},
            "retry_backoff_factor": {"type": "number", "minimum": 1.0},
            "retry_jitter": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "retryable_errors": {"type": "array", "items": {"type": "string"}},
            "enable_circuit_breaker": {"type": "boolean"},
            "circuit_breaker_name": {"type": "string"},
            "timeout": {"type": "number", "minimum": 0},
            "memory_limit": {"type": "integer", "minimum": 0}
        }
        
        # Update the properties in the schema
        if "properties" not in schema:
            schema["properties"] = {}
        
        schema["properties"].update(base_properties)
        
        return schema
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance and resource metrics for this stage.
        
        Returns:
            Dict[str, Any]: Dictionary containing metrics
        """
        return {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "retry_count": self._retry_count,
            "last_duration": self._last_duration,
            "success_rate": (self._success_count / self._execution_count) if self._execution_count > 0 else 0,
            "resources": self._resource_metrics
        }
    
    def get_error_history(self) -> List[Dict[str, Any]]:
        """
        Get the error history for this stage.
        
        Returns:
            List[Dict[str, Any]]: List of error information dictionaries
        """
        # Filter out exception objects for better serialization
        return [{
            k: v for k, v in error.items() 
            if k != 'error' and not isinstance(v, Exception)
        } for error in self._error_history]
    
    def with_retry(self, operation: Callable, max_retries: Optional[int] = None) -> Callable:
        """
        Create a retry wrapper for an operation.
        
        This method provides a convenient way to apply retry logic to specific
        operations within a stage.
        
        Args:
            operation: The operation function to wrap
            max_retries: Optional override for maximum retries
            
        Returns:
            Callable: The wrapped operation function
        """
        max_retries = max_retries or self._max_retries
        
        async def wrapped_operation(*args, **kwargs):
            retry_attempt = 0
            retry_delay = 1.0
            
            while True:
                try:
                    # Attempt the operation
                    return await operation(*args, **kwargs)
                    
                except Exception as e:
                    # Handle the error
                    retry_attempt += 1
                    
                    # Classify the error
                    error_info = self._classify_error(e)
                    
                    # Record the error for analysis
                    self._record_error(error_info)
                    
                    # Check if we should retry
                    if retry_attempt <= max_retries and self._should_retry(error_info):
                        # Apply backoff with jitter
                        jitter = 1.0 + (self._retry_jitter * (2 * (time.time() % 1) - 1))
                        delay = retry_delay * jitter
                        
                        self._logger.info(
                            f"Retrying operation ({retry_attempt}/{max_retries}) after {delay:.2f}s"
                        )
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        
                        # Increase backoff for next attempt
                        retry_delay *= self._retry_backoff_factor
                        
                        # Record retry metrics
                        self._retry_count += 1
                        
                        # Continue to next attempt
                        continue
                    
                    # No more retries or not retryable - re-raise
                    raise
        
        return wrapped_operation
    
    def with_circuit_breaker(self, operation: Callable, circuit_name: Optional[str] = None) -> Callable:
        """
        Create a circuit breaker wrapper for an operation.
        
        This method provides a convenient way to apply circuit breaker protection
        to specific operations within a stage.
        
        Args:
            operation: The operation function to wrap
            circuit_name: Optional override for circuit breaker name
            
        Returns:
            Callable: The wrapped operation function
        """
        circuit_name = circuit_name or f"{self.name}_{operation.__name__}_breaker"
        
        async def wrapped_operation(*args, **kwargs):
            # Get circuit breaker
            circuit = self._get_circuit_breaker(circuit_name)
            
            if circuit and not circuit.allow_request():
                # Circuit is open
                raise OpenCircuitError(f"Circuit {circuit_name} is open")
            
            try:
                # Attempt the operation
                result = await operation(*args, **kwargs)
                
                # Record success
                if circuit:
                    circuit.record_success()
                
                return result
                
            except Exception as e:
                # Record failure
                if circuit:
                    circuit.record_failure()
                
                # Re-raise
                raise