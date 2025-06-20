"""
Base Pipeline Stages Module.

This module defines abstract base classes for different types of pipeline stages,
providing specialized functionality for input, processing, output, and conditional stages.
"""

import abc
import asyncio
import copy
import json
import logging
import time
import traceback
from dataclasses import asdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineMetrics, PipelineRequest, PipelineResponse


class InputStage(PipelineStage):
    """
    Abstract base class for all input stages.
    
    Input stages specialize in data acquisition from various sources
    like HTTP, files, or databases, and preparing it for processing.
    
    Attributes:
        retry_count (int): Maximum number of retry attempts for transient failures.
        retry_delay (float): Delay between retry attempts in seconds.
        throttle_rate (float): Rate limiting for requests (items per second).
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new input stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name if not provided.
            config (Optional[Dict[str, Any]]): Configuration parameters for this stage.
        """
        super().__init__(name, config)
        self.retry_count = self.config.get("retry_count", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.throttle_rate = self.config.get("throttle_rate", 0.0)
        self._last_request_time = 0.0
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the input stage, acquiring data from the specified source.
        
        This method implements retry logic and throttling before delegating
        to the acquire_data method that must be implemented by subclasses.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        # Apply throttling if configured
        if self.throttle_rate > 0:
            await self._apply_throttling()
        
        # Get the input request from context or create default
        request = self._get_request(context)
        
        # Implement retry logic for transient failures
        attempt = 0
        while attempt <= self.retry_count:
            try:
                start_time = time.time()
                
                # Call the acquire_data method that must be implemented by subclasses
                response = await self.acquire_data(request, context)
                
                # Store acquisition metrics
                duration = time.time() - start_time
                self._record_metrics(context, True, duration)
                
                # Store the response data in the context
                if response:
                    self._store_response(response, context)
                    return True
                return False
            
            except Exception as e:
                attempt += 1
                if attempt <= self.retry_count:
                    retry_delay = self.retry_delay * attempt
                    self.logger.warning(
                        f"Attempt {attempt}/{self.retry_count} failed: {str(e)}. "
                        f"Retrying in {retry_delay:.2f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"All {self.retry_count} retry attempts failed")
                    self._record_metrics(context, False, time.time() - start_time)
                    await self.handle_error(context, e)
                    return False
    
    @abc.abstractmethod
    async def acquire_data(self, request: PipelineRequest, context: PipelineContext) -> Optional[PipelineResponse]:
        """
        Acquire data from the specified source.
        
        This method must be implemented by subclasses to perform the actual data acquisition.
        
        Args:
            request (PipelineRequest): The input request data.
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            Optional[PipelineResponse]: The acquired data or None if acquisition failed.
        """
        pass
    
    async def validate_source_config(self) -> bool:
        """
        Validate the configuration for the data source.
        
        Override this method to implement source-specific validation.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        return True
    
    async def _apply_throttling(self) -> None:
        """
        Apply throttling to control request rate.
        
        This method ensures that requests are not sent more frequently
        than the configured throttle_rate.
        """
        if self.throttle_rate <= 0:
            return
            
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.throttle_rate
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            await asyncio.sleep(delay)
            
        self._last_request_time = time.time()
    
    def _get_request(self, context: PipelineContext) -> PipelineRequest:
        """
        Get the input request from context or create a default one.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            PipelineRequest: The input request data.
        """
        # Check if there's a request object in context
        request = context.get("request")
        if request is not None and isinstance(request, PipelineRequest):
            return request
            
        # Create a default request using context data
        return PipelineRequest(
            source=self.config.get("source", ""),
            params=context.data.copy()
        )
    
    def _store_response(self, response: PipelineResponse, context: PipelineContext) -> None:
        """
        Store the response data in the context.
        
        Args:
            response (PipelineResponse): The response data.
            context (PipelineContext): The shared pipeline context.
        """
        # Store the full response object
        context.set("response", response)
        
        # Also store the response data directly in context for easier access
        if response.data:
            context.update(response.data)
    
    def _record_metrics(self, context: PipelineContext, success: bool, duration: float) -> None:
        """
        Record acquisition metrics in the context.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            success (bool): Whether the acquisition succeeded.
            duration (float): Time taken for acquisition in seconds.
        """
        metrics = context.get("input_metrics", {})
        stage_metrics = metrics.get(self.name, {"count": 0, "success": 0, "total_time": 0.0})
        
        stage_metrics["count"] += 1
        if success:
            stage_metrics["success"] += 1
        stage_metrics["total_time"] += duration
        stage_metrics["last_duration"] = duration
        stage_metrics["success_rate"] = stage_metrics["success"] / stage_metrics["count"]
        
        metrics[self.name] = stage_metrics
        context.set("input_metrics", metrics)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for input stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema.
        """
        return {
            "type": "object",
            "properties": {
                "retry_count": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0},
                "throttle_rate": {"type": "number", "minimum": 0},
                "source": {"type": "string"}
            }
        }


class ProcessingMode(Enum):
    """Enumeration of processing modes for ProcessingStage."""
    IN_PLACE = auto()
    COPY = auto()
    NEW = auto()


class ProcessingStage(PipelineStage):
    """
    Abstract base class for all processing stages.
    
    Processing stages specialize in data transformation and manipulation,
    converting input data into the desired format or structure.
    
    Attributes:
        processing_mode (ProcessingMode): How the stage processes data (in-place, copy, new).
        enable_caching (bool): Whether to cache processing results.
        cache_key_template (str): Template for generating cache keys.
        use_circuit_breaker (bool): Whether to use circuit breaker for fault tolerance.
        circuit_breaker_name (str): Name of the circuit breaker to use.
        telemetry_enabled (bool): Whether to collect detailed telemetry data.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new processing stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name if not provided.
            config (Optional[Dict[str, Any]]): Configuration parameters for this stage.
        """
        super().__init__(name, config)
        self.processing_mode = ProcessingMode[self.config.get("processing_mode", "IN_PLACE")]
        self.enable_caching = self.config.get("enable_caching", False)
        self.cache_key_template = self.config.get("cache_key_template", "{stage_name}:{input_hash}")
        self.use_circuit_breaker = self.config.get("use_circuit_breaker", False)
        self.circuit_breaker_name = self.config.get("circuit_breaker_name", f"proc_{self.name}")
        self.telemetry_enabled = self.config.get("telemetry_enabled", True)
        self.max_retries = self.config.get("max_retries", 0)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self._circuit_breaker = None
        self._cache: Dict[str, Any] = {}
        
        # Initialize circuit breaker if enabled
        if self.use_circuit_breaker:
            self._initialize_circuit_breaker()
    
    def _initialize_circuit_breaker(self) -> None:
        """
        Initialize the circuit breaker for this stage.
        """
        try:
            from core.circuit_breaker import CircuitBreaker
            
            # Get circuit breaker configuration from stage config or use defaults
            failure_threshold = self.config.get("circuit_breaker_failure_threshold", 5)
            reset_timeout = self.config.get("circuit_breaker_reset_timeout", 60)
            excluded_exceptions = self.config.get("circuit_breaker_excluded_exceptions", [])
            
            # Convert string exception names to actual exception classes
            excluded_exception_classes = set()
            for exc_name in excluded_exceptions:
                try:
                    exc_class = eval(exc_name)
                    if issubclass(exc_class, Exception):
                        excluded_exception_classes.add(exc_class)
                except (NameError, TypeError):
                    self.logger.warning(f"Invalid exception class: {exc_name}")
            
            # Create the circuit breaker
            self._circuit_breaker = CircuitBreaker(
                name=self.circuit_breaker_name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                excluded_exceptions=excluded_exception_classes
            )
            
            self.logger.info(f"Initialized circuit breaker for stage {self.name}")
        except ImportError:
            self.logger.warning("CircuitBreaker not available, disabling circuit breaker functionality")
            self.use_circuit_breaker = False
        except Exception as e:
            self.logger.error(f"Failed to initialize circuit breaker: {str(e)}")
            self.use_circuit_breaker = False
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the current stage, transforming data from the context.
        
        This method handles processing modes, caching, circuit breaker protection,
        retries, and metrics before delegating to the transform_data method.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        telemetry = {} if self.telemetry_enabled else None
        if telemetry is not None:
            telemetry["stage_name"] = self.name
            telemetry["start_time"] = time.time()
            telemetry["retries"] = 0
        
        # Validate content before processing
        if not await self.validate_content(context):
            self.logger.warning(f"Content validation failed for stage {self.name}")
            context.add_error(self.name, "Content validation failed")
            if telemetry is not None:
                telemetry["status"] = "validation_failed"
                self._store_telemetry(context, telemetry)
            return False
        
        # Check cache if enabled
        if self.enable_caching:
            cache_key = self._generate_cache_key(context)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Using cached result for {self.name}")
                self._store_result(cached_result, context)
                self._record_metrics(context, True, 0.0, True)
                if telemetry is not None:
                    telemetry["status"] = "cache_hit"
                    self._store_telemetry(context, telemetry)
                return True
            elif telemetry is not None:
                telemetry["cache_checked"] = True
        
        # Prepare input data based on processing mode
        input_data = self._prepare_input_data(context)
        
        # Use circuit breaker if enabled
        if self.use_circuit_breaker and self._circuit_breaker:
            return await self._process_with_circuit_breaker(input_data, context, telemetry)
        else:
            # Use retry logic if configured
            if self.max_retries > 0:
                return await self._process_with_retries(input_data, context, telemetry)
            else:
                # Process normally
                return await self._process_internal(input_data, context, telemetry)
    
    async def _process_with_circuit_breaker(self, input_data: Any, context: PipelineContext, telemetry: Optional[Dict[str, Any]]) -> bool:
        """
        Process with circuit breaker protection.
        
        Args:
            input_data (Any): The input data to transform.
            context (PipelineContext): The shared pipeline context.
            telemetry (Optional[Dict[str, Any]]): Telemetry data collection.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        try:
            # Use circuit breaker to protect the processing
            if telemetry is not None:
                telemetry["circuit_breaker_state"] = self._circuit_breaker.state.value
            
            # Execute with circuit breaker protection
            result = await self._circuit_breaker.execute_async(
                self._process_with_retries if self.max_retries > 0 else self._process_internal,
                input_data, context, telemetry
            )
            return result
        except Exception as e:
            self.logger.error(f"Circuit breaker prevented execution: {str(e)}")
            context.add_error(self.name, f"Circuit breaker error: {str(e)}")
            if telemetry is not None:
                telemetry["status"] = "circuit_breaker_error"
                telemetry["error"] = str(e)
                telemetry["end_time"] = time.time()
                self._store_telemetry(context, telemetry)
            return False
    
    async def _process_with_retries(self, input_data: Any, context: PipelineContext, telemetry: Optional[Dict[str, Any]]) -> bool:
        """
        Process with retry logic.
        
        Args:
            input_data (Any): The input data to transform.
            context (PipelineContext): The shared pipeline context.
            telemetry (Optional[Dict[str, Any]]): Telemetry data collection.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt}/{self.max_retries} for stage {self.name}")
                    if telemetry is not None:
                        telemetry["retries"] = attempt
                
                result = await self._process_internal(input_data, context, telemetry)
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt <= self.max_retries:
                    retry_delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed: {str(e)}. "
                        f"Retrying in {retry_delay:.2f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} retry attempts failed")
                    context.add_error(self.name, f"Max retries exceeded: {str(e)}")
                    if telemetry is not None:
                        telemetry["status"] = "max_retries_exceeded"
                        telemetry["error"] = str(e)
                        telemetry["end_time"] = time.time()
                        self._store_telemetry(context, telemetry)
                    return False
        
        # Should never reach here, but just in case
        await self.handle_error(context, last_error or Exception("Unknown error after retries"))
        return False
    
    async def _process_internal(self, input_data: Any, context: PipelineContext, telemetry: Optional[Dict[str, Any]]) -> bool:
        """
        Internal processing logic without circuit breaker or retries.
        
        Args:
            input_data (Any): The input data to transform.
            context (PipelineContext): The shared pipeline context.
            telemetry (Optional[Dict[str, Any]]): Telemetry data collection.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        start_time = time.time()
        try:
            # Call the transform_data method that must be implemented by subclasses
            result = await self.transform_data(input_data, context)
            success = result is not None
            
            # Store the result in context
            if success:
                self._store_result(result, context)
                
                # Cache the result if enabled
                if self.enable_caching:
                    cache_key = self._generate_cache_key(context)
                    self._add_to_cache(cache_key, result)
            
            # Record processing metrics and telemetry
            duration = time.time() - start_time
            self._record_metrics(context, success, duration, False)
            
            if telemetry is not None:
                telemetry["status"] = "success" if success else "failed"
                telemetry["duration"] = duration
                telemetry["end_time"] = time.time()
                telemetry["success"] = success
                self._store_telemetry(context, telemetry)
            
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_metrics(context, False, duration, False)
            
            if telemetry is not None:
                telemetry["status"] = "error"
                telemetry["error"] = str(e)
                telemetry["error_type"] = type(e).__name__
                telemetry["duration"] = duration
                telemetry["end_time"] = time.time()
                telemetry["success"] = False
                self._store_telemetry(context, telemetry)
            
            raise  # Re-raise for retry or circuit breaker to handle
    
    def _store_telemetry(self, context: PipelineContext, telemetry: Dict[str, Any]) -> None:
        """
        Store telemetry data in the context.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            telemetry (Dict[str, Any]): The telemetry data to store.
        """
        stage_telemetry = context.get("stage_telemetry", {})
        
        # Store the current telemetry in the history
        telemetry_history = stage_telemetry.get(self.name, [])
        telemetry_history.append(telemetry)
        
        # Limit history length to avoid memory issues
        max_history = self.config.get("telemetry_history_limit", 20)
        if len(telemetry_history) > max_history:
            telemetry_history = telemetry_history[-max_history:]
        
        stage_telemetry[self.name] = telemetry_history
        context.set("stage_telemetry", stage_telemetry)

    # ... existing methods remain unchanged ...
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for processing stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema.
        """
        return {
            "type": "object",
            "properties": {
                "processing_mode": {"type": "string", "enum": ["IN_PLACE", "COPY", "NEW"]},
                "enable_caching": {"type": "boolean"},
                "cache_key_template": {"type": "string"},
                "input_key": {"type": "string"},
                "output_key": {"type": "string"},
                "use_circuit_breaker": {"type": "boolean"},
                "circuit_breaker_name": {"type": "string"},
                "circuit_breaker_failure_threshold": {"type": "integer", "minimum": 1},
                "circuit_breaker_reset_timeout": {"type": "integer", "minimum": 1},
                "circuit_breaker_excluded_exceptions": {"type": "array", "items": {"type": "string"}},
                "telemetry_enabled": {"type": "boolean"},
                "telemetry_history_limit": {"type": "integer", "minimum": 1},
                "max_retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0.1}
            }
        }


class OutputMode(Enum):
    """Enumeration of output modes for OutputStage."""
    OVERWRITE = auto()
    APPEND = auto()
    UPDATE = auto()


class OutputStage(PipelineStage):
    """
    Abstract base class for all output stages.
    
    Output stages specialize in delivering processed data to destinations
    such as files, databases, or external services.
    
    Attributes:
        output_mode (OutputMode): How to handle existing data (overwrite, append, update).
        backup_enabled (bool): Whether to create backups before changes.
        backup_path_template (str): Template for backup file paths.
        use_circuit_breaker (bool): Whether to use circuit breaker for fault tolerance.
        circuit_breaker_name (str): Name of the circuit breaker to use.
        telemetry_enabled (bool): Whether to collect detailed telemetry data.
        max_retries (int): Maximum number of retry attempts for transient failures.
        retry_delay (float): Base delay between retry attempts in seconds.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new output stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name if not provided.
            config (Optional[Dict[str, Any]]): Configuration parameters for this stage.
        """
        super().__init__(name, config)
        self.output_mode = OutputMode[self.config.get("output_mode", "OVERWRITE")]
        self.backup_enabled = self.config.get("backup_enabled", False)
        self.backup_path_template = self.config.get("backup_path_template", "{path}.bak")
        self.use_circuit_breaker = self.config.get("use_circuit_breaker", False)
        self.circuit_breaker_name = self.config.get("circuit_breaker_name", f"output_{self.name}")
        self.telemetry_enabled = self.config.get("telemetry_enabled", True)
        self.max_retries = self.config.get("max_retries", 2)  # Default to 2 retries for output
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self._circuit_breaker = None
        self._operation_stats: Dict[str, int] = {"success": 0, "failure": 0, "skipped": 0}
        
        # Initialize circuit breaker if enabled
        if self.use_circuit_breaker:
            self._initialize_circuit_breaker()
    
    def _initialize_circuit_breaker(self) -> None:
        """
        Initialize the circuit breaker for this stage.
        """
        try:
            from core.circuit_breaker import CircuitBreaker
            
            # Get circuit breaker configuration from stage config or use defaults
            failure_threshold = self.config.get("circuit_breaker_failure_threshold", 3)
            reset_timeout = self.config.get("circuit_breaker_reset_timeout", 60)
            excluded_exceptions = self.config.get("circuit_breaker_excluded_exceptions", [])
            
            # Convert string exception names to actual exception classes
            excluded_exception_classes = set()
            for exc_name in excluded_exceptions:
                try:
                    exc_class = eval(exc_name)
                    if issubclass(exc_class, Exception):
                        excluded_exception_classes.add(exc_class)
                except (NameError, TypeError):
                    self.logger.warning(f"Invalid exception class: {exc_name}")
            
            # Create the circuit breaker
            self._circuit_breaker = CircuitBreaker(
                name=self.circuit_breaker_name,
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                excluded_exceptions=excluded_exception_classes
            )
            
            self.logger.info(f"Initialized circuit breaker for output stage {self.name}")
        except ImportError:
            self.logger.warning("CircuitBreaker not available, disabling circuit breaker functionality")
            self.use_circuit_breaker = False
        except Exception as e:
            self.logger.error(f"Failed to initialize circuit breaker: {str(e)}")
            self.use_circuit_breaker = False
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the output stage, delivering data to the specified destination.
        
        This method handles output formatting, backups, circuit breaker protection,
        retries, and telemetry before delegating to the deliver_output method.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        telemetry = {} if self.telemetry_enabled else None
        if telemetry is not None:
            telemetry["stage_name"] = self.name
            telemetry["start_time"] = time.time()
            telemetry["retries"] = 0
            telemetry["output_mode"] = self.output_mode.name
        
        # Validate output format
        if not await self.validate_output_format(context):
            self.logger.warning(f"Output format validation failed for stage {self.name}")
            context.add_error(self.name, "Output format validation failed")
            if telemetry is not None:
                telemetry["status"] = "validation_failed"
                self._store_telemetry(context, telemetry)
            return False
        
        # Prepare output data
        output_data = self._prepare_output_data(context)
        if output_data is None:
            self.logger.warning(f"No output data found for stage {self.name}")
            context.add_error(self.name, "No output data found")
            if telemetry is not None:
                telemetry["status"] = "no_output_data"
                self._store_telemetry(context, telemetry)
            return False
        
        # Create backup if enabled
        if self.backup_enabled:
            if not await self._create_backup(context):
                self.logger.warning(f"Failed to create backup for stage {self.name}")
                context.add_error(self.name, "Failed to create backup")
                if telemetry is not None:
                    telemetry["status"] = "backup_failed"
                    self._store_telemetry(context, telemetry)
                return False
        
        # Use circuit breaker if enabled
        if self.use_circuit_breaker and self._circuit_breaker:
            return await self._process_with_circuit_breaker(output_data, context, telemetry)
        else:
            # Use retry logic if configured
            if self.max_retries > 0:
                return await self._process_with_retries(output_data, context, telemetry)
            else:
                # Process normally
                return await self._process_internal(output_data, context, telemetry)
    
    async def _process_with_circuit_breaker(self, output_data: Any, context: PipelineContext, telemetry: Optional[Dict[str, Any]]) -> bool:
        """
        Process with circuit breaker protection.
        
        Args:
            output_data (Any): The output data to deliver.
            context (PipelineContext): The shared pipeline context.
            telemetry (Optional[Dict[str, Any]]): Telemetry data collection.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        try:
            # Use circuit breaker to protect the processing
            if telemetry is not None:
                telemetry["circuit_breaker_state"] = self._circuit_breaker.state.value
            
            # Execute with circuit breaker protection
            result = await self._circuit_breaker.execute_async(
                self._process_with_retries if self.max_retries > 0 else self._process_internal,
                output_data, context, telemetry
            )
            return result
        except Exception as e:
            self.logger.error(f"Circuit breaker prevented execution: {str(e)}")
            context.add_error(self.name, f"Circuit breaker error: {str(e)}")
            if telemetry is not None:
                telemetry["status"] = "circuit_breaker_error"
                telemetry["error"] = str(e)
                telemetry["end_time"] = time.time()
                self._store_telemetry(context, telemetry)
            return False
    
    async def _process_with_retries(self, output_data: Any, context: PipelineContext, telemetry: Optional[Dict[str, Any]]) -> bool:
        """
        Process with retry logic.
        
        Args:
            output_data (Any): The output data to deliver.
            context (PipelineContext): The shared pipeline context.
            telemetry (Optional[Dict[str, Any]]): Telemetry data collection.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt}/{self.max_retries} for stage {self.name}")
                    if telemetry is not None:
                        telemetry["retries"] = attempt
                
                result = await self._process_internal(output_data, context, telemetry)
                return result
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt <= self.max_retries:
                    retry_delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt}/{self.max_retries} failed: {str(e)}. "
                        f"Retrying in {retry_delay:.2f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} retry attempts failed")
                    context.add_error(self.name, f"Max retries exceeded: {str(e)}")
                    if telemetry is not None:
                        telemetry["status"] = "max_retries_exceeded"
                        telemetry["error"] = str(e)
                        telemetry["end_time"] = time.time()
                        self._store_telemetry(context, telemetry)
                    return False
        
        # Should never reach here, but just in case
        await self.handle_error(context, last_error or Exception("Unknown error after retries"))
        return False
    
    async def _process_internal(self, output_data: Any, context: PipelineContext, telemetry: Optional[Dict[str, Any]]) -> bool:
        """
        Internal processing logic without circuit breaker or retries.
        
        Args:
            output_data (Any): The output data to deliver.
            context (PipelineContext): The shared pipeline context.
            telemetry (Optional[Dict[str, Any]]): Telemetry data collection.
            
        Returns:
            bool: True if processing succeeded, False otherwise.
        """
        start_time = time.time()
        try:
            # Call the deliver_output method that must be implemented by subclasses
            result = await self.deliver_output(output_data, context)
            success = result is not None
            
            # Record processing metrics and telemetry
            duration = time.time() - start_time
            self._record_metrics(context, success, duration)
            
            if telemetry is not None:
                telemetry["status"] = "success" if success else "failed"
                telemetry["duration"] = duration
                telemetry["end_time"] = time.time()
                telemetry["success"] = success
                self._store_telemetry(context, telemetry)
            
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_metrics(context, False, duration)
            
            if telemetry is not None:
                telemetry["status"] = "error"
                telemetry["error"] = str(e)
                telemetry["error_type"] = type(e).__name__
                telemetry["duration"] = duration
                telemetry["end_time"] = time.time()
                telemetry["success"] = False
                self._store_telemetry(context, telemetry)
            
            # Rollback from backup if enabled
            if self.backup_enabled:
                await self._rollback_from_backup(context)
            
            raise  # Re-raise for retry or circuit breaker to handle
    
    def _store_telemetry(self, context: PipelineContext, telemetry: Dict[str, Any]) -> None:
        """
        Store telemetry data in the context.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            telemetry (Dict[str, Any]): The telemetry data to store.
        """
        stage_telemetry = context.get("stage_telemetry", {})
        
        # Store the current telemetry in the history
        telemetry_history = stage_telemetry.get(self.name, [])
        telemetry_history.append(telemetry)
        
        # Limit history length to avoid memory issues
        max_history = self.config.get("telemetry_history_limit", 20)
        if len(telemetry_history) > max_history:
            telemetry_history = telemetry_history[-max_history:]
        
        stage_telemetry[self.name] = telemetry_history
        context.set("stage_telemetry", stage_telemetry)
    
    @abc.abstractmethod
    async def deliver_output(self, data: Any, context: PipelineContext) -> Optional[Any]:
        """
        Deliver output data to the specified destination.
        
        This method must be implemented by subclasses to perform the actual data delivery.
        
        Args:
            data (Any): The output data to deliver.
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            Optional[Any]: Delivery result or None if delivery failed.
        """
        pass
    
    async def validate_output_format(self, context: PipelineContext) -> bool:
        """
        Validate the output format before delivery.
        
        Override this method to implement format-specific validation.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if format is valid, False otherwise.
        """
        return True
    
    async def _create_backup(self, context: PipelineContext) -> bool:
        """
        Create a backup before delivering output.
        
        Override this method to implement destination-specific backup logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if backup was created, False otherwise.
        """
        return True
    
    async def _rollback_from_backup(self, context: PipelineContext) -> bool:
        """
        Rollback to backup after failed delivery.
        
        Override this method to implement destination-specific rollback logic.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: True if rollback succeeded, False otherwise.
        """
        return True
    
    def _prepare_output_data(self, context: PipelineContext) -> Optional[Any]:
        """
        Prepare data for output delivery.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            Optional[Any]: The prepared output data.
        """
        data_key = self.config.get("data_key")
        
        # Get data from specific key or use entire context
        if data_key:
            return context.get(data_key)
        else:
            return context.data
    
    def _record_metrics(self, context: PipelineContext, success: bool, duration: float) -> None:
        """
        Record output metrics in the context.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            success (bool): Whether the delivery succeeded.
            duration (float): Time taken for delivery in seconds.
        """
        metrics = context.get("output_metrics", {})
        stage_metrics = metrics.get(self.name, {"count": 0, "success": 0, "total_time": 0.0})
        
        stage_metrics["count"] += 1
        if success:
            stage_metrics["success"] += 1
        stage_metrics["total_time"] += duration
        stage_metrics["last_duration"] = duration
        stage_metrics["success_rate"] = stage_metrics["success"] / stage_metrics["count"]
        
        metrics[self.name] = stage_metrics
        context.set("output_metrics", metrics)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for output stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema.
        """
        return {
            "type": "object",
            "properties": {
                "output_mode": {"type": "string", "enum": ["OVERWRITE", "APPEND", "UPDATE"]},
                "backup_enabled": {"type": "boolean"},
                "backup_path_template": {"type": "string"},
                "use_circuit_breaker": {"type": "boolean"},
                "circuit_breaker_name": {"type": "string"},
                "circuit_breaker_failure_threshold": {"type": "integer", "minimum": 1},
                "circuit_breaker_reset_timeout": {"type": "integer", "minimum": 1},
                "circuit_breaker_excluded_exceptions": {"type": "array", "items": {"type": "string"}},
                "telemetry_enabled": {"type": "boolean"},
                "telemetry_history_limit": {"type": "integer", "minimum": 1},
                "max_retries": {"type": "integer", "minimum": 0},
                "retry_delay": {"type": "number", "minimum": 0.1},
                "data_key": {"type": "string"}
            }
        }


class ConditionalOperator(Enum):
    """Enumeration of conditional operators for ConditionalStage."""
    EQUALS = auto()
    NOT_EQUALS = auto()
    CONTAINS = auto()
    NOT_CONTAINS = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    GREATER_EQUAL = auto()
    LESS_EQUAL = auto()
    EXISTS = auto()
    NOT_EXISTS = auto()
    REGEX_MATCH = auto()
    IS_TYPE = auto()


class ConditionalStage(PipelineStage):
    """
    Abstract base class for all conditional stages.
    
    Conditional stages specialize in evaluating conditions and determining
    whether subsequent stages should be executed based on the results.
    
    Attributes:
        conditions (List[Dict[str, Any]]): List of conditions to evaluate.
        default_result (bool): Default result if evaluation fails.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new conditional stage.
        
        Args:
            name (Optional[str]): Name of the stage, defaults to class name if not provided.
            config (Optional[Dict[str, Any]]): Configuration parameters for this stage.
        """
        super().__init__(name, config)
        self.conditions = self.config.get("conditions", [])
        self.default_result = self.config.get("default_result", True)
        self._evaluation_stats: Dict[str, int] = {"total": 0, "true": 0, "false": 0}
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the conditional stage, evaluating conditions.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: Result of condition evaluation.
        """
        start_time = time.time()
        condition_results = []
        
        try:
            # Evaluate individual conditions
            for condition in self.conditions:
                result = self._evaluate_condition(condition, context)
                condition_results.append(result)
            
            # Combine results based on logical operator
            final_result = self._combine_results(condition_results)
            
            # Record evaluation metrics
            duration = time.time() - start_time
            self._record_metrics(context, True, duration, final_result)
            
            # Update evaluation stats
            self._evaluation_stats["total"] += 1
            if final_result:
                self._evaluation_stats["true"] += 1
            else:
                self._evaluation_stats["false"] += 1
            
            # Store evaluation result in context
            context.set(f"{self.name}_result", final_result)
            context.set(f"{self.name}_condition_results", condition_results)
            
            return final_result
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_metrics(context, False, duration, False)
            await self.handle_error(context, e)
            return self.default_result
    
    def _evaluate_condition(self, condition: Dict[str, Any], context: PipelineContext) -> bool:
        """
        Evaluate a single condition.
        
        Args:
            condition (Dict[str, Any]): The condition configuration.
            context (PipelineContext): The shared pipeline context.
            
        Returns:
            bool: Result of condition evaluation.
        """
        # Extract condition parameters
        key = condition.get("key", "")
        operator_str = condition.get("operator", "EQUALS")
        expected_value = condition.get("value")
        negate = condition.get("negate", False)
        
        # Get actual value from context
        actual_value = context.get(key)
        
        # Convert operator string to enum
        operator = ConditionalOperator[operator_str]
        
        # Evaluate based on operator
        result = False
        
        if operator == ConditionalOperator.EQUALS:
            result = actual_value == expected_value
        elif operator == ConditionalOperator.NOT_EQUALS:
            result = actual_value != expected_value
        elif operator == ConditionalOperator.CONTAINS:
            result = expected_value in actual_value if actual_value else False
        elif operator == ConditionalOperator.NOT_CONTAINS:
            result = expected_value not in actual_value if actual_value else True
        elif operator == ConditionalOperator.GREATER_THAN:
            result = actual_value > expected_value if actual_value is not None else False
        elif operator == ConditionalOperator.LESS_THAN:
            result = actual_value < expected_value if actual_value is not None else False
        elif operator == ConditionalOperator.GREATER_EQUAL:
            result = actual_value >= expected_value if actual_value is not None else False
        elif operator == ConditionalOperator.LESS_EQUAL:
            result = actual_value <= expected_value if actual_value is not None else False
        elif operator == ConditionalOperator.EXISTS:
            result = actual_value is not None
        elif operator == ConditionalOperator.NOT_EXISTS:
            result = actual_value is None
        elif operator == ConditionalOperator.REGEX_MATCH:
            import re
            result = bool(re.match(expected_value, str(actual_value))) if actual_value else False
        elif operator == ConditionalOperator.IS_TYPE:
            result = isinstance(actual_value, eval(expected_value)) if actual_value else False
        
        # Apply negation if specified
        if negate:
            result = not result
            
        return result
    
    def _combine_results(self, results: List[bool]) -> bool:
        """
        Combine multiple condition results using logical operator.
        
        Args:
            results (List[bool]): List of individual condition results.
            
        Returns:
            bool: Combined result.
        """
        if not results:
            return self.default_result
            
        # Get logical operator (AND or OR)
        logical_op = self.config.get("logical_operator", "AND")
        
        if logical_op == "AND":
            return all(results)
        elif logical_op == "OR":
            return any(results)
        else:
            self.logger.warning(f"Unknown logical operator: {logical_op}, defaulting to AND")
            return all(results)
    
    def _record_metrics(self, context: PipelineContext, success: bool, duration: float, result: bool) -> None:
        """
        Record evaluation metrics in the context.
        
        Args:
            context (PipelineContext): The shared pipeline context.
            success (bool): Whether the evaluation succeeded.
            duration (float): Time taken for evaluation in seconds.
            result (bool): Result of the evaluation.
        """
        metrics = context.get("conditional_metrics", {})
        stage_metrics = metrics.get(self.name, {"count": 0, "success": 0, "total_time": 0.0})
        
        stage_metrics["count"] += 1
        if success:
            stage_metrics["success"] += 1
        stage_metrics["total_time"] += duration
        stage_metrics["last_duration"] = duration
        stage_metrics["success_rate"] = stage_metrics["success"] / stage_metrics["count"]
        
        # Add evaluation-specific metrics
        stage_metrics["last_result"] = result
        stage_metrics["true_count"] = self._evaluation_stats["true"]
        stage_metrics["false_count"] = self._evaluation_stats["false"]
        stage_metrics["true_percentage"] = (self._evaluation_stats["true"] / self._evaluation_stats["total"]) * 100 if self._evaluation_stats["total"] > 0 else 0
        
        metrics[self.name] = stage_metrics
        context.set("conditional_metrics", metrics)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for conditional stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema.
        """
        return {
            "type": "object",
            "properties": {
                "conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "operator": {"type": "string", "enum": [op.name for op in ConditionalOperator]},
                            "value": {"type": ["string", "number", "boolean", "null"]},
                            "negate": {"type": "boolean"}
                        },
                        "required": ["key", "operator"]
                    }
                },
                "logical_operator": {"type": "string", "enum": ["AND", "OR"]},
                "default_result": {"type": "boolean"}
            },
            "required": ["conditions"]
        }