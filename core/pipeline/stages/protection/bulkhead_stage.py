"""
Bulkhead Stage Module.

This module provides a BulkheadStage that implements the Bulkhead pattern
to isolate failures and prevent resource exhaustion in a distributed system.
"""

import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class BulkheadStage(PipelineStage):
    """
    A pipeline stage that implements the Bulkhead pattern.
    
    This stage wraps another stage with a bulkhead that limits the number
    of concurrent executions to prevent resource exhaustion and isolate failures.
    
    Features:
    - Configurable max concurrent executions
    - Queue for pending executions with configurable size
    - Timeouts for both queue waiting and execution time
    - Detailed metrics for monitoring bulkhead state
    - Integration with PipelineContext for state tracking
    
    Configuration:
    - wrapped_stage: The stage to protect with the bulkhead
    - max_concurrent_executions: Maximum number of concurrent executions allowed
    - max_queue_size: Maximum number of pending executions allowed
    - queue_timeout_seconds: Maximum time to wait in queue (0 for no timeout)
    - execution_timeout_seconds: Maximum execution time (0 for no timeout)
    """
    
    def __init__(self, 
                 wrapped_stage: PipelineStage,
                 name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the bulkhead stage.
        
        Args:
            wrapped_stage: The stage to protect with the bulkhead
            name: Optional name for the stage (defaults to wrapped stage name)
            config: Optional configuration dictionary
        """
        stage_name = name or f"bulkhead_{wrapped_stage.name}"
        super().__init__(stage_name, config or {})
        
        self.wrapped_stage = wrapped_stage
        
        # Extract configuration
        self.max_concurrent_executions = self.config.get("max_concurrent_executions", 10)
        self.max_queue_size = self.config.get("max_queue_size", 20)
        self.queue_timeout_seconds = self.config.get("queue_timeout_seconds", 0)
        self.execution_timeout_seconds = self.config.get("execution_timeout_seconds", 0)
        
        # Create the semaphore for limiting concurrent executions
        self.semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        
        # Create metrics
        self._current_executions = 0
        self._max_concurrent_seen = 0
        self._total_executions = 0
        self._queue_rejections = 0
        self._queue_timeouts = 0
        self._timed_out_executions = 0
        
        self.logger = logging.getLogger(f"pipeline.bulkhead.{self.name}")
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the stage, respecting bulkhead limits.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        # Check if we've reached the queue limit
        if self._current_executions >= self.max_concurrent_executions + self.max_queue_size:
            error_msg = f"Bulkhead queue full for {self.name}"
            self.logger.warning(error_msg)
            context.add_error(self.name, error_msg)
            self._queue_rejections += 1
            return False
        
        # Try to acquire the semaphore with timeout if configured
        acquired = False
        try:
            if self.queue_timeout_seconds > 0:
                # Use timeout to acquire semaphore
                try:
                    acquired = await asyncio.wait_for(
                        self.semaphore.acquire(),
                        timeout=self.queue_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    error_msg = f"Bulkhead queue timeout for {self.name} after {self.queue_timeout_seconds}s"
                    self.logger.warning(error_msg)
                    context.add_error(self.name, error_msg)
                    self._queue_timeouts += 1
                    return False
            else:
                # No timeout, just wait for semaphore
                await self.semaphore.acquire()
                acquired = True
            
            # Increase metrics
            self._current_executions += 1
            self._total_executions += 1
            self._max_concurrent_seen = max(self._max_concurrent_seen, self._current_executions)
            
            # Add bulkhead state to context
            context.set("bulkhead_execution_count", self._current_executions)
            context.set("bulkhead_name", self.name)
            
            # Execute with timeout if configured
            if self.execution_timeout_seconds > 0:
                try:
                    return await asyncio.wait_for(
                        self._execute_wrapped_stage(context),
                        timeout=self.execution_timeout_seconds
                    )
                except asyncio.TimeoutError:
                self.logger.error(error_msg)
                context.add_error(self.name, error_msg)
                return False
                
            except Exception as e:
                error_msg = f"Error in wrapped stage: {str(e)}"
                self.logger.error(error_msg)
                context.add_error(self.name, error_msg)
                return await self.wrapped_stage.handle_error(context, e)
                
            finally:
                # Always decrement execution count and release semaphore
                self._current_executions -= 1
                self._semaphore.release()
                
        except asyncio.TimeoutError:
            # Could not acquire semaphore within queue timeout
            self._queue_timeouts += 1
            self._rejected_executions += 1
            
            error_msg = (
                f"Bulkhead queue timeout after {self.queue_timeout_seconds}s, "
                f"max_concurrent={self.max_concurrent_executions}, "
                f"current={self._current_executions}"
            )
            self.logger.warning(error_msg)
            context.add_error(self.name, error_msg)
            context.set("bulkhead_rejected", True)
            context.set("bulkhead_rejection_reason", "queue_timeout")
            
            return False
            
        except Exception as e:
            # Other error while trying to acquire semaphore
            self._rejected_executions += 1
            
            error_msg = f"Error entering bulkhead: {str(e)}"
            self.logger.error(error_msg)
            context.add_error(self.name, error_msg)
            context.set("bulkhead_rejected", True)
            context.set("bulkhead_rejection_reason", "unexpected_error")
            
            return False
    
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
        
        # Calculate derived metrics
        avg_execution_time = (
            self._total_execution_time / self._total_executions 
            if self._total_executions > 0 else 0
        )
        avg_queue_time = (
            self._total_queue_time / self._total_executions 
            if self._total_executions > 0 else 0
        )
        rejection_rate = (
            self._rejected_executions / (self._total_executions + self._rejected_executions) * 100
            if (self._total_executions + self._rejected_executions) > 0 else 0
        )
        timeout_rate = (
            self._timed_out_executions / self._total_executions * 100
            if self._total_executions > 0 else 0
        )
        
        # Add bulkhead specific metrics
        bulkhead_metrics = {
            "current_executions": self._current_executions,
            "total_executions": self._total_executions,
            "rejected_executions": self._rejected_executions,
            "timed_out_executions": self._timed_out_executions,
            "queue_timeouts": self._queue_timeouts,
            "avg_execution_time": avg_execution_time,
            "avg_queue_time": avg_queue_time,
            "max_concurrent_executions": self.max_concurrent_executions,
            "max_queue_size": self.max_queue_size,
            "rejection_rate": rejection_rate,
            "timeout_rate": timeout_rate,
            "queue_size": len(self._queue) if hasattr(self._queue, "__len__") else 0
        }
        
        metrics["bulkhead"] = bulkhead_metrics
        
        # Try to get wrapped stage metrics if available
        if hasattr(self.wrapped_stage, "get_metrics"):
            metrics["wrapped_stage"] = self.wrapped_stage.get_metrics()
            
        return metrics
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for bulkhead stage configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing JSON schema
        """
        return {
            "type": "object",
            "properties": {
                "max_concurrent_executions": {"type": "integer", "minimum": 1},
                "max_queue_size": {"type": "integer", "minimum": 0},
                "execution_timeout_seconds": {"type": "number", "minimum": 0},
                "queue_timeout_seconds": {"type": "number", "minimum": 0}
            }
        }