"""
Pipeline Data Transfer Objects.

This module provides data transfer objects (DTOs) for standardized data exchange
between pipeline stages and for capturing performance metrics.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union


class RequestMethod(Enum):
    """HTTP request methods for PipelineRequest."""
    GET = auto()
    POST = auto()
    PUT = auto()
    DELETE = auto()
    HEAD = auto()
    OPTIONS = auto()
    PATCH = auto()


@dataclass
class PipelineRequest:
    """
    Standardized request object for input stages.
    
    Contains information about the source of the data and any parameters
    needed to acquire it.
    
    Attributes:
        source (str): The source identifier (URL, file path, etc.)
        params (Dict[str, Any]): Request parameters.
        headers (Dict[str, str]): Request headers for HTTP requests.
        method (RequestMethod): HTTP method for web requests.
        body (Optional[Any]): Request body for HTTP POST/PUT requests.
        metadata (Dict[str, Any]): Additional metadata about the request.
    """
    source: str
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    method: RequestMethod = RequestMethod.GET
    body: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseStatus(Enum):
    """Status codes for PipelineResponse."""
    SUCCESS = auto()
    PARTIAL = auto()
    ERROR = auto()
    NOT_FOUND = auto()
    RATE_LIMITED = auto()
    UNAUTHORIZED = auto()
    FORBIDDEN = auto()
    TIMEOUT = auto()
    SERVER_ERROR = auto()
    UNKNOWN = auto()


@dataclass
class PipelineResponse:
    """
    Standardized response object from input stages.
    
    Contains the acquired data and information about the acquisition result.
    
    Attributes:
        status (ResponseStatus): The response status.
        data (Optional[Dict[str, Any]]): The acquired data.
        source (str): The source of the data.
        error_message (Optional[str]): Error message if acquisition failed.
        metadata (Dict[str, Any]): Additional metadata about the response.
        headers (Dict[str, str]): Response headers for HTTP responses.
        status_code (Optional[int]): HTTP status code for web requests.
        timestamp (float): Time when the response was created.
    """
    status: ResponseStatus = ResponseStatus.SUCCESS
    data: Optional[Dict[str, Any]] = None
    source: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    status_code: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_success(self) -> bool:
        """Check if the response was successful."""
        return self.status == ResponseStatus.SUCCESS
    
    @property
    def is_error(self) -> bool:
        """Check if the response contains an error."""
        return (
            self.status != ResponseStatus.SUCCESS and 
            self.status != ResponseStatus.PARTIAL
        )
    
    @property
    def should_retry(self) -> bool:
        """Check if the request should be retried."""
        retry_statuses = {
            ResponseStatus.RATE_LIMITED,
            ResponseStatus.TIMEOUT,
            ResponseStatus.SERVER_ERROR
        }
        return self.status in retry_statuses


@dataclass
class StageMetrics:
    """
    Metrics for a single pipeline stage.
    
    Captures performance and operational metrics for a stage execution.
    
    Attributes:
        stage_name (str): Name of the stage.
        start_time (float): Time when stage execution started.
        end_time (float): Time when stage execution ended.
        success (bool): Whether the stage execution succeeded.
        error_message (Optional[str]): Error message if execution failed.
        custom_metrics (Dict[str, Any]): Stage-specific custom metrics.
    """
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Calculate the duration of stage execution."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


@dataclass
class PipelineMetrics:
    """
    Comprehensive metrics for pipeline execution.
    
    Combines metrics from all stages and provides overall pipeline metrics.
    
    Attributes:
        pipeline_name (str): Name of the pipeline.
        start_time (float): Time when pipeline execution started.
        end_time (Optional[float]): Time when pipeline execution ended.
        stages (Dict[str, StageMetrics]): Metrics for each stage.
        error_count (int): Number of errors encountered.
        warning_count (int): Number of warnings encountered.
        resource_usage (Dict[str, Any]): Resource usage metrics.
        custom_metrics (Dict[str, Any]): Pipeline-specific custom metrics.
    """
    pipeline_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Calculate the total duration of pipeline execution."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def successful_stages(self) -> int:
        """Count the number of successful stages."""
        return sum(1 for metrics in self.stages.values() if metrics.success)
    
    @property
    def failed_stages(self) -> int:
        """Count the number of failed stages."""
        return sum(1 for metrics in self.stages.values() if not metrics.success)
    
    @property
    def success_rate(self) -> float:
        """Calculate the stage success rate."""
        if not self.stages:
            return 0.0
        return self.successful_stages / len(self.stages)
    
    def add_stage_metrics(self, metrics: StageMetrics) -> None:
        """Add metrics for a stage."""
        self.stages[metrics.stage_name] = metrics
        if not metrics.success and metrics.error_message:
            self.error_count += 1
    
    def mark_complete(self) -> None:
        """Mark the pipeline execution as complete."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline metrics."""
        return {
            "pipeline_name": self.pipeline_name,
            "duration": self.duration,
            "total_stages": len(self.stages),
            "successful_stages": self.successful_stages,
            "failed_stages": self.failed_stages,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "warning_count": self.warning_count
        }