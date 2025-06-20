"""
Pipeline Context Module.

This module defines the PipelineContext class that manages shared state within a pipeline.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union


class PipelineContext:
    """
    Shared context for pipeline execution with state tracking.
    
    PipelineContext manages the shared state between pipeline stages, including
    data, metrics, errors, and execution tracking. It provides methods for storing
    and retrieving data, tracking stage execution, and collecting metrics.
    
    Attributes:
        data (Dict[str, Any]): The data being processed by the pipeline.
        id (str): Unique identifier for this context instance.
        metadata (Dict[str, Any]): Execution metadata and metrics.
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new pipeline context.
        
        Args:
            initial_data (Optional[Dict[str, Any]]): Initial data to populate the context.
        """
        # Pipeline data
        self.data = initial_data or {}
        self.id = str(uuid.uuid4())
        
        # Execution metadata
        self.metadata: Dict[str, Any] = {
            "pipeline_name": None,
            "start_time": None,
            "end_time": None,
            "current_stage": None,
            "completed_stages": set(),
            "stage_metrics": {},
            "stage_transitions": [],
            "errors": {}
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context data.
        
        Args:
            key (str): The key to retrieve.
            default (Any): Default value if key doesn't exist.
            
        Returns:
            Any: The value associated with the key, or the default value.
        """
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the context data.
        
        Args:
            key (str): The key to set.
            value (Any): The value to store.
        """
        self.data[key] = value
    
    def update(self, values: Dict[str, Any]) -> None:
        """
        Update multiple values in the context data.
        
        Args:
            values (Dict[str, Any]): Dictionary of values to update.
        """
        self.data.update(values)
    
    def has_key(self, key: str) -> bool:
        """
        Check if a key exists in the context data.
        
        Args:
            key (str): The key to check.
            
        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.data
    
    def remove(self, key: str) -> None:
        """
        Remove a key-value pair from the context data.
        
        Args:
            key (str): The key to remove.
        """
        if key in self.data:
            del self.data[key]
    
    def clear(self) -> None:
        """Clear all data in the context."""
        self.data.clear()
    
    def start_pipeline(self, pipeline_name: str) -> None:
        """
        Mark the start of pipeline execution.
        
        Args:
            pipeline_name (str): Name of the executing pipeline.
        """
        self.metadata["pipeline_name"] = pipeline_name
        self.metadata["start_time"] = time.time()
    
    def end_pipeline(self) -> None:
        """Mark the end of pipeline execution."""
        self.metadata["end_time"] = time.time()
    
    def start_stage(self, stage_name: str) -> None:
        """
        Mark the start of a stage's execution.
        
        Args:
            stage_name (str): Name of the current stage.
        """
        self.metadata["current_stage"] = stage_name
        self.metadata["stage_metrics"][stage_name] = {
            "start_time": time.time(),
            "end_time": None,
            "status": "running",
            "execution_time": 0
        }
        
        # Record stage transition
        if self.metadata["stage_transitions"]:
            prev_stage = self.metadata["stage_transitions"][-1]["to"]
            self.metadata["stage_transitions"].append({
                "from": prev_stage,
                "to": stage_name,
                "time": time.time()
            })
        else:
            self.metadata["stage_transitions"].append({
                "from": None,
                "to": stage_name,
                "time": time.time()
            })
    
    def end_stage(self, success: bool = True) -> None:
        """
        Mark the end of a stage's execution.
        
        Args:
            success (bool): Whether the stage completed successfully.
        """
        stage_name = self.metadata["current_stage"]
        if stage_name:
            end_time = time.time()
            self.metadata["completed_stages"].add(stage_name)
            self.metadata["stage_metrics"][stage_name].update({
                "end_time": end_time,
                "status": "success" if success else "failed",
                "execution_time": end_time - self.metadata["stage_metrics"][stage_name]["start_time"]
            })
            self.metadata["current_stage"] = None
    
    def add_error(self, source: str, message: str) -> None:
        """
        Add an error to the context.
        
        Args:
            source (str): Source of the error (e.g., stage name).
            message (str): Error message.
        """
        if source not in self.metadata["errors"]:
            self.metadata["errors"][source] = []
        self.metadata["errors"][source].append(message)
    
    def has_errors(self) -> bool:
        """
        Check if the context has any errors.
        
        Returns:
            bool: True if errors exist, False otherwise.
        """
        return len(self.metadata["errors"]) > 0
    
    def get_errors(self) -> Dict[str, List[str]]:
        """
        Get all errors recorded in the context.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping error sources to lists of error messages.
        """
        return self.metadata["errors"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics for the pipeline.
        
        Returns:
            Dict[str, Any]: Dictionary containing execution metrics.
        """
        total_time = 0
        if self.metadata["start_time"]:
            end_time = self.metadata["end_time"] or time.time()
            total_time = end_time - self.metadata["start_time"]
        
        successful_stages = sum(
            1 for metrics in self.metadata["stage_metrics"].values()
            if metrics["status"] == "success"
        )
        
        return {
            "pipeline_name": self.metadata["pipeline_name"],
            "total_time": total_time,
            "stages": self.metadata["stage_metrics"],
            "successful_stages": successful_stages,
            "total_stages": len(self.metadata["stage_metrics"]),
            "has_errors": self.has_errors(),
            "transitions": self.metadata["stage_transitions"]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the context.
        """
        return {
            "id": self.id,
            "data": self.data,
            "metrics": self.get_metrics(),
            "errors": self.get_errors()
        }