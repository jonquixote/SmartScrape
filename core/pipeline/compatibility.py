"""
Pipeline Compatibility Layer

This module provides compatibility layers to facilitate gradual migration to
the pipeline architecture. It includes feature flags, fallback mechanisms,
performance comparisons, and A/B testing support.
"""

import os
import time
import json
import asyncio
import logging
import contextlib
import random
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar, Generator

from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext
from core.pipeline.registry import PipelineRegistry

# Type variables for generic function signatures
T = TypeVar('T')
R = TypeVar('R')

# Configure logging
logger = logging.getLogger(__name__)
metrics_logger = logging.getLogger("pipeline_metrics")

# Default configuration with all feature flags
DEFAULT_CONFIG = {
    # Master switch for pipeline architecture
    "use_pipeline_architecture": True,
    
    # Enable/disable specific pipeline components
    "pipeline_components": {
        "extraction": True,
        "validation": True,
        "normalization": True
    },
    
    # Fallback settings
    "fallback_to_legacy": True,
    "max_fallback_attempts": 3,
    
    # Comparative execution
    "compare_implementations": False,
    "compare_results": False,
    "log_discrepancies": True,
    
    # Performance monitoring
    "collect_performance_metrics": True,
    "performance_threshold": 1.2,  # 20% slower is acceptable
    
    # A/B testing
    "enable_ab_testing": False,
    "ab_test_ratio": 0.5,  # 50% pipeline, 50% legacy
    
    # Controls percentage of traffic using pipeline
    "pipeline_rollout_percentage": 100,
    
    # Monitoring and metrics
    "metrics_output_path": "pipeline_metrics.json",
    "monitor_memory_usage": False
}


class FeatureFlags:
    """
    Feature flags for controlling pipeline usage.
    
    This class provides a centralized way to manage feature flags for
    the pipeline architecture, allowing for gradual rollout.
    
    Example:
        ```python
        # Check if a feature is enabled
        if FeatureFlags.is_enabled("use_pipeline_architecture"):
            # Use pipeline implementation
            result = pipeline.execute(context)
        else:
            # Use legacy implementation
            result = legacy_component.process(data)
        ```
    """
    
    _config = DEFAULT_CONFIG.copy()
    
    @classmethod
    def initialize(cls, config: Dict[str, Any]) -> None:
        """
        Initialize feature flags with a custom configuration.
        
        Args:
            config: Custom configuration dictionary
        """
        cls._config = DEFAULT_CONFIG.copy()
        cls._config.update(config)
        
    @classmethod
    def is_enabled(cls, feature_name: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if the feature is enabled, False otherwise
        """
        # Handle nested features like "pipeline_components.extraction"
        if "." in feature_name:
            parts = feature_name.split(".")
            config = cls._config
            for part in parts:
                if part not in config:
                    return False
                config = config[part]
            return bool(config)
            
        # Handle top-level features
        return bool(cls._config.get(feature_name, False))
        
    @classmethod
    def should_use_pipeline(cls, component_name: str = None) -> bool:
        """
        Determine if the pipeline should be used based on configuration.
        
        This method considers the master switch, component-specific flags,
        and rollout percentage.
        
        Args:
            component_name: Optional component name to check specific flag
            
        Returns:
            True if the pipeline should be used, False otherwise
        """
        # Check master switch
        if not cls.is_enabled("use_pipeline_architecture"):
            return False
            
        # Check rollout percentage
        rollout_pct = cls._config.get("pipeline_rollout_percentage", 100)
        if rollout_pct < 100 and random.random() * 100 > rollout_pct:
            return False
            
        # Check component-specific flag if provided
        if component_name and "pipeline_components" in cls._config:
            return cls._config["pipeline_components"].get(component_name, True)
            
        return True
        
    @classmethod
    def should_compare_implementations(cls) -> bool:
        """
        Check if implementations should be compared.
        
        Returns:
            True if implementations should be compared, False otherwise
        """
        return cls.is_enabled("compare_implementations")
        
    @classmethod
    def should_fallback_to_legacy(cls) -> bool:
        """
        Check if fallback to legacy is enabled.
        
        Returns:
            True if fallback is enabled, False otherwise
        """
        return cls.is_enabled("fallback_to_legacy")
        
    @classmethod
    def get_flag(cls, flag_name: str, default: Any = None) -> Any:
        """
        Get the value of a specific flag.
        
        Args:
            flag_name: Name of the flag to get
            default: Default value if flag doesn't exist
            
        Returns:
            Flag value or default
        """
        # Handle nested flags
        if "." in flag_name:
            parts = flag_name.split(".")
            config = cls._config
            for part in parts[:-1]:
                if part not in config:
                    return default
                config = config[part]
            return config.get(parts[-1], default)
            
        return cls._config.get(flag_name, default)
        
    @classmethod
    def get_all_flags(cls) -> Dict[str, Any]:
        """
        Get all feature flags.
        
        Returns:
            Dictionary with all feature flags
        """
        return cls._config.copy()


class PerformanceMonitor:
    """
    Monitors and compares performance of pipeline vs legacy implementations.
    
    This class provides utilities for measuring execution time, memory usage,
    and other performance metrics for comparison.
    
    Example:
        ```python
        # Compare performance of two implementations
        with PerformanceMonitor.compare("extraction") as monitor:
            # Track legacy implementation
            with monitor.track("legacy"):
                legacy_result = legacy_extractor.extract(html, url)
                
            # Track pipeline implementation
            with monitor.track("pipeline"):
                pipeline_result = await pipeline.execute({
                    "html_content": html, 
                    "url": url
                })
        ```
    """
    
    _metrics = {
        "comparisons": [],
        "summary": {
            "legacy_faster_count": 0,
            "pipeline_faster_count": 0,
            "total_comparisons": 0
        }
    }
    
    @classmethod
    @contextlib.contextmanager
    def compare(cls, operation_name: str) -> Generator["PerformanceComparison", None, None]:
        """
        Context manager for comparing implementations.
        
        Args:
            operation_name: Name of the operation being compared
            
        Yields:
            PerformanceComparison object
        """
        comparison = PerformanceComparison(operation_name)
        try:
            yield comparison
        finally:
            # Record the comparison
            if FeatureFlags.is_enabled("collect_performance_metrics"):
                cls._record_comparison(comparison)
                
    @classmethod
    @contextlib.contextmanager
    def measure(cls, operation_name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for measuring a single operation.
        
        Args:
            operation_name: Name of the operation being measured
            
        Yields:
            Dictionary to store measurement results
        """
        metrics = {"name": operation_name, "start_time": time.time()}
        try:
            yield metrics
        finally:
            metrics["end_time"] = time.time()
            metrics["duration"] = metrics["end_time"] - metrics["start_time"]
            
    @classmethod
    def _record_comparison(cls, comparison: "PerformanceComparison") -> None:
        """
        Record a performance comparison.
        
        Args:
            comparison: Comparison object with results
        """
        # Create comparison record
        record = {
            "operation": comparison.operation_name,
            "timestamp": datetime.now().isoformat(),
            "measurements": comparison.measurements,
            "summary": comparison.get_summary()
        }
        
        # Update metrics
        cls._metrics["comparisons"].append(record)
        cls._metrics["summary"]["total_comparisons"] += 1
        
        if record["summary"].get("legacy_faster", False):
            cls._metrics["summary"]["legacy_faster_count"] += 1
        elif record["summary"].get("pipeline_faster", False):
            cls._metrics["summary"]["pipeline_faster_count"] += 1
            
        # Log the comparison
        metrics_logger.info(
            f"Performance comparison for {comparison.operation_name}: "
            f"{json.dumps(record['summary'])}"
        )
        
        # Save metrics if configured
        if FeatureFlags.is_enabled("collect_performance_metrics"):
            output_path = FeatureFlags.get_flag("metrics_output_path")
            if output_path:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(cls._metrics, f, indent=2)
                except Exception as e:
                    logger.error(f"Failed to save metrics: {e}")
                    
    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return cls._metrics.copy()
        
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of performance comparisons.
        
        Returns:
            Dictionary with summary metrics
        """
        total = cls._metrics["summary"]["total_comparisons"] or 1  # Avoid division by zero
        return {
            "total_comparisons": cls._metrics["summary"]["total_comparisons"],
            "legacy_faster_count": cls._metrics["summary"]["legacy_faster_count"],
            "pipeline_faster_count": cls._metrics["summary"]["pipeline_faster_count"],
            "legacy_faster_percentage": cls._metrics["summary"]["legacy_faster_count"] / total * 100,
            "pipeline_faster_percentage": cls._metrics["summary"]["pipeline_faster_count"] / total * 100
        }


class PerformanceComparison:
    """
    Helper class for comparing performance of different implementations.
    """
    
    def __init__(self, operation_name: str):
        """
        Initialize a performance comparison.
        
        Args:
            operation_name: Name of the operation being compared
        """
        self.operation_name = operation_name
        self.measurements = {}
        
    @contextlib.contextmanager
    def track(self, implementation_name: str) -> Generator[None, None, None]:
        """
        Context manager for tracking an implementation.
        
        Args:
            implementation_name: Name of the implementation being tracked
        """
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.measurements[implementation_name] = {
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            }
            
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the comparison.
        
        Returns:
            Dictionary with comparison summary
        """
        if len(self.measurements) < 2:
            return {"error": "Not enough measurements for comparison"}
            
        legacy_time = self.measurements.get("legacy", {}).get("duration", 0)
        pipeline_time = self.measurements.get("pipeline", {}).get("duration", 0)
        
        if legacy_time <= 0 or pipeline_time <= 0:
            return {"error": "Invalid measurements"}
            
        ratio = pipeline_time / legacy_time if legacy_time > 0 else 0
        threshold = FeatureFlags.get_flag("performance_threshold", 1.2)
        
        return {
            "legacy_time": legacy_time,
            "pipeline_time": pipeline_time,
            "ratio": ratio,
            "pipeline_overhead": (ratio - 1) * 100,
            "legacy_faster": ratio > threshold,
            "pipeline_faster": ratio < 1.0,
            "within_threshold": ratio <= threshold
        }


class ABTestSelector:
    """
    Selects between implementations for A/B testing.
    
    This class provides utilities for performing A/B tests between
    pipeline and legacy implementations.
    
    Example:
        ```python
        # Determine which implementation to use
        selector = ABTestSelector("extraction")
        if selector.use_pipeline():
            result = await pipeline.execute(context)
        else:
            result = legacy_extractor.extract(html, url)
        ```
    """
    
    def __init__(self, component_name: str = None):
        """
        Initialize the A/B test selector.
        
        Args:
            component_name: Optional component name for specific flags
        """
        self.component_name = component_name
        self.implementation = self._select_implementation()
        
    def _select_implementation(self) -> str:
        """
        Select an implementation based on configuration.
        
        Returns:
            "pipeline" or "legacy"
        """
        # If A/B testing is not enabled, use feature flags
        if not FeatureFlags.is_enabled("enable_ab_testing"):
            return "pipeline" if FeatureFlags.should_use_pipeline(self.component_name) else "legacy"
            
        # Perform A/B test selection
        ratio = FeatureFlags.get_flag("ab_test_ratio", 0.5)
        return "pipeline" if random.random() < ratio else "legacy"
        
    def use_pipeline(self) -> bool:
        """
        Check if the pipeline implementation should be used.
        
        Returns:
            True if the pipeline should be used, False otherwise
        """
        return self.implementation == "pipeline"
        
    def record_result(self, success: bool, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record the result of using the selected implementation.
        
        Args:
            success: Whether the implementation succeeded
            metrics: Optional metrics about the execution
        """
        # Skip if not collecting metrics
        if not FeatureFlags.is_enabled("collect_performance_metrics"):
            return
            
        # Create result record
        record = {
            "component": self.component_name,
            "implementation": self.implementation,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        if metrics:
            record["metrics"] = metrics
            
        # Log the result
        metrics_logger.info(f"AB test result: {json.dumps(record)}")


class FallbackExecutor:
    """
    Executes with fallback to legacy implementation.
    
    This class provides utilities for executing a pipeline with fallback
    to legacy implementation if the pipeline fails.
    
    Example:
        ```python
        # Execute with fallback
        executor = FallbackExecutor(
            pipeline_func=lambda: pipeline.execute(context),
            legacy_func=lambda: legacy_extractor.extract(html, url),
            component_name="extraction"
        )
        result = await executor.execute()
        ```
    """
    
    def __init__(self, 
                pipeline_func: Callable[[], T], 
                legacy_func: Callable[[], T],
                component_name: str = None):
        """
        Initialize the fallback executor.
        
        Args:
            pipeline_func: Function to execute the pipeline implementation
            legacy_func: Function to execute the legacy implementation
            component_name: Optional component name for metrics
        """
        self.pipeline_func = pipeline_func
        self.legacy_func = legacy_func
        self.component_name = component_name
        
    async def execute(self) -> Tuple[T, Dict[str, Any]]:
        """
        Execute with fallback if needed.
        
        Returns:
            Tuple of (result, execution_info)
        """
        # Check if we should use pipeline at all
        if not FeatureFlags.should_use_pipeline(self.component_name):
            # Just use legacy
            start_time = time.time()
            result = await self._execute_func(self.legacy_func)
            end_time = time.time()
            
            return result, {
                "implementation": "legacy",
                "fallback_occurred": False,
                "execution_time": end_time - start_time
            }
            
        # Try pipeline first
        start_time = time.time()
        try:
            result = await self._execute_func(self.pipeline_func)
            end_time = time.time()
            
            return result, {
                "implementation": "pipeline",
                "fallback_occurred": False,
                "execution_time": end_time - start_time
            }
        except Exception as e:
            pipeline_error = str(e)
            logger.warning(
                f"Pipeline execution failed for {self.component_name}: {pipeline_error}"
            )
            
            # Check if fallback is enabled
            if not FeatureFlags.should_fallback_to_legacy():
                # Re-raise the exception
                raise
                
            # Fallback to legacy
            logger.info(f"Falling back to legacy implementation for {self.component_name}")
            try:
                result = await self._execute_func(self.legacy_func)
                end_time = time.time()
                
                return result, {
                    "implementation": "legacy",
                    "fallback_occurred": True,
                    "pipeline_error": pipeline_error,
                    "execution_time": end_time - start_time
                }
            except Exception as fallback_error:
                # Both implementations failed
                logger.error(
                    f"Both pipeline and legacy implementations failed for {self.component_name}: "
                    f"Pipeline error: {pipeline_error}, Legacy error: {str(fallback_error)}"
                )
                # Re-raise the original error
                raise
                
    async def _execute_func(self, func: Callable[[], T]) -> T:
        """
        Execute a function, handling both async and sync functions.
        
        Args:
            func: Function to execute
            
        Returns:
            Function result
        """
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            result = func()
            if asyncio.iscoroutine(result):
                return await result
            return result


class ResultComparator:
    """
    Compares results from pipeline and legacy implementations.
    
    This class provides utilities for comparing results and logging discrepancies.
    
    Example:
        ```python
        # Compare results
        comparator = ResultComparator(component_name="extraction")
        comparator.compare(
            pipeline_result=pipeline_result,
            legacy_result=legacy_result,
            context={"url": "https://example.com"}
        )
        ```
    """
    
    def __init__(self, component_name: str = None):
        """
        Initialize the result comparator.
        
        Args:
            component_name: Optional component name for metrics
        """
        self.component_name = component_name
        
    def compare(self, 
               pipeline_result: Any, 
               legacy_result: Any, 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare results from pipeline and legacy implementations.
        
        Args:
            pipeline_result: Result from pipeline implementation
            legacy_result: Result from legacy implementation
            context: Optional context for debugging (e.g., URL)
            
        Returns:
            Dictionary with comparison results
        """
        # Skip if not logging discrepancies
        if not FeatureFlags.is_enabled("log_discrepancies"):
            return {"compared": False}
            
        comparison_result = {
            "compared": True,
            "equivalent": self._are_equivalent(pipeline_result, legacy_result),
            "context": context or {},
            "component": self.component_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # If not equivalent, add details
        if not comparison_result["equivalent"]:
            comparison_result["differences"] = self._find_differences(
                pipeline_result, legacy_result
            )
            
            # Log discrepancies
            metrics_logger.warning(
                f"Result discrepancy in {self.component_name}: "
                f"{json.dumps(comparison_result['differences'])}"
            )
            
        return comparison_result
        
    def _are_equivalent(self, a: Any, b: Any) -> bool:
        """
        Check if two results are functionally equivalent.
        
        Args:
            a: First result
            b: Second result
            
        Returns:
            True if equivalent, False otherwise
        """
        # Check simple equality first
        if a == b:
            return True
            
        # Handle None
        if a is None or b is None:
            return False
            
        # Handle dictionaries
        if isinstance(a, dict) and isinstance(b, dict):
            # Ignore success/status indicators if present
            a_filtered = {k: v for k, v in a.items() if k not in {"success", "status"}}
            b_filtered = {k: v for k, v in b.items() if k not in {"success", "status"}}
            
            # Check essential data keys
            return self._compare_essential_keys(a_filtered, b_filtered)
            
        # Handle lists
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
                
            # For simple lists, check set equality
            if all(not isinstance(x, (dict, list)) for x in a + b):
                return set(a) == set(b)
                
            # For complex lists, check items one by one
            return all(self._are_equivalent(a_item, b_item) for a_item, b_item in zip(a, b))
            
        # Default to equality
        return a == b
        
    def _compare_essential_keys(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """
        Compare essential keys in two dictionaries.
        
        Args:
            a: First dictionary
            b: Second dictionary
            
        Returns:
            True if essential keys are equivalent, False otherwise
        """
        # Find common essential keys
        a_keys = set(k for k in a.keys() if not k.startswith("_"))
        b_keys = set(k for k in b.keys() if not k.startswith("_"))
        
        # All essential keys should be present in both
        if not (a_keys <= b_keys and b_keys <= a_keys):
            return False
            
        # Check values for essential keys
        for key in a_keys:
            if not self._are_equivalent(a[key], b[key]):
                return False
                
        return True
        
    def _find_differences(self, a: Any, b: Any) -> List[Dict[str, Any]]:
        """
        Find differences between two results.
        
        Args:
            a: Pipeline result
            b: Legacy result
            
        Returns:
            List of differences
        """
        differences = []
        
        # Handle dictionaries
        if isinstance(a, dict) and isinstance(b, dict):
            all_keys = set(a.keys()) | set(b.keys())
            
            for key in all_keys:
                # Check for missing keys
                if key not in a:
                    differences.append({
                        "key": key,
                        "pipeline": "<missing>",
                        "legacy": str(b[key])[:100]
                    })
                elif key not in b:
                    differences.append({
                        "key": key,
                        "pipeline": str(a[key])[:100],
                        "legacy": "<missing>"
                    })
                elif not self._are_equivalent(a[key], b[key]):
                    # Values differ
                    differences.append({
                        "key": key,
                        "pipeline": str(a[key])[:100],
                        "legacy": str(b[key])[:100]
                    })
                    
        # Handle lists
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append({
                    "key": "list_length",
                    "pipeline": len(a),
                    "legacy": len(b)
                })
            else:
                for i, (a_item, b_item) in enumerate(zip(a, b)):
                    if not self._are_equivalent(a_item, b_item):
                        differences.append({
                            "key": f"list_item_{i}",
                            "pipeline": str(a_item)[:100],
                            "legacy": str(b_item)[:100]
                        })
                        
        # Handle primitives
        elif a != b:
            differences.append({
                "key": "value",
                "pipeline": str(a)[:100],
                "legacy": str(b)[:100]
            })
            
        return differences


# Decorators for compatibility layer
def with_fallback(component_name: str = None):
    """
    Decorator for executing a function with fallback to a legacy function.
    
    Args:
        component_name: Optional component name for metrics
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # If pipeline is disabled, just execute the function
            if not FeatureFlags.should_use_pipeline(component_name):
                return await func(*args, **kwargs)
                
            # Get legacy function from kwargs or use default
            legacy_func = kwargs.pop("legacy_func", None)
            if not legacy_func:
                return await func(*args, **kwargs)
                
            # Create fallback executor
            executor = FallbackExecutor(
                pipeline_func=lambda: func(*args, **kwargs),
                legacy_func=lambda: legacy_func(*args, **kwargs),
                component_name=component_name
            )
            
            # Execute with fallback
            result, info = await executor.execute()
            
            # Record metrics if enabled
            if FeatureFlags.is_enabled("collect_performance_metrics"):
                metrics_logger.info(
                    f"Fallback execution {component_name}: {json.dumps(info)}"
                )
                
            return result
        return wrapper
    return decorator


def with_performance_comparison(component_name: str = None):
    """
    Decorator for comparing performance of pipeline vs legacy implementations.
    
    Args:
        component_name: Optional component name for metrics
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # If comparison is disabled, just execute the function
            if not FeatureFlags.should_compare_implementations():
                return await func(*args, **kwargs)
                
            # Get legacy function from kwargs or use default
            legacy_func = kwargs.pop("legacy_func", None)
            if not legacy_func:
                return await func(*args, **kwargs)
                
            # Compare implementations
            with PerformanceMonitor.compare(component_name or func.__name__) as monitor:
                # Track legacy implementation
                with monitor.track("legacy"):
                    legacy_result = await legacy_func(*args, **kwargs)
                    
                # Track pipeline implementation
                with monitor.track("pipeline"):
                    pipeline_result = await func(*args, **kwargs)
                    
            # Compare results if needed
            if FeatureFlags.is_enabled("compare_results"):
                comparator = ResultComparator(component_name or func.__name__)
                comparator.compare(
                    pipeline_result=pipeline_result,
                    legacy_result=legacy_result,
                    context=kwargs
                )
                
            # Return the appropriate result
            # By default, return pipeline result
            return pipeline_result
        return wrapper
    return decorator


def with_ab_testing(component_name: str = None):
    """
    Decorator for A/B testing between pipeline and legacy implementations.
    
    Args:
        component_name: Optional component name for metrics
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # If A/B testing is disabled, just execute the function
            if not FeatureFlags.is_enabled("enable_ab_testing"):
                return await func(*args, **kwargs)
                
            # Get legacy function from kwargs or use default
            legacy_func = kwargs.pop("legacy_func", None)
            if not legacy_func:
                return await func(*args, **kwargs)
                
            # Create A/B test selector
            selector = ABTestSelector(component_name or func.__name__)
            
            try:
                start_time = time.time()
                if selector.use_pipeline():
                    # Use pipeline implementation
                    result = await func(*args, **kwargs)
                    implementation = "pipeline"
                else:
                    # Use legacy implementation
                    result = await legacy_func(*args, **kwargs)
                    implementation = "legacy"
                end_time = time.time()
                
                # Record success
                selector.record_result(
                    success=True,
                    metrics={
                        "execution_time": end_time - start_time,
                        "implementation": implementation
                    }
                )
                
                return result
            except Exception as e:
                # Record failure
                selector.record_result(
                    success=False,
                    metrics={"error": str(e)}
                )
                raise
        return wrapper
    return decorator


# Compatibility functions for controllers
async def execute_with_compatibility(
    pipeline_func: Callable[[], T],
    legacy_func: Callable[[], T],
    component_name: str = None
) -> T:
    """
    Execute with appropriate compatibility layer based on configuration.
    
    This function selects the appropriate execution method based on feature flags.
    
    Args:
        pipeline_func: Function to execute the pipeline implementation
        legacy_func: Function to execute the legacy implementation
        component_name: Optional component name for metrics
        
    Returns:
        Function result
    """
    # If A/B testing is enabled, use that
    if FeatureFlags.is_enabled("enable_ab_testing"):
        selector = ABTestSelector(component_name)
        if selector.use_pipeline():
            return await pipeline_func()
        else:
            return await legacy_func()
            
    # If comparison is enabled, use that
    if FeatureFlags.should_compare_implementations():
        with PerformanceMonitor.compare(component_name or "execution") as monitor:
            # Track legacy implementation
            with monitor.track("legacy"):
                legacy_result = await legacy_func()
                
            # Track pipeline implementation
            with monitor.track("pipeline"):
                pipeline_result = await pipeline_func()
                
        # Compare results if needed
        if FeatureFlags.is_enabled("compare_results"):
            comparator = ResultComparator(component_name)
            comparator.compare(
                pipeline_result=pipeline_result,
                legacy_result=legacy_result
            )
            
        # Return pipeline result by default
        return pipeline_result
        
    # Use pipeline with fallback
    if FeatureFlags.should_use_pipeline(component_name):
        executor = FallbackExecutor(
            pipeline_func=pipeline_func,
            legacy_func=legacy_func,
            component_name=component_name
        )
        result, _ = await executor.execute()
        return result
        
    # Just use legacy
    return await legacy_func()