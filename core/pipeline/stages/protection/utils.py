"""
Protection Utilities Module.

This module provides factory functions and utilities for easily applying
protection patterns to pipeline stages.
"""

from typing import Any, Dict, List, Optional, Union

from core.pipeline.stage import PipelineStage
from core.pipeline.stages.protection.circuit_breaker_stage import CircuitBreakerStage
from core.pipeline.stages.protection.bulkhead_stage import BulkheadStage
from core.pipeline.stages.protection.rate_limiting import RateLimitingStage


def with_circuit_breaker(
    stage: PipelineStage,
    failure_threshold: int = 5,
    reset_timeout: int = 60,
    name: Optional[str] = None,
    use_registry: bool = True,
    half_open_max_calls: int = 1,
    fallback_enabled: bool = False,
    fallback_data: Optional[Dict[str, Any]] = None
) -> CircuitBreakerStage:
    """
    Wrap a pipeline stage with circuit breaker protection.
    
    Args:
        stage: The stage to protect with a circuit breaker
        failure_threshold: Number of failures before opening the circuit
        reset_timeout: Seconds to wait before attempting to half-open the circuit
        name: Optional name for the circuit breaker stage
        use_registry: Whether to use the global circuit breaker registry
        half_open_max_calls: Number of test calls to allow in half-open state
        fallback_enabled: Whether to use fallback behavior when circuit is open
        fallback_data: Data to use for fallback when circuit is open
        
    Returns:
        CircuitBreakerStage: The protected stage
    """
    config = {
        "failure_threshold": failure_threshold,
        "reset_timeout": reset_timeout,
        "use_registry": use_registry,
        "half_open_max_calls": half_open_max_calls,
        "fallback_enabled": fallback_enabled
    }
    
    if fallback_data:
        config["fallback_data"] = fallback_data
        
    return CircuitBreakerStage(
        wrapped_stage=stage,
        name=name,
        config=config
    )


def with_bulkhead(
    stage: PipelineStage,
    max_concurrent_executions: int = 10,
    max_queue_size: int = 20,
    execution_timeout_seconds: float = 30.0,
    queue_timeout_seconds: float = 10.0,
    name: Optional[str] = None
) -> BulkheadStage:
    """
    Wrap a pipeline stage with bulkhead protection.
    
    Args:
        stage: The stage to protect with a bulkhead
        max_concurrent_executions: Maximum number of concurrent executions allowed
        max_queue_size: Maximum size of the pending executions queue
        execution_timeout_seconds: Maximum time to wait for execution to complete
        queue_timeout_seconds: Maximum time to wait in the queue
        name: Optional name for the bulkhead stage
        
    Returns:
        BulkheadStage: The protected stage
    """
    config = {
        "max_concurrent_executions": max_concurrent_executions,
        "max_queue_size": max_queue_size,
        "execution_timeout_seconds": execution_timeout_seconds,
        "queue_timeout_seconds": queue_timeout_seconds
    }
    
    return BulkheadStage(
        wrapped_stage=stage,
        name=name,
        config=config
    )


def with_rate_limiting(
    stage: PipelineStage,
    mode: str = "domain",
    global_rate: float = 1.0,
    domain_rates: Optional[Dict[str, float]] = None,
    default_domain_rate: float = 0.5,
    name: Optional[str] = None
) -> PipelineStage:
    """
    Wrap a pipeline stage with rate limiting protection.
    
    This function creates a pipeline with a RateLimitingStage followed by the provided stage.
    
    Args:
        stage: The stage to protect with rate limiting
        mode: Rate limiting mode ("domain", "global", or "both")
        global_rate: Global rate limit (requests per second)
        domain_rates: Dictionary of domain-specific rate limits
        default_domain_rate: Default rate limit for domains not in domain_rates
        name: Optional name for the rate limiting stage
        
    Returns:
        PipelineStage: A stage that performs rate limiting before executing the provided stage
    """
    # TODO: Implement this function by creating a composite stage that uses RateLimitingStage
    # followed by the provided stage. This requires creating a CompositeStage class first.
    raise NotImplementedError("Rate limiting stage integration not yet implemented")


def with_combined_protection(
    stage: PipelineStage,
    circuit_breaker_config: Optional[Dict[str, Any]] = None,
    bulkhead_config: Optional[Dict[str, Any]] = None,
    name_prefix: Optional[str] = None
) -> PipelineStage:
    """
    Apply multiple protection patterns to a pipeline stage.
    
    This applies both circuit breaker and bulkhead protection.
    The bulkhead is applied first, then the circuit breaker.
    
    Args:
        stage: The stage to protect
        circuit_breaker_config: Configuration for the circuit breaker
        bulkhead_config: Configuration for the bulkhead
        name_prefix: Optional prefix for the stage names
        
    Returns:
        PipelineStage: The protected stage
    """
    prefix = name_prefix or ""
    
    # Apply bulkhead first (inner protection)
    if bulkhead_config:
        bulkhead_name = f"{prefix}bulkhead_{stage.name}" if prefix else None
        stage = with_bulkhead(
            stage=stage,
            name=bulkhead_name,
            **bulkhead_config
        )
    
    # Apply circuit breaker second (outer protection)
    if circuit_breaker_config:
        circuit_breaker_name = f"{prefix}circuit_breaker_{stage.name}" if prefix else None
        stage = with_circuit_breaker(
            stage=stage,
            name=circuit_breaker_name,
            **circuit_breaker_config
        )
    
    return stage