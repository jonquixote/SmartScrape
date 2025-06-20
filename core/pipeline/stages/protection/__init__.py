"""
Pipeline Protection Stages Package.

This package provides stages that implement reliability patterns to improve 
resilience, stability, and fault tolerance in pipeline processing.
"""

from core.pipeline.stages.protection.circuit_breaker_stage import CircuitBreakerStage
from core.pipeline.stages.protection.bulkhead_stage import BulkheadStage
from core.pipeline.stages.protection.rate_limiting import RateLimitingStage
from core.pipeline.stages.protection.utils import (
    with_circuit_breaker,
    with_bulkhead,
    with_combined_protection
)

__all__ = [
    'CircuitBreakerStage',
    'BulkheadStage',
    'RateLimitingStage',
    'with_circuit_breaker',
    'with_bulkhead',
    'with_combined_protection'
]