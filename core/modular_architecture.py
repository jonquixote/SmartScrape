"""
Modular Pipeline Architecture Enhancements

This module provides enhanced modular architecture components
for the SmartScrape pipeline system, enabling better component
isolation, interface standardization, and A/B testing capabilities.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Type, Union, Protocol
from enum import Enum
import threading
import inspect

from core.pipeline.context import PipelineContext
from core.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of pipeline components."""
    INTENT_ANALYZER = "intent_analyzer"
    URL_GENERATOR = "url_generator"
    CRAWLER = "crawler"
    SCHEMA_GENERATOR = "schema_generator"
    PROCESSOR = "processor"
    ENHANCER = "enhancer"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"

class ComponentStatus(Enum):
    """Status of pipeline components."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"

@dataclass
class ComponentMetadata:
    """Metadata for pipeline components."""
    name: str
    component_type: ComponentType
    version: str
    description: str
    status: ComponentStatus = ComponentStatus.ACTIVE
    dependencies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class ComponentInterface(Protocol):
    """Protocol defining the interface for pipeline components."""
    
    metadata: ComponentMetadata
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process the pipeline context."""
        ...
    
    async def validate_input(self, context: PipelineContext) -> bool:
        """Validate input context."""
        ...
    
    async def validate_output(self, context: PipelineContext) -> bool:
        """Validate output context."""
        ...
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        ...

class BaseComponent(ABC):
    """
    Base class for all pipeline components.
    
    Provides common functionality including metrics collection,
    health monitoring, and standardized interfaces.
    """
    
    def __init__(self, metadata: ComponentMetadata):
        self.metadata = metadata
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'avg_duration': 0.0,
            'last_called': None
        }
        self._lock = threading.Lock()
    
    @abstractmethod
    async def _process_impl(self, context: PipelineContext) -> PipelineContext:
        """Implementation-specific processing logic."""
        pass
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """Process the pipeline context with metrics collection."""
        start_time = time.time()
        
        try:
            # Validate input
            if not await self.validate_input(context):
                raise ValueError(f"Input validation failed for {self.metadata.name}")
            
            # Process
            result_context = await self._process_impl(context)
            
            # Validate output
            if not await self.validate_output(result_context):
                raise ValueError(f"Output validation failed for {self.metadata.name}")
            
            # Update success metrics
            with self._lock:
                self.metrics['successful_calls'] += 1
                self.metrics['total_calls'] += 1
                duration = time.time() - start_time
                self.metrics['avg_duration'] = (
                    (self.metrics['avg_duration'] * (self.metrics['total_calls'] - 1) + duration) /
                    self.metrics['total_calls']
                )
                self.metrics['last_called'] = datetime.now()
            
            return result_context
            
        except Exception as e:
            with self._lock:
                self.metrics['failed_calls'] += 1
                self.metrics['total_calls'] += 1
            
            logger.error(f"Component {self.metadata.name} failed: {e}")
            raise
    
    async def validate_input(self, context: PipelineContext) -> bool:
        """Default input validation."""
        return context is not None
    
    async def validate_output(self, context: PipelineContext) -> bool:
        """Default output validation."""
        return context is not None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        with self._lock:
            success_rate = (
                self.metrics['successful_calls'] / self.metrics['total_calls']
                if self.metrics['total_calls'] > 0 else 1.0
            )
            
            health_status = "healthy"
            if success_rate < 0.5:
                health_status = "unhealthy"
            elif success_rate < 0.8:
                health_status = "degraded"
            
            return {
                'status': health_status,
                'success_rate': success_rate,
                'metrics': self.metrics.copy(),
                'metadata': {
                    'name': self.metadata.name,
                    'type': self.metadata.component_type.value,
                    'version': self.metadata.version,
                    'status': self.metadata.status.value
                }
            }

class ComponentRegistry:
    """
    Registry for managing pipeline components.
    
    Provides component discovery, version management,
    and A/B testing capabilities.
    """
    
    def __init__(self):
        self.components: Dict[str, Dict[str, BaseComponent]] = {}  # type -> {name: component}
        self.default_components: Dict[str, str] = {}  # type -> default component name
        self.ab_tests: Dict[str, Dict[str, Any]] = {}  # test_id -> test config
        self._lock = threading.Lock()
    
    def register_component(self, component: BaseComponent, is_default: bool = False) -> None:
        """Register a component in the registry."""
        component_type = component.metadata.component_type.value
        component_name = component.metadata.name
        
        with self._lock:
            if component_type not in self.components:
                self.components[component_type] = {}
            
            self.components[component_type][component_name] = component
            
            if is_default or component_type not in self.default_components:
                self.default_components[component_type] = component_name
        
        logger.info(f"Registered component {component_name} of type {component_type}")
    
    def unregister_component(self, component_type: str, component_name: str) -> bool:
        """Unregister a component."""
        with self._lock:
            if (component_type in self.components and 
                component_name in self.components[component_type]):
                
                del self.components[component_type][component_name]
                
                # If this was the default, select a new default
                if self.default_components.get(component_type) == component_name:
                    remaining = list(self.components[component_type].keys())
                    self.default_components[component_type] = remaining[0] if remaining else None
                
                logger.info(f"Unregistered component {component_name} of type {component_type}")
                return True
        
        return False
    
    def get_component(self, component_type: str, component_name: str = None) -> Optional[BaseComponent]:
        """Get a specific component or the default for a type."""
        if component_name is None:
            component_name = self.default_components.get(component_type)
        
        if (component_type in self.components and 
            component_name in self.components[component_type]):
            return self.components[component_type][component_name]
        
        return None
    
    def list_components(self, component_type: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """List all components or components of a specific type."""
        result = {}
        
        types_to_check = [component_type] if component_type else self.components.keys()
        
        for comp_type in types_to_check:
            if comp_type in self.components:
                result[comp_type] = [
                    {
                        'name': name,
                        'metadata': comp.metadata,
                        'health': comp.get_health_status(),
                        'is_default': self.default_components.get(comp_type) == name
                    }
                    for name, comp in self.components[comp_type].items()
                ]
        
        return result
    
    def create_ab_test(self, test_id: str, component_type: str, 
                      variants: Dict[str, str], traffic_split: Dict[str, float]) -> bool:
        """
        Create an A/B test between component variants.
        
        Args:
            test_id: Unique identifier for the test
            component_type: Type of component to test
            variants: Mapping of variant names to component names
            traffic_split: Mapping of variant names to traffic percentages
            
        Returns:
            True if test created successfully
        """
        # Validate variants exist
        for variant_name, component_name in variants.items():
            if not self.get_component(component_type, component_name):
                logger.error(f"Component {component_name} not found for variant {variant_name}")
                return False
        
        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 0.01:
            logger.error("Traffic split must sum to 1.0")
            return False
        
        with self._lock:
            self.ab_tests[test_id] = {
                'component_type': component_type,
                'variants': variants,
                'traffic_split': traffic_split,
                'created_at': datetime.now(),
                'metrics': {variant: {'calls': 0, 'success': 0, 'avg_duration': 0.0} 
                           for variant in variants.keys()}
            }
        
        logger.info(f"Created A/B test {test_id} for {component_type}")
        return True
    
    def get_ab_test_component(self, test_id: str, session_id: str = None) -> Optional[BaseComponent]:
        """Get component for A/B test based on session."""
        if test_id not in self.ab_tests:
            return None
        
        test_config = self.ab_tests[test_id]
        component_type = test_config['component_type']
        variants = test_config['variants']
        traffic_split = test_config['traffic_split']
        
        # Determine variant based on session hash
        if session_id:
            hash_value = hash(session_id) % 100
        else:
            import random
            hash_value = random.randint(0, 99)
        
        cumulative = 0
        for variant, split in traffic_split.items():
            cumulative += split * 100
            if hash_value < cumulative:
                component_name = variants[variant]
                return self.get_component(component_type, component_name)
        
        # Fallback to first variant
        first_variant = list(variants.keys())[0]
        component_name = variants[first_variant]
        return self.get_component(component_type, component_name)
    
    def record_ab_test_metrics(self, test_id: str, variant: str, 
                              success: bool, duration: float) -> None:
        """Record metrics for A/B test variant."""
        if test_id not in self.ab_tests:
            return
        
        with self._lock:
            metrics = self.ab_tests[test_id]['metrics'][variant]
            metrics['calls'] += 1
            if success:
                metrics['success'] += 1
            
            # Update average duration
            metrics['avg_duration'] = (
                (metrics['avg_duration'] * (metrics['calls'] - 1) + duration) /
                metrics['calls']
            )
    
    def get_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results."""
        if test_id not in self.ab_tests:
            return None
        
        test_config = self.ab_tests[test_id].copy()
        
        # Calculate success rates
        for variant, metrics in test_config['metrics'].items():
            if metrics['calls'] > 0:
                metrics['success_rate'] = metrics['success'] / metrics['calls']
            else:
                metrics['success_rate'] = 0.0
        
        return test_config

class ModularPipelineBuilder:
    """
    Builder for creating modular pipelines with component swapping capabilities.
    """
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.pipeline_stages: List[Dict[str, Any]] = []
        self.pipeline_config = {}
    
    def add_stage(self, component_type: str, component_name: str = None, 
                 config: Dict[str, Any] = None) -> 'ModularPipelineBuilder':
        """Add a stage to the pipeline."""
        self.pipeline_stages.append({
            'component_type': component_type,
            'component_name': component_name,
            'config': config or {}
        })
        return self
    
    def add_ab_test_stage(self, test_id: str) -> 'ModularPipelineBuilder':
        """Add an A/B test stage to the pipeline."""
        self.pipeline_stages.append({
            'ab_test_id': test_id
        })
        return self
    
    def set_config(self, config: Dict[str, Any]) -> 'ModularPipelineBuilder':
        """Set pipeline-level configuration."""
        self.pipeline_config.update(config)
        return self
    
    def build(self, session_id: str = None) -> 'ModularPipeline':
        """Build the modular pipeline."""
        return ModularPipeline(
            self.registry,
            self.pipeline_stages.copy(),
            self.pipeline_config.copy(),
            session_id
        )

class ModularPipeline:
    """
    A modular pipeline that supports component swapping and A/B testing.
    """
    
    def __init__(self, registry: ComponentRegistry, stages: List[Dict[str, Any]], 
                 config: Dict[str, Any], session_id: str = None):
        self.registry = registry
        self.stages = stages
        self.config = config
        self.session_id = session_id or str(uuid.uuid4())
        self.execution_id = None
        self.metrics = []
    
    async def execute(self, initial_context: PipelineContext) -> PipelineContext:
        """Execute the modular pipeline."""
        self.execution_id = str(uuid.uuid4())
        context = initial_context
        context.metadata['pipeline_execution_id'] = self.execution_id
        context.metadata['session_id'] = self.session_id
        
        logger.info(f"Starting modular pipeline execution {self.execution_id}")
        
        for i, stage_config in enumerate(self.stages):
            stage_start = time.time()
            
            try:
                # Get component for this stage
                if 'ab_test_id' in stage_config:
                    component = self.registry.get_ab_test_component(
                        stage_config['ab_test_id'], 
                        self.session_id
                    )
                    if not component:
                        raise ValueError(f"No component found for A/B test {stage_config['ab_test_id']}")
                else:
                    component = self.registry.get_component(
                        stage_config['component_type'],
                        stage_config['component_name']
                    )
                    if not component:
                        raise ValueError(f"No component found for type {stage_config['component_type']}")
                
                # Apply stage-specific config
                stage_context = context.copy()
                stage_context.config.update(stage_config.get('config', {}))
                
                # Execute component
                context = await component.process(stage_context)
                
                # Record metrics
                stage_duration = time.time() - stage_start
                stage_metrics = {
                    'stage': i,
                    'component_name': component.metadata.name,
                    'component_type': component.metadata.component_type.value,
                    'duration': stage_duration,
                    'success': True
                }
                
                # Record A/B test metrics if applicable
                if 'ab_test_id' in stage_config:
                    test_id = stage_config['ab_test_id']
                    test_config = self.registry.ab_tests.get(test_id)
                    if test_config:
                        # Find which variant was used
                        for variant, comp_name in test_config['variants'].items():
                            if comp_name == component.metadata.name:
                                self.registry.record_ab_test_metrics(
                                    test_id, variant, True, stage_duration
                                )
                                stage_metrics['ab_test_variant'] = variant
                                break
                
                self.metrics.append(stage_metrics)
                
                logger.debug(f"Completed stage {i} with {component.metadata.name}")
                
            except Exception as e:
                stage_duration = time.time() - stage_start
                
                # Record failure metrics
                stage_metrics = {
                    'stage': i,
                    'component_name': stage_config.get('component_name', 'unknown'),
                    'component_type': stage_config.get('component_type', 'unknown'),
                    'duration': stage_duration,
                    'success': False,
                    'error': str(e)
                }
                self.metrics.append(stage_metrics)
                
                # Record A/B test failure if applicable
                if 'ab_test_id' in stage_config:
                    test_id = stage_config['ab_test_id']
                    # This is a simplified approach - in practice you'd want to track which variant failed
                    self.registry.record_ab_test_metrics(test_id, 'unknown', False, stage_duration)
                
                logger.error(f"Stage {i} failed: {e}")
                raise
        
        logger.info(f"Completed modular pipeline execution {self.execution_id}")
        return context
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this pipeline run."""
        total_duration = sum(m['duration'] for m in self.metrics)
        successful_stages = sum(1 for m in self.metrics if m['success'])
        
        return {
            'execution_id': self.execution_id,
            'session_id': self.session_id,
            'total_stages': len(self.stages),
            'successful_stages': successful_stages,
            'total_duration': total_duration,
            'success_rate': successful_stages / len(self.stages) if self.stages else 1.0,
            'stage_metrics': self.metrics
        }


# Global component registry
_component_registry: Optional[ComponentRegistry] = None

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry."""
    global _component_registry
    if _component_registry is None:
        _component_registry = ComponentRegistry()
    return _component_registry

def create_pipeline_builder() -> ModularPipelineBuilder:
    """Create a new modular pipeline builder."""
    return ModularPipelineBuilder(get_component_registry())
