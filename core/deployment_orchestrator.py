"""
Phase 10 Integration Layer

This module integrates all Phase 10 components including progressive rollout,
performance optimization, and modular architecture enhancements.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from core.rollout_manager import (
    ProgressiveRolloutManager, 
    get_rollout_manager, 
    SMARTSCRAPE_ROLLOUT_CONFIGS,
    RolloutPhase
)
from core.performance_optimizer import (
    PerformanceOptimizer, 
    get_performance_optimizer,
    ContentChangeDetector,
    StopConditionManager,
    BatchProcessor
)
from core.modular_architecture import (
    ComponentRegistry,
    ModularPipelineBuilder,
    get_component_registry,
    create_pipeline_builder,
    BaseComponent,
    ComponentMetadata,
    ComponentType,
    ComponentStatus
)
from core.pipeline.compatibility import FeatureFlags
from core.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

@dataclass
class Phase10Status:
    """Status tracking for Phase 10 implementation."""
    rollout_manager_active: bool = False
    performance_optimizer_active: bool = False
    modular_architecture_active: bool = False
    feature_flags_configured: bool = False
    active_rollouts: List[str] = None
    optimization_metrics: Dict[str, Any] = None
    component_health: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.active_rollouts is None:
            self.active_rollouts = []
        if self.optimization_metrics is None:
            self.optimization_metrics = {}
        if self.component_health is None:
            self.component_health = {}

class Phase10Manager:
    """
    Main coordinator for Phase 10: Gradual Rollout and Optimization.
    
    This class orchestrates the progressive rollout of features,
    performance optimization, and modular architecture improvements.
    """
    
    def __init__(self):
        self.rollout_manager = get_rollout_manager()
        self.performance_optimizer = get_performance_optimizer()
        self.component_registry = get_component_registry()
        self.active_rollouts: Dict[str, str] = {}  # feature_name -> rollout_id
        self.status = Phase10Status()
        
        # Initialize Phase 10 systems
        self._initialize_phase10()
    
    def _initialize_phase10(self) -> None:
        """Initialize all Phase 10 systems."""
        try:
            # Configure enhanced feature flags for Phase 10
            self._configure_enhanced_feature_flags()
            
            # Initialize performance optimization features
            self._initialize_performance_features()
            
            # Set up modular architecture
            self._initialize_modular_architecture()
            
            # Update status
            self.status.rollout_manager_active = True
            self.status.performance_optimizer_active = True
            self.status.modular_architecture_active = True
            self.status.feature_flags_configured = True
            
            logger.info("Phase 10 systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Phase 10 systems: {e}")
            raise
    
    def _configure_enhanced_feature_flags(self) -> None:
        """Configure enhanced feature flags for Phase 10."""
        phase10_flags = {
            # Master switches for Phase 10 features
            "phase10_enabled": True,
            "progressive_rollout_enabled": True,
            "performance_optimization_enabled": True,
            "modular_architecture_enabled": True,
            
            # Semantic Intent Analysis
            "semantic_intent_analysis_enabled": False,
            "semantic_intent_analysis_rollout_percentage": 0,
            "semantic_intent_analysis_cache_enabled": True,
            "semantic_intent_analysis_batch_size": 10,
            
            # AI Schema Generation
            "ai_schema_generation_enabled": False,
            "ai_schema_generation_rollout_percentage": 0,
            "ai_schema_generation_cache_enabled": True,
            "ai_schema_generation_timeout": 30,
            
            # Intelligent Caching
            "intelligent_caching_enabled": False,
            "intelligent_caching_rollout_percentage": 0,
            "intelligent_caching_ttl": 3600,
            "intelligent_caching_max_size": 1000,
            
            # Resilience Enhancements
            "resilience_enhancements_enabled": False,
            "resilience_enhancements_rollout_percentage": 0,
            "resilience_enhancements_retry_attempts": 3,
            "resilience_enhancements_backoff_factor": 2.0,
            
            # Performance optimization settings
            "content_change_detection_enabled": True,
            "stop_conditions_enabled": True,
            "batch_processing_enabled": True,
            "parallel_processing_enabled": True,
            "performance_monitoring_enabled": True,
            
            # Modular architecture settings
            "component_health_monitoring_enabled": True,
            "ab_testing_enabled": True,
            "component_swapping_enabled": True,
            
            # Pipeline migration settings
            "pipeline_architecture_migration_enabled": True,
            "pipeline_fallback_enabled": True,
            "pipeline_comparison_enabled": False
        }
        
        # Get current flags and update with Phase 10 flags
        current_flags = FeatureFlags.get_all_flags()
        current_flags.update(phase10_flags)
        FeatureFlags.initialize(current_flags)
        
        logger.info("Enhanced feature flags configured for Phase 10")
    
    def _initialize_performance_features(self) -> None:
        """Initialize performance optimization features."""
        # Configure stop conditions
        stop_manager = self.performance_optimizer.stop_manager
        
        # Configure common stop conditions
        stop_manager.configure_condition('item_count', {
            'target_count': 20
        })
        
        stop_manager.configure_condition('time_limit', {
            'time_limit_seconds': 600  # 10 minutes
        })
        
        stop_manager.configure_condition('data_quality', {
            'min_quality_score': 0.8,
            'min_complete_fields': 0.9
        })
        
        # Configure batch processing
        batch_processor = self.performance_optimizer.batch_processor
        batch_processor.max_batch_size = FeatureFlags.get_flag(
            'ai_schema_generation_batch_size', 8192
        )
        
        logger.info("Performance optimization features initialized")
    
    def _initialize_modular_architecture(self) -> None:
        """Initialize modular architecture components."""
        # Register example components for different types
        # This would typically be done by the actual component implementations
        
        # Register a mock intent analyzer component
        from core.modular_architecture import BaseComponent, ComponentMetadata
        
        class MockIntentAnalyzer(BaseComponent):
            async def _process_impl(self, context: PipelineContext) -> PipelineContext:
                # Mock implementation
                await asyncio.sleep(0.1)  # Simulate processing time
                context.data['intent_analysis'] = {
                    'primary_intent': 'data_extraction',
                    'confidence': 0.95,
                    'processed_by': self.metadata.name
                }
                return context
        
        # Create component metadata
        intent_analyzer_metadata = ComponentMetadata(
            name="default_intent_analyzer",
            component_type=ComponentType.INTENT_ANALYZER,
            version="1.0.0",
            description="Default intent analyzer component",
            status=ComponentStatus.ACTIVE
        )
        
        # Register the component
        mock_analyzer = MockIntentAnalyzer(intent_analyzer_metadata)
        self.component_registry.register_component(mock_analyzer, is_default=True)
        
        logger.info("Modular architecture components initialized")
    
    def start_progressive_rollouts(self) -> Dict[str, bool]:
        """Start progressive rollouts for all SmartScrape features."""
        results = {}
        
        for feature_name, config in SMARTSCRAPE_ROLLOUT_CONFIGS.items():
            try:
                # Create rollout
                rollout_id = self.rollout_manager.create_rollout(config)
                
                # Start rollout
                success = self.rollout_manager.start_rollout(rollout_id)
                
                if success:
                    self.active_rollouts[feature_name] = rollout_id
                    self.status.active_rollouts.append(rollout_id)
                    logger.info(f"Started progressive rollout for {feature_name}")
                else:
                    logger.error(f"Failed to start rollout for {feature_name}")
                
                results[feature_name] = success
                
            except Exception as e:
                logger.error(f"Error starting rollout for {feature_name}: {e}")
                results[feature_name] = False
        
        return results
    
    def get_rollout_status_summary(self) -> Dict[str, Any]:
        """Get summary of all active rollouts."""
        summary = {
            'total_rollouts': len(self.active_rollouts),
            'rollouts': {}
        }
        
        for feature_name, rollout_id in self.active_rollouts.items():
            status = self.rollout_manager.get_rollout_status(rollout_id)
            if status:
                summary['rollouts'][feature_name] = {
                    'rollout_id': rollout_id,
                    'status': status['status'],
                    'current_phase': status['current_phase'],
                    'created_at': status['created_at']
                }
        
        return summary
    
    def progress_rollout(self, feature_name: str) -> bool:
        """Manually progress a specific rollout to the next phase."""
        if feature_name not in self.active_rollouts:
            logger.error(f"No active rollout found for {feature_name}")
            return False
        
        rollout_id = self.active_rollouts[feature_name]
        return self.rollout_manager.progress_rollout(rollout_id)
    
    def rollback_feature(self, feature_name: str, reason: str = "Manual rollback") -> bool:
        """Rollback a specific feature."""
        if feature_name not in self.active_rollouts:
            logger.error(f"No active rollout found for {feature_name}")
            return False
        
        rollout_id = self.active_rollouts[feature_name]
        success = self.rollout_manager.rollback_rollout(rollout_id, reason)
        
        if success:
            del self.active_rollouts[feature_name]
            if rollout_id in self.status.active_rollouts:
                self.status.active_rollouts.remove(rollout_id)
        
        return success
    
    def create_ab_test(self, component_type: str, variants: Dict[str, str], 
                      traffic_split: Dict[str, float]) -> str:
        """Create an A/B test for component variants."""
        test_id = f"ab_test_{component_type}_{int(time.time())}"
        
        success = self.component_registry.create_ab_test(
            test_id, component_type, variants, traffic_split
        )
        
        if success:
            logger.info(f"Created A/B test {test_id} for {component_type}")
            return test_id
        else:
            logger.error(f"Failed to create A/B test for {component_type}")
            return None
    
    def get_ab_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results."""
        return self.component_registry.get_ab_test_results(test_id)
    
    def optimize_extraction_pipeline(self, urls: List[str], 
                                   extraction_func: Callable[[str], Dict[str, Any]],
                                   optimization_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run an optimized extraction pipeline with all Phase 10 enhancements.
        
        Args:
            urls: List of URLs to process
            extraction_func: Function to extract data from each URL
            optimization_config: Configuration for optimization features
            
        Returns:
            Extraction results with optimization metadata
        """
        config = optimization_config or {}
        
        # Configure stop conditions
        stop_conditions = config.get('stop_conditions', ['time_limit', 'item_count'])
        
        # Configure performance settings
        max_concurrent = config.get('max_concurrent', 3)
        
        # Configure stop condition parameters
        for condition, params in config.get('stop_condition_params', {}).items():
            self.performance_optimizer.stop_manager.configure_condition(condition, params)
        
        # Run optimized extraction
        results = self.performance_optimizer.optimize_extraction_pipeline(
            urls=urls,
            extraction_func=extraction_func,
            stop_conditions=stop_conditions,
            max_concurrent=max_concurrent
        )
        
        # Add Phase 10 metadata
        results['phase10_metadata'] = {
            'active_rollouts': list(self.active_rollouts.keys()),
            'performance_optimization_enabled': True,
            'modular_architecture_enabled': True,
            'feature_flags_summary': self._get_feature_flags_summary()
        }
        
        return results
    
    def _get_feature_flags_summary(self) -> Dict[str, Any]:
        """Get summary of current feature flag states."""
        flags = FeatureFlags.get_all_flags()
        
        return {
            'phase10_enabled': flags.get('phase10_enabled', False),
            'active_features': [
                feature for feature in [
                    'semantic_intent_analysis',
                    'ai_schema_generation',
                    'intelligent_caching',
                    'resilience_enhancements'
                ] if flags.get(f'{feature}_enabled', False)
            ],
            'optimization_features': [
                feature for feature in [
                    'content_change_detection',
                    'stop_conditions',
                    'batch_processing',
                    'parallel_processing'
                ] if flags.get(f'{feature}_enabled', False)
            ]
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        base_summary = self.performance_optimizer.get_performance_summary()
        
        # Add component health information
        component_health = {}
        for comp_type, components in self.component_registry.list_components().items():
            component_health[comp_type] = [
                {
                    'name': comp['name'],
                    'health_status': comp['health']['status'],
                    'success_rate': comp['health']['success_rate']
                }
                for comp in components
            ]
        
        base_summary.update({
            'component_health': component_health,
            'active_rollouts_count': len(self.active_rollouts),
            'ab_tests_count': len(self.component_registry.ab_tests),
            'phase10_status': {
                'rollout_manager_active': self.status.rollout_manager_active,
                'performance_optimizer_active': self.status.performance_optimizer_active,
                'modular_architecture_active': self.status.modular_architecture_active
            }
        })
        
        return base_summary
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health for Phase 10 components."""
        health_summary = {
            'overall_status': 'healthy',
            'components': {},
            'rollouts': {},
            'performance': {}
        }
        
        # Check component health
        for comp_type, components in self.component_registry.list_components().items():
            unhealthy_count = sum(
                1 for comp in components 
                if comp['health']['status'] != 'healthy'
            )
            health_summary['components'][comp_type] = {
                'total': len(components),
                'healthy': len(components) - unhealthy_count,
                'unhealthy': unhealthy_count
            }
        
        # Check rollout health
        for feature_name, rollout_id in self.active_rollouts.items():
            status = self.rollout_manager.get_rollout_status(rollout_id)
            if status:
                health_summary['rollouts'][feature_name] = status['status']
        
        # Performance health
        perf_summary = self.performance_optimizer.get_performance_summary()
        health_summary['performance'] = {
            'success_rate': perf_summary.get('success_rate', 1.0),
            'avg_duration': perf_summary.get('avg_duration_seconds', 0),
            'optimization_active': True
        }
        
        # Determine overall status
        component_issues = sum(
            comp['unhealthy'] for comp in health_summary['components'].values()
        )
        rollout_issues = sum(
            1 for status in health_summary['rollouts'].values() 
            if status in ['failed', 'rolled_back']
        )
        
        if component_issues > 0 or rollout_issues > 0:
            health_summary['overall_status'] = 'degraded'
        
        if (perf_summary.get('success_rate', 1.0) < 0.8 or 
            component_issues > len(health_summary['components']) // 2):
            health_summary['overall_status'] = 'unhealthy'
        
        return health_summary
    
    def shutdown(self) -> None:
        """Shutdown Phase 10 systems gracefully."""
        try:
            # Shutdown rollout manager
            self.rollout_manager.shutdown()
            
            # Clear active rollouts
            self.active_rollouts.clear()
            self.status.active_rollouts.clear()
            
            # Update status
            self.status.rollout_manager_active = False
            self.status.performance_optimizer_active = False
            self.status.modular_architecture_active = False
            
            logger.info("Phase 10 systems shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during Phase 10 shutdown: {e}")


# Global Phase 10 manager instance
_phase10_manager: Optional[Phase10Manager] = None

def get_phase10_manager() -> Phase10Manager:
    """Get the global Phase 10 manager instance."""
    global _phase10_manager
    if _phase10_manager is None:
        _phase10_manager = Phase10Manager()
    return _phase10_manager

def initialize_phase10() -> Phase10Manager:
    """Initialize Phase 10 systems."""
    return get_phase10_manager()

def start_all_rollouts() -> Dict[str, bool]:
    """Start all progressive rollouts."""
    manager = get_phase10_manager()
    return manager.start_progressive_rollouts()

def get_phase10_status() -> Dict[str, Any]:
    """Get comprehensive Phase 10 status."""
    manager = get_phase10_manager()
    return {
        'system_health': manager.get_system_health(),
        'rollout_summary': manager.get_rollout_status_summary(),
        'performance_summary': manager.get_performance_summary(),
        'feature_flags': manager._get_feature_flags_summary()
    }
