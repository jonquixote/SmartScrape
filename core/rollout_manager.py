"""
Progressive Rollout Manager for SmartScrape

This module manages progressive rollout of new features using feature flags,
A/B testing, and performance monitoring to ensure safe deployment.
"""

import json
import logging
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from core.pipeline.compatibility import FeatureFlags

logger = logging.getLogger(__name__)

class RolloutPhase(Enum):
    """Phases for progressive rollout."""
    DEVELOPMENT = "development"
    CANARY = "canary"
    STAGED_ROLLOUT = "staged_rollout"
    FULL_ROLLOUT = "full_rollout"
    ROLLBACK = "rollback"

class RolloutStatus(Enum):
    """Status of rollout execution."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class RolloutConfig:
    """Configuration for a feature rollout."""
    feature_name: str
    phases: Dict[RolloutPhase, Dict[str, Any]]
    success_criteria: Dict[str, Any]
    rollback_criteria: Dict[str, Any]
    monitoring_interval: int = 300  # 5 minutes
    auto_progress: bool = False
    max_duration: int = 86400  # 24 hours

@dataclass
class RolloutMetrics:
    """Metrics collected during rollout."""
    phase: RolloutPhase
    start_time: datetime
    end_time: Optional[datetime] = None
    requests_total: int = 0
    requests_success: int = 0
    requests_error: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

class ProgressiveRolloutManager:
    """
    Manages progressive rollout of features.
    
    This class orchestrates the gradual deployment of new features,
    monitoring their performance and automatically progressing or
    rolling back based on defined criteria.
    """
    
    def __init__(self, config_file: str = "rollout_configs.json"):
        self.config_file = config_file
        self.rollouts: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, List[RolloutMetrics]] = {}
        self.active_rollouts: Set[str] = set()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
        # Load existing configurations
        self._load_configs()
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _load_configs(self) -> None:
        """Load rollout configurations from file."""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.rollouts = data.get('rollouts', {})
                    # Deserialize metrics
                    raw_metrics = data.get('metrics', {})
                    for feature, metric_list in raw_metrics.items():
                        self.metrics[feature] = [
                            RolloutMetrics(
                                phase=RolloutPhase(m['phase']),
                                start_time=datetime.fromisoformat(m['start_time']),
                                end_time=datetime.fromisoformat(m['end_time']) if m.get('end_time') else None,
                                **{k: v for k, v in m.items() if k not in ['phase', 'start_time', 'end_time']}
                            ) for m in metric_list
                        ]
        except Exception as e:
            logger.error(f"Failed to load rollout configs: {e}")
            self.rollouts = {}
            self.metrics = {}
    
    def _save_configs(self) -> None:
        """Save rollout configurations to file."""
        try:
            # Serialize metrics
            serialized_metrics = {}
            for feature, metric_list in self.metrics.items():
                serialized_metrics[feature] = []
                for metric in metric_list:
                    metric_dict = asdict(metric)
                    metric_dict['phase'] = metric.phase.value
                    metric_dict['start_time'] = metric.start_time.isoformat()
                    if metric.end_time:
                        metric_dict['end_time'] = metric.end_time.isoformat()
                    serialized_metrics[feature].append(metric_dict)
            
            data = {
                'rollouts': self.rollouts,
                'metrics': serialized_metrics
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rollout configs: {e}")
    
    def create_rollout(self, config: RolloutConfig) -> str:
        """
        Create a new progressive rollout.
        
        Args:
            config: Rollout configuration
            
        Returns:
            Rollout ID
        """
        rollout_id = str(uuid.uuid4())
        
        with self._lock:
            # Convert config to serializable format
            config_dict = asdict(config)
            # Convert enum keys to strings
            if 'phases' in config_dict:
                phases_dict = {}
                for phase_key, phase_value in config_dict['phases'].items():
                    if hasattr(phase_key, 'value'):
                        phases_dict[phase_key.value] = phase_value
                    else:
                        phases_dict[str(phase_key)] = phase_value
                config_dict['phases'] = phases_dict
            
            self.rollouts[rollout_id] = {
                'feature_name': config.feature_name,
                'config': config_dict,
                'status': RolloutStatus.PENDING.value,
                'current_phase': RolloutPhase.DEVELOPMENT.value,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            self.metrics[rollout_id] = []
            self._save_configs()
        
        logger.info(f"Created rollout {rollout_id} for feature {config.feature_name}")
        return rollout_id
    
    def start_rollout(self, rollout_id: str) -> bool:
        """
        Start a progressive rollout.
        
        Args:
            rollout_id: ID of the rollout to start
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if rollout_id not in self.rollouts:
                logger.error(f"Rollout {rollout_id} not found")
                return False
            
            rollout = self.rollouts[rollout_id]
            if rollout['status'] != RolloutStatus.PENDING.value:
                logger.error(f"Rollout {rollout_id} is not in pending state")
                return False
            
            # Start with first phase
            first_phase = RolloutPhase.CANARY
            rollout['status'] = RolloutStatus.ACTIVE.value
            rollout['current_phase'] = first_phase.value
            rollout['updated_at'] = datetime.now().isoformat()
            
            # Initialize metrics for this phase
            self.metrics[rollout_id].append(
                RolloutMetrics(
                    phase=first_phase,
                    start_time=datetime.now()
                )
            )
            
            # Configure feature flags for this phase
            self._configure_feature_flags(rollout_id, first_phase)
            
            self.active_rollouts.add(rollout_id)
            self._save_configs()
        
        logger.info(f"Started rollout {rollout_id} in {first_phase.value} phase")
        return True
    
    def _configure_feature_flags(self, rollout_id: str, phase: RolloutPhase) -> None:
        """Configure feature flags for the given rollout phase."""
        rollout = self.rollouts[rollout_id]
        config = rollout['config']
        feature_name = rollout['feature_name']
        
        phase_config = config['phases'].get(phase.value, {})
        
        # Set rollout percentage based on phase
        rollout_percentage = phase_config.get('percentage', {
            RolloutPhase.CANARY.value: 5,
            RolloutPhase.STAGED_ROLLOUT.value: 50,
            RolloutPhase.FULL_ROLLOUT.value: 100
        }.get(phase.value, 0))
        
        # Update feature flags
        current_flags = FeatureFlags.get_all_flags()
        current_flags[f'{feature_name}_enabled'] = True
        current_flags[f'{feature_name}_rollout_percentage'] = rollout_percentage
        
        # Apply phase-specific settings
        for key, value in phase_config.items():
            if key != 'percentage':
                current_flags[f'{feature_name}_{key}'] = value
        
        FeatureFlags.initialize(current_flags)
        
        logger.info(f"Configured feature flags for {feature_name} at {rollout_percentage}% rollout")
    
    def progress_rollout(self, rollout_id: str) -> bool:
        """
        Progress rollout to next phase.
        
        Args:
            rollout_id: ID of the rollout to progress
            
        Returns:
            True if progressed successfully, False otherwise
        """
        with self._lock:
            if rollout_id not in self.rollouts:
                return False
            
            rollout = self.rollouts[rollout_id]
            current_phase = RolloutPhase(rollout['current_phase'])
            
            # Determine next phase
            phase_order = [
                RolloutPhase.CANARY,
                RolloutPhase.STAGED_ROLLOUT,
                RolloutPhase.FULL_ROLLOUT
            ]
            
            try:
                current_index = phase_order.index(current_phase)
                if current_index < len(phase_order) - 1:
                    next_phase = phase_order[current_index + 1]
                else:
                    # Already at final phase
                    rollout['status'] = RolloutStatus.COMPLETED.value
                    self.active_rollouts.discard(rollout_id)
                    logger.info(f"Rollout {rollout_id} completed successfully")
                    return True
            except ValueError:
                logger.error(f"Invalid current phase: {current_phase}")
                return False
            
            # Finalize current phase metrics
            current_metrics = self.metrics[rollout_id][-1]
            current_metrics.end_time = datetime.now()
            
            # Start next phase
            rollout['current_phase'] = next_phase.value
            rollout['updated_at'] = datetime.now().isoformat()
            
            # Initialize metrics for next phase
            self.metrics[rollout_id].append(
                RolloutMetrics(
                    phase=next_phase,
                    start_time=datetime.now()
                )
            )
            
            # Configure feature flags for next phase
            self._configure_feature_flags(rollout_id, next_phase)
            
            self._save_configs()
        
        logger.info(f"Progressed rollout {rollout_id} to {next_phase.value} phase")
        return True
    
    def rollback_rollout(self, rollout_id: str, reason: str = "Manual rollback") -> bool:
        """
        Rollback a rollout.
        
        Args:
            rollout_id: ID of the rollout to rollback
            reason: Reason for rollback
            
        Returns:
            True if rolled back successfully, False otherwise
        """
        with self._lock:
            if rollout_id not in self.rollouts:
                return False
            
            rollout = self.rollouts[rollout_id]
            feature_name = rollout['feature_name']
            
            # Disable the feature
            current_flags = FeatureFlags.get_all_flags()
            current_flags[f'{feature_name}_enabled'] = False
            current_flags[f'{feature_name}_rollout_percentage'] = 0
            FeatureFlags.initialize(current_flags)
            
            # Update rollout status
            rollout['status'] = RolloutStatus.ROLLED_BACK.value
            rollout['rollback_reason'] = reason
            rollout['updated_at'] = datetime.now().isoformat()
            
            # Finalize current phase metrics
            if self.metrics[rollout_id]:
                current_metrics = self.metrics[rollout_id][-1]
                current_metrics.end_time = datetime.now()
            
            self.active_rollouts.discard(rollout_id)
            self._save_configs()
        
        logger.warning(f"Rolled back rollout {rollout_id}: {reason}")
        return True
    
    def get_rollout_status(self, rollout_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a rollout."""
        if rollout_id not in self.rollouts:
            return None
        
        rollout = self.rollouts[rollout_id].copy()
        rollout['metrics'] = [asdict(m) for m in self.metrics.get(rollout_id, [])]
        return rollout

    def find_rollout_by_feature(self, feature_name: str) -> Optional[str]:
        """Find rollout ID by feature name."""
        for rollout_id, rollout in self.rollouts.items():
            if rollout['feature_name'] == feature_name:
                return rollout_id
        return None

    def get_rollout_status_by_feature(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a rollout by feature name."""
        rollout_id = self.find_rollout_by_feature(feature_name)
        if rollout_id:
            return self.get_rollout_status(rollout_id)
        return None

    def progress_rollout_by_feature(self, feature_name: str) -> bool:
        """Progress rollout to next phase by feature name."""
        rollout_id = self.find_rollout_by_feature(feature_name)
        if rollout_id:
            return self.progress_rollout(rollout_id)
        return False
    
    def find_rollout_by_feature(self, feature_name: str) -> Optional[str]:
        """Find rollout ID by feature name."""
        for rollout_id, rollout in self.rollouts.items():
            if rollout['feature_name'] == feature_name:
                return rollout_id
        return None

    def get_rollout_status_by_feature(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a rollout by feature name."""
        rollout_id = self.find_rollout_by_feature(feature_name)
        if rollout_id:
            return self.get_rollout_status(rollout_id)
        return None

    def progress_rollout_by_feature(self, feature_name: str) -> bool:
        """Progress rollout to next phase by feature name."""
        rollout_id = self.find_rollout_by_feature(feature_name)
        if rollout_id:
            return self.progress_rollout(rollout_id)
        return False
    
    def list_rollouts(self) -> List[Dict[str, Any]]:
        """List all rollouts."""
        return [
            {
                'rollout_id': rid,
                'feature_name': rollout['feature_name'],
                'status': rollout['status'],
                'current_phase': rollout['current_phase'],
                'created_at': rollout['created_at']
            }
            for rid, rollout in self.rollouts.items()
        ]
    
    def record_metrics(self, rollout_id: str, **metrics) -> None:
        """Record metrics for an active rollout."""
        if rollout_id not in self.active_rollouts:
            return
        
        with self._lock:
            if self.metrics.get(rollout_id):
                current_metrics = self.metrics[rollout_id][-1]
                
                # Update metrics
                for key, value in metrics.items():
                    if hasattr(current_metrics, key):
                        setattr(current_metrics, key, value)
                    else:
                        current_metrics.performance_metrics[key] = value
    
    def _start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="RolloutMonitoring"
        )
        self._monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                self._check_rollouts()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_rollouts(self) -> None:
        """Check all active rollouts for auto-progression or rollback conditions."""
        active_rollouts = list(self.active_rollouts)
        
        for rollout_id in active_rollouts:
            try:
                self._check_rollout(rollout_id)
            except Exception as e:
                logger.error(f"Error checking rollout {rollout_id}: {e}")
    
    def _check_rollout(self, rollout_id: str) -> None:
        """Check a specific rollout for progression or rollback."""
        rollout = self.rollouts[rollout_id]
        config = rollout['config']
        
        if not config.get('auto_progress', False):
            return
        
        current_metrics = self.metrics[rollout_id][-1] if self.metrics[rollout_id] else None
        if not current_metrics:
            return
        
        # Check rollback criteria
        rollback_criteria = config.get('rollback_criteria', {})
        if self._should_rollback(current_metrics, rollback_criteria):
            reason = f"Auto-rollback: Failed criteria {rollback_criteria}"
            self.rollback_rollout(rollout_id, reason)
            return
        
        # Check success criteria for progression
        success_criteria = config.get('success_criteria', {})
        phase_duration = (datetime.now() - current_metrics.start_time).total_seconds()
        min_duration = success_criteria.get('min_phase_duration', 1800)  # 30 minutes
        
        if (phase_duration >= min_duration and 
            self._meets_success_criteria(current_metrics, success_criteria)):
            self.progress_rollout(rollout_id)
    
    def _should_rollback(self, metrics: RolloutMetrics, criteria: Dict[str, Any]) -> bool:
        """Check if rollback conditions are met."""
        max_error_rate = criteria.get('max_error_rate', 0.05)  # 5%
        max_response_time = criteria.get('max_response_time', 5.0)  # 5 seconds
        min_success_rate = criteria.get('min_success_rate', 0.95)  # 95%
        
        if metrics.error_rate > max_error_rate:
            return True
        
        if metrics.avg_response_time > max_response_time:
            return True
        
        if metrics.requests_total > 0:
            success_rate = metrics.requests_success / metrics.requests_total
            if success_rate < min_success_rate:
                return True
        
        return False
    
    def _meets_success_criteria(self, metrics: RolloutMetrics, criteria: Dict[str, Any]) -> bool:
        """Check if success criteria are met."""
        min_requests = criteria.get('min_requests', 100)
        max_error_rate = criteria.get('max_error_rate', 0.02)  # 2%
        max_response_time = criteria.get('max_response_time', 2.0)  # 2 seconds
        
        if metrics.requests_total < min_requests:
            return False
        
        if metrics.error_rate > max_error_rate:
            return False
        
        if metrics.avg_response_time > max_response_time:
            return False
        
        return True
    
    def shutdown(self) -> None:
        """Shutdown the rollout manager."""
        self._shutdown_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)


# Predefined rollout configurations for SmartScrape features
SMARTSCRAPE_ROLLOUT_CONFIGS = {
    'semantic_intent_analysis': RolloutConfig(
        feature_name='semantic_intent_analysis',
        phases={
            'canary': {'percentage': 5, 'max_requests': 100},
            'staged_rollout': {'percentage': 25, 'max_requests': 1000},
            'full_rollout': {'percentage': 100}
        },
        success_criteria={
            'min_requests': 50,
            'max_error_rate': 0.02,
            'max_response_time': 3.0,
            'min_phase_duration': 1800  # 30 minutes
        },
        rollback_criteria={
            'max_error_rate': 0.05,
            'max_response_time': 10.0,
            'min_success_rate': 0.90
        },
        auto_progress=True
    ),
    
    'ai_schema_generation': RolloutConfig(
        feature_name='ai_schema_generation',
        phases={
            'canary': {'percentage': 10, 'cache_enabled': True},
            'staged_rollout': {'percentage': 50, 'cache_enabled': True},
            'full_rollout': {'percentage': 100, 'cache_enabled': True}
        },
        success_criteria={
            'min_requests': 25,
            'max_error_rate': 0.03,
            'max_response_time': 5.0,
            'min_phase_duration': 3600  # 1 hour
        },
        rollback_criteria={
            'max_error_rate': 0.08,
            'max_response_time': 15.0
        },
        auto_progress=True
    ),
    
    'intelligent_caching': RolloutConfig(
        feature_name='intelligent_caching',
        phases={
            'canary': {'percentage': 15, 'cache_ttl': 300},
            'staged_rollout': {'percentage': 60, 'cache_ttl': 600},
            'full_rollout': {'percentage': 100, 'cache_ttl': 3600}
        },
        success_criteria={
            'min_requests': 100,
            'max_error_rate': 0.01,
            'min_cache_hit_rate': 0.3,
            'min_phase_duration': 1800
        },
        rollback_criteria={
            'max_error_rate': 0.03,
            'cache_hit_rate': 0.1
        },
        auto_progress=True
    ),
    
    'resilience_enhancements': RolloutConfig(
        feature_name='resilience_enhancements',
        phases={
            'canary': {'percentage': 5, 'retry_attempts': 2},
            'staged_rollout': {'percentage': 30, 'retry_attempts': 3},
            'full_rollout': {'percentage': 100, 'retry_attempts': 3}
        },
        success_criteria={
            'min_requests': 75,
            'max_error_rate': 0.02,
            'min_success_rate': 0.95,
            'min_phase_duration': 2700  # 45 minutes
        },
        rollback_criteria={
            'max_error_rate': 0.06,
            'min_success_rate': 0.85
        },
        auto_progress=True
    )
}


# Global rollout manager instance
_rollout_manager: Optional[ProgressiveRolloutManager] = None

def get_rollout_manager() -> ProgressiveRolloutManager:
    """Get the global rollout manager instance."""
    global _rollout_manager
    if _rollout_manager is None:
        _rollout_manager = ProgressiveRolloutManager()
    return _rollout_manager
