"""
Pipeline Orchestrator for SmartScrape

This module provides high-level pipeline orchestration for complex extraction workflows,
batch processing, and task management using Celery.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from celery.result import AsyncResult, GroupResult
from celery import group, chain, chord
from core.celery_config import celery_app
from core.tasks import extract_url_task, batch_extract_task, cache_warm_task, pipeline_health_check_task

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"

@dataclass
class PipelineResult:
    """Pipeline execution result"""
    pipeline_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if pipeline is complete"""
        return self.status in [PipelineStatus.SUCCESS, PipelineStatus.FAILURE, PipelineStatus.CANCELLED]

class PipelineOrchestrator:
    """Orchestrates complex extraction pipelines"""
    
    def __init__(self):
        self.active_pipelines: Dict[str, PipelineResult] = {}
        self.pipeline_callbacks: Dict[str, List[Callable]] = {}
        
    def create_simple_pipeline(self, urls: List[str], strategy: str = None, **kwargs) -> str:
        """
        Create a simple parallel extraction pipeline
        
        Args:
            urls: List of URLs to extract
            strategy: Optional strategy to use
            **kwargs: Additional parameters
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"simple_pipeline_{int(time.time())}"
        
        # Create pipeline result tracker
        pipeline_result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.PENDING,
            start_time=datetime.utcnow(),
            total_tasks=len(urls)
        )
        self.active_pipelines[pipeline_id] = pipeline_result
        
        # Create group of extraction tasks
        job = group(
            extract_url_task.s(url, strategy, **kwargs) 
            for url in urls
        )
        
        # Execute the group
        result = job.apply_async()
        
        # Store the group result for monitoring
        pipeline_result.celery_result = result
        pipeline_result.status = PipelineStatus.STARTED
        
        logger.info(f"Created simple pipeline {pipeline_id} with {len(urls)} tasks")
        return pipeline_id
    
    def create_batch_pipeline(self, url_batches: List[List[str]], strategy: str = None, **kwargs) -> str:
        """
        Create a batch processing pipeline with sequential batch execution
        
        Args:
            url_batches: List of URL batches to process sequentially
            strategy: Optional strategy to use
            **kwargs: Additional parameters
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"batch_pipeline_{int(time.time())}"
        
        total_urls = sum(len(batch) for batch in url_batches)
        pipeline_result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.PENDING,
            start_time=datetime.utcnow(),
            total_tasks=total_urls
        )
        self.active_pipelines[pipeline_id] = pipeline_result
        
        # Create chain of batch tasks
        batch_tasks = [
            batch_extract_task.s(batch, strategy, **kwargs)
            for batch in url_batches
        ]
        
        # Execute as a chain (sequential)
        job = chain(*batch_tasks)
        result = job.apply_async()
        
        pipeline_result.celery_result = result
        pipeline_result.status = PipelineStatus.STARTED
        
        logger.info(f"Created batch pipeline {pipeline_id} with {len(url_batches)} batches ({total_urls} total URLs)")
        return pipeline_id
    
    def create_smart_pipeline(self, urls: List[str], **kwargs) -> str:
        """
        Create an intelligent pipeline that adapts based on URL characteristics
        
        Args:
            urls: List of URLs to extract
            **kwargs: Additional parameters
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"smart_pipeline_{int(time.time())}"
        
        pipeline_result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.PENDING,
            start_time=datetime.utcnow(),
            total_tasks=len(urls)
        )
        self.active_pipelines[pipeline_id] = pipeline_result
        
        # Group URLs by domain for optimization
        domain_groups = self._group_urls_by_domain(urls)
        
        # Create tasks with intelligent strategy selection
        tasks = []
        for domain, domain_urls in domain_groups.items():
            # Use intelligent extraction for each URL
            for url in domain_urls:
                tasks.append(extract_url_task.s(url, strategy=None, **kwargs))
        
        # Execute with optimal concurrency
        job = group(*tasks)
        result = job.apply_async()
        
        pipeline_result.celery_result = result
        pipeline_result.status = PipelineStatus.STARTED
        
        logger.info(f"Created smart pipeline {pipeline_id} with {len(tasks)} tasks across {len(domain_groups)} domains")
        return pipeline_id
    
    def create_pipeline_with_callback(self, urls: List[str], callback_task=None, **kwargs) -> str:
        """
        Create a pipeline with a callback task that runs after all extractions complete
        
        Args:
            urls: List of URLs to extract
            callback_task: Celery task to run after completion
            **kwargs: Additional parameters
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"callback_pipeline_{int(time.time())}"
        
        pipeline_result = PipelineResult(
            pipeline_id=pipeline_id,
            status=PipelineStatus.PENDING,
            start_time=datetime.utcnow(),
            total_tasks=len(urls)
        )
        self.active_pipelines[pipeline_id] = pipeline_result
        
        # Create extraction tasks
        extraction_tasks = group(
            extract_url_task.s(url, **kwargs) 
            for url in urls
        )
        
        # Use chord to run callback after all extractions complete
        if callback_task:
            job = chord(extraction_tasks)(callback_task.s())
        else:
            job = extraction_tasks.apply_async()
        
        pipeline_result.celery_result = job
        pipeline_result.status = PipelineStatus.STARTED
        
        logger.info(f"Created callback pipeline {pipeline_id} with {len(urls)} tasks")
        return pipeline_id
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineResult]:
        """
        Get the current status of a pipeline
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            Pipeline result or None if not found
        """
        if pipeline_id not in self.active_pipelines:
            return None
        
        pipeline_result = self.active_pipelines[pipeline_id]
        
        # Update status from Celery result
        if hasattr(pipeline_result, 'celery_result'):
            celery_result = pipeline_result.celery_result
            
            if isinstance(celery_result, GroupResult):
                # Handle group results
                completed = sum(1 for result in celery_result.results if result.ready())
                failed = sum(1 for result in celery_result.results if result.failed())
                
                pipeline_result.completed_tasks = completed - failed
                pipeline_result.failed_tasks = failed
                
                if celery_result.ready():
                    if celery_result.successful():
                        pipeline_result.status = PipelineStatus.SUCCESS
                    else:
                        pipeline_result.status = PipelineStatus.FAILURE
                    pipeline_result.end_time = datetime.utcnow()
                    pipeline_result.execution_time = (
                        pipeline_result.end_time - pipeline_result.start_time
                    ).total_seconds()
                else:
                    pipeline_result.status = PipelineStatus.PROGRESS
            
            elif isinstance(celery_result, AsyncResult):
                # Handle single task results
                if celery_result.ready():
                    if celery_result.successful():
                        pipeline_result.status = PipelineStatus.SUCCESS
                        pipeline_result.completed_tasks = 1
                    else:
                        pipeline_result.status = PipelineStatus.FAILURE
                        pipeline_result.failed_tasks = 1
                    pipeline_result.end_time = datetime.utcnow()
                    pipeline_result.execution_time = (
                        pipeline_result.end_time - pipeline_result.start_time
                    ).total_seconds()
                else:
                    pipeline_result.status = PipelineStatus.PROGRESS
        
        return pipeline_result
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel a running pipeline
        
        Args:
            pipeline_id: Pipeline ID
            
        Returns:
            True if cancelled successfully
        """
        if pipeline_id not in self.active_pipelines:
            return False
        
        pipeline_result = self.active_pipelines[pipeline_id]
        
        # Cancel Celery tasks
        if hasattr(pipeline_result, 'celery_result'):
            pipeline_result.celery_result.revoke(terminate=True)
        
        # Update status
        pipeline_result.status = PipelineStatus.CANCELLED
        pipeline_result.end_time = datetime.utcnow()
        
        logger.info(f"Cancelled pipeline {pipeline_id}")
        return True
    
    def cleanup_completed_pipelines(self, max_age_hours: int = 24):
        """
        Clean up completed pipelines older than specified age
        
        Args:
            max_age_hours: Maximum age in hours for keeping completed pipelines
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for pipeline_id, pipeline_result in self.active_pipelines.items():
            if (pipeline_result.is_complete and 
                pipeline_result.end_time and 
                pipeline_result.end_time < cutoff_time):
                to_remove.append(pipeline_id)
        
        for pipeline_id in to_remove:
            del self.active_pipelines[pipeline_id]
            if pipeline_id in self.pipeline_callbacks:
                del self.pipeline_callbacks[pipeline_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed pipelines")
    
    def get_system_health(self) -> Dict:
        """
        Get system health information
        
        Returns:
            Dict containing health metrics
        """
        # Queue a health check task
        health_task = pipeline_health_check_task.delay()
        
        # Get basic pipeline statistics
        total_pipelines = len(self.active_pipelines)
        active_pipelines = sum(1 for p in self.active_pipelines.values() if not p.is_complete)
        completed_pipelines = sum(1 for p in self.active_pipelines.values() if p.status == PipelineStatus.SUCCESS)
        failed_pipelines = sum(1 for p in self.active_pipelines.values() if p.status == PipelineStatus.FAILURE)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'health_check_task_id': health_task.id,
            'pipeline_stats': {
                'total': total_pipelines,
                'active': active_pipelines,
                'completed': completed_pipelines,
                'failed': failed_pipelines
            },
            'celery_status': 'operational'  # If we can queue tasks, Celery is working
        }
    
    def _group_urls_by_domain(self, urls: List[str]) -> Dict[str, List[str]]:
        """Group URLs by domain for optimized processing"""
        from urllib.parse import urlparse
        
        domain_groups = {}
        for url in urls:
            try:
                domain = urlparse(url).netloc
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(url)
            except Exception:
                # Use 'unknown' for unparseable URLs
                if 'unknown' not in domain_groups:
                    domain_groups['unknown'] = []
                domain_groups['unknown'].append(url)
        
        return domain_groups

# Global orchestrator instance
pipeline_orchestrator = PipelineOrchestrator()
