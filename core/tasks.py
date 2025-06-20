"""
Celery Tasks for SmartScrape

This module defines asynchronous tasks for extraction operations, batch processing,
and maintenance tasks using Celery.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from celery import current_task
from celery.exceptions import Retry
from core.celery_config import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def extract_url_task(self, url: str, strategy: str = None, **kwargs):
    """
    Async extraction task for a single URL
    
    Args:
        url: URL to extract content from
        strategy: Optional specific strategy to use
        **kwargs: Additional parameters for extraction
    
    Returns:
        Dict containing extraction results
    """
    try:
        # Update task status
        self.update_state(
            state='PROGRESS', 
            meta={
                'status': 'Starting extraction',
                'url': url,
                'strategy': strategy,
                'progress': 0
            }
        )
        
        # Import here to avoid circular imports
        from controllers.extraction_coordinator import ExtractionCoordinator
        
        # Create coordinator instance
        coordinator = ExtractionCoordinator()
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Coordinator initialized',
                'url': url,
                'progress': 25
            }
        )
        
        # Perform extraction
        start_time = time.time()
        
        if strategy:
            # Use specific strategy
            self.update_state(
                state='PROGRESS',
                meta={
                    'status': f'Extracting with {strategy}',
                    'url': url,
                    'progress': 50
                }
            )
            # Note: We need to handle async calls in Celery tasks
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    coordinator.extract_content(url, strategy, **kwargs)
                )
            finally:
                loop.close()
        else:
            # Use intelligent selection or fallback
            self.update_state(
                state='PROGRESS',
                meta={
                    'status': 'Using intelligent strategy selection',
                    'url': url,
                    'progress': 50
                }
            )
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    coordinator.extract_with_intelligent_selection(url, **kwargs)
                )
            finally:
                loop.close()
        
        execution_time = time.time() - start_time
        
        # Final update
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'Extraction completed',
                'url': url,
                'progress': 100,
                'execution_time': execution_time,
                'strategy_used': result.get('strategy', 'unknown'),
                'success': result.get('success', False),
                'content_length': len(result.get('content', ''))
            }
        )
        
        return {
            'url': url,
            'result': result,
            'execution_time': execution_time,
            'task_id': self.request.id,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Extraction task failed for {url}: {str(e)}")
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying extraction for {url} (attempt {self.request.retries + 1})")
            self.update_state(
                state='RETRY',
                meta={
                    'status': f'Retrying (attempt {self.request.retries + 1})',
                    'url': url,
                    'error': str(e),
                    'progress': 0
                }
            )
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        
        # Final failure
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Failed after retries',
                'url': url,
                'error': str(e),
                'retries': self.request.retries
            }
        )
        raise

@celery_app.task(bind=True)
def batch_extract_task(self, urls: List[str], strategy: str = None, **kwargs):
    """
    Batch extraction task for multiple URLs
    
    Args:
        urls: List of URLs to extract
        strategy: Optional strategy to use for all URLs
        **kwargs: Additional parameters
        
    Returns:
        Dict containing batch results and task IDs
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Starting batch extraction',
                'total_urls': len(urls),
                'progress': 0
            }
        )
        
        results = []
        task_ids = []
        
        # Queue individual extraction tasks
        for i, url in enumerate(urls):
            try:
                # Create individual extraction task
                task = extract_url_task.delay(url, strategy, **kwargs)
                task_ids.append(task.id)
                
                results.append({
                    'url': url,
                    'task_id': task.id,
                    'status': 'queued'
                })
                
                # Update progress
                progress = int((i + 1) / len(urls) * 100)
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'status': f'Queued {i + 1}/{len(urls)} tasks',
                        'total_urls': len(urls),
                        'queued': i + 1,
                        'progress': progress
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to queue task for {url}: {str(e)}")
                results.append({
                    'url': url,
                    'error': str(e),
                    'status': 'failed_to_queue'
                })
        
        return {
            'batch_id': self.request.id,
            'total_urls': len(urls),
            'queued_tasks': len(task_ids),
            'task_ids': task_ids,
            'results': results,
            'status': 'batch_queued'
        }
        
    except Exception as e:
        logger.error(f"Batch extraction task failed: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Batch extraction failed',
                'error': str(e),
                'total_urls': len(urls) if urls else 0
            }
        )
        raise

@celery_app.task(bind=True)
def cache_warm_task(self, urls: List[str] = None, strategies: List[str] = None):
    """
    Cache warming task
    
    Args:
        urls: URLs to warm cache for (uses defaults if None)
        strategies: Strategies to use (uses defaults if None)
        
    Returns:
        Dict containing warming results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Starting cache warming', 'progress': 0}
        )
        
        # Import cache warmer
        from scripts.cache_warmer import CacheWarmer
        
        # Initialize warmer
        warmer = CacheWarmer()
        
        # Run warming in async context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(warmer.initialize())
            loop.run_until_complete(warmer.warm_cache(urls, strategies))
        finally:
            loop.close()
        
        # Return results
        return {
            'task_id': self.request.id,
            'status': 'cache_warming_completed',
            'stats': warmer.stats
        }
        
    except Exception as e:
        logger.error(f"Cache warming task failed: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'Cache warming failed',
                'error': str(e)
            }
        )
        raise

@celery_app.task(bind=True)
def pipeline_health_check_task(self):
    """
    Health check task for pipeline monitoring
    
    Returns:
        Dict containing system health information
    """
    try:
        import psutil
        from datetime import datetime
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Redis health check
        try:
            from utils.database_manager import db_manager
            redis_healthy = db_manager.enabled and db_manager.redis_client and db_manager.redis_client.ping()
        except:
            redis_healthy = False
        
        # Database health check
        try:
            from utils.database_manager import db_manager
            db_healthy = db_manager.enabled
        except:
            db_healthy = False
        
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
            },
            'services': {
                'redis': redis_healthy,
                'database': db_healthy,
                'celery': True  # If this task runs, Celery is working
            },
            'task_id': self.request.id,
            'status': 'healthy'
        }
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check task failed: {str(e)}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'unhealthy',
            'error': str(e),
            'task_id': self.request.id
        }
