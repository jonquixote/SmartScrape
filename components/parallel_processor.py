"""
Parallel Processing Component

Handles concurrent URL extraction with:
- Worker pool management
- Asynchronous batch processing
- Resource throttling and rate limiting
- Progress tracking and error aggregation
- Intelligent task distribution

This component provides high-performance parallel extraction capabilities
while maintaining system stability and respecting rate limits.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

# Import core error handling
import functools

def handle_errors_gracefully(func):
    """Decorator for graceful error handling"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in {func.__name__}: {e}")
            return None
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in {func.__name__}: {e}")
            return None
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    # Worker pool settings
    max_workers: int = 10
    max_concurrent_tasks: int = 50
    
    # Rate limiting
    requests_per_second: float = 5.0
    burst_size: int = 10
    
    # Timeouts and retries
    task_timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 1.5
    
    # Resource management
    memory_limit_mb: int = 1024
    cpu_threshold: float = 80.0
    
    # Progress tracking
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None


@dataclass
class TaskResult:
    """Result of a parallel processing task"""
    task_id: str
    url: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    attempt_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a batch processing operation"""
    batch_id: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_time: float
    results: List[TaskResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """Token bucket rate limiter for controlling request rates"""
    
    def __init__(self, rate: float, burst_size: int):
        self.rate = rate
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def wait_for_token(self, tokens: int = 1) -> None:
        """Wait until tokens are available"""
        while not self.acquire(tokens):
            await asyncio.sleep(0.1)


class TaskQueue:
    """Intelligent task queue with priority and retry handling"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue = deque()
        self._retry_queue = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        self._not_full = asyncio.Condition(self._lock)
        self._size = 0
    
    async def put(self, task: Dict[str, Any], priority: int = 0) -> None:
        """Add a task to the queue"""
        async with self._not_full:
            while self._size >= self.max_size:
                await self._not_full.wait()
            
            task['priority'] = priority
            task['created_at'] = time.time()
            
            # Insert based on priority
            if priority > 0:
                self._queue.appendleft(task)
            else:
                self._queue.append(task)
            
            self._size += 1
            self._not_empty.notify()
    
    async def get(self) -> Dict[str, Any]:
        """Get a task from the queue"""
        async with self._not_empty:
            while self._size == 0:
                await self._not_empty.wait()
            
            # Try retry queue first
            if self._retry_queue:
                task = self._retry_queue.popleft()
            else:
                task = self._queue.popleft()
            
            self._size -= 1
            self._not_full.notify()
            return task
    
    async def put_retry(self, task: Dict[str, Any]) -> None:
        """Add a task to the retry queue"""
        async with self._lock:
            self._retry_queue.append(task)
            self._not_empty.notify()
    
    def qsize(self) -> int:
        """Get the current queue size"""
        return self._size


class ProgressTracker:
    """Track progress of parallel processing operations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all tracking metrics"""
        self.start_time = time.time()
        self.total_tasks = 0
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.retry_count = 0
        self.current_active = 0
        self.max_active = 0
        self._lock = threading.Lock()
    
    def set_total(self, total: int) -> None:
        """Set the total number of tasks"""
        with self._lock:
            self.total_tasks = total
    
    def task_started(self) -> None:
        """Mark a task as started"""
        with self._lock:
            self.current_active += 1
            self.max_active = max(self.max_active, self.current_active)
    
    def task_completed(self, success: bool, retry: bool = False) -> None:
        """Mark a task as completed"""
        with self._lock:
            self.current_active -= 1
            self.completed_tasks += 1
            
            if success:
                self.successful_tasks += 1
            else:
                self.failed_tasks += 1
            
            if retry:
                self.retry_count += 1
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress metrics"""
        with self._lock:
            elapsed = time.time() - self.start_time
            
            return {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": self.failed_tasks,
                "retry_count": self.retry_count,
                "current_active": self.current_active,
                "max_active": self.max_active,
                "elapsed_time": elapsed,
                "tasks_per_second": self.completed_tasks / elapsed if elapsed > 0 else 0,
                "success_rate": self.successful_tasks / self.completed_tasks if self.completed_tasks > 0 else 0,
                "completion_percentage": (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
            }


class ParallelProcessor:
    """
    High-performance parallel processing component for URL extraction.
    
    Features:
    - Configurable worker pools
    - Intelligent rate limiting
    - Progress tracking and monitoring
    - Error handling and retry logic
    - Resource usage monitoring
    - Batch processing capabilities
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        super().__init__()
        self.config = config or ParallelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            self.config.requests_per_second,
            self.config.burst_size
        )
        self.task_queue = TaskQueue(self.config.max_concurrent_tasks)
        self.progress_tracker = ProgressTracker()
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._worker_running = False
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_retries": 0,
            "average_task_time": 0.0,
            "peak_concurrency": 0
        }

    async def initialize(self) -> None:
        """Initialize the parallel processor"""
        try:
            self.logger.info("Initializing parallel processor")
            await self._start_workers()
            self.logger.info(f"Started {len(self._workers)} worker tasks")
        except Exception as e:
            self.logger.error(f"Failed to initialize parallel processor: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the parallel processor"""
        try:
            self.logger.info("Shutting down parallel processor")
            self._worker_running = False
            self._shutdown_event.set()
            
            # Wait for workers to finish
            if self._workers:
                await asyncio.gather(*self._workers, return_exceptions=True)
            
            self.logger.info("Parallel processor shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    async def process_urls(
        self,
        urls: List[str],
        extraction_func: Callable[[str, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """
        Process multiple URLs in parallel using the provided extraction function.
        
        Args:
            urls: List of URLs to process
            extraction_func: Function to extract data from each URL
            context: Optional context data to pass to extraction function
            
        Returns:
            BatchResult containing all results and statistics
        """
        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting batch processing: {batch_id} with {len(urls)} URLs")
            
            # Initialize progress tracking
            self.progress_tracker.reset()
            self.progress_tracker.set_total(len(urls))
            
            # Create tasks
            tasks = []
            for i, url in enumerate(urls):
                task = {
                    "task_id": f"{batch_id}_task_{i}",
                    "url": url,
                    "extraction_func": extraction_func,
                    "context": context or {},
                    "attempt": 1,
                    "max_attempts": self.config.max_retries + 1
                }
                tasks.append(task)
            
            # Queue all tasks
            for task in tasks:
                await self.task_queue.put(task)
            
            # Wait for completion
            await self._wait_for_completion(len(tasks))
            
            # Collect results
            results = await self._collect_results(batch_id)
            
            # Calculate final statistics
            total_time = time.time() - start_time
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            batch_result = BatchResult(
                batch_id=batch_id,
                total_tasks=len(urls),
                successful_tasks=successful,
                failed_tasks=failed,
                total_time=total_time,
                results=results,
                metadata={
                    "average_task_time": total_time / len(results) if results else 0,
                    "tasks_per_second": len(results) / total_time if total_time > 0 else 0,
                    "success_rate": successful / len(results) if results else 0
                }
            )
            
            self.logger.info(f"Batch processing complete: {batch_id} - "
                           f"{successful}/{len(urls)} successful in {total_time:.2f}s")
            
            return batch_result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {batch_id} - {str(e)}")
            return BatchResult(
                batch_id=batch_id,
                total_tasks=len(urls),
                successful_tasks=0,
                failed_tasks=len(urls),
                total_time=time.time() - start_time,
                errors=[str(e)]
            )

    async def _start_workers(self) -> None:
        """Start worker tasks for processing the queue"""
        self._worker_running = True
        self._workers = [
            asyncio.create_task(self._worker(f"worker_{i}"))
            for i in range(self.config.max_workers)
        ]

    async def _worker(self, worker_id: str) -> None:
        """Worker task that processes items from the queue"""
        self.logger.debug(f"Worker {worker_id} started")
        
        while self._worker_running and not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process the task
                await self._process_task(task, worker_id)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(0.1)
        
        self.logger.debug(f"Worker {worker_id} stopped")

    async def _process_task(self, task: Dict[str, Any], worker_id: str) -> None:
        """Process a single task"""
        task_id = task["task_id"]
        url = task["url"]
        
        try:
            # Rate limiting
            await self.rate_limiter.wait_for_token()
            
            # Track task start
            self.progress_tracker.task_started()
            
            # Execute the extraction function
            start_time = time.time()
            
            result = await asyncio.wait_for(
                task["extraction_func"](url, task["context"]),
                timeout=self.config.task_timeout
            )
            
            execution_time = time.time() - start_time
            
            # Create successful result
            task_result = TaskResult(
                task_id=task_id,
                url=url,
                success=True,
                data=result,
                execution_time=execution_time,
                attempt_count=task["attempt"],
                metadata={"worker_id": worker_id}
            )
            
            # Store result
            await self._store_result(task_result)
            
            # Track completion
            self.progress_tracker.task_completed(success=True)
            
            # Update statistics
            self.stats["total_processed"] += 1
            self.stats["total_successful"] += 1
            self._update_average_time(execution_time)
            
            self.logger.debug(f"Task {task_id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            
            # Handle retry logic
            if task["attempt"] < task["max_attempts"]:
                # Retry the task
                task["attempt"] += 1
                retry_delay = self.config.retry_backoff ** (task["attempt"] - 1)
                
                self.logger.warning(f"Task {task_id} failed (attempt {task['attempt']}), "
                                  f"retrying in {retry_delay:.1f}s: {str(e)}")
                
                # Schedule retry
                await asyncio.sleep(retry_delay)
                await self.task_queue.put_retry(task)
                
                # Track retry
                self.progress_tracker.task_completed(success=False, retry=True)
                self.stats["total_retries"] += 1
                
            else:
                # Final failure
                task_result = TaskResult(
                    task_id=task_id,
                    url=url,
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    attempt_count=task["attempt"],
                    metadata={"worker_id": worker_id}
                )
                
                await self._store_result(task_result)
                
                # Track failure
                self.progress_tracker.task_completed(success=False)
                self.stats["total_processed"] += 1
                self.stats["total_failed"] += 1
                
                self.logger.error(f"Task {task_id} failed permanently after "
                                f"{task['attempt']} attempts: {str(e)}")

    async def _store_result(self, result: TaskResult) -> None:
        """Store a task result"""
        # In a real implementation, this might store to a database or cache
        # For now, we'll store in memory
        if not hasattr(self, '_results'):
            self._results = {}
        self._results[result.task_id] = result

    async def _collect_results(self, batch_id: str) -> List[TaskResult]:
        """Collect all results for a batch"""
        if not hasattr(self, '_results'):
            return []
        
        # Filter results for this batch
        batch_results = [
            result for task_id, result in self._results.items()
            if task_id.startswith(batch_id)
        ]
        
        # Clean up stored results
        for result in batch_results:
            self._results.pop(result.task_id, None)
        
        return batch_results

    async def _wait_for_completion(self, total_tasks: int) -> None:
        """Wait for all tasks to complete"""
        completed = 0
        last_progress_report = 0
        
        while completed < total_tasks:
            progress = self.progress_tracker.get_progress()
            completed = progress["completed_tasks"]
            
            # Report progress periodically
            if completed - last_progress_report >= 10:
                self.logger.info(f"Progress: {completed}/{total_tasks} tasks completed "
                               f"({progress['completion_percentage']:.1f}%) - "
                               f"Success rate: {progress['success_rate']:.1f}%")
                last_progress_report = completed
                
                # Call progress callback if provided
                if self.config.progress_callback:
                    try:
                        self.config.progress_callback(progress)
                    except Exception as e:
                        self.logger.warning(f"Progress callback error: {str(e)}")
            
            await asyncio.sleep(0.5)

    def _update_average_time(self, execution_time: float) -> None:
        """Update the average task execution time"""
        if self.stats["total_processed"] == 1:
            self.stats["average_task_time"] = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats["average_task_time"] = (
                alpha * execution_time + 
                (1 - alpha) * self.stats["average_task_time"]
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        progress = self.progress_tracker.get_progress()
        
        return {
            **self.stats,
            **progress,
            "queue_size": self.task_queue.qsize(),
            "active_workers": len([w for w in self._workers if not w.done()]),
            "rate_limit_tokens": self.rate_limiter.tokens
        }

    @handle_errors_gracefully
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the parallel processor"""
        try:
            stats = self.get_stats()
            
            # Check system health
            healthy = True
            issues = []
            
            if stats["active_workers"] < self.config.max_workers:
                issues.append("Some workers are not active")
                healthy = False
            
            if stats["queue_size"] > self.config.max_concurrent_tasks * 0.8:
                issues.append("Queue is nearly full")
                healthy = False
            
            return {
                "status": "healthy" if healthy else "degraded",
                "active_workers": stats["active_workers"],
                "queue_size": stats["queue_size"],
                "total_processed": stats["total_processed"],
                "success_rate": stats.get("success_rate", 0),
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def process_urls_parallel(
        self,
        urls: List[str],
        extraction_func: Callable[[str, Dict[str, Any]], Any],
        context: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """
        Alias for process_urls method to match expected interface.
        Process multiple URLs in parallel using the provided extraction function.
        
        Args:
            urls: List of URLs to process
            extraction_func: Function to extract data from each URL
            context: Optional context data to pass to extraction function
            
        Returns:
            BatchResult containing all results and statistics
        """
        return await self.process_urls(urls, extraction_func, context)
    
    async def start_processing(self) -> None:
        """
        Start the parallel processing system.
        This method initializes workers and prepares the system for processing.
        """
        if not self._worker_running:
            await self.initialize()
            self._worker_running = True
            self.logger.info("Parallel processing started")
        else:
            self.logger.info("Parallel processing already running")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information for ongoing processing.
        
        Returns:
            Dictionary containing progress statistics
        """
        try:
            # Get progress from tracker
            progress = self.progress_tracker.get_progress()
            
            # Add processor-specific stats
            progress.update({
                "processor_stats": self.stats,
                "active_workers": len([w for w in self._workers if not w.done()]),
                "queue_size": self.task_queue.qsize(),
                "rate_limit_active": hasattr(self.rate_limiter, 'tokens') and self.rate_limiter.tokens < self.rate_limiter.burst_size,
                "worker_running": self._worker_running
            })
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Error getting progress: {e}")
            return {
                "error": str(e),
                "tasks_completed": 0,
                "tasks_total": 0,
                "completion_percentage": 0,
                "processor_stats": self.stats
            }
