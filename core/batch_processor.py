import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union, Awaitable
from collections import defaultdict, deque

class BatchRequest:
    """Represents a single request within a batch."""
    
    def __init__(self, 
                request_id: str,
                data: Any,
                priority: int = 0,
                metadata: Optional[Dict[str, Any]] = None):
        self.request_id = request_id
        self.data = data
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.processed_at: Optional[float] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.status = "pending"  # pending, processing, completed, failed
    
    def mark_as_processing(self) -> None:
        """Mark the request as being processed."""
        self.status = "processing"
        
    def mark_as_completed(self, result: Any) -> None:
        """Mark the request as completed with a result."""
        self.status = "completed"
        self.result = result
        self.processed_at = time.time()
        
    def mark_as_failed(self, error: Exception) -> None:
        """Mark the request as failed with an error."""
        self.status = "failed"
        self.error = error
        self.processed_at = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the request to a dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            "metadata": self.metadata,
            "priority": self.priority,
            "result": self.result,
            "error": str(self.error) if self.error else None
        }
        
    @property
    def waiting_time(self) -> float:
        """Get the time this request has been waiting in seconds."""
        if self.processed_at:
            return self.processed_at - self.created_at
        return time.time() - self.created_at

class Batch:
    """Represents a collection of requests to be processed together."""
    
    def __init__(self, 
                batch_id: str,
                max_size: int = 10,
                max_waiting_time: float = 5.0,
                metadata: Optional[Dict[str, Any]] = None):
        self.batch_id = batch_id
        self.max_size = max_size
        self.max_waiting_time = max_waiting_time
        self.metadata = metadata or {}
        self.requests: Dict[str, BatchRequest] = {}
        self.created_at = time.time()
        self.processed_at: Optional[float] = None
        self.status = "collecting"  # collecting, processing, completed
        
    def add_request(self, request: BatchRequest) -> bool:
        """Add a request to the batch. Returns True if added, False if batch is full."""
        if self.status != "collecting":
            return False
            
        if len(self.requests) >= self.max_size:
            return False
            
        self.requests[request.request_id] = request
        return True
        
    def mark_as_processing(self) -> None:
        """Mark the batch as being processed."""
        self.status = "processing"
        
    def mark_as_completed(self) -> None:
        """Mark the batch as completed."""
        self.status = "completed"
        self.processed_at = time.time()
        
    @property
    def is_ready(self) -> bool:
        """Check if the batch is ready to be processed."""
        if len(self.requests) >= self.max_size:
            return True
            
        if len(self.requests) > 0 and time.time() - self.created_at >= self.max_waiting_time:
            return True
            
        return False
        
    @property
    def size(self) -> int:
        """Get the number of requests in the batch."""
        return len(self.requests)
        
    @property
    def waiting_time(self) -> float:
        """Get the time this batch has been waiting in seconds."""
        if self.processed_at:
            return self.processed_at - self.created_at
        return time.time() - self.created_at
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the batch to a dictionary."""
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            "size": self.size,
            "metadata": self.metadata,
            "requests": {req_id: req.to_dict() for req_id, req in self.requests.items()}
        }
        
    def get_common_metadata(self) -> Dict[str, Any]:
        """Extract common metadata across all requests."""
        if not self.requests:
            return {}
            
        # Start with all metadata from the first request
        first_req = next(iter(self.requests.values()))
        common_metadata = dict(first_req.metadata)
        
        # Remove keys that aren't present in all requests or have different values
        for req in self.requests.values():
            for key in list(common_metadata.keys()):
                if key not in req.metadata or req.metadata[key] != common_metadata[key]:
                    common_metadata.pop(key, None)
                    
        return common_metadata
        
    def get_data_list(self) -> List[Any]:
        """Get a list of data from all requests in priority order."""
        # Sort requests by priority (highest first)
        sorted_requests = sorted(
            self.requests.values(), 
            key=lambda r: (-r.priority, r.created_at)
        )
        return [req.data for req in sorted_requests]

class BatchProcessor:
    """Processes batched requests for efficient handling."""
    
    def __init__(self, 
                 processor_fn: Callable[[List[Any], Dict[str, Any]], Awaitable[List[Any]]],
                 batch_size: int = 10,
                 max_waiting_time: float = 5.0,
                 max_concurrent_batches: int = 5,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batch processor.
        
        Args:
            processor_fn: Async function that processes a list of data items and returns a list of results
            batch_size: Maximum number of requests per batch
            max_waiting_time: Maximum time to wait before processing a batch even if not full (seconds)
            max_concurrent_batches: Maximum number of batches to process concurrently
            config: Additional configuration
        """
        self.processor_fn = processor_fn
        self.batch_size = batch_size
        self.max_waiting_time = max_waiting_time
        self.max_concurrent_batches = max_concurrent_batches
        self.config = config or {}
        
        self.logger = logging.getLogger("batch_processor")
        
        # Internal state
        self.current_batches: Dict[str, Batch] = {}
        self.batch_queue: deque = deque()
        self.request_futures: Dict[str, asyncio.Future] = {}
        self.processing_batches: Set[str] = set()
        self.batch_results: Dict[str, List[Any]] = {}
        
        # Tracking and statistics
        self.total_requests = 0
        self.total_batches = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Don't create async tasks during synchronous initialization
        # This will be handled in start() method
        self._loop = None
        self._processing_task = None
        self._is_running = False
        
    async def start(self):
        """Start the batch processor's async components."""
        if self._is_running:
            return
            
        # Get the current event loop
        self._loop = asyncio.get_running_loop()
        
        # Start background processing task
        self._processing_task = self._loop.create_task(self._process_batches())
        self._is_running = True
        
    async def add_request(self, 
                        data: Any, 
                        priority: int = 0,
                        metadata: Optional[Dict[str, Any]] = None,
                        request_id: Optional[str] = None) -> Tuple[str, asyncio.Future]:
        """
        Add a request to be batched and processed.
        
        Args:
            data: The data to process
            priority: Priority of the request (higher = processed first within batch)
            metadata: Additional metadata for the request
            request_id: Optional ID for the request, generated if not provided
            
        Returns:
            Tuple of (request_id, future) where future will resolve to the result
        """
        # Ensure the processor is started
        if not self._is_running:
            await self.start()
            
        self.total_requests += 1
        request_id = request_id or str(uuid.uuid4())
        
        # Create a new future for this request
        future = self._loop.create_future()
        self.request_futures[request_id] = future
        
        # Create the request
        request = BatchRequest(
            request_id=request_id,
            data=data,
            priority=priority,
            metadata=metadata
        )
        
        # Try to add the request to an existing batch
        batch_id = await self._add_to_batch(request)
        
        # Return the request ID and future
        return request_id, future
        
    async def _add_to_batch(self, request: BatchRequest) -> str:
        """Add a request to an appropriate batch, creating a new one if needed."""
        # Find an existing batch with room that matches this request's metadata
        for batch_id, batch in self.current_batches.items():
            if batch.status == "collecting" and batch.size < batch.max_size:
                # Check if metadata is compatible
                if self._is_compatible(request.metadata, batch.get_common_metadata()):
                    if batch.add_request(request):
                        self.logger.debug(f"Added request {request.request_id} to existing batch {batch_id}")
                        return batch_id
        
        # No compatible batch found, create a new one
        batch_id = str(uuid.uuid4())
        batch = Batch(
            batch_id=batch_id,
            max_size=self.batch_size,
            max_waiting_time=self.max_waiting_time
        )
        batch.add_request(request)
        self.current_batches[batch_id] = batch
        self.batch_queue.append(batch_id)
        self.total_batches += 1
        self.logger.debug(f"Created new batch {batch_id} for request {request.request_id}")
        return batch_id
        
    def _is_compatible(self, metadata1: Dict[str, Any], metadata2: Dict[str, Any]) -> bool:
        """Check if two metadata dictionaries are compatible for batching."""
        # By default, we consider requests compatible if they share the same 
        # model, task_type, and any other critical parameters
        critical_keys = self.config.get("critical_metadata_keys", 
                                       ["model", "task_type", "content_type"])
        
        for key in critical_keys:
            if key in metadata1 and key in metadata2 and metadata1[key] != metadata2[key]:
                return False
                
        return True
        
    async def _process_batches(self) -> None:
        """Background task that processes batches when they're ready."""
        while self._is_running:
            try:
                # Check for ready batches
                ready_batches = []
                
                # First, check batches at the front of the queue
                while self.batch_queue and len(self.processing_batches) < self.max_concurrent_batches:
                    batch_id = self.batch_queue[0]
                    batch = self.current_batches.get(batch_id)
                    
                    if not batch:
                        # Batch no longer exists, remove from queue
                        self.batch_queue.popleft()
                        continue
                        
                    if batch.is_ready:
                        # Batch is ready, remove from queue and process
                        self.batch_queue.popleft()
                        ready_batches.append(batch_id)
                        self.processing_batches.add(batch_id)
                        batch.mark_as_processing()
                    else:
                        # First batch isn't ready, so no further batches will be ready
                        break
                
                # Process ready batches concurrently
                if ready_batches:
                    # Create processing tasks for each batch
                    tasks = [self._process_batch(batch_id) for batch_id in ready_batches]
                    await asyncio.gather(*tasks)
                else:
                    # No batches ready, wait briefly before checking again
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing task: {e}")
                await asyncio.sleep(1)  # Wait a bit longer on error
                
    async def _process_batch(self, batch_id: str) -> None:
        """Process a single batch."""
        batch = self.current_batches.get(batch_id)
        if not batch:
            self.processing_batches.discard(batch_id)
            return
            
        self.logger.info(f"Processing batch {batch_id} with {batch.size} requests")
        
        try:
            # Extract data and common metadata
            data_list = batch.get_data_list()
            common_metadata = batch.get_common_metadata()
            
            # Process the batch
            try:
                results = await self.processor_fn(data_list, common_metadata)
                
                # Make sure we have the right number of results
                if len(results) != len(data_list):
                    raise ValueError(f"Expected {len(data_list)} results, got {len(results)}")
                    
                # Store the batch results
                self.batch_results[batch_id] = results
                
                # Update each request with its result
                for req_id, request in batch.requests.items():
                    idx = data_list.index(request.data)
                    request.mark_as_completed(results[idx])
                    self.successful_requests += 1
                    
                    # Resolve the future
                    future = self.request_futures.pop(req_id, None)
                    if future and not future.done():
                        future.set_result(results[idx])
                        
            except Exception as e:
                self.logger.error(f"Error processing batch {batch_id}: {e}")
                
                # Mark all requests as failed
                for req_id, request in batch.requests.items():
                    request.mark_as_failed(e)
                    self.failed_requests += 1
                    
                    # Resolve the future with an exception
                    future = self.request_futures.pop(req_id, None)
                    if future and not future.done():
                        future.set_exception(e)
                        
        except Exception as e:
            self.logger.error(f"Unexpected error processing batch {batch_id}: {e}")
            
        finally:
            # Clean up
            batch.mark_as_completed()
            self.processing_batches.discard(batch_id)
            
            # Keep completed batches for a while for inspection, but eventually remove them
            self._loop.create_task(self._cleanup_batch(batch_id))
            
    async def _cleanup_batch(self, batch_id: str) -> None:
        """Clean up a batch after a delay."""
        await asyncio.sleep(self.config.get("batch_retention_seconds", 300))  # Default: 5 minutes
        self.current_batches.pop(batch_id, None)
        self.batch_results.pop(batch_id, None)
        
    async def shutdown(self) -> None:
        """Shut down the batch processor."""
        self._is_running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            
        # Complete any pending futures with an error
        for future in self.request_futures.values():
            if not future.done():
                future.set_exception(RuntimeError("Batch processor shut down"))
                
        self.logger.info("Batch processor shut down")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "pending_requests": len(self.request_futures),
            "active_batches": len(self.current_batches),
            "queued_batches": len(self.batch_queue),
            "processing_batches": len(self.processing_batches)
        }
        
    async def wait_for_request(self, request_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific request to complete."""
        future = self.request_futures.get(request_id)
        if not future:
            raise ValueError(f"No pending request with ID {request_id}")
            
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {request_id} did not complete within {timeout} seconds")