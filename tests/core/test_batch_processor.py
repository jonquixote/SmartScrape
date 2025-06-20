import pytest
import asyncio
from unittest.mock import AsyncMock
from core.batch_processor import BatchProcessor, BatchRequest, Batch

# Define a mock processing function for testing
async def mock_processor(data_list, metadata):
    # Simply return each data item with "processed" prefix
    return [f"processed {item}" for item in data_list]

# Define a slow processor for timeout testing
async def slow_processor(data_list, metadata):
    await asyncio.sleep(0.5)
    return [f"processed {item}" for item in data_list]

@pytest.fixture
async def mock_processor_fn():
    """Create a mock processor function for testing."""
    async def process_batch(data_list, metadata):
        # Simple mock that returns data with a "processed" prefix
        return [f"processed_{item}" for item in data_list]
    
    return AsyncMock(side_effect=process_batch)

@pytest.fixture
async def batch_processor(mock_processor_fn):
    """Create a batch processor for testing."""
    processor = BatchProcessor(
        processor_fn=mock_processor_fn,
        batch_size=3,
        max_waiting_time=0.5,
        max_concurrent_batches=2
    )
    
    yield processor
    
    # Clean up
    processor.shutdown()
    await asyncio.sleep(0.1)  # Give time for shutdown to complete

@pytest.fixture
def slow_batch_processor():
    processor = BatchProcessor(
        processor_fn=slow_processor,
        batch_size=3,
        max_waiting_time=0.5,
        max_concurrent_batches=2
    )
    yield processor
    # Make sure to clean up
    asyncio.run(processor.shutdown())

@pytest.mark.asyncio
async def test_batch_request():
    """Test the BatchRequest class."""
    request = BatchRequest("req1", "test data", priority=1, metadata={"type": "test"})
    
    # Test initial state
    assert request.request_id == "req1"
    assert request.data == "test data"
    assert request.priority == 1
    assert request.metadata == {"type": "test"}
    assert request.status == "pending"
    
    # Test state transitions
    request.mark_as_processing()
    assert request.status == "processing"
    
    request.mark_as_completed("result")
    assert request.status == "completed"
    assert request.result == "result"
    
    # Test to_dict
    data = request.to_dict()
    assert data["request_id"] == "req1"
    assert data["status"] == "completed"
    assert data["result"] == "result"

@pytest.mark.asyncio
async def test_batch():
    """Test the Batch class."""
    batch = Batch("batch1", max_size=2)
    
    # Test initial state
    assert batch.batch_id == "batch1"
    assert batch.max_size == 2
    assert batch.status == "collecting"
    assert batch.size == 0
    
    # Test adding requests
    req1 = BatchRequest("req1", "data1")
    req2 = BatchRequest("req2", "data2")
    req3 = BatchRequest("req3", "data3")
    
    assert batch.add_request(req1) == True
    assert batch.size == 1
    
    assert batch.add_request(req2) == True
    assert batch.size == 2
    
    # Batch is now full
    assert batch.add_request(req3) == False
    assert batch.size == 2
    
    # Test readiness
    assert batch.is_ready == True
    
    # Test state transitions
    batch.mark_as_processing()
    assert batch.status == "processing"
    
    batch.mark_as_completed()
    assert batch.status == "completed"
    
    # Test data extraction
    batch = Batch("batch2", max_size=3)
    req1 = BatchRequest("req1", "data1", priority=1)
    req2 = BatchRequest("req2", "data2", priority=2)  # Higher priority
    req3 = BatchRequest("req3", "data3", priority=0)
    
    batch.add_request(req1)
    batch.add_request(req2)
    batch.add_request(req3)
    
    # Data should be ordered by priority (highest first)
    assert batch.get_data_list() == ["data2", "data1", "data3"]
    
    # Test metadata extraction
    req1.metadata = {"model": "gpt-4", "task": "summarization"}
    req2.metadata = {"model": "gpt-4", "task": "translation"}
    req3.metadata = {"model": "gpt-4", "task": "summarization", "extra": "value"}
    
    common = batch.get_common_metadata()
    assert common == {"model": "gpt-4"}  # Only model is common to all

@pytest.mark.asyncio
async def test_batch_processor_single_request(batch_processor):
    """Test processing a single request."""
    request_id, future = await batch_processor.add_request("test item")
    
    # Wait for processing to complete
    result = await future
    
    assert result == "processed test item"
    assert batch_processor.successful_requests == 1
    assert batch_processor.failed_requests == 0

@pytest.mark.asyncio
async def test_batch_processor_multiple_requests(batch_processor):
    """Test processing multiple requests that should be batched together."""
    # Add 3 requests that should be batched together
    futures = []
    for i in range(3):
        _, future = await batch_processor.add_request(f"item{i}")
        futures.append(future)
    
    # Wait for all to complete
    results = await asyncio.gather(*futures)
    
    # Verify results
    assert len(results) == 3
    assert all(r.startswith("processed item") for r in results)
    assert batch_processor.successful_requests == 3
    
    # Check that they were processed as a single batch
    stats = batch_processor.get_stats()
    assert stats["total_batches"] == 1

@pytest.mark.asyncio
async def test_batch_processor_auto_batching_by_time(slow_batch_processor):
    """Test that the processor automatically processes a batch after max_waiting_time."""
    # Add 2 requests (below the batch size of 3)
    futures = []
    for i in range(2):
        _, future = await slow_batch_processor.add_request(f"item{i}")
        futures.append(future)
    
    # Wait for the batch to be processed automatically (should take about 0.5s + processing time)
    results = await asyncio.gather(*futures)
    
    # Verify results
    assert len(results) == 2
    assert slow_batch_processor.successful_requests == 2
    
    # Check that they were processed as a single batch
    stats = slow_batch_processor.get_stats()
    assert stats["total_batches"] == 1

@pytest.mark.asyncio
async def test_batch_processor_metadata_compatibility(batch_processor):
    """Test that requests with compatible metadata are batched together."""
    # Add 2 requests with compatible metadata
    futures = []
    for i in range(2):
        _, future = await batch_processor.add_request(
            f"item{i}", 
            metadata={"model": "gpt-4", "task_type": "summarization"}
        )
        futures.append(future)
    
    # Add 1 request with incompatible metadata
    _, future3 = await batch_processor.add_request(
        "item2", 
        metadata={"model": "gpt-3.5-turbo", "task_type": "summarization"}
    )
    futures.append(future3)
    
    # Wait for all to complete
    results = await asyncio.gather(*futures)
    
    # Verify results
    assert len(results) == 3
    
    # Should have created 2 batches (one for compatible, one for incompatible)
    stats = batch_processor.get_stats()
    assert stats["total_batches"] == 2

@pytest.mark.asyncio
async def test_batch_processor_priority(batch_processor):
    """Test that requests with higher priority are processed first within a batch."""
    # Add 3 requests with different priorities
    _, future1 = await batch_processor.add_request("low", priority=0)
    _, future2 = await batch_processor.add_request("high", priority=2)
    _, future3 = await batch_processor.add_request("medium", priority=1)
    
    # Wait for all to complete
    results = await asyncio.gather(future1, future2, future3)
    
    # The order of results depends on the implementation of mock_processor
    # In our implementation, the processor preserves the order of the input list
    # which should be ordered by priority
    
    # Verify that all requests were processed
    assert batch_processor.successful_requests == 3
    assert batch_processor.total_batches == 1

@pytest.mark.asyncio
async def test_batch_processor_error_handling(batch_processor):
    """Test error handling in the batch processor."""
    # Override the processor function to throw an error
    original_fn = batch_processor.processor_fn
    
    async def error_processor(data_list, metadata):
        raise ValueError("Test error")
    
    batch_processor.processor_fn = error_processor
    
    # Add a request that should fail
    _, future = await batch_processor.add_request("test")
    
    # Wait for the request to complete (should fail)
    with pytest.raises(ValueError, match="Test error"):
        await future
    
    # Verify the error was tracked
    assert batch_processor.failed_requests == 1
    
    # Restore the original processor function
    batch_processor.processor_fn = original_fn

@pytest.mark.asyncio
async def test_batch_processor_concurrent_batches(slow_batch_processor):
    """Test processing multiple batches concurrently."""
    # Add 6 requests that should be split into 2 batches
    futures = []
    for i in range(6):
        # Add different metadata to force separate batches
        metadata = {"group": i // 3}
        _, future = await slow_batch_processor.add_request(f"item{i}", metadata=metadata)
        futures.append(future)
    
    # Wait for all to complete
    results = await asyncio.gather(*futures)
    
    # Verify results
    assert len(results) == 6
    assert all(r.startswith("processed item") for r in results)
    
    # Should have created 2 batches
    stats = slow_batch_processor.get_stats()
    assert stats["total_batches"] == 2

@pytest.mark.asyncio
async def test_add_request(batch_processor, mock_processor_fn):
    """Test adding requests to the batch processor."""
    # Add a single request
    request_id, future = await batch_processor.add_request("test_data")
    
    # Should return a valid request ID and future
    assert request_id is not None
    assert isinstance(future, asyncio.Future)
    assert not future.done()
    
    # Wait for processing to complete
    await asyncio.sleep(0.6)  # Exceeds max_waiting_time
    
    # Future should now be complete
    assert future.done()
    
    # Check result
    result = future.result()
    assert result == "processed_test_data"
    
    # The mock processor function should have been called
    mock_processor_fn.assert_called_once()

@pytest.mark.asyncio
async def test_batch_grouping(batch_processor, mock_processor_fn):
    """Test that requests are properly grouped into batches."""
    # Reset mock call count
    mock_processor_fn.reset_mock()
    
    # Add multiple requests that should go in the same batch
    futures = []
    for i in range(3):  # batch_size is 3
        _, future = await batch_processor.add_request(f"data_{i}")
        futures.append(future)
    
    # Wait for processing to complete
    await asyncio.sleep(0.1)  # Should process immediately when batch is full
    
    # All futures should be complete
    for future in futures:
        assert future.done()
    
    # The processor function should have been called exactly once with a batch of 3
    mock_processor_fn.assert_called_once()
    args, _ = mock_processor_fn.call_args
    assert len(args[0]) == 3

@pytest.mark.asyncio
async def test_request_timeout(batch_processor, mock_processor_fn):
    """Test that requests are processed after timeout even if batch isn't full."""
    # Reset mock call count
    mock_processor_fn.reset_mock()
    
    # Add a single request
    _, future = await batch_processor.add_request("timeout_test")
    
    # Batch shouldn't be processed immediately (not full)
    await asyncio.sleep(0.1)
    assert not future.done()
    assert mock_processor_fn.call_count == 0
    
    # After timeout, batch should be processed
    await asyncio.sleep(0.5)  # Exceeds max_waiting_time
    assert future.done()
    mock_processor_fn.assert_called_once()

@pytest.mark.asyncio
async def test_request_priority(batch_processor, mock_processor_fn):
    """Test that request priorities are respected."""
    # Configure mock to examine input order
    inputs_received = []
    
    async def track_inputs(data_list, metadata):
        inputs_received.extend(data_list)
        return [f"processed_{item}" for item in data_list]
    
    mock_processor_fn.side_effect = track_inputs
    
    # Add a low-priority request
    await batch_processor.add_request("low_priority", priority=0)
    
    # Add a high-priority request
    await batch_processor.add_request("high_priority", priority=10)
    
    # Add a medium-priority request
    await batch_processor.add_request("medium_priority", priority=5)
    
    # Wait for processing to complete
    await asyncio.sleep(0.1)
    
    # Check the order of processed items
    # Higher priority items should be processed first
    assert inputs_received[0] == "high_priority"
    assert inputs_received[1] == "medium_priority"
    assert inputs_received[2] == "low_priority"

@pytest.mark.asyncio
async def test_concurrent_batches(batch_processor, mock_processor_fn):
    """Test that multiple batches can be processed concurrently."""
    # Slow down the processor function to test concurrency
    slow_future = asyncio.Future()
    
    async def slow_process(data_list, metadata):
        if "batch1" in data_list[0]:
            # Slow down first batch
            await asyncio.sleep(0.5)
            return [f"batch1_processed_{item}" for item in data_list]
        else:
            # Let second batch complete quickly
            return [f"batch2_processed_{item}" for item in data_list]
    
    mock_processor_fn.side_effect = slow_process
    
    # Add first batch requests
    batch1_futures = []
    for i in range(3):
        _, future = await batch_processor.add_request(f"batch1_item_{i}")
        batch1_futures.append(future)
    
    # Add second batch requests
    batch2_futures = []
    for i in range(3):
        _, future = await batch_processor.add_request(f"batch2_item_{i}")
        batch2_futures.append(future)
    
    # Wait for processing to start
    await asyncio.sleep(0.1)
    
    # Second batch should complete before first batch
    assert not all(future.done() for future in batch1_futures)
    await asyncio.sleep(0.1)
    assert all(future.done() for future in batch2_futures)
    
    # Wait for all processing to complete
    await asyncio.sleep(0.5)
    assert all(future.done() for future in batch1_futures)

@pytest.mark.asyncio
async def test_error_handling(batch_processor):
    """Test that errors in processing are handled properly."""
    # Create a processor function that raises an exception
    async def error_processor(data_list, metadata):
        raise ValueError("Test error")
    
    # Use the error processor
    batch_processor.processor_fn = AsyncMock(side_effect=error_processor)
    
    # Add a request
    _, future = await batch_processor.add_request("error_test")
    
    # Wait for processing to complete
    await asyncio.sleep(0.6)
    
    # Future should be done but with an exception
    assert future.done()
    with pytest.raises(ValueError):
        future.result()

@pytest.mark.asyncio
async def test_get_batch_status(batch_processor, mock_processor_fn):
    """Test getting batch status information."""
    # Add a request
    request_id, _ = await batch_processor.add_request("status_test")
    
    # Get status before processing
    batch_id = batch_processor.get_batch_id_for_request(request_id)
    assert batch_id is not None
    
    status = batch_processor.get_batch_status(batch_id)
    assert status is not None
    assert status["status"] in ["collecting", "processing"]
    assert status["size"] == 1
    
    # Wait for processing to complete
    await asyncio.sleep(0.6)
    
    # Get status after processing
    status = batch_processor.get_batch_status(batch_id)
    assert status["status"] == "completed"