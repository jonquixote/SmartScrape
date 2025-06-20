import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from core.pipeline.monitoring import PipelineMonitor
from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineMetrics, StageMetrics


@pytest.fixture
def mock_context():
    """Create a mock pipeline context for testing."""
    context = MagicMock(spec=PipelineContext)
    context.metadata = {
        "pipeline_name": "test_pipeline",
        "start_time": time.time() - 1,  # Started 1 second ago
        "end_time": None,
        "current_stage": "test_stage",
        "completed_stages": set(),
        "stage_metrics": {
            "test_stage": {
                "start_time": time.time() - 0.5,  # Started 0.5 seconds ago
                "end_time": None,
                "status": "running",
                "execution_time": 0
            }
        },
        "errors": {}
    }
    
    context.get_metrics.return_value = {
        "pipeline_name": "test_pipeline",
        "total_time": 1.0,
        "stages": context.metadata["stage_metrics"],
        "successful_stages": 0,
        "total_stages": 1,
        "has_errors": False
    }
    
    context.has_errors.return_value = False
    
    return context


@pytest.fixture
def pipeline_monitor():
    """Create a pipeline monitor instance for testing."""
    return PipelineMonitor(retention_period=1, enable_prometheus=False)


def test_monitor_initialization(pipeline_monitor):
    """Test that the monitor initializes correctly."""
    assert pipeline_monitor is not None
    assert pipeline_monitor.retention_period == 1
    assert not pipeline_monitor.enable_prometheus
    assert isinstance(pipeline_monitor.pipeline_metrics, dict)
    assert isinstance(pipeline_monitor.stage_metrics, dict)
    assert isinstance(pipeline_monitor.historical_data, dict)
    assert isinstance(pipeline_monitor.error_counts, dict)
    assert isinstance(pipeline_monitor.active_pipelines, dict)


def test_register_pipeline_start(pipeline_monitor, mock_context):
    """Test registering the start of a pipeline execution."""
    pipeline_id = "test-pipeline-123"
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    
    # Check active pipelines
    assert pipeline_id in pipeline_monitor.active_pipelines
    active_pipeline = pipeline_monitor.active_pipelines[pipeline_id]
    assert active_pipeline["pipeline_name"] == "test_pipeline"
    assert "start_time" in active_pipeline
    assert active_pipeline["current_stage"] == "test_stage"


def test_register_pipeline_end(pipeline_monitor, mock_context):
    """Test registering the end of a pipeline execution."""
    pipeline_id = "test-pipeline-123"
    
    # First register the start
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    
    # Then register the end
    pipeline_monitor.register_pipeline_end(pipeline_id, mock_context)
    
    # Pipeline should be removed from active pipelines
    assert pipeline_id not in pipeline_monitor.active_pipelines
    
    # Pipeline metrics should be updated
    assert "test_pipeline" in pipeline_monitor.pipeline_metrics
    pipeline_metrics = pipeline_monitor.pipeline_metrics["test_pipeline"]
    assert pipeline_metrics["total_executions"] == 1
    assert pipeline_metrics["successful_executions"] == 1
    assert pipeline_metrics["failed_executions"] == 0
    assert pipeline_metrics["total_duration"] > 0
    assert pipeline_metrics["average_duration"] > 0
    
    # Historical data should be updated
    assert "test_pipeline" in pipeline_monitor.historical_data
    assert len(pipeline_monitor.historical_data["test_pipeline"]) == 1
    
    # Stage metrics should be updated
    stage_key = "test_pipeline:test_stage"
    assert stage_key in pipeline_monitor.stage_metrics
    stage_metrics = pipeline_monitor.stage_metrics[stage_key]
    assert stage_metrics["total_executions"] == 1


def test_register_stage_transition(pipeline_monitor, mock_context):
    """Test registering a stage transition."""
    pipeline_id = "test-pipeline-123"
    
    # First register the start
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    
    # Then register a stage transition
    pipeline_monitor.register_stage_transition(pipeline_id, "test_stage", True)
    
    # Check that stages completed is incremented
    assert pipeline_monitor.active_pipelines[pipeline_id]["stages_completed"] == 1
    assert pipeline_monitor.active_pipelines[pipeline_id]["current_stage"] is None


def test_register_stage_start(pipeline_monitor, mock_context):
    """Test registering the start of a stage."""
    pipeline_id = "test-pipeline-123"
    
    # First register the pipeline start
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    
    # Then register a stage start
    pipeline_monitor.register_stage_start(pipeline_id, "new_stage")
    
    # Check current stage
    assert pipeline_monitor.active_pipelines[pipeline_id]["current_stage"] == "new_stage"


def test_real_time_metrics(pipeline_monitor, mock_context):
    """Test getting real-time metrics."""
    pipeline_id = "test-pipeline-123"
    
    # Register an active pipeline
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    
    # Get real-time metrics
    metrics = pipeline_monitor.real_time_metrics()
    
    # Check metrics structure
    assert "active_pipelines" in metrics
    assert metrics["active_pipelines"] == 1
    assert "pipelines" in metrics
    assert len(metrics["pipelines"]) == 1
    assert "resource_utilization" in metrics
    assert "timestamp" in metrics


def test_performance_history(pipeline_monitor, mock_context):
    """Test getting performance history for a pipeline."""
    pipeline_id = "test-pipeline-123"
    
    # Register pipeline execution
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    pipeline_monitor.register_pipeline_end(pipeline_id, mock_context)
    
    # Get performance history
    history = pipeline_monitor.performance_history("test_pipeline")
    
    # Check history structure
    assert history["pipeline_id"] == "test_pipeline"
    assert history["total_executions"] == 1
    assert history["success_rate"] == 1.0
    assert history["average_duration"] > 0
    assert "history" in history


def test_stage_statistics(pipeline_monitor, mock_context):
    """Test getting statistics for a specific stage."""
    pipeline_id = "test-pipeline-123"
    
    # Register pipeline execution
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    pipeline_monitor.register_pipeline_end(pipeline_id, mock_context)
    
    # Get stage statistics
    stage_key = "test_pipeline:test_stage"
    stats = pipeline_monitor.stage_statistics(stage_key)
    
    # Check statistics structure
    assert stats["stage_id"] == stage_key
    assert stats["total_executions"] == 1
    assert stats["success_rate"] > 0
    assert stats["average_duration"] >= 0


def test_error_analysis(pipeline_monitor):
    """Test error analysis functionality."""
    # Add some error data
    pipeline_monitor.error_counts["test_pipeline"]["timeout"] = 2
    pipeline_monitor.error_counts["test_pipeline"]["validation"] = 1
    
    # Get error analysis
    analysis = pipeline_monitor.error_analysis()
    
    # Check analysis structure
    assert "pipelines" in analysis
    assert "test_pipeline" in analysis["pipelines"]
    assert analysis["pipelines"]["test_pipeline"]["total_errors"] == 3
    assert "error_types" in analysis["pipelines"]["test_pipeline"]
    assert "error_distribution" in analysis["pipelines"]["test_pipeline"]
    assert "global_stats" in analysis


def test_resource_utilization(pipeline_monitor):
    """Test resource utilization monitoring."""
    # Get resource utilization
    utilization = pipeline_monitor.resource_utilization(minutes=5)
    
    # Check utilization structure
    assert "timespan_minutes" in utilization
    assert utilization["timespan_minutes"] == 5
    assert "sample_count" in utilization
    assert "history" in utilization
    assert "statistics" in utilization


def test_export_metrics(pipeline_monitor, mock_context):
    """Test exporting metrics in different formats."""
    pipeline_id = "test-pipeline-123"
    
    # Register pipeline execution
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    pipeline_monitor.register_pipeline_end(pipeline_id, mock_context)
    
    # Export metrics as JSON
    json_metrics = pipeline_monitor.export_metrics(format_type="json")
    
    # Check that metrics can be parsed
    import json
    metrics_data = json.loads(json_metrics)
    
    # Check metrics structure
    assert "pipelines" in metrics_data
    assert "stages" in metrics_data
    assert "errors" in metrics_data
    assert "active_pipelines" in metrics_data
    assert "resource_utilization" in metrics_data
    assert "timestamp" in metrics_data


def test_reset_metrics(pipeline_monitor, mock_context):
    """Test resetting metrics."""
    pipeline_id = "test-pipeline-123"
    
    # Register pipeline execution
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    pipeline_monitor.register_pipeline_end(pipeline_id, mock_context)
    
    # Verify metrics exist
    assert "test_pipeline" in pipeline_monitor.pipeline_metrics
    
    # Reset metrics for specific pipeline
    pipeline_monitor.reset_metrics("test_pipeline")
    
    # Verify metrics were reset
    assert "test_pipeline" not in pipeline_monitor.pipeline_metrics
    assert "test_pipeline" not in pipeline_monitor.historical_data
    
    # Register another execution
    pipeline_monitor.register_pipeline_start(pipeline_id, mock_context)
    pipeline_monitor.register_pipeline_end(pipeline_id, mock_context)
    
    # Reset all metrics
    pipeline_monitor.reset_metrics()
    
    # Verify all metrics were reset
    assert len(pipeline_monitor.pipeline_metrics) == 0
    assert len(pipeline_monitor.stage_metrics) == 0
    assert len(pipeline_monitor.error_counts) == 0
    assert len(pipeline_monitor.historical_data) == 0


@pytest.mark.asyncio
async def test_background_tasks(pipeline_monitor):
    """Test that background tasks run correctly."""
    # Create and keep track of a future
    monitor_task = asyncio.create_task(pipeline_monitor._monitor_resources())
    maintain_task = asyncio.create_task(pipeline_monitor._maintain_metrics())
    
    # Let the tasks run for a short period
    await asyncio.sleep(0.1)
    
    # Cancel the tasks
    monitor_task.cancel()
    maintain_task.cancel()
    
    # Wait for the tasks to be cancelled
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
        
    try:
        await maintain_task
    except asyncio.CancelledError:
        pass