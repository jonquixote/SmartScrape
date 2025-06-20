import pytest
import os
import time
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from core.pipeline.visualization import PipelineVisualizer


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for visualization output."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def visualizer(temp_output_dir):
    """Create a pipeline visualizer instance for testing."""
    return PipelineVisualizer(output_dir=temp_output_dir)


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = MagicMock()
    pipeline.name = "test_pipeline"
    
    # Create mock stages
    stage1 = MagicMock()
    stage1.name = "fetch_data"
    stage1.config = {"url": "https://example.com/api"}
    
    stage2 = MagicMock()
    stage2.name = "process_data"
    stage2.config = {"format": "json"}
    
    stage3 = MagicMock()
    stage3.name = "store_results"
    stage3.config = {"output_file": "results.json"}
    
    # Add stages to pipeline
    pipeline.stages = [stage1, stage2, stage3]
    
    return pipeline


@pytest.fixture
def mock_metrics():
    """Create mock pipeline metrics for testing."""
    current_time = time.time()
    
    return {
        "pipelines": {
            "test_pipeline": {
                "total_executions": 10,
                "successful_executions": 8,
                "failed_executions": 2,
                "total_duration": 50.5,
                "average_duration": 5.05,
                "execution_history": [
                    {
                        "pipeline_id": "test-123",
                        "duration": 5.2,
                        "success": True,
                        "timestamp": current_time - 3600,
                        "stages": {
                            "fetch_data": {"duration": 1.5, "status": "success"},
                            "process_data": {"duration": 2.5, "status": "success"},
                            "store_results": {"duration": 1.2, "status": "success"}
                        }
                    },
                    {
                        "pipeline_id": "test-124",
                        "duration": 6.1,
                        "success": False,
                        "timestamp": current_time - 1800,
                        "stages": {
                            "fetch_data": {"duration": 1.6, "status": "success"},
                            "process_data": {"duration": 2.8, "status": "success"},
                            "store_results": {"duration": 1.7, "status": "failed"}
                        },
                        "error_count": 1
                    }
                ]
            }
        }
    }


@pytest.fixture
def mock_execution_log():
    """Create mock execution log for testing."""
    return {
        "pipeline_name": "test_pipeline",
        "start_time": time.time() - 10,
        "end_time": time.time(),
        "stages": {
            "fetch_data": {
                "start_time": time.time() - 10,
                "end_time": time.time() - 8,
                "status": "success",
                "execution_time": 2.0
            },
            "process_data": {
                "start_time": time.time() - 8,
                "end_time": time.time() - 4,
                "status": "success",
                "execution_time": 4.0
            },
            "store_results": {
                "start_time": time.time() - 4,
                "end_time": time.time(),
                "status": "success",
                "execution_time": 4.0
            }
        }
    }


@pytest.fixture
def mock_monitor_data():
    """Create mock monitor data for testing."""
    current_time = time.time()
    
    return {
        "active_pipelines": [
            {
                "id": "active-pipeline-123",
                "name": "test_pipeline",
                "runtime": 45.2,
                "progress": 66.7,
                "current_stage": "process_data"
            }
        ],
        "resource_utilization": {
            "current": {
                "memory_percent": 42.5,
                "cpu_percent": 28.3
            },
            "history": [
                {
                    "timestamp": current_time - 3600,
                    "memory_percent": 35.0,
                    "cpu_percent": 20.1
                },
                {
                    "timestamp": current_time - 1800,
                    "memory_percent": 40.2,
                    "cpu_percent": 25.3
                },
                {
                    "timestamp": current_time,
                    "memory_percent": 42.5,
                    "cpu_percent": 28.3
                }
            ]
        },
        "error_analysis": {
            "global_stats": {
                "total_errors": 10,
                "error_types": {
                    "timeout": 4,
                    "connection": 3,
                    "validation": 2,
                    "unknown": 1
                },
                "error_distribution": {
                    "timeout": 40,
                    "connection": 30,
                    "validation": 20,
                    "unknown": 10
                }
            }
        },
        "recent_executions": [
            {
                "pipeline_name": "test_pipeline",
                "success": True,
                "duration": 5.2,
                "timestamp": "2025-05-04 14:30:00",
                "stages_completed": 3,
                "stages_total": 3
            },
            {
                "pipeline_name": "test_pipeline",
                "success": False,
                "duration": 6.1,
                "timestamp": "2025-05-04 14:00:00",
                "stages_completed": 2,
                "stages_total": 3
            }
        ]
    }


def test_visualizer_initialization(visualizer, temp_output_dir):
    """Test that the visualizer initializes correctly."""
    assert visualizer is not None
    assert visualizer.output_dir == temp_output_dir
    assert os.path.exists(temp_output_dir)
    
    # Check template directory creation
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(temp_output_dir))), 
                               'core', 'pipeline', 'templates')
    assert os.path.exists(template_dir)


@pytest.mark.skipif(not hasattr(PipelineVisualizer, 'has_networkx') or 
                   not PipelineVisualizer.has_networkx, 
                   reason="NetworkX not available")
def test_generate_flow_diagram(visualizer, mock_pipeline, temp_output_dir):
    """Test generating a flow diagram."""
    output_path = visualizer.generate_flow_diagram(mock_pipeline, output_format='svg')
    
    # Check that file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith('.svg')
    
    # Check with config details
    output_path_with_config = visualizer.generate_flow_diagram(
        mock_pipeline, output_format='svg', show_config=True
    )
    assert output_path_with_config is not None
    assert os.path.exists(output_path_with_config)


@pytest.mark.skipif(not hasattr(PipelineVisualizer, 'has_matplotlib') or 
                   not PipelineVisualizer.has_matplotlib, 
                   reason="Matplotlib not available")
def test_create_performance_chart(visualizer, mock_metrics, temp_output_dir):
    """Test creating a performance chart."""
    output_path = visualizer.create_performance_chart(mock_metrics, output_format='svg')
    
    # Check that file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith('.svg')


@pytest.mark.skipif(not hasattr(PipelineVisualizer, 'has_matplotlib') or 
                   not PipelineVisualizer.has_matplotlib, 
                   reason="Matplotlib not available")
def test_generate_execution_timeline(visualizer, mock_execution_log, temp_output_dir):
    """Test generating an execution timeline."""
    output_path = visualizer.generate_execution_timeline(mock_execution_log, output_format='svg')
    
    # Check that file was created
    assert output_path is not None
    assert os.path.exists(output_path)
    assert output_path.endswith('.svg')


@pytest.mark.skipif(not hasattr(PipelineVisualizer, 'has_jinja2') or 
                   not PipelineVisualizer.has_jinja2, 
                   reason="Jinja2 not available")
def test_create_dashboard(visualizer, mock_metrics, mock_monitor_data, temp_output_dir):
    """Test creating a comprehensive dashboard."""
    # Extract pipeline data from metrics
    pipeline_data = []
    for name, metrics in mock_metrics.get('pipelines', {}).items():
        pipeline_data.append({
            'name': name,
            **metrics
        })
    
    # Create dashboard
    dashboard_path = visualizer.create_dashboard(
        pipeline_data, mock_monitor_data, temp_output_dir
    )
    
    # Check that dashboard was created
    assert dashboard_path is not None
    assert os.path.exists(dashboard_path)
    
    # Check dashboard directory structure
    dashboard_dir = os.path.join(temp_output_dir, 'dashboard')
    assert os.path.exists(dashboard_dir)
    
    # Dashboard HTML should exist
    assert os.path.exists(os.path.join(dashboard_dir, 'dashboard.html'))
    
    # Should have at least one pipeline detail page
    detail_files = [f for f in os.listdir(dashboard_dir) if f.endswith('_detail.html')]
    assert len(detail_files) > 0
    
    # Charts directory should exist
    charts_dir = os.path.join(dashboard_dir, 'charts')
    assert os.path.exists(charts_dir)


def test_generate_resource_chart(visualizer, mock_monitor_data, temp_output_dir):
    """Test generating a resource utilization chart."""
    output_path = visualizer._generate_resource_chart(
        mock_monitor_data, temp_output_dir
    )
    
    # Check if matplotlib is available
    if hasattr(visualizer, 'has_matplotlib') and visualizer.has_matplotlib:
        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith('.svg')
    else:
        assert output_path is None


def test_generate_error_chart(visualizer, mock_monitor_data, temp_output_dir):
    """Test generating an error distribution chart."""
    output_path = visualizer._generate_error_chart(
        mock_monitor_data, temp_output_dir
    )
    
    # Check if matplotlib is available
    if hasattr(visualizer, 'has_matplotlib') and visualizer.has_matplotlib:
        assert output_path is not None
        assert os.path.exists(output_path)
        assert output_path.endswith('.svg')
    else:
        assert output_path is None


def test_extract_recent_executions(visualizer, mock_metrics, mock_monitor_data):
    """Test extracting recent executions data."""
    # Extract from monitor data
    executions = visualizer._extract_recent_executions([], mock_monitor_data)
    assert len(executions) > 0
    
    # Extract from pipeline data
    pipeline_data = []
    for name, metrics in mock_metrics.get('pipelines', {}).items():
        pipeline_data.append({
            'name': name,
            **metrics
        })
    
    executions = visualizer._extract_recent_executions(pipeline_data, None)
    assert len(executions) > 0
    
    # Check first execution data
    first_execution = executions[0]
    assert "pipeline_name" in first_execution
    assert "success" in first_execution
    assert "duration" in first_execution
    assert "timestamp" in first_execution
    assert "stages_completed" in first_execution
    assert "stages_total" in first_execution


def test_generate_pipeline_detail_page(visualizer, mock_metrics, mock_monitor_data, temp_output_dir):
    """Test generating a pipeline detail page."""
    # Extract first pipeline
    pipeline_data = None
    for name, metrics in mock_metrics.get('pipelines', {}).items():
        pipeline_data = {
            'name': name,
            **metrics
        }
        break
    
    assert pipeline_data is not None
    
    # Generate detail page
    detail_path = visualizer._generate_pipeline_detail_page(
        pipeline_data, mock_monitor_data, temp_output_dir
    )
    
    # Check if jinja2 is available
    if hasattr(visualizer, 'has_jinja2') and visualizer.has_jinja2:
        assert detail_path is not None
        assert os.path.exists(detail_path)
        assert detail_path.endswith('.html')
    else:
        assert detail_path is None


def test_generate_stage_performance_chart(visualizer, mock_metrics, temp_output_dir):
    """Test generating a stage performance chart."""
    # Extract first pipeline
    pipeline_data = None
    for name, metrics in mock_metrics.get('pipelines', {}).items():
        pipeline_data = {
            'name': name,
            **metrics
        }
        break
    
    assert pipeline_data is not None
    
    # Generate chart
    chart_path = visualizer._generate_stage_performance_chart(
        pipeline_data, temp_output_dir
    )
    
    # Check if matplotlib is available
    if hasattr(visualizer, 'has_matplotlib') and visualizer.has_matplotlib:
        assert chart_path is not None
        assert os.path.exists(chart_path)
        assert chart_path.endswith('.svg')
    else:
        assert chart_path is None