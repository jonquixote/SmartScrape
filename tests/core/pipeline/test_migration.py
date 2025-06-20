import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List, Optional

from core.pipeline.adapters import (
    StrategyToPipelineAdapter,
    LegacyExtractorAdapter,
    PipelineToLegacyAdapter
)
from core.pipeline.compatibility import (
    FeatureFlags,
    PerformanceMonitor,
    ABTestSelector,
    FallbackExecutor,
    ResultComparator
)
from core.pipeline.pipeline import Pipeline
from core.pipeline.context import PipelineContext
from core.pipeline.stage import PipelineStage


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    strategy = MagicMock()
    strategy.name = "mock_strategy"
    strategy.execute = AsyncMock(return_value={"success": True, "data": {"key": "value"}})
    strategy.get_results = MagicMock(return_value=[{"result1": "value1"}, {"result2": "value2"}])
    return strategy


@pytest.fixture
def mock_extractor():
    """Create a mock legacy extractor for testing."""
    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value={"success": True, "extracted_data": {"title": "Test"}})
    return extractor


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = MagicMock(spec=Pipeline)
    
    async def execute_side_effect(data=None):
        context = PipelineContext(data or {})
        context.set("result", {"title": "Test", "content": "Content"})
        return context
        
    pipeline.execute = AsyncMock(side_effect=execute_side_effect)
    return pipeline


class TestStrategyToPipelineAdapter:
    """Test the StrategyToPipelineAdapter class."""
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, mock_strategy):
        """Test adapter initialization with a strategy."""
        adapter = StrategyToPipelineAdapter(mock_strategy)
        assert adapter.strategy == mock_strategy
        assert adapter.name == "mock_strategy_adapter"
        
    @pytest.mark.asyncio
    async def test_process_method(self, mock_strategy):
        """Test the process method calls strategy execute."""
        adapter = StrategyToPipelineAdapter(mock_strategy)
        context = PipelineContext({"url": "https://example.com"})
        
        result = await adapter.process(context)
        
        assert result is True
        mock_strategy.execute.assert_called_once_with("https://example.com")
        assert context.get("strategy_result") == {"success": True, "data": {"key": "value"}}
        
    @pytest.mark.asyncio
    async def test_process_with_results(self, mock_strategy):
        """Test that strategy results are added to context."""
        adapter = StrategyToPipelineAdapter(mock_strategy, include_all_results=True)
        context = PipelineContext({"url": "https://example.com"})
        
        await adapter.process(context)
        
        assert context.get("strategy_results") == [{"result1": "value1"}, {"result2": "value2"}]


class TestLegacyExtractorAdapter:
    """Test the LegacyExtractorAdapter class."""
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, mock_extractor):
        """Test adapter initialization with an extractor."""
        adapter = LegacyExtractorAdapter(mock_extractor, "content_extractor")
        assert adapter.extractor == mock_extractor
        assert adapter.name == "content_extractor_adapter"
        
    @pytest.mark.asyncio
    async def test_process_method(self, mock_extractor):
        """Test the process method calls extractor extract."""
        adapter = LegacyExtractorAdapter(mock_extractor, "content_extractor")
        context = PipelineContext({
            "html_content": "<html>Test</html>",
            "url": "https://example.com"
        })
        
        result = await adapter.process(context)
        
        assert result is True
        mock_extractor.extract.assert_called_once()
        assert context.get("extraction_result") == {
            "success": True, 
            "extracted_data": {"title": "Test"}
        }
        
    @pytest.mark.asyncio
    async def test_process_with_extraction_parameters(self, mock_extractor):
        """Test that extraction parameters are correctly passed."""
        adapter = LegacyExtractorAdapter(
            mock_extractor, 
            "content_extractor",
            param_mapping={
                "html": "html_content",
                "extraction_params": "params"
            }
        )
        context = PipelineContext({
            "html_content": "<html>Test</html>",
            "url": "https://example.com",
            "params": {"depth": 3}
        })
        
        await adapter.process(context)
        
        # Check that the mapped parameters were used
        mock_extractor.extract.assert_called_once()
        call_kwargs = mock_extractor.extract.call_args[1]
        assert call_kwargs["html"] == "<html>Test</html>"
        assert call_kwargs["extraction_params"] == {"depth": 3}


class TestPipelineToLegacyAdapter:
    """Test the PipelineToLegacyAdapter class."""
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, mock_pipeline):
        """Test adapter initialization with a pipeline."""
        adapter = PipelineToLegacyAdapter(mock_pipeline)
        assert adapter.pipeline == mock_pipeline
        
    @pytest.mark.asyncio
    async def test_extract_method(self, mock_pipeline):
        """Test the extract method calls pipeline execute."""
        adapter = PipelineToLegacyAdapter(mock_pipeline)
        result = await adapter.extract(
            html_content="<html>Test</html>",
            url="https://example.com"
        )
        
        mock_pipeline.execute.assert_called_once()
        assert result == {"title": "Test", "content": "Content"}
        
    @pytest.mark.asyncio
    async def test_custom_result_mapping(self, mock_pipeline):
        """Test custom result mapping from context to result."""
        adapter = PipelineToLegacyAdapter(
            mock_pipeline, 
            result_key="result",
            result_mapping={
                "page_title": "title",
                "page_content": "content"
            }
        )
        
        result = await adapter.extract(
            html_content="<html>Test</html>",
            url="https://example.com"
        )
        
        assert result == {
            "page_title": "Test",
            "page_content": "Content"
        }


class TestCompatibilityLayer:
    """Test the pipeline compatibility layer components."""
    
    def test_feature_flags_initialization(self):
        """Test FeatureFlags initialization and checking."""
        # Initialize with custom config
        custom_config = {
            "use_pipeline_architecture": False,
            "pipeline_components": {
                "extraction": True,
                "validation": False
            }
        }
        
        FeatureFlags.initialize(custom_config)
        
        # Check flags
        assert FeatureFlags.is_enabled("use_pipeline_architecture") is False
        assert FeatureFlags.is_enabled("pipeline_components.extraction") is True
        assert FeatureFlags.is_enabled("pipeline_components.validation") is False
        
    def test_should_use_pipeline(self):
        """Test pipeline usage determination."""
        # Reset to default config
        FeatureFlags.initialize({
            "use_pipeline_architecture": True,
            "pipeline_components": {
                "extraction": True,
                "validation": False
            }
        })
        
        assert FeatureFlags.should_use_pipeline() is True
        assert FeatureFlags.should_use_pipeline("extraction") is True
        assert FeatureFlags.should_use_pipeline("validation") is False
        
    @pytest.mark.asyncio
    async def test_fallback_executor(self):
        """Test the FallbackExecutor with successful pipeline."""
        # Setup mocks
        pipeline_func = AsyncMock(return_value={"status": "success"})
        legacy_func = AsyncMock(return_value={"status": "legacy"})
        
        # Reset to default config with pipeline enabled
        FeatureFlags.initialize({
            "use_pipeline_architecture": True,
            "fallback_to_legacy": True
        })
        
        # Execute with fallback
        executor = FallbackExecutor(
            pipeline_func=pipeline_func,
            legacy_func=legacy_func,
            component_name="test_component"
        )
        
        result, info = await executor.execute()
        
        # Should use pipeline and succeed
        assert result == {"status": "success"}
        assert info["implementation"] == "pipeline"
        assert info["fallback_occurred"] is False
        assert pipeline_func.called is True
        assert legacy_func.called is False
        
    @pytest.mark.asyncio
    async def test_fallback_executor_with_failure(self):
        """Test the FallbackExecutor with failing pipeline."""
        # Setup mocks
        pipeline_func = AsyncMock(side_effect=Exception("Pipeline error"))
        legacy_func = AsyncMock(return_value={"status": "legacy"})
        
        # Reset to default config with pipeline enabled and fallback
        FeatureFlags.initialize({
            "use_pipeline_architecture": True,
            "fallback_to_legacy": True
        })
        
        # Execute with fallback
        executor = FallbackExecutor(
            pipeline_func=pipeline_func,
            legacy_func=legacy_func,
            component_name="test_component"
        )
        
        result, info = await executor.execute()
        
        # Should fallback to legacy and succeed
        assert result == {"status": "legacy"}
        assert info["implementation"] == "legacy"
        assert info["fallback_occurred"] is True
        assert "pipeline_error" in info
        assert pipeline_func.called is True
        assert legacy_func.called is True
        
    def test_result_comparator(self):
        """Test the ResultComparator for equivalent results."""
        # Create comparator
        comparator = ResultComparator("test_component")
        
        # Test equivalent results
        pipeline_result = {
            "success": True,
            "data": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        }
        
        legacy_result = {
            "success": True,
            "data": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        }
        
        comparison = comparator.compare(pipeline_result, legacy_result)
        
        assert comparison["compared"] is True
        assert comparison["equivalent"] is True
        assert "differences" not in comparison
        
    def test_result_comparator_with_differences(self):
        """Test the ResultComparator for different results."""
        # Create comparator
        comparator = ResultComparator("test_component")
        
        # Test different results
        pipeline_result = {
            "success": True,
            "data": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        }
        
        legacy_result = {
            "success": True,
            "data": [{"id": 1, "name": "Different"}, {"id": 3, "name": "Item 3"}]
        }
        
        # Enable discrepancy logging
        FeatureFlags.initialize({"log_discrepancies": True})
        
        comparison = comparator.compare(pipeline_result, legacy_result)
        
        assert comparison["compared"] is True
        assert comparison["equivalent"] is False
        assert "differences" in comparison
        
    @pytest.mark.asyncio
    async def test_performance_monitor(self):
        """Test the PerformanceMonitor for comparing implementations."""
        # Enable performance metrics
        FeatureFlags.initialize({"collect_performance_metrics": True})
        
        # Create mock functions with different execution times
        async def fast_func():
            return {"result": "fast"}
            
        async def slow_func():
            await asyncio.sleep(0.01)  # Simulate slower execution
            return {"result": "slow"}
            
        # Compare performance
        with PerformanceMonitor.compare("test_operation") as monitor:
            with monitor.track("legacy"):
                legacy_result = await slow_func()
                
            with monitor.track("pipeline"):
                pipeline_result = await fast_func()
                
        # Check metrics
        metrics = PerformanceMonitor.get_metrics()
        assert len(metrics["comparisons"]) > 0
        assert metrics["comparisons"][-1]["operation"] == "test_operation"
        
        summary = metrics["comparisons"][-1]["summary"]
        assert summary["pipeline_faster"] is True


class TestMigrationAccuracy:
    """Test the accuracy of migrated components."""
    
    @pytest.mark.asyncio
    async def test_strategy_adapter_maintains_functionality(self, mock_strategy):
        """Test that the strategy adapter maintains the same functionality."""
        # Direct strategy execution
        strategy_result = await mock_strategy.execute("https://example.com")
        
        # Pipeline adapter execution
        adapter = StrategyToPipelineAdapter(mock_strategy)
        context = PipelineContext({"url": "https://example.com"})
        await adapter.process(context)
        adapter_result = context.get("strategy_result")
        
        # Results should be the same
        assert strategy_result == adapter_result
        
    @pytest.mark.asyncio
    async def test_pipeline_adapter_maintains_functionality(self, mock_pipeline):
        """Test that the pipeline adapter maintains the same functionality."""
        # Direct pipeline execution
        pipeline_context = await mock_pipeline.execute({
            "html_content": "<html>Test</html>",
            "url": "https://example.com"
        })
        pipeline_result = pipeline_context.get("result")
        
        # Legacy adapter execution
        adapter = PipelineToLegacyAdapter(mock_pipeline, result_key="result")
        adapter_result = await adapter.extract(
            html_content="<html>Test</html>",
            url="https://example.com"
        )
        
        # Results should be the same
        assert pipeline_result == adapter_result


class TestPerformanceImpact:
    """Test the performance impact of adapters."""
    
    @pytest.mark.asyncio
    async def test_adapter_overhead(self, mock_strategy):
        """Test the overhead added by adapters."""
        # Direct strategy execution
        start_time = asyncio.get_event_loop().time()
        await mock_strategy.execute("https://example.com")
        direct_duration = asyncio.get_event_loop().time() - start_time
        
        # Pipeline adapter execution
        adapter = StrategyToPipelineAdapter(mock_strategy)
        context = PipelineContext({"url": "https://example.com"})
        
        start_time = asyncio.get_event_loop().time()
        await adapter.process(context)
        adapter_duration = asyncio.get_event_loop().time() - start_time
        
        # Adapter should add minimal overhead
        # Note: This is a relative test, small durations may vary
        # The important thing is that the adapter doesn't add significant overhead
        assert adapter_duration < direct_duration * 2  # Allowing for some overhead


class TestBackwardCompatibility:
    """Test backward compatibility of migrated components."""
    
    @pytest.mark.asyncio
    async def test_legacy_code_with_pipeline_adapter(self, mock_pipeline):
        """Test that legacy code works with pipeline adapter."""
        # Create pipeline adapter
        adapter = PipelineToLegacyAdapter(mock_pipeline, result_key="result")
        
        # Function that expects legacy extractor interface
        async def legacy_function(extractor, url):
            result = await extractor.extract(
                html_content="<html>Test</html>",
                url=url
            )
            return result
            
        # Call legacy function with adapter
        result = await legacy_function(adapter, "https://example.com")
        
        # Should work with legacy code
        assert result == {"title": "Test", "content": "Content"}