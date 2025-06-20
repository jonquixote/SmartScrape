import pytest
from core.model_selector import ModelSelector

class TestModelSelector:
    """Test suite for the ModelSelector class."""
    
    @pytest.fixture
    def model_selector(self):
        """Create a model selector with test configuration."""
        config = [
            {
                "name": "test-basic",
                "type": "test",
                "capabilities": {
                    "streaming": True,
                    "function_calling": False
                },
                "quality_score": 5,
                "speed_score": 8,
                "cost_score": 8,
                "context_length": 4000,
                "cost_per_1k_tokens": (0.001, 0.002),
                "task_specializations": ["classification", "qa"]
            },
            {
                "name": "test-advanced",
                "type": "test",
                "capabilities": {
                    "streaming": True,
                    "function_calling": True,
                    "json_mode": True
                },
                "quality_score": 8,
                "speed_score": 5,
                "cost_score": 3,
                "context_length": 8000,
                "cost_per_1k_tokens": (0.01, 0.02),
                "task_specializations": ["analysis", "code"]
            },
            {
                "name": "test-large-context",
                "type": "test",
                "capabilities": {
                    "streaming": True,
                    "function_calling": False
                },
                "quality_score": 7,
                "speed_score": 6,
                "cost_score": 5,
                "context_length": 100000,
                "cost_per_1k_tokens": (0.005, 0.01),
                "task_specializations": ["summarization", "extraction"]
            }
        ]
        selector = ModelSelector(config)
        selector.default_model = "test-basic"
        return selector
    
    def test_initialize_models_info(self, model_selector):
        """Test that models are correctly initialized from config."""
        assert len(model_selector.models_info) == 3
        assert "test-basic" in model_selector.models_info
        assert "test-advanced" in model_selector.models_info
        assert "test-large-context" in model_selector.models_info
        
        # Check specific model properties
        basic_model = model_selector.models_info["test-basic"]
        assert basic_model["type"] == "test"
        assert basic_model["context_length"] == 4000
        assert basic_model["cost_per_1k_tokens"] == (0.001, 0.002)
        assert "classification" in basic_model["task_specializations"]
    
    def test_select_model_by_capability(self, model_selector):
        """Test selecting a model based on required capabilities."""
        # Test selection requiring function calling
        model = model_selector.select_model(
            task_type="general",
            require_capabilities=["function_calling"]
        )
        assert model == "test-advanced"  # Only test-advanced has function_calling
        
        # Test selection with multiple required capabilities
        model = model_selector.select_model(
            task_type="general",
            require_capabilities=["function_calling", "json_mode"]
        )
        assert model == "test-advanced"  # Only test-advanced has both
        
        # Test with non-existent capability - should return default
        model = model_selector.select_model(
            task_type="general",
            require_capabilities=["non_existent_capability"]
        )
        assert model == "test-basic"  # Default model when no match found
    
    def test_select_model_by_content_length(self, model_selector):
        """Test selecting a model based on content length."""
        # Test with content that fits all models
        model = model_selector.select_model(
            task_type="general",
            content_length=3000
        )
        # With balanced priorities, should be test-basic due to speed and cost scores
        assert model == "test-basic"
        
        # Test with content that exceeds basic model
        model = model_selector.select_model(
            task_type="general",
            content_length=5000
        )
        # Only test-advanced and test-large-context can handle this
        assert model in ["test-advanced", "test-large-context"]
        
        # Test with very large content
        model = model_selector.select_model(
            task_type="general",
            content_length=50000
        )
        # Only test-large-context can handle this size
        assert model == "test-large-context"
    
    def test_select_model_by_task_type(self, model_selector):
        """Test selecting a model based on task type."""
        # Test with a task specialized for basic model
        model = model_selector.select_model(
            task_type="classification",
            quality_priority=5,
            speed_priority=5,
            cost_priority=5
        )
        assert model == "test-basic"  # Specialized for classification
        
        # Test with a task specialized for advanced model
        model = model_selector.select_model(
            task_type="code",
            quality_priority=5,
            speed_priority=5,
            cost_priority=5
        )
        assert model == "test-advanced"  # Specialized for code
        
        # Test with a task specialized for large context model
        model = model_selector.select_model(
            task_type="summarization",
            quality_priority=5,
            speed_priority=5,
            cost_priority=5
        )
        assert model == "test-large-context"  # Specialized for summarization
    
    def test_select_model_by_priorities(self, model_selector):
        """Test selecting a model based on different priorities."""
        # Test with high quality priority
        model = model_selector.select_model(
            task_type="general",
            quality_priority=10,
            speed_priority=1,
            cost_priority=1
        )
        assert model == "test-advanced"  # Highest quality score
        
        # Test with high speed priority
        model = model_selector.select_model(
            task_type="general",
            quality_priority=1,
            speed_priority=10,
            cost_priority=1
        )
        assert model == "test-basic"  # Highest speed score
        
        # Test with high cost priority (prefers cheapest)
        model = model_selector.select_model(
            task_type="general",
            quality_priority=1,
            speed_priority=1,
            cost_priority=10
        )
        assert model == "test-basic"  # Lowest cost (highest cost_score)
    
    def test_analyze_task_complexity(self, model_selector):
        """Test analyzing task complexity from description."""
        # Test high complexity task
        high_complexity = model_selector.analyze_task_complexity(
            "Perform a comprehensive analysis of the text, identifying nuanced themes and subtle patterns."
        )
        assert high_complexity["complexity"] == "high"
        assert high_complexity["task_type"] == "analysis"
        assert high_complexity["estimated_tokens"] >= 1000
        
        # Test low complexity task
        low_complexity = model_selector.analyze_task_complexity(
            "Give me a quick summary of this simple article."
        )
        assert low_complexity["complexity"] == "low"
        assert low_complexity["task_type"] == "summarization"
        assert low_complexity["estimated_tokens"] <= 500
        
        # Test task with required capabilities
        json_task = model_selector.analyze_task_complexity(
            "Extract the data and return it as structured JSON output."
        )
        assert "json_mode" in json_task["required_capabilities"]
        
        # Test code task
        code_task = model_selector.analyze_task_complexity(
            "Write a Python function to calculate Fibonacci numbers."
        )
        assert code_task["task_type"] == "code"
        
        # Test task with multiple indicators
        mixed_task = model_selector.analyze_task_complexity(
            "Analyze this complex code and explain how it works in a detailed way."
        )
        assert mixed_task["complexity"] == "high"
        assert mixed_task["task_type"] in ["analysis", "code"]
        assert mixed_task["estimated_tokens"] > 1000
    
    def test_estimate_cost(self, model_selector):
        """Test cost estimation."""
        # Test basic model cost
        basic_cost = model_selector.estimate_cost("test-basic", 1000, 500)
        # (1000 * 0.001 + 500 * 0.002) / 1000 = 0.002
        assert basic_cost == 0.002
        
        # Test advanced model cost
        advanced_cost = model_selector.estimate_cost("test-advanced", 1000, 500)
        # (1000 * 0.01 + 500 * 0.02) / 1000 = 0.02
        assert advanced_cost == 0.02
        
        # Test with auto-estimated output tokens
        large_context_cost = model_selector.estimate_cost("test-large-context", 1200)
        # (1200 * 0.005 + 400 * 0.01) / 1000 = 0.01
        assert large_context_cost == 0.01
        
        # Test unknown model falls back to default
        unknown_cost = model_selector.estimate_cost("unknown-model", 1000, 500)
        # Should use default cost
        assert unknown_cost > 0
    
    def test_suggest_models(self, model_selector):
        """Test model suggestion feature."""
        suggestions = model_selector.suggest_models(
            "Analyze this complex code and explain how it works."
        )
        
        # Should have at least one suggestion
        assert len(suggestions) >= 1
        
        # First suggestion should be marked as primary
        assert suggestions[0]["is_primary_suggestion"] == True
        
        # For a code analysis task, test-advanced should be suggested
        assert any(s["name"] == "test-advanced" for s in suggestions)
        
        # Each suggestion should have rationale and advantages
        for suggestion in suggestions:
            assert "rationale" in suggestion
            assert "advantages" in suggestion
            assert len(suggestion["advantages"]) > 0
            assert "estimated_cost" in suggestion
        
        # Test with a different task type
        summary_suggestions = model_selector.suggest_models(
            "Give me a quick summary of this 30-page document."
        )
        
        # Should suggest the large context model for summarization
        assert any(s["name"] == "test-large-context" for s in summary_suggestions)
        
        # Test suggestions for a very long content
        long_content_suggestions = model_selector.suggest_models(
            "Summarize this document.",
            content_length=60000
        )
        
        # Only the large context model can handle this
        assert len(long_content_suggestions) >= 1
        assert long_content_suggestions[0]["name"] == "test-large-context"
        
        # Alternative suggestions should not be primary
        if len(long_content_suggestions) > 1:
            for suggestion in long_content_suggestions[1:]:
                assert suggestion["is_primary_suggestion"] == False