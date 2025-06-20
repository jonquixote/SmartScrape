import pytest
import asyncio
import logging
from typing import Dict, Any, List

from core.ai_service import AIService
from core.batch_processor import BatchProcessor
from core.model_selector import ModelSelector
from core.content_processor import ContentProcessor
from core.ai_cache import AICache

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestAIServiceIntegration:
    """Integration tests for AI service with other components."""

    @pytest.fixture
    async def ai_service(self):
        """Create an initialized AI service."""
        service = AIService()
        config = {
            "models": [
                {
                    "name": "test-model",
                    "type": "mock", 
                    "api_key": "test-key",
                    "responses": {
                        "default": {"content": "Default response", "model": "test-model"}
                    }
                }
            ],
            "default_model": "test-model",
            "cache": {"backend": "memory", "default_ttl": 3600}
        }
        
        # Patch the model class creation to use a mock
        from unittest.mock import patch
        
        class MockModel:
            def __init__(self, config):
                self.config = config
                self.model_id = config.get("name", "mock")
                self.responses = config.get("responses", {"default": {"content": "Mock response"}})
                
            async def generate(self, prompt, options=None):
                return self.responses.get(prompt, self.responses["default"])
                
            def get_token_count(self, text):
                return len(text) // 4
                
            @property
            def capabilities(self):
                return {"streaming": False, "max_tokens": 1000}
                
            @property
            def cost_per_1k_tokens(self):
                return (0.001, 0.002)
        
        with patch.object(AIService, '_get_model_class', return_value=MockModel):
            service.initialize(config)
            
        yield service
        
        # Clean up
        service.shutdown()

    @pytest.fixture
    async def batch_processor(self):
        """Create a batch processor that uses the AI service."""
        async def process_batch(prompt_list, metadata):
            # Simple mock function that returns processed responses
            return [f"Processed: {prompt}" for prompt in prompt_list]
            
        processor = BatchProcessor(
            processor_fn=process_batch,
            batch_size=3,
            max_waiting_time=0.5
        )
        
        yield processor
        
        # Clean up
        await processor.shutdown()

    @pytest.mark.asyncio
    async def test_ai_service_with_content_processor(self, ai_service):
        """Test that AI service properly uses content processor."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <header>Site Header</header>
                <nav>Navigation</nav>
                <main>
                    <h1>Main Content</h1>
                    <p>This is the important content that should be preserved.</p>
                    <p>Secondary information that's also important.</p>
                </main>
                <footer>Footer content</footer>
            </body>
        </html>
        """
        
        # Test with content preprocessing
        response = await ai_service.generate_response(
            html_content,
            context={
                "content_type": "html",
                "preprocess": True,
                "extract_main": True
            }
        )
        
        # Verify the AI was called with preprocessed content
        assert "content" in response
        assert "_metadata" in response
        
        # Tokens should be less than the original content
        assert response["_metadata"]["input_tokens"] < len(html_content) // 4

    @pytest.mark.asyncio
    async def test_ai_service_with_caching(self, ai_service):
        """Test AI service caching functionality."""
        # First call should not be cached
        response1 = await ai_service.generate_response(
            "Test prompt for caching",
            context={"task_type": "test"}
        )
        
        # Get cache stats before second call
        stats_before = ai_service.cache.get_stats()
        
        # Second identical call should be cached
        response2 = await ai_service.generate_response(
            "Test prompt for caching",
            context={"task_type": "test"}
        )
        
        # Get cache stats after second call
        stats_after = ai_service.cache.get_stats()
        
        # Verify the second call was a cache hit
        assert stats_after["hits"] > stats_before["hits"]
        
        # Responses should be identical
        assert response1["content"] == response2["content"]

    @pytest.mark.asyncio
    async def test_ai_service_with_batch_processor(self, ai_service, batch_processor):
        """Test AI service working with batch processor."""
        from unittest.mock import patch
        
        # Replace the batch processor's function with one that uses the AI service
        async def process_with_ai(prompt_list, metadata):
            results = []
            for prompt in prompt_list:
                response = await ai_service.generate_response(
                    prompt, 
                    context=metadata,
                    model_name=metadata.get("model")
                )
                results.append(response["content"])
            return results
            
        # Patch the processor function
        original_fn = batch_processor.processor_fn
        batch_processor.processor_fn = process_with_ai
        
        try:
            # Submit several requests to be batched
            futures = []
            for i in range(5):
                _, future = await batch_processor.add_request(
                    f"Test prompt {i}",
                    metadata={"model": "test-model", "task_type": "test"}
                )
                futures.append(future)
                
            # Wait for all requests to complete
            results = await asyncio.gather(*futures)
            
            # Verify all results were processed
            assert len(results) == 5
            assert all("Default response" in result for result in results)
            
            # Check batch processor stats
            stats = batch_processor.get_stats()
            assert stats["successful_requests"] == 5
            assert stats["failed_requests"] == 0
            assert stats["total_batches"] >= 1  # Should be at least 1 batch
            
        finally:
            # Restore original processor function
            batch_processor.processor_fn = original_fn

    @pytest.mark.asyncio
    async def test_error_handling_in_batch(self, ai_service, batch_processor):
        """Test error handling when using AI service with batch processor."""
        from unittest.mock import patch
        
        # Replace the batch processor's function with one that simulates errors
        async def process_with_errors(prompt_list, metadata):
            if "error" in prompt_list[0]:
                raise ValueError("Simulated error in batch processing")
            
            results = []
            for prompt in prompt_list:
                response = await ai_service.generate_response(prompt, context=metadata)
                results.append(response["content"])
            return results
            
        # Patch the processor function
        batch_processor.processor_fn = process_with_errors
        
        # Create a request that will cause an error
        _, error_future = await batch_processor.add_request("error_prompt")
        
        # Create a request that should succeed
        _, success_future = await batch_processor.add_request("normal_prompt")
        
        # Wait for the error request to complete (should fail)
        with pytest.raises(ValueError):
            await error_future
            
        # Wait for the success request to complete
        result = await success_future
        assert "Default response" in result
        
        # Check stats
        stats = batch_processor.get_stats()
        assert stats["failed_requests"] >= 1
        assert stats["successful_requests"] >= 1

    @pytest.mark.asyncio
    async def test_rule_engine_integration(self, ai_service):
        """Test integration between AI service and rule engine."""
        from core.rule_engine import RegexRule
        
        # Configure a simple test rule
        test_rule = RegexRule(
            name="test_extraction_rule",
            pattern=r"extract email: ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            template="Email address: {0}",
            group=1,
            priority=10
        )
        
        # Add the rule to the rule engine
        ai_service.rule_engine.add_rule(test_rule)
        
        # Test prompt that should trigger the rule
        test_prompt = "Please extract email: test.user@example.com from this text"
        
        # Generate response with rule engine enabled
        response = await ai_service.generate_response(
            test_prompt,
            context={"use_rule_engine": True}
        )
        
        # Verify rule was used instead of AI model
        assert "source" in response
        assert response["source"] == "rule_engine"
        assert "Email address: test.user@example.com" in response["content"]
        
        # Verify no tokens were used (cost optimization)
        assert response["_metadata"]["input_tokens"] == 0
        assert response["_metadata"]["output_tokens"] == 0
        assert response["_metadata"]["total_cost"] == 0
        
        # Test with rule engine disabled
        response_no_rules = await ai_service.generate_response(
            test_prompt,
            context={"use_rule_engine": False}
        )
        
        # Verify AI model was used instead of rule engine
        assert "source" not in response_no_rules or response_no_rules.get("source") != "rule_engine"
        assert response_no_rules["_metadata"]["input_tokens"] > 0
        
    @pytest.mark.asyncio
    async def test_ai_service_generate_with_rules_and_fallback(self, ai_service):
        """Test the advanced generation with rules and fallback functionality."""
        # Create a fallback generator
        async def custom_fallback(prompt, context):
            return {
                "content": f"Fallback response for: {prompt}",
                "_metadata": {"source": "fallback"}
            }
        
        # Test prompt
        test_prompt = "This should go to AI, then fallback"
        
        # Temporarily modify AI service to fail when generating a response
        original_generate = ai_service.generate_response
        
        async def mock_generate_fail(*args, **kwargs):
            return {"error": "Simulated AI failure", "content": "", "_metadata": {"model": "test-model"}}
        
        try:
            # Replace with failing implementation temporarily
            ai_service.generate_response = mock_generate_fail
            
            # Test fallback generation
            result = await ai_service.generate_with_rules_and_fallback(
                test_prompt, 
                context={},
                fallback_generator=custom_fallback
            )
            
            # Verify fallback was used
            assert "Fallback response for:" in result["content"]
            assert result["_metadata"]["source"] == "custom_fallback"
            
        finally:
            # Restore original method
            ai_service.generate_response = original_generate
            
        # Test the complete chain with rule engine
        from core.rule_engine import RegexRule
        
        # Configure a simple test rule
        test_rule = RegexRule(
            name="special_command_rule",
            pattern=r"special_command: ([a-z_]+)",
            template="Executed special command: {0}",
            group=1,
            priority=10
        )
        
        # Add the rule to the rule engine
        ai_service.rule_engine.add_rule(test_rule)
        
        # Test prompt that should trigger the rule
        rule_test_prompt = "special_command: run_test"
        
        # Generate with full chain
        result = await ai_service.generate_with_rules_and_fallback(
            rule_test_prompt, 
            context={},
            fallback_generator=custom_fallback
        )
        
        # Verify rule was used (first in the chain)
        assert result["source"] == "rule_engine"
        assert "Executed special command: run_test" in result["content"]