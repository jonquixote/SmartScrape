from abc import ABC, abstractmethod
import logging
import asyncio
import time
import os
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from core.service_interface import BaseService
from core.content_processor import ContentProcessor
from core.ai_cache import AICache
from core.model_selector import ModelSelector
from core.batch_processor import BatchProcessor
from core.rule_engine import RuleEngine

class AIModel(ABC):
    """Base interface for AI model implementations."""
    
    @abstractmethod
    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            prompt: The text prompt to send to the model
            options: Optional configuration parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dictionary containing response content and metadata
        """
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for the given text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated number of tokens
        """
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """
        Return model capabilities.
        
        Returns:
            Dictionary of capability flags and parameters (streaming, function_calling, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """
        Return cost per 1k tokens as (input_cost, output_cost).
        
        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in USD
        """
        pass

class AIService(BaseService):
    """Centralized service for all AI interactions with built-in optimization."""
    
    PREDEFINED_MODEL_CONFIGURATIONS = [
        {
            "provider": "google", 
            "model_id": "gemini-2.0-flash-lite", 
            "name": "Gemini 2.0 Flash Lite",
            "capabilities": {"vision": False, "tool_use": True, "long_context": True},
            "notes": "Fast, cost-effective for many tasks."
        },
        {
            "provider": "google", 
            "model_id": "gemini-pro", 
            "name": "Gemini Pro",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True},
            "notes": "Balanced model for text and vision."
        },
        {
            "provider": "google",
            "model_id": "gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True, "video": True},
            "notes": "Latest generation, fast, multi-modal, very long context."
        },
        {
            "provider": "google",
            "model_id": "gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True, "video": True},
            "notes": "Latest generation, most capable, multi-modal, very long context."
        },
        {
            "provider": "openai", 
            "model_id": "gpt-4-turbo", 
            "name": "GPT-4 Turbo",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True},
            "notes": "Advanced model with vision and large context."
        },
        {
            "provider": "openai", 
            "model_id": "gpt-4o", 
            "name": "GPT-4o",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True, "audio": True},
            "notes": "Latest OpenAI model, fast, multimodal."
        },
        {
            "provider": "openai", 
            "model_id": "gpt-3.5-turbo", 
            "name": "GPT-3.5 Turbo",
            "capabilities": {"vision": False, "tool_use": True, "long_context": False}, # 16k context is decent
            "notes": "Fast and cost-effective for general tasks."
        },
        {
            "provider": "anthropic", 
            "model_id": "claude-3-opus-20240229", 
            "name": "Claude 3 Opus",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True},
            "notes": "Most powerful Anthropic model."
        },
        {
            "provider": "anthropic", 
            "model_id": "claude-3-sonnet-20240229", 
            "name": "Claude 3 Sonnet",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True},
            "notes": "Balanced performance and speed."
        },
        {
            "provider": "anthropic", 
            "model_id": "claude-3-haiku-20240307", 
            "name": "Claude 3 Haiku",
            "capabilities": {"vision": True, "tool_use": False, "long_context": True}, # Tool use might be beta
            "notes": "Fastest and most compact model for near-instant responsiveness."
        },
        # Mock model for development/testing
        {
            "provider": "mock",
            "model_id": "mock-model",
            "name": "Mock Model (Dev/Test)",
            "capabilities": {"vision": True, "tool_use": True, "long_context": True},
            "notes": "For development and testing purposes only."
        }
    ]
    
    def __init__(self):
        self._initialized = False
        self._config = None
        self.logger = logging.getLogger("ai_service")
        self.models = {}
        self.default_model_name = None
        
        # Optimization components
        self.content_processor = None
        self.cache = None
        self.model_selector = None
        self.batch_processor = None
        self.rule_engine = None
        
        # Statistics for monitoring and optimization
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "batched_requests": 0,
            "rule_engine_usages": 0,
            "average_latency": 0.0,
            "request_timestamps": []
        }
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the AI service.
        
        Args:
            config: Configuration dictionary with model definitions and settings
        """
        if self._initialized:
            return
        
        self._config = config or {}
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Initialize models
        self.models = self._initialize_models()
        self.default_model_name = self._config.get("default_model", "default")
        
        self._initialized = True
        self.logger.info("AI service initialized with optimization components")
    
    def shutdown(self) -> None:
        """Shutdown the AI service and all components."""
        if not self._initialized:
            return
        
        # Shut down batch processor
        if self.batch_processor:
            asyncio.create_task(self.batch_processor.shutdown())
        
        # Clean up resources
        self.models.clear()
        self._initialized = False
        self.logger.info("AI service shut down")
    
    def refresh_models_from_environment(self) -> None:
        """
        Refresh and reload models based on currently available environment variables.
        This allows the AI service to pick up newly available API keys without restarting.
        """
        if not self._initialized:
            return
        
        self.logger.info("Refreshing AI models based on current environment variables")
        
        # Store the current config for reference
        original_models_count = len(self.models)
        
        # Check which providers now have API keys available
        available_providers = []
        
        # Check for OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            available_providers.append(("openai", "OPENAI_API_KEY"))
        
        # Check for Anthropic
        if os.environ.get("ANTHROPIC_API_KEY"):
            available_providers.append(("anthropic", "ANTHROPIC_API_KEY"))
        
        # Check for Google/Gemini
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            available_providers.append(("google", "GOOGLE_API_KEY"))
        
        # Create new model configurations for available providers
        new_model_configs = []
        
        for provider, env_key in available_providers:
            if provider == "openai":
                # Add GPT models
                new_model_configs.extend([
                    {
                        "name": "gpt-4-turbo",
                        "type": "openai", 
                        "model_id": "gpt-4-turbo",
                        "api_key": os.environ.get("OPENAI_API_KEY")
                    },
                    {
                        "name": "gpt-3.5-turbo",
                        "type": "openai",
                        "model_id": "gpt-3.5-turbo", 
                        "api_key": os.environ.get("OPENAI_API_KEY")
                    }
                ])
            elif provider == "anthropic":
                # Add Claude models
                new_model_configs.append({
                    "name": "claude-3-sonnet",
                    "type": "anthropic",
                    "model_id": "claude-3-sonnet-20240229",
                    "api_key": os.environ.get("ANTHROPIC_API_KEY")
                })
            elif provider == "google":
                # Add Gemini models
                api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                new_model_configs.extend([
                    {
                        "name": "gemini-2.0-flash-lite",
                        "type": "google",
                        "model_id": "gemini-2.0-flash-lite",
                        "api_key": api_key
                    },
                    {
                        "name": "gemini-pro",
                        "type": "google", 
                        "model_id": "gemini-pro",
                        "api_key": api_key
                    }
                ])
        
        # Initialize new models and add them to existing models dict
        new_models_added = 0
        for model_config in new_model_configs:
            model_name = model_config.get("name")
            model_type = model_config.get("type")
            
            # Skip if model already exists
            if model_name in self.models:
                continue
                
            try:
                model_class = self._get_model_class(model_type)
                self.models[model_name] = model_class(model_config)
                new_models_added += 1
                self.logger.info(f"Added new model: {model_name} ({model_type})")
            except Exception as e:
                self.logger.error(f"Failed to initialize new model {model_name}: {str(e)}")
        
        # Update default model if we added real models and current default is mock
        if new_models_added > 0 and self.default_model_name == "mock":
            # Try to find a good default model
            preferred_models = ["gemini-2.0-flash-lite", "gpt-4-turbo", "claude-3-sonnet", "gpt-3.5-turbo", "gemini-pro"]
            for preferred in preferred_models:
                if preferred in self.models:
                    self.default_model_name = preferred
                    self.logger.info(f"Updated default model to: {preferred}")
                    break
        
        # Update model selector with new models if it exists
        if hasattr(self, 'model_selector') and self.model_selector:
            try:
                # Reconstruct model configs for the selector
                updated_model_configs = []
                for model_name, model_instance in self.models.items():
                    # Create model config entries for the model selector
                    model_config = {
                        "name": model_name,
                        "type": getattr(model_instance, '_config', {}).get('type', 'unknown')
                    }
                    
                    # Add capabilities from the model instance
                    if hasattr(model_instance, 'capabilities'):
                        model_config["capabilities"] = model_instance.capabilities
                    
                    # Add other attributes if available
                    if hasattr(model_instance, '_config') and model_instance._config:
                        original_config = model_instance._config
                        model_config.update({
                            "quality_score": original_config.get("quality_score", 5),
                            "speed_score": original_config.get("speed_score", 5),
                            "cost_score": original_config.get("cost_score", 5),
                            "context_length": original_config.get("context_length", 4096),
                            "cost_per_1k_tokens": original_config.get("cost_per_1k_tokens", (0.01, 0.01)),
                            "task_specializations": original_config.get("task_specializations", ["general"]),
                        })
                    
                    updated_model_configs.append(model_config)
                
                # Update the model selector's config and reinitialize
                if updated_model_configs:
                    self.model_selector.config = updated_model_configs
                    self.model_selector.models_info = self.model_selector._initialize_models_info()
                    self.logger.info(f"Updated model selector with {len(updated_model_configs)} models")
                    self.logger.info(f"Model selector now has models: {list(self.model_selector.models_info.keys())}")
            except Exception as e:
                self.logger.warning(f"Failed to update model selector: {str(e)}")
                import traceback
                self.logger.warning(f"Traceback: {traceback.format_exc()}")
        
        self.logger.info(f"Model refresh complete. Added {new_models_added} new models. Total models: {len(self.models)}")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "ai_service"
    
    def _initialize_optimization_components(self):
        """Initialize all optimization components."""
        # Initialize content processor
        content_processor_config = self._config.get("content_processor", {})
        self.content_processor = ContentProcessor(content_processor_config)
        self.logger.info("Content processor initialized")
        
        # Initialize cache
        cache_config = self._config.get("cache", {})
        self.cache = AICache(cache_config)
        self.logger.info(f"AI cache initialized with backend: {self.cache.backend_type}")
        
        # Initialize model selector
        model_selector_config = self._config.get("models", [])
        self.model_selector = ModelSelector(model_selector_config)
        self.logger.info("Model selector initialized")
        
        # Initialize rule engine for fallbacks
        rule_engine_config = self._config.get("rule_engine", {})
        self.rule_engine = RuleEngine(rule_engine_config)
        self.logger.info("Rule engine initialized")
        
        # Initialize batch processor
        batch_config = self._config.get("batch_processor", {})
        batch_size = batch_config.get("batch_size", 10)
        max_waiting_time = batch_config.get("max_waiting_time", 5.0)
        max_concurrent_batches = batch_config.get("max_concurrent_batches", 5)
        
        # Set up batch processor with our processing function
        self.batch_processor = BatchProcessor(
            processor_fn=self._process_batch,
            batch_size=batch_size,
            max_waiting_time=max_waiting_time,
            max_concurrent_batches=max_concurrent_batches,
            config=batch_config
        )
        self.logger.info("Batch processor initialized")
        
    def _initialize_models(self) -> Dict[str, AIModel]:
        """
        Initialize and register all configured models.
        
        Returns:
            Dictionary mapping model names to AIModel instances
        """
        self.logger.info(f"Initializing models with config: {self._config}")
        models = {}
        for model_config in self._config.get("models", []):
            model_type = model_config.get("type")
            model_name = model_config.get("name")
            self.logger.info(f"Processing model config: {model_config}")
            if model_type and model_name:
                try:
                    model_class = self._get_model_class(model_type)
                    models[model_name] = model_class(model_config)
                    self.logger.info(f"Initialized model: {model_name} ({model_type})")
                except Exception as e:
                    self.logger.error(f"Failed to initialize model {model_name}: {str(e)}")
        self.logger.info(f"Total models initialized: {len(models)}, names: {list(models.keys())}")
        return models
    
    def _get_model_class(self, model_type: str) -> type:
        """
        Get the appropriate model class for the specified type.
        
        Args:
            model_type: String identifier for the model type (openai, anthropic, google, etc.)
            
        Returns:
            The model class
            
        Raises:
            ValueError: If the model type is not supported
        """
        # Import here to avoid circular imports
        from core.ai_models import OpenAIModel, AnthropicModel, GoogleModel, MockModel
        
        model_classes = {
            "openai": OpenAIModel,
            "anthropic": AnthropicModel,
            "google": GoogleModel,
            "mock": MockModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model_classes[model_type]
    
    async def _process_batch(self, 
                            prompts: List[str], 
                            common_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts with the same model and parameters.
        
        Args:
            prompts: List of prompts to process
            common_metadata: Common metadata shared by all requests in the batch
            
        Returns:
            List of response dictionaries
        """
        model_name = common_metadata.get("model", self.default_model_name)
        options = common_metadata.get("options", {})
        
        # Debug logging
        self.logger.info(f"Batch process: Requested model: {model_name}")
        self.logger.info(f"Batch process: Available models: {list(self.models.keys())}")
        self.logger.info(f"Batch process: Default model: {self.default_model_name}")
        
        if model_name not in self.models:
            # Try to find an available model that works
            available_models = list(self.models.keys())
            if available_models:
                fallback_model = available_models[0]
                self.logger.warning(f"Model {model_name} not found, using fallback: {fallback_model}")
                model_name = fallback_model
            elif self.default_model_name and self.default_model_name in self.models:
                self.logger.info(f"Batch process: Falling back to default model: {self.default_model_name}")
                model_name = self.default_model_name
            else:
                # Try to use a mock model for development
                self.logger.warning(f"No models available, creating mock response for batch processing")
                # Return mock responses for all prompts
                mock_responses = []
                for prompt in prompts:
                    mock_responses.append({
                        "content": f"Mock response for: {prompt[:100]}...",
                        "model": "mock-fallback",
                        "_metadata": {"warning": "Using mock response due to model unavailability"}
                    })
                return mock_responses
        
        model = self.models[model_name]
        
        # Process all prompts in parallel
        tasks = []
        for prompt in prompts:
            tasks.append(model.generate(prompt, options))
            
        # Wait for all to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                # Handle errors
                processed_responses.append({
                    "error": str(response), 
                    "content": "",
                    "_metadata": {"model": model_name}
                })
            else:
                # Calculate token usage and costs
                prompt_tokens = model.get_token_count(prompts[i])
                output_tokens = model.get_token_count(response.get("content", ""))
                
                # Calculate cost
                input_cost, output_cost = model.cost_per_1k_tokens
                total_cost = (prompt_tokens * input_cost + output_tokens * output_cost) / 1000
                
                # Add metadata
                response["_metadata"] = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_cost": total_cost,
                    "model": model_name,
                    "is_batched": True
                }
                
                processed_responses.append(response)
                
        return processed_responses
        
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for AI interactions with all optimizations applied.
        
        Args:
            prompt: The text prompt to send to the model
            context: Optional context with parameters affecting generation
            model_name: Optional model name to use, defaults to automatic selection
            
        Returns:
            Dictionary containing the response content and metadata
            
        Raises:
            RuntimeError: If the service is not initialized
            ValueError: If the specified model is not available
        """
        if not self._initialized:
            raise RuntimeError("AI service not initialized")
        
        start_time = time.time()
        self.stats["total_requests"] += 1
        context = context or {}
        
        try:
            # 1. Check for rule-based alternative first
            rule_result = await self._try_rule_based_alternative(prompt, context)
            if rule_result:
                self.stats["rule_engine_usages"] += 1
                return rule_result
            
            # 2. Preprocess content to reduce tokens
            preprocessed_prompt = self._preprocess_content(prompt, context)
            
            # 3. Check cache
            cache_key = None
            if context.get("use_cache", True):
                cache_key = self.cache.generate_key(
                    preprocessed_prompt, context, model_name or self.default_model_name
                )
                cached_response = self.cache.get(cache_key)
                if cached_response:
                    self.stats["cache_hits"] += 1
                    
                    # Add cache metadata
                    if "_metadata" not in cached_response:
                        cached_response["_metadata"] = {}
                    cached_response["_metadata"]["from_cache"] = True
                    
                    # Update stats
                    metadata = cached_response.get("_metadata", {})
                    self.stats["total_tokens"] += metadata.get("input_tokens", 0) + metadata.get("output_tokens", 0)
                    self.stats["total_cost"] += metadata.get("total_cost", 0)
                    
                    # Update latency stats
                    elapsed = time.time() - start_time
                    self._update_latency_stats(elapsed)
                    
                    return cached_response
            
            # 4. Determine best model if not specified
            if not model_name:
                # Analyze task and select model
                task_analysis = self.model_selector.analyze_task_complexity(prompt)
                token_estimate = task_analysis["estimated_tokens"]
                
                model_name = self.model_selector.select_model(
                    task_type=task_analysis["task_type"],
                    content_length=token_estimate,
                    require_capabilities=task_analysis["required_capabilities"],
                    quality_priority=context.get("quality_priority", 5),
                    speed_priority=context.get("speed_priority", 5),
                    cost_priority=context.get("cost_priority", 5)
                )
                
                # Add analysis to context
                context["task_analysis"] = task_analysis
            
            # 5. Check if batching is enabled and add to batch if so
            if context.get("use_batching", True) and self.batch_processor:
                self.stats["batched_requests"] += 1
                
                # Add to batch and wait for result
                request_id, future = await self.batch_processor.add_request(
                    data=preprocessed_prompt,
                    priority=context.get("priority", 0),
                    metadata={"model": model_name, "options": context.get("options", {})}
                )
                
                try:
                    # Wait for the batch processing to complete
                    response = await future
                    
                    # Cache the result if caching is enabled
                    if cache_key and context.get("use_cache", True):
                        self.cache.set(cache_key, response)
                    
                    # Update stats
                    self.stats["successful_requests"] += 1
                    metadata = response.get("_metadata", {})
                    self.stats["total_tokens"] += metadata.get("input_tokens", 0) + metadata.get("output_tokens", 0)
                    self.stats["total_cost"] += metadata.get("total_cost", 0)
                    
                    # Update latency stats
                    elapsed = time.time() - start_time
                    self._update_latency_stats(elapsed)
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {str(e)}")
                    self.stats["failed_requests"] += 1
                    
                    # Fall back to direct processing
                    # (continue to the direct model execution below)
            
            # 6. Direct model execution (if batching is disabled or failed)
            # Get the model
            if model_name not in self.models:
                if self.default_model_name in self.models:
                    model_name = self.default_model_name
                else:
                    raise ValueError(f"Model {model_name} not available and no default model found")
                
            model = self.models[model_name]
            
            # Log token usage for cost tracking
            input_tokens = model.get_token_count(preprocessed_prompt)
            self.logger.info(f"Input tokens: {input_tokens} for model {model_name}")
            
            # Generate response
            response = await model.generate(preprocessed_prompt, context.get("options", {}))
            
            # Track output tokens
            if "content" in response:
                output_tokens = model.get_token_count(response["content"])
                self.logger.info(f"Output tokens: {output_tokens} for model {model_name}")
                
                # Calculate and log cost
                input_cost, output_cost = model.cost_per_1k_tokens
                total_cost = (input_tokens * input_cost + output_tokens * output_cost) / 1000
                self.logger.info(f"Estimated cost: ${total_cost:.6f} for model {model_name}")
                
                # Add token and cost info to response metadata
                response["_metadata"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_cost": total_cost,
                    "model": model_name,
                    "preprocessed": True if prompt != preprocessed_prompt else False
                }
                
                # Cache the result if caching is enabled
                if cache_key and context.get("use_cache", True):
                    self.cache.set(cache_key, response)
                
                # Update stats
                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += input_tokens + output_tokens
                self.stats["total_cost"] += total_cost
            
            # Update latency stats
            elapsed = time.time() - start_time
            self._update_latency_stats(elapsed)
            
            return response
                
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            self.stats["failed_requests"] += 1
            
            # Update latency stats even for failures
            elapsed = time.time() - start_time
            self._update_latency_stats(elapsed)
            
            return {"error": str(e), "content": "", "_metadata": {"model": model_name or "unknown"}}
    
    def get_available_model_configurations(self) -> List[Dict[str, Any]]:
        """
        Returns a list of predefined model configurations that the system supports.
        This list indicates models that *could* be used if an API key is provided.
        """
        return self.PREDEFINED_MODEL_CONFIGURATIONS
    
    async def test_api_key(self, provider: str, api_key: str, model_id: Optional[str] = None) -> bool:
        """
        Tests an API key for a given provider by attempting a small interaction.

        Args:
            provider: The AI provider (e.g., "google", "openai", "anthropic").
            api_key: The API key to test.
            model_id: Optional specific model ID to test with. If None, a default/cheap model for the provider will be used.

        Returns:
            True if the API key is valid and a minimal interaction succeeds, False otherwise.
        """
        self.logger.info(f"Testing API key for provider: {provider}")
        if not api_key:
            self.logger.warning("API key is empty. Test failed.")
            return False

        test_model_id = model_id
        model_config_override = {"api_key": api_key}

        if not test_model_id:
            # Select a default, typically low-cost model for testing for each provider
            if provider == "google":
                test_model_id = "gemini-2.0-flash-lite" 
            elif provider == "openai":
                test_model_id = "gpt-3.5-turbo" 
            elif provider == "anthropic":
                test_model_id = "claude-3-haiku-20240307"
            else:
                self.logger.error(f"Unsupported provider for API key test: {provider}")
                return False
        
        model_config_override["model_id"] = test_model_id
        model_config_override["name"] = f"test-{provider}-{test_model_id}"


        try:
            model_class = self._get_model_class(provider)
            # Temporarily instantiate the model with the provided API key
            # The model_config_override provides the necessary 'api_key' and 'model_id'
            temp_model_instance = model_class(config=model_config_override)
            
            # Perform a very small, inexpensive operation
            # For generative models, a very short prompt is usually sufficient.
            # Some provider SDKs might have a more direct way to test credentials (e.g., list models)
            # but a short generation is a universal test.
            test_prompt = "Hello."
            if provider == "anthropic": # Anthropic requires messages format
                response = await temp_model_instance.generate(prompt=[{"role": "user", "content": test_prompt}], options={"max_tokens": 5})
            else:
                response = await temp_model_instance.generate(prompt=test_prompt, options={"max_tokens": 5})

            if response and ("content" in response or isinstance(response.get("content"), list)): # Anthropic might return list of content blocks
                self.logger.info(f"API key for {provider} is valid.")
                return True
            elif response and "error" in response:
                self.logger.warning(f"API key test for {provider} failed with error: {response['error']}")
                return False
            else:
                self.logger.warning(f"API key test for {provider} failed. Unexpected response: {response}")
                return False

        except ValueError as ve: # Handles unsupported model type from _get_model_class
            self.logger.error(f"API key test failed for {provider}: {str(ve)}")
            return False
        except ImportError as ie: # Handles missing SDKs
             self.logger.error(f"API key test failed for {provider} due to missing SDK: {str(ie)}. Please ensure the provider's SDK is installed.")
             return False
        except Exception as e:
            # Catching a broad exception here because different SDKs might raise different auth errors.
            # Specific error types (e.g., AuthenticationError from OpenAI) could be caught if known.
            self.logger.error(f"API key test for {provider} failed with exception: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _preprocess_content(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Preprocess content to reduce token usage.
        
        Args:
            prompt: The original prompt
            context: Context with preprocessing options
            
        Returns:
            Preprocessed prompt
        """
        if not self.content_processor or not context.get("preprocess_content", True):
            return prompt
            
        try:
            # Check if the prompt contains HTML that needs processing
            if "<html" in prompt.lower() or "<!doctype" in prompt.lower():
                # Extract main content from HTML
                preprocessed = self.content_processor.preprocess_html(
                    prompt, 
                    extract_main=context.get("extract_main_content", True),
                    max_tokens=context.get("max_tokens")
                )
                
                return preprocessed
                
            elif context.get("summarize_content", False) and len(prompt) > 2000:
                # Summarize long text content
                return self.content_processor.summarize_content(
                    prompt,
                    ratio=context.get("summary_ratio", 0.3),
                    max_length=context.get("max_length", 1000)
                )
                
            # Chunk content if it's potentially too large
            elif context.get("chunk_content", False) and len(prompt) > 5000:
                chunks = self.content_processor.chunk_content(
                    prompt,
                    max_tokens=context.get("max_tokens_per_chunk", 4000),
                    overlap=context.get("chunk_overlap", 200)
                )
                
                # For now, just use the first chunk
                # In a real implementation, you might process all chunks and combine results
                if chunks:
                    return chunks[0]
            
            # Default: return unchanged
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error preprocessing content: {str(e)}")
            return prompt
    
    async def _try_rule_based_alternative(self, prompt: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Try to use rule-based alternative instead of AI.
        
        Args:
            prompt: The original prompt
            context: Context with rule engine options
            
        Returns:
            Response dict if rule-based alternative succeeded, None otherwise
        """
        if not self.rule_engine or not context.get("use_rule_engine", True):
            return None
            
        try:
            # Try to apply rules
            rule_result = await self.rule_engine.process(prompt, context)
            
            if rule_result and rule_result.get("success", False):
                # Rule was applied successfully
                return {
                    "content": rule_result.get("content", ""),
                    "source": "rule_engine",
                    "_metadata": {
                        "rule_id": rule_result.get("rule_id"),
                        "input_tokens": 0,  # No AI tokens used
                        "output_tokens": 0,  # No AI tokens used
                        "total_cost": 0,     # No AI cost incurred
                        "processing_time": rule_result.get("processing_time", 0)
                    }
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error applying rule-based alternative: {str(e)}")
            return None
    
    def _update_latency_stats(self, elapsed: float) -> None:
        """Update latency statistics with a new data point."""
        # Keep last 100 timestamps for calculating recent average
        self.stats["request_timestamps"].append(time.time())
        if len(self.stats["request_timestamps"]) > 100:
            self.stats["request_timestamps"].pop(0)
            
        # Update the rolling average latency
        if self.stats["average_latency"] == 0:
            self.stats["average_latency"] = elapsed
        else:
            # Weighted average (90% old, 10% new)
            self.stats["average_latency"] = 0.9 * self.stats["average_latency"] + 0.1 * elapsed
    
    def get_ai_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive AI usage statistics.
        
        Returns:
            Dictionary with AI usage statistics
        """
        stats = dict(self.stats)
        
        # Add cache stats
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats["cache"] = cache_stats
        
        # Add batch processor stats
        if self.batch_processor:
            batch_stats = self.batch_processor.get_stats()
            stats["batch_processor"] = batch_stats
        
        # Calculate requests per minute
        if len(self.stats["request_timestamps"]) >= 2:
            oldest = self.stats["request_timestamps"][0]
            newest = self.stats["request_timestamps"][-1]
            time_span = newest - oldest
            if time_span > 0:
                requests_in_window = len(self.stats["request_timestamps"])
                stats["requests_per_minute"] = (requests_in_window / time_span) * 60
            
        # Calculate savings
        cache_hits = stats.get("cache_hits", 0)
        rule_engine_usages = stats.get("rule_engine_usages", 0)
        
        if self.cache:
            cache_token_savings = self.cache.stats.get("token_savings", 0)
            cache_cost_savings = self.cache.stats.get("cost_savings", 0.0)
            stats["estimated_token_savings"] = cache_token_savings
            stats["estimated_cost_savings"] = cache_cost_savings
                
        return stats
        
    async def generate_with_rules_and_fallback(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        fallback_generator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Advanced generation with rule-based processing and custom fallback.
        
        Args:
            prompt: The text prompt to process
            context: Optional context with parameters
            fallback_generator: Optional custom fallback function to use if AI fails
            
        Returns:
            Response dictionary
        """
        context = context or {}
        
        try:
            # First try rule-based approach
            rule_result = await self._try_rule_based_alternative(prompt, context)
            if rule_result:
                return rule_result
                
            # Then try AI approach
            ai_result = await self.generate_response(prompt, context)
            
            # Check if AI succeeded
            if "error" not in ai_result or not ai_result["error"]:
                return ai_result
                
            # If we're here, AI failed and we need to use fallback
            if fallback_generator:
                try:
                    fallback_result = await fallback_generator(prompt, context)
                    fallback_result["_metadata"] = fallback_result.get("_metadata", {})
                    fallback_result["_metadata"]["source"] = "custom_fallback"
                    return fallback_result
                except Exception as e:
                    self.logger.error(f"Fallback generator failed: {str(e)}")
                    
            # If we're here, everything failed
            return ai_result  # Return the failed AI result
            
        except Exception as e:
            self.logger.error(f"Error in generation chain: {str(e)}")
            return {"error": str(e), "content": "", "_metadata": {"source": "error"}}
    
    async def optimize_prompt(self, prompt: str, task_type: str = "general") -> str:
        """
        Optimize a prompt for better results.
        
        Args:
            prompt: The original prompt
            task_type: Type of task for contextual optimization
            
        Returns:
            Optimized prompt
        """
        # If we don't have an AI model available, return the original
        if not self.models or not self.default_model_name in self.models:
            return prompt
            
        try:
            # Create a meta-prompt to optimize the original prompt
            meta_prompt = f"""
            You are a prompt optimization assistant. Your task is to improve the following prompt 
            for a {task_type} task. Make the prompt clearer, more specific, and more likely 
            to generate high-quality results. Do not add any explanations, just return the 
            improved prompt.
            
            ORIGINAL PROMPT:
            {prompt}
            
            OPTIMIZED PROMPT:
            """
            
            # Generate an optimized prompt
            result = await self.generate_response(
                meta_prompt,
                context={
                    "use_cache": True,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more deterministic results
                        "max_tokens": 1000
                    }
                }
            )
            
            if "content" in result and result["content"].strip():
                return result["content"].strip()
            else:
                return prompt
                
        except Exception as e:
            self.logger.error(f"Error optimizing prompt: {str(e)}")
            return prompt
    
    async def generate_collated_responses(
        self,
        prompts: List[str],
        context: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        use_batching: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple related prompts efficiently using collated processing.
        This method optimizes for scenarios where multiple prompts need to be processed
        together, such as extracting different data points from the same content.
        
        Args:
            prompts: List of prompts to process
            context: Optional context shared by all prompts
            model_name: Optional model name to use for all prompts
            use_batching: Whether to use batch processing (recommended for performance)
            
        Returns:
            List of response dictionaries in the same order as input prompts
            
        Raises:
            RuntimeError: If the service is not initialized
        """
        if not self._initialized:
            raise RuntimeError("AI service not initialized")
            
        if not prompts:
            return []
            
        context = context or {}
        start_time = time.time()
        
        try:
            # If batching is enabled and we have multiple prompts, use batch processor
            if use_batching and len(prompts) > 1 and self.batch_processor:
                self.logger.info(f"Processing {len(prompts)} prompts using batch processing")
                
                # Determine the best model if not specified
                if not model_name:
                    # Analyze the first prompt to determine complexity
                    sample_prompt = prompts[0] if prompts else ""
                    task_analysis = self.model_selector.analyze_task_complexity(sample_prompt)
                    
                    model_name = self.model_selector.select_model(
                        task_type=task_analysis["task_type"],
                        content_length=task_analysis["estimated_tokens"],
                        require_capabilities=task_analysis["required_capabilities"],
                        quality_priority=context.get("quality_priority", 5),
                        speed_priority=context.get("speed_priority", 7),  # Prioritize speed for batch
                        cost_priority=context.get("cost_priority", 8)    # Prioritize cost for batch
                    )
                
                # Preprocess all prompts
                preprocessed_prompts = []
                for prompt in prompts:
                    preprocessed_prompt = self._preprocess_content(prompt, context)
                    preprocessed_prompts.append(preprocessed_prompt)
                
                # Check cache for all prompts
                cached_responses = []
                cache_keys = []
                uncached_indices = []
                uncached_prompts = []
                
                if context.get("use_cache", True):
                    for i, preprocessed_prompt in enumerate(preprocessed_prompts):
                        cache_key = self.cache.generate_key(
                            preprocessed_prompt, context, model_name
                        )
                        cache_keys.append(cache_key)
                        
                        cached_response = self.cache.get(cache_key)
                        if cached_response:
                            cached_responses.append((i, cached_response))
                            self.stats["cache_hits"] += 1
                        else:
                            uncached_indices.append(i)
                            uncached_prompts.append(preprocessed_prompt)
                
                # Process uncached prompts in batch
                batch_responses = []
                if uncached_prompts:
                    self.stats["batched_requests"] += len(uncached_prompts)
                    
                    # Submit all uncached prompts to batch processor
                    batch_futures = []
                    for prompt in uncached_prompts:
                        request_id, future = await self.batch_processor.add_request(
                            data=prompt,
                            priority=context.get("priority", 1),  # Higher priority for collated requests
                            metadata={"model": model_name, "options": context.get("options", {})}
                        )
                        batch_futures.append(future)
                    
                    # Wait for all batch processing to complete
                    try:
                        batch_responses = await asyncio.gather(*batch_futures, return_exceptions=True)
                        
                        # Cache the new responses
                        if context.get("use_cache", True):
                            for i, response in enumerate(batch_responses):
                                if not isinstance(response, Exception) and uncached_indices:
                                    original_index = uncached_indices[i]
                                    cache_key = cache_keys[original_index]
                                    self.cache.set(cache_key, response)
                                    
                    except Exception as e:
                        self.logger.error(f"Error in batch processing collated prompts: {str(e)}")
                        # Fall back to individual processing
                        return await self._fallback_individual_processing(prompts, context, model_name)
                
                # Combine cached and batch responses in original order
                final_responses = [None] * len(prompts)
                
                # Place cached responses
                for original_index, cached_response in cached_responses:
                    final_responses[original_index] = cached_response
                
                # Place batch responses
                for i, response in enumerate(batch_responses):
                    if uncached_indices and i < len(uncached_indices):
                        original_index = uncached_indices[i]
                        final_responses[original_index] = response
                
                # Update statistics
                elapsed = time.time() - start_time
                self.stats["total_requests"] += len(prompts)
                self.stats["successful_requests"] += len([r for r in final_responses if r and not isinstance(r, Exception)])
                self._update_latency_stats(elapsed)
                
                return final_responses
                
            else:
                # Fall back to individual processing for small batches or when batching is disabled
                return await self._fallback_individual_processing(prompts, context, model_name)
                
        except Exception as e:
            self.logger.error(f"Error in collated response generation: {str(e)}")
            elapsed = time.time() - start_time
            self.stats["failed_requests"] += len(prompts)
            self._update_latency_stats(elapsed)
            # Ensure to return a list of error dicts matching the number of prompts
            return [{"error": str(e), "content": "", "_metadata": {"source": "error"}} for _ in prompts]

    async def _fallback_individual_processing(
        self,
        prompts: List[str],
        context: Dict[str, Any],
        model_name: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Fall back to processing prompts individually when batch processing is not available or fails.
        
        Args:
            prompts: List of prompts to process
            context: Shared context for all prompts
            model_name: Model to use for processing
            
        Returns:
            List of response dictionaries
        """
        self.logger.info(f"Processing {len(prompts)} prompts individually")
        
        # Process prompts individually with some concurrency
        tasks = []
        for prompt in prompts:
            # Disable batching for individual calls to avoid recursion
            individual_context = dict(context)
            individual_context["use_batching"] = False
            
            task = self.generate_response(prompt, individual_context, model_name)
            tasks.append(task)
        
        # Process with limited concurrency to avoid overwhelming the API
        max_concurrent = context.get("max_concurrent_individual", 3)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_generate(prompt_task):
            async with semaphore:
                return await prompt_task
        
        # Execute with controlled concurrency
        limited_tasks = [limited_generate(task) for task in tasks]
        responses = await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        final_responses = []
        for response in responses:
            if isinstance(response, Exception):
                final_responses.append({
                    "error": str(response),
                    "content": "",
                    "_metadata": {"model": model_name or "unknown"}
                })
            else:
                final_responses.append(response)
        
        return final_responses