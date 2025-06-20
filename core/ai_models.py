import logging
import asyncio
from typing import Dict, Any, Tuple, Optional, List
import re

from core.ai_service import AIModel

class OpenAIModel(AIModel):
    """OpenAI model implementation for GPT models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI model.

        Args:
            config: Configuration with api_key, model_id, and other settings
        """
        self.config = config
        self.api_key = config.get("api_key")
        # Changed default to a placeholder, ensure Gemini is prioritized elsewhere
        self.model_id = config.get("model_id", "placeholder-openai-model")
        self.logger = logging.getLogger("ai_models.openai")
        self._setup()

    def _setup(self):
        """
        Set up the OpenAI client.

        Raises:
            ImportError: If the openai package is not installed
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            self.logger.error("OpenAI package not installed. Try: pip install openai")
            raise

    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response from the OpenAI model.

        Args:
            prompt: The text prompt to send to the model
            options: Optional parameters like temperature, max_tokens, etc.

        Returns:
            Dictionary containing the response content and metadata

        Raises:
            Exception: If the API call fails
        """
        options = options or {}
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("max_tokens", 1000)

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return {
                "content": response.choices[0].message.content,
                "model": self.model_id,
                "finish_reason": response.choices[0].finish_reason,
                "id": response.id
            }

        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for the given text using OpenAI's tokenizer.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated number of tokens
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.model_id)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to rough estimate: ~4 chars per token
            self.logger.warning("Using approximate token count (tiktoken not available)")
            return len(text) // 4

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return model capabilities.

        Returns:
            Dictionary of capability flags and parameters
        """
        capabilities = {
            "streaming": True,
            # Adjusted to reflect that these are OpenAI specific capabilities
            "function_calling": "gpt-4" in self.model_id or "placeholder-openai-model" in self.model_id,
            "json_mode": "gpt-4" in self.model_id or "placeholder-openai-model" in self.model_id,
            "vision": "vision" in self.model_id,
        }

        # Set context window size based on model
        if "gpt-4-32k" in self.model_id:
            capabilities["max_tokens"] = 32768
        elif "gpt-4" in self.model_id:
            capabilities["max_tokens"] = 8192
        elif "gpt-3.5-turbo-16k" in self.model_id: # Ensure this is not used if focusing on Gemini
            capabilities["max_tokens"] = 16384
        elif "placeholder-openai-model" in self.model_id: # Default for placeholder
            capabilities["max_tokens"] = 4096
        else:
            capabilities["max_tokens"] = 4096 # Fallback

        return capabilities

    @property
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """
        Return cost per 1k tokens as (input_cost, output_cost).

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in USD
        """
        # Costs as of 2023-2024
        costs = {
            # Ensure these are not primary if using Gemini
            "placeholder-openai-model": (0.0015, 0.002),
            "gpt-3.5-turbo-16k": (0.003, 0.004),
            "gpt-4": (0.03, 0.06),
            "gpt-4-32k": (0.06, 0.12),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-4-vision": (0.01, 0.03),
        }
        # Find the matching model or use a default
        for model_key, cost_tuple in costs.items():
            if model_key in self.model_id:
                return cost_tuple

        # Default cost if unknown
        self.logger.warning(f"No cost information for model {self.model_id}, using default")
        return (0.01, 0.02)


class AnthropicModel(AIModel):
    """Anthropic model implementation for Claude models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Anthropic model.

        Args:
            config: Configuration with api_key, model_id, and other settings
        """
        self.config = config
        self.api_key = config.get("api_key")
        self.model_id = config.get("model_id", "claude-2")
        self.logger = logging.getLogger("ai_models.anthropic")
        self._setup()

    def _setup(self):
        """
        Set up the Anthropic client.

        Raises:
            ImportError: If the anthropic package is not installed
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            self.logger.error("Anthropic package not installed. Try: pip install anthropic")
            raise

    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response from the Anthropic model.

        Args:
            prompt: The text prompt to send to the model
            options: Optional parameters like temperature, max_tokens, etc.

        Returns:
            Dictionary containing the response content and metadata

        Raises:
            Exception: If the API call fails
        """
        options = options or {}
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("max_tokens", 1000)

        try:
            response = self.client.completions.create(
                model=self.model_id,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                max_tokens_to_sample=max_tokens,
                temperature=temperature
            )

            return {
                "content": response.completion,
                "model": self.model_id,
                "id": response.id
            }

        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {str(e)}")
            raise

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for the given text.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated number of tokens
        """
        try:
            import anthropic
            return anthropic.count_tokens(text)
        except Exception:
            # Fallback to rough estimate: ~4 chars per token
            self.logger.warning("Using approximate token count (anthropic tokenizer not available)")
            return len(text) // 4

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return model capabilities.

        Returns:
            Dictionary of capability flags and parameters
        """
        capabilities = {
            "streaming": True,
            "function_calling": False,
            "json_mode": False,
            "vision": False
        }

        # Set context window size based on model
        if "claude-3" in self.model_id:
            if "opus" in self.model_id:
                capabilities["max_tokens"] = 200000
            elif "sonnet" in self.model_id:
                capabilities["max_tokens"] = 180000
            else:  # haiku
                capabilities["max_tokens"] = 150000
        elif "claude-2" in self.model_id:
            capabilities["max_tokens"] = 100000
        else:
            capabilities["max_tokens"] = 50000

        return capabilities

    @property
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """
        Return cost per 1k tokens as (input_cost, output_cost).

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in USD
        """
        # Costs as of 2023-2024
        costs = {
            "claude-3-opus": (0.015, 0.075),
            "claude-3-sonnet": (0.003, 0.015),
            "claude-3-haiku": (0.00025, 0.00125),
            "claude-2": (0.008, 0.024),
            "claude-instant": (0.0016, 0.0056),
        }

        # Find the matching model or use a default
        for model_key, cost_tuple in costs.items():
            if model_key in self.model_id:
                return cost_tuple

        # Default cost if unknown
        self.logger.warning(f"No cost information for model {self.model_id}, using default")
        return (0.01, 0.03)


class GoogleModel(AIModel):
    """Google model implementation for Gemini models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Google model.

        Args:
            config: Configuration with api_key, model_id, and other settings
        """
        self.config = config
        self.api_key = config.get("api_key")
        self.model_id = config.get("model_id", "gemini-2.0-flash-lite") # Use latest 2.0 model
        self.logger = logging.getLogger("ai_models.google")
        self._setup()

    def _setup(self):
        """
        Set up the Google Gemini client.

        Raises:
            ImportError: If the google.generativeai package is not installed
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(self.model_id)
        except ImportError:
            self.logger.error("Google generativeai package not installed. Try: pip install google-generativeai")
            raise

    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a response from the Google model.

        Args:
            prompt: The text prompt to send to the model
            options: Optional parameters like temperature, etc.

        Returns:
            Dictionary containing the response content and metadata

        Raises:
            Exception: If the API call fails
        """
        options = options or {}
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("max_tokens", 1000)

        try:
            # Configure generation parameters
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": options.get("top_p", 0.8),
                "top_k": options.get("top_k", 40)
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Add debugging information
            self.logger.debug(f"Response type: {type(response)}")
            self.logger.debug(f"Response dir: {dir(response)}")
            if hasattr(response, 'candidates'):
                self.logger.debug(f"Candidates: {len(response.candidates) if response.candidates else 0}")
                if response.candidates:
                    self.logger.debug(f"First candidate type: {type(response.candidates[0])}")
                    self.logger.debug(f"First candidate dir: {dir(response.candidates[0])}")

            # Handle the response more carefully for newer Gemini models
            content = ""
            candidates = []
            
            # Try to get the text directly from response first (newer versions)
            try:
                if hasattr(response, 'text') and response.text:
                    content = response.text
                else:
                    # For newer models, extract content from candidates
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            # Try to access content.parts structure (newer format)
                            try:
                                if hasattr(candidate, 'content') and candidate.content:
                                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                        for part in candidate.content.parts:
                                            if hasattr(part, 'text'):
                                                content += part.text
                                                candidates.append(part.text)
                                    elif hasattr(candidate.content, 'text'):
                                        # Direct text on content
                                        content += candidate.content.text
                                        candidates.append(candidate.content.text)
                            except Exception as part_error:
                                self.logger.debug(f"Error accessing candidate content: {part_error}")
                                continue
                
                # If we still don't have content, try alternative approaches
                if not content:
                    # Check if response has parts directly
                    if hasattr(response, 'parts') and response.parts:
                        for part in response.parts:
                            if hasattr(part, 'text'):
                                content += part.text
                
                # Last resort: try to get any available text
                if not content and hasattr(response, '_result'):
                    # Some versions store the actual result in _result
                    result = getattr(response, '_result', None)
                    if result and hasattr(result, 'candidates'):
                        for candidate in result.candidates:
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text'):
                                        content += part.text
                                        
            except Exception as extraction_error:
                self.logger.error(f"Error extracting content from response: {extraction_error}")
                # Fallback to string parsing
                content_str = str(response)
                if "text:" in content_str:
                    import re
                    text_match = re.search(r'text:\s*["\']([^"\']*)["\']', content_str)
                    if text_match:
                        content = text_match.group(1)

            return {
                "content": content,
                "model": self.model_id,
                "candidates": candidates,
                "finish_reason": getattr(response, 'finish_reason', None) if hasattr(response, 'finish_reason') else None
            }

        except Exception as e:
            self.logger.error(f"Error calling Google Generative AI API: {str(e)}")
            # Log more details for debugging
            self.logger.error(f"Model ID: {self.model_id}")
            self.logger.error(f"Prompt length: {len(prompt)}")
            self.logger.error(f"Options: {options}")
            raise

    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for the given text.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated number of tokens
        """
        try:
            # Google doesn't provide a direct token counter, so approximate
            return len(text) // 4  # Rough estimate: ~4 chars per token
        except Exception:
            return len(text) // 4

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return model capabilities.

        Returns:
            Dictionary of capability flags and parameters
        """
        capabilities = {
            "streaming": True,
            "function_calling": "pro" in self.model_id,
            "json_mode": True,  # Google models can handle JSON output reliably
            "vision": "vision" in self.model_id
        }

        # Set context window size based on model
        if "pro" in self.model_id:
            capabilities["max_tokens"] = 32768
        else:
            capabilities["max_tokens"] = 8192

        return capabilities

    @property
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """
        Return cost per 1k tokens as (input_cost, output_cost).

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in USD
        """
        # Costs as of 2024-2025 (Updated for newer Gemini models)
        costs = {
            "gemini-pro": (0.000125, 0.000375), # Gemini Pro (legacy)
            "gemini-1.5-flash": (0.000125, 0.000375), # Gemini 1.5 Flash
            "gemini-1.5-flash-latest": (0.000125, 0.000375), # Gemini 1.5 Flash Latest
            "gemini-1.5-pro": (0.0025, 0.0075), # Gemini 1.5 Pro
            "gemini-1.5-pro-latest": (0.0025, 0.0075), # Gemini 1.5 Pro Latest
            "gemini-2.0-flash": (0.000125, 0.000375), # Gemini 2.0 Flash
            "gemini-2.0-flash-lite": (0.000075, 0.0003), # Gemini 2.0 Flash Lite (more cost effective)
            "gemini-1.0-pro": (0.000125, 0.000375), # Gemini 1.0 Pro (legacy)
            "gemini-ultra": (0.0100, 0.0300), # Gemini Ultra
        }

        # Find the matching model or use a default
        for model_key, cost_tuple in costs.items():
            if model_key in self.model_id:
                return cost_tuple

        # Default cost if unknown, using Gemini 2.0 Flash Lite pricing
        self.logger.warning(f"No cost information for model {self.model_id}, using default Gemini 2.0 Flash Lite cost")
        return (0.000075, 0.0003)


class MockModel(AIModel):
    """Mock model implementation for testing."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Mock model.

        Args:
            config: Optional configuration with predefined responses
        """
        self.config = config or {}
        self.responses = self.config.get("responses", {
            "default": {"content": "This is a mock response", "model": "mock"}
        })
        self.logger = logging.getLogger("ai_models.mock")

    async def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a mock response.

        Args:
            prompt: The text prompt
            options: Optional parameters (ignored in mock)

        Returns:
            Dictionary containing the mock response
        """
        # Look for specific prompt patterns in the predefined responses
        for pattern, response in self.responses.items():
            if pattern != "default" and re.search(pattern, prompt, re.IGNORECASE):
                return response

        # Return default response if no pattern matches
        return self.responses["default"]

    def get_token_count(self, text: str) -> int:
        """
        Mock token counting function.

        Args:
            text: The text to count tokens for

        Returns:
            Simple approximation of token count
        """
        return len(text) // 4

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return mock capabilities.

        Returns:
            Dictionary of capability flags
        """
        return {
            "streaming": False,
            "function_calling": False,
            "json_mode": False,
            "vision": False,
            "max_tokens": 4096
        }

    @property
    def cost_per_1k_tokens(self) -> Tuple[float, float]:
        """
        Return mock cost (free).

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in USD
        """
        return (0.0, 0.0)