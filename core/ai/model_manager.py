import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AIModelManager:
    """
    Manages available AI models, their configurations, and selection.
    """
    def __init__(self, available_models: Dict[str, Any], default_model_name: str, config_manager: Optional[Any] = None):
        """
        Initialize the AIModelManager.

        Args:
            available_models: A dictionary of available AI models and their properties.
                              Example:
                              {
                                  "gemini-pro": {"score": 100, "available": True, "api_key_env": "GOOGLE_API_KEY"},
                                  "gpt-4": {"score": 90, "available": True, "api_key_env": "OPENAI_API_KEY"}
                              }
            default_model_name: The name of the default AI model to use.
            config_manager: An optional configuration manager to retrieve API keys.
        """
        self.available_models = available_models
        self.default_model_name = default_model_name
        self.config_manager = config_manager # Placeholder for a more robust config solution
        self.loaded_models = {}

        self._validate_models()
        logger.info(f"AIModelManager initialized with {len(self.available_models)} models. Default: {self.default_model_name}")

    def _validate_models(self):
        """Validate the structure of the available_models configuration."""
        if not isinstance(self.available_models, dict):
            raise ValueError("available_models must be a dictionary.")
        for name, config in self.available_models.items():
            if not isinstance(config, dict):
                raise ValueError(f"Configuration for model '{name}' must be a dictionary.")
            if not all(key in config for key in ["score", "available", "api_key_env"]):
                raise ValueError(f"Model '{name}' config is missing required keys (score, available, api_key_env).")
        if self.default_model_name not in self.available_models:
            logger.warning(f"Default model '{self.default_model_name}' not in available models. Falling back to first available.")
            # Fallback to the first available model or the first model if none are explicitly available
            fallback_model = next((name for name, conf in self.available_models.items() if conf.get("available")), None)
            if not fallback_model and self.available_models:
                fallback_model = list(self.available_models.keys())[0]

            if fallback_model:
                self.default_model_name = fallback_model
                logger.info(f"Using fallback default model: {self.default_model_name}")
            else:
                logger.error("No AI models available or configured.")
                # Or raise an error if no models are usable
                # raise ValueError("No AI models available or configured.")


    def get_api_key(self, model_name: str) -> Optional[str]:
        """
        Retrieves the API key for a given model.
        Tries environment variables first, then the config_manager if provided.
        """
        model_config = self.available_models.get(model_name)
        if not model_config:
            logger.warning(f"Model {model_name} not configured.")
            return None

        api_key_env_var = model_config.get("api_key_env")
        api_key = None

        if api_key_env_var:
            api_key = os.getenv(api_key_env_var)

        if not api_key and self.config_manager:
            # This is a placeholder. Actual implementation depends on how config_manager stores/retrieves keys.
            # For example, it might be self.config_manager.get_secret(api_key_env_var)
            # or self.config_manager.get_setting(f"AI_API_KEYS.{model_name}")
            try:
                # Attempt to get key using a hypothetical method from config_manager
                if hasattr(self.config_manager, 'get_api_key'):
                    api_key = self.config_manager.get_api_key(model_name)
                elif hasattr(self.config_manager, 'get_secret'): # Common pattern
                    api_key = self.config_manager.get_secret(api_key_env_var)
                # Add more sophisticated key retrieval logic as needed
            except Exception as e:
                logger.error(f"Error retrieving API key for {model_name} via config_manager: {e}")


        if not api_key:
            logger.warning(f"API key for model {model_name} (env var: {api_key_env_var}) not found.")
        return api_key

    def get_model(self, model_name: Optional[str] = None) -> Optional[Any]:
        """
        Retrieves an instance of the specified AI model.
        If no model_name is provided, uses the default model.
        Handles API key retrieval and model instantiation (currently placeholder).

        Args:
            model_name: The name of the model to retrieve.

        Returns:
            An instance of the AI model, or None if not available or an error occurs.
        """
        target_model_name = model_name or self.default_model_name

        if target_model_name not in self.available_models:
            logger.error(f"Model '{target_model_name}' is not configured in available_models.")
            return None

        model_config = self.available_models[target_model_name]

        if not model_config.get("available", False):
            logger.warning(f"Model '{target_model_name}' is configured but marked as unavailable.")
            return None

        # Check if model already loaded
        if target_model_name in self.loaded_models:
            return self.loaded_models[target_model_name]

        api_key = self.get_api_key(target_model_name)
        if not api_key:
            logger.error(f"Cannot load model '{target_model_name}' due to missing API key.")
            return None

        # Placeholder for actual model loading logic
        # This would involve importing the correct library (e.g., OpenAI, Google GenAI)
        # and initializing the client with the API key.
        try:
            logger.info(f"Loading model: {target_model_name}...")
            # Example:
            # if "gpt" in target_model_name.lower():
            #     from openai import OpenAI
            #     client = OpenAI(api_key=api_key)
            #     self.loaded_models[target_model_name] = client # Or a wrapper around the client
            #     return client
            # elif "gemini" in target_model_name.lower():
            #     import google.generativeai as genai
            #     genai.configure(api_key=api_key)
            #     model_instance = genai.GenerativeModel(target_model_name)
            #     self.loaded_models[target_model_name] = model_instance
            #     return model_instance
            # else:
            #     logger.error(f"Model type for '{target_model_name}' not recognized for loading.")
            #     return None
            
            # For now, return a mock model object
            mock_model = {"name": target_model_name, "api_key_used": bool(api_key), "config": model_config}
            self.loaded_models[target_model_name] = mock_model
            logger.info(f"Successfully loaded mock model: {target_model_name}")
            return mock_model

        except ImportError as e:
            logger.error(f"Failed to import library for model {target_model_name}: {e}")
            self.available_models[target_model_name]["available"] = False # Mark as unavailable
            return None
        except Exception as e:
            logger.error(f"Failed to load model {target_model_name}: {e}")
            self.available_models[target_model_name]["available"] = False # Mark as unavailable
            return None

    def select_model(self, requirements: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Selects the best available model based on requirements (e.g., score, capabilities).
        Currently, it selects the highest-scoring available model.

        Args:
            requirements: A dictionary of requirements (e.g., min_score, capabilities).

        Returns:
            An instance of the selected AI model, or None.
        """
        # Simple selection: highest score among available models
        best_model_name = None
        highest_score = -1

        for name, config in self.available_models.items():
            if config.get("available", False):
                if config["score"] > highest_score:
                    highest_score = config["score"]
                    best_model_name = name
        
        if requirements: # Placeholder for more complex requirement matching
            min_score = requirements.get("min_score")
            # capability_needed = requirements.get("capability")
            
            candidate_models = []
            for name, config in self.available_models.items():
                if config.get("available", False):
                    passes_score = (min_score is None or config["score"] >= min_score)
                    # passes_capability = (capability_needed is None or capability_needed in config.get("capabilities", []))
                    if passes_score: # and passes_capability:
                        candidate_models.append((name, config["score"]))
            
            if candidate_models:
                candidate_models.sort(key=lambda x: x[1], reverse=True) # Sort by score desc
                best_model_name = candidate_models[0][0]


        if best_model_name:
            logger.info(f"Selected model based on requirements/best score: {best_model_name}")
            return self.get_model(best_model_name)
        else:
            logger.warning("No suitable model found based on current selection criteria. Trying default.")
            return self.get_model() # Try default

    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Lists all configured models and their availability status.
        """
        return {
            name: {
                "score": config.get("score"),
                "available": config.get("available", False) and (self.get_api_key(name) is not None),
                "api_key_env": config.get("api_key_env")
            }
            for name, config in self.available_models.items()
        }

# Example Usage (for testing purposes):
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Mock config_manager for testing
    class MockConfigManager:
        def get_secret(self, key_name: str) -> Optional[str]:
            if key_name == "GOOGLE_API_KEY":
                return "test_google_key_from_config"
            # Return None for OPENAI_API_KEY to test fallback to env
            return None

    # Set a dummy env var for testing
    os.environ["OPENAI_API_KEY"] = "test_openai_key_from_env"
    
    DEFAULT_AVAILABLE_MODELS_TEST = {
        "gemini-pro": {"score": 100, "available": True, "api_key_env": "GOOGLE_API_KEY"},
        "gpt-4-turbo": {"score": 110, "available": True, "api_key_env": "OPENAI_API_KEY"},
        "gpt-3.5-turbo": {"score": 90, "available": False, "api_key_env": "OPENAI_API_KEY"},
        "local-llama": {"score": 80, "available": True, "api_key_env": "LOCAL_LLAMA_KEY"} # No key set
    }
    DEFAULT_MODEL_NAME_TEST = "gemini-pro"

    # Test with MockConfigManager
    print("--- Testing with MockConfigManager ---")
    mock_cfg_manager = MockConfigManager()
    ai_manager_with_config = AIModelManager(DEFAULT_AVAILABLE_MODELS_TEST, DEFAULT_MODEL_NAME_TEST, mock_cfg_manager)
    
    print("\nAvailable Models (with config manager):")
    for name, details in ai_manager_with_config.list_available_models().items():
        print(f"  {name}: {details}")

    print(f"\nGetting default model ({ai_manager_with_config.default_model_name}):")
    model = ai_manager_with_config.get_model()
    print(f"  Got model: {model['name'] if model else 'None'}")
    if model: print(f"  Model Config: {model['config']}")


    print("\nGetting gpt-4-turbo (should use env key):")
    gpt4_model = ai_manager_with_config.get_model("gpt-4-turbo")
    print(f"  Got model: {gpt4_model['name'] if gpt4_model else 'None'}")
    if gpt4_model: print(f"  Model Config: {gpt4_model['config']}")
    
    print("\nGetting local-llama (should fail due to no key):")
    llama_model = ai_manager_with_config.get_model("local-llama")
    print(f"  Got model: {llama_model['name'] if llama_model else 'None'}")


    # Test without MockConfigManager (relies purely on environment variables)
    print("\n--- Testing without MockConfigManager (env vars only) ---")
    # Unset GOOGLE_API_KEY to test that path for gemini-pro
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]

    ai_manager_env_only = AIModelManager(DEFAULT_AVAILABLE_MODELS_TEST, DEFAULT_MODEL_NAME_TEST)
    print("\nAvailable Models (env only):")
    for name, details in ai_manager_env_only.list_available_models().items():
        print(f"  {name}: {details}")

    print(f"\nGetting default model ({ai_manager_env_only.default_model_name}) (gemini-pro, should fail as key removed):")
    model_env = ai_manager_env_only.get_model() # gemini-pro
    print(f"  Got model: {model_env['name'] if model_env else 'None'}")

    print("\nSelecting best model (should be gpt-4-turbo as it has env key):")
    selected_model_env = ai_manager_env_only.select_model()
    print(f"  Selected model: {selected_model_env['name'] if selected_model_env else 'None'}")

    # Clean up env var
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
