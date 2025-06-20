import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import re

class ModelSelector:
    """Service for selecting the optimal AI model based on task requirements."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model selector with configuration.
        
        Args:
            config: Configuration dictionary containing model definitions and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger("model_selector")
        self.models_info = self._initialize_models_info()
        
        # Handle both dictionary and list configurations
        if isinstance(self.config, dict):
            self.default_model = self.config.get("default_model", "gemini-2.0-flash-lite") # Use latest 2.0 model
        else:
            # If config is a list (like in the tests), use a sensible default
            self.default_model = "gemini-2.0-flash-lite" # Use latest 2.0 model
            # Attempt to find a model name from the list if it's structured that way
            if isinstance(self.config, list):
                for model_cfg in self.config:
                    if isinstance(model_cfg, dict) and "name" in model_cfg:
                        self.default_model = model_cfg["name"]
                        break
        
    def _initialize_models_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize models information from config.
        
        Returns:
            Dictionary mapping model names to their information
        """
        models_info = {}
        
        # Process models from config
        models_list = []
        if isinstance(self.config, dict) and "models" in self.config:
            # Extract models from the dictionary configuration
            models_list = self.config.get("models", [])
        elif isinstance(self.config, list):
            # Use the list directly if it's already a list
            models_list = self.config
            
        for model_config in models_list:
            name = model_config.get("name")
            if not name:
                continue
                
            models_info[name] = {
                "type": model_config.get("type", "unknown"),
                "capabilities": model_config.get("capabilities", {}),
                "quality_score": model_config.get("quality_score", 5),
                "speed_score": model_config.get("speed_score", 5),
                "cost_score": model_config.get("cost_score", 5),
                "context_length": model_config.get("context_length", 4096),
                "cost_per_1k_tokens": model_config.get("cost_per_1k_tokens", (0.01, 0.01)),
                "task_specializations": model_config.get("task_specializations", []),
            }
            
        # If no models were configured, add some defaults
        if not models_info:
            models_info = {
                "gemini-2.0-flash-lite": { # Primary default - latest and most cost-effective
                    "type": "google",
                    "capabilities": {
                        "streaming": True,
                        "function_calling": True, 
                        "json_mode": True,  # Google models can handle JSON output
                        "vision": True
                    },
                    "quality_score": 8, # High quality for latest model
                    "speed_score": 10,  # Lite version is very fast
                    "cost_score": 10,   # Most cost-effective
                    "context_length": 1048576, # Large context window
                    "cost_per_1k_tokens": (0.000075, 0.0003),
                    "task_specializations": ["general", "summarization", "qa", "extraction", "chat"],
                },
                "gemini-2.0-flash": { # Secondary default - better quality
                    "type": "google",
                    "capabilities": {
                        "streaming": True,
                        "function_calling": True, 
                        "json_mode": True,  # Google models can handle JSON output
                        "vision": True
                    },
                    "quality_score": 9, # Higher quality
                    "speed_score": 9,   # Fast
                    "cost_score": 9,    # Good cost-effectiveness
                    "context_length": 1048576, # Large context window
                    "cost_per_1k_tokens": (0.000125, 0.000375),
                    "task_specializations": ["general", "summarization", "qa", "extraction", "chat", "complex_reasoning"],
                },
                "gemini-1.5-flash": { # Fallback option
                    "type": "google",
                    "capabilities": {
                        "streaming": True,
                        "function_calling": True, 
                        "json_mode": True,  # Google models can handle JSON output
                        "vision": True
                    },
                    "quality_score": 7, # Good quality for a fast model
                    "speed_score": 9,   # Flash models are fast
                    "cost_score": 9,    # Flash models are cost-effective
                    "context_length": 1048576, # Large context window
                    "cost_per_1k_tokens": (0.000125, 0.000375),
                    "task_specializations": ["general", "summarization", "qa", "extraction", "chat"],
                },
                "gpt-3.5-turbo": {
                    "type": "openai",
                    "capabilities": {
                        "streaming": True,
                        "function_calling": True,
                        "json_mode": True
                    },
                    "quality_score": 6,
                    "speed_score": 8,
                    "cost_score": 8,
                    "context_length": 4096,
                    "cost_per_1k_tokens": (0.0015, 0.002),
                    "task_specializations": ["classification", "qa", "summarization"],
                },
                "gpt-4": {
                    "type": "openai",
                    "capabilities": {
                        "streaming": True,
                        "function_calling": True,
                        "json_mode": True
                    },
                    "quality_score": 9,
                    "speed_score": 5,
                    "cost_score": 3,
                    "context_length": 8192,
                    "cost_per_1k_tokens": (0.03, 0.06),
                    "task_specializations": ["analysis", "code", "complex_reasoning"],
                },
                "claude-2": { # Retaining Claude as another option
                    "type": "anthropic",
                    "capabilities": {
                        "streaming": True,
                        "function_calling": False,
                        "json_mode": False
                    },
                    "quality_score": 8,
                    "speed_score": 6,
                    "cost_score": 5,
                    "context_length": 100000,
                    "cost_per_1k_tokens": (0.008, 0.024),
                    "task_specializations": ["summarization", "writing", "conversation"],
                }
            }
            
        return models_info
    
    def select_model(self, 
                     task_type: str = "general", 
                     content_length: Optional[int] = None,
                     require_capabilities: Optional[List[str]] = None,
                     quality_priority: int = 5,
                     speed_priority: int = 5,
                     cost_priority: int = 5) -> str:
        """
        Select the most appropriate model based on task requirements.
        
        Args:
            task_type: Type of task (e.g., "classification", "generation", "extraction")
            content_length: Approximate length of content in tokens
            require_capabilities: List of required capabilities (e.g., "function_calling")
            quality_priority: Priority for quality (1-10, higher means more important)
            speed_priority: Priority for speed (1-10, higher means more important)
            cost_priority: Priority for cost efficiency (1-10, higher means more important)
            
        Returns:
            Name of the selected model
        """
        if not self.models_info:
            self.logger.warning("No models available for selection, using default")
            return self.default_model
            
        candidates = {}
        require_capabilities = require_capabilities or []
        
        # First, filter models by mandatory capabilities
        for name, info in self.models_info.items():
            capabilities = info.get("capabilities", {})
            
            # Skip models that lack required capabilities
            if any(req_cap not in capabilities or not capabilities[req_cap] 
                  for req_cap in require_capabilities):
                continue
                
            # Skip models with insufficient context length
            if content_length and content_length > info.get("context_length", 0):
                continue
                
            candidates[name] = info
        
        # If no candidates, return default
        if not candidates:
            self.logger.warning("No models meet the required capabilities, using default")
            return self.default_model
            
        # Score candidates based on priorities
        scores = {}
        for name, info in candidates.items():
            # Base score starts at 0
            score = 0
            
            # Add weighted quality score
            score += info.get("quality_score", 5) * quality_priority
            
            # Add weighted speed score
            score += info.get("speed_score", 5) * speed_priority
            
            # Add weighted cost score (higher is better/cheaper)
            score += info.get("cost_score", 5) * cost_priority
            
            # Bonus for task specialization
            if task_type in info.get("task_specializations", []):
                score += 10  # Significant bonus for task specialization
                
            # Apply penalties for overkill (using expensive models for simple tasks)
            if task_type in ["classification", "simple_extraction"] and info.get("quality_score", 5) > 7:
                score -= 5  # Penalty for using high-quality models for simple tasks
                
            scores[name] = score
        
        # Get the model with the highest score
        if not scores:
            return self.default_model
            
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"Selected model {best_model} with score {scores[best_model]:.2f} for task {task_type}")
        return best_model
    
    def analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze task description to determine complexity, type, and requirements.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            Dictionary with analysis results including:
            - complexity: "low", "medium", or "high"
            - task_type: Detected task type
            - required_capabilities: List of capabilities needed
            - estimated_tokens: Rough estimate of tokens needed
        """
        # Initialize with default values
        analysis = {
            "complexity": "medium",
            "task_type": "general",
            "required_capabilities": [],
            "estimated_tokens": 500  # Default token estimate
        }
        
        # Check for indicators of complexity
        complexity_indicators = {
            "high": ["analyze in detail", "comprehensive", "complex", "deep dive", 
                    "thorough", "nuanced", "sophisticated", "advanced analysis"],
            "low": ["brief", "quick", "simple", "straightforward", "basic",
                   "short", "just", "only", "merely"]
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in task_description.lower() for indicator in indicators):
                analysis["complexity"] = level
                break
        
        # Check for specific task types
        task_type_indicators = {
            "classification": ["categorize", "classify", "identify type", "label", 
                              "sort into", "which category"],
            "extraction": ["extract", "pull out", "get the", "find the", "identify the"],
            "summarization": ["summarize", "summary", "tl;dr", "brief overview", 
                             "main points", "key points"],
            "qa": ["answer", "question", "explain why", "how does", "what is", 
                  "who is", "where is", "when did"],
            "generation": ["create", "generate", "write", "compose", "draft"],
            "translation": ["translate", "conversion", "convert to", "change to"],
            "code": ["code", "function", "program", "script", "algorithm", 
                    "implement", "coding", "programming language"],
            "analysis": ["analyze", "analysis", "evaluate", "assessment", 
                        "examine", "investigate", "review", "study"]
        }
        
        for task_type, indicators in task_type_indicators.items():
            if any(indicator in task_description.lower() for indicator in indicators):
                analysis["task_type"] = task_type
                break
        
        # Check for capabilities needed
        capability_indicators = {
            "function_calling": ["api", "call function", "tool use", "use tool", 
                                "structured output function"],
            "json_mode": ["json", "structured output", "structured data", 
                         "formatted data", "structured response"],
            "vision": ["image", "picture", "photo", "visual", "diagram", 
                      "screenshot", "graph", "chart"]
        }
        
        for capability, indicators in capability_indicators.items():
            if any(indicator in task_description.lower() for indicator in indicators):
                analysis["required_capabilities"].append(capability)
        
        # Estimate tokens needed based on complexity and task type
        token_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0
        }
        
        base_tokens = {
            "summarization": 800,
            "analysis": 1000,
            "code": 1200,
            "qa": 500,
            "classification": 300,
            "extraction": 400,
            "generation": 800,
            "translation": 600,
            "general": 500
        }
        
        # Calculate estimated tokens based on task type and complexity
        base = base_tokens.get(analysis["task_type"], 500)
        multiplier = token_multipliers.get(analysis["complexity"], 1.0)
        analysis["estimated_tokens"] = int(base * multiplier)
        
        # Further adjust based on specifics in the task description
        if "detailed" in task_description.lower() or "thorough" in task_description.lower():
            analysis["estimated_tokens"] = int(analysis["estimated_tokens"] * 1.5)
            
        if "long" in task_description.lower() or "extensive" in task_description.lower():
            analysis["estimated_tokens"] = int(analysis["estimated_tokens"] * 1.5)
            
        return analysis
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int = None) -> float:
        """
        Estimate the cost of using a specific model for the given token counts.
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (if None, estimated as 1/3 of input)
            
        Returns:
            Estimated cost in USD
        """
        if output_tokens is None:
            output_tokens = input_tokens // 3  # Rough estimate if not provided
            
        if model_name not in self.models_info:
            # Use default cost estimate if model not found
            return (input_tokens + output_tokens) * 0.01 / 1000
            
        input_cost, output_cost = self.models_info[model_name].get("cost_per_1k_tokens", (0.01, 0.01))
        
        # Calculate total cost
        total_cost = (input_tokens * input_cost + output_tokens * output_cost) / 1000
        return total_cost
    
    def suggest_models(self, task_description: str, content_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze task and suggest appropriate models with explanations.
        
        Args:
            task_description: Natural language description of the task
            content_length: Optional length of content in tokens
            
        Returns:
            List of model suggestions with rationales
        """
        # Analyze the task
        analysis = self.analyze_task_complexity(task_description)
        
        # Set content length if provided
        if content_length:
            analysis["estimated_tokens"] = content_length
        
        # Adjust priorities based on analysis
        quality_priority = 8 if analysis["complexity"] == "high" else 5
        speed_priority = 8 if analysis["complexity"] == "low" else 5
        cost_priority = 5  # Default
        
        # Get best model
        best_model = self.select_model(
            task_type=analysis["task_type"],
            content_length=analysis["estimated_tokens"],
            require_capabilities=analysis["required_capabilities"],
            quality_priority=quality_priority,
            speed_priority=speed_priority,
            cost_priority=cost_priority
        )
        
        # Prepare suggestions
        suggestions = []
        
        # Add the best model as primary suggestion
        best_model_info = self.models_info.get(best_model, {})
        suggestions.append({
            "name": best_model,
            "is_primary_suggestion": True,
            "rationale": f"Best overall model for {analysis['task_type']} tasks with {analysis['complexity']} complexity",
            "advantages": [
                f"Good balance of quality, speed, and cost",
                f"Supports {', '.join(cap for cap, enabled in best_model_info.get('capabilities', {}).items() if enabled) if best_model_info.get('capabilities') else 'standard capabilities'}",
                f"Specialized for {', '.join(best_model_info.get('task_specializations', ['general tasks'])) if best_model_info.get('task_specializations') else 'general tasks'}"
            ],
            "estimated_cost": self.estimate_cost(best_model, analysis["estimated_tokens"])
        })
        
        # Add alternative suggestions (e.g., cheaper option, higher quality option)
        
        # Find a cheaper alternative
        if cost_priority < 8:
            cost_focused = self.select_model(
                task_type=analysis["task_type"],
                content_length=analysis["estimated_tokens"],
                require_capabilities=analysis["required_capabilities"],
                quality_priority=3,
                speed_priority=5,
                cost_priority=10
            )
            
            if cost_focused != best_model:
                cost_model_info = self.models_info.get(cost_focused, {})
                cost_estimate = self.estimate_cost(cost_focused, analysis["estimated_tokens"])
                best_cost = self.estimate_cost(best_model, analysis["estimated_tokens"])
                
                if cost_estimate < best_cost:
                    suggestions.append({
                        "name": cost_focused,
                        "is_primary_suggestion": False,
                        "rationale": "More cost-effective alternative",
                        "advantages": [
                            f"Lower cost (${cost_estimate:.4f} vs ${best_cost:.4f})",
                            f"Still suitable for {analysis['task_type']} tasks",
                            f"Good choice for budget-conscious usage"
                        ],
                        "estimated_cost": cost_estimate
                    })
        
        # Find a higher quality alternative
        if quality_priority < 8 and analysis["complexity"] != "low":
            quality_focused = self.select_model(
                task_type=analysis["task_type"],
                content_length=analysis["estimated_tokens"],
                require_capabilities=analysis["required_capabilities"],
                quality_priority=10,
                speed_priority=3,
                cost_priority=3
            )
            
            if quality_focused != best_model:
                quality_model_info = self.models_info.get(quality_focused, {})
                quality_cost_estimate = self.estimate_cost(quality_focused, analysis["estimated_tokens"]) # Renamed variable
                
                suggestions.append({
                    "name": quality_focused,
                    "is_primary_suggestion": False,
                    "rationale": "Higher quality alternative",
                    "advantages": [
                        f"Superior quality for complex tasks",
                        f"Better handles nuanced {analysis['task_type']} tasks",
                        f"Recommended when accuracy is critical"
                    ],
                    "estimated_cost": quality_cost_estimate # Used renamed variable
                })
        
        return suggestions