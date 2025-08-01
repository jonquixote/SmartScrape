"""
AI Intent Parser Module

This module provides functionality to parse user prompts and extract structured intent
data that can drive smart scraping operations. It uses Gemini AI models to analyze
natural language requests and extract relevant entities, properties, and constraints.
"""

import json
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional

import config
from ai_helpers.prompt_generator import parse_user_prompt
from ai_helpers.rule_based_extractor import extract_intent
from utils.ai_cache import AIResponseCache
from core.ai_service import AIService
from core.service_interface import BaseService
from .rule_based_extractor import extract_intent_with_rules

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntentParser")

class IntentParser(BaseService):
    """
    Parses natural language queries into structured intents using the centralized AI service.
    """
    
    def __init__(self, ai_service: Optional[AIService] = None):
        """
        Initialize the IntentParser with an AIService instance.
        
        Args:
            ai_service: Optional AIService instance. If None, will attempt to get from service registry.
        """
        self._initialized = False
        self._ai_service = ai_service
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # No specific configuration needed here, but could be added in the future
        
        self._initialized = True
        logger.info("IntentParser service initialized")
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._initialized = False
        logger.info("IntentParser service shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "intent_parser"
    
    @property
    def ai_service(self) -> AIService:
        """Get the AI service, retrieving from registry if needed."""
        if not self._ai_service:
            # Import here to avoid circular imports
            from core.service_registry import ServiceRegistry
            registry = ServiceRegistry()
            self._ai_service = registry.get_service("ai_service")
            if not self._ai_service:
                raise ValueError("AIService not found in ServiceRegistry")
        return self._ai_service
    
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query to extract structured intent.
        
        Args:
            query: The natural language query
            
        Returns:
            A dictionary containing the structured intent
        """
        # First try rule-based extraction for high-confidence entities
        rule_result = extract_intent_with_rules(query)
        
        # Use AI to extract complete structured information
        prompt = self._build_intent_extraction_prompt(query, rule_result)
        
        # Use the AI service to get a response
        response = await self.ai_service.generate_response(
            prompt=prompt,
            context={
                "use_cache": True,  # Enable caching for intent parsing
                "task_type": "extraction",  # Help model selector pick appropriate model
                "quality_priority": 7,  # Intent parsing benefits from higher quality
                "speed_priority": 8,  # But also needs to be responsive
                "options": {
                    "temperature": 0.2  # Lower temperature for more deterministic extraction
                }
            }
        )
        
        return self._process_intent_response(response, query, rule_result)
    
    async def parse_user_prompt(self, user_prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Parse a user prompt to extract structured intent - alias for parse_query for backwards compatibility.
        
        Args:
            user_prompt: The user's natural language prompt
            context: Optional context information (URLs, etc.)
            
        Returns:
            A dictionary containing the structured intent
        """
        logger.info(f"Parsing user prompt: {user_prompt[:100]}...")
        
        try:
            # Use the existing parse_query method
            result = await self.parse_query(user_prompt)
            
            # Add context information if provided
            if context:
                result["context"] = context
                if "urls" in context:
                    result["start_urls"] = context["urls"]
            
            logger.info(f"Intent parsing successful: {result.get('primary_intent', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing user prompt: {e}")
            # Return a fallback intent structure
            return {
                "primary_intent": user_prompt,
                "entities": [],
                "actions": ["extract"],
                "context": context or {},
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _build_intent_extraction_prompt(self, query: str, rule_result: Dict[str, Any]) -> str:
        """
        Build a structured prompt for intent extraction, focusing on keywords and entities.
        
        Args:
            query: The original user query
            rule_result: Results from rule-based extraction (used as a hint)
            
        Returns:
            A structured prompt for the AI model
        """
        return f"""
        Analyze the following user query to extract key information for a web scraping task.
        
        USER QUERY: "{query}"
        
        Your goal is to identify the core components of the user's request.
        Provide a JSON response with the following structure:
        {{
            "primary_intent": "A concise summary of what the user wants to find or do.",
            "keywords": ["list", "of", "important", "keywords", "from", "the", "query"],
            "entities": ["list", "of", "named", "entities", "like", "people", "organizations", "or", "products"]
        }}
        
        IMPORTANT INSTRUCTIONS:
        1.  **primary_intent**: Summarize the user's goal in a short phrase.
        2.  **keywords**: Extract the most relevant terms that would be useful for a web search. Include action words (e.g., "find", "compare"), topics, and attributes.
        3.  **entities**: Identify specific named things. If the user mentions "Apple stock," "Apple" is the entity. If they mention "reviews for the new iPhone," "iPhone" is the entity.
        
        Focus on extracting information that will help find the right web pages and extract the correct data.
        Return ONLY valid JSON.
        """
    
    def _process_intent_response(self, 
                                response: Dict[str, Any], 
                                original_query: str,
                                rule_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the AI response to extract structured intent.
        
        Args:
            response: The AI service response
            original_query: The original query for fallback
            rule_result: The rule-based extraction result for fallback
            
        Returns:
            The structured intent dictionary
        """
        try:
            content = response.get("content", "")
            
            # Extract JSON from the response
            json_str = self._extract_json_from_text(content)
            if not json_str:
                logger.warning("No JSON found in AI response. Using fallback.")
                return self._create_fallback_intent(original_query, rule_result)
            
            # Parse the JSON
            intent = json.loads(json_str)
            
            # Validate and ensure required fields exist, providing sensible defaults
            validated_intent = {
                "original_query": original_query,
                "primary_intent": intent.get("primary_intent", original_query),
                "keywords": intent.get("keywords", rule_result.get("keywords", [])),
                "entities": intent.get("entities", rule_result.get("entities", []))
            }
            
            # If keywords are empty, generate them from the query as a last resort
            if not validated_intent["keywords"]:
                validated_intent["keywords"] = re.findall(r'\b\w+\b', original_query.lower())

            return validated_intent
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from AI response: {e}. Content: {content[:200]}")
            return self._create_fallback_intent(original_query, rule_result)
        except Exception as e:
            logger.error(f"An unexpected error occurred during intent processing: {e}")
            # Return a fallback intent if any other parsing fails
            return self._create_fallback_intent(original_query, rule_result)
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON from response text."""
        if not text:
            return None
        
        # Try to find JSON block in markdown
        if "```json" in text:
            pattern = r"```json\s*(.+?)\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        elif "```" in text:
            pattern = r"```\s*(.+?)\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
        
        # Try to find JSON-like structure
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            return text[start_idx:end_idx+1]
        
        return None
    
    def _create_fallback_intent(self, query: str, rule_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback intent when AI extraction fails."""
        logger.info(f"Creating fallback intent for query: '{query}'")
        # Use rule results as the primary source for the fallback
        keywords = rule_result.get("keywords", [])
        # If rule-based keywords are empty, generate from query
        if not keywords:
            keywords = re.findall(r'\b\w+\b', query.lower())

        fallback = {
            "original_query": query,
            "primary_intent": query, # Default to the original query
            "keywords": keywords,
            "entities": rule_result.get("entities", [])
        }
        return fallback
    
    async def parse_intent(self, query: str) -> Dict[str, Any]:
        """
        Alias for parse_query for backward compatibility.
        
        Args:
            query: The natural language query
            
        Returns:
            A dictionary containing the structured intent
        """
        return await self.parse_query(query)
    
    async def enhance_intent_with_context(self, intent: Dict[str, Any], html_content: str, url: str) -> Dict[str, Any]:
        """
        Enhance existing intent with context from webpage content.
        
        Args:
            intent: The existing intent dictionary
            html_content: HTML content of the webpage
            url: URL of the webpage
            
        Returns:
            Enhanced intent dictionary
        """
        try:
            # Create a prompt to enhance intent with page context
            prompt = f"""
            Based on this webpage content, enhance the user's search intent with relevant information.
            
            Current Intent:
            {json.dumps(intent, indent=2)}
            
            Webpage URL: {url}
            
            HTML Content (first 3000 chars):
            {html_content[:3000]}
            
            Analyze the page and return an enhanced version of the intent JSON with additional relevant context that could help with data extraction.
            """
            
            response = await self.ai_service.generate_response(
                prompt=prompt,
                context={
                    "task_type": "intent_enhancement",
                    "use_cache": True,
                    "options": {
                        "temperature": 0.2
                    }
                }
            )
            
            if response and "content" in response:
                # Try to extract enhanced intent from response
                content = response["content"]
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        enhanced_intent = json.loads(json_match.group())
                        return enhanced_intent
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # If enhancement fails, return original intent
            return intent
            
        except Exception as e:
            logger.warning(f"Failed to enhance intent with context: {e}")
            return intent
    
    async def analyze_extraction_requirements(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze intent to determine extraction requirements.
        
        Args:
            intent: The user intent dictionary
            
        Returns:
            Dictionary containing extraction requirements
        """
        try:
            prompt = f"""
            Based on this user intent, determine what data needs to be extracted from webpages.
            
            User Intent:
            {json.dumps(intent, indent=2)}
            
            Return a JSON object with extraction requirements including:
            - fields: list of field names to extract
            - data_types: mapping of field names to expected data types
            - required_fields: list of required field names
            - optional_fields: list of optional field names
            - extraction_rules: any specific extraction rules or patterns
            """
            
            response = await self.ai_service.generate_response(
                prompt=prompt,
                context={
                    "task_type": "extraction_analysis",
                    "use_cache": True,
                    "options": {
                        "temperature": 0.2
                    }
                }
            )
            
            if response and "content" in response:
                content = response["content"]
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        requirements = json.loads(json_match.group())
                        return requirements
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Return basic requirements if AI fails
            return {
                "fields": intent.get("properties", []),
                "data_types": {},
                "required_fields": [],
                "optional_fields": intent.get("properties", []),
                "extraction_rules": []
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze extraction requirements: {e}")
            return {
                "fields": intent.get("properties", []),
                "data_types": {},
                "required_fields": [],
                "optional_fields": intent.get("properties", []),
                "extraction_rules": []
            }
    
    async def generate_search_parameters(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate search parameters based on user intent.
        
        Args:
            intent: The user intent dictionary
            
        Returns:
            Dictionary containing search parameters
        """
        try:
            prompt = f"""
            Based on this user intent, generate search parameters that could be used on website search forms.
            
            User Intent:
            {json.dumps(intent, indent=2)}
            
            Return a JSON object with search parameters as key-value pairs that could be used in search forms.
            Include relevant keywords, location filters, category filters, etc.
            """
            
            response = await self.ai_service.generate_response(
                prompt=prompt,
                context={
                    "task_type": "search_params_generation",
                    "use_cache": True,
                    "options": {
                        "temperature": 0.3
                    }
                }
            )
            
            if response and "content" in response:
                content = response["content"]
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        search_params = json.loads(json_match.group())
                        return search_params
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Return basic search parameters if AI fails
            basic_params = {}
            if "keywords" in intent:
                basic_params["keywords"] = " ".join(intent["keywords"])
            if "target_item" in intent:
                basic_params["query"] = intent["target_item"]
            if "location" in intent and intent["location"]:
                if "city" in intent["location"]:
                    basic_params["location"] = intent["location"]["city"]
                if "state" in intent["location"]:
                    basic_params["state"] = intent["location"]["state"]
            
            return basic_params
            
        except Exception as e:
            logger.warning(f"Failed to generate search parameters: {e}")
            # Return basic search parameters
            basic_params = {}
            if "keywords" in intent:
                basic_params["keywords"] = " ".join(intent["keywords"])
            if "target_item" in intent:
                basic_params["query"] = intent["target_item"]
            return basic_params

# Legacy function for backward compatibility
async def parse_user_intent(query: str) -> Dict[str, Any]:
    """Legacy function that uses the IntentParser class."""
    parser = IntentParser()
    return await parser.parse_query(query)

# Function to get a singleton instance
_intent_parser_instance = None

def get_intent_parser(use_ai: bool = True) -> IntentParser:
    """
    Get the singleton intent parser instance.
    
    Args:
        use_ai: Whether to use AI for intent parsing
        
    Returns:
        IntentParser instance
    """
    global _intent_parser_instance
    if _intent_parser_instance is None:
        _intent_parser_instance = IntentParser()
    return _intent_parser_instance
