"""
Semantic Content Extraction

This module provides AI-powered semantic content extraction capabilities,
enabling intelligent data extraction from various content types using
advanced language models and content understanding.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple

from bs4 import BeautifulSoup, Tag
import html2text
import trafilatura
from trafilatura.settings import use_config

from extraction.core.extraction_interface import SemanticExtractor
from extraction.helpers.ai_prompt_templates import (
    customize_prompt, select_prompt_for_content, 
    generate_schema_prompt, GENERAL_EXTRACTION_PROMPT
)
from core.service_interface import BaseService

# Configure logging
logger = logging.getLogger(__name__)

class AISemanticExtractor(SemanticExtractor, BaseService):
    """
    Extractor that uses AI models to extract structured data from content
    through semantic understanding and contextual analysis.
    """
    
    def __init__(self, context=None):
        """
        Initialize the AI semantic extractor.
        
        Args:
            context: Strategy context for accessing shared services
        """
        super().__init__(context)
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0  # No wrapping
        self._initialized = False
        
        # Configure HTML converter
        self.html_converter.protect_links = True
        self.html_converter.unicode_snob = True
        self.html_converter.emphasis_mark = '*'
        
        # Configure trafilatura
        self.traf_config = use_config()
        self.traf_config.set("DEFAULT", "include_comments", "false")
        self.traf_config.set("DEFAULT", "include_tables", "true")
        self.traf_config.set("DEFAULT", "include_images", "true")
        self.traf_config.set("DEFAULT", "include_links", "true")
    
    @property
    def name(self) -> str:
        """Return the name of this service."""
        return "ai_semantic_extractor"
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service with the given configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        if not self._initialized:
            logger.info("Initializing AISemanticExtractor service")
            self._initialized = True
    
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract data from content using AI semantic understanding.
        
        Args:
            content: Content to extract data from
            options: Optional extraction parameters
                
        Returns:
            Dictionary containing extracted data and metadata
        """
        logger.info("Extracting data using AI semantic understanding")
        
        # Implement simple extraction for now
        return {
            "success": True,
            "data": {"message": "AI extraction not fully implemented yet"},
            "extraction_method": "ai_semantic_stub"
        }
    
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this extractor can handle the given content.
        
        Args:
            content: Content to check compatibility with
            content_type: Optional hint about the content type
            
        Returns:
            True if the extractor can handle this content, False otherwise
        """
        return True
    
    def shutdown(self) -> None:
        """Clean up any resources used by the extractor."""
        if self._initialized:
            logger.info("Shutting down AISemanticExtractor")
            self._initialized = False
    
    # Implement required abstract methods
    
    def preprocess_content(self, content: Any) -> Any:
        """
        Preprocess content for AI analysis.
        
        Args:
            content: Raw content to preprocess
            
        Returns:
            Preprocessed content ready for AI analysis
        """
        if isinstance(content, str) and ('<html' in content or '<!DOCTYPE' in content):
            # Use trafilatura to extract main content
            extracted = trafilatura.extract(
                content,
                include_tables=True,
                include_images=True,
                include_links=True,
                output_format='xml',
                config=self.traf_config
            )
            
            if extracted:
                return extracted
            
            # Fall back to HTML to text conversion
            return self.html_converter.handle(content)
        elif isinstance(content, BeautifulSoup) or isinstance(content, Tag):
            # Convert BeautifulSoup to string
            return str(content)
        elif isinstance(content, dict) or isinstance(content, list):
            # Convert JSON to string
            return json.dumps(content, indent=2)
        else:
            # Return content as is
            return content
    
    def generate_prompt(self, content: Any, extraction_goal: str) -> str:
        """
        Generate an AI prompt for extraction.
        
        Args:
            content: Content to extract from
            extraction_goal: Description of what to extract
            
        Returns:
            Formatted prompt string
        """
        # Convert content to text if needed
        if isinstance(content, (dict, list)):
            content_text = json.dumps(content, indent=2)
        elif isinstance(content, (BeautifulSoup, Tag)):
            content_text = str(content)
        else:
            content_text = str(content)
        
        # Select an appropriate prompt template based on the extraction goal
        prompt_template = select_prompt_for_content(extraction_goal) if extraction_goal else GENERAL_EXTRACTION_PROMPT
        
        # Customize the prompt with the content
        prompt = customize_prompt(
            prompt_template,
            content=content_text,
            goal=extraction_goal
        )
        
        return prompt
    
    def parse_ai_response(self, response: str) -> Dict[str, Any]:
        """
        Parse AI model response into structured data.
        
        Args:
            response: Raw response from AI model
            
        Returns:
            Structured data extracted from response
        """
        try:
            # Extract JSON from the response (may contain text before or after JSON)
            json_match = re.search(r'```json\s*({.*?})\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Use the full response as a fallback
                    json_str = response
            
            # Parse the JSON
            extracted_data = json.loads(json_str)
            return extracted_data
            
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON from AI response")
            # Return a simple structure with the raw response
            return {
                "error": "Failed to parse structured data",
                "raw_response": response
            }
    
    def calculate_semantic_confidence(self, result: Dict[str, Any], content: Any) -> float:
        """
        Calculate confidence score for semantic extraction.
        
        Args:
            result: Extraction result to evaluate
            content: Original content for reference
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not result or not isinstance(result, dict):
            return 0.0
        
        # If result already has a confidence score, use it
        if isinstance(result, dict) and "_metadata" in result and "confidence" in result["_metadata"]:
            return float(result["_metadata"]["confidence"])
        
        # Use result data or the whole result
        data = result.get("data", result)
        
        if not data or not isinstance(data, dict):
            return 0.0
        
        # Calculate confidence based on data completeness
        field_count = len([k for k in data.keys() if not k.startswith("_")])
        non_empty_field_count = len([k for k in data.keys() if not k.startswith("_") and data.get(k)])
        
        # Base confidence on ratio of non-empty fields
        if field_count == 0:
            return 0.0
            
        return min(1.0, non_empty_field_count / field_count)
