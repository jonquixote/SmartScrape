"""
Debugging utility for AI guided strategy

This file provides a debugging wrapper to help diagnose issues in the AI guided strategy implementation.
"""

import logging
import traceback
import json
from typing import Dict, Any, Optional
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename="ai_strategy_debug.log")

logger = logging.getLogger("ai_strategy_debug")

def debug_ai_method(func):
    """
    Decorator to debug AI methods and catch any errors
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            logger.debug(f"ENTERING {func.__name__} with args: {args[1:]} and kwargs: {kwargs}")
            result = await func(*args, **kwargs)
            logger.debug(f"EXITING {func.__name__} with result: {result}")
            return result
        except Exception as e:
            logger.error(f"ERROR in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a fallback result
            if func.__name__ == "_determine_extraction_approach":
                return {"method": "standard", "reason": f"Error in determine_extraction_approach: {str(e)}"}
            elif func.__name__ == "_extract_with_ai_guidance":
                # This needs to return the fallback extraction result
                self = args[0]
                html_content = args[1] if len(args) > 1 else ""
                url = args[2] if len(args) > 2 else ""
                config = args[4] if len(args) > 4 else kwargs.get("config", {})
                logger.info(f"Falling back to _fallback_extraction for {url}")
                return await self._fallback_extraction(html_content, url, config)
            return None
    return wrapper

def patch_ai_guided_strategy(strategy_class):
    """
    Patch the AIGuidedStrategy class with debugging wrappers for key methods
    """
    # Patch _determine_extraction_approach
    original_determine_extraction = strategy_class._determine_extraction_approach
    strategy_class._determine_extraction_approach = debug_ai_method(original_determine_extraction)
    
    # Patch _extract_with_ai_guidance
    original_extract_with_ai = strategy_class._extract_with_ai_guidance
    strategy_class._extract_with_ai_guidance = debug_ai_method(original_extract_with_ai)
    
    # Patch _parse_extraction_response
    original_parse = strategy_class._parse_extraction_response
    
    def safe_parse_extraction_response(self, response_content):
        """Safe wrapper for _parse_extraction_response"""
        try:
            logger.debug(f"ENTER _parse_extraction_response with content type: {type(response_content)}")
            logger.debug(f"Content: {response_content}")
            
            # If the response is already a dictionary, return it directly
            if isinstance(response_content, dict):
                logger.debug("Response content is a dictionary, returning directly")
                return response_content
            
            # If None or empty, return empty dict
            if not response_content:
                logger.debug("Response content is empty, returning empty dict")
                return {}
            
            # Convert to string if it's not already
            response_text = str(response_content)
            
            result = original_parse(self, response_text)
            logger.debug(f"EXITING _parse_extraction_response with result: {result}")
            return result
        except Exception as e:
            logger.error(f"ERROR in _parse_extraction_response: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    # Replace the original method with our safe version
    strategy_class._parse_extraction_response = safe_parse_extraction_response
    
    logger.info("AIGuidedStrategy patched with debugging wrappers")
    return strategy_class
