"""
JavaScript Variable Extractor Helper

This module provides extraction capabilities for JavaScript variables and JSON data
embedded in HTML script tags, focusing on extracting structured data from JavaScript
variables, configurations, and JSON objects in the page.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

class JavaScriptExtractor:
    """
    Extractor for JavaScript variables and JSON data in HTML scripts.
    
    This extractor focuses on extracting structured data from JavaScript variables,
    configuration objects, and JSON objects embedded in <script> tags.
    """
    
    def __init__(self):
        """Initialize the JavaScript extractor."""
        self._variable_regex = re.compile(r'(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*({.*?});', re.DOTALL)
        self._json_regex = re.compile(r'({[\s\S]*?})(?=;|$)')
        self._window_property_regex = re.compile(r'window\.([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*({.*?});', re.DOTALL)
        self._config_regex = re.compile(r'(?:config|CONFIG|configuration|CONFIGURATION)(?:\s*=\s*|\s*:\s*)({.*?})(?:;|,|})', re.DOTALL)
    
    def extract_js_variables(self, html: Union[str, BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Extract JavaScript variables from HTML <script> tags.
        
        Args:
            html: HTML content to extract from
            
        Returns:
            Dictionary of extracted variable names and their parsed values
        """
        soup = self._ensure_soup(html)
        variables = {}
        
        # Process all script tags
        for script in soup.find_all("script"):
            if not script.string:
                continue
            
            # Skip script tags with specific types that won't contain JS variables
            if script.get("type") and script.get("type") not in ["text/javascript", "application/javascript", ""]:
                continue
            
            script_content = script.string
            
            # Extract variables defined with var, let, or const
            for var_match in self._variable_regex.finditer(script_content):
                var_name = var_match.group(1)
                var_value = var_match.group(2)
                
                try:
                    # Try to parse the value as JSON
                    parsed_value = self._safe_parse_json(var_value)
                    if parsed_value:
                        variables[var_name] = parsed_value
                except Exception as e:
                    logger.debug(f"Failed to parse variable {var_name}: {str(e)}")
            
            # Extract window properties
            for prop_match in self._window_property_regex.finditer(script_content):
                prop_name = prop_match.group(1)
                prop_value = prop_match.group(2)
                
                try:
                    # Try to parse the value as JSON
                    parsed_value = self._safe_parse_json(prop_value)
                    if parsed_value:
                        variables[f"window.{prop_name}"] = parsed_value
                except Exception as e:
                    logger.debug(f"Failed to parse window property {prop_name}: {str(e)}")
        
        return variables
    
    def extract_json_objects(self, script_content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON objects from JavaScript code.
        
        Args:
            script_content: JavaScript code to extract from
            
        Returns:
            List of extracted JSON objects
        """
        json_objects = []
        
        # Find potential JSON objects
        for json_match in self._json_regex.finditer(script_content):
            json_str = json_match.group(1)
            
            try:
                # Try to parse as JSON
                json_obj = self._safe_parse_json(json_str)
                if json_obj and isinstance(json_obj, dict):
                    json_objects.append(json_obj)
            except Exception as e:
                logger.debug(f"Failed to parse JSON object: {str(e)}")
        
        return json_objects
    
    def extract_config_objects(self, script_content: str) -> Dict[str, Any]:
        """
        Extract configuration objects from JavaScript code.
        
        Args:
            script_content: JavaScript code to extract from
            
        Returns:
            Dictionary of extracted configuration objects
        """
        config_objects = {}
        
        # Find potential config objects
        for config_idx, config_match in enumerate(self._config_regex.finditer(script_content)):
            config_str = config_match.group(1)
            
            try:
                # Try to parse as JSON
                config_obj = self._safe_parse_json(config_str)
                if config_obj and isinstance(config_obj, dict):
                    config_name = f"config_{config_idx}"
                    config_objects[config_name] = config_obj
            except Exception as e:
                logger.debug(f"Failed to parse config object: {str(e)}")
        
        return config_objects
    
    def map_js_data_to_schema(self, js_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map JavaScript data to a standard schema format.
        
        Args:
            js_data: JavaScript data to map
            
        Returns:
            Mapped data in a standardized format
        """
        mapped_data = {}
        
        # Look for common data patterns in the extracted JavaScript
        
        # Look for page metadata
        metadata_keys = [
            "meta", "metadata", "pageData", "pageInfo", "page", "pageMetadata",
            "pageMeta", "siteData", "siteInfo", "site", "siteMetadata", "siteMeta"
        ]
        
        for key in metadata_keys:
            if key in js_data and isinstance(js_data[key], dict):
                mapped_data["metadata"] = js_data[key]
                break
        
        # Look for product data
        product_keys = [
            "product", "productData", "productInfo", "productDetails",
            "item", "itemData", "itemInfo", "itemDetails"
        ]
        
        for key in product_keys:
            if key in js_data and isinstance(js_data[key], dict):
                mapped_data["product"] = js_data[key]
                break
        
        # Look for user data
        user_keys = [
            "user", "userData", "userInfo", "userDetails",
            "customer", "customerData", "customerInfo", "customerDetails"
        ]
        
        for key in user_keys:
            if key in js_data and isinstance(js_data[key], dict):
                mapped_data["user"] = js_data[key]
                break
        
        # Look for configuration
        config_keys = [
            "config", "configuration", "settings", "options",
            "CONFIG", "CONFIGURATION", "SETTINGS", "OPTIONS"
        ]
        
        for key in config_keys:
            if key in js_data and isinstance(js_data[key], dict):
                mapped_data["config"] = js_data[key]
                break
        
        # Look for catalog/listing data
        catalog_keys = [
            "catalog", "products", "items", "listings", "results",
            "catalogData", "productList", "itemList", "listingData"
        ]
        
        for key in catalog_keys:
            if key in js_data and isinstance(js_data[key], (list, dict)):
                mapped_data["catalog"] = js_data[key]
                break
        
        # If no structured data found yet, include all data
        if not mapped_data:
            return {"raw_data": js_data}
        
        return mapped_data
    
    def extract_all_data(self, html: Union[str, BeautifulSoup, Tag]) -> Dict[str, Any]:
        """
        Extract all JavaScript data from HTML.
        
        Args:
            html: HTML content to extract from
            
        Returns:
            Dictionary of all extracted data
        """
        soup = self._ensure_soup(html)
        all_data = {
            "variables": {},
            "configs": {},
            "json_objects": []
        }
        
        # Extract variables
        variables = self.extract_js_variables(soup)
        all_data["variables"] = variables
        
        # Process all script tags for config objects and JSON objects
        for script in soup.find_all("script"):
            if not script.string:
                continue
            
            script_content = script.string
            
            # Extract config objects
            configs = self.extract_config_objects(script_content)
            all_data["configs"].update(configs)
            
            # Extract JSON objects
            json_objects = self.extract_json_objects(script_content)
            all_data["json_objects"].extend(json_objects)
        
        # Create a standardized mapping
        all_data["mapped_data"] = self.map_js_data_to_schema(all_data["variables"])
        
        return all_data
    
    def _safe_parse_json(self, json_str: str) -> Optional[Any]:
        """
        Safely parse a JSON string with fallbacks for JavaScript syntax.
        
        Args:
            json_str: JSON or JavaScript object string to parse
            
        Returns:
            Parsed object or None if parsing fails
        """
        try:
            # Try standard JSON parsing first
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that fails, try to make it valid JSON
            try:
                # Replace single quotes with double quotes
                fixed_str = json_str.replace("'", '"')
                
                # Remove trailing commas in objects and arrays
                fixed_str = re.sub(r',\s*}', '}', fixed_str)
                fixed_str = re.sub(r',\s*]', ']', fixed_str)
                
                # Replace JavaScript undefined with null
                fixed_str = re.sub(r'undefined', 'null', fixed_str)
                
                # Try parsing again
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                # If still failing, try a more aggressive approach
                try:
                    # Remove JavaScript comments
                    fixed_str = re.sub(r'//.*?$|/\*.*?\*/', '', json_str, flags=re.MULTILINE | re.DOTALL)
                    
                    # Handle JavaScript style object keys without quotes
                    fixed_str = re.sub(r'([{,])\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1"\2":', fixed_str)
                    
                    # Replace single quotes with double quotes
                    fixed_str = fixed_str.replace("'", '"')
                    
                    # Remove trailing commas
                    fixed_str = re.sub(r',\s*}', '}', fixed_str)
                    fixed_str = re.sub(r',\s*]', ']', fixed_str)
                    
                    # Replace common JavaScript values
                    fixed_str = re.sub(r'undefined', 'null', fixed_str)
                    fixed_str = re.sub(r'true', 'true', fixed_str)
                    fixed_str = re.sub(r'false', 'false', fixed_str)
                    
                    return json.loads(fixed_str)
                except Exception:
                    # Give up if all approaches fail
                    return None
    
    def _ensure_soup(self, content: Union[str, BeautifulSoup, Tag]) -> BeautifulSoup:
        """
        Ensure we have a BeautifulSoup object to work with.
        
        Args:
            content: Content as string, BeautifulSoup, or Tag
            
        Returns:
            BeautifulSoup object
        """
        if isinstance(content, BeautifulSoup):
            return content
        
        if isinstance(content, Tag):
            return BeautifulSoup(str(content), "lxml")
        
        if isinstance(content, str):
            return BeautifulSoup(content, "lxml")
        
        raise ValueError(f"Unsupported content type: {type(content)}")