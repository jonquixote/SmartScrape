"""
Schema Extraction Module

This module provides schema-driven extraction capabilities for extracting structured data
from web pages using flexible schema definitions and multiple extraction strategies.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter

from bs4 import BeautifulSoup, Tag
import lxml.html
from lxml import etree
import jsonschema

from components.pattern_analyzer.base_analyzer import get_registry
from extraction.content_extraction import ContentExtractor
from utils.extraction_utils import clean_extracted_text, normalize_whitespace
from core.service_interface import BaseService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SchemaExtraction")

@dataclass
class ExtractionField:
    """Represents a field to be extracted with its properties and strategies."""
    name: str
    type: str = "string"  # string, number, boolean, object, array
    required: bool = False
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    default_value: Any = None
    format: Optional[str] = None  # date, time, email, etc.
    confidence_threshold: float = 0.5
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "strategies": self.strategies,
            "default_value": self.default_value,
            "format": self.format,
            "confidence_threshold": self.confidence_threshold,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionField':
        """Create field from dictionary."""
        return cls(
            name=data['name'],
            type=data.get('type', 'string'),
            required=data.get('required', False),
            strategies=data.get('strategies', []),
            default_value=data.get('default_value'),
            format=data.get('format'),
            confidence_threshold=data.get('confidence_threshold', 0.5),
            description=data.get('description')
        )

@dataclass
class ExtractionSchema:
    """Represents a complete extraction schema for structured data."""
    name: str
    version: str = "1.0"
    description: Optional[str] = None
    entity_type: str = "generic"  # generic, product, article, listing, etc.
    fields: List[ExtractionField] = field(default_factory=list)
    item_selector: Optional[List[str]] = None  # For extracting lists of items
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "entity_type": self.entity_type,
            "fields": [field.to_dict() for field in self.fields],
            "item_selector": self.item_selector
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionSchema':
        """Create schema from dictionary."""
        return cls(
            name=data['name'],
            version=data.get('version', '1.0'),
            description=data.get('description'),
            entity_type=data.get('entity_type', 'generic'),
            fields=[ExtractionField.from_dict(field) for field in data.get('fields', [])],
            item_selector=data.get('item_selector')
        )
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format for validation."""
        properties = {}
        required_fields = []
        
        for field in self.fields:
            field_schema = {"type": field.type}
            
            if field.description:
                field_schema["description"] = field.description
                
            if field.format:
                field_schema["format"] = field.format
                
            properties[field.name] = field_schema
            
            if field.required:
                required_fields.append(field.name)
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": self.name,
            "description": self.description or f"Schema for {self.entity_type} extraction",
            "properties": properties
        }
        
        if required_fields:
            schema["required"] = required_fields
            
        return schema

class SchemaExtractor(BaseService):
    """
    Extracts structured data from web pages using schema definitions.
    
    This class provides:
    - Schema-based extraction with multiple strategies
    - Integration with pattern analyzer for smart selector generation
    - Confidence scoring for extracted data
    - Validation and normalization of extracted data
    """
    
    def __init__(self, use_pattern_analyzer: bool = True):
        """
        Initialize the schema extractor.
        
        Args:
            use_pattern_analyzer: Whether to use pattern analyzer for selector discovery
        """
        self.use_pattern_analyzer = use_pattern_analyzer
        self.content_extractor = ContentExtractor()
        self._initialized = False
        
        # Initialize extraction strategies
        self.strategies = {
            "css": self._extract_with_css,
            "xpath": self._extract_with_xpath,
            "regex": self._extract_with_regex,
            "jsonld": self._extract_from_jsonld,
            "microdata": self._extract_from_microdata,
            "ai": self._extract_with_ai,
            "attribute": self._extract_from_attribute,
            "metadata": self._extract_from_metadata
        }
        
        logger.info("SchemaExtractor initialized with %d strategies", len(self.strategies))
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the service with the given configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        if config:
            # Update configuration if provided
            self.use_pattern_analyzer = config.get("use_pattern_analyzer", self.use_pattern_analyzer)
            
        self._initialized = True
        logger.info("SchemaExtractor service initialized")
        
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._initialized = False
        logger.info("SchemaExtractor service shut down")
        
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "schema_extractor"
        
    async def extract(self, 
                    html: str, 
                    url: str,
                    schema: Union[ExtractionSchema, Dict[str, Any]],
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data from HTML using the provided schema.
        
        Args:
            html: HTML content
            url: URL of the page
            schema: ExtractionSchema or dict representation
            context: Additional context for extraction (e.g., user_intent)
            
        Returns:
            Dictionary with extracted data and metadata
        """
        # Ensure we have a proper ExtractionSchema
        if isinstance(schema, dict):
            schema = ExtractionSchema.from_dict(schema)
            
        logger.info(f"Extracting data using schema: {schema.name}")
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'lxml')
            
            # Initialize results
            results = {
                "success": False,
                "schema_name": schema.name,
                "schema_version": schema.version,
                "url": url,
                "entity_type": schema.entity_type,
                "extraction_metadata": {
                    "confidence_scores": {},
                    "strategies_used": {},
                    "extraction_time": 0,
                },
                "data": {}
            }
            
            # Handle list extraction if item_selector is provided
            if schema.item_selector:
                items, item_metadata = await self._extract_items(soup, url, schema, context)
                results["data"]["items"] = items
                results["extraction_metadata"].update(item_metadata)
                results["success"] = True
                return results
            
            # Extract individual fields
            extracted_data = {}
            confidence_scores = {}
            strategies_used = {}
            
            for field in schema.fields:
                value, confidence, strategy = await self._extract_field(soup, url, field, context)
                
                # Store the extracted value
                extracted_data[field.name] = value
                confidence_scores[field.name] = confidence
                strategies_used[field.name] = strategy
                
                logger.debug(f"Extracted {field.name}: {value} (confidence: {confidence}, strategy: {strategy})")
            
            # Validate extracted data
            validation_result = self._validate_extracted_data(extracted_data, schema)
            
            # Store results
            results["data"] = extracted_data
            results["extraction_metadata"]["confidence_scores"] = confidence_scores
            results["extraction_metadata"]["strategies_used"] = strategies_used
            results["extraction_metadata"]["validation_result"] = validation_result
            results["success"] = validation_result.get("valid", False)
            
            # Calculate overall confidence
            confidence_values = list(confidence_scores.values())
            if confidence_values:
                results["extraction_metadata"]["overall_confidence"] = sum(confidence_values) / len(confidence_values)
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting data with schema: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "schema_name": schema.name,
                "url": url
            }
    
    async def _extract_items(self, 
                           soup: BeautifulSoup, 
                           url: str, 
                           schema: ExtractionSchema,
                           context: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract a list of items using the schema.
        
        Args:
            soup: BeautifulSoup object
            url: Page URL
            schema: Extraction schema
            context: Additional context
            
        Returns:
            Tuple of (list of extracted items, metadata)
        """
        items = []
        metadata = {
            "total_found": 0,
            "total_extracted": 0,
            "container_selector_used": None,
            "extraction_success_rate": 0.0
        }
        
        # Try each selector until one works
        container_elements = []
        selector_used = None
        
        for selector in schema.item_selector:
            try:
                # Try to extract with CSS selector
                elements = soup.select(selector)
                
                if elements and len(elements) > 0:
                    container_elements = elements
                    selector_used = selector
                    logger.info(f"Found {len(elements)} items with selector: {selector}")
                    break
            except Exception as e:
                logger.warning(f"Failed to use selector '{selector}': {str(e)}")
        
        # If no containers found with provided selectors, try using pattern analyzer
        if not container_elements and self.use_pattern_analyzer:
            registry = get_registry()
            
            # Check if we have a listing pattern for this domain
            domain = self._get_domain(url)
            listing_pattern = registry.get_pattern("listing", domain)
            
            if listing_pattern and "items_selector" in listing_pattern:
                try:
                    items_selector = listing_pattern["items_selector"]
                    container_elements = soup.select(items_selector)
                    selector_used = items_selector
                    logger.info(f"Using listing pattern from registry with selector: {items_selector}")
                except Exception as e:
                    logger.warning(f"Failed to use pattern analyzer selector: {str(e)}")
        
        # Process each container
        if container_elements:
            metadata["total_found"] = len(container_elements)
            metadata["container_selector_used"] = selector_used
            
            # Limit the number of items to process
            item_limit = min(20, len(container_elements))
            
            for container in container_elements[:item_limit]:
                item_data = {}
                item_confidence = {}
                item_strategies = {}
                
                # Extract each field from the container
                for field in schema.fields:
                    value, confidence, strategy = await self._extract_field(container, url, field, context)
                    
                    # Store the extracted value
                    item_data[field.name] = value
                    item_confidence[field.name] = confidence
                    item_strategies[field.name] = strategy
                
                # Skip items with insufficient data
                if self._is_item_valid(item_data, schema):
                    items.append({
                        "data": item_data,
                        "confidence_scores": item_confidence,
                        "strategies_used": item_strategies
                    })
            
            metadata["total_extracted"] = len(items)
            
            # Calculate success rate
            if metadata["total_found"] > 0:
                metadata["extraction_success_rate"] = metadata["total_extracted"] / metadata["total_found"]
        
        return items, metadata
    
    async def _extract_field(self, 
                           element: BeautifulSoup, 
                           url: str, 
                           field: ExtractionField,
                           context: Dict[str, Any] = None) -> Tuple[Any, float, str]:
        """
        Extract a field using available strategies.
        
        Args:
            element: BeautifulSoup element to extract from
            url: URL of the page
            field: Field definition
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence score, strategy used)
        """
        extracted_value = None
        confidence = 0.0
        strategy_used = "none"
        
        # Try each strategy in order
        for strategy_def in field.strategies:
            strategy_type = strategy_def["type"]
            strategy_config = strategy_def.get("config", {})
            
            # If strategy exists, use it
            if strategy_type in self.strategies:
                value, conf = await self.strategies[strategy_type](element, field.name, strategy_config, url, context)
                
                # If value extracted with sufficient confidence, use it
                if value is not None and conf > confidence:
                    extracted_value = value
                    confidence = conf
                    strategy_used = strategy_type
                    
                    # If confidence is high enough, stop trying more strategies
                    if confidence >= field.confidence_threshold:
                        break
            else:
                logger.warning(f"Unknown extraction strategy: {strategy_type}")
        
        # If no value extracted but we have a default, use it
        if extracted_value is None and field.default_value is not None:
            extracted_value = field.default_value
            confidence = 0.1  # Low confidence for default values
            strategy_used = "default"
        
        # Convert type if needed
        if extracted_value is not None:
            extracted_value = self._convert_type(extracted_value, field.type, field.format)
        
        return extracted_value, confidence, strategy_used
    
    def _convert_type(self, value: Any, type_name: str, format_name: Optional[str] = None) -> Any:
        """
        Convert value to the specified type.
        
        Args:
            value: Value to convert
            type_name: Type to convert to
            format_name: Optional format specification
            
        Returns:
            Converted value
        """
        if value is None:
            return None
            
        try:
            if type_name == "string":
                return str(value)
            elif type_name == "number":
                if isinstance(value, str):
                    # Remove currency symbols and commas
                    cleaned = re.sub(r'[$€£¥,]', '', value)
                    return float(cleaned)
                return float(value)
            elif type_name == "integer":
                if isinstance(value, str):
                    # Remove non-numeric characters
                    cleaned = re.sub(r'[^0-9]', '', value)
                    return int(cleaned)
                return int(value)
            elif type_name == "boolean":
                if isinstance(value, str):
                    lower_value = value.lower()
                    return lower_value in ("yes", "true", "t", "1", "y")
                return bool(value)
            elif type_name == "array":
                if isinstance(value, str):
                    # Split on commas if it's a string
                    return [item.strip() for item in value.split(",")]
                elif isinstance(value, list):
                    return value
                else:
                    return [value]
            else:
                return value
        except Exception as e:
            logger.warning(f"Error converting value '{value}' to {type_name}: {str(e)}")
            return value
    
    def _validate_extracted_data(self, 
                                data: Dict[str, Any], 
                                schema: ExtractionSchema) -> Dict[str, Any]:
        """
        Validate extracted data against the schema.
        
        Args:
            data: Extracted data
            schema: Extraction schema
            
        Returns:
            Validation result dictionary
        """
        # Convert schema to JSON Schema format
        json_schema = schema.to_json_schema()
        
        validation_result = {
            "valid": True,
            "missing_required": [],
            "type_errors": []
        }
        
        # Check for missing required fields
        for field in schema.fields:
            if field.required and (field.name not in data or data[field.name] is None):
                validation_result["valid"] = False
                validation_result["missing_required"].append(field.name)
        
        # Validate against JSON Schema
        try:
            jsonschema.validate(instance=data, schema=json_schema)
        except jsonschema.exceptions.ValidationError as e:
            validation_result["valid"] = False
            validation_result["type_errors"].append(str(e))
        
        return validation_result
    
    def _is_item_valid(self, item_data: Dict[str, Any], schema: ExtractionSchema) -> bool:
        """
        Check if an extracted item has sufficient valid data.
        
        Args:
            item_data: Extracted item data
            schema: Extraction schema
            
        Returns:
            Boolean indicating if item is valid
        """
        # Item must have at least one non-None value
        if not any(value is not None for value in item_data.values()):
            return False
            
        # Check required fields
        required_fields = [field.name for field in schema.fields if field.required]
        
        for field_name in required_fields:
            if field_name not in item_data or item_data[field_name] is None:
                return False
        
        return True
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        
        if not url:
            return ""
            
        parsed = urlparse(url)
        return parsed.netloc
    
    # Extraction strategy implementations
    
    async def _extract_with_css(self, 
                             element: BeautifulSoup, 
                             field_name: str, 
                             config: Dict[str, Any],
                             url: str = None,
                             context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data using CSS selectors.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        selectors = config.get("selectors", [])
        attribute = config.get("attribute", None)
        confidence_base = config.get("confidence_base", 0.7)
        
        # Selector must be provided
        if not selectors:
            return None, 0.0
        
        for selector in selectors:
            try:
                elements = element.select(selector)
                
                if elements:
                    target_element = elements[0]
                    
                    # Extract from specified attribute if provided
                    if attribute:
                        value = target_element.get(attribute)
                        if value:
                            # Adjust confidence based on match quality
                            confidence = confidence_base
                            
                            # If this is a URL and we have a base URL, resolve it
                            if url and attribute in ("href", "src") and value.startswith('/'):
                                from urllib.parse import urljoin
                                value = urljoin(url, value)
                            
                            return value, confidence
                    else:
                        # Default to text content
                        value = clean_extracted_text(target_element.get_text(strip=True))
                        
                        if value:
                            # Adjust confidence based on text length and quality
                            confidence = confidence_base
                            
                            # Adjust confidence based on element type and field name
                            element_type = target_element.name
                            
                            # Higher confidence for certain element types based on field
                            if (field_name == "title" and element_type in ["h1", "h2", "h3"]) or \
                               (field_name == "price" and re.search(r'[$€£¥]\s*\d+', value)) or \
                               (field_name == "description" and element_type in ["p", "div"] and len(value) > 20):
                                confidence += 0.1
                            
                            return value, confidence
            except Exception as e:
                logger.debug(f"CSS extraction error with selector '{selector}': {str(e)}")
        
        return None, 0.0
    
    async def _extract_with_xpath(self, 
                               element: BeautifulSoup, 
                               field_name: str, 
                               config: Dict[str, Any],
                               url: str = None,
                               context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data using XPath.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        xpaths = config.get("xpaths", [])
        attribute = config.get("attribute", None)
        confidence_base = config.get("confidence_base", 0.7)
        
        # XPath must be provided
        if not xpaths:
            return None, 0.0
        
        # Convert BeautifulSoup element to lxml for XPath support
        if isinstance(element, BeautifulSoup):
            html_str = str(element)
            lxml_element = lxml.html.fromstring(html_str)
        elif isinstance(element, Tag):
            html_str = str(element)
            lxml_element = lxml.html.fromstring(html_str)
        else:
            lxml_element = element
        
        for xpath in xpaths:
            try:
                results = lxml_element.xpath(xpath)
                
                if results:
                    result = results[0]
                    
                    # Handle different result types
                    if isinstance(result, etree._Element):
                        if attribute:
                            value = result.get(attribute)
                        else:
                            value = result.text_content().strip()
                    elif isinstance(result, str):
                        value = result
                    else:
                        value = str(result)
                    
                    if value:
                        # Clean and normalize
                        value = clean_extracted_text(value)
                        
                        # Calculate confidence
                        confidence = confidence_base
                        
                        return value, confidence
            except Exception as e:
                logger.debug(f"XPath extraction error with '{xpath}': {str(e)}")
        
        return None, 0.0
    
    async def _extract_with_regex(self, 
                               element: BeautifulSoup, 
                               field_name: str, 
                               config: Dict[str, Any],
                               url: str = None,
                               context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data using regular expressions.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        patterns = config.get("patterns", [])
        search_attribute = config.get("attribute", None)
        group = config.get("group", 0)
        confidence_base = config.get("confidence_base", 0.6)
        
        # Patterns must be provided
        if not patterns:
            return None, 0.0
        
        # Get text to search
        if search_attribute and isinstance(element, (BeautifulSoup, Tag)):
            text = element.get(search_attribute, "")
        else:
            text = element.get_text(strip=True) if isinstance(element, (BeautifulSoup, Tag)) else str(element)
        
        # Try each pattern
        for pattern in patterns:
            try:
                match = re.search(pattern, text)
                
                if match:
                    # Extract the specified group
                    try:
                        value = match.group(group)
                    except IndexError:
                        value = match.group(0)
                    
                    # Clean and normalize
                    value = clean_extracted_text(value)
                    
                    if value:
                        # Calculate confidence based on how specific the match is
                        confidence = confidence_base
                        
                        # Adjust confidence based on match coverage
                        match_length = len(match.group(0))
                        text_length = len(text)
                        
                        if text_length > 0:
                            coverage = match_length / text_length
                            
                            # If match is very specific (small part of text), increase confidence
                            if coverage < 0.1:
                                confidence += 0.1
                            # If match is almost the entire text, slightly lower confidence
                            elif coverage > 0.9:
                                confidence -= 0.1
                        
                        return value, confidence
            except Exception as e:
                logger.debug(f"Regex extraction error with pattern '{pattern}': {str(e)}")
        
        return None, 0.0
    
    async def _extract_from_jsonld(self, 
                                element: BeautifulSoup, 
                                field_name: str, 
                                config: Dict[str, Any],
                                url: str = None,
                                context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data from JSON-LD structured data.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        jsonld_fields = config.get("jsonld_fields", [])
        types = config.get("types", [])
        confidence_base = config.get("confidence_base", 0.9)  # High base confidence for structured data
        
        # Structured data fields must be provided
        if not jsonld_fields:
            return None, 0.0
        
        # Find all JSON-LD script elements
        soup = element if isinstance(element, BeautifulSoup) else BeautifulSoup(str(element), 'lxml')
        jsonld_scripts = soup.find_all("script", type="application/ld+json")
        
        for script in jsonld_scripts:
            try:
                json_content = json.loads(script.string)
                
                # Convert to list if it's a single object
                if not isinstance(json_content, list):
                    json_content = [json_content]
                
                # Check each JSON-LD object
                for item in json_content:
                    # If types are specified, check if this object has the right type
                    item_type = item.get("@type")
                    if types and item_type:
                        if not isinstance(item_type, list):
                            item_type = [item_type]
                        
                        if not any(t in item_type for t in types):
                            continue
                    
                    # Try each possible field path
                    for field_path in jsonld_fields:
                        value = self._get_nested_value(item, field_path)
                        
                        if value is not None:
                            # Handle array values
                            if isinstance(value, list):
                                if len(value) > 0:
                                    # Use the first non-null value
                                    for v in value:
                                        if v is not None:
                                            value = v
                                            break
                                else:
                                    value = None
                            
                            # Clean and normalize if it's a string
                            if isinstance(value, str):
                                value = clean_extracted_text(value)
                            
                            if value is not None:
                                # High confidence for JSON-LD data
                                confidence = confidence_base
                                
                                return value, confidence
            except Exception as e:
                logger.debug(f"JSON-LD extraction error: {str(e)}")
        
        return None, 0.0
    
    async def _extract_from_microdata(self, 
                                    element: BeautifulSoup, 
                                    field_name: str, 
                                    config: Dict[str, Any],
                                    url: str = None,
                                    context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data from HTML microdata.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        properties = config.get("properties", [])
        types = config.get("types", [])
        confidence_base = config.get("confidence_base", 0.85)
        
        # Properties must be specified
        if not properties:
            return None, 0.0
        
        # Parse element
        soup = element if isinstance(element, BeautifulSoup) else BeautifulSoup(str(element), 'lxml')
        
        # Find elements with itemscope
        itemscope_elements = soup.find_all(itemscope=True)
        
        for scope_element in itemscope_elements:
            # Check if element has the right type
            if types:
                itemtype = scope_element.get("itemtype", "")
                if not any(itemtype.endswith(t) for t in types):
                    continue
            
            # Look for elements with specified properties
            for prop in properties:
                prop_elements = scope_element.find_all(itemprop=prop)
                
                if prop_elements:
                    prop_element = prop_elements[0]
                    
                    # Extract value based on tag type
                    tag_name = prop_element.name
                    
                    if tag_name in ["meta"]:
                        value = prop_element.get("content", "")
                    elif tag_name in ["img"]:
                        value = prop_element.get("src", "")
                    elif tag_name in ["a"]:
                        value = prop_element.get("href", "")
                    elif tag_name in ["time"]:
                        value = prop_element.get("datetime", "")
                    else:
                        value = prop_element.get_text(strip=True)
                    
                    # Clean and normalize
                    value = clean_extracted_text(value)
                    
                    if value:
                        confidence = confidence_base
                        return value, confidence
        
        return None, 0.0
    
    async def _extract_with_ai(self, 
                            element: BeautifulSoup, 
                            field_name: str, 
                            config: Dict[str, Any],
                            url: str = None,
                            context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data using AI assistance.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        instructions = config.get("instructions", f"Extract the {field_name} from the HTML content.")
        confidence_base = config.get("confidence_base", 0.6)  # Lower baseline for AI
        
        try:
            # Convert element to HTML string
            html_content = str(element)
            
            # Get user intent from context
            user_intent = context.get("user_intent", {}) if context else {}
            
            # Create context information for AI extraction
            ai_context = {
                "field_name": field_name,
                "field_description": config.get("description", ""),
                "intent": user_intent,
                "url": url
            }
            
            # Import here to avoid circular imports
            from extraction.content_extraction import extract_content_with_ai
            
            # Prepare the desired properties for AI extraction
            desired_properties = [field_name]
            
            # Call AI extraction function
            results = await extract_content_with_ai(
                html_content=html_content,
                url=url,
                user_intent=user_intent,
                desired_properties=desired_properties
            )
            
            # Check if we got any results
            if results and isinstance(results, list) and len(results) > 0:
                # Find the result with our field
                for result in results:
                    if field_name in result:
                        value = result[field_name]
                        if value is not None:
                            confidence = confidence_base
                            return value, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.warning(f"AI extraction error for {field_name}: {str(e)}")
            return None, 0.0
    
    async def _extract_from_attribute(self, 
                                    element: BeautifulSoup, 
                                    field_name: str, 
                                    config: Dict[str, Any],
                                    url: str = None,
                                    context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data from element attributes.
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        selectors = config.get("selectors", [])
        attributes = config.get("attributes", [])
        confidence_base = config.get("confidence_base", 0.75)
        
        # Need both selectors and attributes
        if not selectors or not attributes:
            return None, 0.0
        
        # Parse element if needed
        soup = element if isinstance(element, BeautifulSoup) else BeautifulSoup(str(element), 'lxml')
        
        # Try each selector
        for selector in selectors:
            try:
                elements = soup.select(selector)
                
                if elements:
                    target_element = elements[0]
                    
                    # Try each attribute
                    for attr in attributes:
                        value = target_element.get(attr)
                        
                        if value:
                            # Handle URL attributes that might be relative
                            if url and attr in ["src", "href", "data-src"] and value.startswith('/'):
                                from urllib.parse import urljoin
                                value = urljoin(url, value)
                            
                            # Clean and normalize if it's a string
                            value = clean_extracted_text(value)
                            
                            confidence = confidence_base
                            return value, confidence
            except Exception as e:
                logger.debug(f"Attribute extraction error with selector '{selector}': {str(e)}")
        
        return None, 0.0
    
    async def _extract_from_metadata(self, 
                                  element: BeautifulSoup, 
                                  field_name: str, 
                                  config: Dict[str, Any],
                                  url: str = None,
                                  context: Dict[str, Any] = None) -> Tuple[Any, float]:
        """
        Extract data from HTML metadata (meta tags, Open Graph, etc.).
        
        Args:
            element: BeautifulSoup element
            field_name: Name of the field to extract
            config: Strategy configuration
            url: URL of the page
            context: Additional context
            
        Returns:
            Tuple of (extracted value, confidence)
        """
        meta_names = config.get("meta_names", [])
        meta_properties = config.get("meta_properties", [])
        confidence_base = config.get("confidence_base", 0.8)
        
        # Need at least one type of meta selectors
        if not meta_names and not meta_properties:
            return None, 0.0
        
        # Parse element if needed
        soup = element if isinstance(element, BeautifulSoup) else BeautifulSoup(str(element), 'lxml')
        
        # Try each meta name
        for name in meta_names:
            meta_tag = soup.find("meta", attrs={"name": name})
            if meta_tag and meta_tag.get("content"):
                value = meta_tag.get("content")
                value = clean_extracted_text(value)
                
                if value:
                    confidence = confidence_base
                    return value, confidence
        
        # Try each meta property (OpenGraph, etc.)
        for prop in meta_properties:
            meta_tag = soup.find("meta", attrs={"property": prop})
            if meta_tag and meta_tag.get("content"):
                value = meta_tag.get("content")
                value = clean_extracted_text(value)
                
                if value:
                    confidence = confidence_base
                    return value, confidence
        
        return None, 0.0
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get a nested value from a dictionary using a dot-notation path.
        
        Args:
            data: Dictionary to extract from
            path: Dot-notation path (e.g., "offers.price")
            
        Returns:
            Extracted value or None if not found
        """
        parts = path.split('.')
        current = data
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        
        return current

def create_schema_from_intent(user_intent: Dict[str, Any]) -> ExtractionSchema:
    """
    Create an extraction schema based on user intent.
    
    Args:
        user_intent: User intent dictionary
        
    Returns:
        ExtractionSchema object
    """
    try:
        schema_name = f"intent_generated_{user_intent.get('target_item', 'general')}"
        
        # Determine entity type
        entity_type = "product"
        if "entity_types" in user_intent and user_intent["entity_types"]:
            entity_type = user_intent["entity_types"][0]
        
        # Initialize schema
        schema = ExtractionSchema(
            name=schema_name,
            entity_type=entity_type,
            description=f"Schema generated from user intent: {user_intent.get('description', '')}",
            fields=[]
        )
        
        # Add fields based on requested properties
        if "properties" in user_intent and user_intent["properties"]:
            for prop in user_intent["properties"]:
                field = ExtractionField(
                    name=prop,
                    required=True,
                    strategies=[
                        # CSS strategy
                        {
                            "type": "css",
                            "config": {
                                "selectors": [
                                    f".{prop}", 
                                    f"#{prop}",
                                    f"[data-{prop}]",
                                    f"[itemprop='{prop}']",
                                    f".{prop.replace('_', '-')}"
                                ]
                            }
                        },
                        # Regex strategy for common patterns
                        {
                            "type": "regex",
                            "config": {
                                "patterns": [
                                    # Custom patterns based on property name
                                    f"{prop.replace('_', ' ')}:?\s*([^\\n]+)"
                                ]
                            }
                        },
                        # JSON-LD strategy
                        {
                            "type": "jsonld",
                            "config": {
                                "jsonld_fields": [prop, prop.replace("_", "")]
                            }
                        },
                        # Attribute strategy
                        {
                            "type": "attribute",
                            "config": {
                                "selectors": [f"[data-{prop}]", f"[data-{prop.replace('_', '-')}]"],
                                "attributes": [f"data-{prop}", f"data-{prop.replace('_', '-')}", "content"]
                            }
                        },
                        # Metadata strategy
                        {
                            "type": "metadata",
                            "config": {
                                "meta_names": [prop, prop.replace("_", "-")],
                                "meta_properties": [
                                    f"og:{prop}", 
                                    f"og:{prop.replace('_', '-')}", 
                                    f"product:{prop}", 
                                    f"product:{prop.replace('_', '-')}"
                                ]
                            }
                        },
                        # AI fallback strategy
                        {
                            "type": "ai",
                            "config": {
                                "instructions": f"Extract the {prop.replace('_', ' ')} from the HTML content.",
                                "confidence_base": 0.6
                            }
                        }
                    ]
                )
                
                # Add special strategies for common properties
                if prop == "price" or prop.endswith("_price"):
                    field.strategies.insert(0, {
                        "type": "regex",
                        "config": {
                            "patterns": [
                                r'(\$|€|£|¥)\s*[\d,]+(\.\d{1,2})?',
                                r'[\d,]+(\.\d{1,2})?\s*(\$|€|£|¥)'
                            ]
                        }
                    })
                    field.type = "number"
                
                elif prop == "image" or prop.endswith("_image"):
                    field.strategies.insert(0, {
                        "type": "attribute",
                        "config": {
                            "selectors": ["img", "img.main", "img.primary", ".product-image img"],
                            "attributes": ["src", "data-src", "data-original"]
                        }
                    })
                
                # Add the field to the schema
                schema.fields.append(field)
        
        # Add basic fields if none are specified
        if not schema.fields:
            schema.fields = [
                ExtractionField(
                    name="title",
                    required=True,
                    strategies=[
                        {"type": "css", "config": {"selectors": ["h1", "h2.title", ".title", "header h1"]}},
                        {"type": "metadata", "config": {"meta_properties": ["og:title"], "meta_names": ["title"]}}
                    ]
                ),
                ExtractionField(
                    name="description",
                    strategies=[
                        {"type": "css", "config": {"selectors": ["p.description", ".description", "meta[name='description']"]}},
                        {"type": "metadata", "config": {"meta_properties": ["og:description"], "meta_names": ["description"]}}
                    ]
                )
            ]
            
            # Add common fields based on entity type
            if entity_type == "product":
                schema.fields.extend([
                    ExtractionField(
                        name="price",
                        type="number",
                        strategies=[
                            {"type": "css", "config": {"selectors": [".price", "span.price", "[itemprop='price']"]}},
                            {"type": "regex", "config": {"patterns": [r'(\$|€|£|¥)\s*[\d,]+(\.\d{1,2})?']}}
                        ]
                    ),
                    ExtractionField(
                        name="image",
                        strategies=[
                            {"type": "attribute", "config": {"selectors": ["img.product-image", ".product-img img"], "attributes": ["src", "data-src"]}}
                        ]
                    )
                ])
        
        # Set up item selectors if we're looking for a list
        if entity_type in ["product", "listing", "search_result"]:
            schema.item_selector = [
                ".products > .product",
                ".product-grid > .product",
                ".items > .item",
                "ul.products > li",
                ".search-results > .result",
                ".listings > .listing"
            ]
        
        return schema
    
    except Exception as e:
        logger.error(f"Error creating schema from intent: {str(e)}")
        # Return a minimal fallback schema
        return ExtractionSchema(
            name="fallback_schema",
            entity_type="generic",
            fields=[
                ExtractionField(name="title", required=True, strategies=[
                    {"type": "css", "config": {"selectors": ["h1", "h2", ".title", "header h1"]}}
                ]),
                ExtractionField(name="content", strategies=[
                    {"type": "css", "config": {"selectors": ["p", "article", "main", "#content"]}}
                ])
            ]
        )

async def extract_with_schema(
    html_content: str,
    url: str,
    user_intent: Dict[str, Any] = None,
    custom_schema: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    High-level function to extract data using schema.
    
    Args:
        html_content: HTML content to extract from
        url: URL of the page
        user_intent: User intent dictionary for dynamic schema generation
        custom_schema: Optional custom schema to use
        
    Returns:
        Extraction results
    """
    try:
        extractor = SchemaExtractor()
        
        # Determine which schema to use
        if custom_schema:
            schema = ExtractionSchema.from_dict(custom_schema)
        elif user_intent:
            schema = create_schema_from_intent(user_intent)
        else:
            # Default generic schema
            schema = ExtractionSchema(
                name="generic_schema",
                entity_type="generic",
                fields=[
                    ExtractionField(name="title", required=True, strategies=[
                        {"type": "css", "config": {"selectors": ["h1", "h2", ".title", "header h1"]}}
                    ]),
                    ExtractionField(name="content", strategies=[
                        {"type": "css", "config": {"selectors": ["p", "article", "main", "#content"]}}
                    ])
                ]
            )
        
        # Extract data using the schema
        context = {"user_intent": user_intent} if user_intent else {}
        results = await extractor.extract(html_content, url, schema, context)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in extract_with_schema: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url
        }