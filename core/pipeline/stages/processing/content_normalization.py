"""
Content Normalization Pipeline Stages Module.

This module provides pipeline stages for normalizing and validating different types of content.
"""

import re
import logging
import html
import json
import datetime
import unicodedata
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple, Union, Callable
from collections import defaultdict

from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext


class DataNormalizationStage(PipelineStage):
    """
    Pipeline stage for standardizing data formats.
    
    This stage normalizes various data types into standardized formats, with support for:
    - Date/time parsing and standardization
    - Price/currency normalization
    - Unit conversion
    - Text cleaning and normalization
    - Structured data normalization
    
    Configuration options:
        input_key (str): Key in the context to get the data from
        output_key (str): Key in the context to store the normalized data
        normalization_rules (Dict): Rules for normalizing different data types
        schema_mappings (Dict): Optional field name mappings for standardization
        preserve_original (bool): Whether to preserve original values (default: False)
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a data normalization stage.
        
        Args:
            name: Name of the stage
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Extract configuration options with defaults
        self.input_key = self.config.get("input_key", "extracted_data")
        self.output_key = self.config.get("output_key", "normalized_data")
        self.normalization_rules = self.config.get("normalization_rules", {})
        self.schema_mappings = self.config.get("schema_mappings", {})
        self.preserve_original = self.config.get("preserve_original", False)
        
        # Default date formats to try for parsing
        self.date_formats = self.config.get("date_formats", [
            # ISO format
            "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
            # US format
            "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y",
            # European format
            "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y",
            # With time
            "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
            # Text format
            "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"
        ])
        
        # Field name standardization mappings
        self.field_name_variants = self.config.get("field_name_variants", {
            "price": ["price", "cost", "amount", "value", "fee", "charge", "rate"],
            "date": ["date", "datetime", "time", "published", "created", "posted", "updated", "modified"],
            "title": ["title", "name", "heading", "subject", "label"],
            "description": ["description", "desc", "summary", "overview", "details", "about", "info", "text", "content"],
            "image": ["image", "img", "photo", "picture", "thumbnail", "cover", "banner", "pic", "logo"],
            "url": ["url", "link", "href", "uri", "webpage", "web", "source", "address"],
            "id": ["id", "identifier", "uid", "uuid", "key", "code"],
            "author": ["author", "writer", "creator", "poster", "by", "contributor"],
            "category": ["category", "type", "group", "section", "class", "topic"],
            "location": ["location", "address", "place", "position", "area", "region", "locality"],
            "rating": ["rating", "score", "rank", "stars", "grade"],
            "status": ["status", "state", "condition", "availability"],
            "quantity": ["quantity", "qty", "amount", "count", "number", "total"],
            "brand": ["brand", "make", "manufacturer", "company", "provider", "vendor"],
            "features": ["features", "specs", "specifications", "attributes", "characteristics", "details"]
        })
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains the required data.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.get(self.input_key):
            self.logger.error(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input '{self.input_key}'")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Normalize data in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get the data from the context
            data = context.get(self.input_key)
            
            if not data:
                self.logger.warning(f"Empty data in '{self.input_key}'")
                return False
                
            # Determine the data type and apply appropriate normalization
            if isinstance(data, dict):
                normalized_data = self._normalize_dict(data)
            elif isinstance(data, list):
                normalized_data = self._normalize_list(data)
            else:
                self.logger.warning(f"Unsupported data type: {type(data)}")
                return False
                
            # Store the normalized data in the context
            context.set(self.output_key, normalized_data)
            
            # Store metadata about the normalization
            normalization_metadata = {
                "original_type": type(data).__name__,
                "normalization_stage": self.name,
                "fields_normalized": self._count_normalized_fields(normalized_data),
                "schema_mapping_applied": bool(self.schema_mappings)
            }
            context.set(f"{self.output_key}_metadata", normalization_metadata)
            
            self.logger.info(f"Data normalized successfully: {normalization_metadata['fields_normalized']} fields")
            return True
            
        except Exception as e:
            self.logger.error(f"Error normalizing data: {str(e)}")
            context.add_error(self.name, f"Data normalization failed: {str(e)}")
            return await self.handle_error(context, e)
            
    def _normalize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a dictionary of data.
        
        Args:
            data: Dictionary to normalize
            
        Returns:
            Normalized dictionary
        """
        result = {}
        
        for field_name, value in data.items():
            # Skip metadata fields (they'll be handled separately)
            if field_name.startswith('_'):
                result[field_name] = value
                continue
                
            # Get normalized field name based on mappings
            normalized_field_name = self._normalize_field_name(field_name)
            
            # Save original value if configured
            if self.preserve_original:
                result[f"_original_{normalized_field_name}"] = value
                
            # Determine field type and normalize value
            field_type = self._infer_field_type(normalized_field_name, value)
            normalized_value = self._normalize_field_value(value, field_type)
            
            # Store the normalized value
            result[normalized_field_name] = normalized_value
            
        return result
    
    def _normalize_list(self, data: List[Any]) -> List[Any]:
        """
        Normalize a list of data.
        
        Args:
            data: List to normalize
            
        Returns:
            Normalized list
        """
        normalized_list = []
        
        for item in data:
            if isinstance(item, dict):
                normalized_item = self._normalize_dict(item)
            elif isinstance(item, list):
                normalized_item = self._normalize_list(item)
            else:
                # Try to normalize primitive values
                field_type = self._infer_field_type("", item)
                normalized_item = self._normalize_field_value(item, field_type)
                
            normalized_list.append(normalized_item)
            
        return normalized_list
    
    def _normalize_field_name(self, field_name: str) -> str:
        """
        Normalize a field name to a standard form.
        
        Args:
            field_name: Original field name
            
        Returns:
            Normalized field name
        """
        # Check if we have a direct mapping
        if self.schema_mappings and field_name in self.schema_mappings:
            return self.schema_mappings[field_name]
            
        # Check if field matches any of our known variants
        field_lower = field_name.lower().replace('_', '').replace('-', '')
        
        for standard_name, variants in self.field_name_variants.items():
            for variant in variants:
                variant_clean = variant.lower().replace('_', '').replace('-', '')
                if field_lower == variant_clean or field_lower.endswith(variant_clean):
                    return standard_name
                    
        # Handle common prefixes
        for prefix in ["item", "product", "article", "post", "listing"]:
            if field_lower.startswith(prefix) and len(field_lower) > len(prefix):
                # Try to match the part after the prefix
                remainder = field_lower[len(prefix):]
                for standard_name, variants in self.field_name_variants.items():
                    for variant in variants:
                        variant_clean = variant.lower().replace('_', '').replace('-', '')
                        if remainder == variant_clean:
                            return standard_name
        
        # If no match, keep the original but convert to snake_case
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', field_name).lower()
        return snake_case
    
    def _infer_field_type(self, field_name: str, value: Any) -> str:
        """
        Infer the type of a field based on name and value.
        
        Args:
            field_name: Field name
            value: Field value
            
        Returns:
            Inferred field type as string
        """
        # First check if there's a specific rule for this field
        if field_name in self.normalization_rules:
            return self.normalization_rules[field_name].get("type", "unknown")
            
        field_lower = field_name.lower()
        
        # Check by name first
        for type_name, variants in self.field_name_variants.items():
            if any(variant.lower() in field_lower for variant in variants):
                return type_name
                
        # Check by value type
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            if any(price_term in field_lower for price_term in ["price", "cost", "fee"]):
                return "price"
            else:
                return "number"
        elif isinstance(value, str):
            # Check if it's a date
            if self._looks_like_date(value):
                return "date"
                
            # Check if it's a price
            if self._looks_like_price(value):
                return "price"
                
            # Check if it's a URL
            if self._looks_like_url(value):
                return "url"
                
            # Default to text
            return "text"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
    
    def _normalize_field_value(self, value: Any, field_type: str) -> Any:
        """
        Normalize a field value based on its type.
        
        Args:
            value: Field value to normalize
            field_type: Type of the field
            
        Returns:
            Normalized value
        """
        if value is None:
            return None
            
        # Check for empty string values
        if isinstance(value, str) and value.strip() == "":
            return None
            
        # Apply normalization based on field type
        if field_type == "date":
            return self._normalize_date(value)
        elif field_type == "price":
            return self._normalize_price(value)
        elif field_type == "number":
            return self._normalize_number(value)
        elif field_type == "boolean":
            return self._normalize_boolean(value)
        elif field_type == "text":
            return self._normalize_text(value)
        elif field_type == "url":
            return self._normalize_url(value)
        elif field_type == "list":
            if isinstance(value, list):
                return [self._normalize_field_value(item, self._infer_field_type("", item)) 
                        for item in value]
            else:
                # Convert to list if not already
                return self._normalize_list_value(value)
        elif field_type == "object":
            if isinstance(value, dict):
                return self._normalize_dict(value)
            else:
                return value
        else:
            # Unknown type, return as is
            return value
    
    def _normalize_date(self, date_value: Any) -> Dict[str, Any]:
        """
        Normalize a date value to a standard format.
        
        Args:
            date_value: Date value to normalize
            
        Returns:
            Dictionary with normalized date information
        """
        if date_value is None:
            return None
            
        date_dict = {
            "iso": None,
            "timestamp": None,
            "formatted": None
        }
        
        if isinstance(date_value, str):
            # Try parsing with different formats
            parsed_date = None
            
            for date_format in self.date_formats:
                try:
                    parsed_date = datetime.datetime.strptime(date_value, date_format)
                    break
                except ValueError:
                    continue
                    
            # If parsing failed, try more complex patterns
            if parsed_date is None:
                # Try to handle "X days/hours/minutes ago"
                relative_match = re.search(r'(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago', date_value, re.IGNORECASE)
                if relative_match:
                    num = int(relative_match.group(1))
                    unit = relative_match.group(2).lower()
                    now = datetime.datetime.now()
                    
                    if unit == 'second':
                        parsed_date = now - datetime.timedelta(seconds=num)
                    elif unit == 'minute':
                        parsed_date = now - datetime.timedelta(minutes=num)
                    elif unit == 'hour':
                        parsed_date = now - datetime.timedelta(hours=num)
                    elif unit == 'day':
                        parsed_date = now - datetime.timedelta(days=num)
                    elif unit == 'week':
                        parsed_date = now - datetime.timedelta(weeks=num)
                    elif unit == 'month':
                        # Approximate a month as 30 days
                        parsed_date = now - datetime.timedelta(days=num*30)
                    elif unit == 'year':
                        # Approximate a year as 365 days
                        parsed_date = now - datetime.timedelta(days=num*365)
            
            if parsed_date:
                date_dict["iso"] = parsed_date.isoformat()
                date_dict["timestamp"] = int(parsed_date.timestamp())
                date_dict["formatted"] = parsed_date.strftime("%B %d, %Y")
                
        elif isinstance(date_value, (int, float)):
            # Assume it's a timestamp
            try:
                parsed_date = datetime.datetime.fromtimestamp(date_value)
                date_dict["iso"] = parsed_date.isoformat()
                date_dict["timestamp"] = int(date_value)
                date_dict["formatted"] = parsed_date.strftime("%B %d, %Y")
            except (ValueError, OSError, OverflowError):
                # Invalid timestamp
                pass
                
        elif isinstance(date_value, dict):
            # Already in a dictionary format
            if "iso" in date_value:
                date_dict["iso"] = date_value["iso"]
            if "timestamp" in date_value:
                date_dict["timestamp"] = date_value["timestamp"]
            if "formatted" in date_value:
                date_dict["formatted"] = date_value["formatted"]
                
        return date_dict
    
    def _normalize_price(self, price_value: Any) -> Dict[str, Any]:
        """
        Normalize a price value to a standard format.
        
        Args:
            price_value: Price value to normalize
            
        Returns:
            Dictionary with normalized price information
        """
        if price_value is None:
            return None
            
        price_dict = {
            "amount": None,
            "currency": None,
            "formatted": None
        }
        
        # Handle different input types
        if isinstance(price_value, (int, float)):
            price_dict["amount"] = float(price_value)
            price_dict["formatted"] = f"{price_value:.2f}"
            
        elif isinstance(price_value, str):
            # Extract currency symbol if present
            currency_match = re.search(r'([€$£¥₹]|USD|EUR|GBP|JPY|INR|CAD|AUD)', price_value)
            if currency_match:
                symbol = currency_match.group(1)
                # Map symbol to currency code
                currency_map = {
                    '$': 'USD',
                    '€': 'EUR',
                    '£': 'GBP',
                    '¥': 'JPY',
                    '₹': 'INR',
                    'USD': 'USD',
                    'EUR': 'EUR',
                    'GBP': 'GBP',
                    'JPY': 'JPY',
                    'INR': 'INR',
                    'CAD': 'CAD',
                    'AUD': 'AUD'
                }
                price_dict["currency"] = currency_map.get(symbol, symbol)
            
            # Extract numeric value
            numeric_str = re.sub(r'[^\d.,]', '', price_value)
            
            # Handle different decimal/thousands separators
            if ',' in numeric_str and '.' in numeric_str:
                # Determine which is the decimal separator based on position
                if numeric_str.rindex('.') > numeric_str.rindex(','):
                    # Point is decimal separator
                    numeric_str = numeric_str.replace(',', '')
                else:
                    # Comma is decimal separator
                    numeric_str = numeric_str.replace('.', '').replace(',', '.')
            elif ',' in numeric_str:
                # If only commas, it depends on the pattern
                if re.search(r'\d,\d{3}(?!\d)', numeric_str):
                    # Comma is thousands separator (e.g., 1,234)
                    numeric_str = numeric_str.replace(',', '')
                else:
                    # Comma is decimal separator (e.g., 1,23)
                    numeric_str = numeric_str.replace(',', '.')
            
            try:
                price_dict["amount"] = float(numeric_str)
                price_dict["formatted"] = price_value
            except (ValueError, TypeError):
                # Leave as None if we can't parse a number
                pass
                
        elif isinstance(price_value, dict):
            # Already in a dictionary format, extract relevant fields
            if "amount" in price_value:
                price_dict["amount"] = float(price_value["amount"]) if price_value["amount"] is not None else None
            if "currency" in price_value:
                price_dict["currency"] = price_value["currency"]
            if "formatted" in price_value:
                price_dict["formatted"] = price_value["formatted"]
                
        # If we have amount but no currency, default to USD
        if price_dict["amount"] is not None and price_dict["currency"] is None:
            price_dict["currency"] = "USD"
            
        # If we have amount but no formatted version
        if price_dict["amount"] is not None and price_dict["formatted"] is None:
            currency_symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "INR": "₹"}
            symbol = currency_symbols.get(price_dict["currency"], price_dict["currency"])
            price_dict["formatted"] = f"{symbol}{price_dict['amount']:.2f}"
            
        return price_dict
    
    def _normalize_number(self, number_value: Any) -> float:
        """
        Normalize a numeric value.
        
        Args:
            number_value: Number-like value to normalize
            
        Returns:
            Normalized number as float
        """
        if number_value is None:
            return None
            
        if isinstance(number_value, (int, float)):
            return float(number_value)
            
        if isinstance(number_value, str):
            # Remove non-numeric characters (except decimal point)
            numeric_str = re.sub(r'[^\d.-]', '', number_value)
            try:
                return float(numeric_str)
            except (ValueError, TypeError):
                return None
                
        return None
    
    def _normalize_boolean(self, bool_value: Any) -> bool:
        """
        Normalize a boolean value.
        
        Args:
            bool_value: Boolean-like value to normalize
            
        Returns:
            Normalized boolean
        """
        if isinstance(bool_value, bool):
            return bool_value
            
        if isinstance(bool_value, (int, float)):
            return bool_value != 0
            
        if isinstance(bool_value, str):
            bool_value = bool_value.lower().strip()
            return bool_value in ['true', 'yes', 'y', '1', 'on', 'available', 'in stock']
            
        # None or other types
        return False
    
    def _normalize_text(self, text_value: Any) -> str:
        """
        Normalize a text value.
        
        Args:
            text_value: Text value to normalize
            
        Returns:
            Normalized text string
        """
        if text_value is None:
            return None
            
        if not isinstance(text_value, str):
            text_value = str(text_value)
            
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', text_value).strip()
        
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Convert common HTML entities
        cleaned = html.unescape(cleaned)
        
        # Normalize Unicode characters to their canonical form if configured
        if self.config.get("normalize_unicode", True):
            cleaned = unicodedata.normalize('NFKC', cleaned)
            
        return cleaned
    
    def _normalize_url(self, url_value: Any) -> str:
        """
        Normalize a URL.
        
        Args:
            url_value: URL to normalize
            
        Returns:
            Normalized URL string
        """
        if url_value is None:
            return None
            
        if not isinstance(url_value, str):
            url_value = str(url_value)
            
        url = url_value.strip()
        
        # Fix protocol-relative URLs
        if url.startswith('//'):
            url = 'https:' + url
            
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            if self.config.get("default_to_https", True):
                url = 'https://' + url
            else:
                url = 'http://' + url
                
        return url
    
    def _normalize_list_value(self, list_value: Any) -> List[Any]:
        """
        Normalize a list value.
        
        Args:
            list_value: List or comma-separated string value to normalize
            
        Returns:
            Normalized list
        """
        if list_value is None:
            return []
            
        if isinstance(list_value, str):
            # Split comma-separated string
            items = [item.strip() for item in list_value.split(',')]
            # Filter out empty items
            return [item for item in items if item]
            
        elif isinstance(list_value, list):
            # Clean each item
            return [
                self._normalize_field_value(item, self._infer_field_type("", item)) 
                for item in list_value if item is not None
            ]
            
        # If it's not a list or string, wrap in a list
        return [list_value]
    
    def _looks_like_date(self, value: str) -> bool:
        """
        Check if a string looks like a date.
        
        Args:
            value: String value to check
            
        Returns:
            True if the string looks like a date
        """
        if not isinstance(value, str):
            return False
            
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
            r'\d{2}/\d{2}/\d{2}',  # MM/DD/YY or DD/MM/YY
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # 1 Jan 2020
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',  # Jan 1, 2020
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago'  # X days ago
        ]
        
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in date_patterns)
    
    def _looks_like_price(self, value: str) -> bool:
        """
        Check if a string looks like a price.
        
        Args:
            value: String value to check
            
        Returns:
            True if the string looks like a price
        """
        if not isinstance(value, str):
            return False
            
        # Price patterns with currency symbols
        price_patterns = [
            r'[$€£¥₹]\s*\d+(?:[.,]\d+)?',  # $10.99
            r'\d+(?:[.,]\d+)?\s*[$€£¥₹]',  # 10.99$
            r'USD\s*\d+(?:[.,]\d+)?',      # USD 10.99
            r'\d+(?:[.,]\d+)?\s*USD',      # 10.99 USD
            r'\d+(?:[.,]\d+)?\s*(?:dollars|euros|pounds|yen)'  # 10.99 dollars
        ]
        
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in price_patterns)
    
    def _looks_like_url(self, value: str) -> bool:
        """
        Check if a string looks like a URL.
        
        Args:
            value: String value to check
            
        Returns:
            True if the string looks like a URL
        """
        if not isinstance(value, str):
            return False
            
        url_patterns = [
            r'^https?://',
            r'^www\.',
            r'\.(com|net|org|io|gov|edu|co)(/|$)'
        ]
        
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in url_patterns)
    
    def _count_normalized_fields(self, normalized_data: Any) -> int:
        """
        Count the number of normalized fields in the data.
        
        Args:
            normalized_data: The normalized data
            
        Returns:
            Number of normalized fields
        """
        if isinstance(normalized_data, dict):
            return len([k for k in normalized_data.keys() if not k.startswith('_')])
        elif isinstance(normalized_data, list):
            if not normalized_data:
                return 0
            if all(isinstance(item, dict) for item in normalized_data):
                # For a list of dictionaries, count fields in the first item
                return self._count_normalized_fields(normalized_data[0])
            else:
                # For a simple list, count as 1 field
                return 1
        else:
            return 1


class DataValidationStage(PipelineStage):
    """
    Pipeline stage for validating extracted data.
    
    This stage validates data against schemas, rules, and quality checks, with support for:
    - Schema validation using JSON Schema
    - Data type validation
    - Required field validation
    - Value range validation
    - Pattern matching validation
    - Cross-field validation rules
    
    Configuration options:
        input_key (str): Key in the context to get the data from
        output_key (str): Key in the context to store the validation results
        schema (Dict): JSON Schema for validation
        validation_rules (Dict): Custom validation rules
        required_fields (List): List of required fields
        fail_on_error (bool): Whether to fail the stage on validation errors (default: False)
        add_validation_metadata (bool): Whether to add validation info to fields (default: True)
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a data validation stage.
        
        Args:
            name: Name of the stage
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Extract configuration options with defaults
        self.input_key = self.config.get("input_key", "normalized_data")
        self.output_key = self.config.get("output_key", "validated_data")
        self.schema = self.config.get("schema", {})
        self.validation_rules = self.config.get("validation_rules", {})
        self.required_fields = self.config.get("required_fields", [])
        self.fail_on_error = self.config.get("fail_on_error", False)
        self.add_validation_metadata = self.config.get("add_validation_metadata", True)
        
        # Try to import jsonschema if available
        self.has_jsonschema = False
        try:
            import jsonschema
            self.jsonschema = jsonschema
            self.has_jsonschema = True
        except ImportError:
            self.logger.warning("jsonschema package not available, schema validation will be limited")
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains the required data.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.get(self.input_key):
            self.logger.error(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input '{self.input_key}'")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Validate data in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get the data from the context
            data = context.get(self.input_key)
            
            if not data:
                self.logger.warning(f"Empty data in '{self.input_key}'")
                context.add_error(self.name, "Empty data for validation")
                return not self.fail_on_error
                
            # Initialize validation results
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "field_validations": {}
            }
            
            # Determine the data type and apply appropriate validation
            if isinstance(data, dict):
                self._validate_dict(data, "", validation_results)
            elif isinstance(data, list):
                self._validate_list(data, "", validation_results)
            else:
                error_msg = f"Unsupported data type for validation: {type(data)}"
                validation_results["valid"] = False
                validation_results["errors"].append(error_msg)
                self.logger.warning(error_msg)
                
            # Store the validation results in the context
            context.set(self.output_key, data)  # Store the original data
            context.set(f"{self.output_key}_validation", validation_results)
            
            # If we're adding validation metadata directly to the data
            if self.add_validation_metadata:
                validated_data = self._add_validation_metadata(data, validation_results)
                context.set(self.output_key, validated_data)
            
            # Log validation summary
            error_count = len(validation_results["errors"])
            warning_count = len(validation_results["warnings"])
            
            if error_count > 0:
                self.logger.warning(f"Validation found {error_count} errors and {warning_count} warnings")
                if self.fail_on_error:
                    for error in validation_results["errors"]:
                        context.add_error(self.name, error)
                    return False
            else:
                self.logger.info(f"Validation succeeded with {warning_count} warnings")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            context.add_error(self.name, f"Data validation failed: {str(e)}")
            return await self.handle_error(context, e)
            
    def _validate_dict(self, data: Dict[str, Any], path: str, results: Dict[str, Any]) -> None:
        """
        Validate a dictionary against validation rules.
        
        Args:
            data: Dictionary to validate
            path: Current path in the data structure
            results: Validation results to update
        """
        # Check for required fields
        for field in self.required_fields:
            if field not in data:
                error = f"Required field '{field}' is missing"
                if path:
                    error = f"{path}.{error}"
                results["errors"].append(error)
                results["valid"] = False
                
        # Check JSON Schema if available
        if self.has_jsonschema and self.schema:
            try:
                self.jsonschema.validate(data, self.schema)
            except self.jsonschema.exceptions.ValidationError as e:
                error = f"Schema validation error: {e.message}"
                if path:
                    error = f"{path}: {error}"
                results["errors"].append(error)
                results["valid"] = False
                
        # Validate each field
        for field_name, value in data.items():
            field_path = f"{path}.{field_name}" if path else field_name
            
            # Initialize field validation results
            field_results = {
                "valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Check field-specific validation rules
            if field_name in self.validation_rules:
                rule = self.validation_rules[field_name]
                self._apply_validation_rule(value, rule, field_path, field_results)
                
            # Recursively validate nested structures
            if isinstance(value, dict):
                self._validate_dict(value, field_path, results)
            elif isinstance(value, list):
                self._validate_list(value, field_path, results)
                
            # Update the main results with field results
            if not field_results["valid"]:
                results["valid"] = False
                results["errors"].extend(field_results["errors"])
                results["warnings"].extend(field_results["warnings"])
                
            # Store field validation results
            results["field_validations"][field_path] = field_results
            
    def _validate_list(self, data: List[Any], path: str, results: Dict[str, Any]) -> None:
        """
        Validate a list of items.
        
        Args:
            data: List to validate
            path: Current path in the data structure
            results: Validation results to update
        """
        # Check if list is empty when it shouldn't be
        if not data and path in self.required_fields:
            error = f"Required list '{path}' is empty"
            results["errors"].append(error)
            results["valid"] = False
            
        # Validate each item in the list
        for i, item in enumerate(data):
            item_path = f"{path}[{i}]"
            
            if isinstance(item, dict):
                self._validate_dict(item, item_path, results)
            elif isinstance(item, list):
                self._validate_list(item, item_path, results)
                
    def _apply_validation_rule(self, 
                             value: Any, 
                             rule: Dict[str, Any], 
                             path: str, 
                             results: Dict[str, Any]) -> None:
        """
        Apply a validation rule to a value.
        
        Args:
            value: Value to validate
            rule: Validation rule to apply
            path: Current path in the data structure
            results: Validation results to update
        """
        # Type validation
        if "type" in rule:
            expected_type = rule["type"]
            if expected_type == "string" and not isinstance(value, str):
                results["errors"].append(f"{path}: Expected string, got {type(value).__name__}")
                results["valid"] = False
            elif expected_type == "number" and not isinstance(value, (int, float)):
                results["errors"].append(f"{path}: Expected number, got {type(value).__name__}")
                results["valid"] = False
            elif expected_type == "integer" and not isinstance(value, int):
                results["errors"].append(f"{path}: Expected integer, got {type(value).__name__}")
                results["valid"] = False
            elif expected_type == "boolean" and not isinstance(value, bool):
                results["errors"].append(f"{path}: Expected boolean, got {type(value).__name__}")
                results["valid"] = False
            elif expected_type == "array" and not isinstance(value, list):
                results["errors"].append(f"{path}: Expected array, got {type(value).__name__}")
                results["valid"] = False
            elif expected_type == "object" and not isinstance(value, dict):
                results["errors"].append(f"{path}: Expected object, got {type(value).__name__}")
                results["valid"] = False
                
        # String validation
        if isinstance(value, str):
            # Min length
            if "minLength" in rule and len(value) < rule["minLength"]:
                results["errors"].append(
                    f"{path}: String too short ({len(value)} chars), minimum {rule['minLength']}"
                )
                results["valid"] = False
                
            # Max length
            if "maxLength" in rule and len(value) > rule["maxLength"]:
                results["errors"].append(
                    f"{path}: String too long ({len(value)} chars), maximum {rule['maxLength']}"
                )
                results["valid"] = False
                
            # Pattern matching
            if "pattern" in rule:
                pattern = rule["pattern"]
                if not re.search(pattern, value):
                    results["errors"].append(f"{path}: String does not match pattern '{pattern}'")
                    results["valid"] = False
                    
            # Enumeration
            if "enum" in rule and value not in rule["enum"]:
                results["errors"].append(f"{path}: Value '{value}' not in allowed values {rule['enum']}")
                results["valid"] = False
                
        # Number validation
        if isinstance(value, (int, float)):
            # Minimum value
            if "minimum" in rule and value < rule["minimum"]:
                results["errors"].append(f"{path}: Value {value} less than minimum {rule['minimum']}")
                results["valid"] = False
                
            # Maximum value
            if "maximum" in rule and value > rule["maximum"]:
                results["errors"].append(f"{path}: Value {value} greater than maximum {rule['maximum']}")
                results["valid"] = False
                
            # Multiple of
            if "multipleOf" in rule and value % rule["multipleOf"] != 0:
                results["errors"].append(f"{path}: Value {value} not a multiple of {rule['multipleOf']}")
                results["valid"] = False
                
        # Array validation
        if isinstance(value, list):
            # Min items
            if "minItems" in rule and len(value) < rule["minItems"]:
                results["errors"].append(
                    f"{path}: Array too short ({len(value)} items), minimum {rule['minItems']}"
                )
                results["valid"] = False
                
            # Max items
            if "maxItems" in rule and len(value) > rule["maxItems"]:
                results["errors"].append(
                    f"{path}: Array too long ({len(value)} items), maximum {rule['maxItems']}"
                )
                results["valid"] = False
                
            # Unique items
            if "uniqueItems" in rule and rule["uniqueItems"]:
                unique_items = set()
                duplicates = []
                
                for item in value:
                    # For hashable items
                    if isinstance(item, (str, int, float, bool, tuple)):
                        if item in unique_items:
                            duplicates.append(item)
                        else:
                            unique_items.add(item)
                            
                if duplicates:
                    results["errors"].append(f"{path}: Array contains duplicate items {duplicates}")
                    results["valid"] = False
        
        # Object validation
        if isinstance(value, dict):
            # Required properties
            if "required" in rule:
                for required_prop in rule["required"]:
                    if required_prop not in value:
                        results["errors"].append(f"{path}: Missing required property '{required_prop}'")
                        results["valid"] = False
                        
            # Property dependencies
            if "dependencies" in rule:
                for prop, dependencies in rule["dependencies"].items():
                    if prop in value:
                        if isinstance(dependencies, list):
                            # Property dependencies
                            for dep in dependencies:
                                if dep not in value:
                                    results["errors"].append(
                                        f"{path}: Property '{prop}' depends on '{dep}' which is missing"
                                    )
                                    results["valid"] = False
                        elif isinstance(dependencies, dict):
                            # Schema dependencies
                            # This would require full JSON Schema implementation
                            pass
        
        # Custom validation function
        if "validate" in rule and callable(rule["validate"]):
            try:
                validation_result = rule["validate"](value)
                if validation_result is not True:
                    if isinstance(validation_result, str):
                        error_message = validation_result
                    else:
                        error_message = "Failed custom validation"
                        
                    results["errors"].append(f"{path}: {error_message}")
                    results["valid"] = False
            except Exception as e:
                results["errors"].append(f"{path}: Custom validation error: {str(e)}")
                results["valid"] = False
                
    def _add_validation_metadata(self, 
                               data: Any, 
                               validation_results: Dict[str, Any]) -> Any:
        """
        Add validation metadata to the data structure.
        
        Args:
            data: The data to add validation metadata to
            validation_results: Validation results
            
        Returns:
            Data with added validation metadata
        """
        if not self.add_validation_metadata:
            return data
            
        if isinstance(data, dict):
            result = {}
            # Add overall validation metadata
            result["_validation"] = {
                "valid": validation_results["valid"],
                "error_count": len(validation_results["errors"]),
                "warning_count": len(validation_results["warnings"])
            }
            
            # Add field-specific validation info
            for field_name, value in data.items():
                result[field_name] = value
                
                # Check if we have validation results for this field
                field_path = field_name
                if field_path in validation_results["field_validations"]:
                    field_validation = validation_results["field_validations"][field_path]
                    
                    # If we have errors or warnings, add them
                    if field_validation["errors"] or field_validation["warnings"]:
                        if isinstance(value, dict):
                            # For objects, add validation as a property
                            result[field_name]["_validation"] = {
                                "valid": field_validation["valid"],
                                "errors": field_validation["errors"],
                                "warnings": field_validation["warnings"]
                            }
                        else:
                            # For primitive values, convert to an object with value and validation
                            result[field_name] = {
                                "value": value,
                                "_validation": {
                                    "valid": field_validation["valid"],
                                    "errors": field_validation["errors"],
                                    "warnings": field_validation["warnings"]
                                }
                            }
                            
            return result
            
        elif isinstance(data, list):
            # For lists, we don't modify the structure, just return as is
            return data
            
        else:
            # For primitive values, we don't add validation metadata
            return data


class DataTransformationStage(PipelineStage):
    """
    Pipeline stage for applying transformations to data.
    
    This stage applies transformations to data fields, with support for:
    - Field renaming
    - Value transformation functions
    - Type conversion
    - Structure transformation
    - Field filtering (inclusion/exclusion)
    
    Configuration options:
        input_key (str): Key in the context to get the data from
        output_key (str): Key in the context to store the transformed data
        field_mappings (Dict): Field name mappings for renaming
        transformations (Dict): Transformation functions for fields
        include_fields (List): Fields to include (all others will be excluded)
        exclude_fields (List): Fields to exclude (all others will be included)
        add_fields (Dict): Additional fields to add with static or computed values
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a data transformation stage.
        
        Args:
            name: Name of the stage
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Extract configuration options with defaults
        self.input_key = self.config.get("input_key", "validated_data")
        self.output_key = self.config.get("output_key", "transformed_data")
        self.field_mappings = self.config.get("field_mappings", {})
        self.transformations = self.config.get("transformations", {})
        self.include_fields = self.config.get("include_fields", [])
        self.exclude_fields = self.config.get("exclude_fields", [])
        self.add_fields = self.config.get("add_fields", {})
        
        # Precompile transformation functions
        self.transformation_functions = {}
        self._compile_transformations()
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains the required data.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.get(self.input_key):
            self.logger.error(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input '{self.input_key}'")
            return False
        return True
        
    async def process(self, context: PipelineContext) -> bool:
        """
        Transform data in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Get the data from the context
            data = context.get(self.input_key)
            
            if not data:
                self.logger.warning(f"Empty data in '{self.input_key}'")
                return False
                
            # Determine the data type and apply appropriate transformation
            if isinstance(data, dict):
                transformed_data = self._transform_dict(data)
            elif isinstance(data, list):
                transformed_data = self._transform_list(data)
            else:
                self.logger.warning(f"Unsupported data type: {type(data)}")
                return False
                
            # Store the transformed data in the context
            context.set(self.output_key, transformed_data)
            
            # Add transformation metadata
            transform_metadata = {
                "transformation_stage": self.name,
                "fields_transformed": len(self.transformations) if isinstance(data, dict) else 0,
                "fields_added": len(self.add_fields) if isinstance(data, dict) else 0,
                "fields_renamed": len(self.field_mappings) if isinstance(data, dict) else 0
            }
            context.set(f"{self.output_key}_metadata", transform_metadata)
            
            self.logger.info(f"Data transformed successfully: {transform_metadata}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            context.add_error(self.name, f"Data transformation failed: {str(e)}")
            return await self.handle_error(context, e)
            
    def _compile_transformations(self) -> None:
        """
        Compile transformation functions.
        """
        for field, transform in self.transformations.items():
            if callable(transform):
                # Function is already callable
                self.transformation_functions[field] = transform
            elif isinstance(transform, str):
                # Predefined transformation
                if transform == 'uppercase':
                    self.transformation_functions[field] = lambda x: x.upper() if isinstance(x, str) else x
                elif transform == 'lowercase':
                    self.transformation_functions[field] = lambda x: x.lower() if isinstance(x, str) else x
                elif transform == 'capitalize':
                    self.transformation_functions[field] = lambda x: x.capitalize() if isinstance(x, str) else x
                elif transform == 'strip':
                    self.transformation_functions[field] = lambda x: x.strip() if isinstance(x, str) else x
                elif transform == 'to_int':
                    self.transformation_functions[field] = self._to_int
                elif transform == 'to_float':
                    self.transformation_functions[field] = self._to_float
                elif transform == 'to_string':
                    self.transformation_functions[field] = lambda x: str(x) if x is not None else None
                elif transform == 'to_boolean':
                    self.transformation_functions[field] = self._to_boolean
                elif transform == 'iso_date':
                    self.transformation_functions[field] = self._to_iso_date
                else:
                    self.logger.warning(f"Unknown transformation function '{transform}' for field '{field}'")
            else:
                self.logger.warning(f"Invalid transformation for field '{field}': {transform}")
        
    def _transform_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a dictionary of data.
        
        Args:
            data: Dictionary to transform
            
        Returns:
            Transformed dictionary
        """
        result = {}
        
        # Handle include/exclude fields
        fields_to_process = set(data.keys())
        
        if self.include_fields:
            # Only include specified fields
            fields_to_process = fields_to_process.intersection(set(self.include_fields))
        
        if self.exclude_fields:
            # Exclude specified fields
            fields_to_process = fields_to_process.difference(set(self.exclude_fields))
            
        # Process each field
        for field_name in fields_to_process:
            value = data[field_name]
            
            # Apply field mapping (rename)
            target_field = self.field_mappings.get(field_name, field_name)
            
            # Apply transformation if any
            if field_name in self.transformation_functions:
                try:
                    transform_func = self.transformation_functions[field_name]
                    value = transform_func(value)
                except Exception as e:
                    self.logger.warning(f"Error transforming field '{field_name}': {str(e)}")
                    
            # Handle nested structures
            if isinstance(value, dict):
                value = self._transform_dict(value)
            elif isinstance(value, list):
                value = self._transform_list(value)
                
            # Add the transformed value to the result
            result[target_field] = value
            
        # Add additional fields
        for field_name, field_value in self.add_fields.items():
            if callable(field_value):
                # Computed field
                try:
                    result[field_name] = field_value(data)
                except Exception as e:
                    self.logger.warning(f"Error computing field '{field_name}': {str(e)}")
                    result[field_name] = None
            else:
                # Static field
                result[field_name] = field_value
                
        return result
    
    def _transform_list(self, data: List[Any]) -> List[Any]:
        """
        Transform a list of data.
        
        Args:
            data: List to transform
            
        Returns:
            Transformed list
        """
        transformed_list = []
        
        for item in data:
            if isinstance(item, dict):
                transformed_item = self._transform_dict(item)
            elif isinstance(item, list):
                transformed_item = self._transform_list(item)
            else:
                # For primitive values, no transformation is applied
                transformed_item = item
                
            transformed_list.append(transformed_item)
            
        return transformed_list
    
    # Utility transformation functions
    
    def _to_int(self, value: Any) -> Optional[int]:
        """Convert value to integer."""
        if value is None:
            return None
            
        if isinstance(value, (int, float)):
            return int(value)
            
        if isinstance(value, str):
            # Remove non-numeric characters
            cleaned = re.sub(r'[^\d.-]', '', value)
            try:
                return int(float(cleaned))
            except (ValueError, TypeError):
                return None
                
        return None
        
    def _to_float(self, value: Any) -> Optional[float]:
        """Convert value to float."""
        if value is None:
            return None
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.-]', '', value)
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return None
                
        return None
        
    def _to_boolean(self, value: Any) -> Optional[bool]:
        """Convert value to boolean."""
        if value is None:
            return None
            
        if isinstance(value, bool):
            return value
            
        if isinstance(value, (int, float)):
            return bool(value)
            
        if isinstance(value, str):
            value = value.lower().strip()
            return value in ['true', 'yes', 'y', '1', 'on', 'enabled', 'active']
            
        return bool(value)
        
    def _to_iso_date(self, value: Any) -> Optional[str]:
        """Convert value to ISO format date string."""
        if value is None:
            return None
            
        if isinstance(value, datetime.datetime):
            return value.isoformat()
            
        if isinstance(value, datetime.date):
            return value.isoformat()
            
        if isinstance(value, str):
            # Try to parse with common formats
            for fmt in [
                "%Y-%m-%d", "%Y/%m/%d", 
                "%m/%d/%Y", "%d/%m/%Y", 
                "%m-%d-%Y", "%d-%m-%Y",
                "%b %d, %Y", "%B %d, %Y"
            ]:
                try:
                    date_obj = datetime.datetime.strptime(value, fmt)
                    return date_obj.isoformat()
                except ValueError:
                    continue
                    
            # Try more flexible parsing if dateutil is available
            try:
                from dateutil import parser
                date_obj = parser.parse(value)
                return date_obj.isoformat()
            except (ImportError, ValueError):
                return None
                
        # For numeric values, assume unix timestamp
        if isinstance(value, (int, float)):
            try:
                date_obj = datetime.datetime.fromtimestamp(value)
                return date_obj.isoformat()
            except (ValueError, OverflowError):
                return None
                
        return None
        
    def add_transformation(self, field: str, transformation: Union[Callable, str]) -> None:
        """
        Add a new transformation at runtime.
        
        Args:
            field: Field name to transform
            transformation: Transformation function or name
        """
        self.transformations[field] = transformation
        self._compile_transformations()
        
    def add_field_mapping(self, source_field: str, target_field: str) -> None:
        """
        Add a new field mapping at runtime.
        
        Args:
            source_field: Original field name
            target_field: New field name
        """
        self.field_mappings[source_field] = target_field