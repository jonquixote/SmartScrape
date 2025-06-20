"""
Normalization Stage Module

This module provides the NormalizationStage class, which is responsible for standardizing
and normalizing extracted data in the extraction pipeline.
"""

import logging
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

from core.pipeline.context import PipelineContext
from core.pipeline.stages.base_stages import ProcessingStage

logger = logging.getLogger(__name__)

class NormalizationStage(ProcessingStage):
    """
    Pipeline stage for normalizing and standardizing extracted data.
    
    This stage processes extracted data to ensure consistent formatting,
    data types, and structure across different data sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the normalization stage.
        
        Args:
            config: Configuration dictionary for the stage
        """
        super().__init__(config or {})
        self.input_key = self.config.get("input_key", "extracted_data")
        self.output_key = self.config.get("output_key", "normalized_data")
        self.field_types = self.config.get("field_types", {})
        
        # Configure normalization options
        self.trim_strings = self.config.get("trim_strings", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.standardize_dates = self.config.get("standardize_dates", True)
        self.standardize_prices = self.config.get("standardize_prices", True)
        self.standardize_units = self.config.get("standardize_units", False)
        self.remove_html = self.config.get("remove_html", True)
        self.normalize_lists = self.config.get("normalize_lists", True)
    
    async def process(self, context: PipelineContext) -> bool:
        """
        Process the extracted data and normalize it.
        
        Args:
            context: Pipeline context object containing extracted data
            
        Returns:
            True if normalization succeeded, False otherwise
        """
        # Get the extracted data from the context
        extracted_data = context.get(self.input_key)
        if not extracted_data:
            logger.warning(f"No data found at key '{self.input_key}' for normalization")
            return False
        
        try:
            # Normalize the data
            normalized_data = self._normalize_data(extracted_data)
            
            # Store the normalized data in the context
            context.set(self.output_key, normalized_data)
            
            # Track the normalization in the pipeline metadata
            metadata = context.get_metadata()
            metadata["normalization"] = {
                "normalized_fields": list(normalized_data.keys()),
                "timestamp": datetime.now().isoformat()
            }
            context.set_metadata(metadata)
            
            logger.debug(f"Successfully normalized {len(normalized_data)} fields")
            return True
            
        except Exception as e:
            logger.error(f"Error during data normalization: {str(e)}")
            context.add_error("normalization", f"Failed to normalize data: {str(e)}")
            return False
    
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the extracted data based on configuration.
        
        Args:
            data: Dictionary of extracted data
            
        Returns:
            Dictionary of normalized data
        """
        normalized = {}
        
        # Process each field in the data
        for field_name, field_value in data.items():
            # Skip internal fields (starting with underscore)
            if field_name.startswith('_'):
                normalized[field_name] = field_value
                continue
            
            # Get the expected field type, if specified
            field_type = self.field_types.get(field_name, self._infer_field_type(field_value))
            
            # Normalize based on field type
            normalized_value = self._normalize_field(field_value, field_type)
            normalized[field_name] = normalized_value
        
        return normalized
    
    def _normalize_field(self, value: Any, field_type: str) -> Any:
        """
        Normalize a field value based on its type.
        
        Args:
            value: Original field value
            field_type: Type of the field
            
        Returns:
            Normalized field value
        """
        # Handle None values
        if value is None:
            return None
        
        # Handle different field types
        if field_type == "text" or field_type == "string":
            return self._normalize_text(value)
        elif field_type == "long_text":
            return self._normalize_long_text(value)
        elif field_type == "html":
            return self._normalize_html(value)
        elif field_type == "date":
            return self._normalize_date(value)
        elif field_type == "price":
            return self._normalize_price(value)
        elif field_type == "measurement":
            return self._normalize_measurement(value)
        elif field_type == "list":
            return self._normalize_list(value)
        elif field_type == "key_value_list":
            return self._normalize_key_value_list(value)
        elif field_type == "object":
            return self._normalize_object(value)
        else:
            # Default: return as is for unknown types
            return value
    
    def _infer_field_type(self, value: Any) -> str:
        """
        Infer the type of a field based on its value.
        
        Args:
            value: Field value
            
        Returns:
            Inferred field type
        """
        if value is None:
            return "null"
        
        if isinstance(value, str):
            # Check if it looks like a date
            if re.match(r'^(\d{1,4}[-/]\d{1,2}[-/]\d{1,4})|(\w+ \d{1,2}, \d{4})', value):
                return "date"
            
            # Check if it looks like a price
            if re.match(r'^[$€£¥]?\s*\d+([.,]\d+)?\s*[$€£¥]?$', value):
                return "price"
            
            # Check if it's HTML
            if re.search(r'<[^>]+>', value):
                return "html"
            
            # Check if it's a long text
            if len(value) > 200:
                return "long_text"
                
            return "text"
        
        if isinstance(value, (list, tuple)):
            return "list"
        
        if isinstance(value, dict):
            # Check if it's a key-value list
            if all(isinstance(k, str) for k in value.keys()):
                return "key_value_list"
            return "object"
        
        if isinstance(value, (int, float)):
            return "number"
        
        if isinstance(value, bool):
            return "boolean"
        
        # Default type
        return "unknown"
    
    def _normalize_text(self, value: Any) -> str:
        """
        Normalize a text value.
        
        Args:
            value: Text value
            
        Returns:
            Normalized text
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Trim if configured
        if self.trim_strings:
            value = value.strip()
        
        # Normalize whitespace if configured
        if self.normalize_whitespace:
            value = re.sub(r'\s+', ' ', value)
        
        return value
    
    def _normalize_long_text(self, value: Any) -> str:
        """
        Normalize a long text value.
        
        Args:
            value: Long text value
            
        Returns:
            Normalized long text
        """
        # First normalize as text
        value = self._normalize_text(value)
        
        # Remove HTML if configured
        if self.remove_html:
            value = re.sub(r'<[^>]+>', '', value)
            # Also normalize any resulting whitespace
            value = re.sub(r'\s+', ' ', value)
            value = value.strip()
        
        return value
    
    def _normalize_html(self, value: Any) -> str:
        """
        Normalize HTML content.
        
        Args:
            value: HTML content
            
        Returns:
            Normalized HTML
        """
        if not isinstance(value, str):
            value = str(value)
        
        # Remove unwanted whitespace between tags
        value = re.sub(r'>\s+<', '><', value)
        
        # Normalize whitespace within text nodes
        # This is complex to do completely, so this is a simplified approach
        value = re.sub(r'([^<>])\s+([^<>])', r'\1 \2', value)
        
        return value
    
    def _normalize_date(self, value: Any) -> str:
        """
        Normalize a date value to ISO format.
        
        Args:
            value: Date value
            
        Returns:
            Normalized date in ISO format
        """
        if not self.standardize_dates:
            return value
        
        if not isinstance(value, str):
            value = str(value)
        
        # Try various date formats
        date_formats = [
            '%Y-%m-%d',          # 2023-01-25
            '%d/%m/%Y',          # 25/01/2023
            '%m/%d/%Y',          # 01/25/2023
            '%B %d, %Y',         # January 25, 2023
            '%b %d, %Y',         # Jan 25, 2023
            '%d %B %Y',          # 25 January 2023
            '%d %b %Y',          # 25 Jan 2023
            '%Y/%m/%d',          # 2023/01/25
            '%d-%m-%Y',          # 25-01-2023
            '%m-%d-%Y',          # 01-25-2023
        ]
        
        # Try to parse the date
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(value, fmt)
                return date_obj.isoformat().split('T')[0]  # Return YYYY-MM-DD
            except ValueError:
                continue
        
        # If all formats fail, return the original value
        return value
    
    def _normalize_price(self, value: Any) -> Dict[str, Any]:
        """
        Normalize a price value.
        
        Args:
            value: Price value
            
        Returns:
            Dictionary with normalized price data
        """
        if not self.standardize_prices:
            return value
        
        if isinstance(value, (int, float)):
            return {
                "amount": value,
                "currency": "USD",  # Default currency
                "formatted": f"${value:.2f}"
            }
        
        if not isinstance(value, str):
            value = str(value)
        
        # Extract the numeric value and currency
        # This pattern matches:
        # $10, $10.99, 10.99$, €10.99, 10,99 €, 10.99 USD, etc.
        price_pattern = r'([^\d]*)(\d+[.,]?\d*)([^\d]*)'
        match = re.search(price_pattern, value)
        
        if not match:
            return value
        
        prefix, amount_str, suffix = match.groups()
        
        # Clean and parse the amount
        amount_str = amount_str.replace(',', '.')
        try:
            amount = float(amount_str)
        except ValueError:
            return value
        
        # Determine the currency
        currency = "USD"  # Default
        
        # Check prefix and suffix for currency symbols
        currency_map = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            'kr': 'SEK',  # Swedish Krona
        }
        
        for symbol, curr in currency_map.items():
            if symbol in prefix or symbol in suffix:
                currency = curr
                break
        
        # Check for explicit currency codes
        currency_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY']
        for code in currency_codes:
            if code in suffix.upper() or code in prefix.upper():
                currency = code
                break
        
        # Format the result
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥'
        }
        
        symbol = currency_symbols.get(currency, currency)
        formatted = f"{symbol}{amount:.2f}"
        
        return {
            "amount": amount,
            "currency": currency,
            "formatted": formatted
        }
    
    def _normalize_measurement(self, value: Any) -> Dict[str, Any]:
        """
        Normalize a measurement value.
        
        Args:
            value: Measurement value
            
        Returns:
            Dictionary with normalized measurement data
        """
        if not self.standardize_units:
            return value
        
        if not isinstance(value, str):
            value = str(value)
        
        # Extract the numeric value and unit
        # This pattern matches: 10cm, 10 cm, 10.5cm, 10,5 cm, etc.
        measurement_pattern = r'(\d+[.,]?\d*)\s*([a-zA-Z]+)'
        match = re.search(measurement_pattern, value)
        
        if not match:
            return value
        
        amount_str, unit = match.groups()
        
        # Clean and parse the amount
        amount_str = amount_str.replace(',', '.')
        try:
            amount = float(amount_str)
        except ValueError:
            return value
        
        # Normalize the unit
        unit = unit.lower()
        
        # Unit conversions to standard units
        unit_conversions = {
            # Length
            'cm': {'type': 'length', 'to_base': lambda v: v / 100, 'from_base': lambda v: v * 100, 'base_unit': 'm'},
            'centimeter': {'type': 'length', 'to_base': lambda v: v / 100, 'from_base': lambda v: v * 100, 'base_unit': 'm'},
            'mm': {'type': 'length', 'to_base': lambda v: v / 1000, 'from_base': lambda v: v * 1000, 'base_unit': 'm'},
            'millimeter': {'type': 'length', 'to_base': lambda v: v / 1000, 'from_base': lambda v: v * 1000, 'base_unit': 'm'},
            'm': {'type': 'length', 'to_base': lambda v: v, 'from_base': lambda v: v, 'base_unit': 'm'},
            'meter': {'type': 'length', 'to_base': lambda v: v, 'from_base': lambda v: v, 'base_unit': 'm'},
            'km': {'type': 'length', 'to_base': lambda v: v * 1000, 'from_base': lambda v: v / 1000, 'base_unit': 'm'},
            'kilometer': {'type': 'length', 'to_base': lambda v: v * 1000, 'from_base': lambda v: v / 1000, 'base_unit': 'm'},
            'in': {'type': 'length', 'to_base': lambda v: v * 0.0254, 'from_base': lambda v: v / 0.0254, 'base_unit': 'm'},
            'inch': {'type': 'length', 'to_base': lambda v: v * 0.0254, 'from_base': lambda v: v / 0.0254, 'base_unit': 'm'},
            'ft': {'type': 'length', 'to_base': lambda v: v * 0.3048, 'from_base': lambda v: v / 0.3048, 'base_unit': 'm'},
            'foot': {'type': 'length', 'to_base': lambda v: v * 0.3048, 'from_base': lambda v: v / 0.3048, 'base_unit': 'm'},
            
            # Weight
            'g': {'type': 'weight', 'to_base': lambda v: v / 1000, 'from_base': lambda v: v * 1000, 'base_unit': 'kg'},
            'gram': {'type': 'weight', 'to_base': lambda v: v / 1000, 'from_base': lambda v: v * 1000, 'base_unit': 'kg'},
            'kg': {'type': 'weight', 'to_base': lambda v: v, 'from_base': lambda v: v, 'base_unit': 'kg'},
            'kilogram': {'type': 'weight', 'to_base': lambda v: v, 'from_base': lambda v: v, 'base_unit': 'kg'},
            'lb': {'type': 'weight', 'to_base': lambda v: v * 0.45359237, 'from_base': lambda v: v / 0.45359237, 'base_unit': 'kg'},
            'pound': {'type': 'weight', 'to_base': lambda v: v * 0.45359237, 'from_base': lambda v: v / 0.45359237, 'base_unit': 'kg'},
            'oz': {'type': 'weight', 'to_base': lambda v: v * 0.028349523125, 'from_base': lambda v: v / 0.028349523125, 'base_unit': 'kg'},
            'ounce': {'type': 'weight', 'to_base': lambda v: v * 0.028349523125, 'from_base': lambda v: v / 0.028349523125, 'base_unit': 'kg'},
        }
        
        # See if we can normalize this unit
        unit_info = unit_conversions.get(unit)
        if not unit_info:
            return value
        
        # Convert to base unit
        base_value = unit_info['to_base'](amount)
        
        return {
            "value": amount,
            "unit": unit,
            "type": unit_info['type'],
            "base_value": base_value,
            "base_unit": unit_info['base_unit'],
            "formatted": f"{amount} {unit}"
        }
    
    def _normalize_list(self, value: Any) -> List[Any]:
        """
        Normalize a list value.
        
        Args:
            value: List value
            
        Returns:
            Normalized list
        """
        if not self.normalize_lists:
            return value
        
        # Convert to list if not already
        if isinstance(value, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    value = parsed
                else:
                    # If it's a string but not JSON, split by commas
                    value = [item.strip() for item in value.split(',')]
            except json.JSONDecodeError:
                # If it's not valid JSON, split by commas
                value = [item.strip() for item in value.split(',')]
        
        if not isinstance(value, (list, tuple)):
            # If it's not a list or tuple, wrap it in a list
            return [value] if value is not None else []
        
        # Normalize each item in the list
        normalized_list = []
        for item in value:
            if isinstance(item, dict):
                # Recursively normalize nested objects
                normalized_item = self._normalize_object(item)
            elif isinstance(item, (list, tuple)):
                # Recursively normalize nested lists
                normalized_item = self._normalize_list(item)
            elif isinstance(item, str):
                # Normalize strings
                normalized_item = self._normalize_text(item)
            else:
                # Keep other types as is
                normalized_item = item
            
            # Add non-empty items
            if normalized_item is not None and (not isinstance(normalized_item, str) or normalized_item.strip()):
                normalized_list.append(normalized_item)
        
        return normalized_list
    
    def _normalize_key_value_list(self, value: Any) -> Dict[str, Any]:
        """
        Normalize a key-value list (dictionary).
        
        Args:
            value: Key-value list
            
        Returns:
            Normalized key-value list
        """
        if not isinstance(value, dict):
            # If it's a string, try to parse as JSON
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        value = parsed
                    else:
                        return value  # Not a dictionary, return as is
                except json.JSONDecodeError:
                    return value  # Not valid JSON, return as is
            else:
                return value  # Not a dictionary or string, return as is
        
        # Normalize each value in the dictionary
        normalized_dict = {}
        for key, val in value.items():
            # Normalize the key
            if isinstance(key, str):
                key = self._normalize_text(key)
            
            # Normalize the value based on its type
            field_type = self._infer_field_type(val)
            normalized_val = self._normalize_field(val, field_type)
            
            normalized_dict[key] = normalized_val
        
        return normalized_dict
    
    def _normalize_object(self, value: Any) -> Dict[str, Any]:
        """
        Normalize an object (dictionary).
        
        Args:
            value: Object value
            
        Returns:
            Normalized object
        """
        # Delegate to key_value_list normalization
        return self._normalize_key_value_list(value)