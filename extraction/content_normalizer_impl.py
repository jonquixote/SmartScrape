"""
Content Normalizer Implementation

This module provides a concrete implementation of the ContentNormalizer abstract class
that normalizes text, dates, units, and entities to standard formats.

The ContentNormalizerImpl class is responsible for cleaning and standardizing data
extracted by the extraction framework. It implements various normalization techniques:

1. Text Normalization:
   - Whitespace cleaning and standardization
   - HTML tag removal
   - Character encoding normalization
   - Quote and dash standardization
   - Control character removal

2. Date/Time Normalization:
   - Converting various date formats to ISO 8601
   - Extracting date components (year, month, day)
   - Handling various regional date formats

3. Unit Conversion:
   - Standardizing measurements (length, weight, volume)
   - Converting between metric and imperial units
   - Currency normalization

4. Entity Normalization:
   - Person name parsing
   - Address standardization
   - Phone number formatting
   - URL normalization

5. Type-specific Normalization:
   - Boolean value standardization
   - Price extraction and currency detection
   - List and object recursive normalization
   
The normalizer automatically detects the type of data it's processing and applies
appropriate normalization techniques.
"""

import re
import html
import json
import logging
import unicodedata
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
try:
    import dateutil.parser
except ImportError:
    logging.warning("dateutil.parser not available, date parsing will be limited")
    
try:
    from bs4 import BeautifulSoup
except ImportError:
    logging.warning("BeautifulSoup not available, HTML cleaning will use basic regex")

from extraction.core.extraction_interface import ContentNormalizer
from core.service_interface import BaseService

# Configure logging
logger = logging.getLogger(__name__)

class ContentNormalizerImpl(ContentNormalizer, BaseService):
    """
    Implementation of ContentNormalizer interface.
    
    This class provides methods for cleaning and standardizing extracted data,
    including text normalization, date parsing, unit conversion, and entity recognition
    to ensure consistent output format.
    """
    
    def __init__(self, context=None):
        """Initialize the content normalizer."""
        super().__init__(context)
        self._initialized = False
        self._locale = "en_US"
        self._target_unit_system = "metric"
        self._target_date_format = "%Y-%m-%d"
        self._target_currency = "USD"
        self._remove_html = True
        self._normalize_whitespace = True
        self._standardize_dates = True
        self._standardize_prices = True
        
        # Patterns for value detection
        self._date_patterns = [
            r'\d{4}-\d{1,2}-\d{1,2}',                      # ISO date (YYYY-MM-DD)
            r'\d{1,2}/\d{1,2}/\d{2,4}',                    # US date (MM/DD/YYYY)
            r'\d{1,2}-\d{1,2}-\d{2,4}',                    # Date with dashes
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # 15 Jan 2023
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'  # Jan 15, 2023
        ]
        
        self._price_patterns = [
            r'[$€£¥]\s*\d+(?:[,.]\d+)*',                  # Currency symbol followed by number
            r'\d+(?:[,.]\d+)*\s*[$€£¥]',                  # Number followed by currency symbol
            r'\d+(?:[,.]\d+)*\s*(?:USD|EUR|GBP|JPY|CAD)'  # Number followed by currency code
        ]
        
        self._measurement_patterns = {
            'length': [
                r'\d+(?:[,.]\d+)*\s*(?:m|cm|mm|km|in|inch|inches|ft|foot|feet|yd|yard|yards|mi|mile|miles)',
            ],
            'weight': [
                r'\d+(?:[,.]\d+)*\s*(?:kg|g|mg|lb|lbs|pound|pounds|oz|ounce|ounces)',
            ],
            'volume': [
                r'\d+(?:[,.]\d+)*\s*(?:l|ml|gal|gallon|gallons|qt|quart|quarts|pt|pint|pints|fl oz|fluid ounce|fluid ounces)',
            ],
            'temperature': [
                r'\d+(?:[,.]\d+)*\s*(?:°C|°F|deg C|deg F|degree Celsius|degree Fahrenheit|degrees Celsius|degrees Fahrenheit)',
            ]
        }
        
        # Unit conversion dictionaries
        self._unit_conversions = {
            'length': {
                # To meters
                'mm': 0.001,
                'cm': 0.01,
                'm': 1.0,
                'km': 1000.0,
                'in': 0.0254,
                'inch': 0.0254,
                'inches': 0.0254,
                'ft': 0.3048,
                'foot': 0.3048,
                'feet': 0.3048,
                'yd': 0.9144,
                'yard': 0.9144,
                'yards': 0.9144,
                'mi': 1609.34,
                'mile': 1609.34,
                'miles': 1609.34
            },
            'weight': {
                # To kilograms
                'mg': 0.000001,
                'g': 0.001,
                'kg': 1.0,
                'oz': 0.0283495,
                'ounce': 0.0283495,
                'ounces': 0.0283495,
                'lb': 0.453592,
                'lbs': 0.453592,
                'pound': 0.453592,
                'pounds': 0.453592
            },
            'volume': {
                # To liters
                'ml': 0.001,
                'l': 1.0,
                'fl oz': 0.0295735,
                'fluid ounce': 0.0295735,
                'fluid ounces': 0.0295735,
                'pt': 0.473176,
                'pint': 0.473176,
                'pints': 0.473176,
                'qt': 0.946353,
                'quart': 0.946353,
                'quarts': 0.946353,
                'gal': 3.78541,
                'gallon': 3.78541,
                'gallons': 3.78541
            }
        }
        
        # Currency symbols
        self._currency_symbols = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            '₽': 'RUB',
            '₩': 'KRW',
            '฿': 'THB',
            '₫': 'VND',
            '₴': 'UAH',
            '₺': 'TRY'
        }
        
        # Boolean values mapping
        self._boolean_mapping = {
            'true': True,
            'yes': True,
            'y': True,
            '1': True,
            'on': True,
            'enabled': True,
            'false': False,
            'no': False,
            'n': False,
            '0': False,
            'off': False,
            'disabled': False
        }
        
    @property
    def name(self) -> str:
        """Return the service name."""
        return "content_normalizer"
    
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this normalizer can handle the given content.
        
        Args:
            content: Content to check compatibility with
            content_type: Optional hint about the content type (e.g., "html", "json", "text")
            
        Returns:
            True if the normalizer can handle this content, False otherwise
        """
        # ContentNormalizer can handle most data types for normalization
        if content is None:
            return False
            
        # Handle dictionaries, strings, lists
        if isinstance(content, (dict, str, list)):
            return True
            
        # Handle specific content types
        if content_type in ["html", "json", "text", "product", "article", "listing"]:
            return True
            
        return False
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the normalizer with configuration.
        
        Args:
            config: Optional configuration dictionary with settings
        """
        if config:
            self._locale = config.get('locale', self._locale)
            self._target_unit_system = config.get('unit_system', self._target_unit_system)
            self._target_date_format = config.get('date_format', self._target_date_format)
            self._target_currency = config.get('currency', self._target_currency)
            self._remove_html = config.get('remove_html', self._remove_html)
            self._normalize_whitespace = config.get('normalize_whitespace', self._normalize_whitespace)
            self._standardize_dates = config.get('standardize_dates', self._standardize_dates)
            self._standardize_prices = config.get('standardize_prices', self._standardize_prices)
            
        self._initialized = True
        logger.info(f"ContentNormalizer initialized with locale: {self._locale}, unit system: {self._target_unit_system}")
        
    def shutdown(self) -> None:
        """Clean up resources."""
        self._initialized = False
    
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract and normalize data.
        
        Args:
            content: The content to extract from
            options: Optional extraction configuration
            
        Returns:
            Normalized data dictionary
        """
        if isinstance(content, dict):
            return self.normalize(content, options)
        elif isinstance(content, str):
            # Try to parse as JSON if it's a string
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return self.normalize(data, options)
            except json.JSONDecodeError:
                # Not JSON, normalize as text
                pass
                
            # Default to text normalization
            return {"content": self.normalize_text(content)}
        
        return {"data": content}
    
    def normalize(self, data: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize a data dictionary by standardizing all values.
        
        Args:
            data: Dictionary of data to normalize
            options: Optional normalization configuration
            
        Returns:
            Normalized data dictionary
        """
        if not data:
            return data
            
        options = options or {}
        normalized = {}
        
        # Add metadata field if not present
        if "_metadata" not in normalized:
            normalized["_metadata"] = {}
            
        # Copy metadata from input if available
        if "_metadata" in data:
            normalized["_metadata"].update(data["_metadata"])
        
        # Add normalization metadata
        normalized["_metadata"]["normalized"] = True
        normalized["_metadata"]["normalization_time"] = datetime.now().isoformat()
        
        # Normalize each field
        for key, value in data.items():
            if key == "_metadata":
                continue
                
            try:
                # Determine field type
                if isinstance(value, str):
                    # Check for different types of string data
                    if self._is_date(value):
                        normalized[key] = self._normalize_date_field(value)
                    elif self._is_price(value):
                        normalized[key] = self._normalize_price_field(value)
                    elif self._is_measurement(value):
                        normalized[key] = self._normalize_measurement_field(value)
                    elif self._is_email(value):
                        normalized[key] = {"type": "email", "value": self.normalize_text(value)}
                    elif self._is_phone(value):
                        normalized[key] = self._normalize_phone_field(value)
                    elif self._is_url(value):
                        normalized[key] = self._normalize_url_field(value)
                    elif self._is_boolean_string(value):
                        normalized[key] = self._normalize_boolean_field(value)
                    else:
                        # Default to text normalization
                        normalized[key] = self.normalize_text(value)
                elif isinstance(value, (int, float)):
                    normalized[key] = value
                elif isinstance(value, bool):
                    normalized[key] = value
                elif isinstance(value, list):
                    normalized[key] = self._normalize_list(value)
                elif isinstance(value, dict):
                    normalized[key] = self.normalize(value, options)
                else:
                    # Keep as is
                    normalized[key] = value
            except Exception as e:
                logger.warning(f"Error normalizing field '{key}': {str(e)}")
                normalized[key] = value
                
        return normalized

    def normalize_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Normalize text content by cleaning whitespace, removing control characters,
        and standardizing quotes and dashes.
        
        Args:
            text: Text to normalize
            options: Optional normalization configuration
            
        Returns:
            Normalized text
        """
        if text is None:
            return ""
            
        if not isinstance(text, str):
            return str(text)
            
        # Convert to string and normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove HTML if configured
        if self._remove_html or (options and options.get("remove_html", False)):
            text = self._strip_html(text)
            
        # Clean up whitespace
        if self._normalize_whitespace or (options and options.get("normalize_whitespace", False)):
            text = self._clean_whitespace(text)
            
        # Standardize quotes and dashes
        text = self._standardize_quotes(text)
        text = self._standardize_dashes(text)
        
        # Remove control characters
        text = self._remove_control_chars(text)
        
        return text
    
    def normalize_datetime(self, date_str: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Normalize date and time strings to ISO format.
        
        Args:
            date_str: Date/time string to normalize
            options: Optional normalization configuration
            
        Returns:
            Normalized date/time string in ISO format or original string if parsing fails
        """
        if not date_str:
            return date_str
            
        if not isinstance(date_str, str):
            date_str = str(date_str)
            
        try:
            # Parse the date using dateutil
            dt = dateutil.parser.parse(date_str)
            # Format according to ISO 8601
            return dt.isoformat()
        except (ValueError, TypeError):
            logger.warning(f"Failed to parse date: {date_str}")
            return date_str
    
    def normalize_units(self, value: Union[float, int, str], 
                 unit: str, target_unit: Optional[str] = None,
                 options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize units to standard units based on unit type.
        
        Args:
            value: Value to normalize
            unit: Unit of measurement (e.g., "kg", "inches")
            target_unit: Optional target unit for conversion
            options: Optional normalization configuration
            
        Returns:
            Dictionary with normalized value and unit
        """
        result = {
            "valid": False,
            "original_value": value,
            "original_unit": unit,
            "normalized_value": None,
            "normalized_unit": None
        }
        
        # Determine unit type
        unit_type = None
        for type_name, conversions in self._unit_conversions.items():
            if unit.lower() in conversions:
                unit_type = type_name
                break
                
        if not unit_type:
            return result
            
        # Set unit type in result
        result["unit_type"] = unit_type
        
        # Get conversion factor
        conversion_factor = self._get_conversion_factor(unit, unit_type)
        if not conversion_factor:
            return result
            
        # Convert to base unit
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return result
                
        # Normalize to base unit
        base_unit = "m" if unit_type == "length" else "kg" if unit_type == "weight" else "l"
        normalized_value = value * conversion_factor
        
        result["valid"] = True
        result["normalized_value"] = normalized_value
        result["normalized_unit"] = base_unit
        
        # Convert to target unit if specified
        if target_unit and target_unit in self._unit_conversions.get(unit_type, {}):
            target_conversion = self._unit_conversions[unit_type][target_unit]
            if target_conversion:
                target_value = normalized_value / target_conversion
                result["target_value"] = target_value
                result["target_unit"] = target_unit
        
        return result
    
    def normalize_entity(self, entity: str, entity_type: str) -> Dict[str, Any]:
        """
        Normalize entity references based on entity type.
        
        Args:
            entity: Entity string to normalize
            entity_type: Type of entity (e.g., "person", "organization", "location")
            
        Returns:
            Dictionary with structured entity data
        """
        result = {
            "original": entity,
            "type": entity_type,
            "valid": False
        }
        
        if not entity:
            return result
            
        # Clean the entity text
        cleaned_entity = self.normalize_text(entity)
        result["normalized"] = cleaned_entity
        
        if entity_type == "person":
            # Person name parsing
            parts = cleaned_entity.split()
            if len(parts) == 1:
                # Just a single name
                result["name"] = parts[0]
                result["valid"] = True
            elif len(parts) == 2:
                # First and last name
                result["first_name"] = parts[0]
                result["last_name"] = parts[1]
                result["valid"] = True
            elif len(parts) > 2:
                # Multiple parts, try to parse title, first, middle, last
                if parts[0].lower() in ["mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "prof", "prof."]:
                    result["title"] = parts[0]
                    parts = parts[1:]
                    
                if len(parts) == 2:
                    result["first_name"] = parts[0]
                    result["last_name"] = parts[1]
                elif len(parts) >= 3:
                    result["first_name"] = parts[0]
                    result["middle_name"] = " ".join(parts[1:-1])
                    result["last_name"] = parts[-1]
                result["valid"] = True
                
        elif entity_type == "organization":
            # Organization parsing is simpler
            result["name"] = cleaned_entity
            result["valid"] = True
            
        elif entity_type == "location":
            # Location parsing
            # This could be expanded with address parsing, geocoding, etc.
            result["name"] = cleaned_entity
            
            # Basic pattern for US addresses
            address_match = re.search(r'(\d+\s+[^,]+),\s*([^,]+),\s*([A-Z]{2})\s*(\d{5})?', cleaned_entity)
            if address_match:
                street, city, state, zipcode = address_match.groups()
                result["street"] = street
                result["city"] = city
                result["state"] = state
                if zipcode:
                    result["postal_code"] = zipcode
                result["country"] = "USA"  # Assuming US address format
            
            result["valid"] = True
            
        return result
        
    # Helper methods for normalization
    
    def _normalize_date_field(self, date_str: str) -> Dict[str, Any]:
        """Normalize a date field to a standard format."""
        result = {
            "type": "date",
            "original": date_str,
            "valid": False
        }
        
        try:
            dt = dateutil.parser.parse(date_str)
            result["valid"] = True
            result["iso"] = dt.isoformat()
            result["year"] = dt.year
            result["month"] = dt.month
            result["day"] = dt.day
            
            if dt.hour != 0 or dt.minute != 0 or dt.second != 0:
                result["has_time"] = True
                result["hour"] = dt.hour
                result["minute"] = dt.minute
                result["second"] = dt.second
            
            return result
        except (ValueError, TypeError):
            return result
    
    def _normalize_price_field(self, price_str: str) -> Dict[str, Any]:
        """Normalize a price field to a standard format with currency information."""
        result = {
            "type": "price",
            "original": price_str,
            "valid": False
        }
        
        # Extract currency symbol or code
        currency = self._extract_currency_code(price_str)
        
        # Extract numeric value
        numeric_value = self._extract_numeric_value(price_str)
        
        if numeric_value is not None:
            result["valid"] = True
            result["value"] = numeric_value
            result["currency"] = currency or self._target_currency
            
            # Convert currency if needed
            if currency and currency != self._target_currency:
                result["target_currency"] = self._target_currency
                # In a real system, would use exchange rates here
                
        return result
    
    def _normalize_measurement_field(self, measurement_str: str) -> Dict[str, Any]:
        """Normalize a measurement field to standard units."""
        result = {
            "type": "measurement",
            "original": measurement_str,
            "valid": False
        }
        
        numeric_value, unit = self._extract_numeric_and_unit(measurement_str)
        
        if numeric_value is not None and unit:
            result["valid"] = True
            result["value"] = numeric_value
            result["unit"] = unit.lower()
            
            # Determine unit type
            unit_type = None
            for type_name, conversions in self._unit_conversions.items():
                if unit.lower() in conversions:
                    unit_type = type_name
                    break
                    
            if unit_type:
                result["unit_type"] = unit_type
                
                # Convert to base unit
                base_unit = "m" if unit_type == "length" else "kg" if unit_type == "weight" else "l"
                conversion_factor = self._get_conversion_factor(unit, unit_type)
                
                if conversion_factor:
                    normalized_value = numeric_value * conversion_factor
                    result["normalized_value"] = normalized_value
                    result["normalized_unit"] = base_unit
                    
        return result
    
    def _normalize_phone_field(self, phone_str: str) -> Dict[str, Any]:
        """Normalize a phone number to a standard format."""
        result = {
            "type": "phone",
            "original": phone_str,
            "valid": False
        }
        
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone_str)
        
        if len(digits) >= 10:
            result["valid"] = True
            result["digits"] = digits
            
            # US format parsing
            if len(digits) == 10:
                result["country_code"] = "1"  # Default to US
                result["area_code"] = digits[0:3]
                result["formatted"] = f"({result['area_code']}) {digits[3:6]}-{digits[6:10]}"
            elif len(digits) > 10:
                # International number
                if digits.startswith("1"):
                    result["country_code"] = "1"
                    result["area_code"] = digits[1:4]
                    local = digits[4:]
                    result["formatted"] = f"+1 ({result['area_code']}) {local[:3]}-{local[3:7]}" if len(local) >= 7 else f"+1 ({result['area_code']}) {local}"
                else:
                    # Try to extract country code
                    # In a real system, would use a library like phonenumbers
                    result["country_code"] = digits[:2]
                    result["formatted"] = f"+{result['country_code']} {digits[2:]}"
                    
        return result
    
    def _normalize_url_field(self, url_str: str) -> Dict[str, Any]:
        """Normalize a URL to a standard format."""
        result = {
            "type": "url",
            "original": url_str,
            "valid": False
        }
        
        # Basic URL validation and normalization
        url_match = re.match(r'^(?:(?:https?|ftp):\/\/)?([^\/\s]+)(\/[^\s]*)?$', url_str.strip())
        
        if url_match:
            domain, path = url_match.groups()
            
            # Add scheme if missing
            scheme = "https"
            if "://" in url_str:
                scheme_match = re.match(r'^(https?|ftp)', url_str)
                if scheme_match:
                    scheme = scheme_match.group(1)
            
            result["valid"] = True
            result["scheme"] = scheme
            result["domain"] = domain
            result["path"] = path or "/"
            result["normalized"] = f"{scheme}://{domain}{path or '/'}"
            
        return result
    
    def _normalize_boolean_field(self, bool_str: str) -> bool:
        """Normalize a string to a boolean value."""
        if isinstance(bool_str, bool):
            return bool_str
            
        if isinstance(bool_str, (int, float)):
            return bool(bool_str)
            
        if not isinstance(bool_str, str):
            return False
            
        bool_str = bool_str.lower().strip()
        return self._boolean_mapping.get(bool_str, False)
    
    def _normalize_list(self, items: List[Any]) -> List[Any]:
        """Normalize a list of items."""
        if not items:
            return []
            
        normalized = []
        
        for item in items:
            if item is None:
                continue
                
            if isinstance(item, str):
                normalized.append(self.normalize_text(item))
            elif isinstance(item, dict):
                normalized.append(self.normalize(item))
            elif isinstance(item, list):
                normalized.append(self._normalize_list(item))
            else:
                normalized.append(item)
                
        return normalized
    
    def _strip_html(self, html_text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            html_text: HTML content to clean
            
        Returns:
            Plain text with HTML tags removed
        """
        if not html_text:
            return ""
            
        try:
            # Try to use BeautifulSoup for better HTML parsing
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text and normalize whitespace
            text = soup.get_text()
            return self._clean_whitespace(text)
        except ImportError:
            # Fallback to regex-based approach if BeautifulSoup is not available
            # First remove script and style blocks
            text = re.sub(r'<script[^>]*>.*?</script>', ' ', html_text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL)
            
            # Then remove all HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            
            # Decode HTML entities
            text = html.unescape(text)
            
            return self._clean_whitespace(text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace in text."""
        if not text:
            return ""
            
        # Replace tabs, newlines, etc. with spaces
        text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Trim leading/trailing whitespace
        return text.strip()
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters from text."""
        if not text:
            return ""
            
        # Remove control characters except tab, newline, and carriage return
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    def _standardize_quotes(self, text: str) -> str:
        """Standardize various quote styles to normal quotes."""
        if not text:
            return ""
            
        # Replace fancy quotes with straight quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def _standardize_dashes(self, text: str) -> str:
        """Standardize various dash styles."""
        if not text:
            return ""
            
        # Replace em dash and en dash with standard hyphen
        text = text.replace('—', '--').replace('–', '-')
        
        return text
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract a numeric value from text."""
        if not text:
            return None
            
        # Find all numbers in the string
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        
        if not numbers:
            return None
            
        # Try to convert the first match to a float
        try:
            return float(numbers[0])
        except ValueError:
            return None
    
    def _extract_numeric_and_unit(self, text: str) -> tuple:
        """
        Extract numeric value and unit from text.
        
        Returns:
            Tuple of (numeric_value, unit)
        """
        if not text:
            return None, None
            
        # Match pattern like "10 kg", "5.5 inches", etc.
        match = re.match(r'([-+]?\d*\.?\d+)\s*([a-zA-Z°]+.*)', text.strip())
        
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).strip()
                return value, unit
            except ValueError:
                pass
                
        return None, None
    
    def _extract_currency_code(self, text: str) -> Optional[str]:
        """Extract currency code or symbol from text."""
        if not text:
            return None
            
        # Check for currency symbols
        for symbol, code in self._currency_symbols.items():
            if symbol in text:
                return code
                
        # Check for currency codes
        currency_codes = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY"]
        for code in currency_codes:
            if code in text.upper():
                return code
                
        return None
    
    def _get_conversion_factor(self, unit: str, unit_type: str) -> Optional[float]:
        """Get the conversion factor for a unit to its base unit."""
        if not unit or not unit_type:
            return None
            
        unit = unit.lower()
        if unit_type in self._unit_conversions and unit in self._unit_conversions[unit_type]:
            return self._unit_conversions[unit_type][unit]
            
        return None
    
    def _is_date(self, text: str) -> bool:
        """Check if text appears to be a date."""
        for pattern in self._date_patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def _is_price(self, text: str) -> bool:
        """Check if text appears to be a price."""
        for pattern in self._price_patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def _is_measurement(self, text: str) -> bool:
        """Check if text appears to be a measurement with units."""
        for unit_type, patterns in self._measurement_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return True
                    
        return False
    
    def _is_email(self, text: str) -> bool:
        """Check if text appears to be an email address."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, text))
    
    def _is_phone(self, text: str) -> bool:
        """Check if text appears to be a phone number."""
        # Various phone number patterns
        phone_patterns = [
            r'\(\d{3}\)\s*\d{3}[-\s]?\d{4}',  # (123) 456-7890
            r'\d{3}[-\s]?\d{3}[-\s]?\d{4}',    # 123-456-7890
            r'\+\d{1,3}\s*\(\d{3}\)\s*\d{3}[-\s]?\d{4}',  # +1 (123) 456-7890
            r'\+\d{1,3}\s*\d{3}\s*\d{3}\s*\d{4}'  # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            if re.search(pattern, text):
                return True
                
        return False
    
    def _is_url(self, text: str) -> bool:
        """Check if text appears to be a URL."""
        url_pattern = r'^(?:(?:https?|ftp):\/\/)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:\/.*)?$'
        return bool(re.match(url_pattern, text.strip()))
    
    def _is_boolean_string(self, text: str) -> bool:
        """Check if text appears to be a boolean value."""
        if not isinstance(text, str):
            return False
            
        boolean_strings = ["true", "false", "yes", "no", "y", "n", "1", "0", "on", "off", "enabled", "disabled"]
        return text.lower().strip() in boolean_strings
