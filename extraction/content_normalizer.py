"""
Content Normalization Module

This module provides functionality for cleaning and standardizing extracted data,
including text normalization, date parsing, unit conversion, and more to ensure
consistent high-quality data output.
"""

import re
import html
import json
import logging
from typing import Dict, List, Any, Optional, Union
from bs4 import BeautifulSoup
import unicodedata
from datetime import datetime

from extraction.core.extraction_interface import ContentNormalizer as BaseContentNormalizer
from extraction.helpers.normalization_utils import (
    # Text utilities
    clean_whitespace, remove_control_chars, standardize_quotes, standardize_dashes,
    clean_html_fragments, collapse_newlines,
    # Date utilities
    parse_date, detect_date_format, convert_to_iso, extract_date_parts, standardize_date_separators,
    # Price and unit utilities
    extract_currency_symbol, extract_numeric_value, convert_currency, 
    standardize_units, detect_unit_system
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContentNormalizer")

class ContentNormalizer(BaseContentNormalizer):
    """
    Normalizes and standardizes extracted data for consistent output format.
    
    This class provides methods for cleaning text, standardizing dates,
    normalizing prices, units, and other common data types extracted from
    various sources.
    """
    
    def __init__(self, context=None):
        """
        Initialize the content normalizer.
        
        Args:
            context: Strategy context for accessing shared services
        """
        super().__init__(context)
        self._locale = "en_US"
        self._target_unit_system = "metric"
        self._target_date_format = "iso"
        self._target_currency = "USD"
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the normalizer with configuration."""
        if self._initialized:
            return
            
        super().initialize(config)
        
        # Configure normalization settings
        if config:
            self._locale = config.get("locale", self._locale)
            self._target_unit_system = config.get("unit_system", self._target_unit_system)
            self._target_date_format = config.get("date_format", self._target_date_format)
            self._target_currency = config.get("currency", self._target_currency)
    
    def can_handle(self, data: Any, data_type: Optional[str] = None) -> bool:
        """
        Check if this normalizer can handle the given data type.
        
        Args:
            data: Data to check
            data_type: Optional data type hint
            
        Returns:
            True if the normalizer can handle this data, False otherwise
        """
        # Can handle most data types
        return True
    
    def normalize(self, extraction_result: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Normalize an extraction result by cleaning and standardizing all values.
        
        Args:
            extraction_result: Dictionary containing extracted data
            options: Optional normalization options
            
        Returns:
            Normalized extraction result
        """
        if not extraction_result:
            return extraction_result
            
        options = options or {}
        
        # Apply normalization based on options and field types
        normalized = {}
        
        # Copy metadata
        if "_metadata" in extraction_result:
            normalized["_metadata"] = extraction_result["_metadata"].copy()
        else:
            normalized["_metadata"] = {}
            
        # Add normalization metadata
        normalized["_metadata"]["normalized"] = True
        normalized["_metadata"]["normalization_timestamp"] = datetime.now().isoformat()
        
        # Get schema if provided for type-aware normalization
        schema = options.get("schema", {})
        
        # Process each field
        for key, value in extraction_result.items():
            if key.startswith("_"):
                # Skip metadata fields
                continue
                
            field_type = self._get_field_type(key, value, schema)
            field_options = schema.get(key, {}) if schema else {}
            
            try:
                # Normalize based on field type
                if field_type == "text" or field_type == "string":
                    normalized[key] = self.normalize_text(value)
                elif field_type == "html":
                    normalized[key] = self.normalize_html(value)
                elif field_type == "date":
                    normalized[key] = self.normalize_date(value, field_options)
                elif field_type == "price":
                    normalized[key] = self.normalize_price(value, field_options)
                elif field_type == "measurement":
                    normalized[key] = self.normalize_measurement(value, field_options)
                elif field_type == "boolean":
                    normalized[key] = self.normalize_boolean(value)
                elif field_type == "list":
                    normalized[key] = self.normalize_list(value, field_options)
                elif field_type == "object" or field_type == "dict":
                    normalized[key] = self.normalize_object(value, field_options)
                elif field_type == "name":
                    normalized[key] = self.normalize_names(value)
                elif field_type == "address":
                    normalized[key] = self.normalize_addresses(value)
                elif field_type == "phone":
                    normalized[key] = self.normalize_phone_numbers(value)
                elif field_type == "url":
                    normalized[key] = self.normalize_urls(value)
                elif field_type == "identifier":
                    normalized[key] = self.normalize_identifiers(value, field_options)
                else:
                    # Default to basic text normalization
                    normalized[key] = self.normalize_text(value) if isinstance(value, str) else value
            except Exception as e:
                logger.warning(f"Error normalizing field '{key}': {str(e)}")
                # Keep original value on error
                normalized[key] = value
                # Add normalization error to metadata
                if "_normalization_errors" not in normalized["_metadata"]:
                    normalized["_metadata"]["_normalization_errors"] = {}
                normalized["_metadata"]["_normalization_errors"][key] = str(e)
        
        return normalized
    
    def normalize_text(self, text: Any) -> str:
        """
        Clean and standardize text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if text is None:
            return ""
            
        if not isinstance(text, str):
            text = str(text)
            
        # Apply a series of text normalizations
        result = text
        result = self.normalize_html_entities(result)
        result = remove_control_chars(result)
        result = standardize_quotes(result)
        result = standardize_dashes(result)
        result = clean_html_fragments(result)
        result = self.normalize_whitespace(result)
        
        return result
    
    def normalize_html(self, html_content: Any) -> str:
        """
        Clean and standardize HTML content.
        
        Args:
            html_content: HTML content to normalize
            
        Returns:
            Normalized HTML
        """
        if html_content is None:
            return ""
            
        if not isinstance(html_content, str):
            html_content = str(html_content)
            
        try:
            # Parse and re-render HTML to ensure proper structure
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Remove comments
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()
                
            # Fix common issues
            # - Ensure all tags are properly closed
            # - Remove empty tags
            # - Normalize attributes
            
            # Return clean HTML
            return str(soup)
        except Exception as e:
            logger.warning(f"Error normalizing HTML: {str(e)}")
            # Fallback to basic cleaning
            return clean_html_fragments(html_content)
    
    def normalize_date(self, date_str: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert date to standard format.
        
        Args:
            date_str: Date string to normalize
            options: Optional normalization options
            
        Returns:
            Dictionary with normalized date data
        """
        options = options or {}
        
        # Parse the date
        date_obj = parse_date(date_str)
        
        if not date_obj:
            # Could not parse date
            return {
                "original": date_str,
                "iso": None,
                "valid": False
            }
            
        # Determine output format
        output_format = options.get("format", self._target_date_format)
        
        result = {
            "original": date_str,
            "iso": convert_to_iso(date_obj),
            "valid": True
        }
        
        # Add parts
        if options.get("include_parts", False):
            result.update(extract_date_parts(date_obj))
            
        # Format according to specified output format
        if output_format != "iso":
            try:
                result["formatted"] = date_obj.strftime(output_format)
            except Exception:
                # Invalid format string, default to ISO
                result["formatted"] = result["iso"]
        else:
            result["formatted"] = result["iso"]
        
        return result
    
    def normalize_price(self, price_str: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standardize price and currency format.
        
        Args:
            price_str: Price string to normalize
            options: Optional normalization options
            
        Returns:
            Dictionary with normalized price data
        """
        options = options or {}
        
        # Extract currency and numeric value
        currency = extract_currency_symbol(price_str) if isinstance(price_str, str) else None
        value = extract_numeric_value(price_str)
        
        if value is None:
            # Could not extract numeric value
            return {
                "original": price_str,
                "value": None,
                "currency": currency,
                "valid": False
            }
            
        # Determine target currency
        target_currency = options.get("currency", self._target_currency)
        
        result = {
            "original": price_str,
            "value": value,
            "currency": currency or "UNKNOWN",
            "valid": True
        }
        
        # Convert to target currency if needed
        if currency and currency != target_currency:
            converted = convert_currency(value, currency, target_currency, 
                                      options.get("exchange_rates"))
            if converted is not None:
                result["converted_value"] = converted
                result["converted_currency"] = target_currency
        
        # Format according to locale
        try:
            import locale
            locale.setlocale(locale.LC_ALL, self._locale)
            result["formatted"] = locale.currency(
                result.get("converted_value", value), 
                symbol=target_currency,
                grouping=True
            )
        except Exception:
            # Fallback to basic formatting
            currency_symbol = target_currency
            if target_currency == "USD":
                currency_symbol = "$"
            elif target_currency == "EUR":
                currency_symbol = "€"
            elif target_currency == "GBP":
                currency_symbol = "£"
                
            amount = result.get("converted_value", value)
            result["formatted"] = f"{currency_symbol}{amount:.2f}"
        
        return result
    
    def normalize_measurement(self, value_str: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standardize measurement units.
        
        Args:
            value_str: Measurement string to normalize
            options: Optional normalization options
            
        Returns:
            Dictionary with normalized measurement data
        """
        options = options or {}
        
        # For string input, try to extract numeric value and unit
        if isinstance(value_str, str):
            # Try to match patterns like "10 kg", "5.2 cm", etc.
            match = re.match(r'([\d,.]+)\s*([a-zA-Z°]+)', value_str)
            if match:
                value = extract_numeric_value(match.group(1))
                unit = match.group(2).lower()
            else:
                # No clear unit, try to extract just the numeric value
                value = extract_numeric_value(value_str)
                unit = options.get("unit", "")
        else:
            # Numeric input, use unit from options
            value = extract_numeric_value(value_str)
            unit = options.get("unit", "")
            
        if value is None:
            # Could not extract numeric value
            return {
                "original": value_str,
                "value": None,
                "unit": unit,
                "valid": False
            }
            
        # Determine target unit system
        target_system = options.get("unit_system", self._target_unit_system)
        
        result = {
            "original": value_str,
            "value": value,
            "unit": unit,
            "valid": True
        }
        
        # Convert to target unit system if needed
        if unit:
            conversion = standardize_units(value, unit, target_system)
            if conversion:
                result.update(conversion)
        
        return result
    
    def normalize_boolean(self, value: Any) -> bool:
        """
        Standardize boolean values.
        
        Args:
            value: Value to normalize to boolean
            
        Returns:
            Normalized boolean value
        """
        if isinstance(value, bool):
            return value
            
        if isinstance(value, (int, float)):
            return bool(value)
            
        if isinstance(value, str):
            value_lower = value.lower().strip()
            true_values = ["true", "yes", "1", "y", "t", "on", "enable", "enabled", "✓", "✔", "✅"]
            false_values = ["false", "no", "0", "n", "f", "off", "disable", "disabled", "×", "✗", "✘", "❌"]
            
            if value_lower in true_values:
                return True
            elif value_lower in false_values:
                return False
                
            # Try to cast to int and then to bool
            try:
                return bool(int(value))
            except (ValueError, TypeError):
                pass
        
        # Default conversion
        return bool(value)
    
    def normalize_object(self, obj: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recursively normalize objects/dictionaries.
        
        Args:
            obj: Object/dictionary to normalize
            options: Optional normalization options
            
        Returns:
            Normalized object/dictionary
        """
        options = options or {}
        
        if not isinstance(obj, dict):
            if isinstance(obj, str):
                # Try to parse as JSON
                try:
                    obj = json.loads(obj)
                except json.JSONDecodeError:
                    # Not valid JSON, return as is
                    return obj
            else:
                # Not a dictionary or JSON string, return as is
                return obj
                
        # Recursively normalize each field
        normalized = {}
        
        for key, value in obj.items():
            if key.startswith("_"):
                # Copy metadata fields as is
                normalized[key] = value
                continue
                
            field_options = options.get(key, {})
            
            if isinstance(value, dict):
                # Recursively normalize nested objects
                normalized[key] = self.normalize_object(value, field_options)
            elif isinstance(value, list):
                # Normalize list items
                normalized[key] = self.normalize_list(value, field_options)
            else:
                # Normalize single value based on field type
                field_type = field_options.get("type", self._infer_field_type(value))
                
                if field_type == "text" or field_type == "string":
                    normalized[key] = self.normalize_text(value)
                elif field_type == "date":
                    normalized[key] = self.normalize_date(value, field_options)
                elif field_type == "price":
                    normalized[key] = self.normalize_price(value, field_options)
                elif field_type == "measurement":
                    normalized[key] = self.normalize_measurement(value, field_options)
                elif field_type == "boolean":
                    normalized[key] = self.normalize_boolean(value)
                elif field_type == "name":
                    normalized[key] = self.normalize_names(value)
                elif field_type == "address":
                    normalized[key] = self.normalize_addresses(value)
                elif field_type == "phone":
                    normalized[key] = self.normalize_phone_numbers(value)
                elif field_type == "url":
                    normalized[key] = self.normalize_urls(value)
                elif field_type == "identifier":
                    normalized[key] = self.normalize_identifiers(value, field_options)
                else:
                    # Default to basic text normalization for strings
                    normalized[key] = self.normalize_text(value) if isinstance(value, str) else value
        
        return normalized
    
    def normalize_names(self, name_str: Any) -> Dict[str, Any]:
        """
        Standardize person names.
        
        Args:
            name_str: Name string to normalize
            
        Returns:
            Dictionary with normalized name components
        """
        if name_str is None:
            return {"original": None, "full": None, "valid": False}
            
        if not isinstance(name_str, str):
            name_str = str(name_str)
            
        # Clean the name
        clean_name = self.normalize_text(name_str)
        
        # Basic result
        result = {
            "original": name_str,
            "full": clean_name,
            "valid": bool(clean_name)
        }
        
        if not clean_name:
            return result
            
        # Try to split name into parts
        name_parts = clean_name.split()
        
        if len(name_parts) == 1:
            # Just one word, assume it's a first name
            result["first"] = name_parts[0]
        elif len(name_parts) == 2:
            # Two words, assume first and last name
            result["first"] = name_parts[0]
            result["last"] = name_parts[1]
        elif len(name_parts) >= 3:
            # More words, try to identify parts
            result["first"] = name_parts[0]
            
            # Check for common prefixes/titles
            titles = ["mr", "mrs", "miss", "ms", "dr", "prof", "professor", "rev", "reverend"]
            if name_parts[0].lower().rstrip('.') in titles:
                result["title"] = name_parts[0]
                result["first"] = name_parts[1]
                
                if len(name_parts) >= 4:
                    result["middle"] = ' '.join(name_parts[2:-1])
                    result["last"] = name_parts[-1]
                else:
                    result["last"] = name_parts[2]
            else:
                # No title, assume first, middle, last
                result["middle"] = ' '.join(name_parts[1:-1])
                result["last"] = name_parts[-1]
                
        # Check for suffix
        suffixes = ["jr", "sr", "ii", "iii", "iv", "v"]
        if "last" in result and result["last"].lower().rstrip('.') in suffixes:
            # Last part is actually a suffix
            if "middle" in result:
                parts = result["middle"].split()
                if parts:
                    result["suffix"] = result["last"]
                    result["last"] = parts[-1]
                    if len(parts) > 1:
                        result["middle"] = ' '.join(parts[:-1])
                    else:
                        del result["middle"]
            else:
                result["suffix"] = result["last"]
                result["last"] = result["first"]
                del result["first"]
        
        return result
    
    def normalize_addresses(self, address_str: Any) -> Dict[str, Any]:
        """
        Standardize postal addresses.
        
        Args:
            address_str: Address string to normalize
            
        Returns:
            Dictionary with normalized address components
        """
        if address_str is None:
            return {"original": None, "full": None, "valid": False}
            
        if not isinstance(address_str, str):
            address_str = str(address_str)
            
        # Clean the address
        clean_address = self.normalize_text(address_str)
        
        # Basic result
        result = {
            "original": address_str,
            "full": clean_address,
            "valid": bool(clean_address)
        }
        
        if not clean_address:
            return result
            
        # Try to extract components
        
        # Look for postal code
        postal_code_pattern = r'(?:zip|postal)?\s*(?:code)?\s*[#:]?\s*(\d{5}(?:-\d{4})?)'
        postal_match = re.search(postal_code_pattern, clean_address, re.IGNORECASE)
        if postal_match:
            result["postal_code"] = postal_match.group(1)
            
        # Look for state/province
        # US states and territories
        us_states = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia', 'PR': 'Puerto Rico', 'VI': 'Virgin Islands', 'GU': 'Guam'
        }
        
        # Build regex pattern for states
        state_abbr_pattern = r'\b(' + '|'.join(us_states.keys()) + r')\b'
        state_name_pattern = r'\b(' + '|'.join(us_states.values()) + r')\b'
        
        # Try to find state abbreviation
        state_abbr_match = re.search(state_abbr_pattern, clean_address)
        if state_abbr_match:
            result["state"] = state_abbr_match.group(1)
            result["state_full"] = us_states[state_abbr_match.group(1)]
        else:
            # Try to find state name
            state_name_match = re.search(state_name_pattern, clean_address, re.IGNORECASE)
            if state_name_match:
                result["state_full"] = state_name_match.group(1)
                # Find abbreviation
                for abbr, name in us_states.items():
                    if name.lower() == state_name_match.group(1).lower():
                        result["state"] = abbr
                        break
        
        # Try to extract city
        # This is tricky without more context, but one approach is to look for "city_name, state"
        if "state" in result:
            city_pattern = r'\b([A-Za-z\s]+),\s*' + result["state"] + r'\b'
            city_match = re.search(city_pattern, clean_address)
            if city_match:
                result["city"] = city_match.group(1).strip()
        
        # Try to extract street address
        # This is also difficult without more context
        # One approach is to look for patterns like "123 Main St"
        street_pattern = r'\b(\d+\s+[A-Za-z0-9\s]+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Way|Place|Pl|Court|Ct)\.?)\b'
        street_match = re.search(street_pattern, clean_address, re.IGNORECASE)
        if street_match:
            result["street"] = street_match.group(1)
        
        return result
    
    def normalize_phone_numbers(self, phone_str: Any) -> Dict[str, Any]:
        """
        Standardize phone number formats.
        
        Args:
            phone_str: Phone number string to normalize
            
        Returns:
            Dictionary with normalized phone components
        """
        if phone_str is None:
            return {"original": None, "full": None, "valid": False}
            
        if not isinstance(phone_str, str):
            phone_str = str(phone_str)
            
        # Clean the phone number
        clean_phone = re.sub(r'[^\d+]', '', phone_str)
        
        # Basic result
        result = {
            "original": phone_str,
            "digits": clean_phone,
            "valid": len(clean_phone) >= 7  # Minimum valid length
        }
        
        if not clean_phone:
            return result
            
        # Determine country code
        if clean_phone.startswith('+'):
            # International format
            if clean_phone.startswith('+1'):
                # US/Canada
                result["country_code"] = "1"
                clean_phone = clean_phone[2:]
            else:
                # Try to extract country code (typically 1-3 digits)
                match = re.match(r'\+(\d{1,3})', clean_phone)
                if match:
                    result["country_code"] = match.group(1)
                    clean_phone = clean_phone[len(match.group(1))+1:]
        elif len(clean_phone) == 10:
            # US/Canada without country code
            result["country_code"] = "1"
        elif len(clean_phone) == 11 and clean_phone.startswith('1'):
            # US/Canada with country code
            result["country_code"] = "1"
            clean_phone = clean_phone[1:]
            
        # Format for US/Canada
        if result.get("country_code") == "1" and len(clean_phone) == 10:
            result["area_code"] = clean_phone[:3]
            result["exchange"] = clean_phone[3:6]
            result["line_number"] = clean_phone[6:]
            result["formatted"] = f"({result['area_code']}) {result['exchange']}-{result['line_number']}"
            result["e164"] = f"+{result['country_code']}{clean_phone}"
        else:
            # Generic formatting
            if "country_code" in result:
                result["e164"] = f"+{result['country_code']}{clean_phone}"
                result["formatted"] = f"+{result['country_code']} {clean_phone}"
            else:
                result["formatted"] = clean_phone
        
        return result
    
    def normalize_urls(self, url: Any) -> Dict[str, Any]:
        """
        Clean and standardize URLs.
        
        Args:
            url: URL to normalize
            
        Returns:
            Dictionary with normalized URL components
        """
        if url is None:
            return {"original": None, "full": None, "valid": False}
            
        if not isinstance(url, str):
            url = str(url)
            
        # Clean the URL
        clean_url = url.strip()
        
        # Basic result
        result = {
            "original": url,
            "full": clean_url,
            "valid": bool(clean_url)
        }
        
        if not clean_url:
            return result
            
        # Try to use urllib.parse for proper URL parsing
        try:
            from urllib.parse import urlparse, urlunparse
            
            # Add scheme if missing
            if not clean_url.startswith(('http://', 'https://', 'ftp://')):
                clean_url = 'https://' + clean_url
                
            # Parse URL
            parsed = urlparse(clean_url)
            
            # Extract components
            result["scheme"] = parsed.scheme
            result["netloc"] = parsed.netloc
            result["domain"] = parsed.netloc.split(':')[0]  # Remove port if present
            result["path"] = parsed.path
            result["params"] = parsed.params
            result["query"] = parsed.query
            result["fragment"] = parsed.fragment
            
            # Remove www. from domain
            if result["domain"].startswith('www.'):
                result["domain_no_www"] = result["domain"][4:]
            else:
                result["domain_no_www"] = result["domain"]
                
            # Create canonical URL (lowercase, no trailing slash)
            canonical = urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip('/') or '/',
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            result["canonical"] = canonical
            
            # Update validity based on proper parsing
            result["valid"] = bool(parsed.netloc)
            
        except (ImportError, ValueError) as e:
            # Fallback to simple URL cleaning
            logger.debug(f"URL parsing failed: {str(e)}")
            
            # Remove fragments
            if '#' in clean_url:
                clean_url = clean_url.split('#')[0]
                
            # Add scheme if missing
            if not clean_url.startswith(('http://', 'https://', 'ftp://')):
                clean_url = 'https://' + clean_url
                
            result["full"] = clean_url
        
        return result
    
    def normalize_identifiers(self, id_str: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standardize product IDs, SKUs, etc.
        
        Args:
            id_str: Identifier string to normalize
            options: Optional normalization options
            
        Returns:
            Dictionary with normalized identifier information
        """
        if id_str is None:
            return {"original": None, "value": None, "valid": False}
            
        options = options or {}
        
        # Convert to string
        if not isinstance(id_str, str):
            id_str = str(id_str)
            
        # Clean the identifier
        clean_id = id_str.strip()
        
        # Determine format type
        id_type = options.get("type", "unknown")
        
        # Basic result
        result = {
            "original": id_str,
            "value": clean_id,
            "type": id_type,
            "valid": bool(clean_id)
        }
        
        if not clean_id:
            return result
            
        # Auto-detect type if not specified
        if id_type == "unknown":
            # Try to detect common formats
            if re.match(r'^[0-9]{13}$', clean_id):
                id_type = "isbn13"
            elif re.match(r'^[0-9]{10}$', clean_id) or re.match(r'^[0-9]{9}X$', clean_id, re.IGNORECASE):
                id_type = "isbn10"
            elif re.match(r'^[0-9]{12}$', clean_id):
                id_type = "upc"
            elif re.match(r'^[0-9]{8}$', clean_id):
                id_type = "issn"
            elif re.match(r'^[A-Z0-9]{10}$', clean_id):
                id_type = "sku"
            elif re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', clean_id, re.IGNORECASE):
                id_type = "uuid"
            
            result["type"] = id_type
            
        # Format based on type
        if id_type == "isbn13":
            # Format ISBN-13
            if re.match(r'^[0-9]{13}$', clean_id):
                result["formatted"] = f"{clean_id[0:3]}-{clean_id[3]}-{clean_id[4:8]}-{clean_id[8:12]}-{clean_id[12]}"
                result["valid"] = True
        elif id_type == "isbn10":
            # Format ISBN-10
            if re.match(r'^[0-9]{9}[0-9X]$', clean_id, re.IGNORECASE):
                result["formatted"] = f"{clean_id[0]}-{clean_id[1:5]}-{clean_id[5:9]}-{clean_id[9]}"
                result["valid"] = True
        elif id_type == "upc":
            # Format UPC
            if re.match(r'^[0-9]{12}$', clean_id):
                result["formatted"] = clean_id
                result["valid"] = True
        elif id_type == "uuid":
            # Format UUID
            uuid_match = re.match(r'^([0-9a-f]{8})-?([0-9a-f]{4})-?([0-9a-f]{4})-?([0-9a-f]{4})-?([0-9a-f]{12})$', clean_id, re.IGNORECASE)
            if uuid_match:
                result["formatted"] = f"{uuid_match.group(1)}-{uuid_match.group(2)}-{uuid_match.group(3)}-{uuid_match.group(4)}-{uuid_match.group(5)}"
                result["valid"] = True
        else:
            # Generic identifier, just return cleaned version
            result["formatted"] = clean_id
            
        return result
    
    def normalize_html_entities(self, text: str) -> str:
        """
        Decode HTML entities in text.
        
        Args:
            text: Text containing HTML entities
            
        Returns:
            Text with decoded HTML entities
        """
        if not text or not isinstance(text, str):
            return text
            
        return html.unescape(text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Standardize whitespace.
        
        Args:
            text: Text with potentially inconsistent whitespace
            
        Returns:
            Text with standardized whitespace
        """
        if not text or not isinstance(text, str):
            return text
            
        # Use the utility function
        return clean_whitespace(text)
    
    def normalize_list(self, items: List[Any], options: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Normalize list items.
        
        Args:
            items: List of items to normalize
            options: Optional normalization options
            
        Returns:
            List with normalized items
        """
        if items is None:
            return []
            
        if isinstance(items, str):
            # Try to parse as JSON
            try:
                items = json.loads(items)
            except json.JSONDecodeError:
                # Not valid JSON, split by common separators
                items = [item.strip() for item in re.split(r'[,;|]', items)]
                
        if not isinstance(items, list):
            # Convert to list
            items = [items]
            
        options = options or {}
        
        # Determine item type
        item_type = options.get("items", {}).get("type")
        if not item_type:
            # Try to infer from first non-None item
            for item in items:
                if item is not None:
                    item_type = self._infer_field_type(item)
                    break
            
        # Normalize each item
        normalized = []
        
        for item in items:
            if item is None:
                # Skip None items if specified
                if not options.get("include_none", False):
                    continue
                normalized.append(None)
                continue
                
            # Normalize based on item type
            if item_type == "text" or item_type == "string":
                normalized.append(self.normalize_text(item))
            elif item_type == "date":
                normalized.append(self.normalize_date(item, options.get("items", {})))
            elif item_type == "price":
                normalized.append(self.normalize_price(item, options.get("items", {})))
            elif item_type == "measurement":
                normalized.append(self.normalize_measurement(item, options.get("items", {})))
            elif item_type == "boolean":
                normalized.append(self.normalize_boolean(item))
            elif item_type == "object" or item_type == "dict":
                normalized.append(self.normalize_object(item, options.get("items", {})))
            elif item_type == "name":
                normalized.append(self.normalize_names(item))
            elif item_type == "address":
                normalized.append(self.normalize_addresses(item))
            elif item_type == "phone":
                normalized.append(self.normalize_phone_numbers(item))
            elif item_type == "url":
                normalized.append(self.normalize_urls(item))
            elif item_type == "identifier":
                normalized.append(self.normalize_identifiers(item, options.get("items", {})))
            else:
                # Default to basic normalization
                if isinstance(item, str):
                    normalized.append(self.normalize_text(item))
                elif isinstance(item, dict):
                    normalized.append(self.normalize_object(item, options.get("items", {})))
                elif isinstance(item, list):
                    normalized.append(self.normalize_list(item, options.get("items", {})))
                else:
                    normalized.append(item)
        
        # Remove duplicates if specified
        if options.get("remove_duplicates", False):
            # Try to convert to hashable types for deduplication
            hashable_items = []
            for item in normalized:
                if isinstance(item, dict):
                    # Convert dict to tuple of items
                    hashable_items.append(tuple(sorted(item.items())))
                elif isinstance(item, list):
                    # Convert list to tuple
                    hashable_items.append(tuple(item))
                else:
                    hashable_items.append(item)
                    
            # Find unique indices
            seen = set()
            unique_indices = [i for i, item in enumerate(hashable_items) if item not in seen and not seen.add(item)]
            
            # Filter original list
            normalized = [normalized[i] for i in unique_indices]
        
        return normalized
    
    def _get_field_type(self, field_name: str, value: Any, schema: Optional[Dict[str, Any]] = None) -> str:
        """
        Determine field type from name, value, and schema.
        
        Args:
            field_name: Name of the field
            value: Field value
            schema: Optional schema information
            
        Returns:
            Field type string
        """
        # Check schema first
        if schema and field_name in schema:
            field_type = schema[field_name].get("type")
            if field_type:
                return field_type
                
        # Use name-based heuristics
        name_lower = field_name.lower()
        
        if "price" in name_lower or "cost" in name_lower or "fee" in name_lower or "amount" in name_lower:
            return "price"
            
        if "date" in name_lower or "time" in name_lower or "day" in name_lower or "year" in name_lower or "month" in name_lower:
            return "date"
            
        if "url" in name_lower or "link" in name_lower or "href" in name_lower or "website" in name_lower:
            return "url"
            
        if "phone" in name_lower or "tel" in name_lower or "mobile" in name_lower or "cell" in name_lower:
            return "phone"
            
        if "name" in name_lower or "person" in name_lower or "author" in name_lower or "editor" in name_lower:
            return "name"
            
        if "address" in name_lower or "location" in name_lower or "street" in name_lower or "city" in name_lower or "state" in name_lower or "country" in name_lower:
            return "address"
            
        if "id" in name_lower or "identifier" in name_lower or "code" in name_lower or "sku" in name_lower or "isbn" in name_lower or "number" in name_lower:
            return "identifier"
            
        if "weight" in name_lower or "height" in name_lower or "width" in name_lower or "length" in name_lower or "size" in name_lower:
            return "measurement"
            
        if "enabled" in name_lower or "active" in name_lower or "available" in name_lower or "is_" in name_lower or "has_" in name_lower:
            return "boolean"
            
        if "html" in name_lower or "markup" in name_lower:
            return "html"
            
        # Infer type from value
        return self._infer_field_type(value)
    
    def _infer_field_type(self, value: Any) -> str:
        """
        Infer field type from value.
        
        Args:
            value: Field value to analyze
            
        Returns:
            Inferred field type string
        """
        if value is None:
            return "null"
            
        if isinstance(value, bool):
            return "boolean"
            
        if isinstance(value, (int, float)):
            return "number"
            
        if isinstance(value, (list, tuple)):
            return "list"
            
        if isinstance(value, dict):
            return "object"
            
        if isinstance(value, str):
            # Try to detect specific string types
            value_lower = value.lower().strip()
            
            # Check for boolean-like strings
            if value_lower in ["true", "false", "yes", "no", "y", "n", "t", "f", "1", "0", "on", "off"]:
                return "boolean"
                
            # Check for date-like strings
            if re.match(r'\d{4}-\d{2}-\d{2}', value) or re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', value):
                return "date"
                
            # Check for price-like strings
            if re.match(r'^[$€£¥]', value) or re.search(r'\d+\s*(?:USD|EUR|GBP|JPY)', value):
                return "price"
                
            # Check for measurement-like strings
            if re.search(r'\d+\s*(?:cm|mm|m|km|in|inch|ft|feet|yd|yards|mi|miles|kg|g|lb|pounds|oz|ounces)', value):
                return "measurement"
                
            # Check for URL-like strings
            if re.match(r'^https?://', value) or re.match(r'^www\.', value):
                return "url"
                
            # Check for phone-like strings
            if re.match(r'^\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$', value):
                return "phone"
                
            # Check for HTML-like strings
            if value.startswith('<') and value.endswith('>') and ('</' in value or '/>' in value):
                return "html"
                
            # Default to generic text
            return "text"
            
        # Default for unknown types
        return "unknown"