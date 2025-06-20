"""
Content Validation and Quality Analysis Module

This module provides tools for validating extracted content against schemas, 
performing semantic validation of extracted values, calculating confidence scores,
and handling edge cases in extraction results.

Features:
- Schema-based validation of extraction results
- Semantic validation of extracted values
- Confidence scoring for extraction results
- Automatic correction and normalization of common value formats
- Edge case handling and fallback mechanisms
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
import re
import json
import logging
import datetime
from collections import defaultdict
from urllib.parse import urljoin, urlparse

# Try to import jsonschema for schema validation
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContentValidation")

class ValidationResult:
    """Class to hold validation results with details about what passed/failed."""
    
    def __init__(self, is_valid: bool = True, score: float = 1.0):
        self.is_valid = is_valid
        self.score = score  # 0.0 to 1.0
        self.errors = []
        self.warnings = []
        self.corrections = {}
        self.field_scores = {}
    
    def add_error(self, field: str, message: str, score_impact: float = 0.5):
        """Add a validation error."""
        self.errors.append({"field": field, "message": message})
        self.is_valid = False
        self.score *= (1 - score_impact)
        
    def add_warning(self, field: str, message: str, score_impact: float = 0.1):
        """Add a validation warning."""
        self.warnings.append({"field": field, "message": message})
        self.score *= (1 - score_impact)
    
    def add_correction(self, field: str, original: Any, corrected: Any, confidence: float = 0.9):
        """Add a correction made to a field value."""
        self.corrections[field] = {
            "original": original,
            "corrected": corrected,
            "confidence": confidence
        }
    
    def set_field_score(self, field: str, score: float):
        """Set confidence score for a specific field."""
        self.field_scores[field] = max(0.0, min(1.0, score))
    
    def get_field_score(self, field: str) -> float:
        """Get confidence score for a specific field."""
        return self.field_scores.get(field, 1.0)
    
    def update_overall_score(self):
        """Update the overall score based on field scores."""
        if self.field_scores:
            self.score = sum(self.field_scores.values()) / len(self.field_scores)

class ContentValidator:
    """
    Validates extracted content against schemas and performs semantic validation.
    
    This class provides tools for:
    1. Schema validation against a JSON Schema
    2. Semantic validation of field values
    3. Automatic correction of common formatting issues
    4. Confidence scoring for extraction quality
    """
    
    # Type validation patterns
    URL_PATTERN = re.compile(r'^(https?://[^\s/$.?#].[^\s]*|/[^\s]*|\.{0,2}/[^\s]*)$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?[\d\s()\-\.]{7,}$')
    DATE_PATTERNS = [
        re.compile(r'^\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}$'),  # MM/DD/YYYY
        re.compile(r'^\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}$'),    # YYYY/MM/DD
        re.compile(r'^[A-Za-z]{3,9} \d{1,2},? \d{4}$'),      # Month DD, YYYY
        re.compile(r'^\d{1,2} [A-Za-z]{3,9},? \d{4}$')       # DD Month YYYY
    ]
    PRICE_PATTERN = re.compile(r'^[$£€¥]?\s*\d+([.,]\d{1,2})?\s*[$£€¥]?$')
    
    def __init__(self, user_intent: Optional[Dict[str, Any]] = None):
        """
        Initialize the content validator.
        
        Args:
            user_intent: Optional user intent dictionary to use for validation
        """
        self.user_intent = user_intent
        self.validators = {
            "url": self._validate_url,
            "email": self._validate_email,
            "phone": self._validate_phone,
            "date": self._validate_date,
            "price": self._validate_price,
            "number": self._validate_number,
            "text": self._validate_text,
            "image": self._validate_image,
            "html": self._validate_html
        }
        
        # Cache of known valid formats for quicker processing
        self.valid_format_cache = {}
        
        # Correction handlers for various field types
        self.correctors = {
            "price": self._correct_price,
            "date": self._correct_date,
            "url": self._correct_url,
            "phone": self._correct_phone,
            "text": self._correct_text
        }
    
    def validate_extraction_result(self, 
                                  data: Dict[str, Any], 
                                  schema: Optional[Dict[str, Any]] = None,
                                  base_url: Optional[str] = None,
                                  field_types: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Validate an extraction result against a schema and perform semantic validation.
        
        Args:
            data: The extracted data to validate
            schema: Optional JSON schema to validate against
            base_url: Optional base URL for resolving relative URLs
            field_types: Optional mapping of field names to expected types
            
        Returns:
            ValidationResult with validation details
        """
        # Initialize validation result
        result = ValidationResult()
        
        # If no data, return invalid result
        if not data:
            result.add_error("", "No data provided for validation", 1.0)
            return result
        
        # Perform JSON Schema validation if schema provided and jsonschema available
        schema_result = self._validate_against_schema(data, schema)
        if not schema_result.is_valid:
            # Copy schema validation errors to our result
            for error in schema_result.errors:
                result.add_error(error["field"], error["message"], 0.3)
        
        # Get field types from schema if available and not provided
        if not field_types and schema and "properties" in schema:
            field_types = {}
            for field, prop in schema["properties"].items():
                if "type" in prop:
                    field_types[field] = prop["type"]
        
        # Infer field types if not provided
        if not field_types:
            field_types = self._infer_field_types(data)
        
        # Perform semantic validation on each field
        for field, value in data.items():
            # Skip null values
            if value is None:
                result.set_field_score(field, 0.5)  # Missing data gets a medium-low score
                continue
                
            # Get the field type
            field_type = field_types.get(field, self._infer_field_type(field, value))
            
            # Validate field value
            field_result = self._validate_field(field, value, field_type, base_url)
            
            # Add any errors or warnings to the overall result
            for error in field_result.errors:
                result.add_error(field, error["message"], 0.2)
            
            for warning in field_result.warnings:
                result.add_warning(field, warning["message"], 0.1)
            
            # Apply corrections if any
            if field in field_result.corrections:
                correction = field_result.corrections[field]
                result.add_correction(field, correction["original"], correction["corrected"], correction["confidence"])
                
                # Use the corrected value in the data
                data[field] = correction["corrected"]
            
            # Set field score
            result.set_field_score(field, field_result.get_field_score(field))
        
        # Check for consistency across multiple items if data is a list
        if isinstance(data, list) and len(data) > 1:
            consistency_result = self._check_consistency_across_items(data)
            
            # Apply consistency scores to field scores
            for field, score in consistency_result.field_scores.items():
                avg_score = (result.get_field_score(field) + score) / 2
                result.set_field_score(field, avg_score)
        
        # Check if extracted data matches user intent
        if self.user_intent:
            intent_match_result = self._validate_against_intent(data, self.user_intent)
            
            # Add intent validation warnings/errors
            for warning in intent_match_result.warnings:
                result.add_warning(warning["field"], warning["message"], 0.1)
            
            # Adjust overall score based on intent match
            result.score *= intent_match_result.score
        
        # Update the overall score based on field scores
        result.update_overall_score()
        
        return result
    
    def _validate_against_schema(self, 
                               data: Dict[str, Any], 
                               schema: Optional[Dict[str, Any]]) -> ValidationResult:
        """
        Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema to validate against
            
        Returns:
            ValidationResult with schema validation results
        """
        result = ValidationResult()
        
        # If no schema or jsonschema not available, return valid result
        if not schema or not HAS_JSONSCHEMA:
            if not HAS_JSONSCHEMA and schema:
                result.add_warning("", "jsonschema library not available for schema validation", 0.1)
            return result
        
        try:
            # Validate against the schema
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            # Add error details
            path = "/".join(str(p) for p in e.path) if e.path else ""
            result.add_error(path, f"Schema validation error: {e.message}", 0.3)
            result.is_valid = False
        except Exception as e:
            # Add general error
            result.add_error("", f"Schema validation error: {str(e)}", 0.3)
            result.is_valid = False
        
        return result
    
    def _validate_field(self, 
                      field: str, 
                      value: Any, 
                      field_type: str,
                      base_url: Optional[str] = None) -> ValidationResult:
        """
        Validate a single field value.
        
        Args:
            field: Field name
            value: Field value
            field_type: Expected field type
            base_url: Optional base URL for resolving relative URLs
            
        Returns:
            ValidationResult for the field
        """
        result = ValidationResult()
        
        # Skip validation for None values
        if value is None:
            result.score = 0.5
            return result
        
        # Call the appropriate validator based on field type
        validator = self.validators.get(field_type, self._validate_text)
        validator_result = validator(field, value, base_url)
        
        # Merge validator result with our result
        for error in validator_result.errors:
            result.add_error(error["field"], error["message"], 0.2)
        
        for warning in validator_result.warnings:
            result.add_warning(warning["field"], warning["message"], 0.1)
        
        # Apply corrections if any
        if field in validator_result.corrections:
            result.add_correction(
                field, 
                validator_result.corrections[field]["original"],
                validator_result.corrections[field]["corrected"],
                validator_result.corrections[field]["confidence"]
            )
        
        # Update score
        result.score = validator_result.score
        
        return result
    
    def _validate_url(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate a URL field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"URL must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_error(field, "URL is empty", 0.5)
            return result
        
        # Check for valid URL format
        if not self.URL_PATTERN.match(value):
            result.add_error(field, f"Invalid URL format: {value}", 0.4)
            
            # Try to correct the URL
            corrected = self._correct_url(value, base_url)
            if corrected != value:
                result.add_correction(field, value, corrected, 0.8)
                result.score = 0.8  # Partial validity for corrected URLs
            return result
        
        # Check for relative URLs and resolve if base_url provided
        if base_url and (value.startswith('/') or value.startswith('./') or value.startswith('../')):
            corrected = urljoin(base_url, value)
            result.add_correction(field, value, corrected, 0.9)
            result.score = 0.9  # High validity for resolved relative URLs
        
        return result
    
    def _validate_email(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate an email field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"Email must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_error(field, "Email is empty", 0.5)
            return result
        
        # Check for valid email format
        if not self.EMAIL_PATTERN.match(value):
            result.add_error(field, f"Invalid email format: {value}", 0.4)
            return result
        
        return result
    
    def _validate_phone(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate a phone number field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"Phone number must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_error(field, "Phone number is empty", 0.5)
            return result
        
        # Check for valid phone format
        if not self.PHONE_PATTERN.match(value):
            result.add_error(field, f"Invalid phone number format: {value}", 0.4)
            
            # Try to correct the phone number
            corrected = self._correct_phone(value)
            if corrected != value:
                result.add_correction(field, value, corrected, 0.8)
                result.score = 0.8  # Partial validity for corrected phone numbers
            
            return result
        
        # Clean up phone number format
        corrected = self._correct_phone(value)
        if corrected != value:
            result.add_correction(field, value, corrected, 0.9)
        
        return result
    
    def _validate_date(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate a date field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"Date must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_error(field, "Date is empty", 0.5)
            return result
        
        # Check for valid date format
        valid_format = any(pattern.match(value) for pattern in self.DATE_PATTERNS)
        if not valid_format:
            result.add_error(field, f"Invalid date format: {value}", 0.4)
            
            # Try to correct the date
            corrected = self._correct_date(value)
            if corrected != value:
                result.add_correction(field, value, corrected, 0.8)
                result.score = 0.8  # Partial validity for corrected dates
            
            return result
        
        # Try to standardize the date format
        corrected = self._correct_date(value)
        if corrected != value:
            result.add_correction(field, value, corrected, 0.9)
        
        return result
    
    def _validate_price(self, field: str, value: Any, base_url: Optional[str] = None) -> ValidationResult:
        """Validate a price field."""
        result = ValidationResult()
        
        # Allow both string and numeric price values
        if not isinstance(value, (str, int, float)):
            result.add_error(field, f"Price must be a string or number, got {type(value).__name__}", 0.5)
            return result
        
        # Convert to string for further validation
        value_str = str(value)
        
        # Check if empty
        if not value_str.strip():
            result.add_error(field, "Price is empty", 0.5)
            return result
        
        # Check for valid price format if it's a string
        if isinstance(value, str) and not self.PRICE_PATTERN.match(value):
            result.add_error(field, f"Invalid price format: {value}", 0.4)
            
            # Try to correct the price
            corrected = self._correct_price(value)
            if corrected != value:
                result.add_correction(field, value, corrected, 0.8)
                result.score = 0.8  # Partial validity for corrected prices
            
            return result
        
        # If a float or numeric string, it's already in a valid format
        if isinstance(value, (int, float)) or re.match(r'^\d+(\.\d+)?$', value_str):
            return result
        
        # Clean up price format
        corrected = self._correct_price(value_str)
        if corrected != value_str:
            result.add_correction(field, value, corrected, 0.9)
        
        return result
    
    def _validate_number(self, field: str, value: Any, base_url: Optional[str] = None) -> ValidationResult:
        """Validate a numeric field."""
        result = ValidationResult()
        
        # Check if it's a number or can be converted to one
        if not isinstance(value, (int, float)):
            if isinstance(value, str):
                # Try to convert string to number
                try:
                    float(value)
                except ValueError:
                    result.add_error(field, f"Cannot convert string to number: {value}", 0.5)
                    return result
            else:
                result.add_error(field, f"Number must be a numeric type, got {type(value).__name__}", 0.5)
                return result
        
        return result
    
    def _validate_text(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate a text field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"Text must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_warning(field, "Text is empty", 0.2)
            result.score = 0.8
            return result
        
        # Check for suspiciously short or long text
        if len(value) < 2:
            result.add_warning(field, f"Text is very short ({len(value)} chars)", 0.1)
            result.score = 0.9
        elif len(value) > 5000:
            result.add_warning(field, f"Text is very long ({len(value)} chars)", 0.1)
            result.score = 0.9
        
        # Clean up text
        corrected = self._correct_text(value)
        if corrected != value:
            result.add_correction(field, value, corrected, 0.9)
        
        return result
    
    def _validate_image(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate an image URL field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"Image URL must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_error(field, "Image URL is empty", 0.5)
            return result
        
        # Check for valid URL format
        if not self.URL_PATTERN.match(value):
            result.add_error(field, f"Invalid image URL format: {value}", 0.4)
            
            # Try to correct the URL
            corrected = self._correct_url(value, base_url)
            if corrected != value:
                result.add_correction(field, value, corrected, 0.8)
                result.score = 0.8  # Partial validity for corrected URLs
            
            return result
        
        # Check for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp']
        has_image_ext = any(value.lower().endswith(ext) for ext in image_extensions)
        
        # Check for image URL patterns
        contains_image_pattern = ('images' in value.lower() or 'img' in value.lower() or 'photo' in value.lower())
        
        # If it doesn't look like an image URL, add a warning
        if not has_image_ext and not contains_image_pattern:
            result.add_warning(field, f"URL does not appear to be an image: {value}", 0.2)
            result.score = 0.8
        
        # Check for relative URLs and resolve if base_url provided
        if base_url and (value.startswith('/') or value.startswith('./') or value.startswith('../')):
            corrected = urljoin(base_url, value)
            result.add_correction(field, value, corrected, 0.9)
            result.score = 0.9  # High validity for resolved relative URLs
        
        return result
    
    def _validate_html(self, field: str, value: str, base_url: Optional[str] = None) -> ValidationResult:
        """Validate an HTML field."""
        result = ValidationResult()
        
        # Check if it's a string
        if not isinstance(value, str):
            result.add_error(field, f"HTML must be a string, got {type(value).__name__}", 0.5)
            return result
        
        # Check if empty
        if not value.strip():
            result.add_warning(field, "HTML is empty", 0.2)
            result.score = 0.8
            return result
        
        # Check for HTML tags
        has_html_tags = '<' in value and '>' in value
        if not has_html_tags:
            result.add_warning(field, f"Content does not appear to be HTML: {value[:50]}...", 0.2)
            result.score = 0.8
        
        return result
    
    def _correct_price(self, value: str) -> str:
        """Correct common price format issues."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove extra whitespace
        corrected = value.strip()
        
        # Remove non-numeric characters except for currency symbols, commas, and decimal points
        allowed_chars = '$€£¥.,0123456789'
        corrected = ''.join(c for c in corrected if c in allowed_chars or c.isdigit())
        
        # Make sure we have a consistent format with currency symbol at the start
        # and up to 2 decimal places
        currency_symbols = ['$', '€', '£', '¥']
        has_currency = any(sym in corrected for sym in currency_symbols)
        
        # Extract numeric part
        numeric_part = ''.join(c for c in corrected if c.isdigit() or c in '.,')
        
        # Replace comma with decimal point if needed
        if ',' in numeric_part and '.' not in numeric_part:
            # If comma is used as decimal separator
            if numeric_part.count(',') == 1 and numeric_part.rindex(',') > len(numeric_part) - 4:
                numeric_part = numeric_part.replace(',', '.')
            # If comma is used as thousands separator
            else:
                numeric_part = numeric_part.replace(',', '')
        
        # Format the price with up to 2 decimal places
        try:
            numeric_value = float(numeric_part)
            formatted_numeric = f"{numeric_value:.2f}" if '.' in numeric_part else str(int(numeric_value))
            
            # Add currency symbol if original had one
            if has_currency:
                # Find which currency symbol was used
                used_symbol = next((sym for sym in currency_symbols if sym in corrected), '$')
                return f"{used_symbol}{formatted_numeric}"
            else:
                return formatted_numeric
        except ValueError:
            # If conversion fails, return original value
            return value
    
    def _correct_date(self, value: str) -> str:
        """Standardize date formats."""
        if not isinstance(value, str):
            return str(value)
        
        # Try various date parsing approaches
        try:
            # Use dateutil parser for flexible date parsing
            from dateutil import parser
            
            # Parse the date
            parsed_date = parser.parse(value, fuzzy=True)
            
            # Return ISO format (YYYY-MM-DD)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            # Try common date formats with datetime
            date_formats = [
                '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d',
                '%m.%d.%Y', '%d.%m.%Y', '%Y.%m.%d',
                '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.datetime.strptime(value, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except:
                    continue
            
            # If all parsing attempts fail, return original value
            return value
    
    def _correct_url(self, value: str, base_url: Optional[str] = None) -> str:
        """Correct common URL format issues."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove extra whitespace
        corrected = value.strip()
        
        # Add http:// if missing
        if not corrected.startswith(('http://', 'https://', '/', './', '../')):
            # Check if it looks like a domain
            if re.match(r'^[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}(/.*)?$', corrected):
                corrected = 'https://' + corrected
        
        # Handle relative URLs if base_url provided
        if base_url and (corrected.startswith('/') or corrected.startswith('./') or corrected.startswith('../')):
            try:
                corrected = urljoin(base_url, corrected)
            except:
                # If urljoin fails, return the original value
                return value
        
        return corrected
    
    def _correct_phone(self, value: str) -> str:
        """Standardize phone number formats."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove all non-digit characters except + for country code
        digits_only = ''.join(c for c in value if c.isdigit() or c == '+')
        
        # Check if it starts with a country code
        if digits_only.startswith('+'):
            return digits_only
        
        # Format based on length for US numbers
        if len(digits_only) == 10:
            return f"+1 {digits_only[:3]}-{digits_only[3:6]}-{digits_only[6:]}"
        elif len(digits_only) == 11 and digits_only.startswith('1'):
            return f"+{digits_only[0]} {digits_only[1:4]}-{digits_only[4:7]}-{digits_only[7:]}"
        else:
            # For other formats, just return the digits with a plus if it seems to be a country code
            if len(digits_only) > 10 and not digits_only.startswith('+'):
                return '+' + digits_only
            return digits_only
    
    def _correct_text(self, value: str) -> str:
        """Clean up text content."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove excess whitespace
        corrected = re.sub(r'\s+', ' ', value).strip()
        
        # Remove HTML tags if present
        if '<' in corrected and '>' in corrected:
            corrected = re.sub(r'<[^>]+>', '', corrected)
        
        # Fix common encodings issues
        corrected = corrected.replace('&amp;', '&')
        corrected = corrected.replace('&lt;', '<')
        corrected = corrected.replace('&gt;', '>')
        corrected = corrected.replace('&quot;', '"')
        corrected = corrected.replace('&apos;', "'")
        
        return corrected
    
    def _infer_field_types(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Infer field types from data and field names."""
        field_types = {}
        
        for field, value in data.items():
            field_types[field] = self._infer_field_type(field, value)
        
        return field_types
    
    def _infer_field_type(self, field: str, value: Any) -> str:
        """Infer the type of a field based on its name and value."""
        field_lower = field.lower()
        
        # Check field name for type hints
        if any(x in field_lower for x in ['url', 'link', 'href']):
            return 'url'
        elif any(x in field_lower for x in ['email', 'e-mail']):
            return 'email'
        elif any(x in field_lower for x in ['phone', 'tel', 'cell', 'mobile']):
            return 'phone'
        elif any(x in field_lower for x in ['date', 'time', 'day', 'month', 'year']):
            return 'date'
        elif any(x in field_lower for x in ['price', 'cost', 'fee', 'amount', '$', '€', '£', '¥']):
            return 'price'
        elif any(x in field_lower for x in ['num', 'count', 'qty', 'quantity', 'age']):
            return 'number'
        elif any(x in field_lower for x in ['image', 'img', 'photo', 'picture', 'thumbnail']):
            return 'image'
        elif any(x in field_lower for x in ['html', 'content', 'body']):
            return 'html'
        
        # Check value type for hints
        if isinstance(value, (int, float)):
            return 'number'
        elif isinstance(value, str):
            # Check if it matches common patterns
            if self.URL_PATTERN.match(value):
                if any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    return 'image'
                return 'url'
            elif self.EMAIL_PATTERN.match(value):
                return 'email'
            elif self.PHONE_PATTERN.match(value):
                return 'phone'
            elif any(pattern.match(value) for pattern in self.DATE_PATTERNS):
                return 'date'
            elif self.PRICE_PATTERN.match(value):
                return 'price'
            elif '<' in value and '>' in value and ('</' in value or '/>' in value):
                return 'html'
            elif value.isdigit() or (value.replace('.', '', 1).isdigit() and value.count('.') == 1):
                return 'number'
        
        # Default to text for anything else
        return 'text'
    
    def _check_consistency_across_items(self, items: List[Dict[str, Any]]) -> ValidationResult:
        """Check consistency of fields across multiple items."""
        result = ValidationResult()
        
        # Analyze value patterns across items
        field_values = defaultdict(list)
        
        # Collect values for each field
        for item in items:
            for field, value in item.items():
                field_values[field].append(value)
        
        # Check consistency for each field
        for field, values in field_values.items():
            # Skip fields that don't appear in all items
            if len(values) < len(items) * 0.5:  # At least 50% of items should have the field
                continue
            
            # Check for value type consistency
            value_types = [type(v) for v in values if v is not None]
            if len(set(value_types)) > 1:
                result.add_warning(field, f"Inconsistent value types: {set(type(v).__name__ for v in values if v is not None)}", 0.1)
                result.set_field_score(field, 0.7)
                continue
            
            # For numeric fields, check value range consistency
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                non_null_values = [v for v in values if v is not None]
                if non_null_values:
                    min_val = min(non_null_values)
                    max_val = max(non_null_values)
                    
                    # If the range is abnormally large, it might indicate an issue
                    if max_val > min_val * 100 and max_val - min_val > 1000:
                        result.add_warning(field, f"Wide value range: {min_val} to {max_val}", 0.1)
                        result.set_field_score(field, 0.8)
                    else:
                        result.set_field_score(field, 1.0)
            
            # For string fields, check length consistency
            elif all(isinstance(v, str) for v in values if v is not None):
                non_null_values = [v for v in values if v is not None and v.strip()]
                if non_null_values:
                    lengths = [len(v) for v in non_null_values]
                    avg_length = sum(lengths) / len(lengths)
                    
                    # Check for outliers (3x longer or shorter than average)
                    outliers = [v for v, l in zip(non_null_values, lengths) if l < avg_length / 3 or l > avg_length * 3]
                    
                    if outliers and len(outliers) < len(non_null_values) * 0.2:  # Less than 20% are outliers
                        result.add_warning(field, f"Some values have inconsistent length compared to others", 0.1)
                        result.set_field_score(field, 0.9)
                    else:
                        result.set_field_score(field, 1.0)
        
        return result
    
    def _validate_against_intent(self, 
                               data: Dict[str, Any], 
                               user_intent: Dict[str, Any]) -> ValidationResult:
        """Check if extracted data matches user intent."""
        result = ValidationResult()
        
        # Extract key parts of the user intent
        target_item = user_intent.get('target_item', '')
        entity_type = user_intent.get('entity_type', '')
        properties = user_intent.get('properties', [])
        keywords = user_intent.get('keywords', [])
        
        # 1. Check if the extracted data has the expected fields
        if properties:
            missing_properties = [prop for prop in properties if prop not in data or data[prop] is None]
            if missing_properties:
                result.add_warning("", f"Missing expected properties: {', '.join(missing_properties)}", 0.1)
                result.score *= 0.9
        
        # 2. Check if the content matches the expected entity type or target item
        if entity_type or target_item:
            # Convert item data to lowercase string for keyword matching
            text_content = ' '.join(str(v).lower() for v in data.values() if v is not None)
            
            # Create a list of keywords to check for
            check_keywords = []
            if entity_type:
                check_keywords.append(entity_type.lower())
            if target_item:
                check_keywords.append(target_item.lower())
                # Add variations (singular/plural forms)
                if target_item.endswith('s'):
                    check_keywords.append(target_item[:-1].lower())
                else:
                    check_keywords.append(f"{target_item.lower()}s")
            
            # Check if any keywords are present
            found_keywords = [kw for kw in check_keywords if kw in text_content]
            if not found_keywords:
                result.add_warning("", f"Content may not match expected {entity_type or target_item}", 0.2)
                result.score *= 0.8
        
        # 3. Check for specific keywords from user intent
        if keywords:
            # Check if any of the keywords appear in the data
            text_content = ' '.join(str(v).lower() for v in data.values() if v is not None)
            found_keywords = [kw for kw in keywords if kw.lower() in text_content]
            
            # Calculate a score based on percentage of keywords found
            if keywords:
                keyword_match_score = len(found_keywords) / len(keywords)
                result.score *= 0.5 + (0.5 * keyword_match_score)  # Scale between 0.5 and 1.0
        
        return result

class EdgeCaseHandler:
    """
    Handles edge cases and special scenarios in extraction data.
    
    This class provides tools for:
    1. Handling missing data
    2. Extracting data from alternative sources
    3. Advanced attribute extraction
    4. Handling embedded JSON/JS data
    """
    
    def __init__(self):
        """Initialize the edge case handler."""
        pass
    
    def handle_missing_data(self, data: Dict[str, Any], schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle missing data by providing default values where appropriate.
        
        Args:
            data: The extracted data
            schema: Optional schema with default values
            
        Returns:
            Updated data with defaults for missing fields
        """
        # If no data or schema, return as is
        if not data:
            return data
        
        # Create a copy to avoid modifying the original
        result = dict(data)
        
        # If schema is provided, use it to set defaults
        if schema and 'properties' in schema:
            for field, prop in schema['properties'].items():
                if field not in result or result[field] is None:
                    # Use default value from schema if available
                    if 'default' in prop:
                        result[field] = prop['default']
                    # Otherwise, use type-appropriate default
                    elif 'type' in prop:
                        if prop['type'] == 'string':
                            result[field] = ""
                        elif prop['type'] == 'number' or prop['type'] == 'integer':
                            result[field] = 0
                        elif prop['type'] == 'boolean':
                            result[field] = False
                        elif prop['type'] == 'array':
                            result[field] = []
                        elif prop['type'] == 'object':
                            result[field] = {}
        
        return result
    
    def extract_from_attributes(self, soup, field_mapping: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Extract data from various HTML attributes.
        
        Args:
            soup: BeautifulSoup object to extract from
            field_mapping: Mapping of field names to attribute selectors
            
        Returns:
            Extracted data from attributes
        """
        result = {}
        
        for field, selectors in field_mapping.items():
            for selector in selectors:
                # Parse the selector to get element and attribute
                if '::attr(' in selector:
                    # Format: "css_selector::attr(attribute_name)"
                    css_part, attr_part = selector.split('::attr(')
                    attr_name = attr_part.rstrip(')')
                    
                    elements = soup.select(css_part)
                    if elements:
                        attr_value = elements[0].get(attr_name)
                        if attr_value:
                            result[field] = attr_value
                            break
                elif '@' in selector:
                    # Format: "css_selector@attribute_name"
                    css_part, attr_name = selector.split('@')
                    
                    elements = soup.select(css_part)
                    if elements:
                        attr_value = elements[0].get(attr_name)
                        if attr_value:
                            result[field] = attr_value
                            break
        
        return result
    
    def extract_from_json_ld(self, soup) -> List[Dict[str, Any]]:
        """
        Extract structured data from JSON-LD script tags.
        
        Args:
            soup: BeautifulSoup object to extract from
            
        Returns:
            List of extracted structured data objects
        """
        results = []
        
        # Look for JSON-LD script tags
        ld_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        
        for script in ld_scripts:
            try:
                # Parse the JSON content
                data = json.loads(script.string)
                
                # Handle both arrays and single objects
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
            except Exception as e:
                logger.warning(f"Error parsing JSON-LD: {str(e)}")
        
        return results
    
    def extract_from_javascript(self, html_content: str) -> Dict[str, Any]:
        """
        Extract data embedded in JavaScript variables.
        
        Args:
            html_content: HTML content to search for JS variables
            
        Returns:
            Dictionary of extracted data from JavaScript
        """
        results = {}
        
        # Common JavaScript variable patterns to look for
        patterns = [
            # Object literal: var someObject = { key: "value" };
            r'var\s+(\w+)\s*=\s*({[^;]+});',
            # JSON data: var someVar = JSON.parse('{"key":"value"}');
            r'var\s+(\w+)\s*=\s*JSON\.parse\([\'"]({[^\'"]+})[\'"]\)',
            # Array literal: var someArray = [...];
            r'var\s+(\w+)\s*=\s*(\[[^\]]+\]);',
            # Window property: window.dataLayer = [...];
            r'window\.(\w+)\s*=\s*({[^;]+});',
            # Window property array: window.dataLayer = [...];
            r'window\.(\w+)\s*=\s*(\[[^\]]+\]);',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content)
            
            for match in matches:
                var_name, data_str = match
                
                try:
                    # Try to parse as JSON
                    data = json.loads(data_str)
                    results[var_name] = data
                except json.JSONDecodeError:
                    # If it's not valid JSON, it might be a JS object literal
                    # Here we capture only very simple cases
                    if data_str.startswith('{') and data_str.endswith('}'):
                        try:
                            # Replace JS-style quotes with JSON-style
                            json_str = re.sub(r'([{,])\s*(\w+):', r'\1"\2":', data_str)
                            json_str = json_str.replace("'", '"')
                            data = json.loads(json_str)
                            results[var_name] = data
                        except:
                            pass
        
        return results
    
    def consolidate_duplicate_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate duplicate fields with different names.
        
        Args:
            data: The extracted data
            
        Returns:
            Data with consolidated duplicate fields
        """
        if not data:
            return data
        
        # Create a copy to avoid modifying the original
        result = dict(data)
        
        # Define groups of field names that might refer to the same data
        field_groups = [
            ["price", "cost", "amount", "value"],
            ["title", "name", "heading"],
            ["description", "summary", "details", "content"],
            ["image", "img", "thumbnail", "photo", "picture"],
            ["url", "link", "href"],
            ["date", "datetime", "timestamp"],
            ["author", "creator", "by"],
            ["category", "type", "class"]
        ]
        
        # Check each group for multiple occurrences
        for group in field_groups:
            # Find which fields from the group are present in data
            present_fields = [field for field in group if field in result]
            
            # If more than one field from the group is present
            if len(present_fields) > 1:
                # Choose the primary field (first one in the group that exists)
                primary_field = present_fields[0]
                
                # If the primary field is empty, try to fill it from other fields
                if result[primary_field] is None or (isinstance(result[primary_field], str) and not result[primary_field].strip()):
                    for field in present_fields[1:]:
                        if result[field] is not None and (not isinstance(result[field], str) or result[field].strip()):
                            result[primary_field] = result[field]
                            break
                
                # Remove the secondary fields
                for field in present_fields[1:]:
                    del result[field]
        
        return result

def create_validator(user_intent: Optional[Dict[str, Any]] = None) -> ContentValidator:
    """Factory function to create a content validator."""
    return ContentValidator(user_intent=user_intent)

def create_edge_case_handler() -> EdgeCaseHandler:
    """Factory function to create an edge case handler."""
    return EdgeCaseHandler()