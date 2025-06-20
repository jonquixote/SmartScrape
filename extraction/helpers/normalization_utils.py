"""
Normalization Utilities Module

This module provides utility functions for normalizing different types of content,
including text, dates, currencies, measurements, and more. These utilities are used
by the ContentNormalizer to standardize extracted data.
"""

import re
import html
from typing import Dict, Any, Optional, List, Union, Tuple
import json
from datetime import datetime, date
import unicodedata
import urllib.parse
from decimal import Decimal

# ===== Text Normalization Utilities =====

def clean_whitespace(text: Optional[str]) -> Optional[str]:
    """
    Clean and normalize whitespace in text.
    
    Args:
        text: Text to clean
        
    Returns:
        Text with normalized whitespace
    """
    if text is None:
        return None
        
    # Replace all whitespace (including tabs and newlines) with a single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading and trailing whitespace
    return text.strip()

def remove_control_chars(text: Optional[str]) -> Optional[str]:
    """
    Remove control characters from text.
    
    Args:
        text: Text to clean
        
    Returns:
        Text without control characters
    """
    if text is None:
        return None
        
    # Remove ASCII control characters (0-31 and 127)
    return re.sub(r'[\x00-\x1F\x7F]', '', text)

def standardize_quotes(text: Optional[str]) -> Optional[str]:
    """
    Standardize various quote types to basic ASCII quotes.
    
    Args:
        text: Text to standardize
        
    Returns:
        Text with standardized quotes
    """
    if text is None:
        return None
        
    # Replace curly/smart quotes with straight quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace('’', "'").replace('’', "'")
    return text

def standardize_dashes(text: Optional[str]) -> Optional[str]:
    """
    Standardize various dash types to basic ASCII dashes.
    
    Args:
        text: Text to standardize
        
    Returns:
        Text with standardized dashes
    """
    if text is None:
        return None
        
    # Replace em-dash and en-dash with standard ASCII equivalents
    text = text.replace('—', '--')  # em-dash to double hyphen
    text = text.replace('–', '-')   # en-dash to hyphen
    return text

def clean_html_fragments(text: Optional[str]) -> Optional[str]:
    """
    Remove HTML tags from text.
    
    Args:
        text: Text that may contain HTML fragments
        
    Returns:
        Text with HTML tags removed
    """
    if text is None:
        return None
        
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Clean up whitespace
    return clean_whitespace(text)

def collapse_newlines(text: Optional[str]) -> Optional[str]:
    """
    Collapse multiple consecutive newlines into a single newline.
    
    Args:
        text: Text to clean
        
    Returns:
        Text with normalized newlines
    """
    if text is None:
        return None
        
    # Collapse multiple newlines into a single newline
    return re.sub(r'\n{2,}', '\n', text)

def normalize_text_full(text: Optional[str], remove_html: bool = True) -> str:
    """
    Apply all text normalization functions in sequence.
    
    Args:
        text: Text to normalize
        remove_html: Whether to remove HTML tags
        
    Returns:
        Fully normalized text
    """
    if text is None:
        return ""
        
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Apply normalizations in sequence
    text = remove_control_chars(text)
    if remove_html:
        text = clean_html_fragments(text)
    text = standardize_quotes(text)
    text = standardize_dashes(text)
    text = clean_whitespace(text)
    
    return text


# ===== Date Normalization Utilities =====

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a date string using multiple formats.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    # List of common date formats to try
    formats = [
        # ISO formats
        "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S",
        # US formats
        "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%Y %H:%M:%S",
        # European formats
        "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%Y %H:%M:%S",
        # Month name formats
        "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y",
        # Year first with month name
        "%Y %b %d", "%Y %B %d",
    ]
    
    # Clean the date string
    date_str = clean_whitespace(date_str)
    
    # Try each format
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try more lenient parsing with dateutil if available
    try:
        from dateutil import parser
        return parser.parse(date_str)
    except:
        # If dateutil is not available or fails
        return None

def detect_date_format(date_str: Optional[str]) -> Optional[str]:
    """
    Detect the format of a date string.
    
    Args:
        date_str: Date string to analyze
        
    Returns:
        Format string (strftime-compatible) or None if format can't be detected
    """
    if not date_str:
        return None
    
    # Common patterns and their corresponding format strings
    patterns = [
        (r'^\d{4}-\d{2}-\d{2}$', "%Y-%m-%d"),  # ISO date (YYYY-MM-DD)
        (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', "%Y-%m-%dT%H:%M:%S"),  # ISO datetime
        (r'^\d{1,2}/\d{1,2}/\d{4}$', "%m/%d/%Y"),  # US date (MM/DD/YYYY)
        (r'^\d{1,2}-\d{1,2}-\d{4}$', "%m-%d-%Y"),  # US date with hyphens
        (r'^\d{1,2}/\d{1,2}/\d{2}$', "%m/%d/%y"),  # US date, 2-digit year
        (r'^\d{4}/\d{1,2}/\d{1,2}$', "%Y/%m/%d"),  # Year first with slashes
        (r'^[A-Za-z]{3,9} \d{1,2}, \d{4}$', "%B %d, %Y"),  # Month name, day, year
        (r'^\d{1,2} [A-Za-z]{3,9} \d{4}$', "%d %B %Y"),  # Day, month name, year
    ]
    
    for pattern, fmt in patterns:
        if re.match(pattern, date_str):
            return fmt
    
    return None

def convert_to_iso(dt: Optional[Union[datetime, date]]) -> Optional[str]:
    """
    Convert a datetime object to ISO 8601 format.
    
    Args:
        dt: Datetime or date object to convert
        
    Returns:
        ISO 8601 formatted string (YYYY-MM-DDTHH:MM:SS)
    """
    if dt is None:
        return None
        
    # If it's a date object, convert to datetime
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime.combine(dt, datetime.min.time())
        
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def extract_date_parts(dt: Optional[datetime]) -> Dict[str, Any]:
    """
    Extract individual parts from a datetime object.
    
    Args:
        dt: Datetime object to extract from
        
    Returns:
        Dictionary of date parts
    """
    if dt is None:
        return {}
        
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    return {
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
        "second": dt.second,
        "month_name": month_names[dt.month - 1],
        "month_short": month_names[dt.month - 1][:3],
        "day_of_week": dt.strftime("%A"),
        "day_of_week_short": dt.strftime("%a"),
        "iso_date": dt.strftime("%Y-%m-%d"),
        "iso_time": dt.strftime("%H:%M:%S"),
        "is_weekend": dt.weekday() >= 5,  # 5=Saturday, 6=Sunday
    }

def standardize_date_separators(date_str: Optional[str]) -> Optional[str]:
    """
    Standardize date separators to a consistent format.
    
    Args:
        date_str: Date string to standardize
        
    Returns:
        Date string with standardized separators
    """
    if date_str is None:
        return None
    
    # For ISO format (YYYY-MM-DD or YYYY/MM/DD)
    if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', date_str):
        return re.sub(r'/', '-', date_str)
    
    # For US/European format (MM/DD/YYYY or DD/MM/YYYY)
    if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$', date_str):
        return re.sub(r'-', '/', date_str)
    
    # If no specific format is detected, return as-is
    return date_str


# ===== Price and Currency Utilities =====

def extract_currency_symbol(price_str: Optional[str]) -> Optional[str]:
    """
    Extract currency symbol or code from a price string.
    
    Args:
        price_str: Price string to analyze
        
    Returns:
        Currency code (e.g., "USD", "EUR")
    """
    if not price_str:
        return None
    
    # Map of common currency symbols to codes
    currency_map = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₹": "INR",
        "₽": "RUB",
        "₩": "KRW",
        "₿": "BTC",
        "CA$": "CAD",
        "A$": "AUD",
        "HK$": "HKD",
    }
    
    # Check for currency symbols
    for symbol, code in currency_map.items():
        if symbol in price_str:
            return code
    
    # Check for currency codes
    currency_codes = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "INR", "BTC"]
    for code in currency_codes:
        if code in price_str.upper():
            return code
    
    return None

def extract_numeric_value(value_str: Optional[str]) -> Optional[float]:
    """
    Extract numeric value from a string, handling different formats.
    
    Args:
        value_str: String containing a numeric value
        
    Returns:
        Extracted numeric value as float
    """
    if not value_str:
        return None
    
    # Convert to string if not already
    if not isinstance(value_str, str):
        value_str = str(value_str)
    
    # Remove currency symbols and other non-numeric characters except . and ,
    cleaned = re.sub(r'[^\d.,]', '', value_str)
    
    # Handle different numeric formats
    try:
        # US format: 1,234.56
        if re.search(r'\d,\d{3}', cleaned):
            cleaned = cleaned.replace(',', '')
            return float(cleaned)
        
        # European format: 1.234,56
        if re.search(r'\d\.\d{3}', cleaned) and ',' in cleaned:
            cleaned = cleaned.replace('.', '').replace(',', '.')
            return float(cleaned)
        
        # Simple format with comma as decimal: 1234,56
        if ',' in cleaned and '.' not in cleaned:
            cleaned = cleaned.replace(',', '.')
            return float(cleaned)
        
        # Simple format with period as decimal: 1234.56
        return float(cleaned)
    
    except ValueError:
        return None

def convert_currency(amount: Optional[float], from_currency: Optional[str], 
                     to_currency: Optional[str]) -> Optional[float]:
    """
    Convert an amount from one currency to another.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code
        to_currency: Target currency code
        
    Returns:
        Converted amount
    """
    if amount is None or from_currency is None or to_currency is None:
        return None
    
    # If same currency, no conversion needed
    if from_currency == to_currency:
        return amount
    
    # Default conversion rates (as of a fixed date)
    # In a real implementation, these should be fetched from an API
    rates = {
        "USD": 1.0,      # Base currency
        "EUR": 0.85,     # 1 USD = 0.85 EUR
        "GBP": 0.75,     # 1 USD = 0.75 GBP
        "JPY": 110.0,    # 1 USD = 110 JPY
        "CAD": 1.25,     # 1 USD = 1.25 CAD
        "AUD": 1.35,     # 1 USD = 1.35 AUD
        "CHF": 0.92,     # 1 USD = 0.92 CHF
        "CNY": 6.45,     # 1 USD = 6.45 CNY
        "INR": 74.5,     # 1 USD = 74.5 INR
        "BTC": 0.000024, # 1 USD = 0.000024 BTC
    }
    
    if from_currency not in rates or to_currency not in rates:
        return None
    
    # Convert via USD as the base currency
    usd_amount = amount / rates[from_currency]
    target_amount = usd_amount * rates[to_currency]
    
    return round(target_amount, 2)


# ===== Unit Measurement Utilities =====

def standardize_units(value: Optional[float], unit: Optional[str], 
                      target_system: str = "metric") -> Optional[Dict[str, Any]]:
    """
    Standardize units of measurement to a consistent system.
    
    Args:
        value: Numeric value
        unit: Unit of measurement
        target_system: Target unit system ("metric" or "imperial")
        
    Returns:
        Dictionary with standardized value and unit information
    """
    if value is None or not unit:
        return None
    
    # Normalize unit string
    unit = unit.lower().strip()
    
    # Detect the unit type
    unit_type = "unknown"
    if unit in ["mm", "cm", "m", "km", "inch", "in", "ft", "foot", "feet", "yd", "yard", "mi", "mile"]:
        unit_type = "length"
    elif unit in ["g", "gram", "kg", "kilogram", "oz", "ounce", "lb", "pound"]:
        unit_type = "weight"
    elif unit in ["ml", "l", "liter", "litre", "fl oz", "fluid ounce", "pint", "pt", "quart", "qt", "gal", "gallon"]:
        unit_type = "volume"
    elif unit in ["c", "celsius", "f", "fahrenheit", "k", "kelvin"]:
        unit_type = "temperature"
    elif unit in ["m2", "sq m", "square meter", "sq ft", "square foot", "square feet", "acre", "hectare"]:
        unit_type = "area"
    
    # Get the current unit system
    current_system = detect_unit_system(value, unit)
    
    # If already in the target system, return as is
    if current_system == target_system:
        return {
            "value": value,
            "unit": unit,
            "system": current_system,
            "type": unit_type
        }
    
    # Conversion factors for common units
    conversion_result = {
        "original_value": value,
        "original_unit": unit,
        "original_system": current_system,
        "system": target_system,
        "type": unit_type
    }
    
    # Perform conversion based on unit type
    if unit_type == "length":
        if target_system == "metric":
            # Convert to centimeters
            if unit in ["inch", "in"]:
                conversion_result["converted_value"] = value * 2.54
                conversion_result["unit"] = "cm"
            elif unit in ["ft", "foot", "feet"]:
                conversion_result["converted_value"] = value * 30.48
                conversion_result["unit"] = "cm"
            elif unit in ["yd", "yard"]:
                conversion_result["converted_value"] = value * 91.44
                conversion_result["unit"] = "cm"
            elif unit in ["mi", "mile"]:
                conversion_result["converted_value"] = value * 1.60934
                conversion_result["unit"] = "km"
        else:  # target_system == "imperial"
            # Convert to inches
            if unit == "mm":
                conversion_result["converted_value"] = value * 0.0393701
                conversion_result["unit"] = "inch"
            elif unit == "cm":
                conversion_result["converted_value"] = value * 0.393701
                conversion_result["unit"] = "inch"
            elif unit == "m":
                conversion_result["converted_value"] = value * 39.3701
                conversion_result["unit"] = "inch"
            elif unit == "km":
                conversion_result["converted_value"] = value * 0.621371
                conversion_result["unit"] = "mile"
    
    elif unit_type == "weight":
        if target_system == "metric":
            # Convert to grams
            if unit in ["oz", "ounce"]:
                conversion_result["converted_value"] = value * 28.3495
                conversion_result["unit"] = "g"
            elif unit in ["lb", "pound"]:
                conversion_result["converted_value"] = value * 453.592
                conversion_result["unit"] = "g"
        else:  # target_system == "imperial"
            # Convert to ounces
            if unit == "g" or unit == "gram":
                conversion_result["converted_value"] = value * 0.035274
                conversion_result["unit"] = "oz"
            elif unit in ["kg", "kilogram"]:
                conversion_result["converted_value"] = value * 2.20462
                conversion_result["unit"] = "lb"
    
    elif unit_type == "volume":
        if target_system == "metric":
            # Convert to milliliters
            if unit in ["fl oz", "fluid ounce"]:
                conversion_result["converted_value"] = value * 29.5735
                conversion_result["unit"] = "ml"
            elif unit in ["pint", "pt"]:
                conversion_result["converted_value"] = value * 473.176
                conversion_result["unit"] = "ml"
            elif unit in ["quart", "qt"]:
                conversion_result["converted_value"] = value * 946.353
                conversion_result["unit"] = "ml"
            elif unit in ["gal", "gallon"]:
                conversion_result["converted_value"] = value * 3785.41
                conversion_result["unit"] = "ml"
        else:  # target_system == "imperial"
            # Convert to fluid ounces
            if unit == "ml":
                conversion_result["converted_value"] = value * 0.033814
                conversion_result["unit"] = "fl oz"
            elif unit in ["l", "liter", "litre"]:
                conversion_result["converted_value"] = value * 33.814
                conversion_result["unit"] = "fl oz"
    
    elif unit_type == "temperature":
        if target_system == "metric":
            # Convert to Celsius
            if unit in ["f", "fahrenheit"]:
                conversion_result["converted_value"] = (value - 32) * 5/9
                conversion_result["unit"] = "c"
            elif unit in ["k", "kelvin"]:
                conversion_result["converted_value"] = value - 273.15
                conversion_result["unit"] = "c"
        else:  # target_system == "imperial"
            # Convert to Fahrenheit
            if unit in ["c", "celsius"]:
                conversion_result["converted_value"] = (value * 9/5) + 32
                conversion_result["unit"] = "f"
            elif unit in ["k", "kelvin"]:
                conversion_result["converted_value"] = (value - 273.15) * 9/5 + 32
                conversion_result["unit"] = "f"
    
    elif unit_type == "area":
        if target_system == "metric":
            # Convert to square meters
            if unit in ["sq ft", "square foot", "square feet"]:
                conversion_result["converted_value"] = value * 0.092903
                conversion_result["unit"] = "m2"
            elif unit in ["acre"]:
                conversion_result["converted_value"] = value * 4046.86
                conversion_result["unit"] = "m2"
        else:  # target_system == "imperial"
            # Convert to square feet
            if unit in ["m2", "sq m", "square meter"]:
                conversion_result["converted_value"] = value * 10.7639
                conversion_result["unit"] = "sq ft"
            elif unit in ["hectare"]:
                conversion_result["converted_value"] = value * 107639
                conversion_result["unit"] = "sq ft"
    
    # Round the converted value for better readability
    if "converted_value" in conversion_result:
        conversion_result["converted_value"] = round(conversion_result["converted_value"], 3)
    
    return conversion_result

def detect_unit_system(value: Optional[float], unit: Optional[str]) -> str:
    """
    Detect the unit system of a measurement.
    
    Args:
        value: Numeric value
        unit: Unit of measurement
        
    Returns:
        Unit system ("metric", "imperial", or "unknown")
    """
    if unit is None:
        return "unknown"
    
    # Normalize unit string
    unit = unit.lower().strip()
    
    # Metric units
    metric_units = ["mm", "cm", "m", "km", "g", "gram", "kg", "kilogram", 
                    "ml", "l", "liter", "litre", "c", "celsius", "k", "kelvin",
                    "m2", "sq m", "square meter", "hectare"]
    
    # Imperial units
    imperial_units = ["inch", "in", "ft", "foot", "feet", "yd", "yard", "mi", "mile",
                     "oz", "ounce", "lb", "pound", "fl oz", "fluid ounce", 
                     "pint", "pt", "quart", "qt", "gal", "gallon", "f", "fahrenheit",
                     "sq ft", "square foot", "square feet", "acre"]
    
    for metric_unit in metric_units:
        if metric_unit in unit:
            return "metric"
    
    for imperial_unit in imperial_units:
        if imperial_unit in unit:
            return "imperial"
    
    return "unknown"


# ===== Pattern Matching Utilities =====

def is_email(text: str) -> bool:
    """Check if text is a valid email address."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text))

def is_url(text: str) -> bool:
    """Check if text is a valid URL."""
    url_pattern = r'^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
    return bool(re.match(url_pattern, text))

def is_phone_number(text: str) -> bool:
    """Check if text is a phone number."""
    # This is a simplified pattern, real phone validation is more complex
    phone_pattern = r'^\+?[\d\s\(\)\-\.]{7,20}$'
    return bool(re.match(phone_pattern, text))

def is_date(text: str) -> bool:
    """Check if text is a date."""
    return parse_date(text) is not None

def is_price(text: str) -> bool:
    """Check if text represents a price."""
    # Look for currency symbols or formats like $123.45, 123.45 USD
    price_pattern = r'^(\p{Sc}|[A-Z]{3}\s?)?[\d\s,\.]+(\s?[A-Z]{3})?$'
    return bool(re.search(price_pattern, text, re.UNICODE))

def is_measurement(text: str) -> bool:
    """Check if text represents a measurement with units."""
    # Look for number followed by units like 10kg, 10 cm, 10.5 inches
    measurement_pattern = r'^[\d\s,\.]+\s?([a-zA-Z]+\.?|[²³°])$'
    return bool(re.match(measurement_pattern, text))

def is_html(text: str) -> bool:
    """Check if text contains HTML markup."""
    return '<' in text and '>' in text and re.search(r'<[a-z]+[^>]*>', text, re.IGNORECASE) is not None

def extract_pattern_match(text: str, pattern: str) -> Optional[str]:
    """
    Extract text matching a specific regex pattern.
    
    Args:
        text: Text to search in
        pattern: Regex pattern to match
        
    Returns:
        Matched string or None if no match
    """
    match = re.search(pattern, text)
    return match.group(0) if match else None


# ===== Name Parsing Utilities =====

def parse_name(name: str) -> Dict[str, str]:
    """
    Parse a name into components.
    
    Args:
        name: Full name to parse
        
    Returns:
        Dictionary with name parts
    """
    parts = name.strip().split()
    result = {"full": name.strip()}
    
    # Handle empty name
    if not parts:
        return {"full": "", "valid": False}
    
    # Check for title (Dr., Mr., Mrs., etc.)
    titles = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Rev.", "Sir", "Madam"]
    if parts[0] in titles:
        result["title"] = parts[0]
        parts = parts[1:]
    
    # Check for suffix (Jr., Sr., III, etc.)
    suffixes = ["Jr.", "Sr.", "I", "II", "III", "IV", "V", "PhD", "MD", "DDS"]
    if parts and parts[-1] in suffixes:
        result["suffix"] = parts[-1]
        parts = parts[:-1]
    
    # Remaining parts are first, middle, last names
    if len(parts) == 1:
        result["first"] = parts[0]
    elif len(parts) == 2:
        result["first"] = parts[0]
        result["last"] = parts[1]
    elif len(parts) >= 3:
        result["first"] = parts[0]
        result["last"] = parts[-1]
        result["middle"] = " ".join(parts[1:-1])
    
    result["valid"] = True
    return result


# ===== Address Parsing Utilities =====

def parse_us_address(address: str) -> Dict[str, str]:
    """
    Parse a US address into components.
    
    Args:
        address: Address string to parse
        
    Returns:
        Dictionary with address parts
    """
    if not address:
        return {"valid": False}
    
    result = {"full": address.strip(), "valid": True}
    
    # Try to extract postal code
    postal_code_match = re.search(r'(\d{5}(-\d{4})?)', address)
    if postal_code_match:
        result["postal_code"] = postal_code_match.group(1)
    
    # Try to extract state
    state_match = re.search(r'\b([A-Z]{2})\b', address)
    if state_match:
        result["state"] = state_match.group(1)
    
    # Try to extract city
    if "state" in result:
        city_match = re.search(r',\s+([^,]+),\s+' + result["state"], address)
        if city_match:
            result["city"] = city_match.group(1)
    
    # Try to extract street
    if "city" in result:
        street_parts = address.split(", " + result["city"])
        if street_parts:
            result["street"] = street_parts[0].strip()
    
    return result


# ===== Phone Number Parsing Utilities =====

def parse_phone_number(phone: str) -> Dict[str, str]:
    """
    Parse a phone number into components.
    
    Args:
        phone: Phone number string to parse
        
    Returns:
        Dictionary with phone number parts
    """
    if not phone:
        return {"valid": False}
    
    # Strip all non-digit characters to get raw digits
    digits = re.sub(r'\D', '', phone)
    
    result = {
        "full": phone.strip(),
        "digits": digits,
        "valid": len(digits) >= 7  # Basic validation, should have at least 7 digits
    }
    
    # Try to extract country code
    if len(digits) > 10 and digits.startswith("1"):
        result["country_code"] = "1"  # US
        digits = digits[1:]
    elif len(digits) > 10:
        # Assume first 1-3 digits are country code for international numbers
        result["country_code"] = digits[:min(3, len(digits) - 7)]
        digits = digits[len(result["country_code"]):]
    else:
        # Default to US if no country code is found
        result["country_code"] = "1"
    
    # Extract area code (3 digits in North America)
    if len(digits) >= 10:
        result["area_code"] = digits[:3]
        digits = digits[3:]
    
    # Extract local number
    if len(digits) >= 7:
        result["local_number"] = digits
    
    return result


# ===== URL Parsing Utilities =====

def parse_url(url: str) -> Dict[str, str]:
    """
    Parse a URL into components.
    
    Args:
        url: URL string to parse
        
    Returns:
        Dictionary with URL parts
    """
    if not url:
        return {"valid": False}
    
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urllib.parse.urlparse(url)
        
        result = {
            "full": url,
            "scheme": parsed.scheme,
            "domain": parsed.netloc,
            "path": parsed.path if parsed.path else "/",
            "valid": bool(parsed.netloc)  # Valid if domain exists
        }
        
        if parsed.query:
            result["query"] = parsed.query
            result["query_params"] = dict(urllib.parse.parse_qsl(parsed.query))
        
        if parsed.fragment:
            result["fragment"] = parsed.fragment
        
        return result
        
    except Exception:
        return {"full": url, "valid": False}


# ===== Identifier Parsing Utilities =====

def validate_isbn13(isbn: str) -> bool:
    """
    Validate ISBN-13 check digit.
    
    Args:
        isbn: ISBN-13 string
        
    Returns:
        True if valid, False otherwise
    """
    # Remove hyphens and spaces
    isbn = re.sub(r'[-\s]', '', isbn)
    
    # Check if it's 13 digits
    if not re.match(r'^\d{13}$', isbn):
        return False
    
    # Calculate check digit
    sum_odd = sum(int(isbn[i]) for i in range(0, 12, 2))
    sum_even = sum(int(isbn[i]) * 3 for i in range(1, 12, 2))
    total = sum_odd + sum_even
    check = (10 - (total % 10)) % 10
    
    # Validate check digit
    return int(isbn[12]) == check

def validate_uuid(uuid: str) -> bool:
    """
    Validate a UUID string.
    
    Args:
        uuid: UUID string
        
    Returns:
        True if valid, False otherwise
    """
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, uuid.lower()))

def parse_identifier(identifier: str, id_type: str = None) -> Dict[str, Any]:
    """
    Parse and validate various identifier types.
    
    Args:
        identifier: Identifier string
        id_type: Type of identifier (isbn13, uuid, etc.)
        
    Returns:
        Dictionary with identifier validation info
    """
    if not identifier:
        return {"valid": False}
    
    # Clean identifier
    clean_id = re.sub(r'[-\s]', '', identifier)
    result = {"raw": identifier, "cleaned": clean_id}
    
    # Auto-detect type if not provided
    if not id_type:
        if len(clean_id) == 13 and clean_id.isdigit():
            id_type = "isbn13"
        elif re.match(r'^[0-9a-f]{32}$', clean_id.lower()):
            id_type = "uuid"
        else:
            id_type = "unknown"
    
    result["type"] = id_type
    
    # Validate based on type
    if id_type == "isbn13":
        result["valid"] = validate_isbn13(clean_id)
        if result["valid"]:
            # Format with hyphens for readability
            result["formatted"] = f"{clean_id[0:3]}-{clean_id[3:4]}-{clean_id[4:9]}-{clean_id[9:12]}-{clean_id[12]}"
    
    elif id_type == "uuid":
        # Format as UUID if it's just a hex string
        if re.match(r'^[0-9a-f]{32}$', clean_id.lower()):
            formatted = f"{clean_id[0:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:32]}"
            result["valid"] = True
            result["formatted"] = formatted.lower()
        else:
            result["valid"] = validate_uuid(identifier)
            result["formatted"] = identifier.lower() if result["valid"] else None
    
    else:
        # For unknown types, just store the cleaned version
        result["valid"] = True
        result["formatted"] = clean_id
    
    return result


# ===== Boolean Normalization Utilities =====

def normalize_boolean_value(value: Any) -> bool:
    """
    Normalize any value to a boolean.
    
    Args:
        value: Value to normalize
        
    Returns:
        Normalized boolean value
    """
    if value is None:
        return False
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return value != 0
    
    if isinstance(value, str):
        value = value.lower().strip()
        
        # Handle common "true" strings
        true_values = ["yes", "true", "t", "y", "1", "on", "enabled", "enable"]
        if value in true_values:
            return True
        
        # Handle common "false" strings
        false_values = ["no", "false", "f", "n", "0", "off", "disabled", "disable"]
        if value in false_values:
            return False
        
        # Non-empty string that doesn't match known false values is considered true
        return value != ""
    
    # Non-empty collections (list, dict, etc.) are considered true
    if hasattr(value, "__len__"):
        return len(value) > 0
    
    # Default to True for objects that exist
    return True