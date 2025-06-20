"""
Quality Metrics Module

This module provides utility functions for calculating various quality metrics
for extracted data, including confidence scores, plausibility checks, and
validation functions.
"""

import re
import math
import string
import logging
from typing import Dict, Any, List, Union, Optional, Tuple, Set
from datetime import datetime
import dateutil.parser
from collections import Counter

logger = logging.getLogger(__name__)

def calculate_text_quality(text: str) -> float:
    """
    Evaluate the quality of text content.
    
    Args:
        text: Text to evaluate
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    if not text:
        return 0.0
        
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Initialize quality score
    quality = 1.0
    
    # Check for very short text
    if len(text) < 3:
        quality *= 0.5
    
    # Check for very long text
    if len(text) > 10000:
        quality *= 0.8  # Long text is not necessarily bad
    
    # Check for abnormal character distributions
    char_counts = Counter(text.lower())
    
    # Calculate letter percentage
    letter_count = sum(char_counts.get(c, 0) for c in string.ascii_lowercase)
    letter_percentage = letter_count / max(1, len(text))
    
    # Penalize text with too few letters
    if letter_percentage < 0.3:
        quality *= 0.5 + (letter_percentage)
    
    # Check for repetitive characters
    most_common_char, most_common_count = char_counts.most_common(1)[0] if char_counts else ('', 0)
    char_repetition_ratio = most_common_count / max(1, len(text))
    
    if char_repetition_ratio > 0.5:
        quality *= 0.5
    
    # Check for capitalization consistency
    capitalization_issues = 0
    words = text.split()
    
    # Skip for very short texts
    if len(words) >= 3:
        # Check for ALL CAPS
        all_caps = sum(1 for word in words if word.isupper() and len(word) > 1)
        all_caps_ratio = all_caps / len(words)
        
        # Check for weird capitalization
        weird_caps = sum(1 for word in words if not word.isupper() and not word.islower() 
                        and not word[0].isupper() and len(word) > 1)
        weird_caps_ratio = weird_caps / len(words)
        
        if all_caps_ratio > 0.8:
            capitalization_issues += 0.5  # ALL CAPS text is often lower quality
        
        if weird_caps_ratio > 0.3:
            capitalization_issues += 0.5  # Weird capitalization suggests poor quality
        
        quality *= max(0.3, 1.0 - capitalization_issues)
    
    # Check for HTML remnants
    if re.search(r'</?[a-z]+[^>]*>', text, re.IGNORECASE):
        quality *= 0.6
    
    # Check for code/JSON fragments
    if ('{' in text and '}' in text) or ('[' in text and ']' in text):
        if re.search(r'"[a-z_]+":\s*', text, re.IGNORECASE):
            quality *= 0.7
    
    # Check for URL quality if this is a URL
    if re.match(r'^https?://', text):
        # URLs with more path segments are often better
        segments = text.count('/')
        if segments <= 3:
            quality *= 0.8
    
    # Check for common error patterns
    error_patterns = [
        r'\b(undefined|null|none|nan|error|not found)\b',
        r'\b404\b',
        r'\bnot available\b',
        r'\bcoming soon\b'
    ]
    
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in error_patterns):
        quality *= 0.4
    
    # Ensure final score is in 0.0-1.0 range
    return max(0.0, min(1.0, quality))

def calculate_field_confidence(value: Any, expected_pattern: Optional[str] = None) -> float:
    """
    Calculate confidence score for a field value based on expected patterns.
    
    Args:
        value: Field value to evaluate
        expected_pattern: Optional regex pattern the value should match
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Handle None/empty values
    if value is None:
        return 0.0
    
    # Convert to string for pattern matching
    if not isinstance(value, str):
        # For non-string types, if we have a specific pattern, confidence is lower
        if expected_pattern:
            return 0.3
        
        # Otherwise, non-string types are usually reliable
        return 0.9
    
    # Empty strings have zero confidence
    if value == "":
        return 0.0
    
    # Very short strings have lower confidence
    if len(value) < 3:
        return 0.5
    
    # Check if the value matches the expected pattern
    if expected_pattern:
        try:
            if re.match(expected_pattern, value):
                return 0.95  # High confidence for pattern match
            else:
                return 0.2  # Low confidence for pattern mismatch
        except:
            # In case of regex error
            logger.warning(f"Invalid regex pattern: {expected_pattern}")
    
    # Default confidence for strings without pattern
    base_confidence = 0.7
    
    # Adjust confidence based on string characteristics
    
    # Check for common error indicators
    error_indicators = ["null", "undefined", "none", "n/a", "error", "not found", "not available"]
    if any(value.lower() == indicator for indicator in error_indicators):
        return 0.1
    
    # Check for potentially invalid characters
    invalid_chars = sum(1 for c in value if ord(c) < 32 or ord(c) > 126)
    if invalid_chars > 0:
        base_confidence *= max(0.2, 1.0 - (invalid_chars / len(value)))
    
    # Check for very long strings (potential data issues)
    if len(value) > 1000:
        base_confidence *= 0.9
    
    # Bonus for well-formatted values
    if re.match(r'^[A-Z][a-z]+((\s|\-)[A-Z][a-z]+)*$', value):  # Proper names
        base_confidence *= 1.2
    
    # Ensure final score is in 0.0-1.0 range
    return max(0.0, min(1.0, base_confidence))

def measure_numerical_plausibility(value: Union[int, float], 
                                expected_range: Optional[Tuple[float, float]] = None) -> float:
    """
    Check if a numeric value is plausible (within expected range).
    
    Args:
        value: Numeric value to check
        expected_range: Optional tuple with (min, max) expected range
        
    Returns:
        Plausibility score between 0.0 and 1.0
    """
    if not isinstance(value, (int, float)):
        return 0.0
    
    # Check for NaN or infinity
    if math.isnan(value) or math.isinf(value):
        return 0.0
    
    # If range is provided, check if value is within range
    if expected_range:
        min_val, max_val = expected_range
        
        if min_val <= value <= max_val:
            # Inside range - fully plausible
            return 1.0
        
        # Outside range - calculate how far outside
        if value < min_val:
            # Calculate how far below the range
            distance_ratio = (min_val - value) / max(1.0, min_val)
            # For small deviations, score is still relatively high
            if distance_ratio < 0.5:
                return max(0.0, 1.0 - distance_ratio)
            else:
                return max(0.0, 0.5 - (distance_ratio * 0.5))
        else:  # value > max_val
            # Calculate how far above the range
            distance_ratio = (value - max_val) / max(1.0, max_val)
            # For small deviations, score is still relatively high
            if distance_ratio < 0.5:
                return max(0.0, 1.0 - distance_ratio)
            else:
                return max(0.0, 0.5 - (distance_ratio * 0.5))
    
    # Without expected range, use heuristics
    
    # Extremely large values are suspicious
    if abs(value) > 1e10:
        return 0.3
    
    # Negative values are suspicious for many fields
    if value < 0:
        return 0.5
    
    # Round numbers are slightly more likely to be defaults/placeholders
    if value > 10 and value % 10 == 0:
        return 0.8
    
    # Default case - neutral plausibility
    return 0.9

def check_date_validity(date_str: str) -> float:
    """
    Validate a date string.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        Validity score between 0.0 and 1.0
    """
    if not date_str or not isinstance(date_str, str):
        return 0.0
    
    # Clean the date string
    date_str = date_str.strip()
    
    # Check if it's an empty string
    if not date_str:
        return 0.0
    
    # Check for obvious date formats
    date_formats = [
        r'^\d{4}-\d{2}-\d{2}$',  # ISO format (YYYY-MM-DD)
        r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
        r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
        r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
        r'^\d{1,2}\s+[A-Za-z]+\s+\d{4}$',  # D Month YYYY
        r'^[A-Za-z]+\s+\d{1,2},\s+\d{4}$'  # Month D, YYYY
    ]
    
    # Check for format match
    format_score = 0.0
    for pattern in date_formats:
        if re.match(pattern, date_str):
            format_score = 0.9
            break
    
    # If no format match, check for date-like content
    if format_score == 0.0:
        # Check for year presence
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            format_score = 0.3
            
            # Check for month names
            months = ["jan", "feb", "mar", "apr", "may", "jun", 
                      "jul", "aug", "sep", "oct", "nov", "dec"]
            if any(month in date_str.lower() for month in months):
                format_score = 0.5
                
            # Check for day presence
            if re.search(r'\b\d{1,2}\b', date_str):
                format_score = 0.6
    
    # Try to parse the date
    validation_score = 0.0
    try:
        parsed_date = dateutil.parser.parse(date_str, fuzzy=True)
        
        # If we get here, the date is at least parseable
        validation_score = 0.7
        
        # Check if date is reasonable (not too far in the past or future)
        current_year = datetime.now().year
        if 1900 <= parsed_date.year <= current_year + 10:
            validation_score = 1.0
        else:
            # Date is technically valid but not reasonable
            validation_score = 0.4
            
    except (ValueError, OverflowError):
        # Not parseable
        validation_score = 0.0
    
    # Combine format and validation scores, with validation weighted higher
    return (format_score * 0.4) + (validation_score * 0.6)

def check_url_validity(url: str) -> float:
    """
    Validate a URL string.
    
    Args:
        url: URL string to validate
        
    Returns:
        Validity score between 0.0 and 1.0
    """
    if not url or not isinstance(url, str):
        return 0.0
    
    # Clean the URL string
    url = url.strip()
    
    # Check if it's an empty string
    if not url:
        return 0.0
    
    # Basic URL pattern
    basic_pattern = r'^(https?://)?([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(/.*)?$'
    
    # More strict URL pattern
    strict_pattern = r'^https?://([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(/[a-zA-Z0-9._~:/?#[\]@!$&\'()*+,;=%-]*)?$'
    
    # Check for basic pattern match
    if re.match(basic_pattern, url):
        # Basic pattern matches
        basic_score = 0.7
        
        # Check for scheme
        if url.startswith(('http://', 'https://')):
            basic_score = 0.8
            
            # HTTPS is slightly more valid than HTTP
            if url.startswith('https://'):
                basic_score = 0.9
        
        # Check for strict pattern match
        if re.match(strict_pattern, url):
            return min(1.0, basic_score + 0.1)
            
        return basic_score
    
    # No match, but check for partial URL validity
    if '.' in url and not url.startswith('@'):
        domain_parts = url.split('.')
        if len(domain_parts) >= 2 and all(part.strip() for part in domain_parts):
            # Might be a domain without scheme
            tld = domain_parts[-1].lower()
            
            # Check for common TLDs
            common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'io', 'co', 'ai', 'app', 'dev']
            if tld in common_tlds or len(tld) == 2:  # Country code TLDs
                return 0.5
            else:
                return 0.3
    
    # Not a valid URL
    return 0.0

def measure_enum_validity(value: Any, allowed_values: List[Any]) -> float:
    """
    Check if a value belongs to a set of allowed values.
    
    Args:
        value: Value to check
        allowed_values: List of allowed values
        
    Returns:
        Validity score between 0.0 and 1.0
    """
    if not allowed_values:
        return 0.5  # No allowed values to check against
    
    # Direct match
    if value in allowed_values:
        return 1.0
    
    # For string values, try case-insensitive matching
    if isinstance(value, str):
        value_lower = value.lower()
        if any(str(allowed).lower() == value_lower for allowed in allowed_values):
            return 0.9  # Case-insensitive match
        
        # Check for partial matches
        partial_matches = [
            allowed for allowed in allowed_values 
            if isinstance(allowed, str) and (
                allowed.lower() in value_lower or value_lower in allowed.lower()
            )
        ]
        
        if partial_matches:
            # Stronger match if value is contained in allowed value
            for match in partial_matches:
                if value_lower in str(match).lower():
                    return 0.7
            
            # Weaker match if allowed value is contained in value
            return 0.5
    
    # For numeric values, check if value is close to any allowed value
    if isinstance(value, (int, float)):
        if any(isinstance(allowed, (int, float)) for allowed in allowed_values):
            numeric_allowed = [allowed for allowed in allowed_values if isinstance(allowed, (int, float))]
            closest = min(numeric_allowed, key=lambda x: abs(x - value))
            ratio = min(value, closest) / max(value, closest) if max(value, closest) != 0 else 0
            
            # If values are very close (within 10%)
            if ratio > 0.9:
                return 0.6
    
    # No match
    return 0.0

def check_field_relationships(data: Dict[str, Any], relationships: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Check relationships between fields in the data.
    
    Args:
        data: Data dictionary to check
        relationships: List of relationship definitions to check
        
    Returns:
        Dictionary mapping relationship IDs to validity scores
    """
    results = {}
    
    for rel in relationships:
        rel_id = rel.get("id")
        rel_type = rel.get("type")
        fields = rel.get("fields", [])
        
        # Skip if missing required information
        if not rel_id or not rel_type or not fields:
            continue
            
        # Skip if not all fields are present
        if not all(field in data for field in fields):
            results[rel_id] = 0.0
            continue
        
        # Check relationship based on type
        if rel_type == "dependency":
            # First field depends on second field
            if len(fields) >= 2:
                source = fields[0]
                target = fields[1]
                
                if data[source] and not data[target]:
                    results[rel_id] = 0.0
                else:
                    results[rel_id] = 1.0
        
        elif rel_type == "numeric_comparison":
            # Compare numeric values
            if len(fields) >= 2:
                field1 = fields[0]
                field2 = fields[1]
                condition = rel.get("condition", "<")
                
                # Check if both fields have numeric values
                if isinstance(data[field1], (int, float)) and isinstance(data[field2], (int, float)):
                    if condition == "<" and data[field1] < data[field2]:
                        results[rel_id] = 1.0
                    elif condition == "<=" and data[field1] <= data[field2]:
                        results[rel_id] = 1.0
                    elif condition == ">" and data[field1] > data[field2]:
                        results[rel_id] = 1.0
                    elif condition == ">=" and data[field1] >= data[field2]:
                        results[rel_id] = 1.0
                    elif condition == "==" and data[field1] == data[field2]:
                        results[rel_id] = 1.0
                    else:
                        # How far off is the relationship
                        if condition in ["<", "<="]:
                            if data[field1] > data[field2]:
                                ratio = data[field2] / data[field1] if data[field1] != 0 else 0
                                results[rel_id] = max(0.0, ratio)
                            else:
                                results[rel_id] = 1.0
                        elif condition in [">", ">="]:
                            if data[field1] < data[field2]:
                                ratio = data[field1] / data[field2] if data[field2] != 0 else 0
                                results[rel_id] = max(0.0, ratio)
                            else:
                                results[rel_id] = 1.0
                else:
                    # Not numeric values
                    results[rel_id] = 0.0
        
        elif rel_type == "consistent_units":
            # Check for consistent units across fields
            # Assume fields have similar numeric magnitudes
            numeric_fields = [field for field in fields if isinstance(data[field], (int, float))]
            
            if not numeric_fields:
                results[rel_id] = 0.0
                continue
                
            # Get values and compute order of magnitude
            values = [data[field] for field in numeric_fields]
            magnitudes = [math.floor(math.log10(abs(val))) if val != 0 else 0 for val in values]
            
            # Check if magnitudes are consistent (within 1 order of magnitude)
            if max(magnitudes) - min(magnitudes) <= 1:
                results[rel_id] = 1.0
            else:
                # Calculate consistency based on magnitude differences
                max_diff = max(magnitudes) - min(magnitudes)
                results[rel_id] = max(0.0, 1.0 - (max_diff / 10))
        
        elif rel_type == "format_consistency":
            # Check for consistent formatting across fields
            if all(isinstance(data[field], str) for field in fields):
                # Check for format patterns
                patterns = []
                
                for field in fields:
                    value = data[field]
                    
                    # Detect format pattern
                    if re.match(r'^\d+$', value):
                        patterns.append("number")
                    elif re.match(r'^\d+\.\d+$', value):
                        patterns.append("decimal")
                    elif re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                        patterns.append("iso-date")
                    elif re.match(r'^\d{2}/\d{2}/\d{4}$', value):
                        patterns.append("mm/dd/yyyy")
                    elif re.match(r'^[A-Z][a-z]+$', value):
                        patterns.append("capitalized")
                    elif value.isupper():
                        patterns.append("uppercase")
                    elif value.islower():
                        patterns.append("lowercase")
                    else:
                        patterns.append("other")
                
                # Check consistency
                if len(set(patterns)) == 1:
                    results[rel_id] = 1.0
                else:
                    # Partial consistency
                    results[rel_id] = 0.5
            else:
                results[rel_id] = 0.0
        
        else:
            # Unknown relationship type
            results[rel_id] = 0.5
    
    return results

def calculate_overall_quality_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate an overall quality score from multiple metrics.
    
    Args:
        metrics: Dictionary of quality metrics
        
    Returns:
        Overall quality score between 0.0 and 1.0
    """
    if not metrics:
        return 0.0
    
    # Define weights for different metrics
    weights = {
        "completeness": 0.3,
        "confidence": 0.25,
        "consistency": 0.15,
        "type_validation.valid_rate": 0.2,
        "relevance": 0.1
    }
    
    score_components = {}
    
    # Extract scores from metrics
    for metric_key, weight in weights.items():
        if '.' in metric_key:
            # Handle nested metrics
            outer_key, inner_key = metric_key.split('.', 1)
            if outer_key in metrics and inner_key in metrics[outer_key]:
                score_components[metric_key] = metrics[outer_key][inner_key] * weight
        elif metric_key in metrics:
            if isinstance(metrics[metric_key], (int, float)):
                score_components[metric_key] = metrics[metric_key] * weight
    
    # Calculate total
    if not score_components:
        return 0.5  # Default score if no components available
        
    total_weight = sum(weight for metric_key, weight in weights.items() 
                     if metric_key in score_components)
                     
    if total_weight == 0:
        return 0.5  # Default score if no weights
        
    total_score = sum(score_components.values())
    
    # Normalize by actual weights used
    return total_score / total_weight

def generate_quality_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a statistical profile of the data quality.
    
    Args:
        data: Data to analyze
        
    Returns:
        Dictionary with quality profile
    """
    if not data:
        return {
            "field_count": 0,
            "completeness": 0.0,
            "overall_score": 0.0,
            "field_types": {}
        }
    
    # Exclude metadata fields
    data_fields = {k: v for k, v in data.items() if not k.startswith("_")}
    
    # Count fields
    field_count = len(data_fields)
    
    # Count non-empty fields
    non_empty_count = sum(1 for v in data_fields.values() if v is not None and v != "")
    
    # Calculate completeness
    completeness = non_empty_count / max(1, field_count)
    
    # Count field types
    field_types = {}
    for value in data_fields.values():
        field_type = type(value).__name__
        field_types[field_type] = field_types.get(field_type, 0) + 1
    
    # Calculate percentage for each type
    for field_type in field_types:
        field_types[field_type] = field_types[field_type] / field_count
    
    # Calculate a basic overall score
    # Simple heuristic: completeness * 0.7 + proportion of fields with good types * 0.3
    good_types = sum(field_types.get(t, 0) for t in ["str", "int", "float", "dict", "list"])
    
    overall_score = (completeness * 0.7) + (good_types * 0.3)
    
    return {
        "field_count": field_count,
        "completeness": completeness,
        "overall_score": overall_score,
        "field_types": field_types
    }

def identify_improvement_opportunities(data: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Identify opportunities to improve data quality.
    
    Args:
        data: Data to analyze
        metrics: Quality metrics already calculated
        
    Returns:
        List of improvement opportunities
    """
    opportunities = []
    
    # Check completeness
    completeness = metrics.get("completeness", 0.0)
    if completeness < 0.8:
        missing_count = sum(1 for v in data.values() if v is None or v == "")
        opportunities.append({
            "type": "completeness",
            "message": f"Improve data completeness. {missing_count} fields are empty or null."
        })
    
    # Check confidence
    confidence = metrics.get("confidence", 0.0)
    if confidence < 0.7:
        opportunities.append({
            "type": "confidence",
            "message": "Improve extraction confidence. Consider using more reliable extraction methods."
        })
    
    # Check for anomalies
    if "anomalies" in metrics and metrics["anomalies"]:
        anomaly_count = len(metrics["anomalies"])
        opportunities.append({
            "type": "anomalies",
            "message": f"Address {anomaly_count} anomalous fields that may indicate extraction errors."
        })
    
    # Check for missing required fields
    if "missing_required_fields" in metrics and metrics["missing_required_fields"]:
        missing = metrics["missing_required_fields"]
        if isinstance(missing, list) and missing:
            opportunities.append({
                "type": "required_fields",
                "message": f"Add missing required fields: {', '.join(missing)}"
            })
    
    # Check type validation
    if "type_validation" in metrics:
        type_validation = metrics["type_validation"]
        if "invalid_fields" in type_validation:
            invalid_fields = type_validation["invalid_fields"]
            if invalid_fields:
                opportunities.append({
                    "type": "type_validation",
                    "message": f"Fix type mismatches in {len(invalid_fields)} fields."
                })
    
    # Check text quality
    if "text_quality" in metrics:
        low_quality_text = []
        for field, score in metrics["text_quality"].items():
            if score < 0.6:
                low_quality_text.append(field)
        
        if low_quality_text:
            opportunities.append({
                "type": "text_quality",
                "message": f"Improve text quality in fields: {', '.join(low_quality_text)}"
            })
    
    # Add general recommendations based on data types
    string_fields = sum(1 for v in data.values() if isinstance(v, str))
    numeric_fields = sum(1 for v in data.values() if isinstance(v, (int, float)))
    
    if string_fields > 5 and numeric_fields == 0:
        opportunities.append({
            "type": "data_types",
            "message": "Add numeric fields for better data analytics. Current data is primarily text."
        })
    
    return opportunities

def calculate_schema_compliance_rate(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate how well the data complies with the schema.
    
    Args:
        data: Data to check
        schema: Schema to validate against
        
    Returns:
        Dictionary with compliance metrics
    """
    result = {
        "compliance_rate": 0.0,
        "missing_required_fields": [],
        "type_mismatches": [],
        "constraint_violations": []
    }
    
    # Extract required fields and types from schema
    required_fields = []
    field_types = {}
    constraints = {}
    
    # Handle JSON Schema format
    if "required" in schema:
        required_fields = schema["required"]
    
    if "properties" in schema:
        for field, field_schema in schema["properties"].items():
            if "type" in field_schema:
                field_types[field] = field_schema["type"]
            
            # Extract constraints
            for key, value in field_schema.items():
                if key in ["minimum", "maximum", "minLength", "maxLength", "pattern", "enum"]:
                    if field not in constraints:
                        constraints[field] = []
                    constraints[field].append({"type": key, "value": value})
    
    # Handle custom schema format
    elif "fields" in schema:
        for field in schema["fields"]:
            if field.get("required", False) and "name" in field:
                required_fields.append(field["name"])
            
            if "name" in field and "type" in field:
                field_types[field["name"]] = field["type"]
            
            # Extract constraints
            if "name" in field:
                field_name = field["name"]
                for key, value in field.items():
                    if key in ["min", "max", "min_length", "max_length", "pattern", "enum", "allowed_values"]:
                        if field_name not in constraints:
                            constraints[field_name] = []
                        constraints[field_name].append({"type": key, "value": value})
    
    # Check required fields
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            result["missing_required_fields"].append(field)
    
    # Check field types
    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            value = data[field]
            
            # Check type
            type_matches = False
            
            if expected_type == "string" and isinstance(value, str):
                type_matches = True
            elif expected_type == "number" and isinstance(value, (int, float)):
                type_matches = True
            elif expected_type == "integer" and isinstance(value, int):
                type_matches = True
            elif expected_type == "boolean" and isinstance(value, bool):
                type_matches = True
            elif expected_type == "array" and isinstance(value, (list, tuple)):
                type_matches = True
            elif expected_type == "object" and isinstance(value, dict):
                type_matches = True
            
            if not type_matches:
                result["type_mismatches"].append({
                    "field": field,
                    "expected_type": expected_type,
                    "actual_type": type(value).__name__
                })
    
    # Check constraints
    for field, field_constraints in constraints.items():
        if field in data and data[field] is not None:
            value = data[field]
            
            for constraint in field_constraints:
                constraint_type = constraint["type"]
                constraint_value = constraint["value"]
                
                # Check constraint
                if constraint_type == "minimum" and isinstance(value, (int, float)):
                    if value < constraint_value:
                        result["constraint_violations"].append({
                            "field": field,
                            "constraint": "minimum",
                            "expected": constraint_value,
                            "actual": value
                        })
                
                elif constraint_type == "maximum" and isinstance(value, (int, float)):
                    if value > constraint_value:
                        result["constraint_violations"].append({
                            "field": field,
                            "constraint": "maximum",
                            "expected": constraint_value,
                            "actual": value
                        })
                
                elif constraint_type in ["minLength", "min_length"] and isinstance(value, str):
                    if len(value) < constraint_value:
                        result["constraint_violations"].append({
                            "field": field,
                            "constraint": "minLength",
                            "expected": constraint_value,
                            "actual": len(value)
                        })
                
                elif constraint_type in ["maxLength", "max_length"] and isinstance(value, str):
                    if len(value) > constraint_value:
                        result["constraint_violations"].append({
                            "field": field,
                            "constraint": "maxLength",
                            "expected": constraint_value,
                            "actual": len(value)
                        })
                
                elif constraint_type == "pattern" and isinstance(value, str):
                    if not re.match(constraint_value, value):
                        result["constraint_violations"].append({
                            "field": field,
                            "constraint": "pattern",
                            "expected": constraint_value,
                            "actual": value
                        })
                
                elif constraint_type in ["enum", "allowed_values"]:
                    if value not in constraint_value:
                        result["constraint_violations"].append({
                            "field": field,
                            "constraint": "enum",
                            "expected": constraint_value,
                            "actual": value
                        })
    
    # Calculate compliance rate
    total_checks = len(required_fields) + len(field_types) + sum(len(c) for c in constraints.values())
    violations = len(result["missing_required_fields"]) + len(result["type_mismatches"]) + len(result["constraint_violations"])
    
    if total_checks > 0:
        result["compliance_rate"] = max(0.0, min(1.0, 1.0 - (violations / total_checks)))
    else:
        result["compliance_rate"] = 1.0  # No checks, assume compliant
    
    return result

def detect_outliers(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect statistical outliers in the data.
    
    Args:
        data: Data to analyze
        
    Returns:
        Dictionary mapping field names to outlier information
    """
    outliers = {}
    
    # Group fields by type
    numeric_fields = {k: v for k, v in data.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    string_fields = {k: v for k, v in data.items() if isinstance(v, str) and not k.startswith("_")}
    
    # Analyze numeric fields for outliers
    if len(numeric_fields) >= 3:  # Need at least a few values for meaningful analysis
        values = list(numeric_fields.values())
        
        # Calculate mean and standard deviation
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        
        # Identify outliers (more than 2 standard deviations from mean)
        for field, value in numeric_fields.items():
            z_score = (value - mean) / std_dev if std_dev != 0 else 0
            if abs(z_score) > 2:
                outliers[field] = {
                    "value": value,
                    "z_score": z_score,
                    "mean": mean,
                    "std_dev": std_dev
                }
    
    # Analyze string fields for length outliers
    if len(string_fields) >= 3:
        lengths = [len(v) for v in string_fields.values()]
        
        # Calculate mean and standard deviation of lengths
        mean_length = sum(lengths) / len(lengths)
        std_dev_length = math.sqrt(sum((x - mean_length) ** 2 for x in lengths) / len(lengths))
        
        # Identify outliers in string length
        for field, value in string_fields.items():
            length = len(value)
            z_score = (length - mean_length) / std_dev_length if std_dev_length != 0 else 0
            if abs(z_score) > 2:
                outliers[field] = {
                    "length": length,
                    "z_score": z_score,
                    "mean_length": mean_length,
                    "std_dev_length": std_dev_length
                }
    
    return outliers

def measure_data_coherence(data: Dict[str, Any]) -> float:
    """
    Measure the internal coherence of the data.
    
    Args:
        data: Data to analyze
        
    Returns:
        Coherence score between 0.0 and 1.0
    """
    if not data:
        return 0.0
    
    # Exclude metadata fields
    data_fields = {k: v for k, v in data.items() if not k.startswith("_")}
    
    # If not enough fields, high coherence by default
    if len(data_fields) < 3:
        return 1.0
    
    # Check field naming consistency
    field_parts = [field.lower().split('_') for field in data_fields.keys()]
    common_terms = set()
    
    # Find terms that appear in multiple field names
    for parts in field_parts:
        for part in parts:
            # Skip short parts and common words
            if len(part) <= 2 or part in ["the", "and", "of", "in", "for"]:
                continue
                
            # Count occurrences across fields
            occurrences = sum(1 for other_parts in field_parts if part in other_parts)
            if occurrences > 1:
                common_terms.add(part)
    
    # Calculate naming coherence based on common terms
    naming_coherence = min(1.0, len(common_terms) / max(1, len(data_fields) / 2))
    
    # Check value type coherence
    type_counts = {}
    for value in data_fields.values():
        value_type = type(value).__name__
        type_counts[value_type] = type_counts.get(value_type, 0) + 1
    
    # Calculate type coherence (higher if fewer types)
    dominant_type_count = max(type_counts.values()) if type_counts else 0
    type_coherence = dominant_type_count / len(data_fields)
    
    # Check string format coherence for string fields
    string_fields = {k: v for k, v in data_fields.items() if isinstance(v, str)}
    format_coherence = 1.0
    
    if len(string_fields) >= 2:
        # Look for format groups
        date_like = []
        url_like = []
        numeric_like = []
        capitalized = []
        
        for field, value in string_fields.items():
            if check_date_validity(value) > 0.7:
                date_like.append(field)
            elif check_url_validity(value) > 0.7:
                url_like.append(field)
            elif re.match(r'^[\d.,]+$', value):
                numeric_like.append(field)
            elif value and value[0].isupper():
                capitalized.append(field)
        
        # Check if similar fields have similar formats
        format_issues = 0
        
        # Check date formats
        date_formats = set()
        for field in date_like:
            value = string_fields[field]
            if re.match(r'^\d{4}-\d{2}-\d{2}', value):
                date_formats.add("ISO")
            elif re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}', value):
                date_formats.add("MM/DD/YYYY")
            else:
                date_formats.add("other")
        
        if len(date_formats) > 1:
            format_issues += 1
        
        # Check URL formats
        url_formats = set()
        for field in url_like:
            value = string_fields[field]
            if value.startswith("https://"):
                url_formats.add("https")
            elif value.startswith("http://"):
                url_formats.add("http")
            else:
                url_formats.add("other")
        
        if len(url_formats) > 1:
            format_issues += 1
        
        # Adjust format coherence
        if format_issues > 0:
            format_coherence = max(0.0, 1.0 - (format_issues * 0.2))
    
    # Calculate overall coherence
    return (naming_coherence * 0.3) + (type_coherence * 0.4) + (format_coherence * 0.3)