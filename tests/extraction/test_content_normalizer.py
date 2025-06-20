"""
Content Normalizer Tests

This module contains unit tests for the ContentNormalizer class and its 
associated utility functions in the normalization_utils module.
"""

import unittest
from datetime import datetime
import json
import re
from unittest.mock import patch, MagicMock

# Import the modules to test
from extraction.content_normalizer import ContentNormalizer
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

class TestNormalizationUtils(unittest.TestCase):
    """Tests for the normalization utility functions."""
    
    def test_clean_whitespace(self):
        """Test whitespace cleaning."""
        # Test basic whitespace normalization
        self.assertEqual(clean_whitespace("  hello  world  "), "hello world")
        
        # Test with tabs and newlines
        self.assertEqual(clean_whitespace("hello\t\tworld\n\nnow"), "hello world now")
        
        # Test with empty input
        self.assertEqual(clean_whitespace(""), "")
        self.assertEqual(clean_whitespace(None), None)
    
    def test_remove_control_chars(self):
        """Test control character removal."""
        # Test with control characters
        self.assertEqual(remove_control_chars("hello\x00world"), "helloworld")
        
        # Test with empty input
        self.assertEqual(remove_control_chars(""), "")
        self.assertEqual(remove_control_chars(None), None)
    
    def test_standardize_quotes(self):
        """Test quote standardization."""
        # Test with various quote styles
        self.assertEqual(standardize_quotes(""hello" 'world'"), '"hello" \'world\'')
        
        # Test with empty input
        self.assertEqual(standardize_quotes(""), "")
        self.assertEqual(standardize_quotes(None), None)
    
    def test_standardize_dashes(self):
        """Test dash standardization."""
        # Test with various dash styles
        self.assertEqual(standardize_dashes("hello–world—now"), "hello-world--now")
        
        # Test with empty input
        self.assertEqual(standardize_dashes(""), "")
        self.assertEqual(standardize_dashes(None), None)
    
    def test_clean_html_fragments(self):
        """Test HTML fragment cleaning."""
        # Test with HTML fragments
        self.assertEqual(clean_html_fragments("hello <b>world</b>"), "hello world")
        
        # Test with empty input
        self.assertEqual(clean_html_fragments(""), "")
        self.assertEqual(clean_html_fragments(None), None)
    
    def test_collapse_newlines(self):
        """Test newline collapsing."""
        # Test with multiple newlines
        self.assertEqual(collapse_newlines("hello\n\n\nworld"), "hello\nworld")
        
        # Test with empty input
        self.assertEqual(collapse_newlines(""), "")
        self.assertEqual(collapse_newlines(None), None)
    
    def test_parse_date(self):
        """Test date parsing."""
        # Test with ISO format
        date_obj = parse_date("2023-01-15")
        self.assertEqual(date_obj.year, 2023)
        self.assertEqual(date_obj.month, 1)
        self.assertEqual(date_obj.day, 15)
        
        # Test with US format
        date_obj = parse_date("01/15/2023")
        self.assertEqual(date_obj.year, 2023)
        self.assertEqual(date_obj.month, 1)
        self.assertEqual(date_obj.day, 15)
        
        # Test with text format
        date_obj = parse_date("Jan 15, 2023")
        self.assertEqual(date_obj.year, 2023)
        self.assertEqual(date_obj.month, 1)
        self.assertEqual(date_obj.day, 15)
        
        # Test with invalid input
        self.assertIsNone(parse_date("not a date"))
        self.assertIsNone(parse_date(None))
    
    def test_detect_date_format(self):
        """Test date format detection."""
        # Test with ISO format
        self.assertEqual(detect_date_format("2023-01-15"), "%Y-%m-%d")
        
        # Test with US format
        format_str = detect_date_format("01/15/2023")
        self.assertTrue(format_str in ["%m/%d/%Y", "%d/%m/%Y"])  # Note: This is ambiguous
        
        # Test with invalid input
        self.assertIsNone(detect_date_format("not a date"))
        self.assertIsNone(detect_date_format(None))
    
    def test_convert_to_iso(self):
        """Test ISO date conversion."""
        # Test with datetime object
        date_obj = datetime(2023, 1, 15, 12, 30, 45)
        self.assertEqual(convert_to_iso(date_obj), "2023-01-15T12:30:45")
        
        # Test with None
        self.assertIsNone(convert_to_iso(None))
    
    def test_extract_date_parts(self):
        """Test extraction of date parts."""
        # Test with datetime object
        date_obj = datetime(2023, 1, 15, 12, 30, 45)
        parts = extract_date_parts(date_obj)
        
        self.assertEqual(parts["year"], 2023)
        self.assertEqual(parts["month"], 1)
        self.assertEqual(parts["day"], 15)
        self.assertEqual(parts["hour"], 12)
        self.assertEqual(parts["minute"], 30)
        self.assertEqual(parts["second"], 45)
        self.assertEqual(parts["month_name"], "January")
        
        # Test with None
        self.assertEqual(extract_date_parts(None), {})
    
    def test_standardize_date_separators(self):
        """Test standardization of date separators."""
        # Test with various separators
        self.assertEqual(standardize_date_separators("2023-01-15"), "2023-01-15")
        self.assertEqual(standardize_date_separators("2023/01/15"), "2023-01-15")
        self.assertEqual(standardize_date_separators("01/15/2023"), "01/15/2023")
        self.assertEqual(standardize_date_separators("01-15-2023"), "01/15/2023")
        
        # Test with None
        self.assertEqual(standardize_date_separators(None), None)
    
    def test_extract_currency_symbol(self):
        """Test currency symbol extraction."""
        # Test with common symbols
        self.assertEqual(extract_currency_symbol("$100"), "USD")
        self.assertEqual(extract_currency_symbol("€100"), "EUR")
        self.assertEqual(extract_currency_symbol("£100"), "GBP")
        
        # Test with currency codes
        self.assertEqual(extract_currency_symbol("100 USD"), "USD")
        self.assertEqual(extract_currency_symbol("100 EUR"), "EUR")
        
        # Test with None
        self.assertIsNone(extract_currency_symbol(None))
        
        # Test with no currency
        self.assertIsNone(extract_currency_symbol("100"))
    
    def test_extract_numeric_value(self):
        """Test numeric value extraction."""
        # Test with simple numbers
        self.assertEqual(extract_numeric_value("100"), 100.0)
        self.assertEqual(extract_numeric_value("100.50"), 100.5)
        
        # Test with currency symbols
        self.assertEqual(extract_numeric_value("$100"), 100.0)
        self.assertEqual(extract_numeric_value("€100.50"), 100.5)
        
        # Test with thousands separators
        self.assertEqual(extract_numeric_value("1,000"), 1000.0)
        self.assertEqual(extract_numeric_value("1,000.50"), 1000.5)
        
        # Test with European format (comma as decimal)
        self.assertEqual(extract_numeric_value("1.000,50"), 1000.5)
        
        # Test with non-numeric
        self.assertIsNone(extract_numeric_value("not a number"))
        self.assertIsNone(extract_numeric_value(None))
    
    def test_convert_currency(self):
        """Test currency conversion."""
        # Test same currency (no conversion needed)
        self.assertEqual(convert_currency(100.0, "USD", "USD"), 100.0)
        
        # Test conversion with default rates
        # Note: This depends on the default rates in the function
        converted = convert_currency(100.0, "USD", "EUR")
        self.assertIsNotNone(converted)
        
        # Test with None values
        self.assertIsNone(convert_currency(None, "USD", "EUR"))
        self.assertIsNone(convert_currency(100.0, None, "EUR"))
        self.assertIsNone(convert_currency(100.0, "USD", None))
    
    def test_standardize_units(self):
        """Test unit standardization."""
        # Test length conversion to metric
        result = standardize_units(10.0, "inch", "metric")
        self.assertIsNotNone(result)
        self.assertEqual(result["system"], "metric")
        
        # Test length conversion to imperial
        result = standardize_units(10.0, "cm", "imperial")
        self.assertIsNotNone(result)
        self.assertEqual(result["system"], "imperial")
        
        # Test weight conversion
        result = standardize_units(10.0, "lb", "metric")
        self.assertIsNotNone(result)
        self.assertEqual(result["system"], "metric")
        
        # Test temperature conversion
        result = standardize_units(32.0, "f", "metric")
        self.assertIsNotNone(result)
        self.assertEqual(result["system"], "metric")
        self.assertAlmostEqual(result["converted_value"], 0.0, places=1)
        
        # Test with None values
        self.assertIsNone(standardize_units(None, "inch", "metric"))
        self.assertIsNone(standardize_units(10.0, "", "metric"))
    
    def test_detect_unit_system(self):
        """Test unit system detection."""
        # Test metric units
        self.assertEqual(detect_unit_system(10.0, "cm"), "metric")
        self.assertEqual(detect_unit_system(10.0, "kg"), "metric")
        self.assertEqual(detect_unit_system(10.0, "c"), "metric")
        
        # Test imperial units
        self.assertEqual(detect_unit_system(10.0, "inch"), "imperial")
        self.assertEqual(detect_unit_system(10.0, "lb"), "imperial")
        self.assertEqual(detect_unit_system(10.0, "f"), "imperial")
        
        # Test unknown units
        self.assertEqual(detect_unit_system(10.0, "unknown"), "unknown")
        self.assertEqual(detect_unit_system(10.0, ""), "unknown")
        self.assertEqual(detect_unit_system(10.0, None), "unknown")


class TestContentNormalizer(unittest.TestCase):
    """Tests for the ContentNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = ContentNormalizer()
        self.normalizer.initialize()
    
    def test_initialize(self):
        """Test initializing the normalizer."""
        normalizer = ContentNormalizer()
        normalizer.initialize({"locale": "en_GB", "unit_system": "imperial", 
                             "date_format": "%d/%m/%Y", "currency": "GBP"})
        
        self.assertEqual(normalizer._locale, "en_GB")
        self.assertEqual(normalizer._target_unit_system, "imperial")
        self.assertEqual(normalizer._target_date_format, "%d/%m/%Y")
        self.assertEqual(normalizer._target_currency, "GBP")
    
    def test_can_handle(self):
        """Test can_handle method."""
        # ContentNormalizer should handle most data types
        self.assertTrue(self.normalizer.can_handle("test"))
        self.assertTrue(self.normalizer.can_handle({"key": "value"}))
        self.assertTrue(self.normalizer.can_handle(123))
        self.assertTrue(self.normalizer.can_handle(None))
    
    def test_normalize(self):
        """Test normalize method."""
        # Test with a simple extraction result
        result = {
            "title": "  Test Product  ",
            "price": "$99.99",
            "date": "2023-01-15",
            "in_stock": "Yes",
            "dimensions": "10 inches x 5 inches",
            "description": "<p>This is a <b>test</b> description.</p>"
        }
        
        normalized = self.normalizer.normalize(result)
        
        # Check that normalization adds metadata
        self.assertTrue("_metadata" in normalized)
        self.assertTrue(normalized["_metadata"]["normalized"])
        
        # Check that fields are normalized
        self.assertEqual(normalized["title"], "Test Product")
        self.assertTrue("value" in normalized["price"])
        self.assertEqual(normalized["price"]["value"], 99.99)
        self.assertTrue("iso" in normalized["date"])
        self.assertTrue(normalized["in_stock"])  # Should be converted to boolean
        
        # Test with an empty extraction result
        self.assertEqual(self.normalizer.normalize({}), {})
        
        # Test with None
        self.assertIsNone(self.normalizer.normalize(None))
    
    def test_normalize_text(self):
        """Test text normalization."""
        # Test with messy text
        text = "  Hello  \t\n world! This is a \"test\" with—various—characters.  "
        normalized = self.normalizer.normalize_text(text)
        
        # Text should be cleaned up
        self.assertEqual(normalized, 'Hello world! This is a "test" with--various--characters.')
        
        # Test with HTML
        html = "<p>This is <b>bold</b> text.</p>"
        normalized = self.normalizer.normalize_text(html)
        
        # HTML tags should be removed
        self.assertEqual(normalized, "This is bold text.")
        
        # Test with None
        self.assertEqual(self.normalizer.normalize_text(None), "")
        
        # Test with non-string
        self.assertEqual(self.normalizer.normalize_text(123), "123")
    
    def test_normalize_html(self):
        """Test HTML normalization."""
        # Test with HTML content
        html = "<p>This is <b>bold</b> text.</p><script>alert('test');</script>"
        normalized = self.normalizer.normalize_html(html)
        
        # Script tags should be removed
        self.assertNotIn("script", normalized)
        self.assertNotIn("alert", normalized)
        
        # Structure should be preserved
        self.assertIn("<p>", normalized)
        self.assertIn("<b>", normalized)
        
        # Test with None
        self.assertEqual(self.normalizer.normalize_html(None), "")
    
    def test_normalize_date(self):
        """Test date normalization."""
        # Test with ISO date
        result = self.normalizer.normalize_date("2023-01-15")
        self.assertTrue(result["valid"])
        self.assertEqual(result["iso"], "2023-01-15T00:00:00")
        
        # Test with US format
        result = self.normalizer.normalize_date("01/15/2023")
        self.assertTrue(result["valid"])
        
        # Test with text format
        result = self.normalizer.normalize_date("Jan 15, 2023")
        self.assertTrue(result["valid"])
        
        # Test with invalid date
        result = self.normalizer.normalize_date("not a date")
        self.assertFalse(result["valid"])
        self.assertIsNone(result["iso"])
        
        # Test with None
        result = self.normalizer.normalize_date(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_price(self):
        """Test price normalization."""
        # Test with dollar amount
        result = self.normalizer.normalize_price("$99.99")
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 99.99)
        self.assertEqual(result["currency"], "USD")
        
        # Test with euro amount
        result = self.normalizer.normalize_price("€99.99")
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 99.99)
        self.assertEqual(result["currency"], "EUR")
        self.assertTrue("converted_value" in result)  # Should convert to target currency
        
        # Test with currency code
        result = self.normalizer.normalize_price("99.99 GBP")
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 99.99)
        self.assertEqual(result["currency"], "GBP")
        
        # Test with numeric value
        result = self.normalizer.normalize_price(99.99)
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 99.99)
        
        # Test with invalid price
        result = self.normalizer.normalize_price("not a price")
        self.assertFalse(result["valid"])
        
        # Test with None
        result = self.normalizer.normalize_price(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_measurement(self):
        """Test measurement normalization."""
        # Test with inch measurement
        result = self.normalizer.normalize_measurement("10 inches")
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 10.0)
        self.assertEqual(result["unit"], "inches")
        
        # Test with centimeter measurement
        result = self.normalizer.normalize_measurement("10 cm")
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 10.0)
        self.assertEqual(result["unit"], "cm")
        
        # Test with temperature
        result = self.normalizer.normalize_measurement("98.6 F")
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 98.6)
        self.assertEqual(result["unit"], "f")
        
        # Test with numeric value and unit in options
        result = self.normalizer.normalize_measurement(10.0, {"unit": "kg"})
        self.assertTrue(result["valid"])
        self.assertEqual(result["value"], 10.0)
        self.assertEqual(result["unit"], "kg")
        
        # Test with invalid measurement
        result = self.normalizer.normalize_measurement("not a measurement")
        self.assertFalse(result["valid"])
        
        # Test with None
        result = self.normalizer.normalize_measurement(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_boolean(self):
        """Test boolean normalization."""
        # Test with boolean values
        self.assertTrue(self.normalizer.normalize_boolean(True))
        self.assertFalse(self.normalizer.normalize_boolean(False))
        
        # Test with numeric values
        self.assertTrue(self.normalizer.normalize_boolean(1))
        self.assertFalse(self.normalizer.normalize_boolean(0))
        
        # Test with string values
        self.assertTrue(self.normalizer.normalize_boolean("yes"))
        self.assertTrue(self.normalizer.normalize_boolean("true"))
        self.assertTrue(self.normalizer.normalize_boolean("on"))
        self.assertTrue(self.normalizer.normalize_boolean("1"))
        
        self.assertFalse(self.normalizer.normalize_boolean("no"))
        self.assertFalse(self.normalizer.normalize_boolean("false"))
        self.assertFalse(self.normalizer.normalize_boolean("off"))
        self.assertFalse(self.normalizer.normalize_boolean("0"))
        
        # Test with None (should default to False)
        self.assertFalse(self.normalizer.normalize_boolean(None))
    
    def test_normalize_names(self):
        """Test name normalization."""
        # Test with simple name
        result = self.normalizer.normalize_names("John Smith")
        self.assertTrue(result["valid"])
        self.assertEqual(result["first"], "John")
        self.assertEqual(result["last"], "Smith")
        
        # Test with title
        result = self.normalizer.normalize_names("Dr. John Smith")
        self.assertTrue(result["valid"])
        self.assertEqual(result["title"], "Dr.")
        self.assertEqual(result["first"], "John")
        self.assertEqual(result["last"], "Smith")
        
        # Test with middle name
        result = self.normalizer.normalize_names("John Robert Smith")
        self.assertTrue(result["valid"])
        self.assertEqual(result["first"], "John")
        self.assertEqual(result["middle"], "Robert")
        self.assertEqual(result["last"], "Smith")
        
        # Test with suffix
        result = self.normalizer.normalize_names("John Smith Jr.")
        self.assertTrue(result["valid"])
        self.assertEqual(result["first"], "John")
        self.assertEqual(result["last"], "Smith")
        self.assertEqual(result["suffix"], "Jr.")
        
        # Test with single name
        result = self.normalizer.normalize_names("John")
        self.assertTrue(result["valid"])
        self.assertEqual(result["first"], "John")
        
        # Test with None
        result = self.normalizer.normalize_names(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_addresses(self):
        """Test address normalization."""
        # Test with US address
        address = "123 Main St, New York, NY 10001"
        result = self.normalizer.normalize_addresses(address)
        self.assertTrue(result["valid"])
        self.assertEqual(result["full"], address)
        self.assertEqual(result["street"], "123 Main St")
        self.assertEqual(result["state"], "NY")
        self.assertEqual(result["postal_code"], "10001")
        
        # Test with None
        result = self.normalizer.normalize_addresses(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_phone_numbers(self):
        """Test phone number normalization."""
        # Test with US number
        result = self.normalizer.normalize_phone_numbers("(123) 456-7890")
        self.assertTrue(result["valid"])
        self.assertEqual(result["digits"], "1234567890")
        self.assertEqual(result["country_code"], "1")
        self.assertEqual(result["area_code"], "123")
        
        # Test with international number
        result = self.normalizer.normalize_phone_numbers("+44 20 1234 5678")
        self.assertTrue(result["valid"])
        self.assertEqual(result["country_code"], "44")
        
        # Test with None
        result = self.normalizer.normalize_phone_numbers(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_urls(self):
        """Test URL normalization."""
        # Test with full URL
        result = self.normalizer.normalize_urls("https://www.example.com/path?query=value#fragment")
        self.assertTrue(result["valid"])
        self.assertEqual(result["scheme"], "https")
        self.assertEqual(result["domain"], "www.example.com")
        self.assertEqual(result["path"], "/path")
        
        # Test with domain only
        result = self.normalizer.normalize_urls("example.com")
        self.assertTrue(result["valid"])
        self.assertEqual(result["scheme"], "https")  # Should add default scheme
        self.assertEqual(result["domain"], "example.com")
        
        # Test with None
        result = self.normalizer.normalize_urls(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_identifiers(self):
        """Test identifier normalization."""
        # Test with ISBN-13
        result = self.normalizer.normalize_identifiers("9781234567890", {"type": "isbn13"})
        self.assertTrue(result["valid"])
        self.assertEqual(result["type"], "isbn13")
        
        # Test with UUID
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = self.normalizer.normalize_identifiers(uuid, {"type": "uuid"})
        self.assertTrue(result["valid"])
        self.assertEqual(result["type"], "uuid")
        self.assertEqual(result["formatted"], uuid)
        
        # Test with None
        result = self.normalizer.normalize_identifiers(None)
        self.assertFalse(result["valid"])
    
    def test_normalize_list(self):
        """Test list normalization."""
        # Test with list of strings
        strings = ["  item 1  ", "<b>item 2</b>", "item 3"]
        result = self.normalizer.normalize_list(strings)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "item 1")
        self.assertEqual(result[1], "item 2")
        
        # Test with comma-separated string
        result = self.normalizer.normalize_list("item 1, item 2, item 3")
        self.assertEqual(len(result), 3)
        
        # Test with list of dicts
        dicts = [{"name": "  John Smith  "}, {"name": "<b>Jane Doe</b>"}]
        result = self.normalizer.normalize_list(dicts)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "John Smith")
        
        # Test with empty list
        self.assertEqual(self.normalizer.normalize_list([]), [])
        
        # Test with None
        self.assertEqual(self.normalizer.normalize_list(None), [])
    
    def test_normalize_object(self):
        """Test object normalization."""
        # Test with nested dictionary
        obj = {
            "name": "  John Smith  ",
            "age": "30",
            "contact": {
                "email": "john@example.com",
                "phone": "(123) 456-7890"
            },
            "interests": ["  programming  ", "  music  "]
        }
        
        result = self.normalizer.normalize_object(obj)
        
        # Object should be recursively normalized
        self.assertEqual(result["name"], "John Smith")
        self.assertEqual(result["contact"]["email"], "john@example.com")
        self.assertEqual(result["interests"][0], "programming")
        
        # Test with JSON string
        json_str = '{"name": "John Smith", "age": 30}'
        result = self.normalizer.normalize_object(json_str)
        self.assertEqual(result["name"], "John Smith")
        
        # Test with None
        self.assertIsNone(self.normalizer.normalize_object(None))
    
    def test_infer_field_type(self):
        """Test field type inference."""
        # Test with different value types
        self.assertEqual(self.normalizer._infer_field_type(True), "boolean")
        self.assertEqual(self.normalizer._infer_field_type(123), "number")
        self.assertEqual(self.normalizer._infer_field_type([1, 2, 3]), "list")
        self.assertEqual(self.normalizer._infer_field_type({"key": "value"}), "object")
        
        # Test with different string formats
        self.assertEqual(self.normalizer._infer_field_type("2023-01-15"), "date")
        self.assertEqual(self.normalizer._infer_field_type("$99.99"), "price")
        self.assertEqual(self.normalizer._infer_field_type("10 kg"), "measurement")
        self.assertEqual(self.normalizer._infer_field_type("https://example.com"), "url")
        self.assertEqual(self.normalizer._infer_field_type("(123) 456-7890"), "phone")
        self.assertEqual(self.normalizer._infer_field_type("<p>HTML</p>"), "html")
        
        # Default for simple strings
        self.assertEqual(self.normalizer._infer_field_type("simple text"), "text")
        
        # Test with None
        self.assertEqual(self.normalizer._infer_field_type(None), "null")


if __name__ == '__main__':
    unittest.main()