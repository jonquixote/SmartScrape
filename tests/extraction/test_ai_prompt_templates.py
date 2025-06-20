"""
Tests for AI Prompt Templates Helper Module

This module tests the functionality of the AI prompt templates helper module,
which provides customizable prompt templates for AI-powered semantic extraction.
"""

import json
import unittest
from unittest.mock import patch, MagicMock
import pytest

from extraction.helpers.ai_prompt_templates import (
    customize_prompt, select_prompt_for_content, generate_schema_prompt, 
    optimize_prompt_for_model, generate_refinement_prompt,
    GENERAL_EXTRACTION_PROMPT, PRODUCT_EXTRACTION_PROMPT, ARTICLE_EXTRACTION_PROMPT,
    TABULAR_DATA_PROMPT, SCHEMA_GUIDED_EXTRACTION_PROMPT, SEMANTIC_UNDERSTANDING_PROMPT
)


class TestAIPromptTemplates(unittest.TestCase):
    """Test cases for AI prompt templates helper functions."""
    
    def test_customize_prompt(self):
        """Test customizing prompt templates with specific parameters."""
        template = "Extract {field1} and {field2} from the content."
        customized = customize_prompt(template, field1="title", field2="price")
        
        self.assertEqual(customized, "Extract title and price from the content.")
        
        # Test with more complex parameters
        template = "Content: {content}\nFields: {fields}"
        content = "This is some test content"
        fields = ["title", "author", "date"]
        
        customized = customize_prompt(template, content=content, fields=", ".join(fields))
        expected = f"Content: {content}\nFields: title, author, date"
        
        self.assertEqual(customized, expected)
    
    def test_select_prompt_for_content(self):
        """Test selecting appropriate prompt templates for content types."""
        # Test known content types
        self.assertEqual(select_prompt_for_content("product"), PRODUCT_EXTRACTION_PROMPT)
        self.assertEqual(select_prompt_for_content("article"), ARTICLE_EXTRACTION_PROMPT)
        self.assertEqual(select_prompt_for_content("table"), TABULAR_DATA_PROMPT)
        self.assertEqual(select_prompt_for_content("semantic"), SEMANTIC_UNDERSTANDING_PROMPT)
        
        # Test case insensitivity
        self.assertEqual(select_prompt_for_content("PRODUCT"), PRODUCT_EXTRACTION_PROMPT)
        self.assertEqual(select_prompt_for_content("Article"), ARTICLE_EXTRACTION_PROMPT)
        
        # Test unknown content type (should default to general extraction)
        self.assertEqual(select_prompt_for_content("unknown_type"), GENERAL_EXTRACTION_PROMPT)
    
    def test_generate_schema_prompt(self):
        """Test generating prompts tailored to specific schemas."""
        # Test with product schema
        product_schema = {
            "_type": "product",
            "name": "string",
            "price": "number",
            "description": "string"
        }
        content = "<html><body><div>Product content</div></body></html>"
        
        prompt = generate_schema_prompt(product_schema, content)
        
        # Should use the product template for product schema
        self.assertIn("You are a specialized product data extraction system", prompt)
        self.assertIn("Product content", prompt)
        
        # Test with article schema
        article_schema = {
            "_type": "article",
            "title": "string",
            "author": "string",
            "content": "string"
        }
        
        prompt = generate_schema_prompt(article_schema, content)
        
        # Should use the article template for article schema
        self.assertIn("You are a specialized article content extraction system", prompt)
        
        # Test with generic schema (no _type)
        generic_schema = {
            "field1": "string",
            "field2": "number"
        }
        
        prompt = generate_schema_prompt(generic_schema, content)
        
        # Should use the schema-guided template for generic schema
        self.assertIn("You are a specialized data extraction system", prompt)
        self.assertIn("EXTRACTION SCHEMA:", prompt)
    
    def test_optimize_prompt_for_model(self):
        """Test optimizing prompts for specific AI models."""
        test_prompt = """
        This is a test prompt.
        
        It has multiple lines.
            With different indentation.
        
        For testing prompt optimization.
        """
        
        # Test optimization for GPT-4
        gpt4_optimized = optimize_prompt_for_model(test_prompt, "gpt-4")
        
        # Should trim excess whitespace and normalize
        self.assertNotIn("        This is a test prompt.", gpt4_optimized)
        self.assertIn("This is a test prompt.", gpt4_optimized)
        self.assertFalse(gpt4_optimized.endswith("\n\n"))
        
        # Test specialized optimization for GPT-3.5
        test_prompt_with_indicators = """
        Perform a comprehensive extraction and analysis of the data.
        Return in a structured JSON format matching the schema exactly.
        """
        
        gpt35_optimized = optimize_prompt_for_model(test_prompt_with_indicators, "gpt-3.5-turbo")
        
        # Should simplify instructions for GPT-3.5
        self.assertIn("Extract the following information", gpt35_optimized)
        self.assertIn("JSON format following this structure", gpt35_optimized)
    
    def test_generate_refinement_prompt(self):
        """Test generating prompts for refining extraction results."""
        initial_result = {
            "title": "Test Product",
            "price": 29.99,
            "description": None  # Missing field
        }
        
        content = "<html><body><div>Product content with description</div></body></html>"
        missing_fields = ["description", "features"]
        
        refinement_prompt = generate_refinement_prompt(initial_result, content, missing_fields)
        
        # Should include current extraction
        self.assertIn("CURRENT EXTRACTION:", refinement_prompt)
        self.assertIn('"title": "Test Product"', refinement_prompt)
        
        # Should list fields to improve
        self.assertIn("FIELDS TO IMPROVE:", refinement_prompt)
        self.assertIn("- description", refinement_prompt)
        self.assertIn("- features", refinement_prompt)
        
        # Should include original content
        self.assertIn("ORIGINAL CONTENT:", refinement_prompt)
        self.assertIn("Product content with description", refinement_prompt)


# Pytest-style tests for more complex scenarios
@pytest.mark.parametrize("content_type,expected_fragment", [
    ("product", "Extract the following product details"),
    ("article", "Extract the following article details"),
    ("list", "Extract items from the list"),
    ("table", "Extract data from tables"),
    ("semantic", "semantic extraction system with deep language understanding"),
    ("document", "document structure analysis system"),
    ("knowledge", "knowledge extraction system"),
    ("competitive", "competitive intelligence extraction system"),
])
def test_prompt_selection_contains_expected_content(content_type, expected_fragment):
    """Test that selected prompts contain expected content fragments."""
    prompt = select_prompt_for_content(content_type)
    assert expected_fragment in prompt, f"Expected '{expected_fragment}' not found in {content_type} prompt"


def test_schema_guided_prompt_complexity():
    """Test that schema-guided prompts adapt to schema complexity."""
    simple_schema = {"name": "string", "value": "number"}
    complex_schema = {
        "product": {
            "name": "string",
            "price": {"current": "number", "original": "number"},
            "variants": [{"color": "string", "size": "string", "stock": "number"}],
            "specifications": {"type": "object", "properties": {}}
        }
    }
    
    simple_prompt = generate_schema_prompt(simple_schema, "content")
    complex_prompt = generate_schema_prompt(complex_schema, "content")
    
    # Both should contain the schema
    assert json.dumps(simple_schema) in simple_prompt
    assert json.dumps(complex_schema) in complex_prompt
    
    # Should use the schema-guided template
    assert "Extract information according to the provided schema" in simple_prompt
    assert "Extract information according to the provided schema" in complex_prompt


if __name__ == "__main__":
    unittest.main()