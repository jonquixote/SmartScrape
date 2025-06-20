"""
Test module for content extraction pipeline stages.

This module contains tests for the TextExtractionStage, StructuredDataExtractionStage,
and PatternExtractionStage classes.
"""

import pytest
import asyncio
from bs4 import BeautifulSoup
from typing import Dict, Any

from core.pipeline.context import PipelineContext
from core.pipeline.stages.processing.content_extraction import (
    TextExtractionStage, 
    StructuredDataExtractionStage, 
    PatternExtractionStage
)

# Sample HTML content for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page for extraction">
</head>
<body>
    <header>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About</a></li>
            </ul>
        </nav>
    </header>
    <main>
        <article>
            <h1>Test Article</h1>
            <p>This is a paragraph with some <b>bold text</b> and a <a href="https://example.com">link</a>.</p>
            <p>Here's another paragraph with some text.</p>
            <ul>
                <li>List item 1</li>
                <li>List item 2 with <a href="#">a link</a></li>
            </ul>
            <table>
                <caption>Sample Table</caption>
                <thead>
                    <tr>
                        <th>Header 1</th>
                        <th>Header 2</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Cell 1</td>
                        <td>Cell 2</td>
                    </tr>
                    <tr>
                        <td>Cell 3</td>
                        <td>Cell 4</td>
                    </tr>
                </tbody>
            </table>
        </article>
        <aside>
            <div class="sidebar">
                <h2>Sidebar</h2>
                <p>Some sidebar content</p>
            </div>
        </aside>
    </main>
    <footer>
        <p>Footer text with an email: test@example.com and phone: (123) 456-7890</p>
    </footer>
</body>
</html>
"""

# Create test fixtures
@pytest.fixture
def pipeline_context():
    """Create a pipeline context with sample HTML content."""
    context = PipelineContext()
    context.set("html_content", SAMPLE_HTML)
    return context

@pytest.fixture
def extracted_text_context():
    """Create a pipeline context with already extracted text."""
    context = PipelineContext()
    context.set("extracted_text", "This is some sample text with an email: test@example.com and phone: (123) 456-7890")
    return context


class TestTextExtractionStage:
    """Tests for the TextExtractionStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = TextExtractionStage()
        assert stage.input_key == "html_content"
        assert stage.output_key == "extracted_text"
        
        # Custom initialization
        config = {
            "input_key": "custom_input",
            "output_key": "custom_output",
            "selector": "article",
            "preserve_formatting": True,
        }
        stage = TextExtractionStage(name="custom_text_extractor", config=config)
        assert stage.name == "custom_text_extractor"
        assert stage.input_key == "custom_input"
        assert stage.output_key == "custom_output"
        assert stage.selector == "article"
        assert stage.preserve_formatting == True
    
    @pytest.mark.asyncio
    async def test_validate_input(self, pipeline_context):
        """Test input validation."""
        stage = TextExtractionStage()
        # Valid input
        result = await stage.validate_input(pipeline_context)
        assert result == True
        
        # Invalid input
        empty_context = PipelineContext()
        result = await stage.validate_input(empty_context)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process(self, pipeline_context):
        """Test the main processing method."""
        stage = TextExtractionStage()
        result = await stage.process(pipeline_context)
        
        # Check success
        assert result == True
        
        # Check that text was extracted
        extracted_text = pipeline_context.get("extracted_text")
        assert extracted_text is not None
        assert isinstance(extracted_text, str)
        assert "Test Article" in extracted_text
        assert "This is a paragraph" in extracted_text
        
        # Check metadata
        metadata = pipeline_context.get("extracted_text_metadata")
        assert metadata is not None
        assert "original_length" in metadata
        assert "extracted_length" in metadata
    
    @pytest.mark.asyncio
    async def test_process_with_selector(self, pipeline_context):
        """Test processing with a specific selector."""
        config = {"selector": "article", "preserve_formatting": False}
        stage = TextExtractionStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check success
        assert result == True
        
        # Check that only article content was extracted
        extracted_text = pipeline_context.get("extracted_text")
        assert "Test Article" in extracted_text
        assert "Sidebar" not in extracted_text
        assert "Footer text" not in extracted_text
    
    @pytest.mark.asyncio
    async def test_process_with_formatting(self, pipeline_context):
        """Test processing with preserved formatting."""
        config = {"preserve_formatting": True}
        stage = TextExtractionStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check that formatting was preserved
        extracted_text = pipeline_context.get("extracted_text")
        assert "\n" in extracted_text
        
        # With formatting disabled
        config = {"preserve_formatting": False}
        stage = TextExtractionStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check that text was normalized
        plain_text = pipeline_context.get("extracted_text")
        assert plain_text.count("\n") < extracted_text.count("\n")


class TestStructuredDataExtractionStage:
    """Tests for the StructuredDataExtractionStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = StructuredDataExtractionStage()
        assert stage.input_key == "html_content"
        assert stage.output_key == "structured_data"
        assert stage.extract_tables == True
        
        # Custom initialization
        config = {
            "input_key": "custom_input",
            "output_key": "custom_output",
            "extract_tables": True,
            "extract_lists": False,
            "extract_forms": True,
            "table_selector": "table.data"
        }
        stage = StructuredDataExtractionStage(name="custom_extractor", config=config)
        assert stage.name == "custom_extractor"
        assert stage.input_key == "custom_input"
        assert stage.output_key == "custom_output"
        assert stage.extract_tables == True
        assert stage.extract_lists == False
        assert stage.extract_forms == True
        assert stage.table_selector == "table.data"
    
    @pytest.mark.asyncio
    async def test_validate_input(self, pipeline_context):
        """Test input validation."""
        stage = StructuredDataExtractionStage()
        # Valid input
        result = await stage.validate_input(pipeline_context)
        assert result == True
        
        # Invalid input
        empty_context = PipelineContext()
        result = await stage.validate_input(empty_context)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process(self, pipeline_context):
        """Test the main processing method."""
        stage = StructuredDataExtractionStage()
        result = await stage.process(pipeline_context)
        
        # Check success
        assert result == True
        
        # Check that structured data was extracted
        structured_data = pipeline_context.get("structured_data")
        assert structured_data is not None
        assert isinstance(structured_data, dict)
        
        # Check tables
        assert "tables" in structured_data
        tables = structured_data["tables"]
        assert len(tables) == 1
        assert tables[0]["rows"] is not None
        assert len(tables[0]["rows"]) > 0
        assert tables[0]["caption"] == "Sample Table"
        
        # Check lists
        assert "lists" in structured_data
        lists = structured_data["lists"]
        assert len(lists) >= 1
        
        # Check metadata
        assert "metadata" in structured_data
        
        # Check extraction metadata
        metadata = pipeline_context.get("structured_data_metadata")
        assert metadata is not None
        assert metadata["tables_count"] == 1
    
    @pytest.mark.asyncio
    async def test_extract_tables_only(self, pipeline_context):
        """Test extracting only tables."""
        config = {
            "extract_tables": True,
            "extract_lists": False,
            "extract_metadata": False
        }
        stage = StructuredDataExtractionStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check that only tables were extracted
        structured_data = pipeline_context.get("structured_data")
        assert "tables" in structured_data
        assert "lists" in structured_data  # Will be empty list
        assert len(structured_data["lists"]) == 0
        assert "metadata" not in structured_data
    
    @pytest.mark.asyncio
    async def test_schema_mapping(self, pipeline_context):
        """Test schema mapping functionality."""
        # Create a simple schema mapping
        config = {
            "schema_mapping": {
                "tables": "data_tables",
                "lists": "data_lists"
            }
        }
        stage = StructuredDataExtractionStage(config=config)
        result = await stage.process(pipeline_context)
        
        # Check that schema mapping was applied
        structured_data = pipeline_context.get("structured_data")
        # This would be more complex in a real schema mapping test


class TestPatternExtractionStage:
    """Tests for the PatternExtractionStage class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initialization with different configurations."""
        # Default initialization
        stage = PatternExtractionStage()
        assert stage.input_key == "text_content"
        assert stage.output_key == "extracted_patterns"
        
        # Custom initialization with patterns
        config = {
            "input_key": "custom_input",
            "output_key": "custom_output",
            "patterns": {
                "custom_email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            },
            "built_in_patterns": ["phone_us", "url"],
            "min_confidence": 0.7
        }
        stage = PatternExtractionStage(name="pattern_extractor", config=config)
        assert stage.name == "pattern_extractor"
        assert stage.input_key == "custom_input"
        assert stage.output_key == "custom_output"
        assert "custom_email" in stage.patterns
        assert "phone_us" in stage.built_in_patterns
        assert stage.min_confidence == 0.7
    
    @pytest.mark.asyncio
    async def test_validate_input(self, extracted_text_context):
        """Test input validation."""
        stage = PatternExtractionStage(config={"input_key": "extracted_text"})
        # Valid input
        result = await stage.validate_input(extracted_text_context)
        assert result == True
        
        # Invalid input
        empty_context = PipelineContext()
        result = await stage.validate_input(empty_context)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_process(self, extracted_text_context):
        """Test the main processing method."""
        # Configure with built-in patterns
        config = {
            "input_key": "extracted_text",
            "built_in_patterns": ["email", "phone_us"]
        }
        stage = PatternExtractionStage(config=config)
        result = await stage.process(extracted_text_context)
        
        # Check success
        assert result == True
        
        # Check that patterns were extracted
        patterns = extracted_text_context.get("extracted_patterns")
        assert patterns is not None
        assert isinstance(patterns, dict)
        
        # Check email extraction
        assert "email" in patterns
        assert len(patterns["email"]) > 0
        assert patterns["email"][0]["text"] == "test@example.com"
        
        # Check phone extraction
        assert "phone_us" in patterns
        assert len(patterns["phone_us"]) > 0
        assert "(123) 456-7890" in patterns["phone_us"][0]["text"]
        
        # Check metadata
        metadata = extracted_text_context.get("extracted_patterns_metadata")
        assert metadata is not None
        assert metadata["patterns_with_matches"] >= 2
    
    @pytest.mark.asyncio
    async def test_custom_patterns(self, extracted_text_context):
        """Test extraction with custom patterns."""
        config = {
            "input_key": "extracted_text",
            "patterns": {
                "word_test": r'\btest\b',
                "sample": r'sample'
            },
            "min_confidence": 0.5
        }
        stage = PatternExtractionStage(config=config)
        result = await stage.process(extracted_text_context)
        
        # Check that custom patterns were extracted
        patterns = extracted_text_context.get("extracted_patterns")
        assert "word_test" in patterns
        assert "sample" in patterns
    
    @pytest.mark.asyncio
    async def test_add_pattern(self, extracted_text_context):
        """Test adding a pattern at runtime."""
        stage = PatternExtractionStage(config={"input_key": "extracted_text"})
        
        # Add a new pattern
        stage.add_pattern("word_test", r'\btest\b', 0.9)
        
        # Process with the new pattern
        result = await stage.process(extracted_text_context)
        
        # Check that the new pattern was used
        patterns = extracted_text_context.get("extracted_patterns")
        assert "word_test" in patterns
        
        # Test removing a pattern
        success = stage.remove_pattern("word_test")
        assert success == True
        
        # Check pattern info
        pattern_info = stage.get_pattern_info()
        assert "word_test" not in pattern_info


if __name__ == "__main__":
    pytest.main()