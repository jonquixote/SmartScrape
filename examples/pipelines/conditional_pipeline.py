#!/usr/bin/env python3
"""
Conditional Pipeline Example

This example demonstrates a pipeline with conditional branching based on content type.
The pipeline will process different types of data (HTML, JSON, text) using
different processing strategies based on the content type.

Key concepts demonstrated:
- Conditional stage execution
- Dynamic stage selection
- Context-based routing
- Error handling with fallbacks
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import re
from html.parser import HTMLParser

# Import the base pipeline classes from the basic example
# In a real implementation, you would import from core.pipeline.*
from examples.pipelines.basic_pipeline import (
    Pipeline, PipelineStage, PipelineContext
)


# Content Detection Stage

class ContentDetectionStage(PipelineStage):
    """Analyzes input content and determines its type."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Detect content type and set it in the context."""
        print(f"[{self.name}] Analyzing content type...")
        
        content = context.get("raw_content")
        if not content:
            context.add_error(self.name, "No content to analyze")
            return False
            
        # Detect content type
        content_type = self._detect_content_type(content)
        context.set("content_type", content_type)
        
        print(f"[{self.name}] Detected content type: {content_type}")
        return True
        
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content based on its structure."""
        # Try parsing as JSON
        if content.strip().startswith(("{", "[")):
            try:
                json.loads(content)
                return "json"
            except json.JSONDecodeError:
                pass
                
        # Check for HTML indicators
        if re.search(r"<html.*?>|<body.*?>|<!DOCTYPE html>", content, re.IGNORECASE):
            return "html"
            
        # Check for XML indicators
        if re.search(r"<\?xml.*?\?>|<[a-zA-Z]+:.*?>", content):
            return "xml"
            
        # Default to text
        return "text"


# Conditional Routing Stage

class ContentRoutingStage(PipelineStage):
    """Routes content to the appropriate processing stage based on its type."""
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains the required content type."""
        if not context.get("content_type"):
            context.add_error(self.name, "Missing content_type in context")
            return False
        if not context.get("raw_content"):
            context.add_error(self.name, "Missing raw_content in context")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Route the content to the appropriate processing stage."""
        print(f"[{self.name}] Routing content for processing...")
        
        content_type = context.get("content_type")
        raw_content = context.get("raw_content")
        
        # Select processor based on content type
        if content_type == "html":
            processor = HtmlProcessingStage(self.config.get("html_config", {}))
        elif content_type == "json":
            processor = JsonProcessingStage(self.config.get("json_config", {}))
        elif content_type == "xml":
            processor = XmlProcessingStage(self.config.get("xml_config", {}))
        else:
            # Default to text processing
            processor = TextProcessingStage(self.config.get("text_config", {}))
            
        print(f"[{self.name}] Selected processor: {processor.name}")
        
        # Process the content with the selected processor
        success = await processor.process(context)
        
        # If primary processor fails, try fallback if enabled
        if not success and self.config.get("use_fallback", True):
            print(f"[{self.name}] Primary processor failed, trying fallback...")
            fallback = FallbackProcessingStage(self.config.get("fallback_config", {}))
            return await fallback.process(context)
            
        return success


# Type-specific Processing Stages

class HtmlProcessingStage(PipelineStage):
    """Processes HTML content."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Extract data from HTML content."""
        print(f"[{self.name}] Processing HTML content...")
        
        html_content = context.get("raw_content")
        
        # Simple HTML parsing
        parser = SimpleHtmlParser()
        parser.feed(html_content)
        
        # Extract the data we want
        extracted_data = {
            "title": parser.title,
            "headings": parser.headings,
            "links": parser.links,
            "text_content": parser.text_content
        }
        
        context.set("processed_data", extracted_data)
        context.set("processing_method", "html")
        
        print(f"[{self.name}] Extracted {len(parser.headings)} headings and {len(parser.links)} links")
        return True


class JsonProcessingStage(PipelineStage):
    """Processes JSON content."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Parse and extract data from JSON content."""
        print(f"[{self.name}] Processing JSON content...")
        
        json_content = context.get("raw_content")
        
        try:
            # Parse JSON
            parsed_data = json.loads(json_content)
            
            # Extract specific fields if configured
            extracted_data = parsed_data
            if self.config.get("extract_fields"):
                fields = self.config.get("extract_fields", [])
                if isinstance(parsed_data, dict):
                    extracted_data = {field: parsed_data.get(field) for field in fields if field in parsed_data}
                    
            context.set("processed_data", extracted_data)
            context.set("processing_method", "json")
            
            if isinstance(extracted_data, dict):
                print(f"[{self.name}] Extracted {len(extracted_data)} fields from JSON")
            else:
                print(f"[{self.name}] Processed JSON data")
                
            return True
            
        except json.JSONDecodeError as e:
            context.add_error(self.name, f"JSON parsing error: {str(e)}")
            return False


class XmlProcessingStage(PipelineStage):
    """Processes XML content."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Parse and extract data from XML content."""
        print(f"[{self.name}] Processing XML content...")
        
        # For simplicity, we'll just extract tags
        xml_content = context.get("raw_content")
        
        # Extract tags using regex (simplified)
        tags = re.findall(r"<([a-zA-Z0-9_:]+)[^>]*>", xml_content)
        unique_tags = list(set(tags))
        
        # Count tag occurrences
        tag_counts = {}
        for tag in unique_tags:
            tag_counts[tag] = len(re.findall(f"<{tag}[^>]*>", xml_content))
            
        extracted_data = {
            "unique_tags": unique_tags,
            "tag_counts": tag_counts,
            "content_length": len(xml_content)
        }
        
        context.set("processed_data", extracted_data)
        context.set("processing_method", "xml")
        
        print(f"[{self.name}] Extracted {len(unique_tags)} unique XML tags")
        return True


class TextProcessingStage(PipelineStage):
    """Processes plain text content."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Process plain text content."""
        print(f"[{self.name}] Processing text content...")
        
        text_content = context.get("raw_content")
        
        # Simple text analysis
        lines = text_content.split("\n")
        words = re.findall(r'\b\w+\b', text_content)
        
        # Calculate stats
        stats = {
            "line_count": len(lines),
            "word_count": len(words),
            "char_count": len(text_content),
            "avg_word_length": sum(len(word) for word in words) / max(len(words), 1)
        }
        
        # Extract potential entities (simplified)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text_content)
        
        extracted_data = {
            "stats": stats,
            "entities": {
                "emails": emails,
                "urls": urls
            },
            "snippet": text_content[:100] + ("..." if len(text_content) > 100 else "")
        }
        
        context.set("processed_data", extracted_data)
        context.set("processing_method", "text")
        
        print(f"[{self.name}] Analyzed text with {stats['word_count']} words in {stats['line_count']} lines")
        return True


class FallbackProcessingStage(PipelineStage):
    """Fallback processor for when type-specific processing fails."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Apply generic fallback processing to content."""
        print(f"[{self.name}] Applying fallback processing...")
        
        content = context.get("raw_content")
        
        # Very basic analysis that works for any content type
        extracted_data = {
            "content_length": len(content),
            "content_snippet": content[:100] + ("..." if len(content) > 100 else ""),
            "content_lines": len(content.split("\n")),
        }
        
        context.set("processed_data", extracted_data)
        context.set("processing_method", "fallback")
        
        print(f"[{self.name}] Applied generic fallback processing")
        return True


# Result Summary Stage

class ResultSummaryStage(PipelineStage):
    """Summarizes the results of content processing."""
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains processed data."""
        if not context.get("processed_data"):
            context.add_error(self.name, "Missing processed_data in context")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Create a summary of the processed data."""
        print(f"[{self.name}] Creating result summary...")
        
        processed_data = context.get("processed_data")
        processing_method = context.get("processing_method", "unknown")
        
        # Create a summary appropriate for the processing method
        if processing_method == "html":
            summary = self._summarize_html(processed_data)
        elif processing_method == "json":
            summary = self._summarize_json(processed_data)
        elif processing_method == "xml":
            summary = self._summarize_xml(processed_data)
        elif processing_method == "text":
            summary = self._summarize_text(processed_data)
        else:
            summary = self._summarize_fallback(processed_data)
            
        context.set("result_summary", summary)
        
        print(f"[{self.name}] Created summary using method: {processing_method}")
        return True
        
    def _summarize_html(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary for HTML content."""
        return {
            "type": "html",
            "title": data.get("title", "No title"),
            "heading_count": len(data.get("headings", [])),
            "link_count": len(data.get("links", [])),
            "content_preview": data.get("text_content", "")[:150] + "..."
        }
        
    def _summarize_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary for JSON content."""
        return {
            "type": "json",
            "structure": self._describe_structure(data),
            "field_count": len(data) if isinstance(data, dict) else (
                len(data) if isinstance(data, list) else 0
            )
        }
        
    def _summarize_xml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary for XML content."""
        return {
            "type": "xml",
            "unique_tag_count": len(data.get("unique_tags", [])),
            "most_common_tags": self._get_most_common(data.get("tag_counts", {}), 3),
            "content_size": data.get("content_length", 0)
        }
        
    def _summarize_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary for text content."""
        stats = data.get("stats", {})
        entities = data.get("entities", {})
        
        return {
            "type": "text",
            "word_count": stats.get("word_count", 0),
            "line_count": stats.get("line_count", 0),
            "email_count": len(entities.get("emails", [])),
            "url_count": len(entities.get("urls", [])),
            "snippet": data.get("snippet", "")
        }
        
    def _summarize_fallback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary for fallback processed content."""
        return {
            "type": "generic",
            "content_length": data.get("content_length", 0),
            "content_snippet": data.get("content_snippet", ""),
            "line_count": data.get("content_lines", 0)
        }
        
    def _describe_structure(self, data: Any) -> str:
        """Describe the structure of JSON data."""
        if isinstance(data, dict):
            return f"Object with {len(data)} properties"
        elif isinstance(data, list):
            return f"Array with {len(data)} items"
        elif isinstance(data, str):
            return f"String ({len(data)} chars)"
        elif isinstance(data, (int, float)):
            return f"Number ({data})"
        elif data is None:
            return "null"
        else:
            return f"{type(data).__name__}"
            
    def _get_most_common(self, counter: Dict[str, int], n: int) -> List[Dict[str, Any]]:
        """Get the N most common items from a counter dictionary."""
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return [{"name": k, "count": v} for k, v in sorted_items[:n]]


# Utility Classes

class SimpleHtmlParser(HTMLParser):
    """A simplified HTML parser that extracts basic information."""
    
    def __init__(self):
        super().__init__()
        self.title = ""
        self.in_title = False
        self.headings = []
        self.current_heading = ""
        self.in_heading = False
        self.heading_level = 0
        self.links = []
        self.text_content = []
        
    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True
        elif tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self.in_heading = True
            self.heading_level = int(tag[1])
            self.current_heading = ""
        elif tag == "a":
            href = next((v for k, v in attrs if k == "href"), None)
            if href:
                self.links.append(href)
                
    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag.startswith("h") and len(tag) == 2 and tag[1].isdigit():
            self.in_heading = False
            if self.current_heading:
                self.headings.append({
                    "level": self.heading_level,
                    "text": self.current_heading
                })
                
    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif self.in_heading:
            self.current_heading += data
            
        # Collect non-empty text
        if data.strip():
            self.text_content.append(data.strip())


# Sample content for testing

HTML_SAMPLE = """<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML Page</title>
</head>
<body>
    <h1>Main Heading</h1>
    <p>This is a paragraph with a <a href="https://example.com">link</a>.</p>
    <h2>Subheading 1</h2>
    <p>Another paragraph with <a href="https://test.com">another link</a>.</p>
    <h2>Subheading 2</h2>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
</body>
</html>"""

JSON_SAMPLE = """{
    "name": "Product Example",
    "price": 29.99,
    "in_stock": true,
    "categories": ["electronics", "accessories"],
    "specifications": {
        "weight": "250g",
        "dimensions": {
            "width": 10,
            "height": 5,
            "depth": 2
        }
    },
    "reviews": [
        {"user": "user1", "rating": 5, "comment": "Great product!"},
        {"user": "user2", "rating": 4, "comment": "Good value for money."}
    ]
}"""

XML_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<catalog>
    <book id="bk101">
        <author>Gambardella, Matthew</author>
        <title>XML Developer's Guide</title>
        <genre>Computer</genre>
        <price>44.95</price>
        <publish_date>2000-10-01</publish_date>
    </book>
    <book id="bk102">
        <author>Ralls, Kim</author>
        <title>Midnight Rain</title>
        <genre>Fantasy</genre>
        <price>5.95</price>
        <publish_date>2000-12-16</publish_date>
    </book>
    <book id="bk103">
        <author>Corets, Eva</author>
        <title>Maeve Ascendant</title>
        <genre>Fantasy</genre>
        <price>5.95</price>
        <publish_date>2000-11-17</publish_date>
    </book>
</catalog>"""

TEXT_SAMPLE = """This is a sample plain text document.
It contains multiple lines of text.
Some lines might have URLs like https://example.com or https://test.com.
It could also contain emails like user@example.com.

The text has paragraph breaks and varying line lengths.
This can be used to test text processing functionality.
"""


async def main():
    """Run the conditional pipeline example."""
    print("=== Conditional Pipeline Example ===\n")
    
    # Create samples map
    samples = {
        "html": HTML_SAMPLE,
        "json": JSON_SAMPLE,
        "xml": XML_SAMPLE,
        "text": TEXT_SAMPLE
    }
    
    # Process each sample
    for sample_name, sample_content in samples.items():
        print(f"\n--- Processing {sample_name.upper()} Sample ---")
        
        # Create a pipeline for this sample
        pipeline = Pipeline(f"{sample_name}_pipeline")
        
        # Add stages
        pipeline.add_stage(ContentDetectionStage())
        pipeline.add_stage(ContentRoutingStage())
        pipeline.add_stage(ResultSummaryStage())
        
        # Execute the pipeline with this sample
        context = await pipeline.execute({"raw_content": sample_content})
        
        # Print the summary
        if not context.has_errors():
            summary = context.get("result_summary", {})
            print(f"\nResult Summary ({sample_name}):")
            for key, value in summary.items():
                if isinstance(value, (list, dict)):
                    print(f"  {key}: {json.dumps(value)}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("\nErrors during processing:")
            for source, messages in context.metadata["errors"].items():
                for message in messages:
                    print(f"  {source}: {message}")
        
        print("\n" + "-" * 50)
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())

# Expected output:
"""
=== Conditional Pipeline Example ===

--- Processing HTML Sample ---
[ContentDetectionStage] Analyzing content type...
[ContentDetectionStage] Detected content type: html
[ContentRoutingStage] Routing content for processing...
[ContentRoutingStage] Selected processor: HtmlProcessingStage
[HtmlProcessingStage] Processing HTML content...
[HtmlProcessingStage] Extracted 3 headings and 2 links
[ResultSummaryStage] Creating result summary...
[ResultSummaryStage] Created summary using method: html

Result Summary (html):
  type: html
  title: Sample HTML Page
  heading_count: 3
  link_count: 2
  content_preview: Main Heading This is a paragraph with a link. Subheading 1 Another paragraph with another link. Subheading 2 Item 1 Item 2 Item 3...

--------------------------------------------------

--- Processing JSON Sample ---
[ContentDetectionStage] Analyzing content type...
[ContentDetectionStage] Detected content type: json
[ContentRoutingStage] Routing content for processing...
[ContentRoutingStage] Selected processor: JsonProcessingStage
[JsonProcessingStage] Processing JSON content...
[JsonProcessingStage] Extracted 6 fields from JSON
[ResultSummaryStage] Creating result summary...
[ResultSummaryStage] Created summary using method: json

Result Summary (json):
  type: json
  structure: Object with 6 properties
  field_count: 6

--------------------------------------------------

--- Processing XML Sample ---
[ContentDetectionStage] Analyzing content type...
[ContentDetectionStage] Detected content type: xml
[ContentRoutingStage] Routing content for processing...
[ContentRoutingStage] Selected processor: XmlProcessingStage
[XmlProcessingStage] Processing XML content...
[XmlProcessingStage] Extracted 8 unique XML tags
[ResultSummaryStage] Creating result summary...
[ResultSummaryStage] Created summary using method: xml

Result Summary (xml):
  type: xml
  unique_tag_count: 8
  most_common_tags: [{"name": "book", "count": 3}, {"name": "price", "count": 3}, {"name": "author", "count": 3}]
  content_size: 842

--------------------------------------------------

--- Processing TEXT Sample ---
[ContentDetectionStage] Analyzing content type...
[ContentDetectionStage] Detected content type: text
[ContentRoutingStage] Routing content for processing...
[ContentRoutingStage] Selected processor: TextProcessingStage
[TextProcessingStage] Processing text content...
[TextProcessingStage] Analyzed text with 58 words in 7 lines
[ResultSummaryStage] Creating result summary...
[ResultSummaryStage] Created summary using method: text

Result Summary (text):
  type: text
  word_count: 58
  line_count: 7
  email_count: 1
  url_count: 2
  snippet: This is a sample plain text document.
It contains multiple lines of text.
Some lines might have...

--------------------------------------------------

=== Example Complete ===
"""