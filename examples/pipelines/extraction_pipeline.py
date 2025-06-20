#!/usr/bin/env python3
"""
Extraction Pipeline Example

This example demonstrates a complete content extraction pipeline that:
1. Fetches HTML content from a URL
2. Cleans and processes the HTML
3. Extracts structured data
4. Normalizes the extracted data
5. Validates the results against a schema
6. Outputs the results in JSON format
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.pipeline.registry import PipelineRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('extraction_pipeline')


class HttpInputStage(PipelineStage):
    """Stage that fetches content from a URL."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Fetch content from the configured URL.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if successful, False otherwise
        """
        url = self.config.get("url")
        if not url:
            context.add_error(self.name, "No URL provided in configuration")
            return False
            
        logger.info(f"Fetching content from {url}")
        
        # In a real implementation, this would use aiohttp or similar
        # For this example, we'll simulate a successful fetch
        context.set("url", url)
        context.set("html_content", f"<html><body><h1>Example Page</h1><div class='product'><h2>Product Name</h2><p class='price'>$99.99</p><p class='description'>This is a sample product description.</p></div></body></html>")
        context.set("http_status", 200)
        
        return True


class HtmlProcessingStage(PipelineStage):
    """Stage that cleans and processes HTML content."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Process the HTML content.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if successful, False otherwise
        """
        html_content = context.get("html_content")
        if not html_content:
            context.add_error(self.name, "No HTML content to process")
            return False
            
        logger.info("Processing HTML content")
        
        # In a real implementation, this would use BeautifulSoup or similar
        # For this example, we'll simulate HTML processing
        
        # Remove scripts if configured
        if self.config.get("remove_scripts", True):
            logger.info("Removing script tags")
            html_content = html_content.replace("<script>", "").replace("</script>", "")
            
        # Remove styles if configured
        if self.config.get("remove_styles", True):
            logger.info("Removing style tags")
            html_content = html_content.replace("<style>", "").replace("</style>", "")
            
        # Extract main content if configured
        if self.config.get("extract_main_content", True):
            logger.info("Extracting main content")
            # Simple simulation - in reality would use more sophisticated techniques
            if "<div class='product'>" in html_content:
                main_content = html_content.split("<div class='product'>")[1].split("</div>")[0]
                context.set("main_content", main_content)
            
        # Store processed HTML
        context.set("processed_html", html_content)
        
        return True


class ContentExtractionStage(PipelineStage):
    """Stage that extracts structured data from processed HTML."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Extract structured data from the processed HTML.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if successful, False otherwise
        """
        html_content = context.get("processed_html")
        if not html_content:
            context.add_error(self.name, "No processed HTML to extract from")
            return False
            
        logger.info("Extracting structured data")
        
        # In a real implementation, this would use sophisticated extraction techniques
        # For this example, we'll simulate data extraction with simple string operations
        extracted_data = {}
        
        # Extract product name
        if "<h2>" in html_content and "</h2>" in html_content:
            product_name = html_content.split("<h2>")[1].split("</h2>")[0]
            extracted_data["product_name"] = product_name
            
        # Extract price
        if "<p class='price'>" in html_content and "</p>" in html_content:
            price_text = html_content.split("<p class='price'>")[1].split("</p>")[0]
            extracted_data["price"] = price_text
            
        # Extract description
        if "<p class='description'>" in html_content and "</p>" in html_content:
            description = html_content.split("<p class='description'>")[1].split("</p>")[0]
            extracted_data["description"] = description
            
        # Store extracted data
        context.set("extracted_data", extracted_data)
        
        return True


class DataNormalizationStage(PipelineStage):
    """Stage that normalizes extracted data."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Normalize the extracted data.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if successful, False otherwise
        """
        extracted_data = context.get("extracted_data")
        if not extracted_data:
            context.add_error(self.name, "No extracted data to normalize")
            return False
            
        logger.info("Normalizing extracted data")
        
        # Create a new normalized data dictionary
        normalized_data = {}
        
        # Normalize product name
        if "product_name" in extracted_data:
            normalized_data["name"] = extracted_data["product_name"].strip()
            
        # Normalize price
        if "price" in extracted_data:
            # Convert price string to float
            price_str = extracted_data["price"]
            try:
                # Remove currency symbol and convert to float
                price_value = float(price_str.replace("$", "").strip())
                normalized_data["price"] = {
                    "value": price_value,
                    "currency": "USD"
                }
            except ValueError:
                context.add_error(self.name, f"Could not parse price: {price_str}")
                
        # Normalize description
        if "description" in extracted_data:
            normalized_data["description"] = extracted_data["description"].strip()
            
        # Add timestamp
        import datetime
        normalized_data["extracted_at"] = datetime.datetime.now().isoformat()
        
        # Store normalized data
        context.set("normalized_data", normalized_data)
        
        return True


class SchemaValidationStage(PipelineStage):
    """Stage that validates data against a schema."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Validate the normalized data against a schema.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if validation succeeds, False otherwise
        """
        normalized_data = context.get("normalized_data")
        if not normalized_data:
            context.add_error(self.name, "No normalized data to validate")
            return False
            
        logger.info("Validating data against schema")
        
        # In a real implementation, this would use jsonschema or similar
        # For this example, we'll do basic validation
        
        # Required fields validation
        required_fields = ["name", "price"]
        missing_fields = []
        
        for field in required_fields:
            if field not in normalized_data:
                missing_fields.append(field)
                
        if missing_fields:
            context.add_error(
                self.name, 
                f"Validation failed: Missing required fields: {', '.join(missing_fields)}"
            )
            return False
            
        # Type validation
        if not isinstance(normalized_data.get("name", ""), str):
            context.add_error(self.name, "Validation failed: 'name' must be a string")
            return False
            
        if not isinstance(normalized_data.get("price", {}).get("value", 0), (int, float)):
            context.add_error(self.name, "Validation failed: 'price.value' must be a number")
            return False
            
        # Add validation status to context
        context.set("validation_passed", True)
        
        return True


class JsonOutputStage(PipelineStage):
    """Stage that formats the final output as JSON."""
    
    async def process(self, context: PipelineContext) -> bool:
        """Format the normalized and validated data as JSON.
        
        Args:
            context: The pipeline context
            
        Returns:
            bool: True if successful, False otherwise
        """
        normalized_data = context.get("normalized_data")
        if not normalized_data:
            context.add_error(self.name, "No normalized data for output")
            return False
            
        logger.info("Generating JSON output")
        
        # Generate JSON
        try:
            # Determine if pretty printing is enabled
            pretty_print = self.config.get("pretty_print", False)
            indent = 2 if pretty_print else None
            
            # Convert to JSON
            json_output = json.dumps(normalized_data, indent=indent)
            
            # Store in context
            context.set("json_output", json_output)
            
            # Log success
            if pretty_print:
                logger.info(f"Generated pretty-printed JSON output:\n{json_output}")
            else:
                logger.info("Generated JSON output")
                
            return True
            
        except Exception as e:
            context.add_error(self.name, f"JSON serialization error: {str(e)}")
            return False


async def run_extraction_pipeline(url: str) -> Dict[str, Any]:
    """Run the extraction pipeline with the specified URL.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        The extraction results
    """
    # Create a pipeline
    pipeline = Pipeline("extraction_pipeline", {
        "continue_on_error": False,
        "parallel_execution": False
    })
    
    # Add stages
    pipeline.add_stage(HttpInputStage({"url": url}))
    pipeline.add_stage(HtmlProcessingStage({
        "remove_scripts": True,
        "remove_styles": True,
        "extract_main_content": True
    }))
    pipeline.add_stage(ContentExtractionStage())
    pipeline.add_stage(DataNormalizationStage())
    pipeline.add_stage(SchemaValidationStage())
    pipeline.add_stage(JsonOutputStage({"pretty_print": True}))
    
    # Execute pipeline
    context = await pipeline.execute()
    
    # Prepare result
    result = {
        "success": not context.has_errors(),
        "url": url,
    }
    
    # Add output if available
    if context.get("json_output"):
        result["data"] = json.loads(context.get("json_output"))
    
    # Add errors if any
    if context.has_errors():
        result["errors"] = context.metadata["errors"]
    
    # Add metrics
    result["metrics"] = context.get_metrics()
    
    return result


async def run_extraction_with_registry(url: str) -> Dict[str, Any]:
    """Run the extraction pipeline using the registry.
    
    Args:
        url: The URL to extract content from
        
    Returns:
        The extraction results
    """
    # Get or create registry
    registry = PipelineRegistry()
    
    # Register stage types
    registry.register_stage(HttpInputStage)
    registry.register_stage(HtmlProcessingStage)
    registry.register_stage(ContentExtractionStage)
    registry.register_stage(DataNormalizationStage)
    registry.register_stage(SchemaValidationStage)
    registry.register_stage(JsonOutputStage)
    
    # Register pipeline configuration
    registry.register_pipeline("extraction", {
        "name": "extraction_pipeline",
        "config": {
            "continue_on_error": False,
            "parallel_execution": False
        },
        "stages": [
            {
                "type": "HttpInputStage",
                "config": {"url": url}
            },
            {
                "type": "HtmlProcessingStage",
                "config": {
                    "remove_scripts": True,
                    "remove_styles": True,
                    "extract_main_content": True
                }
            },
            {
                "type": "ContentExtractionStage",
                "config": {}
            },
            {
                "type": "DataNormalizationStage",
                "config": {}
            },
            {
                "type": "SchemaValidationStage",
                "config": {}
            },
            {
                "type": "JsonOutputStage",
                "config": {"pretty_print": True}
            }
        ]
    })
    
    # Create and execute pipeline
    pipeline = await registry.create_pipeline("extraction")
    context = await pipeline.execute()
    
    # Prepare result
    result = {
        "success": not context.has_errors(),
        "url": url,
    }
    
    # Add output if available
    if context.get("json_output"):
        result["data"] = json.loads(context.get("json_output"))
    
    # Add errors if any
    if context.has_errors():
        result["errors"] = context.metadata["errors"]
    
    # Add metrics
    result["metrics"] = context.get_metrics()
    
    return result


async def main():
    """Main function to run the example."""
    logger.info("=== Running Extraction Pipeline Example ===")
    
    # URL to extract from
    url = "https://example.com/product"
    
    # Run with direct pipeline creation
    logger.info("\n=== Running with direct pipeline creation ===")
    result1 = await run_extraction_pipeline(url)
    
    # Print result
    print("\nExtraction Result (Direct Pipeline):")
    print(json.dumps(result1, indent=2))
    
    # Run with registry
    logger.info("\n=== Running with pipeline registry ===")
    result2 = await run_extraction_with_registry(url)
    
    # Print result
    print("\nExtraction Result (Registry Pipeline):")
    print(json.dumps(result2, indent=2))
    
    logger.info("=== Example completed ===")


if __name__ == "__main__":
    asyncio.run(main())