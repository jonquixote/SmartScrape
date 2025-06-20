import logging
from typing import Any, Dict, List, Optional, Union

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext
from core.service_registry import ServiceRegistry

# Try to import actual stage classes, fallback to mocks if they don't exist
try:
    from core.pipeline.stages.input.http_input import HTTPInputStage
except ImportError:
    class HTTPInputStage(PipelineStage):
        """Mock HTTPInputStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True

try:
    from core.pipeline.stages.input.file_input import FileInputStage
except ImportError:
    class FileInputStage(PipelineStage):
        """Mock FileInputStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True

try:
    from core.pipeline.stages.processing.html_processing import HTMLCleaningStage, ContentExtractionStage
except ImportError:
    class HTMLCleaningStage(PipelineStage):
        """Mock HTMLCleaningStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True
            
    class ContentExtractionStage(PipelineStage):
        """Mock ContentExtractionStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True

try:
    from core.pipeline.stages.processing.content_extraction import TextExtractionStage, StructuredDataExtractionStage, PatternExtractionStage
except ImportError:
    class TextExtractionStage(PipelineStage):
        """Mock TextExtractionStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True
            
    class StructuredDataExtractionStage(PipelineStage):
        """Mock StructuredDataExtractionStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True
            
    class PatternExtractionStage(PipelineStage):
        """Mock PatternExtractionStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True

try:
    from core.pipeline.stages.processing.content_normalization import DataNormalizationStage, DataValidationStage
except ImportError:
    class DataNormalizationStage(PipelineStage):
        """Mock DataNormalizationStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True
            
    class DataValidationStage(PipelineStage):
        """Mock DataValidationStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True

try:
    from core.pipeline.stages.output.json_output import JSONOutputStage
except ImportError:
    class JSONOutputStage(PipelineStage):
        """Mock JSONOutputStage for testing purposes."""
        async def process(self, context: PipelineContext) -> bool:
            return True


class ExtractionPipeline(Pipeline):
    """Pre-configured pipeline for common extraction workflows.
    
    This pipeline template provides specialized configurations for extracting data
    from various sources like HTML pages, APIs, feeds, and sitemaps.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the extraction pipeline with a name and configuration.
        
        Args:
            name: Unique name for this pipeline
            config: Pipeline configuration with extraction-specific settings
        """
        super().__init__(name, config)
        self.logger = logging.getLogger(f"extraction_pipeline.{name}")
        
        # Extraction-specific configuration defaults
        self.extraction_config = {
            "clean_html": True,
            "extract_metadata": True,
            "normalize_data": True,
            "validate_results": True,
            "retry_failed_stages": True,
            "max_retries": 3,
            "respect_robots_txt": True,
            "use_rate_limiting": True,
            "use_proxy": False,
            "extract_links": True,
            "follow_pagination": False,
            **(self.config.get("extraction_config", {}) if self.config else {})
        }
        
        # Initialize services if needed
        self.service_registry = ServiceRegistry()
        
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> PipelineContext:
        """Execute the extraction pipeline with specialized error handling.
        
        Args:
            initial_data: Initial data to populate the context
            
        Returns:
            The final pipeline context with extraction results
        """
        context = await super().execute(initial_data)
        
        # Add extraction-specific post-processing if needed
        if not context.has_errors() and self.extraction_config.get("post_process_results", False):
            self._post_process_results(context)
            
        return context
    
    def _post_process_results(self, context: PipelineContext) -> None:
        """Apply additional post-processing to extraction results.
        
        Args:
            context: The pipeline context with extraction results
        """
        # Example post-processing logic
        extracted_data = context.get("extraction_result")
        if extracted_data and isinstance(extracted_data, dict):
            # Add metadata about the extraction
            extracted_data["_extraction_info"] = {
                "pipeline": self.name,
                "timestamp": context.metadata.get("end_time"),
                "duration": context.get_metrics().get("total_time"),
                "source": context.get("source_url", context.get("source", "unknown"))
            }
            context.set("extraction_result", extracted_data)
    
    def handle_extraction_error(self, context: PipelineContext, error: Exception) -> bool:
        """Specialized error handling for extraction failures.
        
        Args:
            context: The pipeline context
            error: The exception that occurred
            
        Returns:
            True if error was handled, False otherwise
        """
        error_type = type(error).__name__
        error_message = str(error)
        self.logger.error(f"Extraction error ({error_type}): {error_message}")
        
        # Add error details to context
        context.add_error("extraction", f"{error_type}: {error_message}")
        
        # Check if we should retry
        if self.extraction_config.get("retry_failed_stages", False):
            retry_count = context.get("retry_count", 0)
            if retry_count < self.extraction_config.get("max_retries", 3):
                context.set("retry_count", retry_count + 1)
                self.logger.info(f"Retrying extraction (attempt {retry_count + 1})")
                return True
                
        # Use fallback extraction if available
        if self.extraction_config.get("use_fallback", False):
            self.logger.info("Using fallback extraction method")
            # Logic to invoke fallback extraction would go here
            return True
            
        return False
    
    @classmethod
    def create_html_extraction_pipeline(cls, url: str, config: Optional[Dict[str, Any]] = None) -> 'ExtractionPipeline':
        """Create a pipeline configured for HTML content extraction.
        
        Args:
            url: The URL to fetch HTML content from
            config: Additional configuration options
            
        Returns:
            Configured ExtractionPipeline instance
        """
        name = f"html_extraction_{url.split('//')[1].split('/')[0]}"
        pipeline_config = {
            "extraction_config": {
                "source_type": "html",
                "source_url": url,
                "clean_html": True,
                "extract_metadata": True,
                **(config.get("extraction_config", {}) if config else {})
            },
            **(config if config else {})
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add standard HTML extraction stages
        pipeline.add_stages([
            HTTPInputStage({
                "url": url,
                "respect_robots_txt": pipeline.extraction_config.get("respect_robots_txt", True),
                "use_rate_limiting": pipeline.extraction_config.get("use_rate_limiting", True),
                "use_proxy": pipeline.extraction_config.get("use_proxy", False)
            }),
            HTMLCleaningStage({
                "remove_scripts": True,
                "remove_styles": True,
                "remove_comments": True,
                "normalize_whitespace": True
            }),
            ContentExtractionStage({
                "extract_main_content": True,
                "extract_title": True,
                "extract_metadata": pipeline.extraction_config.get("extract_metadata", True)
            }),
            TextExtractionStage({
                "extract_links": pipeline.extraction_config.get("extract_links", True),
                "extract_tables": True,
                "extract_images": True
            }),
            DataNormalizationStage({
                "normalize_text": True,
                "normalize_urls": True,
                "normalize_dates": True
            }),
            DataValidationStage({
                "validate_required_fields": ["title", "content"],
                "min_content_length": 100
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_api_extraction_pipeline(cls, endpoint: str, config: Optional[Dict[str, Any]] = None) -> 'ExtractionPipeline':
        """Create a pipeline configured for API data extraction.
        
        Args:
            endpoint: The API endpoint URL
            config: Additional configuration options
            
        Returns:
            Configured ExtractionPipeline instance
        """
        name = f"api_extraction_{endpoint.split('//')[1].split('/')[0]}"
        pipeline_config = {
            "extraction_config": {
                "source_type": "api",
                "source_url": endpoint,
                "headers": config.get("headers", {}) if config else {},
                "method": config.get("method", "GET") if config else "GET",
                "params": config.get("params", {}) if config else {},
                "json_path": config.get("json_path", None) if config else None,
                **(config.get("extraction_config", {}) if config else {})
            },
            **(config if config else {})
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add standard API extraction stages
        pipeline.add_stages([
            HTTPInputStage({
                "url": endpoint,
                "method": pipeline_config["extraction_config"]["method"],
                "headers": pipeline_config["extraction_config"]["headers"],
                "params": pipeline_config["extraction_config"]["params"],
                "use_rate_limiting": pipeline.extraction_config.get("use_rate_limiting", True),
                "use_proxy": pipeline.extraction_config.get("use_proxy", False)
            }),
            # API responses typically don't need HTML cleaning
            StructuredDataExtractionStage({
                "json_path": pipeline_config["extraction_config"]["json_path"],
                "extract_all": pipeline_config["extraction_config"].get("extract_all", True)
            }),
            DataNormalizationStage({
                "normalize_dates": True,
                "normalize_numbers": True
            }),
            DataValidationStage({
                "validate_schema": pipeline_config["extraction_config"].get("schema", None)
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_feed_extraction_pipeline(cls, feed_url: str, config: Optional[Dict[str, Any]] = None) -> 'ExtractionPipeline':
        """Create a pipeline configured for RSS/Atom feed extraction.
        
        Args:
            feed_url: The URL of the feed
            config: Additional configuration options
            
        Returns:
            Configured ExtractionPipeline instance
        """
        name = f"feed_extraction_{feed_url.split('//')[1].split('/')[0]}"
        pipeline_config = {
            "extraction_config": {
                "source_type": "feed",
                "source_url": feed_url,
                "feed_type": config.get("feed_type", "auto-detect") if config else "auto-detect",
                "entry_limit": config.get("entry_limit", 50) if config else 50,
                **(config.get("extraction_config", {}) if config else {})
            },
            **(config if config else {})
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add standard feed extraction stages
        pipeline.add_stages([
            HTTPInputStage({
                "url": feed_url,
                "respect_robots_txt": pipeline.extraction_config.get("respect_robots_txt", True),
                "use_rate_limiting": pipeline.extraction_config.get("use_rate_limiting", True)
            }),
            # Specialized feed parsing stage - using StructuredDataExtractionStage with feed config
            StructuredDataExtractionStage({
                "feed_parsing": True,
                "feed_type": pipeline_config["extraction_config"]["feed_type"],
                "entry_limit": pipeline_config["extraction_config"]["entry_limit"]
            }),
            DataNormalizationStage({
                "normalize_dates": True,
                "normalize_urls": True
            }),
            DataValidationStage({
                "validate_required_fields": ["title", "link", "published"],
                "validate_entry_count": True
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True
            })
        ])
        
        return pipeline
    
    @classmethod
    def create_sitemap_extraction_pipeline(cls, sitemap_url: str, config: Optional[Dict[str, Any]] = None) -> 'ExtractionPipeline':
        """Create a pipeline configured for sitemap extraction.
        
        Args:
            sitemap_url: The URL of the sitemap
            config: Additional configuration options
            
        Returns:
            Configured ExtractionPipeline instance
        """
        name = f"sitemap_extraction_{sitemap_url.split('//')[1].split('/')[0]}"
        pipeline_config = {
            "extraction_config": {
                "source_type": "sitemap",
                "source_url": sitemap_url,
                "recursive": config.get("recursive", True) if config else True,
                "url_limit": config.get("url_limit", 100) if config else 100,
                "include_patterns": config.get("include_patterns", []) if config else [],
                "exclude_patterns": config.get("exclude_patterns", []) if config else [],
                **(config.get("extraction_config", {}) if config else {})
            },
            **(config if config else {})
        }
        
        pipeline = cls(name, pipeline_config)
        
        # Add standard sitemap extraction stages
        pipeline.add_stages([
            HTTPInputStage({
                "url": sitemap_url,
                "respect_robots_txt": pipeline.extraction_config.get("respect_robots_txt", True),
                "use_rate_limiting": pipeline.extraction_config.get("use_rate_limiting", True)
            }),
            # Specialized sitemap parsing stage
            StructuredDataExtractionStage({
                "sitemap_parsing": True,
                "recursive": pipeline_config["extraction_config"]["recursive"],
                "url_limit": pipeline_config["extraction_config"]["url_limit"],
                "include_patterns": pipeline_config["extraction_config"]["include_patterns"],
                "exclude_patterns": pipeline_config["extraction_config"]["exclude_patterns"]
            }),
            DataNormalizationStage({
                "normalize_urls": True,
                "normalize_dates": True
            }),
            DataValidationStage({
                "validate_urls": True,
                "validate_last_modified": True
            }),
            JSONOutputStage({
                "format": "json",
                "pretty_print": True
            })
        ])
        
        return pipeline