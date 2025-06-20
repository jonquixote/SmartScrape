"""
Content Normalization Stage Module

This module provides a pipeline stage for cleaning and standardizing extracted data,
applying formatting rules, and ensuring data consistency.
"""

import logging
import re
import json
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

from core.pipeline.stages.base_stages import ProcessingStage
from core.pipeline.context import PipelineContext
from extraction.content_normalizer_impl import ContentNormalizerImpl

logger = logging.getLogger(__name__)

class ContentNormalizationStage(ProcessingStage):
    """
    Pipeline stage that cleans and standardizes extracted data.
    
    This stage applies normalization rules to ensure data consistency,
    performs cleaning operations, and standardizes formats for fields
    like dates, prices, and dimensions.
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the content normalization stage with configuration.
        
        Args:
            name: Name of this stage (defaults to class name)
            config: Configuration dictionary
        """
        super().__init__(name, config)
        self.normalizer = None
        self.input_key = self.config.get("input_key", "extracted_data")
        self.output_key = self.config.get("output_key", "normalized_data")
        self.normalize_fields = self.config.get("normalize_fields", [
            "title", "description", "content", "price", "date", "dimensions", "specs"
        ])
        self.trim_strings = self.config.get("trim_strings", True)
        self.normalize_whitespace = self.config.get("normalize_whitespace", True)
        self.standardize_dates = self.config.get("standardize_dates", True)
        self.standardize_prices = self.config.get("standardize_prices", True)
        self.remove_html = self.config.get("remove_html", True)
        self.locale = self.config.get("locale", "en_US")
        self.currency = self.config.get("currency", "USD")
        
    async def initialize(self) -> None:
        """Initialize the normalizer and stage resources."""
        if self._initialized:
            return
            
        # Create the content normalizer
        self.normalizer = ContentNormalizerImpl()
        
        # Initialize the normalizer with configuration
        normalizer_config = {
            "locale": self.locale,
            "currency": self.currency,
            "normalize_whitespace": self.normalize_whitespace,
            "trim_strings": self.trim_strings,
            "standardize_dates": self.standardize_dates,
            "standardize_prices": self.standardize_prices,
            "remove_html": self.remove_html
        }
        self.normalizer.initialize(normalizer_config)
        
        await super().initialize()
        logger.debug(f"{self.name} initialized with content normalizer")
        
    async def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        if self.normalizer:
            self.normalizer.shutdown()
            
        await super().cleanup()
        logger.debug(f"{self.name} cleaned up")
        
    async def validate_input(self, context: PipelineContext) -> bool:
        """
        Validate that the required inputs are present in the context.
        
        Args:
            context: Pipeline context containing data
            
        Returns:
            True if validation passes, False otherwise
        """
        if not context.has_key(self.input_key):
            logger.warning(f"Missing required input '{self.input_key}' in context")
            context.add_error(self.name, f"Missing required input: {self.input_key}")
            return False
            
        # Check if input is a dictionary (we can normalize)
        input_data = context.get(self.input_key)
        if not isinstance(input_data, dict):
            logger.warning(f"Invalid input type for normalization: {type(input_data)}")
            context.add_error(self.name, f"Invalid input type: {type(input_data)}")
            return False
            
        return True
        
    async def transform_data(self, data: Dict[str, Any], context: PipelineContext) -> Optional[Dict[str, Any]]:
        """
        Normalize extracted data by cleaning and standardizing values.
        
        Args:
            data: Input data (not used, we get data from context)
            context: Pipeline context containing extracted data
            
        Returns:
            Dictionary containing normalized data or None if normalization fails
        """
        try:
            if not self.normalizer:
                self.normalizer = ContentNormalizer()
                normalizer_config = {
                    "locale": self.locale,
                    "currency": self.currency,
                    "normalize_whitespace": self.normalize_whitespace,
                    "trim_strings": self.trim_strings,
                    "standardize_dates": self.standardize_dates,
                    "standardize_prices": self.standardize_prices,
                    "remove_html": self.remove_html
                }
                self.normalizer.initialize(normalizer_config)
            
            # Get extracted data from context
            extracted_data = context.get(self.input_key)
            
            # Get content type if available for specialized normalization
            content_type = context.get("content_type", "unknown")
            
            # Create normalization options
            options = {
                "content_type": content_type,
                "normalize_fields": self.normalize_fields,
                "locale": self.locale,
                "currency": self.currency
            }
            
            # Normalize data
            logger.info(f"Normalizing extracted data for content type: {content_type}")
            normalized_data = self.normalizer.normalize(extracted_data, options)
            
            # Store the original extraction method in normalized data metadata
            if "_metadata" not in normalized_data:
                normalized_data["_metadata"] = {}
                
            if "_metadata" in extracted_data:
                extraction_method = extracted_data["_metadata"].get("extraction_method", "unknown")
                normalized_data["_metadata"]["extraction_method"] = extraction_method
                
                # Copy any other useful metadata
                for key in ["item_count", "confidence_scores"]:
                    if key in extracted_data["_metadata"]:
                        normalized_data["_metadata"][key] = extracted_data["_metadata"][key]
            
            # Add normalization metadata
            normalized_data["_metadata"]["normalized"] = True
            normalized_data["_metadata"]["normalization_time"] = datetime.now().isoformat()
            
            # Apply content-type specific normalization
            if content_type == "product":
                normalized_data = self._normalize_product(normalized_data)
            elif content_type == "article":
                normalized_data = self._normalize_article(normalized_data)
            elif content_type in ["listing", "search_results"]:
                normalized_data = self._normalize_listing(normalized_data)
            
            # Track modifications for debugging
            modifications = []
            for field, value in normalized_data.items():
                if field != "_metadata" and field in extracted_data:
                    original_value = extracted_data[field]
                    if value != original_value:
                        modifications.append(field)
            
            if modifications:
                normalized_data["_metadata"]["modified_fields"] = modifications
                logger.debug(f"Modified fields during normalization: {', '.join(modifications)}")
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error in content normalization: {str(e)}")
            context.add_error(self.name, f"Normalization error: {str(e)}")
            
            # Return the original data as a fallback
            return context.get(self.input_key)
    
    def _normalize_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply product-specific normalization rules.
        
        Args:
            product_data: Product data to normalize
            
        Returns:
            Normalized product data
        """
        # Ensure we're working with a copy to avoid modifying the original
        normalized = dict(product_data)
        
        # Normalize price field
        if "price" in normalized:
            try:
                # Handle price as a string with currency symbol
                if isinstance(normalized["price"], str):
                    # Extract numeric part and convert to float
                    price_str = normalized["price"].strip()
                    price_value = self._extract_price(price_str)
                    if price_value is not None:
                        normalized["price_value"] = price_value
                        normalized["price_currency"] = self._extract_currency(price_str)
            except Exception as e:
                logger.warning(f"Error normalizing price: {str(e)}")
        
        # Normalize specifications
        if "specifications" in normalized and isinstance(normalized["specifications"], dict):
            specs = normalized["specifications"]
            normalized_specs = {}
            
            for key, value in specs.items():
                # Normalize keys (lowercase, replace spaces with underscores)
                normalized_key = key.lower().replace(" ", "_")
                
                # Normalize values
                if isinstance(value, str):
                    value = value.strip()
                    
                normalized_specs[normalized_key] = value
                
            normalized["specifications"] = normalized_specs
            
        # Ensure images is a list
        if "images" in normalized:
            if not isinstance(normalized["images"], list):
                if isinstance(normalized["images"], str):
                    normalized["images"] = [normalized["images"]]
                else:
                    normalized["images"] = []
        
        # Ensure variants is a list of dictionaries
        if "variants" in normalized:
            if not isinstance(normalized["variants"], list):
                normalized["variants"] = []
            else:
                # Normalize each variant
                for i, variant in enumerate(normalized["variants"]):
                    if isinstance(variant, dict):
                        # Normalize variant price if present
                        if "price" in variant and isinstance(variant["price"], str):
                            price_value = self._extract_price(variant["price"])
                            if price_value is not None:
                                variant["price_value"] = price_value
                    else:
                        # Replace non-dict variants with empty dict
                        normalized["variants"][i] = {}
        
        return normalized
    
    def _normalize_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply article-specific normalization rules.
        
        Args:
            article_data: Article data to normalize
            
        Returns:
            Normalized article data
        """
        # Ensure we're working with a copy to avoid modifying the original
        normalized = dict(article_data)
        
        # Normalize publication date
        if "date_published" in normalized:
            try:
                # Convert to ISO format
                date_str = normalized["date_published"]
                iso_date = self._parse_date(date_str)
                if iso_date:
                    normalized["date_published"] = iso_date
            except Exception as e:
                logger.warning(f"Error normalizing publication date: {str(e)}")
                
        # Similar for modified date
        if "date_modified" in normalized:
            try:
                date_str = normalized["date_modified"]
                iso_date = self._parse_date(date_str)
                if iso_date:
                    normalized["date_modified"] = iso_date
            except Exception as e:
                logger.warning(f"Error normalizing modified date: {str(e)}")
        
        # Normalize content field - remove excessive whitespace
        if "content" in normalized and isinstance(normalized["content"], str):
            # Normalize whitespace
            content = normalized["content"]
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Remove HTML if configured
            if self.remove_html:
                content = self._strip_html(content)
                
            normalized["content"] = content
            
            # Add word count
            normalized["word_count"] = len(content.split())
        
        # Ensure tags is a list
        if "tags" in normalized:
            if not isinstance(normalized["tags"], list):
                if isinstance(normalized["tags"], str):
                    # Split comma-separated tags
                    tags = [tag.strip() for tag in normalized["tags"].split(",")]
                    normalized["tags"] = tags
                else:
                    normalized["tags"] = []
        
        # Normalize author field
        if "author" in normalized:
            if isinstance(normalized["author"], dict):
                # Keep author as is, it's already structured
                pass
            elif isinstance(normalized["author"], str):
                # Convert to structure with just name
                normalized["author"] = {"name": normalized["author"].strip()}
            else:
                # Invalid author, remove it
                normalized.pop("author")
        
        return normalized
    
    def _normalize_listing(self, listing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply listing-specific normalization rules.
        
        Args:
            listing_data: Listing data to normalize
            
        Returns:
            Normalized listing data
        """
        # Ensure we're working with a copy to avoid modifying the original
        normalized = dict(listing_data)
        
        # Normalize items
        if "items" in normalized and isinstance(normalized["items"], list):
            normalized_items = []
            
            for item in normalized["items"]:
                if not isinstance(item, dict):
                    continue  # Skip non-dict items
                    
                # Normalize each item
                normalized_item = dict(item)
                
                # Normalize item title
                if "title" in normalized_item and isinstance(normalized_item["title"], str):
                    normalized_item["title"] = normalized_item["title"].strip()
                
                # Normalize item price
                if "price" in normalized_item and isinstance(normalized_item["price"], str):
                    price_value = self._extract_price(normalized_item["price"])
                    if price_value is not None:
                        normalized_item["price_value"] = price_value
                        normalized_item["price_currency"] = self._extract_currency(normalized_item["price"])
                
                # Ensure item URL is absolute
                if "url" in normalized_item and isinstance(normalized_item["url"], str):
                    # URL normalization would go here, but we need the base URL
                    pass
                
                normalized_items.append(normalized_item)
                
            normalized["items"] = normalized_items
            
            # Update item count
            if "_metadata" not in normalized:
                normalized["_metadata"] = {}
            normalized["_metadata"]["item_count"] = len(normalized_items)
        
        # Normalize pagination info
        if "pagination" in normalized and isinstance(normalized["pagination"], dict):
            pagination = normalized["pagination"]
            
            # Ensure numeric types for page numbers
            if "current_page" in pagination and pagination["current_page"]:
                try:
                    pagination["current_page"] = int(pagination["current_page"])
                except (ValueError, TypeError):
                    pass
                    
            if "total_pages" in pagination and pagination["total_pages"]:
                try:
                    pagination["total_pages"] = int(pagination["total_pages"])
                except (ValueError, TypeError):
                    pass
        
        return normalized
    
    def _extract_price(self, price_str: str) -> Optional[float]:
        """
        Extract numeric price value from a price string.
        
        Args:
            price_str: Price string (e.g., "$19.99")
            
        Returns:
            Price as float or None if extraction fails
        """
        try:
            # Remove currency symbols and non-numeric chars except decimal point
            numeric_str = re.sub(r'[^\d.,]', '', price_str)
            
            # Handle different decimal separators
            if ',' in numeric_str and '.' in numeric_str:
                # Both comma and period exist, assume comma is thousands separator
                numeric_str = numeric_str.replace(',', '')
            elif ',' in numeric_str and '.' not in numeric_str:
                # Only comma exists, assume it's decimal separator
                numeric_str = numeric_str.replace(',', '.')
            
            return float(numeric_str)
        except (ValueError, TypeError):
            return None
    
    def _extract_currency(self, price_str: str) -> str:
        """
        Extract currency symbol or code from a price string.
        
        Args:
            price_str: Price string (e.g., "$19.99")
            
        Returns:
            Currency code (e.g., "USD") or empty string if extraction fails
        """
        # Map common currency symbols to codes
        currency_map = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            '₽': 'RUB',
            'CA$': 'CAD',
            'A$': 'AUD',
            'NZ$': 'NZD'
        }
        
        # Check for currency symbols
        for symbol, code in currency_map.items():
            if symbol in price_str:
                return code
        
        # Check for currency codes
        currency_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'NZD', 'CHF']
        for code in currency_codes:
            if code in price_str:
                return code
        
        # Default to the configured currency
        return self.currency
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse a date string into ISO format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            ISO format date string or None if parsing fails
        """
        if not date_str:
            return None
            
        try:
            from dateutil import parser
            dt = parser.parse(date_str)
            return dt.isoformat()
        except Exception:
            # Try some common formats
            formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%d/%m/%Y',
                '%b %d, %Y',
                '%d %b %Y',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
                    
            return None
    
    def _strip_html(self, html_str: str) -> str:
        """
        Strip HTML tags from a string.
        
        Args:
            html_str: String containing HTML
            
        Returns:
            String with HTML tags removed
        """
        # Simple regex-based HTML tag removal
        text = re.sub(r'<[^>]+>', '', html_str)
        
        # Replace common HTML entities
        entities = {
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&',
            '&quot;': '"',
            '&apos;': "'",
            '&lsquo;': ''',
            '&rsquo;': ''',
            '&ldquo;': '"',
            '&rdquo;': '"',
            '&ndash;': '–',
            '&mdash;': '—',
            '&hellip;': '…'
        }
        
        for entity, replacement in entities.items():
            text = text.replace(entity, replacement)
            
        return text