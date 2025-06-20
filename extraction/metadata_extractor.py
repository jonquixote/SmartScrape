"""
Metadata Extractor Module

This module provides extraction capabilities for various metadata formats
embedded in content, with a focus on HTML metadata like meta tags, 
OpenGraph, Twitter Cards, JSON-LD structured data, and microdata.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin

from extraction.core.extraction_interface import MetadataExtractor, BaseExtractor
from strategies.core.strategy_context import StrategyContext
from core.service_interface import BaseService

logger = logging.getLogger(__name__)

class MetadataExtractorImpl(MetadataExtractor, BaseService):
    """
    Implementation of the MetadataExtractor interface.
    
    This extractor aggregates metadata from various sources and formats, including:
    - HTML meta tags
    - Open Graph and social media tags
    - JSON-LD structured data
    - HTML5 Microdata attributes
    """
    
    def __init__(self, context: Optional[StrategyContext] = None):
        """Initialize the extractor with an optional strategy context."""
        super().__init__(context)
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Return the name of this service."""
        return "metadata_extractor"
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the metadata extractor.
        
        Args:
            config: Optional configuration dictionary
        """
        if self._initialized:
            return
            
        logger.info("Initializing MetadataExtractor")
        self._initialized = True
    
    def shutdown(self) -> None:
        """Clean up any resources used by the extractor."""
        if self._initialized:
            logger.info("Shutting down MetadataExtractor")
            self._initialized = False
    
    def extract(self, content: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract metadata from content.
        
        Args:
            content: Content to extract metadata from (typically HTML)
            options: Optional extraction parameters
                
        Returns:
            Dictionary containing extracted metadata
        """
        options = options or {}
        base_url = options.get('url', '')
        
        try:
            # Convert to string if not already
            if not isinstance(content, str):
                content = str(content)
                
            # Extract metadata from different sources
            jsonld_data = self.extract_jsonld(content)
            opengraph_data = self.extract_opengraph(content)
            microdata = self.extract_microdata(content)
            
            # Parse basic HTML metadata
            html_metadata = self._extract_html_metadata(content, base_url)
            
            # Combine and standardize all metadata
            combined_metadata = {
                "basic": html_metadata,
                "jsonld": jsonld_data,
                "opengraph": opengraph_data,
                "microdata": microdata
            }
            
            # Create a consolidated view with standardized metadata
            consolidated = self._consolidate_metadata(combined_metadata)
            consolidated["_metadata"] = {
                "sources": list(combined_metadata.keys()),
                "extraction_method": "composite_metadata"
            }
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {"error": str(e), "_metadata": {"success": False}}
    
    def extract_jsonld(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON-LD structured data from content.
        
        Args:
            content: Content to extract from
            
        Returns:
            List of extracted JSON-LD objects
        """
        jsonld_data = []
        
        try:
            # Parse HTML content
            soup = BeautifulSoup(content, 'lxml')
            
            # Find all script tags with type="application/ld+json"
            jsonld_scripts = soup.find_all('script', type='application/ld+json')
            
            for script in jsonld_scripts:
                try:
                    # Parse JSON content
                    script_content = script.string.strip() if script.string else '{}'
                    json_data = json.loads(script_content)
                    
                    # Handle single item or array
                    if isinstance(json_data, list):
                        jsonld_data.extend(json_data)
                    else:
                        jsonld_data.append(json_data)
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.warning(f"Error parsing JSON-LD: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error extracting JSON-LD data: {str(e)}")
        
        return jsonld_data
    
    def extract_opengraph(self, content: str) -> Dict[str, Any]:
        """
        Extract OpenGraph metadata from content.
        
        Args:
            content: Content to extract from
            
        Returns:
            Dictionary of OpenGraph metadata
        """
        opengraph_data = {}
        
        try:
            # Parse HTML content
            soup = BeautifulSoup(content, 'lxml')
            
            # Find all OpenGraph meta tags
            og_meta_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            
            # Extract data from OpenGraph meta tags
            for meta in og_meta_tags:
                if meta.has_attr('property') and meta.has_attr('content'):
                    property_name = meta['property'][3:]  # Remove 'og:' prefix
                    content_value = meta['content']
                    opengraph_data[property_name] = content_value
            
            # Add Twitter Card metadata if present
            twitter_meta_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
            twitter_data = {}
            
            for meta in twitter_meta_tags:
                if meta.has_attr('name') and meta.has_attr('content'):
                    property_name = meta['name'][8:]  # Remove 'twitter:' prefix
                    content_value = meta['content']
                    twitter_data[property_name] = content_value
            
            if twitter_data:
                opengraph_data['twitter'] = twitter_data
                
        except Exception as e:
            logger.error(f"Error extracting OpenGraph data: {str(e)}")
        
        return opengraph_data
    
    def extract_microdata(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract Microdata from content.
        
        Args:
            content: Content to extract from
            
        Returns:
            List of extracted Microdata items
        """
        microdata_items = []
        
        try:
            # Parse HTML content
            soup = BeautifulSoup(content, 'lxml')
            
            # Find all elements with itemscope attribute
            itemscope_elements = soup.find_all(attrs={"itemscope": True})
            
            for element in itemscope_elements:
                item_data = {}
                
                # Get item type
                if element.has_attr('itemtype'):
                    item_data['@type'] = element['itemtype']
                
                # Process itemprop elements
                self._extract_item_properties(element, item_data)
                
                # Only add non-empty items
                if item_data and len(item_data) > 1:  # More than just @type
                    microdata_items.append(item_data)
        
        except Exception as e:
            logger.error(f"Error extracting Microdata: {str(e)}")
        
        return microdata_items
    
    def standardize_metadata(self, metadata: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Standardize metadata from different sources into a common format.
        
        Args:
            metadata: Raw metadata to standardize
            source: Source of the metadata (e.g., "jsonld", "opengraph")
            
        Returns:
            Standardized metadata dictionary
        """
        standardized = {}
        
        # Define field mappings from source-specific to standardized fields
        field_mappings = {
            "jsonld": {
                "name": ["name", "headline"],
                "description": ["description", "about"],
                "image": ["image", "thumbnailUrl"],
                "url": ["url", "mainEntityOfPage"],
                "datePublished": ["datePublished", "dateCreated"],
                "dateModified": ["dateModified"],
                "author": ["author", "creator"],
                "publisher": ["publisher"]
            },
            "opengraph": {
                "title": ["title"],
                "description": ["description"],
                "image": ["image"],
                "url": ["url"],
                "type": ["type"],
                "site_name": ["site_name"]
            },
            "basic": {
                "title": ["title"],
                "description": ["description"],
                "author": ["author"],
                "canonical": ["canonical"]
            },
            "microdata": {
                "name": ["name"],
                "description": ["description"],
                "image": ["image"],
                "url": ["url"]
            }
        }
        
        # Use appropriate mapping based on source
        mapping = field_mappings.get(source, {})
        
        # Apply the mapping
        for std_field, source_fields in mapping.items():
            for source_field in source_fields:
                if isinstance(metadata, dict) and source_field in metadata:
                    value = metadata[source_field]
                    standardized[std_field] = value
                    break
        
        return standardized
    
    def can_handle(self, content: Any, content_type: Optional[str] = None) -> bool:
        """
        Check if this extractor can handle the given content.
        
        Args:
            content: Content to check compatibility with
            content_type: Optional hint about the content type
            
        Returns:
            True if the extractor can handle this content, False otherwise
        """
        # This extractor can handle HTML or text content
        if content_type in ["html", "text/html"]:
            return True
            
        # Check if content looks like HTML
        if isinstance(content, str) and ('<html' in content.lower() or '<!doctype html' in content.lower()):
            return True
            
        return False
    
    def _extract_html_metadata(self, content: str, base_url: str = '') -> Dict[str, Any]:
        """
        Extract basic HTML metadata from content.
        
        Args:
            content: HTML content to extract from
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Dictionary of basic HTML metadata
        """
        metadata = {}
        
        try:
            # Parse HTML content
            soup = BeautifulSoup(content, 'lxml')
            
            # Get title
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                metadata['title'] = title_tag.string.strip()
            
            # Get meta description
            description_tag = soup.find('meta', attrs={'name': 'description'})
            if description_tag and description_tag.has_attr('content'):
                metadata['description'] = description_tag['content']
            
            # Get meta keywords
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag and keywords_tag.has_attr('content'):
                metadata['keywords'] = keywords_tag['content']
            
            # Get author
            author_tag = soup.find('meta', attrs={'name': 'author'})
            if author_tag and author_tag.has_attr('content'):
                metadata['author'] = author_tag['content']
            
            # Get canonical URL
            canonical_tag = soup.find('link', attrs={'rel': 'canonical'})
            if canonical_tag and canonical_tag.has_attr('href'):
                canonical_url = canonical_tag['href']
                if base_url and not canonical_url.startswith(('http://', 'https://')):
                    canonical_url = urljoin(base_url, canonical_url)
                metadata['canonical'] = canonical_url
            
            # Get favicon
            favicon_tag = soup.find('link', attrs={'rel': lambda r: r and ('icon' in r or 'shortcut icon' in r)})
            if favicon_tag and favicon_tag.has_attr('href'):
                favicon_url = favicon_tag['href']
                if base_url and not favicon_url.startswith(('http://', 'https://')):
                    favicon_url = urljoin(base_url, favicon_url)
                metadata['favicon'] = favicon_url
        
        except Exception as e:
            logger.error(f"Error extracting HTML metadata: {str(e)}")
        
        return metadata
    
    def _extract_item_properties(self, element: Tag, item_data: Dict[str, Any]) -> None:
        """
        Recursively extract item properties from an element with itemscope.
        
        Args:
            element: Element with itemscope to process
            item_data: Dictionary to populate with extracted properties
        """
        # Process direct properties
        for prop_element in element.find_all(attrs={"itemprop": True}, recursive=False):
            prop_name = prop_element["itemprop"]
            
            # Handle nested itemscope
            if prop_element.has_attr("itemscope"):
                nested_data = {}
                if prop_element.has_attr("itemtype"):
                    nested_data["@type"] = prop_element["itemtype"]
                self._extract_item_properties(prop_element, nested_data)
                item_data[prop_name] = nested_data
            else:
                # Extract property value based on element type
                if prop_element.name == "meta":
                    item_data[prop_name] = prop_element.get("content", "")
                elif prop_element.name == "img":
                    item_data[prop_name] = prop_element.get("src", "")
                elif prop_element.name == "a":
                    item_data[prop_name] = prop_element.get("href", "")
                elif prop_element.name == "time":
                    item_data[prop_name] = prop_element.get("datetime", prop_element.get_text().strip())
                else:
                    item_data[prop_name] = prop_element.get_text().strip()
        
        # Process properties in child elements
        for child in element.find_all(attrs={"itemprop": True}, recursive=True):
            # Skip direct children as they've already been processed
            if child.parent == element:
                continue
                
            # Skip elements that are within nested itemscope elements
            parent = child.parent
            is_nested = False
            while parent and parent != element:
                if parent.has_attr("itemscope"):
                    is_nested = True
                    break
                parent = parent.parent
                
            if not is_nested:
                prop_name = child["itemprop"]
                
                # Extract property value based on element type
                if child.name == "meta":
                    item_data[prop_name] = child.get("content", "")
                elif child.name == "img":
                    item_data[prop_name] = child.get("src", "")
                elif child.name == "a":
                    item_data[prop_name] = child.get("href", "")
                elif child.name == "time":
                    item_data[prop_name] = child.get("datetime", child.get_text().strip())
                else:
                    item_data[prop_name] = child.get_text().strip()
    
    def _consolidate_metadata(self, metadata_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate metadata from different sources into a single representation.
        
        Args:
            metadata_sources: Dictionary with metadata from different sources
            
        Returns:
            Consolidated metadata dictionary
        """
        consolidated = {}
        
        # Priority order for metadata sources (higher index = higher priority)
        source_priority = ["basic", "microdata", "opengraph", "jsonld"]
        
        # Standardize all sources
        standardized_sources = {}
        for source, data in metadata_sources.items():
            if source == "jsonld" and isinstance(data, list):
                # Find the most relevant JSON-LD item
                main_item = {}
                for item in data:
                    std_item = self.standardize_metadata(item, source)
                    # Prefer items with more fields or specific types
                    if len(std_item) > len(main_item):
                        main_item = std_item
                standardized_sources[source] = main_item
            elif source == "microdata" and isinstance(data, list):
                # Find the most relevant microdata item
                main_item = {}
                for item in data:
                    std_item = self.standardize_metadata(item, source)
                    if len(std_item) > len(main_item):
                        main_item = std_item
                standardized_sources[source] = main_item
            else:
                standardized_sources[source] = self.standardize_metadata(data, source)
        
        # Combine standardized sources with priority
        for field in ["title", "description", "image", "url", "author", "datePublished", "dateModified", "publisher"]:
            for source in sorted(standardized_sources.keys(), key=lambda s: source_priority.index(s) if s in source_priority else -1):
                source_data = standardized_sources[source]
                if field in source_data and source_data[field]:
                    consolidated[field] = source_data[field]
                    break
        
        # Add all additional fields from all sources that weren't already added
        for source, data in standardized_sources.items():
            for field, value in data.items():
                if field not in consolidated and value:
                    consolidated[field] = value
        
        return consolidated
