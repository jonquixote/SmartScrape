"""
Trafilatura Extraction Strategy for SmartScrape

This strategy uses Trafilatura, a specialized library for main content extraction
from web pages. It's particularly effective at removing boilerplate content and
extracting clean, readable text.
"""

import trafilatura
import logging
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from .base_strategy import BaseExtractionStrategy

logger = logging.getLogger(__name__)

class TrafilaturaStrategy(BaseExtractionStrategy):
    """Extraction strategy using Trafilatura library"""
    
    def __init__(self):
        super().__init__()
        self.name = "trafilatura"
        self.description = "Uses Trafilatura library for main content extraction"
        self.priority = 50  # Medium priority fallback
        
    async def extract(self, url: str, html: str = None, **kwargs) -> Dict[str, Any]:
        """
        Extract content using Trafilatura
        
        Args:
            url: The URL to extract content from
            html: Optional pre-fetched HTML content
            **kwargs: Additional options
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Fetch HTML if not provided
            if not html:
                html = await self._fetch_html(url)
                if not html:
                    return self._create_error_result(url, "Failed to fetch HTML content")
            
            # Extract with Trafilatura
            extracted_text = trafilatura.extract(
                html,
                include_comments=kwargs.get('include_comments', False),
                include_tables=kwargs.get('include_tables', True),
                include_links=kwargs.get('include_links', True),
                favor_precision=kwargs.get('favor_precision', True),
                favor_recall=kwargs.get('favor_recall', False),
                url=url
            )
            
            # Extract metadata
            metadata = None
            try:
                metadata = trafilatura.extract_metadata(html)
            except Exception as e:
                logger.warning(f"Failed to extract metadata with Trafilatura: {e}")
            
            # Extract with HTML output for structure preservation
            extracted_html = None
            if kwargs.get('include_html', False):
                try:
                    extracted_html = trafilatura.extract(
                        html,
                        output_format="html",
                        include_comments=False,
                        include_tables=True,
                        include_links=True,
                        url=url
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract HTML with Trafilatura: {e}")
            
            # Build result
            result = {
                'content': extracted_text or '',
                'url': url,
                'strategy': self.name,
                'success': bool(extracted_text),
                'metadata': {
                    'title': metadata.title if metadata else None,
                    'author': metadata.author if metadata else None,
                    'date': metadata.date if metadata else None,
                    'sitename': metadata.sitename if metadata else None,
                    'description': metadata.description if metadata else None,
                    'categories': metadata.categories if metadata else None,
                    'tags': metadata.tags if metadata else None,
                    'language': metadata.language if metadata else None,
                } if metadata else {},
                'html_content': extracted_html,
                'word_count': len(extracted_text.split()) if extracted_text else 0,
                'extraction_method': 'trafilatura'
            }
            
            # Add quality metrics
            if extracted_text:
                result['quality_metrics'] = {
                    'content_length': len(extracted_text),
                    'word_count': len(extracted_text.split()),
                    'has_metadata': bool(metadata),
                    'has_title': bool(metadata and metadata.title) if metadata else False,
                    'has_date': bool(metadata and metadata.date) if metadata else False,
                }
            
            logger.info(f"Trafilatura extraction completed for {url}")
            return result
            
        except Exception as e:
            error_msg = f"Trafilatura extraction failed: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(url, error_msg)
    
    async def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL"""
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
        except Exception as e:
            logger.error(f"Failed to fetch HTML from {url}: {e}")
            return None
    
    def _create_error_result(self, url: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'content': '',
            'url': url,
            'strategy': self.name,
            'success': False,
            'error': error_message,
            'metadata': {},
            'extraction_method': 'trafilatura'
        }
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for this strategy"""
        return {
            'include_comments': {
                'type': 'boolean',
                'default': False,
                'description': 'Include comments in extracted content'
            },
            'include_tables': {
                'type': 'boolean', 
                'name': True,
                'description': 'Include tables in extracted content'
            },
            'include_links': {
                'type': 'boolean',
                'default': True,
                'description': 'Preserve links in extracted content'
            },
            'favor_precision': {
                'type': 'boolean',
                'default': True,
                'description': 'Favor precision over recall in extraction'
            },
            'favor_recall': {
                'type': 'boolean',
                'default': False,
                'description': 'Favor recall over precision in extraction'
            },
            'include_html': {
                'type': 'boolean',
                'default': False,
                'description': 'Also extract HTML structure'
            }
        }
