"""
Default Pattern Analyzer implementation.

This module provides a basic concrete implementation of the PatternAnalyzer abstract class.
"""

import logging
from typing import Dict, Any
from components.pattern_analyzer.base_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)

class DefaultPatternAnalyzer(PatternAnalyzer):
    """
    Default concrete implementation of the PatternAnalyzer.
    
    This class provides a basic implementation of the abstract analyze method.
    """
    
    async def analyze(self, html: str, url: str) -> Dict[str, Any]:
        """
        Analyze a page to detect patterns.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            
        Returns:
            Dictionary with detected patterns and their properties
        """
        domain = self.get_domain(url)
        logger.info(f"Analyzing patterns for URL: {url} (domain: {domain})")
        
        # Parse the HTML
        soup = self.parse_html(html)
        
        # Default implementation just returns basic page info
        result = {
            "url": url,
            "domain": domain,
            "patterns": {},
            "page_info": {
                "title": soup.title.text.strip() if soup.title else "",
                "meta_description": "",
                "has_forms": len(soup.find_all("form")) > 0,
                "has_tables": len(soup.find_all("table")) > 0,
                "link_count": len(soup.find_all("a")),
                "image_count": len(soup.find_all("img"))
            }
        }
        
        # Try to extract meta description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc and "content" in meta_desc.attrs:
            result["page_info"]["meta_description"] = meta_desc["content"]
        
        return result