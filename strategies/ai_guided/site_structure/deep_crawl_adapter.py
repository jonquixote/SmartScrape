"""
Deep Crawl Adapter Module

Integrates Crawl4AI's BestFirstCrawlingStrategy with our AI-guided strategy
and intelligent site analysis.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import urlparse

from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import (
    KeywordRelevanceScorer, 
    PathDepthScorer, 
    CompositeScorer
)
from crawl4ai.deep_crawling.filters import (
    FilterChain, 
    URLPatternFilter, 
    ContentTypeFilter,
    DomainFilter
)

from strategies.ai_guided.site_structure.site_analyzer import SiteStructureAnalyzer
from components.site_discovery import SiteDiscovery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepCrawlAdapter")

class DeepCrawlAdapter:
    """
    Adapts Crawl4AI's BestFirstCrawlingStrategy for use with our
    AI-guided strategy, integrating intelligent site analysis and
    content understanding.
    """
    
    def __init__(self, site_analyzer: Optional[SiteStructureAnalyzer] = None):
        """
        Initialize the deep crawl adapter.
        
        Args:
            site_analyzer: SiteStructureAnalyzer instance
        """
        self.site_analyzer = site_analyzer or SiteStructureAnalyzer()
        self.site_discovery = SiteDiscovery()  # Initialize site discovery component
    
    async def create_deep_crawl_strategy(
        self, 
        url: str, 
        html_content: str, 
        crawl_intent: Dict[str, Any],
        max_depth: int = 3,
        max_pages: int = 50,
        include_external: bool = False
    ) -> BestFirstCrawlingStrategy:
        """
        Create an intelligent BestFirstCrawlingStrategy based on site structure and user intent.
        
        Args:
            url: Starting URL
            html_content: HTML content of the starting page
            crawl_intent: Dictionary containing user intent information
            max_depth: Maximum depth to crawl
            max_pages: Maximum pages to crawl
            include_external: Whether to include external domains
            
        Returns:
            Configured BestFirstCrawlingStrategy instance
        """
        # Analyze the site structure if we have a site analyzer
        if self.site_analyzer:
            site_analysis = await self.site_analyzer.analyze(url, html_content, crawl_intent)
            logger.info(f"Site analysis completed: {site_analysis['site_type']} site detected")
        else:
            site_analysis = {"site_type": "unknown"}
            logger.info("No site analyzer available, using default configuration")
        
        # Detect website type from URL if not already detected
        if site_analysis.get("site_type") == "unknown":
            site_type = self.site_discovery.detect_site_type_from_url(url)
            if site_type != "unknown":
                site_analysis["site_type"] = site_type
                logger.info(f"Detected site type from URL: {site_type}")
        
        # Extract keywords from crawl intent for scoring
        keywords = self._extract_keywords_from_intent(crawl_intent)
        logger.info(f"Extracted keywords for scoring: {keywords}")
        
        # Create URL pattern filter based on site analysis and intent
        url_pattern_filter = self._create_url_pattern_filter(url, site_analysis, crawl_intent)
        
        # Create content type filter to exclude non-HTML content
        content_type_filter = ContentTypeFilter(allowed_types=["text/html"])
        
        # Create domain filter based on include_external setting
        domain_filter = None
        if not include_external:
            domain = urlparse(url).netloc
            domain_filter = DomainFilter(allowed_domains=[domain])
            logger.info(f"Restricting crawling to domain: {domain}")
        
        # Create a filter chain with all filters
        filters = FilterChain()
        filters.add_filter(url_pattern_filter)
        filters.add_filter(content_type_filter)
        if domain_filter:
            filters.add_filter(domain_filter)
        
        # Create a custom scorer that takes into account site structure and user intent
        scorer = self._create_custom_scorer(url, site_analysis, crawl_intent)
        
        # Adjust crawl settings based on site analysis and user requirements
        crawl_settings = self._get_optimal_crawl_settings(site_analysis, max_depth, max_pages)
        
        # Create and return the best-first crawling strategy
        strategy = BestFirstCrawlingStrategy(
            scorer=scorer,
            filters=filters,
            max_depth=crawl_settings["max_depth"],
            max_pages=crawl_settings["max_pages"],
            revisit_pages=crawl_settings.get("revisit_pages", False),
            follow_redirects=True
        )
        
        logger.info(f"Created deep crawl strategy with max_depth={crawl_settings['max_depth']}, " +
                   f"max_pages={crawl_settings['max_pages']}")
        
        return strategy
    
    def _extract_keywords_from_intent(self, crawl_intent: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from the crawl intent for use in scoring.
        
        Args:
            crawl_intent: Dictionary with crawl intent information
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Extract explicit keywords if available
        if "keywords" in crawl_intent and isinstance(crawl_intent["keywords"], list):
            keywords.extend(crawl_intent["keywords"])
        
        # Extract from extract_description
        if "extract_description" in crawl_intent:
            desc = crawl_intent["extract_description"]
            # Simple keyword extraction - split by spaces and filter short words
            desc_words = [w.lower() for w in desc.split() if len(w) > 3 and w.isalpha()]
            keywords.extend(desc_words)
        
        # If specific entities are mentioned
        if "entities" in crawl_intent and isinstance(crawl_intent["entities"], list):
            keywords.extend(crawl_intent["entities"])
            
        # Deduplicate keywords while preserving order
        unique_keywords = []
        for kw in keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
                
        return unique_keywords
    
    def _create_url_pattern_filter(
        self, 
        start_url: str, 
        site_analysis: Dict[str, Any],
        crawl_intent: Dict[str, Any]
    ) -> URLPatternFilter:
        """
        Create URL pattern filter based on site structure and intent.
        
        Args:
            start_url: Starting URL
            site_analysis: Site structure analysis
            crawl_intent: User intent information
            
        Returns:
            Configured URLPatternFilter
        """
        # Patterns to exclude (common non-content paths)
        exclude_patterns = [
            r'\.(?:css|js|png|jpg|jpeg|gif|svg|woff|ttf|ico)$',  # Static assets
            r'/(?:login|signin|signup|register|logout)',  # Authentication pages
            r'/(?:cart|checkout|basket)',  # Cart pages
            r'/(?:privacy|terms|contact|about)',  # Policy pages
            r'/wp-(?:admin|includes|content/plugins)'  # WordPress admin paths
        ]
        
        # Check site type and add more specific patterns
        site_type = site_analysis.get("site_type", "unknown")
        
        if site_type == "ecommerce":
            # For ecommerce, focus on product and category pages
            include_patterns = [
                r'/(?:product|item|category|collection)s?/',
                r'/(?:p|c)/[\w-]+',
                r'/catalog/'
            ]
        elif site_type == "real_estate":
            # For real estate, focus on property listings
            include_patterns = [
                r'/(?:property|properties|home|homes|house|houses)/',
                r'/(?:listing|listings|for-sale|for-rent)/',
                r'/(?:search|results)/'
            ]
            # Add MLS ID pattern if present in the URL or domain
            if "mls" in start_url.lower():
                include_patterns.append(r'/(?:mls|listing)-\d+')
        elif site_type == "blog":
            # For blogs, focus on posts and categories
            include_patterns = [
                r'/(?:post|article|blog|news)s?/',
                r'/\d{4}/\d{2}/[\w-]+',  # Common date-based blog URLs
                r'/category/[\w-]+'
            ]
        else:
            # Generic include patterns - fairly inclusive
            include_patterns = []
        
        # Check crawl intent for additional clues
        if "content_paths" in crawl_intent and isinstance(crawl_intent["content_paths"], list):
            include_patterns.extend(crawl_intent["content_paths"])
            
        # Check if site analysis found important path patterns
        if "important_sections" in site_analysis and site_analysis["important_sections"]:
            for section in site_analysis["important_sections"]:
                if "path" in section:
                    path_pattern = section["path"].replace("/", r'\/')
                    include_patterns.append(f"{path_pattern}")
        
        # Create and return the URL pattern filter
        return URLPatternFilter(
            include_patterns=include_patterns if include_patterns else None,
            exclude_patterns=exclude_patterns
        )
    
    def _create_custom_scorer(
        self, 
        start_url: str, 
        site_analysis: Dict[str, Any],
        crawl_intent: Dict[str, Any]
    ) -> CompositeScorer:
        """
        Create a custom scorer based on site analysis and crawl intent.
        
        Args:
            start_url: Starting URL
            site_analysis: Site structure analysis
            crawl_intent: User intent information
            
        Returns:
            Configured CompositeScorer
        """
        scorers = []
        weights = []
        
        # Add keyword relevance scorer if we have keywords
        keywords = self._extract_keywords_from_intent(crawl_intent)
        if keywords:
            keyword_scorer = KeywordRelevanceScorer(keywords=keywords)
            scorers.append(keyword_scorer)
            weights.append(0.5)  # High weight for keyword relevance
        
        # Add path depth scorer - prefer shallower paths generally
        depth_scorer = PathDepthScorer(prefer_shallow=True)
        scorers.append(depth_scorer)
        weights.append(0.3)
        
        # Add URL pattern scorer for preferred content sections
        pattern_weights = {}
        
        # Add weights for common content patterns based on site type
        site_type = site_analysis.get("site_type", "unknown")
        if site_type == "ecommerce":
            pattern_weights.update({
                r'/product': 0.9,
                r'/category': 0.7,
                r'/collection': 0.7,
                r'/brand': 0.6
            })
        elif site_type == "real_estate":
            pattern_weights.update({
                r'/property': 0.9,
                r'/listing': 0.9,
                r'/home': 0.8,
                r'/search-results': 0.7
            })
        elif site_type == "blog":
            pattern_weights.update({
                r'/post': 0.9,
                r'/article': 0.9,
                r'/blog': 0.8,
                r'/category': 0.7
            })
            
        # Check if site analysis found important path patterns
        if "important_sections" in site_analysis and site_analysis["important_sections"]:
            for section in site_analysis["important_sections"]:
                if "path" in section and "relevance" in section:
                    path_pattern = section["path"].replace("/", r'\/')
                    pattern_weights[path_pattern] = section["relevance"]
        
        # If we have pattern weights, add a URL pattern scorer
        if pattern_weights:
            # URLPatternScorer has been renamed/removed in newer crawl4ai versions
            # Create a simple keyword relevance scorer instead as a fallback
            logger.info("Using keyword relevance scorer as fallback for URL pattern scoring")
            keywords = list(pattern_weights.keys())
            weights_list = list(pattern_weights.values())
            # Normalize weights
            if weights_list:
                max_weight = max(weights_list)
                normalized_weights = [w/max_weight for w in weights_list]
                fallback_scorer = KeywordRelevanceScorer(keywords=keywords, weights=normalized_weights)
                scorers.append(fallback_scorer)
                weights.append(0.3)
        
        # Create and return the composite scorer
        return CompositeScorer(scorers=scorers, weights=weights)
    
    def _get_optimal_crawl_settings(
        self, 
        site_analysis: Dict[str, Any], 
        max_depth: int,
        max_pages: int
    ) -> Dict[str, Any]:
        """
        Get optimal crawl settings based on site analysis.
        
        Args:
            site_analysis: Site structure analysis
            max_depth: Default maximum depth
            max_pages: Default maximum pages
            
        Returns:
            Dictionary with crawl settings
        """
        settings = {
            "max_depth": max_depth,
            "max_pages": max_pages,
            "revisit_pages": False
        }
        
        site_type = site_analysis.get("site_type", "unknown")
        
        # Adjust based on site type
        if site_type == "ecommerce":
            # E-commerce sites often need deeper crawling
            settings["max_depth"] = max(4, max_depth)
            settings["max_pages"] = max(100, max_pages)
        elif site_type == "blog":
            # Blogs might need less depth but more pages
            settings["max_depth"] = min(3, max_depth)
            settings["max_pages"] = max(150, max_pages)
        elif site_type == "real_estate":
            # Real estate sites often need more pages
            settings["max_pages"] = max(200, max_pages)
            
        # Consider recommended approach from site analysis
        approach = site_analysis.get("recommended_approach", "")
        if approach == "breadth_first":
            # For breadth-first, we want to limit depth but increase pages
            settings["max_depth"] = min(2, max_depth)
            settings["max_pages"] = max(150, max_pages)
        elif approach == "depth_first":
            # For depth-first, we want to increase depth
            settings["max_depth"] = max(5, max_depth)
            
        # Check if pagination was detected
        if site_analysis.get("has_pagination", False):
            # If pagination exists, we might want to focus more on paginated content
            settings["max_depth"] += 1
            
        return settings