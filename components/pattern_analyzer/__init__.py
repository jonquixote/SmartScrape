"""
Pattern Analyzer Module

This module provides components for detecting and analyzing common patterns in websites.
It helps SmartScrape adapt to different website structures without site-specific rules.
"""

from components.pattern_analyzer.base_analyzer import PatternAnalyzer, get_registry
from components.pattern_analyzer.search_form_analyzer import SearchFormAnalyzer
from components.pattern_analyzer.listing_analyzer import ListingAnalyzer
from components.pattern_analyzer.table_analyzer import TableAnalyzer
from components.pattern_analyzer.pagination_analyzer import PaginationAnalyzer
from components.pattern_analyzer.detail_analyzer import DetailAnalyzer
from components.pattern_analyzer.selector_generator import SelectorGenerator
from components.pattern_analyzer.pattern_cache import PatternCache

__all__ = [
    'PatternAnalyzer',
    'SearchFormAnalyzer',
    'ListingAnalyzer',
    'TableAnalyzer',
    'PaginationAnalyzer',
    'DetailAnalyzer',
    'SelectorGenerator',
    'PatternCache',
    'get_registry',
]