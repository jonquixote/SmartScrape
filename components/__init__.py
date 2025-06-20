"""
Components Module

This module contains core components for the SmartScrape application,
including domain intelligence, pagination handling, search automation,
template integration, site discovery, and template storage.
"""

# Import all component classes for easier access
from .domain_intelligence import DomainIntelligence
from .pagination_handler import PaginationHandler
from .search_automation import SearchFormDetector, SearchAutomator
from .search_template_integration import SearchTemplateIntegrator
from .site_discovery import SiteDiscovery
from .template_storage import TemplateStorage