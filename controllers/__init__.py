"""
Controllers Module for SmartScrape

This package contains controllers that coordinate various components
of the system for scraping and data extraction tasks.
"""

# Import core services and registry
from core.service_registry import ServiceRegistry
from core.url_service import URLService
from core.html_service import HTMLService
from core.session_manager import SessionManager

# Create a global service registry that can be used during imports
global_registry = ServiceRegistry()

# Register core services
global_registry.register_service("url_service", URLService())
global_registry.register_service("html_service", HTMLService())
global_registry.register_service("session_manager", SessionManager())

# Export modules
__all__ = ['adaptive_scraper', 'global_registry']