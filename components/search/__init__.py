"""
Search components for SmartScrape.

This package provides modular components for handling search operations:
- form_detection.py: Detection of search forms on web pages
- form_interaction.py: Interaction with detected search forms
- api_detection.py: Detection and analysis of search APIs
- ajax_handler.py: Handling of AJAX responses for search results
- browser_interaction.py: Browser automation for search interactions
- autocomplete_handler.py: Handling of autocomplete/typeahead interfaces
- api_request_builder.py: Construction of API requests for search
- visual_suggestion_recognition.py: Visual recognition of search suggestions
- search_orchestrator.py: Coordination of search components
"""

from .form_detection import SearchFormDetector
from .ajax_handler import AJAXHandler
from .api_detection import APIParameterAnalyzer
from .browser_interaction import BrowserInteraction
from .search_orchestrator import SearchCoordinator
from .form_interaction import FormFieldIdentifier, FormInteraction

# Re-export for backward compatibility
__all__ = [
    'SearchFormDetector',
    'AJAXHandler', 
    'APIParameterAnalyzer',
    'BrowserInteraction',
    'SearchCoordinator',
    'FormFieldIdentifier',
    'FormInteraction'
]