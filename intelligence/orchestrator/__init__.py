"""
Universal Scraping Orchestrator

This module provides the core orchestration layer for the SmartScrape system,
coordinating all aspects of intelligent scraping from intent analysis to result delivery.
"""

from .universal_orchestrator import UniversalOrchestrator
from .discovery_coordinator import DiscoveryCoordinator
from .extraction_pipeline import ExtractionPipeline
from .quality_controller import QualityController

__all__ = [
    'UniversalOrchestrator',
    'DiscoveryCoordinator', 
    'ExtractionPipeline',
    'QualityController'
]
