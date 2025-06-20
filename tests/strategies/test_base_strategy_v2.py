"""
Tests for the BaseStrategyV2 class.

This module contains unit tests for the improved base strategy implementation.
"""

import unittest
import time
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional

from bs4 import BeautifulSoup

from strategies.base_strategy_v2 import BaseStrategyV2
from strategies.core.strategy_types import StrategyType, StrategyCapability

class MockContext:
    """Mock strategy context for testing."""
    
    def __init__(self):
        self.logger = MagicMock()
        self.services = {
            "url_service": MagicMock(),
            "html_service": MagicMock()
        }
        
    def get_service(self, name):
        return self.services.get(name)

class TestBaseStrategyV2(unittest.TestCase):
    """Test case for BaseStrategyV2."""
    
    def setUp(self):
        """Set up tests."""
        self.mock_context = MockContext()
        self.mock_context.services["url_service"].normalize_url.return_value = "https://example.com/normalized"
        self.mock_context.services["url_service"].is_allowed.return_value = True
        
        # Sample HTML content for testing
        self.html_content = """
        <html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="This is a test page">
            </head>
            <body>
                <div id="main">
                    <h1>Welcome to Test Page</h1>
                    <p>This is sample content for testing.</p>
                    <a href="https://example.com/page1">Page 1</a>
                    <a href="https://example.com/page2">Page 2</a>
                    <div class="pagination">
                        <a href="https://example.com/page?p=1">1</a>
                        <a href="https://example.com/page?p=2">2</a>
                        <a href="https://example.com/page?p=3">3</a>
                        <a href="https://example.com/page?p=next">Next</a>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # Configure mock HTML service
        self.mock_context.services["html_service"].extract_links.return_value = [
            {"url": "https://example.com/page1", "text": "Page 1"},
            {"url": "https://example.com/page2", "text": "Page 2"}
        ]
        self.mock_context.services["html_service"].clean_html.return_value = self.html_content
        self.mock_context.services["html_service"].extract_main_content.return_value = "<div><p>This is sample content for testing.</p></div>"
        self.mock_context.services["html_service"].extract_metadata.return_value = {"title": "Test Page", "description": "This is a test page"}
        
        # Create strategy instance
        self.strategy = BaseStrategyV2(self.mock_context)
        
    def tearDown(self):
        """Clean up after tests."""
        self.strategy.shutdown()
    
    def test_initialization(self):
        """Test strategy initialization."""
        # Test default property values
        self.assertEqual(self.strategy.name, "base_v2")
        self.assertEqual(self.strategy.config['max_depth'], 2)
        self.assertEqual(self.strategy.config['max_pages'], 100)
        self.assertFalse(self.strategy.config['include_external'])
        
        # Test with custom config
        custom_config = {
            'max_depth': 5,
            'max_pages': 200,
            'include_external': True
        }
        strategy = BaseStrategyV2(self.mock_context, custom_config)
        self.assertEqual(strategy.config['max_depth'], 5)
        self.assertEqual(strategy.config['max_pages'], 200)
        self.assertTrue(strategy.config['include_external'])
    
    def test_metadata_decorator(self):
        """Test strategy metadata decorator."""
        # Test that metadata was applied correctly
        self.assertEqual(self.strategy._metadata.strategy_type, StrategyType.SPECIAL_PURPOSE)
        self.assertTrue(StrategyCapability.ROBOTS_TXT_ADHERENCE in self.strategy._metadata.capabilities)
        self.assertTrue(StrategyCapability.RATE_LIMITING in self.strategy._metadata.capabilities)
        self.assertTrue(StrategyCapability.ERROR_HANDLING in self.strategy._metadata.capabilities)
    
    def test_execute_success(self):
        """Test successful execution of strategy."""
        # Mock _fetch_url to return sample HTML
        with patch.object(self.strategy, '_fetch_url', return_value=self.html_content):
            result = self.strategy.execute("https://example.com")
            
            # Verify result contains expected data
            self.assertIsNotNone(result)
            self.assertEqual(result['title'], "Test Page")
            self.assertEqual(result['description'], "This is a test page")
            self.assertEqual(result['link_count'], 2)
            
            # Verify metrics were updated
            self.assertEqual(self.strategy._metrics['pages_visited'], 1)
            self.assertEqual(self.strategy._metrics['successful_extractions'], 1)
    
    def test_execute_failure(self):
        """Test failed execution of strategy."""
        # Mock _fetch_url to return None (failed)
        with patch.object(self.strategy, '_fetch_url', return_value=None):
            result = self.strategy.execute("https://example.com")
            
            # Verify result is None
            self.assertIsNone(result)
            
            # Verify metrics were updated
            self.assertEqual(self.strategy._metrics['pages_visited'], 0)
            self.assertEqual(self.strategy._metrics['successful_extractions'], 0)
    
    def test_crawl(self):
        """Test crawling functionality."""
        # Mock _fetch_url to return sample HTML
        with patch.object(self.strategy, '_fetch_url', return_value=self.html_content):
            result = self.strategy.crawl("https://example.com")
            
            # Verify result structure
            self.assertIsNotNone(result)
            self.assertIn('results', result)
            self.assertIn('metrics', result)
            
            # Verify at least the starting URL was visited
            self.assertTrue(len(self.strategy.visited_urls) >= 1)
    
    def test_extract(self):
        """Test data extraction."""
        result = self.strategy.extract(self.html_content, "https://example.com")
        
        # Verify result contains expected data
        self.assertIsNotNone(result)
        self.assertEqual(result['title'], "Test Page")
        self.assertEqual(result['description'], "This is a test page")
        self.assertEqual(result['url'], "https://example.com")
        
        # Verify services were called
        self.mock_context.services["html_service"].clean_html.assert_called_once()
        self.mock_context.services["html_service"].extract_links.assert_called_once()
        self.mock_context.services["html_service"].extract_metadata.assert_called_once()
    
    def test_handle_pagination(self):
        """Test pagination handling."""
        pagination_urls = self.strategy.handle_pagination(self.html_content, "https://example.com")
        
        # Verify pagination URLs were extracted
        self.assertEqual(len(pagination_urls), 4)  # 3 numbered pages + Next
    
    def test_should_visit(self):
        """Test URL filtering."""
        # Setup
        self.strategy.main_domain = "example.com"
        self.strategy.visited_urls = {"https://example.com/visited"}
        
        # Test already visited URL
        self.assertFalse(self.strategy.should_visit("https://example.com/visited"))
        
        # Test new URL on same domain
        self.assertTrue(self.strategy.should_visit("https://example.com/new"))
        
        # Test external URL with default config (include_external=False)
        self.assertFalse(self.strategy.should_visit("https://other-domain.com"))
        
        # Test external URL with include_external=True
        self.strategy.config['include_external'] = True
        self.assertTrue(self.strategy.should_visit("https://other-domain.com"))
    
    def test_score_url(self):
        """Test URL scoring."""
        # Same domain, short path
        score1 = self.strategy.score_url("https://example.com/page", "https://example.com")
        
        # Same domain, deep path
        score2 = self.strategy.score_url("https://example.com/category/subcategory/page", "https://example.com")
        
        # Different domain
        score3 = self.strategy.score_url("https://other-domain.com/page", "https://example.com")
        
        # Verify scoring logic
        self.assertGreater(score1, score2)  # Shorter path should score higher
        self.assertGreater(score1, score3)  # Same domain should score higher than different domain
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            'max_depth': 3,
            'max_pages': 100
        }
        self.assertTrue(self.strategy.validate_config(valid_config))
        
        # Invalid config - missing required params
        invalid_config1 = {
            'max_depth': 3
            # missing max_pages
        }
        self.assertFalse(self.strategy.validate_config(invalid_config1))
        
        # Invalid config - wrong type
        invalid_config2 = {
            'max_depth': "3",  # should be int
            'max_pages': 100
        }
        self.assertFalse(self.strategy.validate_config(invalid_config2))
    
    def test_metrics(self):
        """Test metrics tracking."""
        # Execute strategy to generate metrics
        with patch.object(self.strategy, '_fetch_url', return_value=self.html_content):
            self.strategy.execute("https://example.com")
            
        # Get metrics
        metrics = self.strategy.get_metrics()
        
        # Verify metrics structure
        self.assertIn('pages_visited', metrics)
        self.assertIn('successful_extractions', metrics)
        self.assertIn('failed_extractions', metrics)
        self.assertIn('total_time', metrics)
        
        # Verify metrics values
        self.assertEqual(metrics['pages_visited'], 1)
        self.assertEqual(metrics['successful_extractions'], 1)
        self.assertGreaterEqual(metrics['total_time'], 0)
    
    def test_pause_resume(self):
        """Test pausing and resuming strategy execution."""
        # Initially not paused
        self.assertFalse(self.strategy.is_paused)
        
        # Pause strategy
        self.strategy.pause()
        self.assertTrue(self.strategy.is_paused)
        
        # Resume strategy
        self.strategy.resume()
        self.assertFalse(self.strategy.is_paused)
    
    def test_add_result(self):
        """Test adding results manually."""
        # Initially empty
        self.assertEqual(len(self.strategy.get_results()), 0)
        
        # Add a result
        sample_result = {
            'url': 'https://example.com',
            'title': 'Example',
            'content': 'Sample content'
        }
        self.strategy.add_result(sample_result)
        
        # Verify result was added
        results = self.strategy.get_results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], sample_result)
        
        # Clear results
        self.strategy.clear_results()
        self.assertEqual(len(self.strategy.get_results()), 0)


if __name__ == '__main__':
    unittest.main()