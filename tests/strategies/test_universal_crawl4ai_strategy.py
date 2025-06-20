"""
Unit tests for UniversalCrawl4AIStrategy component.

This module tests the advanced crawling strategy using crawl4ai with intelligent
pathfinding, progressive data collection, AI-driven content extraction, and
memory-adaptive resource management.
"""

import unittest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from strategies.universal_crawl4ai_strategy import (
    UniversalCrawl4AIStrategy, PageData, CrawlSession
)
from strategies.core.strategy_types import StrategyCapability
from strategies.base_strategy import StrategyContext


class TestUniversalCrawl4AIStrategy(unittest.TestCase):
    """Test suite for UniversalCrawl4AIStrategy component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock strategy context
        self.mock_context = Mock(spec=StrategyContext)
        self.mock_context.config = Mock()
        self.mock_context.config.CRAWL4AI_ENABLED = True
        self.mock_context.config.CRAWL4AI_MAX_PAGES = 10
        self.mock_context.config.CRAWL4AI_DEEP_CRAWL = True
        self.mock_context.config.CRAWL4AI_MEMORY_THRESHOLD = 80.0
        self.mock_context.config.CRAWL4AI_AI_PATHFINDING = True
        self.mock_context.config.USE_UNDETECTED_CHROMEDRIVER = True
        self.mock_context.config.PROGRESSIVE_DATA_COLLECTION = True
        self.mock_context.config.DATA_CONSISTENCY_CHECKS = True
        self.mock_context.config.CIRCUIT_BREAKER_ENABLED = True
        
        # Mock crawl4ai availability
        with patch('strategies.universal_crawl4ai_strategy.CRAWL4AI_AVAILABLE', True):
            self.strategy = UniversalCrawl4AIStrategy(context=self.mock_context)
    
    def test_initialization_success(self):
        """Test successful UniversalCrawl4AIStrategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.context, self.mock_context)
        self.assertTrue(hasattr(self.strategy, 'crawler_config'))
        self.assertTrue(hasattr(self.strategy, 'deep_crawl_enabled'))
        self.assertTrue(hasattr(self.strategy, 'use_undetected_driver'))
        self.assertTrue(hasattr(self.strategy, 'ai_pathfinding'))
    
    @patch('strategies.universal_crawl4ai_strategy.CRAWL4AI_AVAILABLE', False)
    def test_initialization_no_crawl4ai(self):
        """Test initialization failure when crawl4ai is not available."""
        with self.assertRaises((ImportError, ValueError)):
            UniversalCrawl4AIStrategy(context=self.mock_context)
    
    def test_strategy_capabilities(self):
        """Test that strategy reports correct capabilities."""
        capabilities = self.strategy.get_capabilities()
        
        expected_capabilities = {
            StrategyCapability.AI_ASSISTED,
            StrategyCapability.PROGRESSIVE_CRAWLING,
            StrategyCapability.SEMANTIC_SEARCH,
            StrategyCapability.INTENT_ANALYSIS,
            StrategyCapability.AI_PATHFINDING,
            StrategyCapability.EARLY_RELEVANCE_TERMINATION,
            StrategyCapability.MEMORY_ADAPTIVE,
            StrategyCapability.CIRCUIT_BREAKER,
            StrategyCapability.CONSOLIDATED_AI_PROCESSING,
            StrategyCapability.JAVASCRIPT_EXECUTION,
            StrategyCapability.DYNAMIC_CONTENT,
            StrategyCapability.ERROR_HANDLING,
            StrategyCapability.RETRY_MECHANISM,
            StrategyCapability.RATE_LIMITING
        }
        
        for capability in expected_capabilities:
            self.assertIn(capability, capabilities)
    
    @patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler')
    async def test_execute_basic_crawl(self, mock_crawler_class):
        """Test basic crawl execution."""
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        # Mock crawl result
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = "<html><body><h1>Test Content</h1></body></html>"
        mock_result.cleaned_html = "<h1>Test Content</h1>"
        mock_result.markdown = "# Test Content"
        mock_result.extracted_content = "Test Content"
        mock_result.metadata = {"title": "Test Page"}
        mock_result.links = {"internal": ["https://example.com/page2"]}
        
        mock_crawler.arun.return_value = mock_result
        
        # Mock other methods
        with patch.object(self.strategy, '_create_crawler_config') as mock_config:
            with patch.object(self.strategy, '_evaluate_page_relevance', return_value=True):
                with patch.object(self.strategy, '_extract_page_data') as mock_extract:
                    mock_extract.return_value = PageData(
                        url="https://example.com",
                        title="Test Page",
                        main_content="Test Content",
                        structured_data={"title": "Test Page"},
                        relevance_score=0.8,
                        timestamp=1234567890.0,
                        metadata={"source": "crawl4ai"}
                    )
                    
                    result = await self.strategy.execute("https://example.com")
                    
                    self.assertIsNotNone(result)
                    self.assertIn('extracted_data', result)
                    self.assertIn('metadata', result)
    
    @patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler')
    async def test_execute_with_intent_analysis(self, mock_crawler_class):
        """Test crawl execution with intent analysis integration."""
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.html = "<html><body><h1>AI Research</h1></body></html>"
        mock_result.cleaned_html = "<h1>AI Research</h1>"
        mock_result.markdown = "# AI Research"
        mock_result.extracted_content = "AI Research Content"
        mock_result.metadata = {"title": "AI Research"}
        mock_result.links = {"internal": []}
        
        mock_crawler.arun.return_value = mock_result
        
        # Intent analysis data
        intent_analysis = {
            'intent_type': 'research',
            'entities': [{'text': 'artificial intelligence', 'label': 'TOPIC'}],
            'semantic_keywords': ['AI', 'machine learning', 'research'],
            'confidence': 0.9
        }
        
        with patch.object(self.strategy, '_create_crawler_config'):
            with patch.object(self.strategy, '_evaluate_page_relevance', return_value=True):
                with patch.object(self.strategy, '_extract_page_data') as mock_extract:
                    mock_extract.return_value = PageData(
                        url="https://example.com",
                        title="AI Research",
                        main_content="AI Research Content",
                        structured_data={"title": "AI Research"},
                        relevance_score=0.9,
                        timestamp=1234567890.0,
                        metadata={"source": "crawl4ai"}
                    )
                    
                    result = await self.strategy.execute(
                        "https://example.com", 
                        intent_analysis=intent_analysis
                    )
                    
                    self.assertIsNotNone(result)
                    self.assertGreater(result['metadata']['relevance_score'], 0.8)
    
    @patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler')
    async def test_progressive_data_collection(self, mock_crawler_class):
        """Test progressive data collection from multiple pages."""
        # Setup mock crawler
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        # Mock multiple page results
        page_results = [
            Mock(success=True, html=f"<html><body><h1>Page {i}</h1></body></html>", 
                 cleaned_html=f"<h1>Page {i}</h1>", markdown=f"# Page {i}",
                 extracted_content=f"Page {i} Content", metadata={"title": f"Page {i}"},
                 links={"internal": []})
            for i in range(1, 4)
        ]
        
        mock_crawler.arun.side_effect = page_results
        
        with patch.object(self.strategy, '_create_crawler_config'):
            with patch.object(self.strategy, '_evaluate_page_relevance', return_value=True):
                with patch.object(self.strategy, '_extract_page_data') as mock_extract:
                    mock_extract.side_effect = [
                        PageData(f"https://example.com/page{i}", f"Page {i}", 
                                f"Page {i} Content", {"title": f"Page {i}"}, 
                                0.8, 1234567890.0, {"source": "crawl4ai"})
                        for i in range(1, 4)
                    ]
                    with patch.object(self.strategy, '_get_progressive_urls') as mock_urls:
                        mock_urls.return_value = [
                            "https://example.com/page2", 
                            "https://example.com/page3"
                        ]
                        
                        result = await self.strategy.execute(
                            "https://example.com/page1",
                            progressive_crawl=True,
                            max_pages=3
                        )
                        
                        self.assertIsNotNone(result)
                        self.assertIn('aggregated_data', result)
                        # Should have collected data from multiple pages
                        self.assertGreater(len(result['aggregated_data']['pages']), 1)
    
    async def test_early_relevance_termination(self):
        """Test early termination of crawling for low-relevance pages."""
        with patch.object(self.strategy, '_evaluate_page_relevance', return_value=False):
            with patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler') as mock_crawler_class:
                mock_crawler = AsyncMock()
                mock_crawler_class.return_value = mock_crawler
                mock_crawler.__aenter__.return_value = mock_crawler
                mock_crawler.__aexit__.return_value = None
                
                # Mock initial crawl result
                mock_result = Mock()
                mock_result.success = True
                mock_result.html = "<html><body><h1>Irrelevant Content</h1></body></html>"
                mock_result.cleaned_html = "<h1>Irrelevant Content</h1>"
                mock_result.markdown = "# Irrelevant Content"
                mock_result.extracted_content = "Irrelevant Content"
                mock_result.metadata = {"title": "Irrelevant Page"}
                mock_result.links = {"internal": []}
                
                mock_crawler.arun.return_value = mock_result
                
                with patch.object(self.strategy, '_create_crawler_config'):
                    result = await self.strategy.execute("https://example.com")
                    
                    # Should terminate early and return minimal result
                    self.assertIsNotNone(result)
                    self.assertIn('terminated_early', result['metadata'])
                    self.assertTrue(result['metadata']['terminated_early'])
    
    def test_memory_adaptive_configuration(self):
        """Test memory-adaptive configuration setup."""
        with patch.object(self.strategy, '_get_memory_usage', return_value=75.0):
            config = self.strategy._create_crawler_config()
            
            self.assertIsNotNone(config)
            # Should configure for memory-adaptive operation
    
    def test_memory_adaptive_throttling(self):
        """Test memory usage monitoring and throttling."""
        # Test high memory usage scenario
        with patch.object(self.strategy, '_get_memory_usage', return_value=85.0):
            should_continue = self.strategy._should_continue_crawling(current_pages=5)
            
            # Should throttle or stop when memory usage is high
            self.assertFalse(should_continue)
        
        # Test normal memory usage scenario
        with patch.object(self.strategy, '_get_memory_usage', return_value=60.0):
            should_continue = self.strategy._should_continue_crawling(current_pages=5)
            
            # Should continue when memory usage is normal
            self.assertTrue(should_continue)
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern for resilience."""
        # Test circuit breaker tripping after multiple failures
        for i in range(5):
            self.strategy._record_failure("test_error")
        
        self.assertTrue(self.strategy._is_circuit_open())
        
        # Test circuit breaker reset after cooldown
        with patch('time.time', return_value=9999999):  # Future time
            self.strategy._try_reset_circuit()
            self.assertFalse(self.strategy._is_circuit_open())
    
    @patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler')
    async def test_error_handling_and_retry(self, mock_crawler_class):
        """Test error handling and retry mechanism."""
        # Setup mock crawler with initial failure
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        # First call fails, second succeeds
        mock_result_fail = Mock()
        mock_result_fail.success = False
        mock_result_fail.error = "Network error"
        
        mock_result_success = Mock()
        mock_result_success.success = True
        mock_result_success.html = "<html><body><h1>Success</h1></body></html>"
        mock_result_success.cleaned_html = "<h1>Success</h1>"
        mock_result_success.markdown = "# Success"
        mock_result_success.extracted_content = "Success Content"
        mock_result_success.metadata = {"title": "Success Page"}
        mock_result_success.links = {"internal": []}
        
        mock_crawler.arun.side_effect = [mock_result_fail, mock_result_success]
        
        with patch.object(self.strategy, '_create_crawler_config'):
            with patch.object(self.strategy, '_evaluate_page_relevance', return_value=True):
                with patch.object(self.strategy, '_extract_page_data') as mock_extract:
                    mock_extract.return_value = PageData(
                        url="https://example.com",
                        title="Success Page",
                        main_content="Success Content",
                        structured_data={"title": "Success Page"},
                        relevance_score=0.8,
                        timestamp=1234567890.0,
                        metadata={"source": "crawl4ai"}
                    )
                    
                    result = await self.strategy.execute("https://example.com")
                    
                    # Should succeed after retry
                    self.assertIsNotNone(result)
                    self.assertIn('retry_count', result['metadata'])
    
    def test_page_data_dataclass(self):
        """Test PageData dataclass functionality."""
        page_data = PageData(
            url="https://example.com",
            title="Test Page",
            main_content="Test content",
            structured_data={"key": "value"},
            relevance_score=0.8,
            timestamp=1234567890.0,
            metadata={"source": "test"}
        )
        
        self.assertEqual(page_data.url, "https://example.com")
        self.assertEqual(page_data.title, "Test Page")
        self.assertEqual(page_data.main_content, "Test content")
        self.assertEqual(page_data.structured_data["key"], "value")
        self.assertEqual(page_data.relevance_score, 0.8)
        self.assertEqual(page_data.timestamp, 1234567890.0)
        self.assertEqual(page_data.metadata["source"], "test")
    
    def test_crawl_session_dataclass(self):
        """Test CrawlSession dataclass functionality."""
        session = CrawlSession(
            query="test query",
            intent_analysis={"intent_type": "test"},
            collected_pages=[],
            visited_urls=set(),
            pending_urls=[],
            total_relevance=0.0,
            start_time=1234567890.0,
            memory_usage=0.0
        )
        
        self.assertEqual(session.query, "test query")
        self.assertEqual(session.intent_analysis["intent_type"], "test")
        self.assertEqual(len(session.collected_pages), 0)
        self.assertEqual(len(session.visited_urls), 0)
        self.assertEqual(len(session.pending_urls), 0)
        self.assertEqual(session.total_relevance, 0.0)
        self.assertEqual(session.start_time, 1234567890.0)
        self.assertEqual(session.memory_usage, 0.0)
    
    def test_crawler_config_creation(self):
        """Test crawler configuration creation."""
        config = self.strategy._create_crawler_config()
        
        self.assertIsNotNone(config)
        # Should include settings for memory management, undetected chrome, etc.
    
    async def test_consolidated_ai_processing(self):
        """Test consolidated AI processing of aggregated data."""
        # Mock aggregated data from multiple pages
        aggregated_data = [
            PageData("https://example.com/1", "Page 1", "Content 1", 
                    {"title": "Page 1"}, 0.8, 1234567890.0, {}),
            PageData("https://example.com/2", "Page 2", "Content 2", 
                    {"title": "Page 2"}, 0.7, 1234567890.0, {}),
            PageData("https://example.com/3", "Page 3", "Content 3", 
                    {"title": "Page 3"}, 0.9, 1234567890.0, {})
        ]
        
        with patch.object(self.strategy, '_apply_ai_processing') as mock_ai:
            mock_ai.return_value = {
                "consolidated_data": {
                    "title": "Consolidated Title",
                    "content": "Consolidated Content",
                    "summary": "AI-generated summary"
                },
                "confidence": 0.85
            }
            
            result = self.strategy._process_aggregated_data(aggregated_data)
            
            self.assertIsNotNone(result)
            self.assertIn('consolidated_data', result)
            self.assertIn('confidence', result)
            mock_ai.assert_called_once()
    
    def test_url_prioritization_logic(self):
        """Test URL prioritization based on relevance scores."""
        urls = [
            ("https://example.com/low", 0.3),
            ("https://example.com/high", 0.9),
            ("https://example.com/medium", 0.6)
        ]
        
        prioritized = self.strategy._prioritize_urls(urls)
        
        # Should be sorted by priority score (descending)
        self.assertEqual(prioritized[0][0], "https://example.com/high")
        self.assertEqual(prioritized[1][0], "https://example.com/medium")
        self.assertEqual(prioritized[2][0], "https://example.com/low")
    
    def test_data_consistency_checks(self):
        """Test data consistency validation."""
        consistent_data = {
            "title": "Test Title",
            "content": "Test content with sufficient length",
            "url": "https://example.com"
        }
        
        inconsistent_data = {
            "title": "",  # Empty title
            "content": "",  # Empty content
            "url": "not-a-url"  # Invalid URL
        }
        
        self.assertTrue(self.strategy._validate_data_consistency(consistent_data))
        self.assertFalse(self.strategy._validate_data_consistency(inconsistent_data))
    
    @patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler')
    async def test_rate_limiting(self, mock_crawler_class):
        """Test rate limiting functionality."""
        mock_crawler = AsyncMock()
        mock_crawler_class.return_value = mock_crawler
        mock_crawler.__aenter__.return_value = mock_crawler
        mock_crawler.__aexit__.return_value = None
        
        # Mock delayed response to simulate rate limiting
        async def delayed_response():
            await asyncio.sleep(0.1)  # Small delay for testing
            result = Mock()
            result.success = True
            result.html = "<html><body><h1>Rate Limited</h1></body></html>"
            result.cleaned_html = "<h1>Rate Limited</h1>"
            result.markdown = "# Rate Limited"
            result.extracted_content = "Rate Limited Content"
            result.metadata = {"title": "Rate Limited Page"}
            result.links = {"internal": []}
            return result
        
        mock_crawler.arun.side_effect = delayed_response
        
        with patch.object(self.strategy, '_create_crawler_config'):
            with patch.object(self.strategy, '_should_apply_rate_limit', return_value=True):
                start_time = asyncio.get_event_loop().time()
                
                await self.strategy.execute("https://example.com")
                
                end_time = asyncio.get_event_loop().time()
                
                # Should have applied some delay for rate limiting
                self.assertGreater(end_time - start_time, 0.05)


class TestUniversalCrawl4AIStrategyIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for UniversalCrawl4AIStrategy."""
    
    async def test_full_crawl_workflow(self):
        """Test complete crawl workflow integration."""
        # This would be an integration test with actual crawl4ai
        # For now, we'll use mocks but test the full flow
        
        mock_context = Mock(spec=StrategyContext)
        mock_context.config = Mock()
        mock_context.config.CRAWL4AI_ENABLED = True
        mock_context.config.CRAWL4AI_MAX_PAGES = 3
        mock_context.config.CRAWL4AI_DEEP_CRAWL = True
        mock_context.config.CRAWL4AI_MEMORY_THRESHOLD = 80.0
        mock_context.config.CRAWL4AI_AI_PATHFINDING = True
        mock_context.config.USE_UNDETECTED_CHROMEDRIVER = True
        mock_context.config.PROGRESSIVE_DATA_COLLECTION = True
        mock_context.config.DATA_CONSISTENCY_CHECKS = True
        mock_context.config.CIRCUIT_BREAKER_ENABLED = True
        
        with patch('strategies.universal_crawl4ai_strategy.CRAWL4AI_AVAILABLE', True):
            strategy = UniversalCrawl4AIStrategy(context=mock_context)
            
            # Mock the entire crawl workflow
            with patch('strategies.universal_crawl4ai_strategy.AsyncWebCrawler') as mock_crawler_class:
                mock_crawler = AsyncMock()
                mock_crawler_class.return_value = mock_crawler
                mock_crawler.__aenter__.return_value = mock_crawler
                mock_crawler.__aexit__.return_value = None
                
                # Mock successful crawl results
                mock_result = Mock()
                mock_result.success = True
                mock_result.html = "<html><body><h1>Integration Test</h1></body></html>"
                mock_result.cleaned_html = "<h1>Integration Test</h1>"
                mock_result.markdown = "# Integration Test"
                mock_result.extracted_content = "Integration Test Content"
                mock_result.metadata = {"title": "Integration Test"}
                mock_result.links = {"internal": []}
                
                mock_crawler.arun.return_value = mock_result
                
                with patch.object(strategy, '_create_crawler_config'):
                    with patch.object(strategy, '_evaluate_page_relevance', return_value=True):
                        with patch.object(strategy, '_extract_page_data') as mock_extract:
                            mock_extract.return_value = PageData(
                                url="https://example.com",
                                title="Integration Test",
                                main_content="Integration Test Content",
                                structured_data={"title": "Integration Test"},
                                relevance_score=0.8,
                                timestamp=1234567890.0,
                                metadata={"source": "crawl4ai"}
                            )
                            
                            result = await strategy.execute(
                                "https://example.com",
                                intent_analysis={
                                    'intent_type': 'integration_test',
                                    'entities': [],
                                    'confidence': 0.8
                                }
                            )
                            
                            # Verify complete workflow execution
                            self.assertIsNotNone(result)
                            self.assertIn('extracted_data', result)
                            self.assertIn('metadata', result)
                            self.assertGreater(result['metadata']['relevance_score'], 0)


if __name__ == '__main__':
    unittest.main()
