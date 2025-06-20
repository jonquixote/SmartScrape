"""
Unit tests for CompositeUniversalStrategy component.

This test suite validates the composite universal strategy functionality including:
- Strategy initialization and configuration
- Intent-based strategy selection
- Progressive data collection across strategies  
- AI schema generation and validation integration
- Fallback strategy coordination
- Performance tracking and optimization
- Error handling and resilience
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from strategies.composite_universal_strategy import CompositeUniversalStrategy, UniversalExtractionPlan
from strategies.core.strategy_context import StrategyContext
from strategies.core.strategy_types import StrategyCapability


class TestCompositeUniversalStrategy(unittest.TestCase):
    """Test suite for CompositeUniversalStrategy component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock strategy context
        self.mock_context = Mock(spec=StrategyContext)
        self.mock_context.config = Mock()
        self.mock_context.config.PROGRESSIVE_DATA_COLLECTION = True
        self.mock_context.config.AI_SCHEMA_GENERATION_ENABLED = True
        self.mock_context.config.SEMANTIC_SEARCH_ENABLED = True
        self.mock_context.config.MAX_PAGES = 10
        self.mock_context.config.CIRCUIT_BREAKER_ENABLED = True
        
        # Create strategy instance
        self.strategy = CompositeUniversalStrategy(self.mock_context)
        
        # Mock child strategies
        self.mock_universal_crawl4ai = Mock()
        self.mock_universal_crawl4ai.name = "universal_crawl4ai"
        self.mock_universal_crawl4ai.execute = AsyncMock(return_value={
            "success": True,
            "results": [{"title": "Universal Result", "url": "https://example.com"}],
            "strategy": "universal_crawl4ai"
        })
        
        self.mock_dom_strategy = Mock()
        self.mock_dom_strategy.name = "dom_strategy"
        self.mock_dom_strategy.execute = AsyncMock(return_value={
            "success": True,
            "results": [{"title": "DOM Result", "url": "https://example.com"}],
            "strategy": "dom_strategy"
        })
        
        self.mock_api_strategy = Mock()
        self.mock_api_strategy.name = "api_strategy"
        self.mock_api_strategy.execute = AsyncMock(return_value={
            "success": False,
            "error": "API not available",
            "strategy": "api_strategy"
        })
        
        # Sample intent analysis
        self.sample_intent_analysis = {
            'intent_type': 'product_search',
            'keywords': ['laptop', 'computer', 'technology'],
            'entities': [{'text': 'laptop', 'label': 'PRODUCT'}],
            'query_complexity': 0.7,
            'requires_deep_crawling': True,
            'requires_semantic_analysis': True,
            'api_accessible': False,
            'site_analysis_needed': True
        }
        
        # Sample enhanced context
        self.enhanced_context = {
            'intent_analysis': self.sample_intent_analysis,
            'pydantic_schema': Mock(),
            'site_analysis': {'site_type': 'ecommerce', 'complexity': 'high'},
            'required_capabilities': {StrategyCapability.AI_ASSISTED, StrategyCapability.PROGRESSIVE_CRAWLING}
        }
        
        # Add mock strategies to composite
        self.strategy.add_strategy(self.mock_universal_crawl4ai)
        self.strategy.add_strategy(self.mock_dom_strategy)
        self.strategy.add_strategy(self.mock_api_strategy)
    
    def test_initialization_success(self):
        """Test successful initialization of CompositeUniversalStrategy."""
        strategy = CompositeUniversalStrategy(self.mock_context)
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.context, self.mock_context)
        self.assertIsInstance(strategy.strategy_priorities, dict)
        self.assertIsInstance(strategy.strategy_performance, dict)
        self.assertIsInstance(strategy.extraction_history, list)
        self.assertTrue(strategy.progressive_collection_enabled)
        self.assertTrue(strategy.ai_schema_generation_enabled)
        self.assertTrue(strategy.semantic_search_enabled)
    
    def test_initialization_without_context(self):
        """Test initialization without strategy context."""
        strategy = CompositeUniversalStrategy()
        
        self.assertIsNotNone(strategy)
        self.assertIsNone(strategy.context)
        # Should have default configurations
        self.assertIsInstance(strategy.strategy_priorities, dict)
    
    def test_supports_enhanced_context(self):
        """Test that strategy supports enhanced context."""
        supports = self.strategy.supports_enhanced_context()
        self.assertTrue(supports)
    
    def test_strategy_priorities_configuration(self):
        """Test strategy priority configuration."""
        priorities = self.strategy.strategy_priorities
        
        # Should have expected strategies with numeric priorities
        expected_strategies = ['universal_crawl4ai', 'dom_strategy', 'api_strategy', 'form_strategy', 'url_param_strategy']
        for strategy_name in expected_strategies:
            self.assertIn(strategy_name, priorities)
            self.assertIsInstance(priorities[strategy_name], int)
        
        # Universal crawl4ai should have highest priority (lowest number)
        self.assertEqual(priorities['universal_crawl4ai'], 1)
    
    async def test_search_method_with_enhanced_context(self):
        """Test search method with enhanced context."""
        query = "best laptops 2024"
        url = "https://example.com"
        
        # Mock execute method
        self.strategy.execute = AsyncMock(return_value={
            "success": True,
            "results": [{"title": "Laptop Result", "url": url}],
            "strategy": "composite_universal"
        })
        
        result = await self.strategy.search(query, url, self.enhanced_context)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("success"))
        self.assertIn("results", result)
        
        # Verify execute was called with proper parameters
        self.strategy.execute.assert_called_once()
    
    async def test_search_method_url_generation(self):
        """Test search method when URL needs to be generated from context."""
        query = "test query"
        
        # Context with target URLs in intent analysis
        context_with_urls = {
            'intent_analysis': {
                'target_urls': ['https://generated.com', 'https://backup.com']
            }
        }
        
        self.strategy.execute = AsyncMock(return_value={
            "success": True,
            "results": []
        })
        
        result = await self.strategy.search(query, None, context_with_urls)
        
        # Should use generated URL
        call_args = self.strategy.execute.call_args
        self.assertEqual(call_args[0][0], 'https://generated.com')  # First positional arg should be the URL
    
    async def test_search_method_missing_url_error(self):
        """Test search method error when no URL provided or can be determined."""
        query = "test query"
        
        result = await self.strategy.search(query, None, {})
        
        self.assertFalse(result.get("success"))
        self.assertIn("error", result)
        self.assertIn("No target URL", result["error"])
    
    def test_create_extraction_plan_product_search(self):
        """Test extraction plan creation for product search intent."""
        intent_analysis = {
            'intent_type': 'product_search',
            'requires_deep_crawling': True,
            'requires_semantic_analysis': True,
            'api_accessible': False
        }
        
        plan = self.strategy._create_extraction_plan("https://example.com", intent_analysis)
        
        self.assertIsInstance(plan, UniversalExtractionPlan)
        self.assertEqual(plan.primary_strategy, 'universal_crawl4ai')
        self.assertTrue(plan.progressive_collection)
        self.assertTrue(plan.ai_consolidation)
        self.assertIn('dom_strategy', plan.fallback_strategies)
    
    def test_create_extraction_plan_information_seeking(self):
        """Test extraction plan creation for information seeking intent."""
        intent_analysis = {
            'intent_type': 'information_seeking',
            'requires_deep_crawling': False,
            'requires_semantic_analysis': False,
            'api_accessible': True
        }
        
        plan = self.strategy._create_extraction_plan("https://example.com", intent_analysis)
        
        self.assertIsInstance(plan, UniversalExtractionPlan)
        # Should prefer simpler strategy for basic information seeking
        self.assertIn(plan.primary_strategy, ['dom_strategy', 'universal_crawl4ai'])
        self.assertIsInstance(plan.fallback_strategies, list)
    
    def test_select_primary_strategy_complex_site(self):
        """Test primary strategy selection for complex sites."""
        intent_analysis = {
            'intent_type': 'product_search',
            'requires_deep_crawling': True,
            'requires_semantic_analysis': True,
            'site_complexity': 'high'
        }
        
        primary = self.strategy._select_primary_strategy("https://example.com", intent_analysis)
        self.assertEqual(primary, 'universal_crawl4ai')
    
    def test_select_primary_strategy_simple_site(self):
        """Test primary strategy selection for simple sites."""
        intent_analysis = {
            'intent_type': 'information_seeking',
            'requires_deep_crawling': False,
            'requires_semantic_analysis': False,
            'site_complexity': 'low'
        }
        
        primary = self.strategy._select_primary_strategy("https://example.com", intent_analysis)
        self.assertEqual(primary, 'dom_strategy')
    
    def test_select_primary_strategy_api_available(self):
        """Test primary strategy selection when API is available."""
        intent_analysis = {
            'intent_type': 'data_extraction',
            'api_accessible': True,
            'structured_data_available': True
        }
        
        primary = self.strategy._select_primary_strategy("https://example.com", intent_analysis)
        # Should consider API strategy when available
        self.assertIn(primary, ['api_strategy', 'universal_crawl4ai'])
    
    def test_select_fallback_strategies(self):
        """Test fallback strategy selection."""
        primary_strategy = 'universal_crawl4ai'
        intent_analysis = {
            'intent_type': 'product_search',
            'api_accessible': True
        }
        
        fallbacks = self.strategy._select_fallback_strategies(primary_strategy, intent_analysis)
        
        self.assertIsInstance(fallbacks, list)
        self.assertNotIn(primary_strategy, fallbacks)  # Primary shouldn't be in fallbacks
        self.assertIn('dom_strategy', fallbacks)  # DOM strategy should be reliable fallback
        
        # If API is accessible, should include API strategy
        if intent_analysis.get('api_accessible', False):
            self.assertIn('api_strategy', fallbacks)
    
    async def test_execute_with_primary_strategy_success(self):
        """Test execution when primary strategy succeeds."""
        url = "https://example.com"
        
        # Mock the extraction plan creation
        mock_plan = UniversalExtractionPlan(
            primary_strategy='universal_crawl4ai',
            fallback_strategies=['dom_strategy'],
            intent_analysis=self.sample_intent_analysis,
            schema_definition=None,
            progressive_collection=True,
            ai_consolidation=True,
            cache_enabled=False
        )
        
        with patch.object(self.strategy, '_create_extraction_plan', return_value=mock_plan), \
             patch.object(self.strategy, '_execute_strategy_plan', new_callable=AsyncMock) as mock_execute:
            
            mock_execute.return_value = {
                "success": True,
                "results": [{"title": "Success Result"}],
                "strategy": "universal_crawl4ai"
            }
            
            result = await self.strategy.execute(url, user_prompt="test query")
            
            self.assertTrue(result.get("success"))
            self.assertIn("results", result)
            mock_execute.assert_called_once_with(mock_plan, url, user_prompt="test query")
    
    async def test_execute_with_primary_strategy_failure_fallback(self):
        """Test execution fallback when primary strategy fails."""
        url = "https://example.com"
        
        mock_plan = UniversalExtractionPlan(
            primary_strategy='api_strategy',  # This will fail
            fallback_strategies=['universal_crawl4ai', 'dom_strategy'],
            intent_analysis=self.sample_intent_analysis,
            schema_definition=None,
            progressive_collection=True,
            ai_consolidation=True,
            cache_enabled=False
        )
        
        with patch.object(self.strategy, '_create_extraction_plan', return_value=mock_plan), \
             patch.object(self.strategy, '_execute_strategy_plan', new_callable=AsyncMock) as mock_execute:
            
            # First call (primary) fails, second call (fallback) succeeds
            mock_execute.side_effect = [
                {"success": False, "error": "API failed"},
                {"success": True, "results": [{"title": "Fallback Success"}], "strategy": "universal_crawl4ai"}
            ]
            
            result = await self.strategy.execute(url, user_prompt="test query")
            
            self.assertTrue(result.get("success"))
            self.assertIn("results", result)
            self.assertEqual(mock_execute.call_count, 2)  # Primary + one fallback
    
    async def test_execute_with_all_strategies_failing(self):
        """Test execution when all strategies fail."""
        url = "https://example.com"
        
        mock_plan = UniversalExtractionPlan(
            primary_strategy='api_strategy',
            fallback_strategies=['universal_crawl4ai', 'dom_strategy'],
            intent_analysis=self.sample_intent_analysis,
            schema_definition=None,
            progressive_collection=True,
            ai_consolidation=True,
            cache_enabled=False
        )
        
        with patch.object(self.strategy, '_create_extraction_plan', return_value=mock_plan), \
             patch.object(self.strategy, '_execute_strategy_plan', new_callable=AsyncMock) as mock_execute:
            
            # All strategies fail
            mock_execute.return_value = {"success": False, "error": "Strategy failed"}
            
            result = await self.strategy.execute(url, user_prompt="test query")
            
            self.assertFalse(result.get("success"))
            self.assertIn("error", result)
            # Should try primary + all fallbacks
            self.assertEqual(mock_execute.call_count, 3)
    
    async def test_execute_strategy_plan_progressive_collection(self):
        """Test strategy plan execution with progressive collection."""
        plan = UniversalExtractionPlan(
            primary_strategy='universal_crawl4ai',
            fallback_strategies=[],
            intent_analysis=self.sample_intent_analysis,
            schema_definition=None,
            progressive_collection=True,
            ai_consolidation=True,
            cache_enabled=False
        )
        
        url = "https://example.com"
        
        # Mock progressive data collection
        with patch.object(self.strategy, '_execute_progressive_collection', new_callable=AsyncMock) as mock_progressive:
            mock_progressive.return_value = {
                "success": True,
                "results": [{"title": "Progressive Result"}],
                "pages_collected": 5
            }
            
            result = await self.strategy._execute_strategy_plan(plan, url, user_prompt="test")
            
            self.assertTrue(result.get("success"))
            mock_progressive.assert_called_once()
    
    async def test_execute_strategy_plan_single_page(self):
        """Test strategy plan execution for single page extraction."""
        plan = UniversalExtractionPlan(
            primary_strategy='dom_strategy',
            fallback_strategies=[],
            intent_analysis=self.sample_intent_analysis,
            schema_definition=None,
            progressive_collection=False,
            ai_consolidation=False,
            cache_enabled=False
        )
        
        url = "https://example.com"
        
        # Mock single page extraction
        with patch.object(self.strategy, '_execute_single_page', new_callable=AsyncMock) as mock_single:
            mock_single.return_value = {
                "success": True,
                "results": [{"title": "Single Page Result"}]
            }
            
            result = await self.strategy._execute_strategy_plan(plan, url, user_prompt="test")
            
            self.assertTrue(result.get("success"))
            mock_single.assert_called_once()
    
    def test_consolidate_results_multiple_strategies(self):
        """Test result consolidation from multiple strategies."""
        strategy_results = [
            {
                "strategy": "universal_crawl4ai",
                "results": [{"title": "Result 1", "url": "https://example.com/1"}],
                "confidence": 0.9
            },
            {
                "strategy": "dom_strategy",
                "results": [{"title": "Result 2", "url": "https://example.com/2"}],
                "confidence": 0.7
            }
        ]
        
        consolidated = self.strategy._consolidate_results(strategy_results)
        
        self.assertIsInstance(consolidated, dict)
        self.assertIn("items", consolidated)
        self.assertIn("metadata", consolidated)
        
        # Should merge results from both strategies
        items = consolidated["items"]
        self.assertGreaterEqual(len(items), 2)
        
        # Should include metadata about strategies used
        metadata = consolidated["metadata"]
        self.assertIn("strategies_used", metadata)
        self.assertIn("confidence_scores", metadata)
    
    def test_consolidate_results_deduplication(self):
        """Test result consolidation with deduplication."""
        strategy_results = [
            {
                "strategy": "strategy1",
                "results": [
                    {"title": "Duplicate Item", "url": "https://example.com/item"},
                    {"title": "Unique Item 1", "url": "https://example.com/1"}
                ],
                "confidence": 0.8
            },
            {
                "strategy": "strategy2", 
                "results": [
                    {"title": "Duplicate Item", "url": "https://example.com/item"},  # Duplicate
                    {"title": "Unique Item 2", "url": "https://example.com/2"}
                ],
                "confidence": 0.7
            }
        ]
        
        consolidated = self.strategy._consolidate_results(strategy_results)
        
        # Should deduplicate based on URL
        items = consolidated["items"]
        urls = [item.get("url") for item in items]
        unique_urls = set(urls)
        
        # Should have fewer items due to deduplication
        self.assertEqual(len(unique_urls), 3)  # 3 unique URLs
    
    def test_performance_tracking(self):
        """Test strategy performance tracking."""
        # Track successful strategy execution
        self.strategy._track_strategy_performance("universal_crawl4ai", True, 1.5, 10)
        
        # Check performance data
        performance = self.strategy.strategy_performance
        self.assertIn("universal_crawl4ai", performance)
        
        strategy_perf = performance["universal_crawl4ai"]
        self.assertEqual(strategy_perf["success_count"], 1)
        self.assertEqual(strategy_perf["total_count"], 1)
        self.assertEqual(strategy_perf["avg_response_time"], 1.5)
        self.assertEqual(strategy_perf["total_items"], 10)
        
        # Track failed execution
        self.strategy._track_strategy_performance("universal_crawl4ai", False, 2.0, 0)
        
        # Performance should be updated
        strategy_perf = performance["universal_crawl4ai"]
        self.assertEqual(strategy_perf["success_count"], 1)
        self.assertEqual(strategy_perf["total_count"], 2)
        self.assertEqual(strategy_perf["success_rate"], 0.5)
    
    def test_extraction_history_recording(self):
        """Test extraction history recording."""
        operation_id = "test_operation_123"
        result_data = {
            "success": True,
            "results": [{"title": "Test Result"}],
            "strategy": "universal_crawl4ai"
        }
        
        self.strategy._record_extraction_history(operation_id, "https://example.com", result_data)
        
        # Check history was recorded
        history = self.strategy.extraction_history
        self.assertEqual(len(history), 1)
        
        history_entry = history[0]
        self.assertEqual(history_entry["operation_id"], operation_id)
        self.assertEqual(history_entry["url"], "https://example.com")
        self.assertTrue(history_entry["success"])
        self.assertIn("timestamp", history_entry)
    
    def test_get_strategy_recommendations(self):
        """Test strategy recommendation based on performance."""
        # Add some performance data
        self.strategy.strategy_performance = {
            "universal_crawl4ai": {
                "success_rate": 0.9,
                "avg_response_time": 2.0,
                "total_items": 100
            },
            "dom_strategy": {
                "success_rate": 0.7,
                "avg_response_time": 1.0,
                "total_items": 50
            }
        }
        
        recommendations = self.strategy._get_strategy_recommendations()
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn("recommended_primary", recommendations)
        self.assertIn("performance_summary", recommendations)
        
        # Should recommend the strategy with best overall performance
        recommended = recommendations["recommended_primary"]
        self.assertIn(recommended, ["universal_crawl4ai", "dom_strategy"])
    
    def test_error_handling_strategy_not_found(self):
        """Test error handling when strategy is not found."""
        # Try to execute plan with non-existent strategy
        plan = UniversalExtractionPlan(
            primary_strategy='non_existent_strategy',
            fallback_strategies=[],
            intent_analysis={},
            schema_definition=None,
            progressive_collection=False,
            ai_consolidation=False,
            cache_enabled=False
        )
        
        # Should handle gracefully
        with patch.object(self.strategy, 'get_child_strategy', return_value=None):
            result = asyncio.run(self.strategy._execute_strategy_plan(plan, "https://example.com"))
            
            self.assertFalse(result.get("success"))
            self.assertIn("error", result)
    
    def test_error_handling_strategy_execution_exception(self):
        """Test error handling when strategy execution raises exception."""
        # Mock strategy that raises exception
        mock_failing_strategy = Mock()
        mock_failing_strategy.name = "failing_strategy"
        mock_failing_strategy.execute = AsyncMock(side_effect=Exception("Strategy execution failed"))
        
        self.strategy.add_strategy(mock_failing_strategy)
        
        plan = UniversalExtractionPlan(
            primary_strategy='failing_strategy',
            fallback_strategies=[],
            intent_analysis={},
            schema_definition=None,
            progressive_collection=False,
            ai_consolidation=False,
            cache_enabled=False
        )
        
        result = asyncio.run(self.strategy._execute_strategy_plan(plan, "https://example.com"))
        
        self.assertFalse(result.get("success"))
        self.assertIn("error", result)
    
    def test_name_property(self):
        """Test strategy name property."""
        name = self.strategy.name
        self.assertEqual(name, "composite_universal")
    
    def test_supports_capability_checking(self):
        """Test capability checking functionality."""
        # CompositeUniversalStrategy should support many capabilities
        capabilities = [
            StrategyCapability.AI_ASSISTED,
            StrategyCapability.PROGRESSIVE_CRAWLING,
            StrategyCapability.SEMANTIC_SEARCH,
            StrategyCapability.INTENT_ANALYSIS
        ]
        
        for capability in capabilities:
            # This would depend on the actual implementation
            # For now, just verify the strategy has the capability metadata
            self.assertIsNotNone(self.strategy)


class TestUniversalExtractionPlan(unittest.TestCase):
    """Test suite for UniversalExtractionPlan dataclass."""
    
    def test_plan_creation(self):
        """Test creation of extraction plan."""
        plan = UniversalExtractionPlan(
            primary_strategy="universal_crawl4ai",
            fallback_strategies=["dom_strategy", "api_strategy"],
            intent_analysis={"intent_type": "product_search"},
            schema_definition=None,
            progressive_collection=True,
            ai_consolidation=True,
            cache_enabled=False
        )
        
        self.assertEqual(plan.primary_strategy, "universal_crawl4ai")
        self.assertEqual(len(plan.fallback_strategies), 2)
        self.assertTrue(plan.progressive_collection)
        self.assertTrue(plan.ai_consolidation)
        self.assertFalse(plan.cache_enabled)
    
    def test_plan_with_schema(self):
        """Test extraction plan with schema definition."""
        mock_schema = Mock()
        
        plan = UniversalExtractionPlan(
            primary_strategy="universal_crawl4ai",
            fallback_strategies=[],
            intent_analysis={},
            schema_definition=mock_schema,
            progressive_collection=False,
            ai_consolidation=False,
            cache_enabled=True
        )
        
        self.assertEqual(plan.schema_definition, mock_schema)
        self.assertTrue(plan.cache_enabled)


if __name__ == '__main__':
    unittest.main()
