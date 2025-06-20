"""
Unit tests for ExtractionCoordinator controller.

This test suite validates the extraction coordination functionality including:
- Coordinator initialization and component setup
- Intent analysis and extraction planning
- Multi-page data aggregation coordination
- AI schema generation integration
- Unified output processing and validation
- Caching and performance tracking
- Error handling and resilience
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import sys
import os
from typing import Dict, List, Any, Optional

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from controllers.extraction_coordinator import ExtractionCoordinator


class TestExtractionCoordinator(unittest.TestCase):
    """Test suite for ExtractionCoordinator controller."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock config
        self.mock_config = {
            'AI_SCHEMA_GENERATION_ENABLED': True,
            'REDIS_CACHE_ENABLED': False,  # Disable Redis for testing
            'PYDANTIC_VALIDATION_ENABLED': True,
            'MAX_PAGES': 10,
            'MAX_URLS': 20
        }
        
        # Create coordinator with mocked dependencies
        with patch('controllers.extraction_coordinator.get_config', return_value=self.mock_config), \
             patch('controllers.extraction_coordinator.UniversalIntentAnalyzer') as mock_intent, \
             patch('controllers.extraction_coordinator.IntelligentURLGenerator') as mock_url_gen, \
             patch('controllers.extraction_coordinator.AISchemaGenerator') as mock_schema_gen, \
             patch('controllers.extraction_coordinator.get_adaptive_scraper') as mock_scraper:
            
            # Set up mock components
            self.mock_intent_analyzer = Mock()
            self.mock_url_generator = Mock()
            self.mock_schema_generator = Mock()
            self.mock_adaptive_scraper = Mock()
            
            mock_intent.return_value = self.mock_intent_analyzer
            mock_url_gen.return_value = self.mock_url_generator
            mock_schema_gen.return_value = self.mock_schema_generator
            mock_scraper.return_value = self.mock_adaptive_scraper
            
            self.coordinator = ExtractionCoordinator()
        
        # Sample test data
        self.test_query = "best laptops under $1000"
        self.test_options = {
            'max_urls': 5,
            'max_pages': 10,
            'strategy': 'auto'
        }
        
        self.sample_intent_analysis = {
            'intent_type': 'product_search',
            'keywords': ['laptop', 'computer', 'technology'],
            'entities': [{'text': 'laptop', 'label': 'PRODUCT'}],
            'query_complexity': 0.7,
            'requires_deep_crawling': True,
            'target_urls': ['https://example.com/laptops']
        }
        
        self.sample_extraction_plan = {
            'operation_id': 'test_op_123',
            'strategy': 'composite_universal',
            'target_urls': ['https://example.com/laptops'],
            'intent_analysis': self.sample_intent_analysis,
            'pydantic_schema': Mock(),
            'progressive_collection': True,
            'ai_consolidation': True
        }
    
    def test_initialization_success(self):
        """Test successful initialization of ExtractionCoordinator."""
        self.assertIsNotNone(self.coordinator)
        self.assertIsNotNone(self.coordinator.intent_analyzer)
        self.assertIsNotNone(self.coordinator.url_generator)
        self.assertIsNotNone(self.coordinator.adaptive_scraper)
        self.assertIsNotNone(self.coordinator.schema_generator)  # Should be enabled
        self.assertIsNone(self.coordinator.redis_client)  # Disabled in test config
        self.assertIsInstance(self.coordinator.extraction_metrics, dict)
        self.assertIsInstance(self.coordinator.session_cache, dict)
    
    def test_initialization_without_schema_generator(self):
        """Test initialization when AI schema generation is disabled."""
        config_no_schema = self.mock_config.copy()
        config_no_schema['AI_SCHEMA_GENERATION_ENABLED'] = False
        
        with patch('controllers.extraction_coordinator.get_config', return_value=config_no_schema), \
             patch('controllers.extraction_coordinator.UniversalIntentAnalyzer'), \
             patch('controllers.extraction_coordinator.IntelligentURLGenerator'), \
             patch('controllers.extraction_coordinator.get_adaptive_scraper'):
            
            coordinator = ExtractionCoordinator()
            self.assertIsNone(coordinator.schema_generator)
    
    def test_initialization_with_redis_success(self):
        """Test initialization with successful Redis connection."""
        config_with_redis = self.mock_config.copy()
        config_with_redis['REDIS_CACHE_ENABLED'] = True
        
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        
        with patch('controllers.extraction_coordinator.get_config', return_value=config_with_redis), \
             patch('controllers.extraction_coordinator.REDIS_AVAILABLE', True), \
             patch('controllers.extraction_coordinator.redis.Redis', return_value=mock_redis), \
             patch('controllers.extraction_coordinator.UniversalIntentAnalyzer'), \
             patch('controllers.extraction_coordinator.IntelligentURLGenerator'), \
             patch('controllers.extraction_coordinator.get_adaptive_scraper'):
            
            coordinator = ExtractionCoordinator()
            self.assertEqual(coordinator.redis_client, mock_redis)
            mock_redis.ping.assert_called_once()
    
    def test_initialization_with_redis_failure(self):
        """Test initialization when Redis connection fails."""
        config_with_redis = self.mock_config.copy()
        config_with_redis['REDIS_CACHE_ENABLED'] = True
        
        mock_redis = Mock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        with patch('controllers.extraction_coordinator.get_config', return_value=config_with_redis), \
             patch('controllers.extraction_coordinator.REDIS_AVAILABLE', True), \
             patch('controllers.extraction_coordinator.redis.Redis', return_value=mock_redis), \
             patch('controllers.extraction_coordinator.UniversalIntentAnalyzer'), \
             patch('controllers.extraction_coordinator.IntelligentURLGenerator'), \
             patch('controllers.extraction_coordinator.get_adaptive_scraper'):
            
            coordinator = ExtractionCoordinator()
            self.assertIsNone(coordinator.redis_client)
    
    async def test_analyze_and_plan_success(self):
        """Test successful analysis and planning."""
        # Mock intent analyzer response
        self.mock_intent_analyzer.analyze_intent.return_value = self.sample_intent_analysis
        
        # Mock URL generator response
        self.mock_url_generator.generate_urls.return_value = {
            'urls': ['https://example.com/laptops', 'https://example.com/computers'],
            'confidence': 0.8
        }
        
        # Mock schema generator response
        mock_schema = Mock()
        self.mock_schema_generator.generate_schema_from_intent.return_value = mock_schema
        
        plan = await self.coordinator.analyze_and_plan(self.test_query, self.test_options)
        
        self.assertIsInstance(plan, dict)
        self.assertIn('operation_id', plan)
        self.assertIn('strategy', plan)
        self.assertIn('target_urls', plan)
        self.assertIn('intent_analysis', plan)
        self.assertIn('pydantic_schema', plan)
        
        # Verify component calls
        self.mock_intent_analyzer.analyze_intent.assert_called_once_with(self.test_query)
        self.mock_url_generator.generate_urls.assert_called_once()
        self.mock_schema_generator.generate_schema_from_intent.assert_called_once()
    
    async def test_analyze_and_plan_without_schema_generator(self):
        """Test analysis and planning when schema generator is disabled."""
        # Temporarily disable schema generator
        original_schema_gen = self.coordinator.schema_generator
        self.coordinator.schema_generator = None
        
        try:
            self.mock_intent_analyzer.analyze_intent.return_value = self.sample_intent_analysis
            self.mock_url_generator.generate_urls.return_value = {
                'urls': ['https://example.com/laptops'],
                'confidence': 0.8
            }
            
            plan = await self.coordinator.analyze_and_plan(self.test_query, self.test_options)
            
            self.assertIsInstance(plan, dict)
            self.assertNotIn('pydantic_schema', plan)  # Should not include schema
            
        finally:
            # Restore schema generator
            self.coordinator.schema_generator = original_schema_gen
    
    async def test_coordinate_extraction_success(self):
        """Test successful extraction coordination."""
        # Mock the analyze_and_plan method
        self.coordinator.analyze_and_plan = AsyncMock(return_value=self.sample_extraction_plan)
        
        # Mock the execute_extraction_plan method
        mock_execution_result = {
            'success': True,
            'results': {
                'items': [
                    {'title': 'Best Laptop 1', 'price': '$800'},
                    {'title': 'Best Laptop 2', 'price': '$900'}
                ],
                'metadata': {'pages_processed': 3, 'total_items': 2}
            },
            'metrics': {
                'execution_time': 5.2,
                'data_quality_score': 0.85,
                'pages_processed': 3
            }
        }
        self.coordinator.execute_extraction_plan = AsyncMock(return_value=mock_execution_result)
        
        # Mock caching and metrics methods
        self.coordinator._cache_extraction_results = AsyncMock()
        self.coordinator._record_extraction_metrics = AsyncMock()
        
        result = await self.coordinator.coordinate_extraction(self.test_query, self.test_options)
        
        self.assertTrue(result.get('success'))
        self.assertIn('results', result)
        self.assertIn('metrics', result)
        self.assertEqual(len(result['results']['items']), 2)
        
        # Verify method calls
        self.coordinator.analyze_and_plan.assert_called_once_with(self.test_query, self.test_options)
        self.coordinator.execute_extraction_plan.assert_called_once_with(self.sample_extraction_plan)
        self.coordinator._cache_extraction_results.assert_called_once()
        self.coordinator._record_extraction_metrics.assert_called_once()
    
    async def test_coordinate_extraction_with_cached_results(self):
        """Test extraction coordination with cached results."""
        # Mock cached results
        cached_result = {
            'success': True,
            'results': {'items': [{'title': 'Cached Laptop', 'price': '$750'}]},
            'cached': True
        }
        self.coordinator.get_cached_results = AsyncMock(return_value=cached_result)
        
        result = await self.coordinator.coordinate_extraction(self.test_query, self.test_options)
        
        self.assertTrue(result.get('success'))
        self.assertTrue(result.get('cached'))
        self.assertEqual(len(result['results']['items']), 1)
        
        # Should not proceed with extraction planning
        self.coordinator.get_cached_results.assert_called_once()
    
    async def test_coordinate_extraction_planning_failure(self):
        """Test extraction coordination when planning fails."""
        # Mock planning failure
        self.coordinator.analyze_and_plan = AsyncMock(side_effect=Exception("Planning failed"))
        
        result = await self.coordinator.coordinate_extraction(self.test_query, self.test_options)
        
        self.assertFalse(result.get('success'))
        self.assertIn('error', result)
        self.assertIn('Planning failed', result['error'])
    
    async def test_coordinate_extraction_execution_failure(self):
        """Test extraction coordination when execution fails."""
        # Mock successful planning but failed execution
        self.coordinator.analyze_and_plan = AsyncMock(return_value=self.sample_extraction_plan)
        self.coordinator.execute_extraction_plan = AsyncMock(return_value={
            'success': False,
            'error': 'Extraction failed'
        })
        
        result = await self.coordinator.coordinate_extraction(self.test_query, self.test_options)
        
        self.assertFalse(result.get('success'))
        self.assertIn('error', result)
    
    async def test_execute_extraction_plan_composite_strategy(self):
        """Test execution plan with composite universal strategy."""
        # Mock composite strategy execution
        self.mock_adaptive_scraper.execute_search_pipeline = AsyncMock(return_value={
            'success': True,
            'results': [
                {'title': 'Laptop A', 'price': '$800'},
                {'title': 'Laptop B', 'price': '$900'}
            ],
            'metrics': {'execution_time': 3.5}
        })
        
        result = await self.coordinator.execute_extraction_plan(self.sample_extraction_plan)
        
        self.assertTrue(result.get('success'))
        self.assertIn('results', result)
        self.assertIn('items', result['results'])
        self.assertEqual(len(result['results']['items']), 2)
        
        # Verify scraper was called with correct parameters
        self.mock_adaptive_scraper.execute_search_pipeline.assert_called_once()
        call_args = self.mock_adaptive_scraper.execute_search_pipeline.call_args
        self.assertIn('use_extraction_coordinator', call_args.kwargs)
        self.assertFalse(call_args.kwargs['use_extraction_coordinator'])  # Prevent recursion
    
    async def test_execute_extraction_plan_progressive_collection(self):
        """Test execution plan with progressive data collection."""
        plan_with_progressive = self.sample_extraction_plan.copy()
        plan_with_progressive['progressive_collection'] = True
        plan_with_progressive['target_urls'] = [
            'https://example.com/page1',
            'https://example.com/page2',
            'https://example.com/page3'
        ]
        
        # Mock multi-page results
        page_results = [
            {'success': True, 'results': [{'title': 'Item 1'}]},
            {'success': True, 'results': [{'title': 'Item 2'}]},
            {'success': True, 'results': [{'title': 'Item 3'}]}
        ]
        
        self.mock_adaptive_scraper.execute_search_pipeline = AsyncMock(side_effect=page_results)
        
        # Mock consolidation
        self.coordinator._consolidate_progressive_results = AsyncMock(return_value={
            'items': [{'title': 'Item 1'}, {'title': 'Item 2'}, {'title': 'Item 3'}],
            'metadata': {'pages_processed': 3, 'total_items': 3}
        })
        
        result = await self.coordinator.execute_extraction_plan(plan_with_progressive)
        
        self.assertTrue(result.get('success'))
        self.assertEqual(len(result['results']['items']), 3)
        
        # Should call scraper for each URL
        self.assertEqual(self.mock_adaptive_scraper.execute_search_pipeline.call_count, 3)
        self.coordinator._consolidate_progressive_results.assert_called_once()
    
    async def test_execute_extraction_plan_with_validation(self):
        """Test execution plan with Pydantic validation."""
        plan_with_validation = self.sample_extraction_plan.copy()
        plan_with_validation['pydantic_schema'] = Mock()
        
        # Mock extraction results
        self.mock_adaptive_scraper.execute_search_pipeline = AsyncMock(return_value={
            'success': True,
            'results': [{'title': 'Valid Item', 'price': '$800'}]
        })
        
        # Mock validation
        self.coordinator._validate_extracted_data = AsyncMock(return_value={
            'valid_items': [{'title': 'Valid Item', 'price': '$800'}],
            'validation_errors': [],
            'validation_status': 'passed'
        })
        
        result = await self.coordinator.execute_extraction_plan(plan_with_validation)
        
        self.assertTrue(result.get('success'))
        self.coordinator._validate_extracted_data.assert_called_once()
    
    def test_consolidate_progressive_results(self):
        """Test consolidation of progressive collection results."""
        page_results = [
            {'results': [{'title': 'Item 1', 'url': 'https://example.com/1'}]},
            {'results': [{'title': 'Item 2', 'url': 'https://example.com/2'}]},
            {'results': [{'title': 'Item 1', 'url': 'https://example.com/1'}]}  # Duplicate
        ]
        
        consolidated = self.coordinator._consolidate_progressive_results(page_results)
        
        self.assertIn('items', consolidated)
        self.assertIn('metadata', consolidated)
        
        # Should deduplicate based on URL
        items = consolidated['items']
        self.assertEqual(len(items), 2)  # Should remove duplicate
        
        metadata = consolidated['metadata']
        self.assertEqual(metadata['pages_processed'], 3)
        self.assertEqual(metadata['total_items'], 2)  # After deduplication
    
    async def test_validate_extracted_data_success(self):
        """Test successful data validation."""
        mock_schema = Mock()
        mock_schema.validate.return_value = {'title': 'Valid Item', 'price': '$800'}
        
        items = [{'title': 'Valid Item', 'price': '$800'}]
        
        result = await self.coordinator._validate_extracted_data(items, mock_schema)
        
        self.assertIn('valid_items', result)
        self.assertIn('validation_errors', result)
        self.assertIn('validation_status', result)
        self.assertEqual(len(result['valid_items']), 1)
        self.assertEqual(len(result['validation_errors']), 0)
        self.assertEqual(result['validation_status'], 'passed')
    
    async def test_validate_extracted_data_with_errors(self):
        """Test data validation with validation errors."""
        mock_schema = Mock()
        mock_schema.validate.side_effect = [
            {'title': 'Valid Item', 'price': '$800'},  # Valid
            Exception("Validation error")  # Invalid
        ]
        
        items = [
            {'title': 'Valid Item', 'price': '$800'},
            {'title': 'Invalid Item', 'price': 'invalid'}
        ]
        
        result = await self.coordinator._validate_extracted_data(items, mock_schema)
        
        self.assertEqual(len(result['valid_items']), 1)
        self.assertEqual(len(result['validation_errors']), 1)
        self.assertEqual(result['validation_status'], 'partial')
    
    async def test_cache_extraction_results_redis(self):
        """Test caching extraction results with Redis."""
        # Mock Redis client
        self.coordinator.redis_client = Mock()
        self.coordinator.redis_client.setex.return_value = True
        
        operation_id = "test_op_123"
        results = {'items': [{'title': 'Test Item'}]}
        
        await self.coordinator._cache_extraction_results(operation_id, results)
        
        # Verify Redis setex was called
        self.coordinator.redis_client.setex.assert_called_once()
        call_args = self.coordinator.redis_client.setex.call_args
        self.assertIn(operation_id, call_args[0][0])  # Key should contain operation_id
    
    async def test_cache_extraction_results_memory_fallback(self):
        """Test caching extraction results with memory fallback."""
        # No Redis client
        self.coordinator.redis_client = None
        
        operation_id = "test_op_123"
        results = {'items': [{'title': 'Test Item'}]}
        
        await self.coordinator._cache_extraction_results(operation_id, results)
        
        # Should use in-memory cache
        cache_key = f"extraction_result_{operation_id}"
        self.assertIn(cache_key, self.coordinator.session_cache)
        self.assertEqual(self.coordinator.session_cache[cache_key]['results'], results)
    
    async def test_get_cached_results_redis(self):
        """Test retrieving cached results from Redis."""
        # Mock Redis client
        self.coordinator.redis_client = Mock()
        cached_data = {
            'results': {'items': [{'title': 'Cached Item'}]},
            'metrics': {'cached': True}
        }
        self.coordinator.redis_client.get.return_value = json.dumps(cached_data)
        
        result = await self.coordinator.get_cached_results(self.test_query, self.test_options)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['results']['items'][0]['title'], 'Cached Item')
        self.assertTrue(result['metrics']['cached'])
    
    async def test_get_cached_results_memory(self):
        """Test retrieving cached results from memory."""
        # No Redis client
        self.coordinator.redis_client = None
        
        # Set up memory cache
        cache_key = f"extraction_result_{hash(self.test_query + str(self.test_options))}"
        cached_data = {
            'results': {'items': [{'title': 'Memory Cached Item'}]},
            'metrics': {'cached': True},
            'timestamp': time.time()
        }
        self.coordinator.session_cache[cache_key] = cached_data
        
        result = await self.coordinator.get_cached_results(self.test_query, self.test_options)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['results']['items'][0]['title'], 'Memory Cached Item')
    
    async def test_get_cached_results_not_found(self):
        """Test retrieving cached results when none exist."""
        # Mock Redis returning None
        self.coordinator.redis_client = Mock()
        self.coordinator.redis_client.get.return_value = None
        
        result = await self.coordinator.get_cached_results(self.test_query, self.test_options)
        
        self.assertIsNone(result)
    
    async def test_record_extraction_metrics(self):
        """Test extraction metrics recording."""
        operation_id = "test_op_123"
        execution_time = 5.2
        results = {
            'results': {'items': [{'title': 'Item 1'}, {'title': 'Item 2'}]},
            'metrics': {'data_quality_score': 0.85}
        }
        
        await self.coordinator._record_extraction_metrics(operation_id, execution_time, results)
        
        # Verify metrics were recorded
        self.assertIn(operation_id, self.coordinator.extraction_metrics)
        
        metrics = self.coordinator.extraction_metrics[operation_id]
        self.assertEqual(metrics['execution_time'], execution_time)
        self.assertEqual(metrics['items_extracted'], 2)
        self.assertEqual(metrics['data_quality_score'], 0.85)
        self.assertIn('timestamp', metrics)
    
    def test_get_extraction_metrics(self):
        """Test extraction metrics retrieval."""
        # Add some test metrics
        self.coordinator.extraction_metrics = {
            'op1': {
                'execution_time': 3.0,
                'items_extracted': 5,
                'data_quality_score': 0.8,
                'timestamp': time.time()
            },
            'op2': {
                'execution_time': 4.0,
                'items_extracted': 10,
                'data_quality_score': 0.9,
                'timestamp': time.time()
            }
        }
        
        metrics = self.coordinator.get_extraction_metrics()
        
        self.assertIn('total_operations', metrics)
        self.assertIn('average_execution_time', metrics)
        self.assertIn('average_items_per_operation', metrics)
        self.assertIn('average_quality_score', metrics)
        self.assertIn('recent_operations', metrics)
        
        self.assertEqual(metrics['total_operations'], 2)
        self.assertEqual(metrics['average_execution_time'], 3.5)
        self.assertEqual(metrics['average_items_per_operation'], 7.5)
        self.assertEqual(metrics['average_quality_score'], 0.85)
    
    def test_get_extraction_metrics_empty(self):
        """Test extraction metrics retrieval when no metrics exist."""
        self.coordinator.extraction_metrics = {}
        
        metrics = self.coordinator.get_extraction_metrics()
        
        self.assertEqual(metrics, {})
    
    async def test_shutdown(self):
        """Test coordinator shutdown."""
        # Mock Redis client
        self.coordinator.redis_client = Mock()
        self.coordinator.redis_client.close.return_value = None
        
        # Add some session cache data
        self.coordinator.session_cache['test_key'] = {'data': 'test'}
        
        await self.coordinator.shutdown()
        
        # Verify Redis client was closed
        self.coordinator.redis_client.close.assert_called_once()
        
        # Verify session cache was cleared
        self.assertEqual(len(self.coordinator.session_cache), 0)
    
    async def test_shutdown_redis_error(self):
        """Test coordinator shutdown with Redis error."""
        # Mock Redis client with error
        self.coordinator.redis_client = Mock()
        self.coordinator.redis_client.close.side_effect = Exception("Redis close error")
        
        # Should not raise exception
        await self.coordinator.shutdown()
        
        # Session cache should still be cleared
        self.assertEqual(len(self.coordinator.session_cache), 0)
    
    def test_error_handling_malformed_options(self):
        """Test error handling with malformed options."""
        malformed_options = "not a dict"
        
        # Should handle gracefully without crashing
        result = asyncio.run(
            self.coordinator.coordinate_extraction(self.test_query, malformed_options)
        )
        
        # Should return error result
        self.assertFalse(result.get('success'))
        self.assertIn('error', result)


class TestExtractionCoordinatorIntegration(unittest.TestCase):
    """Integration tests for ExtractionCoordinator with real components."""
    
    def test_singleton_pattern(self):
        """Test that get_extraction_coordinator returns singleton instance."""
        from controllers.extraction_coordinator import get_extraction_coordinator
        
        # Mock dependencies to avoid actual initialization
        with patch('controllers.extraction_coordinator.UniversalIntentAnalyzer'), \
             patch('controllers.extraction_coordinator.IntelligentURLGenerator'), \
             patch('controllers.extraction_coordinator.AISchemaGenerator'), \
             patch('controllers.extraction_coordinator.get_adaptive_scraper'):
            
            coordinator1 = get_extraction_coordinator()
            coordinator2 = get_extraction_coordinator()
            
            # Should return the same instance
            self.assertIs(coordinator1, coordinator2)


if __name__ == '__main__':
    unittest.main()
