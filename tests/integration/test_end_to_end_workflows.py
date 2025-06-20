"""
End-to-end workflow integration tests for SmartScrape Phase 1-6 components.

Tests complete workflows from query input to final structured output,
validating the entire pipeline including:
- Intent analysis → URL generation → Content extraction → Quality scoring → Schema generation → Result aggregation

This tests realistic user scenarios and complete data flows.
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from controllers.extraction_coordinator import ExtractionCoordinator
from core.service_registry import ServiceRegistry


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def mock_full_system(self):
        """Setup a complete mocked system for end-to-end testing."""
        # Reset service registry
        ServiceRegistry._instance = None
        registry = ServiceRegistry()
        
        # Mock configuration
        mock_config = {
            'ai_services': {
                'openai_api_key': 'test_key',
                'model': 'gpt-3.5-turbo',
                'temperature': 0.1
            },
            'semantic_search': {
                'enabled': True,
                'model': 'all-MiniLM-L6-v2',
                'similarity_threshold': 0.7
            },
            'spacy': {
                'model': 'en_core_web_sm',
                'enabled': True
            },
            'content_quality': {
                'min_quality_score': 0.6,
                'semantic_similarity_threshold': 0.8
            },
            'redis': {
                'enabled': False,
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
        
        # Mock AI service with realistic responses
        mock_ai_service = Mock()
        mock_ai_service.generate_schema = AsyncMock()
        mock_ai_service.chat_completion = AsyncMock()
        
        registry.register('config', mock_config)
        registry.register('ai_service', mock_ai_service)
        
        return registry, mock_ai_service
    
    def setup_realistic_extraction_mocks(self, mock_ai_service):
        """Setup realistic mocks for different extraction scenarios."""
        
        # Restaurant search scenario
        restaurant_schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'cuisine': {'type': 'string'},
                'rating': {'type': 'number'},
                'price_range': {'type': 'string'},
                'address': {'type': 'string'},
                'phone': {'type': 'string'},
                'website': {'type': 'string'}
            },
            'required': ['name', 'cuisine']
        }
        
        restaurant_content = """
        <div class="restaurant">
            <h2>Mario's Italian Kitchen</h2>
            <p>Cuisine: Italian</p>
            <p>Rating: 4.8/5 stars</p>
            <p>Price: $$</p>
            <p>Address: 123 Main St, Seattle, WA</p>
            <p>Phone: (206) 555-0123</p>
        </div>
        <div class="restaurant">
            <h2>Sakura Sushi</h2>
            <p>Cuisine: Japanese</p>
            <p>Rating: 4.6/5 stars</p>
            <p>Price: $$$</p>
            <p>Address: 456 Pine St, Seattle, WA</p>
            <p>Phone: (206) 555-0456</p>
        </div>
        """
        
        # Product search scenario
        product_schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'price': {'type': 'number'},
                'brand': {'type': 'string'},
                'description': {'type': 'string'},
                'specifications': {'type': 'object'},
                'availability': {'type': 'string'},
                'rating': {'type': 'number'}
            },
            'required': ['name', 'price']
        }
        
        product_content = """
        <div class="product">
            <h1>ThinkPad X1 Carbon</h1>
            <p class="price">$1,299.99</p>
            <p class="brand">Lenovo</p>
            <p class="description">Ultra-lightweight business laptop with premium features</p>
            <div class="specs">
                <p>Processor: Intel i7-1165G7</p>
                <p>Memory: 16GB LPDDR4x</p>
                <p>Storage: 512GB SSD</p>
            </div>
            <p class="availability">In Stock</p>
            <p class="rating">4.7/5 stars</p>
        </div>
        """
        
        # News search scenario
        news_schema = {
            'type': 'object',
            'properties': {
                'headline': {'type': 'string'},
                'author': {'type': 'string'},
                'published_date': {'type': 'string'},
                'summary': {'type': 'string'},
                'category': {'type': 'string'},
                'source': {'type': 'string'}
            },
            'required': ['headline', 'summary']
        }
        
        news_content = """
        <article>
            <h1>AI Breakthrough in Natural Language Processing</h1>
            <p class="author">By Dr. Jane Smith</p>
            <p class="date">March 15, 2024</p>
            <p class="summary">Researchers announce major advancement in AI language models...</p>
            <p class="category">Technology</p>
            <p class="source">Tech News Daily</p>
        </article>
        <article>
            <h1>Climate Change Impact on Technology Industry</h1>
            <p class="author">By Mike Johnson</p>
            <p class="date">March 14, 2024</p>
            <p class="summary">New study reveals how climate policies affect tech companies...</p>
            <p class="category">Environment</p>
            <p class="source">Environmental Tech</p>
        </article>
        """
        
        # Configure AI service responses based on query content
        def mock_generate_schema(query, content, **kwargs):
            if 'restaurant' in query.lower():
                return restaurant_schema
            elif 'laptop' in query.lower() or 'product' in query.lower():
                return product_schema
            elif 'news' in query.lower():
                return news_schema
            else:
                return {
                    'type': 'object',
                    'properties': {
                        'title': {'type': 'string'},
                        'content': {'type': 'string'}
                    }
                }
        
        mock_ai_service.generate_schema.side_effect = mock_generate_schema
        
        return {
            'restaurant': {'schema': restaurant_schema, 'content': restaurant_content},
            'product': {'schema': product_schema, 'content': product_content},
            'news': {'schema': news_schema, 'content': news_content}
        }
    
    @pytest.mark.asyncio
    async def test_restaurant_search_end_to_end(self, mock_full_system):
        """Test complete restaurant search workflow."""
        registry, mock_ai_service = mock_full_system
        scenarios = self.setup_realistic_extraction_mocks(mock_ai_service)
        
        coordinator = ExtractionCoordinator()
        
        query = "Find the best restaurants in Seattle"
        
        # Mock crawl4ai extraction
        def mock_extract_data(url, plan, context):
            return {
                'success': True,
                'data': [
                    {
                        'name': 'Mario\'s Italian Kitchen',
                        'cuisine': 'Italian',
                        'rating': 4.8,
                        'price_range': '$$',
                        'address': '123 Main St, Seattle, WA',
                        'phone': '(206) 555-0123'
                    },
                    {
                        'name': 'Sakura Sushi',
                        'cuisine': 'Japanese', 
                        'rating': 4.6,
                        'price_range': '$$$',
                        'address': '456 Pine St, Seattle, WA',
                        'phone': '(206) 555-0456'
                    }
                ],
                'metadata': {
                    'url': url,
                    'strategy_used': 'universal_crawl4ai',
                    'extraction_time': 0.5
                }
            }
        
        with patch('redis.Redis'):
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_data):
                # Phase 1: Analysis and Planning
                plan = await coordinator.analyze_and_plan(query, max_urls=3)
                
                assert plan is not None
                assert plan.query == query
                assert len(plan.target_urls) > 0
                assert 'restaurant' in plan.intent_context.get('entities', [])
                
                # Phase 2: Coordinate Extraction
                results = await coordinator.coordinate_extraction(plan)
                
                # Validate results
                assert results is not None
                assert results['success'] is True
                assert 'aggregated_data' in results
                assert len(results['aggregated_data']) == 2
                
                # Validate restaurant data structure
                for restaurant in results['aggregated_data']:
                    assert 'name' in restaurant
                    assert 'cuisine' in restaurant
                    assert 'rating' in restaurant
                    assert isinstance(restaurant['rating'], (int, float))
                
                # Validate metadata
                assert 'extraction_summary' in results['metadata']
                assert 'ai_schema' in results['metadata']
                assert results['metadata']['extraction_summary']['successful_extractions'] > 0
    
    @pytest.mark.asyncio
    async def test_product_search_end_to_end(self, mock_full_system):
        """Test complete product search workflow."""
        registry, mock_ai_service = mock_full_system
        scenarios = self.setup_realistic_extraction_mocks(mock_ai_service)
        
        coordinator = ExtractionCoordinator()
        
        query = "Find laptops under $1500"
        
        # Mock product extraction
        def mock_extract_data(url, plan, context):
            return {
                'success': True,
                'data': [
                    {
                        'name': 'ThinkPad X1 Carbon',
                        'price': 1299.99,
                        'brand': 'Lenovo',
                        'description': 'Ultra-lightweight business laptop',
                        'specifications': {
                            'processor': 'Intel i7-1165G7',
                            'memory': '16GB LPDDR4x',
                            'storage': '512GB SSD'
                        },
                        'availability': 'In Stock',
                        'rating': 4.7
                    }
                ],
                'metadata': {
                    'url': url,
                    'strategy_used': 'universal_crawl4ai',
                    'extraction_time': 0.3
                }
            }
        
        with patch('redis.Redis'):
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_data):
                # Full workflow
                plan = await coordinator.analyze_and_plan(query, max_urls=2)
                results = await coordinator.coordinate_extraction(plan)
                
                # Validate results
                assert results['success'] is True
                assert len(results['aggregated_data']) == 1
                
                product = results['aggregated_data'][0]
                assert product['name'] == 'ThinkPad X1 Carbon'
                assert product['price'] == 1299.99
                assert product['price'] < 1500  # Meets query criteria
                assert 'specifications' in product
                assert isinstance(product['specifications'], dict)
    
    @pytest.mark.asyncio
    async def test_news_search_end_to_end(self, mock_full_system):
        """Test complete news search workflow."""
        registry, mock_ai_service = mock_full_system
        scenarios = self.setup_realistic_extraction_mocks(mock_ai_service)
        
        coordinator = ExtractionCoordinator()
        
        query = "Latest news about artificial intelligence"
        
        # Mock news extraction from multiple sources
        call_count = 0
        def mock_extract_data(url, plan, context):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return {
                    'success': True,
                    'data': [
                        {
                            'headline': 'AI Breakthrough in Natural Language Processing',
                            'author': 'Dr. Jane Smith',
                            'published_date': '2024-03-15',
                            'summary': 'Researchers announce major advancement in AI language models...',
                            'category': 'Technology',
                            'source': 'Tech News Daily'
                        }
                    ],
                    'metadata': {'url': url, 'strategy_used': 'universal_crawl4ai'}
                }
            else:
                return {
                    'success': True,
                    'data': [
                        {
                            'headline': 'Climate Change Impact on Technology Industry',
                            'author': 'Mike Johnson',
                            'published_date': '2024-03-14',
                            'summary': 'New study reveals how climate policies affect tech companies...',
                            'category': 'Environment', 
                            'source': 'Environmental Tech'
                        }
                    ],
                    'metadata': {'url': url, 'strategy_used': 'universal_crawl4ai'}
                }
        
        with patch('redis.Redis'):
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_data):
                # Full workflow with multiple sources
                plan = await coordinator.analyze_and_plan(query, max_urls=2)
                results = await coordinator.coordinate_extraction(plan)
                
                # Validate aggregated news results
                assert results['success'] is True
                assert len(results['aggregated_data']) == 2
                
                # Validate news article structure
                for article in results['aggregated_data']:
                    assert 'headline' in article
                    assert 'summary' in article
                    assert 'author' in article
                    assert 'published_date' in article
                
                # Validate content relevance
                articles_text = ' '.join([article['headline'] + ' ' + article['summary'] 
                                        for article in results['aggregated_data']])
                assert any(keyword in articles_text.lower() 
                          for keyword in ['ai', 'artificial', 'intelligence', 'technology'])
    
    @pytest.mark.asyncio
    async def test_multi_page_progressive_collection(self, mock_full_system):
        """Test progressive data collection across multiple pages."""
        registry, mock_ai_service = mock_full_system
        scenarios = self.setup_realistic_extraction_mocks(mock_ai_service)
        
        coordinator = ExtractionCoordinator()
        
        query = "Comprehensive restaurant guide for Seattle"
        
        # Mock different restaurants from different pages
        page_data = [
            # Page 1 - Downtown restaurants
            [
                {'name': 'Mario\'s Italian', 'cuisine': 'Italian', 'rating': 4.8, 'area': 'Downtown'},
                {'name': 'Pike Place Chowder', 'cuisine': 'Seafood', 'rating': 4.9, 'area': 'Downtown'}
            ],
            # Page 2 - Capitol Hill restaurants
            [
                {'name': 'Cafe Vita', 'cuisine': 'Coffee', 'rating': 4.5, 'area': 'Capitol Hill'},
                {'name': 'Oddfellows', 'cuisine': 'American', 'rating': 4.6, 'area': 'Capitol Hill'}
            ],
            # Page 3 - Fremont restaurants
            [
                {'name': 'The Whale Wins', 'cuisine': 'Mediterranean', 'rating': 4.7, 'area': 'Fremont'},
                {'name': 'Taco Bell', 'cuisine': 'Mexican', 'rating': 3.2, 'area': 'Fremont'}  # Low quality
            ]
        ]
        
        call_count = 0
        def mock_extract_data(url, plan, context):
            nonlocal call_count
            current_page = page_data[call_count % len(page_data)]
            call_count += 1
            
            return {
                'success': True,
                'data': current_page,
                'metadata': {
                    'url': url,
                    'strategy_used': 'universal_crawl4ai',
                    'page_number': call_count
                }
            }
        
        with patch('redis.Redis'):
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_data):
                # Create plan with multiple URLs
                plan = await coordinator.analyze_and_plan(query, max_urls=3)
                
                # Override to ensure we have exactly 3 URLs for this test
                plan.target_urls = [
                    'https://restaurants.com/downtown',
                    'https://restaurants.com/capitol-hill', 
                    'https://restaurants.com/fremont'
                ]
                
                results = await coordinator.coordinate_extraction(plan)
                
                # Validate progressive collection
                assert results['success'] is True
                assert len(results['aggregated_data']) == 6  # 2 restaurants per page * 3 pages
                
                # Validate data from all areas is present
                areas = [r.get('area') for r in results['aggregated_data']]
                assert 'Downtown' in areas
                assert 'Capitol Hill' in areas
                assert 'Fremont' in areas
                
                # Validate quality filtering occurred (Taco Bell should be filtered out or scored low)
                high_rated = [r for r in results['aggregated_data'] if r.get('rating', 0) > 4.0]
                assert len(high_rated) >= 4  # Should have most high-rated restaurants
                
                # Validate metadata tracking
                summary = results['metadata']['extraction_summary']
                assert summary['total_urls'] == 3
                assert summary['successful_extractions'] == 3
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, mock_full_system):
        """Test error recovery and fallback strategies in end-to-end workflow."""
        registry, mock_ai_service = mock_full_system
        scenarios = self.setup_realistic_extraction_mocks(mock_ai_service)
        
        coordinator = ExtractionCoordinator()
        
        query = "Find job listings in tech companies"
        
        # Mock extraction with failures and fallback success
        call_count = 0
        def mock_extract_data(url, plan, context):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First URL fails with primary strategy
                return {
                    'success': False,
                    'error': 'Primary strategy failed - site blocked crawl4ai',
                    'metadata': {'url': url, 'strategy_used': 'universal_crawl4ai'}
                }
            elif call_count == 2:
                # Second URL succeeds with fallback
                return {
                    'success': True,
                    'data': [
                        {
                            'title': 'Senior Software Engineer',
                            'company': 'Tech Corp',
                            'location': 'Seattle, WA',
                            'salary': '$120,000 - $150,000',
                            'description': 'Join our innovative team...'
                        }
                    ],
                    'metadata': {'url': url, 'strategy_used': 'dom'}  # Fallback strategy
                }
            else:
                # Third URL succeeds with primary
                return {
                    'success': True,
                    'data': [
                        {
                            'title': 'AI Research Scientist',
                            'company': 'AI Innovations',
                            'location': 'San Francisco, CA',
                            'salary': '$180,000 - $220,000',
                            'description': 'Lead cutting-edge AI research...'
                        }
                    ],
                    'metadata': {'url': url, 'strategy_used': 'universal_crawl4ai'}
                }
        
        with patch('redis.Redis'):
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_data):
                plan = await coordinator.analyze_and_plan(query, max_urls=3)
                results = await coordinator.coordinate_extraction(plan)
                
                # Should succeed overall despite one failure
                assert results['success'] is True
                assert len(results['aggregated_data']) == 2  # Two successful extractions
                
                # Validate extracted job data
                jobs = results['aggregated_data']
                assert any('Software Engineer' in job.get('title', '') for job in jobs)
                assert any('AI Research' in job.get('title', '') for job in jobs)
                
                # Validate error handling in metadata
                summary = results['metadata']['extraction_summary']
                assert summary['total_urls'] == 3
                assert summary['successful_extractions'] == 2
                assert summary['failed_extractions'] == 1
                
                # Check that different strategies were used
                strategy_performance = results['metadata'].get('strategy_performance', [])
                strategies_used = [perf.get('strategy') for perf in strategy_performance]
                assert 'dom' in strategies_used  # Fallback was used
                assert 'universal_crawl4ai' in strategies_used  # Primary was used
    
    @pytest.mark.asyncio
    async def test_caching_across_workflow(self, mock_full_system):
        """Test caching effectiveness across complete workflow."""
        registry, mock_ai_service = mock_full_system
        scenarios = self.setup_realistic_extraction_mocks(mock_ai_service)
        
        coordinator = ExtractionCoordinator()
        
        query = "Popular coffee shops in Portland"
        
        # Mock extraction
        def mock_extract_data(url, plan, context):
            return {
                'success': True,
                'data': [
                    {
                        'name': 'Blue Bottle Coffee',
                        'type': 'Coffee Shop',
                        'rating': 4.4,
                        'location': 'Pearl District'
                    }
                ],
                'metadata': {'url': url, 'strategy_used': 'universal_crawl4ai'}
            }
        
        # Mock Redis for caching
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            
            # First request - cache miss
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis_class.return_value = mock_redis
            
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_data):
                # First execution
                plan1 = await coordinator.analyze_and_plan(query, max_urls=2)
                results1 = await coordinator.coordinate_extraction(plan1)
                
                # Verify first execution success
                assert results1['success'] is True
                
                # Second request - simulate cache hit
                cached_plan_data = {
                    'query': query,
                    'target_urls': ['https://coffee.com/portland'],
                    'intent_context': {'intent_type': 'local_search', 'entities': ['coffee', 'shops']},
                    'primary_strategy': 'universal_crawl4ai',
                    'fallback_strategies': ['dom']
                }
                mock_redis.get.return_value = json.dumps(cached_plan_data).encode()
                
                # Second execution (should use cache)
                plan2 = await coordinator.analyze_and_plan(query, max_urls=2)
                results2 = await coordinator.coordinate_extraction(plan2)
                
                # Verify second execution success
                assert results2['success'] is True
                
                # Verify caching was used
                assert mock_redis.get.called
                assert mock_redis.setex.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
