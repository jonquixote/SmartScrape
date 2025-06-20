"""
Integration tests for new SmartScrape components (Phases 1-6).

Tests the interaction between:
- UniversalIntentAnalyzer
- IntelligentURLGenerator  
- UniversalCrawl4AIStrategy
- AISchemaGenerator
- ContentQualityScorer
- CompositeUniversalStrategy
- ExtractionCoordinator

These tests verify that components work together correctly in realistic scenarios.
"""

import os
import sys
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any
import tempfile
import redis

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from components.universal_intent_analyzer import UniversalIntentAnalyzer
from components.intelligent_url_generator import IntelligentURLGenerator
from components.ai_schema_generator import AISchemaGenerator
from processors.content_quality_scorer import ContentQualityScorer
from strategies.universal_crawl4ai_strategy import UniversalCrawl4AIStrategy
from strategies.composite_universal_strategy import CompositeUniversalStrategy, UniversalExtractionPlan
from controllers.extraction_coordinator import ExtractionCoordinator
from core.service_registry import ServiceRegistry
from core.configuration import get_resource_config


class TestComponentIntegration:
    """Test interactions between major components."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
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
                'enabled': False,  # Use memory fallback for tests
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for testing."""
        return {
            'high_quality': """
            This is a comprehensive article about machine learning advances in 2024.
            The field has seen remarkable progress in natural language processing,
            computer vision, and reinforcement learning. Key developments include
            improved transformer architectures, better training methodologies,
            and more efficient model compression techniques.
            """,
            'low_quality': "Click here! Buy now! Limited time offer!!!",
            'product_content': """
            Product Name: Premium Laptop Pro
            Price: $1,299.99
            Specifications: 16GB RAM, 512GB SSD, Intel i7 processor
            Customer Rating: 4.8/5 stars
            Available Colors: Silver, Space Gray, Gold
            """,
            'news_content': """
            Breaking: Tech Company Announces Major AI Breakthrough
            Published: March 15, 2024
            A leading technology company today unveiled a new artificial
            intelligence system that promises to revolutionize data analysis.
            The system combines machine learning with quantum computing.
            """
        }
    
    @pytest.fixture
    def mock_services(self, mock_config):
        """Setup mock services and registry."""
        # Reset service registry
        ServiceRegistry._instance = None
        registry = ServiceRegistry()
        
        # Mock AI service
        mock_ai_service = Mock()
        mock_ai_service.generate_schema.return_value = {
            'type': 'object',
            'properties': {
                'title': {'type': 'string'},
                'content': {'type': 'string'},
                'metadata': {'type': 'object'}
            }
        }
        mock_ai_service.chat_completion = AsyncMock(return_value={
            'choices': [{'message': {'content': 'Generated response'}}]
        })
        
        registry.register('ai_service', mock_ai_service)
        registry.register('config', mock_config)
        
        return registry

    @pytest.mark.asyncio
    async def test_intent_analysis_to_url_generation_flow(self, mock_services, sample_content):
        """Test intent analysis flowing into URL generation."""
        # Setup components
        intent_analyzer = UniversalIntentAnalyzer()
        url_generator = IntelligentURLGenerator()
        
        # Test intent analysis
        query = "Find the best coffee shops in Seattle"
        intent_result = await intent_analyzer.analyze_intent(query)
        
        assert intent_result is not None
        assert 'intent_type' in intent_result
        assert 'entities' in intent_result
        
        # Test URL generation using intent
        urls = await url_generator.generate_intelligent_urls(
            query, 
            intent_result,
            max_urls=3
        )
        
        assert isinstance(urls, list)
        assert len(urls) > 0
        
        # Verify URLs are reasonable
        for url in urls:
            assert isinstance(url, str)
            assert url.startswith(('http://', 'https://'))

    @pytest.mark.asyncio
    async def test_content_quality_scoring_integration(self, mock_services, sample_content):
        """Test ContentQualityScorer integration with intent analysis."""
        # Setup components
        intent_analyzer = UniversalIntentAnalyzer()
        quality_scorer = ContentQualityScorer()
        
        # Analyze intent for content evaluation
        query = "Find high-quality tech articles"
        intent_result = await intent_analyzer.analyze_intent(query)
        
        # Score different content types
        high_quality_score = quality_scorer.score_content_quality(
            sample_content['high_quality'],
            intent_result.get('keywords', [])
        )
        
        low_quality_score = quality_scorer.score_content_quality(
            sample_content['low_quality'],
            intent_result.get('keywords', [])
        )
        
        # High quality content should score better
        assert high_quality_score > low_quality_score
        assert high_quality_score > 0.5  # Should be decent quality
        assert low_quality_score < 0.4   # Should be poor quality

    @pytest.mark.asyncio
    async def test_schema_generation_with_content_analysis(self, mock_services, sample_content):
        """Test AI schema generation integrated with content analysis."""
        # Setup components
        schema_generator = AISchemaGenerator()
        intent_analyzer = UniversalIntentAnalyzer()
        
        # Analyze intent first
        query = "Extract product information"
        intent_result = await intent_analyzer.analyze_intent(query)
        
        # Generate schema based on intent and sample content
        schema = await schema_generator.generate_schema(
            query,
            sample_content['product_content'],
            intent_context=intent_result
        )
        
        assert schema is not None
        assert 'type' in schema
        assert schema['type'] == 'object'
        assert 'properties' in schema
        
        # Schema should be relevant to product extraction
        properties = schema['properties']
        assert any(key.lower() in ['name', 'title', 'product'] for key in properties.keys())

    @pytest.mark.asyncio
    async def test_composite_strategy_coordination(self, mock_services):
        """Test CompositeUniversalStrategy coordinating multiple extraction methods."""
        # Setup strategy
        composite_strategy = CompositeUniversalStrategy()
        
        # Mock the URL and context
        test_url = "https://example.com/products"
        mock_context = Mock()
        mock_context.get_session.return_value = Mock()
        
        # Create extraction plan
        plan = UniversalExtractionPlan(
            query="Find product details",
            target_urls=[test_url],
            primary_strategy="universal_crawl4ai",
            fallback_strategies=["dom", "ai_guided"],
            intent_context={
                'intent_type': 'data_extraction',
                'entities': ['products'],
                'keywords': ['product', 'price', 'details']
            }
        )
        
        # Mock strategy results
        with patch.object(composite_strategy, '_execute_strategy') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'data': [{'title': 'Test Product', 'price': '$99.99'}],
                'metadata': {'strategy_used': 'universal_crawl4ai'}
            }
            
            # Execute extraction
            result = await composite_strategy.extract_data(test_url, plan, mock_context)
            
            assert result is not None
            assert result['success'] is True
            assert 'data' in result
            assert len(result['data']) > 0

    @pytest.mark.asyncio
    async def test_extraction_coordinator_full_workflow(self, mock_services, sample_content):
        """Test ExtractionCoordinator orchestrating the complete workflow."""
        # Setup coordinator
        coordinator = ExtractionCoordinator()
        
        query = "Find the best restaurants in San Francisco"
        
        # Mock Redis to avoid external dependencies
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis_class.return_value = mock_redis
            
            # Phase 1: Analysis and Planning
            plan = await coordinator.analyze_and_plan(query, max_urls=2)
            
            assert plan is not None
            assert hasattr(plan, 'query')
            assert hasattr(plan, 'target_urls')
            assert hasattr(plan, 'intent_context')
            
            # Phase 2: Execute extraction with mocked data
            with patch.object(coordinator.composite_strategy, 'extract_data') as mock_extract:
                mock_extract.return_value = {
                    'success': True,
                    'data': [
                        {'name': 'Restaurant A', 'rating': '4.5', 'cuisine': 'Italian'},
                        {'name': 'Restaurant B', 'rating': '4.8', 'cuisine': 'French'}
                    ],
                    'metadata': {'url': 'https://example.com', 'strategy': 'universal_crawl4ai'}
                }
                
                results = await coordinator.coordinate_extraction(plan)
                
                assert results is not None
                assert 'success' in results
                assert 'aggregated_data' in results
                assert 'metadata' in results

    @pytest.mark.asyncio
    async def test_progressive_data_collection(self, mock_services):
        """Test progressive data collection across multiple URLs."""
        coordinator = ExtractionCoordinator()
        
        # Create plan with multiple URLs
        plan = UniversalExtractionPlan(
            query="Collect news articles",
            target_urls=[
                "https://example.com/news1",
                "https://example.com/news2", 
                "https://example.com/news3"
            ],
            primary_strategy="universal_crawl4ai",
            fallback_strategies=["dom"],
            intent_context={
                'intent_type': 'content_aggregation',
                'entities': ['news', 'articles'],
                'keywords': ['news', 'articles', 'latest']
            }
        )
        
        # Mock extraction results for each URL
        mock_results = [
            {
                'success': True,
                'data': [{'title': 'News 1', 'content': 'Content 1'}],
                'metadata': {'url': 'https://example.com/news1'}
            },
            {
                'success': True,
                'data': [{'title': 'News 2', 'content': 'Content 2'}],
                'metadata': {'url': 'https://example.com/news2'}
            },
            {
                'success': True,
                'data': [{'title': 'News 3', 'content': 'Content 3'}],
                'metadata': {'url': 'https://example.com/news3'}
            }
        ]
        
        with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_results):
            results = await coordinator.coordinate_extraction(plan)
            
            # Verify progressive collection worked
            assert results['success'] is True
            assert len(results['aggregated_data']) == 3
            assert 'extraction_summary' in results['metadata']
            assert results['metadata']['extraction_summary']['total_urls'] == 3
            assert results['metadata']['extraction_summary']['successful_extractions'] == 3

    @pytest.mark.asyncio
    async def test_error_handling_and_fallback_strategies(self, mock_services):
        """Test error handling and fallback strategy coordination."""
        coordinator = ExtractionCoordinator()
        
        plan = UniversalExtractionPlan(
            query="Extract data with fallbacks",
            target_urls=["https://example.com/test"],
            primary_strategy="universal_crawl4ai",
            fallback_strategies=["dom", "ai_guided"],
            intent_context={'intent_type': 'data_extraction'}
        )
        
        # Mock primary strategy failure, fallback success
        def mock_extract_side_effect(url, plan, context):
            if hasattr(mock_extract_side_effect, 'call_count'):
                mock_extract_side_effect.call_count += 1
            else:
                mock_extract_side_effect.call_count = 1
                
            if mock_extract_side_effect.call_count == 1:
                # Primary strategy fails
                return {
                    'success': False,
                    'error': 'Primary strategy failed',
                    'metadata': {'strategy_used': 'universal_crawl4ai'}
                }
            else:
                # Fallback succeeds
                return {
                    'success': True,
                    'data': [{'title': 'Fallback Success', 'content': 'Data from fallback'}],
                    'metadata': {'strategy_used': 'dom'}
                }
        
        with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract_side_effect):
            results = await coordinator.coordinate_extraction(plan)
            
            # Should succeed with fallback
            assert results['success'] is True
            assert len(results['aggregated_data']) > 0
            # Should indicate fallback was used
            assert any('dom' in str(item) for item in results['metadata'].get('strategy_performance', []))

    @pytest.mark.asyncio
    async def test_caching_integration(self, mock_services):
        """Test caching integration across components."""
        coordinator = ExtractionCoordinator()
        
        query = "Test caching functionality"
        
        # Mock Redis for caching
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            
            # First call - cache miss
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            
            mock_redis_class.return_value = mock_redis
            
            # First extraction
            plan1 = await coordinator.analyze_and_plan(query, max_urls=1)
            
            # Second call - simulate cache hit
            cached_plan = {
                'query': query,
                'target_urls': ['https://example.com'],
                'intent_context': {'intent_type': 'test'},
                'primary_strategy': 'universal_crawl4ai'
            }
            mock_redis.get.return_value = json.dumps(cached_plan).encode()
            
            plan2 = await coordinator.analyze_and_plan(query, max_urls=1)
            
            # Both plans should be valid
            assert plan1 is not None
            assert plan2 is not None
            
            # Verify caching was attempted
            assert mock_redis.get.called
            assert mock_redis.setex.called

    @pytest.mark.asyncio 
    async def test_ai_schema_validation_integration(self, mock_services, sample_content):
        """Test AI-generated schema validation with extracted data."""
        coordinator = ExtractionCoordinator()
        schema_generator = AISchemaGenerator()
        
        # Generate schema for product data
        query = "Extract product information"
        schema = await schema_generator.generate_schema(
            query,
            sample_content['product_content']
        )
        
        # Mock extraction that should match schema
        mock_extracted_data = [
            {
                'title': 'Premium Laptop Pro',
                'content': 'High-performance laptop with excellent specifications',
                'metadata': {'price': '$1,299.99', 'rating': '4.8/5'}
            }
        ]
        
        # Test schema validation integration
        plan = UniversalExtractionPlan(
            query=query,
            target_urls=["https://example.com/product"],
            primary_strategy="universal_crawl4ai",
            fallback_strategies=[],
            intent_context={'intent_type': 'product_extraction'},
            ai_schema=schema
        )
        
        with patch.object(coordinator.composite_strategy, 'extract_data') as mock_extract:
            mock_extract.return_value = {
                'success': True,
                'data': mock_extracted_data,
                'metadata': {'strategy_used': 'universal_crawl4ai'}
            }
            
            results = await coordinator.coordinate_extraction(plan)
            
            # Should have successful extraction with schema
            assert results['success'] is True
            assert 'aggregated_data' in results
            assert 'ai_schema' in results['metadata']


class TestComponentPerformanceIntegration:
    """Test performance aspects of component integration."""
    
    @pytest.mark.asyncio
    async def test_concurrent_extraction_coordination(self, mock_services):
        """Test concurrent extraction across multiple URLs."""
        coordinator = ExtractionCoordinator()
        
        # Create plan with multiple URLs for concurrent processing
        urls = [f"https://example.com/page{i}" for i in range(5)]
        plan = UniversalExtractionPlan(
            query="Concurrent extraction test",
            target_urls=urls,
            primary_strategy="universal_crawl4ai",
            fallback_strategies=["dom"],
            intent_context={'intent_type': 'performance_test'}
        )
        
        # Mock fast extraction responses
        def mock_extract(url, plan, context):
            return {
                'success': True,
                'data': [{'url': url, 'content': f'Content from {url}'}],
                'metadata': {'extraction_time': 0.1}
            }
        
        with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract):
            import time
            start_time = time.time()
            
            results = await coordinator.coordinate_extraction(plan)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should complete reasonably quickly
            assert total_time < 5.0  # Should be much faster with mocking
            assert results['success'] is True
            assert len(results['aggregated_data']) == 5

    @pytest.mark.asyncio
    async def test_memory_efficient_large_data_handling(self, mock_services):
        """Test memory efficiency with large amounts of data."""
        coordinator = ExtractionCoordinator()
        quality_scorer = ContentQualityScorer()
        
        # Simulate large content processing
        large_content = "Large content section. " * 1000  # Simulate large text
        
        # Test quality scoring doesn't consume excessive memory
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large content
        score = quality_scorer.score_content_quality(large_content, ['test', 'content'])
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100
        assert isinstance(score, float)
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
