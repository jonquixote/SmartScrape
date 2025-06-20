"""
Performance tests for SmartScrape Phase 1-6 components.

Tests performance, memory usage, and scalability of:
- UniversalIntentAnalyzer
- IntelligentURLGenerator
- UniversalCrawl4AIStrategy  
- AISchemaGenerator
- ContentQualityScorer
- CompositeUniversalStrategy
- ExtractionCoordinator

Includes benchmarks, stress tests, and optimization validation.
"""

import os
import sys
import pytest
import asyncio
import time
import psutil
import gc
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any
import concurrent.futures
import threading
from memory_profiler import profile
import cProfile
import pstats
from io import StringIO

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


class PerformanceTestBase:
    """Base class for performance testing utilities."""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def time_async_function(func, *args, **kwargs):
        """Time an async function execution."""
        async def timed_execution():
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time
        
        return asyncio.run(timed_execution())
    
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """Profile a function and return stats."""
        profiler = cProfile.Profile()
        profiler.enable()
        
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
        
        profiler.disable()
        
        # Get stats
        stats_stream = StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats()
        
        return result, stats_stream.getvalue()


class TestUniversalIntentAnalyzerPerformance(PerformanceTestBase):
    """Performance tests for UniversalIntentAnalyzer."""
    
    @pytest.fixture
    def intent_analyzer(self):
        """Setup intent analyzer with mocked dependencies."""
        with patch('spacy.load') as mock_spacy:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_doc.ents = []
            mock_doc.noun_chunks = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            analyzer = UniversalIntentAnalyzer()
            return analyzer
    
    @pytest.mark.asyncio
    async def test_intent_analysis_latency(self, intent_analyzer):
        """Test intent analysis response time."""
        queries = [
            "Find the best restaurants in New York",
            "Get latest news about technology",
            "Search for jobs in software engineering",
            "Compare prices of laptops",
            "Book a flight to Paris"
        ]
        
        total_time = 0
        max_time = 0
        min_time = float('inf')
        
        for query in queries:
            start_time = time.perf_counter()
            result = await intent_analyzer.analyze_intent(query)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            total_time += execution_time
            max_time = max(max_time, execution_time)
            min_time = min(min_time, execution_time)
            
            # Verify result is valid
            assert result is not None
            assert isinstance(result, dict)
        
        avg_time = total_time / len(queries)
        
        # Performance assertions
        assert avg_time < 0.5, f"Average analysis time {avg_time:.3f}s exceeds 500ms"
        assert max_time < 1.0, f"Max analysis time {max_time:.3f}s exceeds 1s"
        
        print(f"Intent Analysis Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Min: {min_time:.3f}s")
        print(f"  Max: {max_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_intent_analysis_memory_usage(self, intent_analyzer):
        """Test memory usage during intent analysis."""
        # Generate large query text
        large_query = "Find information about " + " ".join([
            f"topic_{i} with details and specifications" 
            for i in range(100)
        ])
        
        memory_before = self.get_memory_usage()
        
        # Process multiple large queries
        for _ in range(10):
            await intent_analyzer.analyze_intent(large_query)
            gc.collect()  # Force garbage collection
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Memory increase {memory_increase:.1f}MB too high"
        
        print(f"Intent Analysis Memory Usage:")
        print(f"  Before: {memory_before:.1f}MB")
        print(f"  After: {memory_after:.1f}MB") 
        print(f"  Increase: {memory_increase:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_concurrent_intent_analysis(self, intent_analyzer):
        """Test concurrent intent analysis performance."""
        queries = [f"Query number {i} about various topics" for i in range(20)]
        
        start_time = time.perf_counter()
        
        # Process concurrently
        tasks = [intent_analyzer.analyze_intent(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # All results should be valid
        assert len(results) == len(queries)
        for result in results:
            assert result is not None
            assert isinstance(result, dict)
        
        # Should be faster than sequential processing
        avg_time_per_query = total_time / len(queries)
        assert avg_time_per_query < 0.2, f"Concurrent processing too slow: {avg_time_per_query:.3f}s per query"
        
        print(f"Concurrent Intent Analysis:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg per query: {avg_time_per_query:.3f}s")


class TestContentQualityScorerPerformance(PerformanceTestBase):
    """Performance tests for ContentQualityScorer."""
    
    @pytest.fixture
    def quality_scorer(self):
        """Setup quality scorer with mocked dependencies."""
        return ContentQualityScorer()
    
    def test_quality_scoring_performance(self, quality_scorer):
        """Test content quality scoring performance."""
        # Generate content of various sizes
        test_contents = [
            "Short content piece.",
            "Medium length content with more details and information. " * 10,
            "Long content piece with extensive information and details. " * 100,
            "Very long content piece with comprehensive information. " * 500
        ]
        
        keywords = ['test', 'content', 'information', 'details']
        
        for i, content in enumerate(test_contents):
            start_time = time.perf_counter()
            
            # Score content quality
            score = quality_scorer.score_content_quality(content, keywords)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Verify valid score
            assert isinstance(score, float)
            assert 0 <= score <= 1
            
            # Performance should scale reasonably with content size
            content_length = len(content)
            time_per_char = execution_time / content_length
            
            assert time_per_char < 0.001, f"Scoring too slow for content size {content_length}"
            
            print(f"Content {i+1} ({content_length} chars): {execution_time:.4f}s ({time_per_char:.6f}s/char)")
    
    def test_duplicate_detection_performance(self, quality_scorer):
        """Test duplicate content detection performance."""
        # Create content with some duplicates
        base_content = "This is a sample content piece for testing duplicate detection. "
        contents = []
        
        # Add original content
        for i in range(10):
            contents.append(base_content + f"Unique part {i}")
        
        # Add some duplicates
        for i in range(5):
            contents.append(contents[i])  # Exact duplicates
        
        # Add near-duplicates
        for i in range(3):
            contents.append(base_content + f"Unique part {i}" + " with slight modification")
        
        start_time = time.perf_counter()
        
        # Filter duplicates
        filtered_content = quality_scorer.filter_duplicates(contents)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should remove some duplicates
        assert len(filtered_content) < len(contents)
        assert len(filtered_content) >= 10  # At least original content
        
        # Performance should be reasonable
        assert execution_time < 2.0, f"Duplicate detection too slow: {execution_time:.3f}s"
        
        print(f"Duplicate Detection Performance:")
        print(f"  Input items: {len(contents)}")
        print(f"  Output items: {len(filtered_content)}")
        print(f"  Execution time: {execution_time:.3f}s")
    
    def test_semantic_similarity_performance(self, quality_scorer):
        """Test semantic similarity calculation performance."""
        # Test texts for similarity
        text1 = "Machine learning algorithms are transforming data analysis."
        text2 = "AI techniques are revolutionizing data processing."
        text3 = "The weather is nice today."
        
        texts = [text1, text2, text3]
        
        start_time = time.perf_counter()
        
        # Calculate all pairwise similarities
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = quality_scorer.calculate_semantic_similarity(texts[i], texts[j])
                assert isinstance(similarity, float)
                assert 0 <= similarity <= 1
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Should be reasonably fast
        assert execution_time < 1.0, f"Semantic similarity too slow: {execution_time:.3f}s"
        
        print(f"Semantic Similarity Performance: {execution_time:.3f}s for {len(texts)} texts")


class TestExtractionCoordinatorPerformance(PerformanceTestBase):
    """Performance tests for ExtractionCoordinator."""
    
    @pytest.fixture
    def coordinator(self):
        """Setup extraction coordinator with mocked dependencies."""
        # Reset service registry
        ServiceRegistry._instance = None
        registry = ServiceRegistry()
        
        # Mock services
        mock_config = {
            'redis': {'enabled': False},
            'ai_services': {'openai_api_key': 'test'},
            'semantic_search': {'enabled': False}
        }
        
        mock_ai_service = Mock()
        mock_ai_service.generate_schema = AsyncMock(return_value={'type': 'object'})
        
        registry.register('config', mock_config)
        registry.register('ai_service', mock_ai_service)
        
        return ExtractionCoordinator()
    
    @pytest.mark.asyncio
    async def test_analysis_and_planning_performance(self, coordinator):
        """Test analysis and planning phase performance."""
        queries = [
            "Find restaurants in Seattle",
            "Get news about technology", 
            "Search for jobs in AI",
            "Compare laptop prices",
            "Book hotels in Paris"
        ]
        
        total_time = 0
        
        for query in queries:
            start_time = time.perf_counter()
            
            plan = await coordinator.analyze_and_plan(query, max_urls=3)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            total_time += execution_time
            
            # Verify plan is valid
            assert plan is not None
            assert hasattr(plan, 'query')
            assert hasattr(plan, 'target_urls')
        
        avg_time = total_time / len(queries)
        
        # Performance assertions
        assert avg_time < 1.0, f"Average planning time {avg_time:.3f}s exceeds 1s"
        
        print(f"Planning Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_extraction_coordination_scalability(self, coordinator):
        """Test extraction coordination with varying numbers of URLs."""
        url_counts = [1, 5, 10, 20]
        
        for url_count in url_counts:
            # Create plan with specified number of URLs
            urls = [f"https://example.com/page{i}" for i in range(url_count)]
            plan = UniversalExtractionPlan(
                query=f"Test with {url_count} URLs",
                target_urls=urls,
                primary_strategy="universal_crawl4ai",
                fallback_strategies=["dom"],
                intent_context={'intent_type': 'scalability_test'}
            )
            
            # Mock extraction results
            def mock_extract(url, plan, context):
                return {
                    'success': True,
                    'data': [{'url': url, 'content': f'Content from {url}'}],
                    'metadata': {'extraction_time': 0.1}
                }
            
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract):
                start_time = time.perf_counter()
                
                results = await coordinator.coordinate_extraction(plan)
                
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Verify results
                assert results['success'] is True
                assert len(results['aggregated_data']) == url_count
                
                # Time should scale reasonably
                time_per_url = execution_time / url_count
                assert time_per_url < 0.5, f"Time per URL too high: {time_per_url:.3f}s"
                
                print(f"Scalability Test - {url_count} URLs:")
                print(f"  Total time: {execution_time:.3f}s")
                print(f"  Time per URL: {time_per_url:.3f}s")
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_large_extraction(self, coordinator):
        """Test memory usage during large-scale extraction."""
        # Create plan with many URLs
        urls = [f"https://example.com/page{i}" for i in range(50)]
        plan = UniversalExtractionPlan(
            query="Large scale extraction test",
            target_urls=urls,
            primary_strategy="universal_crawl4ai",
            fallback_strategies=[],
            intent_context={'intent_type': 'memory_test'}
        )
        
        memory_before = self.get_memory_usage()
        
        # Mock extraction with moderate data size
        def mock_extract(url, plan, context):
            return {
                'success': True,
                'data': [{'url': url, 'content': 'X' * 1000}],  # 1KB per item
                'metadata': {'extraction_time': 0.05}
            }
        
        with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract):
            results = await coordinator.coordinate_extraction(plan)
            
            memory_after = self.get_memory_usage()
            memory_increase = memory_after - memory_before
            
            # Verify extraction worked
            assert results['success'] is True
            assert len(results['aggregated_data']) == 50
            
            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f}MB"
            
            print(f"Large Extraction Memory Usage:")
            print(f"  URLs processed: {len(urls)}")
            print(f"  Memory before: {memory_before:.1f}MB")
            print(f"  Memory after: {memory_after:.1f}MB")
            print(f"  Memory increase: {memory_increase:.1f}MB")


class TestCompositeStrategyPerformance(PerformanceTestBase):
    """Performance tests for CompositeUniversalStrategy."""
    
    @pytest.fixture
    def composite_strategy(self):
        """Setup composite strategy with mocked dependencies."""
        return CompositeUniversalStrategy()
    
    @pytest.mark.asyncio
    async def test_strategy_selection_performance(self, composite_strategy):
        """Test strategy selection performance."""
        # Test different types of plans
        plans = [
            UniversalExtractionPlan(
                query="Simple extraction",
                target_urls=["https://example.com"],
                primary_strategy="universal_crawl4ai",
                fallback_strategies=["dom"],
                intent_context={'intent_type': 'simple_extraction'}
            ),
            UniversalExtractionPlan(
                query="Complex extraction with AI",
                target_urls=["https://complex-site.com"],
                primary_strategy="ai_guided",
                fallback_strategies=["universal_crawl4ai", "dom"],
                intent_context={'intent_type': 'complex_extraction', 'complexity': 'high'}
            ),
            UniversalExtractionPlan(
                query="Product data extraction",
                target_urls=["https://ecommerce.com"],
                primary_strategy="universal_crawl4ai",
                fallback_strategies=["dom"],
                intent_context={'intent_type': 'product_extraction', 'entities': ['products']}
            )
        ]
        
        mock_context = Mock()
        mock_context.get_session.return_value = Mock()
        
        total_time = 0
        
        for plan in plans:
            with patch.object(composite_strategy, '_execute_strategy') as mock_execute:
                mock_execute.return_value = {
                    'success': True,
                    'data': [{'test': 'data'}],
                    'metadata': {'strategy_used': plan.primary_strategy}
                }
                
                start_time = time.perf_counter()
                
                result = await composite_strategy.extract_data(
                    plan.target_urls[0], plan, mock_context
                )
                
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                total_time += execution_time
                
                # Verify result
                assert result is not None
                assert result['success'] is True
        
        avg_time = total_time / len(plans)
        
        # Strategy selection should be fast
        assert avg_time < 0.1, f"Strategy selection too slow: {avg_time:.3f}s"
        
        print(f"Strategy Selection Performance:")
        print(f"  Average: {avg_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_performance(self, composite_strategy):
        """Test fallback strategy coordination performance."""
        plan = UniversalExtractionPlan(
            query="Fallback test",
            target_urls=["https://example.com"],
            primary_strategy="universal_crawl4ai",
            fallback_strategies=["dom", "ai_guided"],
            intent_context={'intent_type': 'fallback_test'}
        )
        
        mock_context = Mock()
        mock_context.get_session.return_value = Mock()
        
        # Mock primary strategy failure, first fallback failure, second success
        call_count = 0
        def mock_execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:  # Primary and first fallback fail
                return {
                    'success': False,
                    'error': f'Strategy {call_count} failed',
                    'metadata': {'strategy_used': f'strategy_{call_count}'}
                }
            else:  # Second fallback succeeds
                return {
                    'success': True,
                    'data': [{'test': 'fallback_success'}],
                    'metadata': {'strategy_used': 'ai_guided'}
                }
        
        with patch.object(composite_strategy, '_execute_strategy', side_effect=mock_execute_side_effect):
            start_time = time.perf_counter()
            
            result = await composite_strategy.extract_data(
                plan.target_urls[0], plan, mock_context
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Should eventually succeed with fallback
            assert result is not None
            assert result['success'] is True
            
            # Fallback coordination should be reasonably fast
            assert execution_time < 1.0, f"Fallback coordination too slow: {execution_time:.3f}s"
            
            print(f"Fallback Performance: {execution_time:.3f}s for 3 strategy attempts")


class TestStressTests(PerformanceTestBase):
    """Stress tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """Test system under high concurrency load."""
        # Setup coordinator
        coordinator = ExtractionCoordinator()
        
        # Mock services to avoid external dependencies
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis.get.return_value = None
            mock_redis.setex.return_value = True
            mock_redis_class.return_value = mock_redis
            
            # Create many concurrent extraction requests
            queries = [f"Concurrent query {i}" for i in range(50)]
            
            start_time = time.perf_counter()
            
            # Mock extraction to return quickly
            def mock_extract(url, plan, context):
                return {
                    'success': True,
                    'data': [{'query': plan.query, 'url': url}],
                    'metadata': {'extraction_time': 0.01}
                }
            
            with patch.object(coordinator.composite_strategy, 'extract_data', side_effect=mock_extract):
                # Run planning phase concurrently
                planning_tasks = [
                    coordinator.analyze_and_plan(query, max_urls=2) 
                    for query in queries
                ]
                plans = await asyncio.gather(*planning_tasks)
                
                # Run extraction phase concurrently
                extraction_tasks = [
                    coordinator.coordinate_extraction(plan) 
                    for plan in plans if plan is not None
                ]
                results = await asyncio.gather(*extraction_tasks)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Verify all extractions completed
            successful_results = [r for r in results if r and r.get('success')]
            success_rate = len(successful_results) / len(queries)
            
            assert success_rate > 0.9, f"Success rate too low: {success_rate:.2%}"
            assert total_time < 10.0, f"High concurrency test too slow: {total_time:.3f}s"
            
            print(f"High Concurrency Stress Test:")
            print(f"  Queries: {len(queries)}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg time per query: {total_time/len(queries):.3f}s")
    
    def test_memory_stress_large_content(self):
        """Test memory handling with very large content."""
        quality_scorer = ContentQualityScorer()
        
        # Generate very large content
        large_content = "Large content section with detailed information. " * 10000  # ~500KB
        keywords = ['content', 'information', 'detailed', 'section']
        
        memory_before = self.get_memory_usage()
        
        # Process large content multiple times
        for i in range(10):
            score = quality_scorer.score_content_quality(large_content, keywords)
            assert isinstance(score, float)
            assert 0 <= score <= 1
            
            # Force garbage collection periodically
            if i % 3 == 0:
                gc.collect()
        
        memory_after = self.get_memory_usage()
        memory_increase = memory_after - memory_before
        
        # Memory increase should be controlled
        assert memory_increase < 200, f"Memory increase too high: {memory_increase:.1f}MB"
        
        print(f"Memory Stress Test:")
        print(f"  Content size: ~{len(large_content) / 1024:.1f}KB")
        print(f"  Iterations: 10")
        print(f"  Memory increase: {memory_increase:.1f}MB")


if __name__ == "__main__":
    # Run with performance-specific options
    pytest.main([
        __file__, 
        "-v", 
        "-s",  # Don't capture output so we can see performance metrics
        "--tb=short"
    ])
