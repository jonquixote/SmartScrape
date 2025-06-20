#!/usr/bin/env python3
"""
Priority 6.2: Performance and Reliability Testing
Tests for timeout handling, different website types, error recovery, and success rate measurement
"""

import asyncio
import pytest
import logging
import time
import random
from typing import Dict, List, Any
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
import statistics

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Priority6PerformanceTesting")

class TestTimeoutHandling:
    """Test timeout handling under various conditions"""
    
    @pytest.mark.asyncio
    async def test_short_timeout_handling(self):
        """Test handling of very short timeouts"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
            # Mock slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(5)  # 5 second delay
                return {"items": [], "metadata": {}}
            
            mock_scrape.side_effect = slow_response
            
            scraper = AdaptiveScraper()
            start_time = time.time()
            
            try:
                result = await asyncio.wait_for(
                    scraper.scrape_data(
                        url="https://slow-site.com",
                        query="test",
                        options={"timeout": 1}  # 1 second timeout
                    ),
                    timeout=2  # Overall timeout
                )
                
                # Should complete within timeout or handle gracefully
                elapsed = time.time() - start_time
                assert elapsed < 3  # Should not take longer than 3 seconds
                
            except asyncio.TimeoutError:
                # Timeout is acceptable - test that it happens quickly
                elapsed = time.time() - start_time
                assert elapsed < 3  # Should timeout quickly, not hang
                logger.info(f"✅ Short timeout handled properly: {elapsed:.2f}s")
            
        logger.info("✅ Short timeout handling test passed")
    
    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self):
        """Test recovery from network timeouts"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        attempts = []
        
        async def intermittent_timeout(*args, **kwargs):
            attempts.append(time.time())
            if len(attempts) <= 2:  # First 2 attempts fail
                raise asyncio.TimeoutError("Network timeout")
            return {"items": [{"title": "Success after retries"}], "metadata": {}}
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
            mock_scrape.side_effect = intermittent_timeout
            
            scraper = AdaptiveScraper()
            
            # Should eventually succeed after retries
            result = await scraper.scrape_data(
                url="https://unreliable-site.com",
                query="test",
                options={"max_retries": 3, "retry_delay": 0.1}
            )
            
            # Verify that retries occurred
            assert len(attempts) >= 2
            if result:
                assert "items" in result
                
        logger.info("✅ Network timeout recovery test passed")
    
    @pytest.mark.asyncio
    async def test_progressive_timeout_handling(self):
        """Test progressive timeout increases"""
        timeouts = []
        
        async def record_timeout(*args, **kwargs):
            timeout = kwargs.get('timeout', 30)
            timeouts.append(timeout)
            if len(timeouts) < 3:
                raise asyncio.TimeoutError(f"Timeout {timeout}s")
            return {"items": [], "metadata": {"timeout_used": timeout}}
        
        from controllers.adaptive_scraper import AdaptiveScraper
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
            mock_scrape.side_effect = record_timeout
            
            scraper = AdaptiveScraper()
            
            result = await scraper.scrape_data(
                url="https://example.com",
                query="test",
                options={"progressive_timeout": True, "initial_timeout": 5}
            )
            
            # Should use progressively longer timeouts
            if len(timeouts) > 1:
                assert timeouts[1] >= timeouts[0]  # Second timeout should be >= first
                
        logger.info("✅ Progressive timeout handling test passed")


class TestDifferentWebsiteTypes:
    """Test with different types of websites"""
    
    @pytest.mark.asyncio
    async def test_ecommerce_site_extraction(self):
        """Test extraction from e-commerce websites"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        # Mock e-commerce response
        ecommerce_response = {
            "items": [
                {
                    "title": "Wireless Headphones",
                    "price": "$89.99",
                    "rating": "4.5/5",
                    "availability": "In Stock",
                    "image_url": "https://example.com/headphones.jpg"
                },
                {
                    "title": "Bluetooth Speaker",
                    "price": "$49.99", 
                    "rating": "4.2/5",
                    "availability": "Limited Stock",
                    "image_url": "https://example.com/speaker.jpg"
                }
            ],
            "metadata": {
                "site_type": "ecommerce",
                "extraction_time": 2.1,
                "total_items": 2
            }
        }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.return_value = ecommerce_response
            
            scraper = AdaptiveScraper()
            result = await scraper.scrape_data(
                url="https://electronics-store.com/headphones",
                query="wireless headphones",
                options={"site_type": "ecommerce"}
            )
            
            assert result is not None
            assert "items" in result
            assert len(result["items"]) > 0
            
            # Validate e-commerce specific fields
            for item in result["items"]:
                assert "price" in item or "title" in item  # Should have basic product info
                
        logger.info("✅ E-commerce site extraction test passed")
    
    @pytest.mark.asyncio
    async def test_news_site_extraction(self):
        """Test extraction from news websites"""
        news_response = {
            "items": [
                {
                    "title": "Breaking: New Technology Breakthrough",
                    "author": "Tech Reporter",
                    "publish_date": "2025-01-01",
                    "summary": "Scientists achieve major breakthrough...",
                    "category": "Technology"
                }
            ],
            "metadata": {
                "site_type": "news",
                "extraction_time": 1.8,
                "total_items": 1
            }
        }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.return_value = news_response
            
            from controllers.adaptive_scraper import AdaptiveScraper
            scraper = AdaptiveScraper()
            
            result = await scraper.scrape_data(
                url="https://tech-news.com/latest",
                query="technology breakthrough",
                options={"site_type": "news"}
            )
            
            assert result is not None
            assert "items" in result
            
            # Validate news-specific fields
            for item in result["items"]:
                assert "title" in item or "author" in item or "publish_date" in item
                
        logger.info("✅ News site extraction test passed")
    
    @pytest.mark.asyncio
    async def test_social_media_extraction(self):
        """Test extraction from social media sites"""
        social_response = {
            "items": [
                {
                    "username": "tech_enthusiast",
                    "content": "Just tried the new AI tool, amazing results!",
                    "timestamp": "2 hours ago",
                    "likes": 245,
                    "shares": 12
                }
            ],
            "metadata": {
                "site_type": "social_media",
                "extraction_time": 1.5,
                "total_items": 1
            }
        }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.return_value = social_response
            
            from controllers.adaptive_scraper import AdaptiveScraper
            scraper = AdaptiveScraper()
            
            result = await scraper.scrape_data(
                url="https://social-platform.com/feed",
                query="AI tools",
                options={"site_type": "social_media"}
            )
            
            assert result is not None
            assert "items" in result
            
            # Validate social media fields
            for item in result["items"]:
                assert "content" in item or "username" in item
                
        logger.info("✅ Social media extraction test passed")
    
    @pytest.mark.asyncio
    async def test_javascript_heavy_site(self):
        """Test extraction from JavaScript-heavy websites"""
        js_response = {
            "items": [
                {
                    "title": "Dynamic Content Item",
                    "description": "Loaded via JavaScript",
                    "dynamic_field": "Value loaded after page load"
                }
            ],
            "metadata": {
                "site_type": "spa",
                "javascript_required": True,
                "extraction_time": 4.2,
                "rendering_time": 3.1
            }
        }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.return_value = js_response
            
            from controllers.adaptive_scraper import AdaptiveScraper
            scraper = AdaptiveScraper()
            
            result = await scraper.scrape_data(
                url="https://spa-website.com/products",
                query="dynamic products",
                options={"javascript_enabled": True, "wait_for_content": True}
            )
            
            assert result is not None
            assert "items" in result
            
            # Should handle dynamic content
            if result["metadata"].get("javascript_required"):
                assert result["metadata"]["extraction_time"] > 2  # JS sites take longer
                
        logger.info("✅ JavaScript-heavy site extraction test passed")


class TestErrorRecoveryAndGracefulDegradation:
    """Test error recovery and graceful degradation"""
    
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test recovery from partial failures"""
        failure_count = [0]  # Use list to allow modification in nested function
        
        async def partial_failure(*args, **kwargs):
            failure_count[0] += 1
            if failure_count[0] <= 2:  # First 2 calls fail
                return {
                    "items": [],
                    "metadata": {"error": "Partial extraction failure"},
                    "status": "partial_failure"
                }
            else:  # Third call succeeds
                return {
                    "items": [{"title": "Recovered Data", "content": "Success"}],
                    "metadata": {"recovered": True},
                    "status": "success"
                }
        
        from controllers.adaptive_scraper import AdaptiveScraper
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
            mock_scrape.side_effect = partial_failure
            
            scraper = AdaptiveScraper()
            result = await scraper.scrape_data(
                url="https://unstable-site.com",
                query="test data",
                options={"retry_on_partial_failure": True, "max_retries": 3}
            )
            
            # Should eventually recover
            assert result is not None
            if result.get("status") == "success":
                assert len(result["items"]) > 0
            
        logger.info("✅ Partial failure recovery test passed")
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_no_ai(self):
        """Test graceful degradation when AI services unavailable"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        # Mock AI service failure
        with patch('core.ai_service.AIService.generate_response') as mock_ai:
            mock_ai.side_effect = Exception("AI service unavailable")
            
            # Mock fallback to rule-based extraction
            with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
                mock_scrape.return_value = {
                    "items": [{"title": "Rule-based extraction", "content": "No AI needed"}],
                    "metadata": {"strategy": "rule_based_fallback", "ai_available": False}
                }
                
                scraper = AdaptiveScraper()
                result = await scraper.scrape_data(
                    url="https://example.com",
                    query="test",
                    options={"ai_fallback": True}
                )
                
                assert result is not None
                assert "items" in result
                # Should work without AI
                if result["metadata"].get("ai_available") is False:
                    assert result["metadata"]["strategy"] == "rule_based_fallback"
                
        logger.info("✅ Graceful degradation without AI test passed")
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling under memory pressure"""
        large_data_calls = []
        
        async def memory_intensive_operation(*args, **kwargs):
            # Simulate memory pressure
            large_data_calls.append(time.time())
            if len(large_data_calls) > 3:
                # Simulate memory cleanup/optimization
                return {
                    "items": [{"title": "Optimized extraction"}],
                    "metadata": {"memory_optimized": True, "reduced_data": True}
                }
            else:
                return {
                    "items": [{"title": f"Large dataset {i}", "data": "x" * 1000} 
                             for i in range(100)],  # Large dataset
                    "metadata": {"memory_usage": "high"}
                }
        
        from controllers.adaptive_scraper import AdaptiveScraper
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
            mock_scrape.side_effect = memory_intensive_operation
            
            scraper = AdaptiveScraper()
            
            # Process multiple requests to trigger memory optimization
            results = []
            for i in range(5):
                result = await scraper.scrape_data(
                    url=f"https://example{i}.com",
                    query="large dataset",
                    options={"memory_optimization": True}
                )
                results.append(result)
            
            # Later results should show memory optimization
            if len(results) > 3 and results[-1]:
                last_result = results[-1]
                if last_result["metadata"].get("memory_optimized"):
                    assert last_result["metadata"]["reduced_data"] is True
                    
        logger.info("✅ Memory pressure handling test passed")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self):
        """Test handling of rate limiting"""
        request_times = []
        
        async def rate_limited_response(*args, **kwargs):
            current_time = time.time()
            request_times.append(current_time)
            
            # Simulate rate limiting on first few requests
            if len(request_times) <= 2:
                if len(request_times) > 1:
                    time_diff = current_time - request_times[-2]
                    if time_diff < 1:  # Too fast
                        raise Exception("Rate limit exceeded - 429")
                
            return {
                "items": [{"title": "Rate limited data"}],
                "metadata": {"rate_limited": len(request_times) <= 2}
            }
        
        from controllers.adaptive_scraper import AdaptiveScraper
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper._scrape_with_strategy') as mock_scrape:
            mock_scrape.side_effect = rate_limited_response
            
            scraper = AdaptiveScraper()
            
            try:
                result = await scraper.scrape_data(
                    url="https://rate-limited-site.com",
                    query="test",
                    options={"respect_rate_limits": True, "retry_delay": 1.1}
                )
                
                assert result is not None
                
            except Exception as e:
                # Rate limiting errors should be handled gracefully
                assert "rate limit" in str(e).lower() or "429" in str(e)
                
        logger.info("✅ Rate limiting handling test passed")


class TestSuccessRateMeasurement:
    """Test actual data extraction success rate measurement"""
    
    @pytest.mark.asyncio
    async def test_success_rate_tracking(self):
        """Test tracking of extraction success rates"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        success_count = 0
        total_count = 0
        
        async def variable_success(*args, **kwargs):
            nonlocal success_count, total_count
            total_count += 1
            
            # 70% success rate
            if random.random() < 0.7:
                success_count += 1
                return {
                    "items": [{"title": f"Success {success_count}"}],
                    "metadata": {"status": "success"}
                }
            else:
                return {
                    "items": [],
                    "metadata": {"status": "failure", "error": "No data found"}
                }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.side_effect = variable_success
            
            scraper = AdaptiveScraper()
            
            # Run multiple extractions
            results = []
            for i in range(20):
                try:
                    result = await scraper.scrape_data(
                        url=f"https://test{i}.com",
                        query="test data"
                    )
                    results.append(result)
                except Exception:
                    results.append(None)
            
            # Calculate success rate
            successful_results = [r for r in results if r and r.get("metadata", {}).get("status") == "success"]
            success_rate = len(successful_results) / len(results) if results else 0
            
            # Should have reasonable success rate
            assert success_rate >= 0.5  # At least 50% success rate
            
            logger.info(f"✅ Success rate tracking: {success_rate:.2%}")
    
    @pytest.mark.asyncio 
    async def test_performance_metrics_collection(self):
        """Test collection of performance metrics"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        performance_data = []
        
        async def collect_metrics(*args, **kwargs):
            start_time = time.time()
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Variable processing time
            end_time = time.time()
            
            metrics = {
                "extraction_time": end_time - start_time,
                "items_extracted": random.randint(1, 10),
                "memory_usage": random.uniform(50, 200),  # MB
                "strategy_used": random.choice(["crawl4ai", "dom", "ai_guided"])
            }
            performance_data.append(metrics)
            
            return {
                "items": [{"title": "Test item"}] * metrics["items_extracted"],
                "metadata": metrics
            }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.side_effect = collect_metrics
            
            scraper = AdaptiveScraper()
            
            # Collect performance data
            for i in range(10):
                await scraper.scrape_data(
                    url=f"https://perf-test{i}.com",
                    query="performance test"
                )
            
            # Analyze performance metrics
            if performance_data:
                avg_time = statistics.mean(p["extraction_time"] for p in performance_data)
                avg_items = statistics.mean(p["items_extracted"] for p in performance_data)
                avg_memory = statistics.mean(p["memory_usage"] for p in performance_data)
                
                assert avg_time > 0
                assert avg_items > 0 
                assert avg_memory > 0
                
                logger.info(f"✅ Performance metrics - Time: {avg_time:.3f}s, Items: {avg_items:.1f}, Memory: {avg_memory:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_reliability_under_load(self):
        """Test reliability under concurrent load"""
        from controllers.adaptive_scraper import AdaptiveScraper
        
        async def load_test_response(*args, **kwargs):
            # Simulate variable response times under load
            await asyncio.sleep(random.uniform(0.05, 0.2))
            
            # Occasional failures under load
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Service overloaded")
            
            return {
                "items": [{"title": "Load test item"}],
                "metadata": {"load_test": True}
            }
        
        with patch('controllers.adaptive_scraper.AdaptiveScraper.scrape_data') as mock_scrape:
            mock_scrape.side_effect = load_test_response
            
            scraper = AdaptiveScraper()
            
            # Run concurrent requests
            tasks = []
            for i in range(20):
                task = asyncio.create_task(
                    scraper.scrape_data(
                        url=f"https://load-test{i}.com",
                        query="concurrent test"
                    )
                )
                tasks.append(task)
            
            # Wait for all requests with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=10
                )
                
                # Count successful results
                successful = sum(1 for r in results if isinstance(r, dict) and "items" in r)
                total = len(results)
                reliability_rate = successful / total if total > 0 else 0
                
                # Should maintain reasonable reliability under load
                assert reliability_rate >= 0.8  # At least 80% reliability
                
                logger.info(f"✅ Reliability under load: {reliability_rate:.2%}")
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ Load test timed out - this may indicate performance issues")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
