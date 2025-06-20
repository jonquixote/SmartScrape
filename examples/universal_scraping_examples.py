#!/usr/bin/env python3
"""
Universal Scraping Examples for SmartScrape

This module demonstrates end-to-end capabilities of SmartScrape including
semantic intent analysis, AI schema generation, resilience features,
caching strategies, user feedback integration, and progressive collection.

Examples cover real-world scenarios and best practices for using all
the advanced features implemented in Phases 1-7.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from controllers.adaptive_scraper import AdaptiveScraper
from core.service_registry import ServiceRegistry
from core.configuration import Configuration
from components.semantic_intent.universal_intent_analyzer import UniversalIntentAnalyzer
from components.ai_schema.ai_schema_generator import AISchemaGenerator
from components.caching.cache_manager import CacheManager
from components.feedback.feedback_collector import FeedbackCollector
from components.progressive.collection_coordinator import CollectionCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalScrapingExamples:
    """
    Comprehensive examples showcasing SmartScrape's advanced capabilities
    """
    
    def __init__(self):
        """Initialize the example system with all advanced features"""
        self.config = Configuration()
        self.service_registry = ServiceRegistry()
        self.adaptive_scraper = AdaptiveScraper(self.service_registry)
        
        # Initialize advanced components
        self.intent_analyzer = UniversalIntentAnalyzer(self.config)
        self.schema_generator = AISchemaGenerator(self.config)
        self.cache_manager = CacheManager(self.config)
        self.feedback_collector = FeedbackCollector(self.config)
        self.collection_coordinator = CollectionCoordinator(self.config)
    
    async def example_1_ecommerce_product_research(self):
        """
        Example 1: E-commerce Product Research
        
        Demonstrates:
        - Semantic intent analysis for product queries
        - Dynamic schema generation for product data
        - Progressive collection across multiple pages
        - Intelligent caching of product information
        """
        logger.info("=== Example 1: E-commerce Product Research ===")
        
        # User query with natural language
        user_query = "Find the best laptops under $1000 with good reviews"
        
        # Step 1: Semantic Intent Analysis
        logger.info("Analyzing user intent...")
        intent_result = await self.intent_analyzer.analyze_intent(user_query)
        
        print(f"Detected Intent: {intent_result.intent_category}")
        print(f"Expanded Terms: {intent_result.expanded_terms}")
        print(f"Entities: {intent_result.entities}")
        
        # Step 2: Generate dynamic schema for product data
        logger.info("Generating product schema...")
        product_schema = await self.schema_generator.generate_schema(
            domain="e_commerce",
            content_type="product",
            sample_data=None,
            intent=intent_result
        )
        
        print(f"Generated Schema Fields: {list(product_schema.model_fields.keys())}")
        
        # Step 3: Configure progressive collection
        collection_config = {
            "max_pages": 20,
            "batch_size": 5,
            "enable_deduplication": True,
            "quality_threshold": 0.8
        }
        
        # Step 4: Execute progressive collection
        logger.info("Starting progressive collection...")
        results = await self.collection_coordinator.collect_progressively(
            query=user_query,
            intent=intent_result,
            schema=product_schema,
            config=collection_config
        )
        
        # Step 5: Display results with quality metrics
        logger.info(f"Collected {len(results.items)} products")
        logger.info(f"Average quality score: {results.average_quality:.2f}")
        logger.info(f"Deduplication removed: {results.duplicates_removed} items")
        
        # Step 6: Cache results for future use
        cache_key = f"products:laptops:under_1000:{datetime.now().strftime('%Y%m%d')}"
        await self.cache_manager.set(cache_key, results.to_dict(), ttl=3600)
        
        return results
    
    async def example_2_news_sentiment_analysis(self):
        """
        Example 2: News Sentiment Analysis
        
        Demonstrates:
        - Intent analysis for news queries
        - Cross-domain schema adaptation
        - Real-time feedback integration
        - Consolidated AI processing for sentiment
        """
        logger.info("=== Example 2: News Sentiment Analysis ===")
        
        user_query = "Latest news about AI and machine learning breakthroughs"
        
        # Step 1: Analyze intent with news focus
        intent_result = await self.intent_analyzer.analyze_intent(
            query=user_query,
            domain_hint="news"
        )
        
        # Step 2: Generate news-specific schema
        news_schema = await self.schema_generator.generate_schema(
            domain="news",
            content_type="article",
            additional_fields=["sentiment_score", "credibility_rating"]
        )
        
        # Step 3: Execute search with resilience features
        search_config = {
            "enable_proxy_rotation": True,
            "anti_detection": True,
            "max_retries": 3,
            "circuit_breaker": True
        }
        
        results = await self.adaptive_scraper.scrape(
            query=user_query,
            intent=intent_result,
            schema=news_schema,
            config=search_config
        )
        
        # Step 4: Process with consolidated AI for sentiment analysis
        enhanced_results = await self.collection_coordinator.process_with_ai(
            results,
            tasks=["sentiment_analysis", "credibility_assessment", "topic_extraction"]
        )
        
        # Step 5: Collect user feedback
        feedback_request = {
            "results": enhanced_results,
            "user_query": user_query,
            "feedback_type": "quality_rating"
        }
        
        # Simulate user feedback (in real use, this would come from UI)
        simulated_feedback = {
            "overall_rating": 4,
            "relevance": 5,
            "accuracy": 4,
            "comments": "Good coverage but some sources seem biased"
        }
        
        await self.feedback_collector.collect_feedback(
            feedback_request,
            simulated_feedback
        )
        
        return enhanced_results
    
    async def example_3_research_data_mining(self):
        """
        Example 3: Academic Research Data Mining
        
        Demonstrates:
        - Multi-source data collection
        - Hierarchical schema structures
        - Advanced deduplication
        - Cross-page relationship analysis
        """
        logger.info("=== Example 3: Academic Research Data Mining ===")
        
        research_query = "Recent papers on quantum computing applications"
        
        # Step 1: Multi-domain intent analysis
        intent_result = await self.intent_analyzer.analyze_intent(
            query=research_query,
            enable_multi_domain=True,
            domains=["research", "academic", "technology"]
        )
        
        # Step 2: Generate hierarchical schema for research papers
        research_schema = await self.schema_generator.generate_hierarchical_schema(
            base_domain="research",
            hierarchical_levels={
                "paper": ["title", "authors", "abstract", "publication_date"],
                "author": ["name", "affiliation", "email"],
                "citation": ["cited_paper", "citation_count", "context"]
            }
        )
        
        # Step 3: Configure multi-source collection
        sources = [
            "arxiv.org",
            "scholar.google.com",
            "ieee.org",
            "acm.org"
        ]
        
        # Step 4: Execute progressive collection across sources
        all_results = []
        for source in sources:
            logger.info(f"Collecting from {source}...")
            
            source_results = await self.collection_coordinator.collect_from_source(
                source=source,
                query=research_query,
                intent=intent_result,
                schema=research_schema
            )
            
            all_results.extend(source_results.items)
        
        # Step 5: Advanced deduplication and relationship analysis
        logger.info("Performing advanced deduplication...")
        deduplicated_results = await self.collection_coordinator.deduplicate_advanced(
            all_results,
            strategies=["exact_match", "fuzzy_match", "semantic_similarity"],
            cross_reference_analysis=True
        )
        
        # Step 6: Analyze relationships between papers
        relationships = await self.collection_coordinator.analyze_relationships(
            deduplicated_results,
            relationship_types=["citation", "author_collaboration", "topic_similarity"]
        )
        
        logger.info(f"Found {len(relationships)} relationships between papers")
        
        return {
            "papers": deduplicated_results,
            "relationships": relationships,
            "collection_stats": {
                "total_sources": len(sources),
                "original_count": len(all_results),
                "deduplicated_count": len(deduplicated_results),
                "deduplication_rate": 1 - (len(deduplicated_results) / len(all_results))
            }
        }
    
    async def example_4_real_estate_market_analysis(self):
        """
        Example 4: Real Estate Market Analysis
        
        Demonstrates:
        - Geographic intent analysis
        - Time-series data collection
        - Intelligent caching strategies
        - Performance optimization
        """
        logger.info("=== Example 4: Real Estate Market Analysis ===")
        
        market_query = "Housing prices in San Francisco Bay Area last 6 months"
        
        # Step 1: Geographic and temporal intent analysis
        intent_result = await self.intent_analyzer.analyze_intent(
            query=market_query,
            enable_geographic_analysis=True,
            enable_temporal_analysis=True
        )
        
        print(f"Geographic Entities: {intent_result.geographic_entities}")
        print(f"Temporal Entities: {intent_result.temporal_entities}")
        
        # Step 2: Generate schema with geographic and temporal fields
        real_estate_schema = await self.schema_generator.generate_schema(
            domain="real_estate",
            content_type="listing",
            geographic_context=intent_result.geographic_entities,
            temporal_context=intent_result.temporal_entities
        )
        
        # Step 3: Check cache for recent data
        cache_key = f"real_estate:{intent_result.geographic_hash}:{intent_result.temporal_hash}"
        cached_results = await self.cache_manager.get(cache_key)
        
        if cached_results:
            logger.info("Using cached real estate data")
            return cached_results
        
        # Step 4: Configure high-performance collection
        performance_config = {
            "parallel_workers": 10,
            "batch_size": 20,
            "enable_compression": True,
            "memory_optimization": True,
            "streaming_mode": True
        }
        
        # Step 5: Execute collection with performance monitoring
        start_time = datetime.now()
        
        results = await self.collection_coordinator.collect_with_monitoring(
            query=market_query,
            intent=intent_result,
            schema=real_estate_schema,
            config=performance_config
        )
        
        end_time = datetime.now()
        collection_time = (end_time - start_time).total_seconds()
        
        # Step 6: Analyze market trends
        trend_analysis = await self.analyze_market_trends(results)
        
        # Step 7: Cache results with intelligent TTL
        intelligent_ttl = self.calculate_intelligent_ttl(
            data_type="real_estate",
            volatility=trend_analysis.get("volatility", 0.5),
            update_frequency=intent_result.temporal_entities.get("frequency", "daily")
        )
        
        final_results = {
            "listings": results.items,
            "trend_analysis": trend_analysis,
            "performance_metrics": {
                "collection_time": collection_time,
                "items_per_second": len(results.items) / collection_time,
                "cache_hit_ratio": results.cache_hit_ratio
            }
        }
        
        await self.cache_manager.set(cache_key, final_results, ttl=intelligent_ttl)
        
        return final_results
    
    async def example_5_social_media_monitoring(self):
        """
        Example 5: Social Media Brand Monitoring
        
        Demonstrates:
        - Multi-platform data collection
        - Real-time sentiment tracking
        - User feedback integration
        - Adaptive personalization
        """
        logger.info("=== Example 5: Social Media Brand Monitoring ===")
        
        brand_query = "Mentions of Tesla on social media platforms"
        
        # Step 1: Social media intent analysis
        intent_result = await self.intent_analyzer.analyze_intent(
            query=brand_query,
            domain_hint="social_media",
            enable_sentiment_preprocessing=True
        )
        
        # Step 2: Multi-platform schema generation
        social_schema = await self.schema_generator.generate_multi_platform_schema(
            platforms=["twitter", "reddit", "facebook", "instagram"],
            content_types=["post", "comment", "mention"],
            sentiment_analysis=True
        )
        
        # Step 3: Configure resilient collection for social platforms
        resilience_config = {
            "rate_limiting": {
                "adaptive": True,
                "platform_specific": True
            },
            "anti_detection": {
                "user_agent_rotation": True,
                "request_timing_randomization": True,
                "proxy_rotation": True
            },
            "error_handling": {
                "platform_specific_retries": True,
                "graceful_degradation": True
            }
        }
        
        # Step 4: Execute multi-platform collection
        platform_results = {}
        platforms = ["twitter", "reddit", "facebook"]
        
        for platform in platforms:
            try:
                logger.info(f"Collecting from {platform}...")
                
                platform_result = await self.collection_coordinator.collect_from_platform(
                    platform=platform,
                    query=brand_query,
                    intent=intent_result,
                    schema=social_schema,
                    config=resilience_config
                )
                
                platform_results[platform] = platform_result
                
            except Exception as e:
                logger.warning(f"Failed to collect from {platform}: {e}")
                # Continue with other platforms due to graceful degradation
        
        # Step 5: Consolidated sentiment analysis
        all_mentions = []
        for platform, result in platform_results.items():
            all_mentions.extend(result.items)
        
        sentiment_analysis = await self.collection_coordinator.analyze_sentiment_batch(
            all_mentions,
            include_confidence=True,
            include_emotion_detection=True
        )
        
        # Step 6: Generate monitoring report
        monitoring_report = {
            "total_mentions": len(all_mentions),
            "platform_breakdown": {
                platform: len(result.items) 
                for platform, result in platform_results.items()
            },
            "sentiment_summary": {
                "positive": sentiment_analysis.positive_count,
                "negative": sentiment_analysis.negative_count,
                "neutral": sentiment_analysis.neutral_count,
                "average_sentiment": sentiment_analysis.average_sentiment
            },
            "trending_topics": sentiment_analysis.trending_topics,
            "collection_timestamp": datetime.now().isoformat()
        }
        
        # Step 7: Collect user feedback for monitoring accuracy
        feedback_data = {
            "monitoring_accuracy": 4.5,
            "relevance_score": 4.2,
            "platform_coverage": 4.8,
            "sentiment_accuracy": 4.0,
            "suggestions": ["Include LinkedIn", "Add emoji sentiment analysis"]
        }
        
        await self.feedback_collector.collect_feedback(
            {"report": monitoring_report, "query": brand_query},
            feedback_data
        )
        
        # Step 8: Update personalization based on feedback
        await self.feedback_collector.update_personalization(
            user_preferences={
                "preferred_platforms": ["twitter", "reddit"],
                "sentiment_detail_level": "high",
                "update_frequency": "hourly"
            }
        )
        
        return monitoring_report
    
    async def analyze_market_trends(self, results) -> Dict[str, Any]:
        """Analyze market trends from real estate data"""
        # Implementation would include statistical analysis
        return {
            "price_trend": "increasing",
            "volatility": 0.3,
            "market_temperature": "hot",
            "prediction_confidence": 0.85
        }
    
    def calculate_intelligent_ttl(self, data_type: str, volatility: float, 
                                update_frequency: str) -> int:
        """Calculate intelligent TTL based on data characteristics"""
        base_ttl = {
            "real_estate": 3600,  # 1 hour
            "news": 1800,        # 30 minutes
            "social_media": 900,  # 15 minutes
            "research": 86400    # 24 hours
        }
        
        frequency_multiplier = {
            "real_time": 0.1,
            "hourly": 0.5,
            "daily": 1.0,
            "weekly": 7.0
        }
        
        volatility_adjustment = 1.0 - volatility  # High volatility = lower TTL
        
        return int(
            base_ttl.get(data_type, 3600) * 
            frequency_multiplier.get(update_frequency, 1.0) * 
            volatility_adjustment
        )
    
    async def example_6_comprehensive_integration(self):
        """
        Example 6: Comprehensive Feature Integration
        
        Demonstrates all features working together in a complex scenario
        """
        logger.info("=== Example 6: Comprehensive Feature Integration ===")
        
        # Complex multi-domain query
        complex_query = """
        Find competitive analysis data for electric vehicle companies,
        including stock prices, news sentiment, social media mentions,
        and recent patent filings
        """
        
        # Step 1: Advanced intent analysis with multi-domain support
        intent_result = await self.intent_analyzer.analyze_intent(
            query=complex_query,
            enable_multi_domain=True,
            domains=["finance", "news", "social_media", "research"],
            enable_cross_domain_relationships=True
        )
        
        # Step 2: Generate unified schema across domains
        unified_schema = await self.schema_generator.generate_unified_schema(
            domains=intent_result.detected_domains,
            relationship_mapping=intent_result.domain_relationships,
            enable_cross_domain_validation=True
        )
        
        # Step 3: Execute coordinated collection across all domains
        collection_plan = await self.collection_coordinator.create_collection_plan(
            query=complex_query,
            intent=intent_result,
            schema=unified_schema,
            optimization_target="comprehensive_coverage"
        )
        
        # Step 4: Execute plan with full feature utilization
        comprehensive_results = await self.collection_coordinator.execute_comprehensive_plan(
            plan=collection_plan,
            features={
                "semantic_analysis": True,
                "progressive_collection": True,
                "intelligent_caching": True,
                "resilience_management": True,
                "quality_assessment": True,
                "feedback_integration": True,
                "performance_optimization": True
            }
        )
        
        # Step 5: Generate comprehensive analysis report
        analysis_report = await self.generate_comprehensive_report(
            comprehensive_results,
            intent_result,
            collection_plan
        )
        
        logger.info("Comprehensive integration example completed successfully")
        return analysis_report
    
    async def generate_comprehensive_report(self, results, intent, plan) -> Dict[str, Any]:
        """Generate a comprehensive analysis report"""
        return {
            "executive_summary": {
                "total_data_points": sum(len(domain_results) for domain_results in results.values()),
                "domains_covered": list(results.keys()),
                "collection_efficiency": plan.efficiency_metrics,
                "overall_quality_score": results.get("quality_metrics", {}).get("average", 0.0)
            },
            "domain_analysis": results,
            "cross_domain_insights": await self.extract_cross_domain_insights(results),
            "recommendations": await self.generate_recommendations(results, intent),
            "metadata": {
                "collection_timestamp": datetime.now().isoformat(),
                "processing_time": plan.total_processing_time,
                "features_used": plan.features_utilized
            }
        }
    
    async def extract_cross_domain_insights(self, results) -> List[Dict[str, Any]]:
        """Extract insights that span multiple domains"""
        # Implementation would analyze relationships between different data types
        return [
            {
                "insight_type": "correlation",
                "domains": ["finance", "social_media"],
                "finding": "Strong correlation between social sentiment and stock price movements"
            }
        ]
    
    async def generate_recommendations(self, results, intent) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        return [
            "Focus on improving social media sentiment through targeted campaigns",
            "Monitor patent filings for competitive intelligence",
            "Increase news monitoring frequency during market volatility"
        ]


async def run_all_examples():
    """Run all universal scraping examples"""
    examples = UniversalScrapingExamples()
    
    print("🚀 Starting SmartScrape Universal Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        await examples.example_1_ecommerce_product_research()
        await examples.example_2_news_sentiment_analysis()
        await examples.example_3_research_data_mining()
        await examples.example_4_real_estate_market_analysis()
        await examples.example_5_social_media_monitoring()
        await examples.example_6_comprehensive_integration()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())
