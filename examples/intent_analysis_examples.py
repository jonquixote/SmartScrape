#!/usr/bin/env python3
"""
Intent Analysis Examples for SmartScrape

This module demonstrates the semantic intent analysis capabilities
including query expansion, entity recognition, semantic search,
and contextual understanding features.

Examples show how the UniversalIntentAnalyzer processes natural language
queries and generates actionable insights for web scraping.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from components.semantic_intent.universal_intent_analyzer import UniversalIntentAnalyzer
from components.semantic_intent.spacy_integration import SpacyNLPProcessor
from components.semantic_intent.semantic_search import SemanticSearchEngine
from core.configuration import Configuration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentAnalysisExamples:
    """
    Examples demonstrating semantic intent analysis capabilities
    """
    
    def __init__(self):
        """Initialize the intent analysis system"""
        self.config = Configuration()
        self.intent_analyzer = UniversalIntentAnalyzer(self.config)
        self.nlp_processor = SpacyNLPProcessor(self.config)
        self.semantic_search = SemanticSearchEngine(self.config)
    
    async def example_1_basic_intent_analysis(self):
        """
        Example 1: Basic Intent Analysis
        
        Demonstrates fundamental intent analysis capabilities:
        - Intent category classification
        - Entity extraction
        - Query expansion
        - Confidence scoring
        """
        logger.info("=== Example 1: Basic Intent Analysis ===")
        
        # Test queries with different intents
        test_queries = [
            "Find the best smartphones under $500",
            "Latest news about artificial intelligence",
            "Research papers on climate change",
            "Tesla stock price today",
            "Restaurants near me with good reviews",
            "How to install Python on Windows"
        ]
        
        results = []
        
        for query in test_queries:
            print(f"\n🔍 Analyzing: '{query}'")
            
            # Analyze intent
            intent_result = await self.intent_analyzer.analyze_intent(query)
            
            # Display results
            print(f"  📂 Category: {intent_result.intent_category}")
            print(f"  🎯 Confidence: {intent_result.confidence:.2f}")
            print(f"  📝 Entities: {intent_result.entities}")
            print(f"  🔄 Expanded Terms: {intent_result.expanded_terms[:3]}...")  # Show first 3
            print(f"  🏷️  Keywords: {intent_result.keywords}")
            
            results.append({
                "query": query,
                "intent": intent_result.intent_category,
                "confidence": intent_result.confidence,
                "entities": intent_result.entities,
                "expanded_terms": intent_result.expanded_terms
            })
        
        return results
    
    async def example_2_entity_recognition_deep_dive(self):
        """
        Example 2: Entity Recognition Deep Dive
        
        Demonstrates advanced entity recognition:
        - Named entity recognition (NER)
        - Custom entity types
        - Entity relationships
        - Geographic and temporal entities
        """
        logger.info("=== Example 2: Entity Recognition Deep Dive ===")
        
        # Complex queries with various entity types
        complex_queries = [
            "Find Apple iPhone 14 Pro prices in New York and Los Angeles",
            "News about Elon Musk's Tesla announcement yesterday",
            "Research on COVID-19 vaccines published in Nature journal",
            "Amazon stock performance from January to March 2024",
            "Hotels in Paris near Eiffel Tower under 200 euros per night"
        ]
        
        for query in complex_queries:
            print(f"\n🔍 Deep Analysis: '{query}'")
            
            # Perform detailed entity analysis
            entity_analysis = await self.nlp_processor.extract_entities_detailed(query)
            
            print("  📍 Entities by Type:")
            for entity_type, entities in entity_analysis.entities_by_type.items():
                print(f"    {entity_type}: {entities}")
            
            print("  🌍 Geographic Entities:")
            for geo_entity in entity_analysis.geographic_entities:
                print(f"    {geo_entity.text}: {geo_entity.label_} ({geo_entity.geo_type})")
            
            print("  ⏰ Temporal Entities:")
            for temp_entity in entity_analysis.temporal_entities:
                print(f"    {temp_entity.text}: {temp_entity.normalized_date}")
            
            print("  💰 Monetary Entities:")
            for money_entity in entity_analysis.monetary_entities:
                print(f"    {money_entity.text}: {money_entity.amount} {money_entity.currency}")
            
            print("  🔗 Entity Relationships:")
            for relationship in entity_analysis.entity_relationships:
                print(f"    {relationship.subject} --{relationship.relation}--> {relationship.object}")
    
    async def example_3_query_expansion_strategies(self):
        """
        Example 3: Query Expansion Strategies
        
        Demonstrates different query expansion approaches:
        - Semantic expansion using word embeddings
        - Synonym-based expansion
        - Context-aware expansion
        - Domain-specific expansion
        """
        logger.info("=== Example 3: Query Expansion Strategies ===")
        
        # Base queries for expansion
        base_queries = [
            {
                "query": "cheap laptops",
                "domain": "e_commerce",
                "context": "budget shopping"
            },
            {
                "query": "breaking news",
                "domain": "news",
                "context": "current events"
            },
            {
                "query": "machine learning",
                "domain": "research",
                "context": "academic papers"
            }
        ]
        
        for query_info in base_queries:
            query = query_info["query"]
            domain = query_info["domain"]
            context = query_info["context"]
            
            print(f"\n🔍 Expanding: '{query}' (Domain: {domain})")
            
            # Semantic expansion
            semantic_expansion = await self.intent_analyzer.expand_query_semantic(
                query, max_terms=5
            )
            print(f"  🧠 Semantic: {semantic_expansion}")
            
            # Synonym expansion
            synonym_expansion = await self.intent_analyzer.expand_query_synonyms(
                query, max_synonyms=3
            )
            print(f"  📖 Synonyms: {synonym_expansion}")
            
            # Context-aware expansion
            context_expansion = await self.intent_analyzer.expand_query_contextual(
                query, context=context, max_terms=4
            )
            print(f"  🎯 Contextual: {context_expansion}")
            
            # Domain-specific expansion
            domain_expansion = await self.intent_analyzer.expand_query_domain_specific(
                query, domain=domain, max_terms=5
            )
            print(f"  🏷️  Domain-specific: {domain_expansion}")
            
            # Combined expansion strategy
            combined_expansion = await self.intent_analyzer.expand_query_combined(
                query,
                domain=domain,
                context=context,
                strategies=["semantic", "synonym", "contextual", "domain_specific"]
            )
            print(f"  🌟 Combined: {combined_expansion[:8]}...")  # Show first 8 terms
    
    async def example_4_semantic_search_integration(self):
        """
        Example 4: Semantic Search Integration
        
        Demonstrates how intent analysis integrates with semantic search:
        - Semantic similarity matching
        - Vector-based search
        - Contextual ranking
        - Multi-modal search
        """
        logger.info("=== Example 4: Semantic Search Integration ===")
        
        # Sample content database for semantic search
        sample_content = [
            {
                "url": "https://example-tech.com/reviews/iphone",
                "title": "iPhone 14 Pro Review: Camera and Performance",
                "content": "The latest iPhone offers exceptional camera quality and fast performance...",
                "domain": "technology"
            },
            {
                "url": "https://example-news.com/ai-breakthrough",
                "title": "Major AI Breakthrough in Language Understanding",
                "content": "Researchers announce significant progress in natural language processing...",
                "domain": "news"
            },
            {
                "url": "https://example-shop.com/laptops/gaming",
                "title": "Best Gaming Laptops Under $1000",
                "content": "Discover powerful gaming laptops that won't break the bank...",
                "domain": "e_commerce"
            },
            {
                "url": "https://example-research.com/climate-study",
                "title": "Climate Change Impact on Ocean Temperatures",
                "content": "New research reveals accelerating ocean warming trends...",
                "domain": "research"
            }
        ]
        
        # Initialize semantic search with sample content
        await self.semantic_search.index_content(sample_content)
        
        # Test queries for semantic search
        search_queries = [
            "smartphone camera quality reviews",
            "artificial intelligence language processing news",
            "affordable gaming computers",
            "environmental climate research studies"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Semantic Search: '{query}'")
            
            # Analyze intent first
            intent_result = await self.intent_analyzer.analyze_intent(query)
            
            # Perform semantic search with intent context
            search_results = await self.semantic_search.search_with_intent(
                query=query,
                intent=intent_result,
                max_results=3,
                similarity_threshold=0.7
            )
            
            print(f"  📊 Intent Category: {intent_result.intent_category}")
            print("  🎯 Top Results:")
            
            for i, result in enumerate(search_results, 1):
                print(f"    {i}. {result.title}")
                print(f"       Similarity: {result.similarity_score:.3f}")
                print(f"       Domain: {result.domain}")
                print(f"       URL: {result.url}")
    
    async def example_5_contextual_understanding(self):
        """
        Example 5: Contextual Understanding
        
        Demonstrates advanced contextual analysis:
        - Multi-turn conversation context
        - Temporal context analysis
        - Geographic context understanding
        - User preference learning
        """
        logger.info("=== Example 5: Contextual Understanding ===")
        
        # Simulate a conversation with context building
        conversation = [
            {
                "turn": 1,
                "query": "I'm looking for a new laptop",
                "context": {}
            },
            {
                "turn": 2,
                "query": "Something good for gaming",
                "context": {"previous_intent": "laptop_search"}
            },
            {
                "turn": 3,
                "query": "Under $1500 budget",
                "context": {
                    "previous_intent": "laptop_search",
                    "specifications": ["gaming"],
                    "budget_constraint": True
                }
            },
            {
                "turn": 4,
                "query": "Available in San Francisco stores",
                "context": {
                    "previous_intent": "laptop_search",
                    "specifications": ["gaming"],
                    "budget": "$1500",
                    "location_preference": True
                }
            }
        ]
        
        accumulated_context = {}
        
        for turn_data in conversation:
            turn = turn_data["turn"]
            query = turn_data["query"]
            context = turn_data["context"]
            
            print(f"\n💬 Turn {turn}: '{query}'")
            
            # Analyze intent with accumulated context
            intent_result = await self.intent_analyzer.analyze_intent_with_context(
                query=query,
                conversation_context=accumulated_context,
                turn_context=context
            )
            
            print(f"  🎯 Resolved Intent: {intent_result.intent_category}")
            print(f"  📝 Complete Query: {intent_result.resolved_query}")
            print(f"  🔄 Context Updates: {intent_result.context_updates}")
            
            # Update accumulated context
            accumulated_context.update(intent_result.context_updates)
            accumulated_context["turn"] = turn
            
            print(f"  💾 Accumulated Context: {accumulated_context}")
    
    async def example_6_domain_specific_analysis(self):
        """
        Example 6: Domain-Specific Analysis
        
        Demonstrates domain-specific intent analysis:
        - E-commerce intent patterns
        - News and media analysis
        - Research and academic queries
        - Financial data requests
        """
        logger.info("=== Example 6: Domain-Specific Analysis ===")
        
        # Domain-specific queries
        domain_queries = {
            "e_commerce": [
                "Best deals on wireless headphones",
                "iPhone 14 vs Samsung Galaxy S23 comparison",
                "Free shipping electronics under $100"
            ],
            "news": [
                "Breaking: Stock market volatility today",
                "Update on climate change summit",
                "Tech earnings reports this week"
            ],
            "research": [
                "Recent studies on renewable energy efficiency",
                "Peer-reviewed papers on quantum computing",
                "Meta-analysis of COVID-19 treatments"
            ],
            "finance": [
                "Tesla quarterly earnings forecast",
                "Cryptocurrency market trends analysis",
                "Real estate investment opportunities"
            ]
        }
        
        for domain, queries in domain_queries.items():
            print(f"\n🏷️  Domain: {domain.upper()}")
            print("=" * 40)
            
            for query in queries:
                print(f"\n🔍 Query: '{query}'")
                
                # Domain-specific analysis
                domain_result = await self.intent_analyzer.analyze_domain_specific(
                    query=query,
                    domain=domain
                )
                
                print(f"  📂 Sub-category: {domain_result.sub_category}")
                print(f"  🎯 Action Type: {domain_result.action_type}")
                print(f"  📊 Attributes: {domain_result.domain_attributes}")
                print(f"  🔄 Suggested URLs: {domain_result.suggested_url_patterns[:2]}")
                
                # Domain-specific entity extraction
                domain_entities = await self.intent_analyzer.extract_domain_entities(
                    query=query,
                    domain=domain
                )
                
                print(f"  🏷️  Domain Entities: {domain_entities}")
    
    async def example_7_performance_optimization(self):
        """
        Example 7: Performance Optimization
        
        Demonstrates performance optimization features:
        - Batch processing
        - Caching strategies
        - Asynchronous processing
        - Memory optimization
        """
        logger.info("=== Example 7: Performance Optimization ===")
        
        # Generate test queries for batch processing
        test_queries = [
            f"Find {product} reviews and prices" 
            for product in [
                "laptops", "smartphones", "tablets", "headphones", "cameras",
                "speakers", "keyboards", "monitors", "mice", "printers"
            ]
        ]
        
        print(f"🚀 Processing {len(test_queries)} queries...")
        
        # Batch processing with timing
        start_time = datetime.now()
        
        batch_results = await self.intent_analyzer.analyze_batch(
            queries=test_queries,
            enable_caching=True,
            parallel_processing=True,
            batch_size=5
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Total Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Queries per Second: {len(test_queries) / processing_time:.2f}")
        print(f"💾 Cache Hit Ratio: {batch_results.cache_hit_ratio:.2%}")
        print(f"🧠 Memory Usage: {batch_results.memory_usage_mb:.1f} MB")
        
        # Show performance breakdown
        print("\n📈 Performance Breakdown:")
        for stage, time_ms in batch_results.timing_breakdown.items():
            print(f"  {stage}: {time_ms:.1f}ms")
        
        return batch_results
    
    async def example_8_advanced_features_showcase(self):
        """
        Example 8: Advanced Features Showcase
        
        Demonstrates cutting-edge features:
        - Multi-language support
        - Cross-domain intent transfer
        - Predictive intent modeling
        - Real-time learning
        """
        logger.info("=== Example 8: Advanced Features Showcase ===")
        
        # Multi-language queries
        multilingual_queries = [
            ("Find restaurants in Paris", "en"),
            ("Buscar hoteles en Barcelona", "es"),
            ("Recherche d'appartements à Londres", "fr"),
            ("Suche nach Laptops in Deutschland", "de")
        ]
        
        print("🌍 Multi-language Intent Analysis:")
        for query, language in multilingual_queries:
            print(f"\n🔍 {language.upper()}: '{query}'")
            
            # Multi-language analysis
            result = await self.intent_analyzer.analyze_multilingual(
                query=query,
                source_language=language,
                target_language="en"
            )
            
            print(f"  🔄 Translated: '{result.translated_query}'")
            print(f"  📂 Intent: {result.intent_category}")
            print(f"  🌍 Location: {result.detected_location}")
        
        # Cross-domain intent transfer
        print("\n🔄 Cross-Domain Intent Transfer:")
        base_intent = await self.intent_analyzer.analyze_intent(
            "Find investment opportunities in renewable energy"
        )
        
        transferred_intents = await self.intent_analyzer.transfer_intent_across_domains(
            base_intent=base_intent,
            target_domains=["news", "research", "e_commerce"]
        )
        
        for domain, transferred_intent in transferred_intents.items():
            print(f"  {domain}: {transferred_intent.adapted_query}")
        
        # Predictive intent modeling
        print("\n🔮 Predictive Intent Modeling:")
        user_history = [
            "Tesla stock price",
            "Electric vehicle news",
            "EV charging stations near me",
            "Tesla Model 3 reviews"
        ]
        
        predicted_intents = await self.intent_analyzer.predict_next_intents(
            user_history=user_history,
            prediction_horizon=3
        )
        
        print("  🎯 Predicted Next Queries:")
        for i, prediction in enumerate(predicted_intents, 1):
            print(f"    {i}. {prediction.query} (confidence: {prediction.confidence:.2f})")


async def run_intent_analysis_examples():
    """Run all intent analysis examples"""
    examples = IntentAnalysisExamples()
    
    print("🧠 Starting SmartScrape Intent Analysis Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        await examples.example_1_basic_intent_analysis()
        await examples.example_2_entity_recognition_deep_dive()
        await examples.example_3_query_expansion_strategies()
        await examples.example_4_semantic_search_integration()
        await examples.example_5_contextual_understanding()
        await examples.example_6_domain_specific_analysis()
        await examples.example_7_performance_optimization()
        await examples.example_8_advanced_features_showcase()
        
        print("\n✅ All intent analysis examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Intent analysis examples failed: {e}")
        raise


async def interactive_intent_analyzer():
    """Interactive intent analyzer for testing"""
    examples = IntentAnalysisExamples()
    
    print("🚀 Interactive Intent Analyzer")
    print("Type 'exit' to quit, 'help' for commands")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n💭 Enter your query: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  exit - Quit the analyzer")
                print("  help - Show this help")
                print("  Just type any query to analyze it!")
                continue
            elif not user_input:
                continue
            
            # Analyze the user's query
            result = await examples.intent_analyzer.analyze_intent(user_input)
            
            print(f"\n📊 Analysis Results:")
            print(f"  Category: {result.intent_category}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Entities: {result.entities}")
            print(f"  Keywords: {result.keywords}")
            print(f"  Expanded Terms: {result.expanded_terms[:5]}...")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Run interactive mode
        asyncio.run(interactive_intent_analyzer())
    else:
        # Run all examples
        asyncio.run(run_intent_analysis_examples())
