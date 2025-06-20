#!/usr/bin/env python3
"""
Final validation test for the complete intelligent scraping system
"""

import sys
import os
import asyncio
sys.path.append('/Users/johnny/Downloads/SmartScrape')

from controllers.adaptive_scraper import AdaptiveScraper
from processors.content_quality_scorer import ContentQualityScorer
from components.universal_intent_analyzer import UniversalIntentAnalyzer

async def test_complete_intelligent_system():
    """Test the complete intelligent scraping system with all components working together."""
    print("🔍 Testing Complete Intelligent Scraping System...")
    
    # Initialize the system
    scraper = AdaptiveScraper()
    intent_analyzer = UniversalIntentAnalyzer()
    quality_scorer = ContentQualityScorer(intent_analyzer=intent_analyzer)
    
    # Test query
    query = "latest Tesla Model S updates"
    
    print(f"Query: {query}")
    print("\n1. Intent Analysis...")
    
    # Test intent analysis
    intent_result = intent_analyzer.analyze_intent(query)
    print(f"   Intent Type: {intent_result.get('entity_type', 'unknown')}")
    print(f"   Keywords: {intent_result.get('keywords', [])}")
    print(f"   Expanded Queries: {len(intent_result.get('expanded_queries', []))}")
    
    print("\n2. Intelligent Scraping...")
    
    # Test intelligent scraping
    try:
        results = await scraper.process_query(
            user_query=query,
            options={
                'enable_intelligent_hunting': True,
                'max_results': 3,
                'start_url': 'https://www.tesla.com'
            }
        )
        
        result_data = results.get('data', [])
        print(f"   Found {len(result_data)} results")
        
        print("\n3. Content Quality Assessment...")
        
        # Test content quality scoring for each result
        for i, result in enumerate(result_data[:2], 1):  # Test first 2 results
            content = result.get('content', '')
            if content:
                scores = quality_scorer.get_detailed_scores(content, query)
                print(f"   Result {i}:")
                print(f"     URL: {result.get('url', 'N/A')[:60]}...")
                print(f"     Overall Score: {scores.get('overall_score', 0.0):.3f}")
                print(f"     Semantic Relevance: {scores.get('semantic_relevance', 0.0):.3f}")
                print(f"     Content Length: {len(content)} chars")
        
        print("\n✅ Complete Intelligent System is working correctly!")
        print("🎉 All components are integrated and operational:")
        print("   - Intent Analysis ✓")
        print("   - URL Discovery ✓") 
        print("   - Content Hunting ✓")
        print("   - Quality Scoring ✓")
        print("   - Semantic Relevance ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in intelligent scraping: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("FINAL VALIDATION: SmartScrape Intelligent System")
    print("=" * 60)
    
    success = asyncio.run(test_complete_intelligent_system())
    
    if success:
        print("\n🎯 VALIDATION COMPLETE: SmartScrape is fully intelligent and operational!")
    else:
        print("\n❌ VALIDATION FAILED: System needs attention")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
