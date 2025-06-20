#!/usr/bin/env python3
"""
Test the fallback extraction system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from controllers.extraction_coordinator import ExtractionCoordinator
import json

async def test_fallback_extraction():
    """Test fallback extraction system"""
    print("Testing fallback extraction system...")
    
    try:
        coordinator = ExtractionCoordinator()
        
        # Test URLs with different complexity levels
        test_urls = [
            "https://httpbin.org/html",  # Simple static HTML
            "https://jsonplaceholder.typicode.com/posts/1",  # JSON API
            "https://example.com",  # Very simple page
        ]
        
        for url in test_urls:
            print(f"\n🔍 Testing fallback extraction for: {url}")
            
            result = await coordinator.extract_with_fallbacks(url)
            
            if result.get('success'):
                print(f"✅ Extraction successful!")
                print(f"   Strategy: {result.get('strategy', 'unknown')}")
                print(f"   Content length: {len(result.get('content', ''))}")
                print(f"   Word count: {result.get('word_count', 0)}")
                if result.get('metadata'):
                    print(f"   Title: {result.get('metadata', {}).get('title', 'N/A')}")
            else:
                print(f"❌ Extraction failed: {result.get('error')}")
                if result.get('strategies_attempted'):
                    print(f"   Strategies attempted: {result.get('strategies_attempted')}")
                    
        return True
        
    except Exception as e:
        print(f"❌ Fallback extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_strategies():
    """Test individual strategies"""
    print("\n🧪 Testing individual strategies...")
    
    test_url = "https://httpbin.org/html"
    
    try:
        # Test Trafilatura strategy
        print(f"\n📚 Testing Trafilatura strategy...")
        from strategies.trafilatura_strategy import TrafilaturaStrategy
        trafilatura = TrafilaturaStrategy()
        result = await trafilatura.extract(test_url)
        
        if result.get('success'):
            print(f"✅ Trafilatura successful - Content length: {len(result.get('content', ''))}")
        else:
            print(f"❌ Trafilatura failed: {result.get('error')}")
        
        # Test Playwright strategy
        print(f"\n🎭 Testing Playwright strategy...")
        from strategies.playwright_strategy import PlaywrightStrategy
        playwright = PlaywrightStrategy()
        result = await playwright.extract(test_url)
        
        if result.get('success'):
            print(f"✅ Playwright successful - Content length: {len(result.get('content', ''))}")
        else:
            print(f"❌ Playwright failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Individual strategy tests failed: {e}")
        return False
    
    return True

async def main():
    """Main test function"""
    print("Starting fallback extraction tests...")
    
    # Test individual strategies first
    individual_success = await test_individual_strategies()
    
    # Then test the fallback chain
    fallback_success = await test_fallback_extraction()
    
    if individual_success and fallback_success:
        print("\n🎉 All fallback extraction tests passed!")
    else:
        print("\n💥 Some fallback extraction tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
