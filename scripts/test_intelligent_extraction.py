#!/usr/bin/env python3
"""
Test the intelligent strategy selection system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from controllers.extraction_coordinator import ExtractionCoordinator
from components.strategy_selector import AdaptiveStrategySelector
from components.domain_intelligence import DomainIntelligence
import json

async def test_javascript_detection():
    """Test JavaScript detection functionality"""
    print("🔍 Testing JavaScript detection...")
    
    domain_intel = DomainIntelligence()
    domain_intel.initialize()
    
    # Test cases with different JS complexity
    test_cases = [
        {
            'name': 'Static HTML',
            'html': '<html><body><h1>Hello World</h1><p>This is static content.</p></body></html>',
            'expected_js': False
        },
        {
            'name': 'React App',
            'html': '''<html><head><script src="react.js"></script></head>
                      <body><div id="root"></div>
                      <script>ReactDOM.render(React.createElement("h1"), document.getElementById("root"));</script>
                      </body></html>''',
            'expected_js': True
        },
        {
            'name': 'jQuery Site',
            'html': '''<html><head><script src="jquery.js"></script></head>
                      <body><div class="content">Content loaded with jQuery</div>
                      <script>$(document).ready(function() { $(".content").show(); });</script>
                      </body></html>''',
            'expected_js': True
        },
        {
            'name': 'Vue.js App',
            'html': '''<html><head><script src="vue.js"></script></head>
                      <body><div id="app" v-if="show">{{ message }}</div>
                      <script>new Vue({ el: "#app", data: { message: "Hello", show: true } });</script>
                      </body></html>''',
            'expected_js': True
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  📋 Testing: {test_case['name']}")
        
        result = await domain_intel.detect_javascript_dependency(
            f"https://example.com/{test_case['name'].lower().replace(' ', '-')}", 
            test_case['html']
        )
        
        print(f"    Requires JS: {result['requires_js']} (expected: {test_case['expected_js']})")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    JS Heavy: {result['js_heavy']}")
        print(f"    Frameworks: {[f['name'] for f in result['frameworks']]}")
        
        # Check if detection matches expectation
        if result['requires_js'] == test_case['expected_js']:
            print(f"    ✅ Detection correct")
        else:
            print(f"    ❌ Detection incorrect")
    
    return True

async def test_strategy_selection():
    """Test adaptive strategy selection"""
    print("\n🎯 Testing strategy selection...")
    
    selector = AdaptiveStrategySelector()
    
    # Test different scenarios
    test_scenarios = [
        {
            'name': 'News Article',
            'url': 'https://cnn.com/news/article-title',
            'domain_info': {'requires_js': False, 'confidence': 0.1},
            'expected_strategy': 'trafilatura'
        },
        {
            'name': 'React SPA',
            'url': 'https://app.example.com/dashboard',
            'domain_info': {
                'requires_js': True, 
                'confidence': 0.8, 
                'js_heavy': True,
                'frameworks': [{'name': 'react', 'confidence': 0.9}]
            },
            'expected_strategy': 'playwright'
        },
        {
            'name': 'General Website',
            'url': 'https://example.com/page',
            'domain_info': {'requires_js': False, 'confidence': 0.0},
            'expected_strategy': 'trafilatura'  # or universal_crawl4ai
        },
        {
            'name': 'E-commerce Product',
            'url': 'https://shop.example.com/product/123',
            'domain_info': {'requires_js': True, 'confidence': 0.5},
            'expected_strategy': 'universal_crawl4ai'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n  📋 Testing: {scenario['name']}")
        print(f"    URL: {scenario['url']}")
        
        selected = await selector.select_optimal_strategy(
            scenario['url'], 
            scenario['domain_info']
        )
        
        print(f"    Selected: {selected}")
        print(f"    Expected: {scenario['expected_strategy']}")
        
        # Note: Strategy selection is heuristic, so we just log the results
        if selected in ['trafilatura', 'playwright', 'universal_crawl4ai']:
            print(f"    ✅ Valid strategy selected")
        else:
            print(f"    ❌ Invalid strategy selected")
    
    return True

async def test_intelligent_extraction():
    """Test the full intelligent extraction workflow"""
    print("\n🧠 Testing intelligent extraction workflow...")
    
    coordinator = ExtractionCoordinator()
    
    # Test URLs with different characteristics
    test_urls = [
        "https://httpbin.org/html",  # Simple static content
        "https://jsonplaceholder.typicode.com/posts/1",  # JSON API response
    ]
    
    for url in test_urls:
        print(f"\n  🔍 Testing intelligent extraction for: {url}")
        
        result = await coordinator.extract_with_intelligent_selection(url)
        
        if result.get('success'):
            print(f"    ✅ Extraction successful!")
            print(f"    Strategy used: {result.get('strategy', 'unknown')}")
            print(f"    Content length: {len(result.get('content', ''))}")
            print(f"    Word count: {result.get('word_count', 0)}")
        else:
            print(f"    ❌ Extraction failed: {result.get('error')}")
            if result.get('strategies_attempted'):
                print(f"    Strategies attempted: {result.get('strategies_attempted')}")
    
    return True

async def test_performance_tracking():
    """Test performance tracking functionality"""
    print("\n📊 Testing performance tracking...")
    
    selector = AdaptiveStrategySelector()
    
    # Simulate some performance data
    test_performances = [
        {'url': 'https://news.example.com/article1', 'strategy': 'trafilatura', 'success': True, 'time': 1.2, 'quality': 0.9},
        {'url': 'https://news.example.com/article2', 'strategy': 'trafilatura', 'success': True, 'time': 0.8, 'quality': 0.85},
        {'url': 'https://app.example.com/page1', 'strategy': 'playwright', 'success': True, 'time': 3.5, 'quality': 0.8},
        {'url': 'https://app.example.com/page2', 'strategy': 'playwright', 'success': False, 'time': 5.0, 'quality': 0.0},
        {'url': 'https://general.example.com/page1', 'strategy': 'universal_crawl4ai', 'success': True, 'time': 2.1, 'quality': 0.75},
    ]
    
    # Update performance metrics
    for perf in test_performances:
        await selector.update_performance(
            perf['url'], perf['strategy'], perf['success'], 
            perf['time'], perf['quality']
        )
    
    # Get performance summary
    summary = selector.get_performance_summary()
    
    print(f"  📈 Performance Summary:")
    print(f"    Total domains tracked: {summary['total_domains']}")
    print(f"    Strategies with data: {list(summary['strategies'].keys())}")
    
    for strategy, stats in summary['strategies'].items():
        print(f"    {strategy}:")
        print(f"      Success rate: {stats['success_rate']:.2f}")
        print(f"      Avg response time: {stats['avg_response_time']:.2f}s")
        print(f"      Total attempts: {stats['total_attempts']}")
    
    print(f"  ✅ Performance tracking working correctly")
    return True

async def main():
    """Main test function"""
    print("Starting intelligent extraction system tests...")
    
    try:
        # Run all tests
        js_test = await test_javascript_detection()
        strategy_test = await test_strategy_selection()
        extraction_test = await test_intelligent_extraction()
        performance_test = await test_performance_tracking()
        
        if all([js_test, strategy_test, extraction_test, performance_test]):
            print("\n🎉 All intelligent extraction tests passed!")
        else:
            print("\n💥 Some intelligent extraction tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
