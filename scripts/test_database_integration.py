#!/usr/bin/env python3
"""
Test database integration for SmartScrape
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.database_manager import db_manager
from controllers.extraction_coordinator import ExtractionCoordinator
import config

async def test_database_initialization():
    """Test database initialization"""
    print("🔧 Testing database initialization...")
    
    # Override config to enable database for testing
    original_enabled = config.DATABASE_ENABLED
    config.DATABASE_ENABLED = True
    
    try:
        success = await db_manager.initialize()
        if success:
            print("  ✅ Database initialized successfully")
        else:
            print("  ❌ Database initialization failed")
        
        return success
        
    finally:
        # Restore original setting
        config.DATABASE_ENABLED = original_enabled

async def test_extraction_result_storage():
    """Test storing extraction results"""
    print("\n💾 Testing extraction result storage...")
    
    if not db_manager.enabled:
        print("  ⏭️ Database disabled, skipping test")
        return True
    
    # Test extraction result
    test_result = {
        'url': 'https://test.example.com/page1',
        'strategy': 'test_strategy',
        'content': 'This is test content for database storage',
        'word_count': 8,
        'metadata': {'title': 'Test Page', 'description': 'Test description'},
        'success': True,
        'response_time': 1.5,
        'quality_score': 0.8
    }
    
    try:
        result_id = await db_manager.save_extraction_result(test_result)
        
        if result_id:
            print(f"  ✅ Extraction result saved with ID: {result_id}")
            return True
        else:
            print("  ❌ Failed to save extraction result")
            return False
            
    except Exception as e:
        print(f"  ❌ Error saving extraction result: {e}")
        return False

async def test_domain_profile_management():
    """Test domain profile storage and retrieval"""
    print("\n🌐 Testing domain profile management...")
    
    if not db_manager.enabled:
        print("  ⏭️ Database disabled, skipping test")
        return True
    
    test_domain = "test.example.com"
    profile_data = {
        'optimal_strategy': 'trafilatura',
        'requires_js': False,
        'content_type': 'news',
        'avg_response_time': 1.2,
        'success_rate': 0.95,
        'total_extractions': 10,
        'analysis_confidence': 0.8
    }
    
    try:
        # Update domain profile
        success = await db_manager.update_domain_profile(test_domain, profile_data)
        if not success:
            print("  ❌ Failed to update domain profile")
            return False
        
        # Retrieve domain profile
        retrieved_profile = await db_manager.get_domain_profile(test_domain)
        if retrieved_profile:
            print(f"  ✅ Domain profile saved and retrieved successfully")
            print(f"    Domain: {retrieved_profile['domain']}")
            print(f"    Optimal strategy: {retrieved_profile['optimal_strategy']}")
            print(f"    Success rate: {retrieved_profile['success_rate']}")
            return True
        else:
            print("  ❌ Failed to retrieve domain profile")
            return False
            
    except Exception as e:
        print(f"  ❌ Error managing domain profile: {e}")
        return False

async def test_strategy_performance_tracking():
    """Test strategy performance tracking"""
    print("\n📊 Testing strategy performance tracking...")
    
    if not db_manager.enabled:
        print("  ⏭️ Database disabled, skipping test")
        return True
    
    test_data = [
        {'domain': 'news.example.com', 'strategy': 'trafilatura', 'success': True, 'time': 1.2, 'quality': 0.9},
        {'domain': 'news.example.com', 'strategy': 'trafilatura', 'success': True, 'time': 0.8, 'quality': 0.85},
        {'domain': 'app.example.com', 'strategy': 'playwright', 'success': True, 'time': 3.5, 'quality': 0.8},
        {'domain': 'app.example.com', 'strategy': 'playwright', 'success': False, 'time': 5.0},
    ]
    
    try:
        # Update performance metrics
        for data in test_data:
            success = await db_manager.update_strategy_performance(
                data['domain'], data['strategy'], data['success'], 
                data['time'], data.get('quality')
            )
            if not success:
                print(f"  ❌ Failed to update performance for {data['domain']}/{data['strategy']}")
                return False
        
        # Retrieve performance data
        performance_data = await db_manager.get_strategy_performance()
        
        if performance_data:
            print(f"  ✅ Performance tracking working, {len(performance_data)} records")
            for perf in performance_data[:2]:  # Show first 2 records
                print(f"    {perf['domain']}/{perf['strategy']}: "
                     f"{perf['success_rate']:.2f} success rate, "
                     f"{perf['avg_response_time']:.2f}s avg time")
            return True
        else:
            print("  ❌ No performance data retrieved")
            return False
            
    except Exception as e:
        print(f"  ❌ Error tracking strategy performance: {e}")
        return False

async def test_system_metrics():
    """Test system metrics recording"""
    print("\n📈 Testing system metrics recording...")
    
    if not db_manager.enabled:
        print("  ⏭️ Database disabled, skipping test")
        return True
    
    try:
        # Record some test metrics
        metrics = [
            {'name': 'extraction_count', 'value': 42.0, 'unit': 'count', 'component': 'coordinator'},
            {'name': 'avg_response_time', 'value': 1.5, 'unit': 'seconds', 'component': 'extractor'},
            {'name': 'cache_hit_rate', 'value': 0.75, 'unit': 'percentage', 'component': 'cache'}
        ]
        
        for metric in metrics:
            success = await db_manager.record_system_metric(
                metric['name'], metric['value'], metric['unit'], 
                metric['component']
            )
            if not success:
                print(f"  ❌ Failed to record metric {metric['name']}")
                return False
        
        print(f"  ✅ System metrics recorded successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error recording system metrics: {e}")
        return False

async def test_integrated_extraction():
    """Test extraction with database integration"""
    print("\n🔬 Testing integrated extraction with database...")
    
    if not db_manager.enabled:
        print("  ⏭️ Database disabled, skipping test")
        return True
    
    try:
        coordinator = ExtractionCoordinator()
        
        # Test URL
        test_url = "https://httpbin.org/html"
        
        print(f"  🔍 Testing extraction for: {test_url}")
        result = await coordinator.extract_with_intelligent_selection(test_url)
        
        if result.get('success'):
            print(f"  ✅ Extraction successful with database integration")
            print(f"    Strategy: {result.get('strategy')}")
            print(f"    Content length: {len(result.get('content', ''))}")
            
            # Check if data was saved to database
            extraction_stats = await db_manager.get_extraction_stats(days=1)
            print(f"  📊 Recent extractions: {extraction_stats.get('total_extractions', 0)}")
            
            return True
        else:
            print(f"  ❌ Extraction failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error in integrated extraction: {e}")
        return False

async def test_database_stats():
    """Test database statistics"""
    print("\n📊 Testing database statistics...")
    
    if not db_manager.enabled:
        print("  ⏭️ Database disabled, skipping test")
        return True
    
    try:
        stats = await db_manager.get_extraction_stats(days=7)
        
        print(f"  📈 Extraction Statistics (last 7 days):")
        print(f"    Total extractions: {stats.get('total_extractions', 0)}")
        print(f"    Successful extractions: {stats.get('successful_extractions', 0)}")
        print(f"    Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"    Unique domains: {stats.get('unique_domains', 0)}")
        
        strategy_breakdown = stats.get('strategy_breakdown', [])
        if strategy_breakdown:
            print(f"    Strategy breakdown:")
            for strategy in strategy_breakdown:
                print(f"      {strategy['strategy']}: {strategy['success_rate']:.2%} success, "
                     f"{strategy['avg_response_time']:.2f}s avg")
        
        print(f"  ✅ Database statistics retrieved successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error getting database statistics: {e}")
        return False

async def main():
    """Main test function"""
    print("Starting database integration tests...")
    
    try:
        # Run tests
        tests = [
            test_database_initialization(),
            test_extraction_result_storage(),
            test_domain_profile_management(),
            test_strategy_performance_tracking(),
            test_system_metrics(),
            test_integrated_extraction(),
            test_database_stats()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Check results
        passed = 0
        failed = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"\n❌ Test {i+1} failed with exception: {result}")
                failed += 1
            elif result:
                passed += 1
            else:
                failed += 1
        
        print(f"\n📋 Test Summary:")
        print(f"  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"  📊 Total: {len(tests)}")
        
        if failed == 0:
            print("\n🎉 All database integration tests passed!")
        else:
            print(f"\n💥 {failed} database integration tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
