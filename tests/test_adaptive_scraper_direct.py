"""
Direct Integration Tests for Adaptive Scraper

This module provides comprehensive tests for directly testing the AdaptiveScraper class
across different real-world websites including:
- Real estate listings in Cleveland, OH
- E-commerce product searches
- News article extraction
- Directory listings

These tests don't require the web API to be running, as they directly instantiate
and use the AdaptiveScraper class.
"""

import os
import sys
import json
import asyncio
import unittest
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components without monkey patching
from controllers.adaptive_scraper import AdaptiveScraper, get_adaptive_scraper
from strategies.core.strategy_types import StrategyCapability
from components.search_orchestrator import SearchOrchestrator
from components.site_discovery import SiteDiscovery
from components.pattern_analyzer import PatternAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='direct_integration_tests.log'
)
logger = logging.getLogger("AdaptiveScraperDirectTests")

# Test case categories
TEST_CATEGORIES = {
    "real_estate": {
        "sites": [
            {
                "url": "https://www.ohiobrokerdirect.com/",
                "name": "Ohio Broker Direct",
                "search_query": "Cleveland, OH",
                "location": "Cleveland, OH"
            },
            {
                "url": "https://www.realtor.com/realestateandhomes-search/Cleveland_OH",
                "name": "Realtor.com Cleveland",
                "search_query": "homes for sale",
                "location": "Cleveland, OH"
            }
        ],
        "expected_fields": ["price", "bedrooms", "bathrooms", "address", "sqft", "description"]
    },
    "e_commerce": {
        "sites": [
            {
                "url": "https://books.toscrape.com/",
                "name": "Books To Scrape",
                "search_query": "fiction books under $20",
                "category": "books"
            }
        ],
        "expected_fields": ["title", "price", "image", "rating", "description", "url"]
    },
    "news": {
        "sites": [
            {
                "url": "https://news.ycombinator.com/",
                "name": "Hacker News",
                "search_query": "technology news",
                "category": "tech"
            }
        ],
        "expected_fields": ["title", "url", "points", "comments", "author", "date"]
    }
}

class AdaptiveScraperDirectTests(unittest.TestCase):
    """
    Direct integration tests for the AdaptiveScraper across different website types.
    
    Tests the scraper's ability to:
    1. Extract structured data from different website categories
    2. Adapt to different page layouts and structures
    3. Handle pagination and multi-page extraction
    4. Apply the right strategies for different sites
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        # Create directory for results
        cls.results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(cls.results_dir, exist_ok=True)
        
        # Initialize test results tracking
        cls.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": "0%",
            "start_time": datetime.now().isoformat(),
            "results": []
        }
        
        # We'll create a fresh instance of AdaptiveScraper in each test
        # rather than sharing one instance across all tests
        # This ensures tests are isolated and don't impact each other
        
        # Log setup complete
        logger.info("Test setup complete, required services will be initialized in each test")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.test_results["end_time"] = datetime.now().isoformat()
        if cls.test_results["total_tests"] > 0:
            cls.test_results["success_rate"] = f"{(cls.test_results['passed'] / cls.test_results['total_tests']) * 100:.1f}%"
        
        # Save test results
        with open(os.path.join(cls.results_dir, 'direct_integration_test_results.json'), 'w') as f:
            json.dump(cls.test_results, f, indent=2, default=str)
        
        logger.info(f"Tests completed. Success rate: {cls.test_results['success_rate']}")
        logger.info(f"Results saved to {os.path.join(cls.results_dir, 'direct_integration_test_results.json')}")

    def _run_test_for_site(self, category: str, site_info: Dict[str, str], expected_fields: List[str]) -> Dict[str, Any]:
        """Run a test for a specific site and return the results."""
        site_name = site_info["name"]
        url = site_info["url"]
        search_query = site_info.get("search_query", "")
        
        logger.info(f"Testing {site_name} with query: '{search_query}'")
        
        result = {
            "site": site_name,
            "url": url,
            "query": search_query,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "results_count": 0,
            "execution_time": 0,
            "field_coverage": {},
            "error": None
        }
        
        start_time = datetime.now()
        
        try:
            # Create user intent based on category
            user_intent = self._create_user_intent_for_category(category, site_info)
            
            # Get a fresh instance of AdaptiveScraper with appropriate configurations
            scraper = get_adaptive_scraper(
                use_ai=True,
                max_pages=5,
                max_depth=3,
                use_pipelines=True
            )
            
            # Execute the scrape operation
            scrape_result = scraper.scrape(
                url=url,
                required_capabilities={StrategyCapability.AI_ASSISTED},
                options={
                    "intent": user_intent,
                    "max_pages": 5,
                    "use_extraction_pipelines": True,
                    "location": site_info.get("location", ""),
                    "follow_pagination": True
                }
            )
            
            # Check if result is a coroutine (async method)
            if asyncio.iscoroutine(scrape_result):
                # Run the coroutine in an event loop
                loop = asyncio.get_event_loop()
                scrape_result = loop.run_until_complete(scrape_result)
            
            # Process the results
            if scrape_result and scrape_result.get("success", False):
                result["success"] = True
                
                # Get list of results
                results_data = scrape_result.get("results", [])
                if isinstance(results_data, dict):
                    # Handle dictionary results
                    results_data = [results_data]
                
                # Save the raw results
                result["raw_results"] = results_data
                result["results_count"] = len(results_data)
                
                # Check for expected fields
                for field in expected_fields:
                    has_field = False
                    for item in results_data:
                        if not isinstance(item, dict):
                            continue
                            
                        # Check for exact field match
                        if field in item:
                            has_field = True
                            break
                            
                        # Check for field match in nested data
                        if "data" in item and isinstance(item["data"], dict) and field in item["data"]:
                            has_field = True
                            break
                            
                        # Check for fields with different names but same meaning
                        item_fields = list(item.keys())
                        if any(field.lower() in key.lower() for key in item_fields):
                            has_field = True
                            break
                            
                    result["field_coverage"][field] = has_field
                
                # Calculate field coverage percentage
                covered_fields = sum(1 for v in result["field_coverage"].values() if v)
                result["field_coverage_pct"] = round((covered_fields / len(expected_fields)) * 100, 2)
                
                # Save results to a file for this test
                filename = f"{site_info['name'].replace(' ', '_').lower()}_direct_test_results.json"
                with open(os.path.join(self.results_dir, filename), 'w') as f:
                    json.dump(scrape_result, f, indent=2, default=str)
                
                result["results_file"] = filename
                
            else:
                result["success"] = False
                result["error"] = scrape_result.get("error", "Unknown error") if scrape_result else "No result returned"
                
        except Exception as e:
            logger.error(f"Error testing {site_name}: {str(e)}")
            logger.error(traceback.format_exc())
            result["success"] = False
            result["error"] = str(e)
        
        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        return result

    def _create_user_intent_for_category(self, category: str, site_info: Dict[str, str]) -> Dict[str, Any]:
        """Create a user intent dictionary based on the category and site info."""
        base_intent = {
            "description": site_info.get("search_query", ""),
            "extraction_type": category
        }
        
        # Add category-specific fields
        if category == "real_estate":
            base_intent.update({
                "location": site_info.get("location", ""),
                "property_type": "home",
                "min_bedrooms": 2,
                "min_bathrooms": 1
            })
        elif category == "e_commerce":
            base_intent.update({
                "product_category": site_info.get("category", ""),
                "price_range": {
                    "max": site_info.get("max_price", 100)
                }
            })
        elif category == "news":
            base_intent.update({
                "topic": site_info.get("category", ""),
                "recency": "recent"
            })
        
        return base_intent

    def _process_test_results(self, scrape_result, site_info, expected_fields):
        """Process and analyze scrape results for test reporting."""
        result = {
            "site": site_info["name"],
            "url": site_info["url"],
            "query": site_info.get("search_query", ""),
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "results_count": 0,
            "execution_time": 0,
            "field_coverage": {},
            "error": None
        }
        
        # Process the results
        if scrape_result and scrape_result.get("success", False):
            result["success"] = True
            
            # Get list of results
            results_data = scrape_result.get("results", [])
            if isinstance(results_data, dict):
                # Handle dictionary results
                results_data = [results_data]
            
            # Save the raw results
            result["raw_results"] = results_data
            result["results_count"] = len(results_data)
            
            # Check for expected fields
            field_occurrences = {field: 0 for field in expected_fields}
            
            for item in results_data:
                if not isinstance(item, dict):
                    continue
                    
                # Track fields in this item
                item_fields = set()
                
                # Check for exact field match
                for field in expected_fields:
                    if field in item:
                        item_fields.add(field)
                        field_occurrences[field] += 1
                        
                    # Check for field match in nested data
                    elif "data" in item and isinstance(item["data"], dict) and field in item["data"]:
                        item_fields.add(field)
                        field_occurrences[field] += 1
                    
                    # Check for fields in _metadata or metadata
                    elif "_metadata" in item and isinstance(item["_metadata"], dict) and field in item["_metadata"]:
                        item_fields.add(field)
                        field_occurrences[field] += 1
                    elif "metadata" in item and isinstance(item["metadata"], dict) and field in item["metadata"]:
                        item_fields.add(field)
                        field_occurrences[field] += 1
                        
                    # Check for fields with different names but same meaning
                    else:
                        item_keys = list(item.keys())
                        if any(field.lower() in key.lower() for key in item_keys):
                            item_fields.add(field)
                            field_occurrences[field] += 1
            
            # Calculate field presence (in at least one result)
            for field in expected_fields:
                result["field_coverage"][field] = field_occurrences[field] > 0
            
            # Add field occurrence stats
            result["field_occurrences"] = field_occurrences
            result["field_occurrence_pct"] = {
                field: (count / len(results_data) * 100) if results_data else 0 
                for field, count in field_occurrences.items()
            }
            
            # Calculate field coverage percentage
            covered_fields = sum(1 for v in result["field_coverage"].values() if v)
            result["field_coverage_pct"] = round((covered_fields / len(expected_fields)) * 100, 2)
            
            # Add sample data for visualization
            if results_data:
                result["sample_result"] = results_data[0]
            
            # Save results to a file for this test
            filename = f"{site_info['name'].replace(' ', '_').lower()}_direct_test_results.json"
            with open(os.path.join(self.__class__.results_dir, filename), 'w') as f:
                json.dump(scrape_result, f, indent=2, default=str)
            
            result["results_file"] = filename
            
            # Add execution time if available
            if "execution_time" in scrape_result:
                result["execution_time"] = scrape_result["execution_time"]
            elif "metrics" in scrape_result and "execution_time" in scrape_result["metrics"]:
                result["execution_time"] = scrape_result["metrics"]["execution_time"]
            
            # Add chosen strategy information if available
            if "strategy" in scrape_result:
                result["strategy"] = scrape_result["strategy"]
            
            # Add content analysis information if available
            if "content_analysis" in scrape_result:
                result["content_analysis"] = scrape_result["content_analysis"]
            
            # Add extraction quality metrics if available
            if "quality_metrics" in scrape_result:
                result["quality_metrics"] = scrape_result["quality_metrics"]
                
        else:
            result["success"] = False
            result["error"] = scrape_result.get("error", "Unknown error") if scrape_result else "No result returned"
            
        return result

    def test_ohio_broker_direct(self):
        """Test scraper on Ohio Broker Direct for homes in Cleveland, OH."""
        category = "real_estate"
        site_info = next((site for site in TEST_CATEGORIES[category]["sites"] 
                          if site["name"] == "Ohio Broker Direct"), None)
        
        if not site_info:
            self.skipTest("Ohio Broker Direct site info not found in test categories")
            
        logger.info(f"Running Ohio Broker Direct test for homes in Cleveland, OH")
        
        # Set the search query explicitly to "homes in Cleveland, OH"
        site_info["search_query"] = "homes in Cleveland, OH"
        
        # Create user intent with the search query
        user_intent = self._create_user_intent_for_category(category, site_info)
        user_intent["description"] = "homes in Cleveland, OH"
        
        # Execute the scrape operation starting from the homepage
        try:
            # Start at the homepage
            url = site_info["url"]  # https://www.ohiobrokerdirect.com/
            query = "homes in Cleveland, OH"  # Explicit search query
            
            logger.info(f"Starting search at {url} with query: {query}")
            
            # Get a fresh instance of AdaptiveScraper with appropriate configurations
            scraper = get_adaptive_scraper(
                use_ai=True,
                max_pages=5,
                max_depth=3,
                use_pipelines=True
            )
            
            # Let the SearchOrchestrator handle strategy selection naturally
            scrape_result = scraper.scrape(
                url=url,
                options={
                    "intent": user_intent,
                    "query": query,
                    "max_pages": 5,
                    "use_extraction_pipelines": True,
                    "location": "Cleveland, OH",
                    "follow_pagination": True,
                    # No forced strategy selection - let the scraper decide based on site analysis
                }
            )
            
            # Check if result is a coroutine (async method)
            if asyncio.iscoroutine(scrape_result):
                # Run the coroutine in an event loop
                loop = asyncio.get_event_loop()
                scrape_result = loop.run_until_complete(scrape_result)
            
            # Process results and assertions
            test_result = self._process_test_results(
                scrape_result, 
                site_info, 
                TEST_CATEGORIES[category]["expected_fields"]
            )
            
            # Log detailed results for debugging
            logger.info(f"Strategy used: {scrape_result.get('strategy', 'Unknown')}")
            logger.info(f"Results count: {test_result['results_count']}")
            logger.info(f"Field coverage: {test_result['field_coverage_pct']}%")
            logger.info(f"Execution time: {test_result.get('execution_time', 'Unknown')} seconds")
            
            # Log field coverage details
            if 'field_coverage' in test_result:
                logger.info("Field coverage details:")
                for field, has_field in test_result['field_coverage'].items():
                    logger.info(f"  - Field '{field}': {'✓' if has_field else '✗'}")
                    
            # Log sample data from result for inspection
            if 'raw_results' in test_result and test_result['raw_results']:
                sample = test_result['raw_results'][0]
                logger.info(f"Sample data from first result: {json.dumps(sample, indent=2)}")
            
            # Update overall results
            self.__class__.test_results["total_tests"] += 1
            if test_result["success"]:
                self.__class__.test_results["passed"] += 1
            else:
                self.__class__.test_results["failed"] += 1
            
            self.__class__.test_results["results"].append({
                "case": {
                    "category": category,
                    "site_name": site_info["name"],
                    "url": site_info["url"],
                    "query": query
                },
                **test_result
            })
            
            # Save output to a dedicated file for this test run
            result_filename = f"ohio_broker_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            result_path = os.path.join(self.results_dir, result_filename)
            with open(result_path, 'w') as f:
                json.dump(test_result, f, indent=2, default=str)
            logger.info(f"Detailed test results saved to {result_path}")
            
            # Assert success for unittest
            self.assertTrue(
                test_result["success"], 
                f"Failed to extract data from {site_info['name']}: {test_result.get('error', 'Unknown error')}"
            )
            
            # Verify we have results
            self.assertGreater(
                test_result["results_count"], 0, 
                f"No results found for {site_info['name']} using query '{query}'. " +
                f"Check that the scraper is properly handling the search form and extracting results."
            )
            
            # Verify at least some of the expected fields are present
            min_coverage = 50  # At least 50% of expected fields should be covered
            self.assertGreaterEqual(
                test_result["field_coverage_pct"], min_coverage,
                f"Field coverage too low: {test_result['field_coverage_pct']}% < {min_coverage}%\n" +
                f"Fields found: {[f for f, has_f in test_result['field_coverage'].items() if has_f]}\n" +
                f"Fields missing: {[f for f, has_f in test_result['field_coverage'].items() if not has_f]}\n" +
                f"This indicates the scraper is not extracting all required fields. Check extraction strategy."
            )
            
            logger.info(f"Test completed for {site_info['name']} with {test_result['results_count']} results and " +
                       f"{test_result['field_coverage_pct']}% field coverage")
        
        except Exception as e:
            logger.error(f"Error executing test for {site_info['name']}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Save error details to file for easier debugging
            error_filename = f"ohio_broker_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            error_path = os.path.join(self.results_dir, error_filename)
            with open(error_path, 'w') as f:
                f.write(f"Error executing test: {str(e)}\n\n")
                f.write(traceback.format_exc())
            logger.error(f"Error details saved to {error_path}")
            
            raise

    def test_books_to_scrape(self):
        """Test scraper on Books To Scrape for fiction books under $20."""
        category = "e_commerce"
        site_info = next((site for site in TEST_CATEGORIES[category]["sites"] 
                          if site["name"] == "Books To Scrape"), None)
        
        if not site_info:
            self.skipTest("Books To Scrape site info not found in test categories")
            
        logger.info(f"Running Books To Scrape test for fiction books under $20")
        
        test_result = self._run_test_for_site(
            category, 
            site_info, 
            TEST_CATEGORIES[category]["expected_fields"]
        )
        
        # Update overall results
        self.__class__.test_results["total_tests"] += 1
        if test_result["success"]:
            self.__class__.test_results["passed"] += 1
        else:
            self.__class__.test_results["failed"] += 1
        
        self.__class__.test_results["results"].append({
            "case": {
                "category": category,
                "site_name": site_info["name"],
                "url": site_info["url"]
            },
            **test_result
        })
        
        # Assert success for unittest
        self.assertTrue(
            test_result["success"], 
            f"Failed to extract data from {site_info['name']}: {test_result.get('error', 'Unknown error')}"
        )
        
        # Verify we have results
        self.assertGreater(
            test_result["results_count"], 0, 
            f"No results found for {site_info['name']}"
        )
        
        # Verify at least some of the expected fields are present
        min_coverage = 50  # At least 50% of expected fields should be covered
        self.assertGreaterEqual(
            test_result["field_coverage_pct"], min_coverage,
            f"Field coverage too low: {test_result['field_coverage_pct']}% < {min_coverage}%"
        )
        
        logger.info(f"Test completed for {site_info['name']} with {test_result['results_count']} results and " +
                   f"{test_result['field_coverage_pct']}% field coverage")

    def test_hacker_news(self):
        """Test scraper on Hacker News for technology news."""
        category = "news"
        site_info = next((site for site in TEST_CATEGORIES[category]["sites"] 
                          if site["name"] == "Hacker News"), None)
        
        if not site_info:
            self.skipTest("Hacker News site info not found in test categories")
            
        logger.info(f"Running Hacker News test for technology news")
        
        test_result = self._run_test_for_site(
            category, 
            site_info, 
            TEST_CATEGORIES[category]["expected_fields"]
        )
        
        # Update overall results
        self.__class__.test_results["total_tests"] += 1
        if test_result["success"]:
            self.__class__.test_results["passed"] += 1
        else:
            self.__class__.test_results["failed"] += 1
        
        self.__class__.test_results["results"].append({
            "case": {
                "category": category,
                "site_name": site_info["name"],
                "url": site_info["url"]
            },
            **test_result
        })
        
        # Assert success for unittest
        self.assertTrue(
            test_result["success"], 
            f"Failed to extract data from {site_info['name']}: {test_result.get('error', 'Unknown error')}"
        )
        
        # Verify we have results
        self.assertGreater(
            test_result["results_count"], 0, 
            f"No results found for {site_info['name']}"
        )
        
        # Verify at least some of the expected fields are present
        min_coverage = 50  # At least 50% of expected fields should be covered
        self.assertGreaterEqual(
            test_result["field_coverage_pct"], min_coverage,
            f"Field coverage too low: {test_result['field_coverage_pct']}% < {min_coverage}%"
        )
        
        logger.info(f"Test completed for {site_info['name']} with {test_result['results_count']} results and " +
                   f"{test_result['field_coverage_pct']}% field coverage")

def run_tests():
    """Run all direct integration tests."""
    logger.info("Starting AdaptiveScraper direct integration tests")
    unittest.main(module=__name__, exit=False)
    logger.info("AdaptiveScraper direct integration tests completed")

if __name__ == "__main__":
    run_tests()