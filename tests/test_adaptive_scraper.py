"""
Integration Tests for Adaptive Scraper

This module provides comprehensive tests for the adaptive scraper
across different website types including:
- Real estate listings
- E-commerce product pages
- Content/article sites
- Directory/listing sites
"""

import os
import sys
import json
import unittest
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.adaptive_scraper import AdaptiveScraper
from utils.metrics_analyzer import ScraperMetricsAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='integration_tests.log'
)
logger = logging.getLogger("AdaptiveScraperTests")

# Test case categories
TEST_CATEGORIES = {
    "real_estate": {
        "sites": [
            {"url": "https://www.zillow.com/homes/for_sale/", "name": "Zillow"},
            {"url": "https://www.realtor.com/realestateandhomes-search/", "name": "Realtor"},
            {"url": "https://www.redfin.com/city/", "name": "Redfin"},
        ],
        "expected_fields": ["price", "bedrooms", "bathrooms", "address", "sqft", "description"]
    },
    "e_commerce": {
        "sites": [
            {"url": "https://books.toscrape.com/", "name": "Books To Scrape"},
            {"url": "https://www.amazon.com/s?k=books", "name": "Amazon Books"},
            {"url": "https://shop.adidas.com/us/men-shoes/", "name": "Adidas Shop"},
        ],
        "expected_fields": ["title", "price", "image", "rating", "description", "url"]
    },
    "content_sites": {
        "sites": [
            {"url": "https://news.ycombinator.com/", "name": "Hacker News"},
            {"url": "https://www.bbc.com/news", "name": "BBC News"},
            {"url": "https://techcrunch.com/", "name": "TechCrunch"},
        ],
        "expected_fields": ["title", "author", "date", "content", "url"]
    },
    "directory_sites": {
        "sites": [
            {"url": "https://www.yelp.com/search?find_desc=Restaurants", "name": "Yelp"},
            {"url": "https://www.tripadvisor.com/Restaurants", "name": "TripAdvisor"},
            {"url": "https://www.indeed.com/jobs?q=software+engineer", "name": "Indeed"},
        ],
        "expected_fields": ["name", "rating", "address", "description", "url"]
    }
}

# Functions to support the Phase 7 test suite

def setup_test_suite():
    """
    Set up the test suite configuration for Phase 7 testing.
    
    This function prepares the test configurations for different website categories
    with appropriate test prompts and expected outcomes.
    
    Returns:
        Dictionary containing test suite configuration
    """
    # Initialize test suite
    test_suite = {
        "real_estate_sites": [],
        "ecommerce_sites": [],
        "content_sites": [],
        "directory_sites": []
    }
    
    # Configure real estate site tests
    for site in TEST_CATEGORIES["real_estate"]["sites"]:
        test_suite["real_estate_sites"].append({
            "site": site["name"],
            "url": site["url"],
            "prompt": "Find homes for sale with 3 bedrooms in Seattle, WA",
            "expected_fields": TEST_CATEGORIES["real_estate"]["expected_fields"],
            "min_results": 5,
            "location": "Seattle, WA"
        })
    
    # Configure e-commerce site tests
    for site in TEST_CATEGORIES["e_commerce"]["sites"]:
        test_suite["ecommerce_sites"].append({
            "site": site["name"],
            "url": site["url"],
            "prompt": "Find bestselling fiction books under $20",
            "expected_fields": TEST_CATEGORIES["e_commerce"]["expected_fields"],
            "min_results": 5,
            "product_category": "books",
            "max_price": 20
        })
    
    # Configure content site tests
    for site in TEST_CATEGORIES["content_sites"]["sites"]:
        test_suite["content_sites"].append({
            "site": site["name"],
            "url": site["url"],
            "prompt": "Find the latest technology news articles",
            "expected_fields": TEST_CATEGORIES["content_sites"]["expected_fields"],
            "min_results": 3,
            "content_type": "news"
        })
    
    # Configure directory site tests
    for site in TEST_CATEGORIES["directory_sites"]["sites"]:
        test_suite["directory_sites"].append({
            "site": site["name"],
            "url": site["url"],
            "prompt": "Find top-rated restaurants in San Francisco",
            "expected_fields": TEST_CATEGORIES["directory_sites"]["expected_fields"],
            "min_results": 5,
            "location": "San Francisco"
        })
    
    return test_suite

def run_real_estate_tests(scraper: AdaptiveScraper, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run tests for real estate websites.
    
    Args:
        scraper: Instance of AdaptiveScraper
        test_configs: List of test configuration dictionaries
        
    Returns:
        List of test result dictionaries
    """
    results = []
    logger.info(f"Running {len(test_configs)} real estate site tests")
    
    for config in test_configs:
        logger.info(f"Testing {config['site']} with prompt: '{config['prompt']}'")
        result = {
            "site": config["site"],
            "url": config["url"],
            "prompt": config["prompt"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "results_count": 0,
            "execution_time": 0,
            "field_coverage": {},
            "error": None
        }
        
        start_time = datetime.now()
        
        try:
            # Create user intent for real estate
            user_intent = {
                "description": config["prompt"],
                "target_item": "homes",
                "location": config.get("location", ""),
                "specific_criteria": {
                    "bedrooms": 3
                },
                "entity_type": "property",
                "properties": config["expected_fields"]
            }
            
            # Run the scraper
            scrape_result = asyncio.run(scraper.scrape(config["url"], user_intent=user_intent))
            
            # Extract results
            items = scrape_result.get("results", [])
            result["results_count"] = len(items)
            
            # Check success criteria
            if len(items) >= config["min_results"]:
                result["success"] = True
                
                # Check field coverage
                for field in config["expected_fields"]:
                    has_field = False
                    for item in items:
                        item_fields = set(key.lower() for key in item.keys())
                        if any(field.lower() in key for key in item_fields):
                            has_field = True
                            break
                    result["field_coverage"][field] = has_field
                
                # Calculate field coverage percentage
                covered_fields = sum(1 for v in result["field_coverage"].values() if v)
                result["field_coverage_pct"] = round((covered_fields / len(config["expected_fields"])) * 100, 2)
            
        except Exception as e:
            logger.error(f"Error testing {config['site']}: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
        
        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        results.append(result)
    
    return results

def run_ecommerce_tests(scraper: AdaptiveScraper, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run tests for e-commerce websites.
    
    Args:
        scraper: Instance of AdaptiveScraper
        test_configs: List of test configuration dictionaries
        
    Returns:
        List of test result dictionaries
    """
    results = []
    logger.info(f"Running {len(test_configs)} e-commerce site tests")
    
    for config in test_configs:
        logger.info(f"Testing {config['site']} with prompt: '{config['prompt']}'")
        result = {
            "site": config["site"],
            "url": config["url"],
            "prompt": config["prompt"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "results_count": 0,
            "execution_time": 0,
            "field_coverage": {},
            "error": None
        }
        
        start_time = datetime.now()
        
        try:
            # Create user intent for e-commerce
            user_intent = {
                "description": config["prompt"],
                "target_item": config.get("product_category", "books"),
                "specific_criteria": {
                    "max_price": config.get("max_price", 20),
                    "category": "fiction",
                    "bestselling": True
                },
                "entity_type": "product",
                "properties": config["expected_fields"]
            }
            
            # Run the scraper
            scrape_result = asyncio.run(scraper.scrape(config["url"], user_intent=user_intent))
            
            # Extract results
            items = scrape_result.get("results", [])
            result["results_count"] = len(items)
            
            # Check success criteria
            if len(items) >= config["min_results"]:
                result["success"] = True
                
                # Check field coverage
                for field in config["expected_fields"]:
                    has_field = False
                    for item in items:
                        item_fields = set(key.lower() for key in item.keys())
                        if any(field.lower() in key for key in item_fields):
                            has_field = True
                            break
                    result["field_coverage"][field] = has_field
                
                # Calculate field coverage percentage
                covered_fields = sum(1 for v in result["field_coverage"].values() if v)
                result["field_coverage_pct"] = round((covered_fields / len(config["expected_fields"])) * 100, 2)
            
        except Exception as e:
            logger.error(f"Error testing {config['site']}: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
        
        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        results.append(result)
    
    return results

def run_content_site_tests(scraper: AdaptiveScraper, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run tests for content websites.
    
    Args:
        scraper: Instance of AdaptiveScraper
        test_configs: List of test configuration dictionaries
        
    Returns:
        List of test result dictionaries
    """
    results = []
    logger.info(f"Running {len(test_configs)} content site tests")
    
    for config in test_configs:
        logger.info(f"Testing {config['site']} with prompt: '{config['prompt']}'")
        result = {
            "site": config["site"],
            "url": config["url"],
            "prompt": config["prompt"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "results_count": 0,
            "execution_time": 0,
            "field_coverage": {},
            "error": None
        }
        
        start_time = datetime.now()
        
        try:
            # Create user intent for content sites
            user_intent = {
                "description": config["prompt"],
                "target_item": "articles",
                "specific_criteria": {
                    "topic": "technology",
                    "content_type": config.get("content_type", "news"),
                    "recency": "latest"
                },
                "entity_type": "article",
                "properties": config["expected_fields"]
            }
            
            # Run the scraper
            scrape_result = asyncio.run(scraper.scrape(config["url"], user_intent=user_intent))
            
            # Extract results
            items = scrape_result.get("results", [])
            result["results_count"] = len(items)
            
            # Check success criteria
            if len(items) >= config["min_results"]:
                result["success"] = True
                
                # Check field coverage
                for field in config["expected_fields"]:
                    has_field = False
                    for item in items:
                        item_fields = set(key.lower() for key in item.keys())
                        if any(field.lower() in key for key in item_fields):
                            has_field = True
                            break
                    result["field_coverage"][field] = has_field
                
                # Calculate field coverage percentage
                covered_fields = sum(1 for v in result["field_coverage"].values() if v)
                result["field_coverage_pct"] = round((covered_fields / len(config["expected_fields"])) * 100, 2)
            
        except Exception as e:
            logger.error(f"Error testing {config['site']}: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
        
        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        results.append(result)
    
    return results

def run_directory_site_tests(scraper: AdaptiveScraper, test_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run tests for directory websites.
    
    Args:
        scraper: Instance of AdaptiveScraper
        test_configs: List of test configuration dictionaries
        
    Returns:
        List of test result dictionaries
    """
    results = []
    logger.info(f"Running {len(test_configs)} directory site tests")
    
    for config in test_configs:
        logger.info(f"Testing {config['site']} with prompt: '{config['prompt']}'")
        result = {
            "site": config["site"],
            "url": config["url"],
            "prompt": config["prompt"],
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "results_count": 0,
            "execution_time": 0,
            "field_coverage": {},
            "error": None
        }
        
        start_time = datetime.now()
        
        try:
            # Create user intent for directory sites
            user_intent = {
                "description": config["prompt"],
                "target_item": "restaurants",
                "location": config.get("location", "San Francisco"),
                "specific_criteria": {
                    "rating": "top-rated"
                },
                "entity_type": "business",
                "properties": config["expected_fields"]
            }
            
            # Run the scraper
            scrape_result = asyncio.run(scraper.scrape(config["url"], user_intent=user_intent))
            
            # Extract results
            items = scrape_result.get("results", [])
            result["results_count"] = len(items)
            
            # Check success criteria
            if len(items) >= config["min_results"]:
                result["success"] = True
                
                # Check field coverage
                for field in config["expected_fields"]:
                    has_field = False
                    for item in items:
                        item_fields = set(key.lower() for key in item.keys())
                        if any(field.lower() in key for key in item_fields):
                            has_field = True
                            break
                    result["field_coverage"][field] = has_field
                
                # Calculate field coverage percentage
                covered_fields = sum(1 for v in result["field_coverage"].values() if v)
                result["field_coverage_pct"] = round((covered_fields / len(config["expected_fields"])) * 100, 2)
            
        except Exception as e:
            logger.error(f"Error testing {config['site']}: {str(e)}")
            result["success"] = False
            result["error"] = str(e)
        
        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        results.append(result)
    
    return results

class AdaptiveScraperIntegrationTests(unittest.TestCase):
    """
    Integration tests for the AdaptiveScraper across different website types.
    
    Tests the scraper's ability to:
    1. Extract structured data from different website categories
    2. Adapt to different page layouts and structures
    3. Handle pagination and multi-page extraction
    4. Apply the right strategies for different sites
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.scraper = AdaptiveScraper()
        cls.results_dir = os.path.join(os.path.dirname(__file__), '..', 'test_results')
        os.makedirs(cls.results_dir, exist_ok=True)
        cls.test_results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": "0%",
            "start_time": datetime.now().isoformat(),
            "results": []
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.test_results["end_time"] = datetime.now().isoformat()
        if cls.test_results["total_tests"] > 0:
            cls.test_results["success_rate"] = f"{(cls.test_results['passed'] / cls.test_results['total_tests']) * 100:.1f}%"
        
        # Save test results
        with open(os.path.join(cls.results_dir, 'integration_test_results.json'), 'w') as f:
            json.dump(cls.test_results, f, indent=2, default=str)
        
        # Run metrics analysis
        analyzer = ScraperMetricsAnalyzer(results_dir=cls.results_dir)
        analyzer.generate_report(output_file=os.path.join(cls.results_dir, 'metrics_report.json'))
        analyzer.generate_charts(output_dir=os.path.join(cls.results_dir, 'charts'))
    
    def _run_test_for_site(self, category: str, site_info: Dict[str, str], expected_fields: List[str]) -> Dict[str, Any]:
        """
        Run test for a specific site.
        
        Args:
            category: Site category
            site_info: Site URL and name
            expected_fields: List of expected fields
            
        Returns:
            Dictionary with test results
        """
        start_time = datetime.now()
        url = site_info["url"]
        site_name = site_info["name"]
        
        logger.info(f"Testing {site_name} ({category}) - {url}")
        
        test_result = {
            "category": category,
            "site_name": site_name,
            "url": url,
            "success": False,
            "start_time": start_time.isoformat(),
            "execution_time": 0,
            "results_count": 0,
            "field_coverage": {field: False for field in expected_fields},
            "errors": []
        }
        
        try:
            # Create user intent for this category
            user_intent = self._create_user_intent_for_category(category)
            
            # Run the scraper
            result = asyncio.run(self.scraper.scrape(url, user_intent=user_intent))
            
            # Get the results
            items = result.get("results", [])
            test_result["results_count"] = len(items)
            
            # Check success conditions
            if len(items) > 0:
                test_result["success"] = True
                
                # Check field coverage
                all_fields = set()
                for item in items:
                    all_fields.update(item.keys())
                
                for field in expected_fields:
                    for item_field in all_fields:
                        if field.lower() in item_field.lower():
                            test_result["field_coverage"][field] = True
                            break
                            
                # Save the scrape result
                result_file = os.path.join(
                    self.results_dir, 
                    f"{site_name.lower().replace(' ', '_')}_{category}_scrape_results.json"
                )
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error testing {site_name}: {str(e)}")
            test_result["success"] = False
            test_result["errors"].append(str(e))
        
        # Calculate execution time
        end_time = datetime.now()
        test_result["end_time"] = end_time.isoformat()
        test_result["execution_time"] = (end_time - start_time).total_seconds()
        
        return test_result
    
    def _create_user_intent_for_category(self, category: str) -> Dict[str, Any]:
        """
        Create a user intent dictionary based on the site category.
        
        Args:
            category: Site category
            
        Returns:
            Dictionary with user intent
        """
        if category == "real_estate":
            return {
                "intent": "extract_listings",
                "entity_type": "property",
                "properties": ["price", "bedrooms", "bathrooms", "address", "sqft", "description"],
                "max_items": 20
            }
        elif category == "e_commerce":
            return {
                "intent": "extract_products",
                "entity_type": "product",
                "properties": ["title", "price", "image", "rating", "description", "url"],
                "max_items": 20
            }
        elif category == "content_sites":
            return {
                "intent": "extract_articles",
                "entity_type": "article",
                "properties": ["title", "author", "date", "content", "url"],
                "max_items": 20
            }
        elif category == "directory_sites":
            return {
                "intent": "extract_listings",
                "entity_type": "business",
                "properties": ["name", "rating", "address", "description", "url"],
                "max_items": 20
            }
        else:
            return {
                "intent": "extract_data",
                "properties": ["title", "description", "url"],
                "max_items": 20
            }
    
    def test_real_estate_sites(self):
        """Test scraper on real estate sites."""
        category = "real_estate"
        logger.info(f"Running {category} tests")
        
        for site_info in TEST_CATEGORIES[category]["sites"]:
            test_result = self._run_test_for_site(
                category, 
                site_info, 
                TEST_CATEGORIES[category]["expected_fields"]
            )
            
            # Update overall results
            self.test_results["total_tests"] += 1
            if test_result["success"]:
                self.test_results["passed"] += 1
            else:
                self.test_results["failed"] += 1
            
            self.test_results["results"].append({
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
                f"Failed to extract data from {site_info['name']}"
            )
    
    def test_e_commerce_sites(self):
        """Test scraper on e-commerce sites."""
        category = "e_commerce"
        logger.info(f"Running {category} tests")
        
        for site_info in TEST_CATEGORIES[category]["sites"]:
            test_result = self._run_test_for_site(
                category, 
                site_info, 
                TEST_CATEGORIES[category]["expected_fields"]
            )
            
            # Update overall results
            self.test_results["total_tests"] += 1
            if test_result["success"]:
                self.test_results["passed"] += 1
            else:
                self.test_results["failed"] += 1
            
            self.test_results["results"].append({
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
                f"Failed to extract data from {site_info['name']}"
            )
    
    def test_content_sites(self):
        """Test scraper on content sites."""
        category = "content_sites"
        logger.info(f"Running {category} tests")
        
        for site_info in TEST_CATEGORIES[category]["sites"]:
            test_result = self._run_test_for_site(
                category, 
                site_info, 
                TEST_CATEGORIES[category]["expected_fields"]
            )
            
            # Update overall results
            self.test_results["total_tests"] += 1
            if test_result["success"]:
                self.test_results["passed"] += 1
            else:
                self.test_results["failed"] += 1
            
            self.test_results["results"].append({
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
                f"Failed to extract data from {site_info['name']}"
            )
    
    def test_directory_sites(self):
        """Test scraper on directory sites."""
        category = "directory_sites"
        logger.info(f"Running {category} tests")
        
        for site_info in TEST_CATEGORIES[category]["sites"]:
            test_result = self._run_test_for_site(
                category, 
                site_info, 
                TEST_CATEGORIES[category]["expected_fields"]
            )
            
            # Update overall results
            self.test_results["total_tests"] += 1
            if test_result["success"]:
                self.test_results["passed"] += 1
            else:
                self.test_results["failed"] += 1
            
            self.test_results["results"].append({
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
                f"Failed to extract data from {site_info['name']}"
            )

def run_tests():
    """Run integration tests."""
    logger.info("Starting integration tests for AdaptiveScraper")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    logger.info("Completed integration tests")

if __name__ == "__main__":
    run_tests()
