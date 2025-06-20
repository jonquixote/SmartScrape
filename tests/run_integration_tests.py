"""
Run all adaptive scraper integration tests

This script runs integration tests across various website types to validate
the SmartScrape system's ability to extract content from diverse sources.
"""

import asyncio
import json
import logging
import argparse
import os
import time
from typing import Dict, List, Any, Optional

from controllers.adaptive_scraper import AdaptiveScraper, get_adaptive_scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegrationTests")

# Define test cases for different site types
TEST_CASES = [
    # E-commerce tests
    {
        "category": "e-commerce",
        "site_name": "Books to Scrape",
        "start_url": "https://books.toscrape.com/",
        "user_prompt": "Get information about science fiction books with their prices and ratings",
        "expected_fields": ["title", "price", "rating"],
        "min_results": 5
    },
    # Real estate tests
    {
        "category": "real_estate",
        "site_name": "Ohio Broker Direct (demo only)",
        "start_url": "https://www.ohiobrokerdirect.com/",
        "user_prompt": "Find houses in Columbus with 3 bedrooms",
        "expected_fields": ["address", "price", "bedrooms"],
        "min_results": 1
    },
    # Content site tests
    {
        "category": "content",
        "site_name": "Python Documentation",
        "start_url": "https://docs.python.org/3/",
        "user_prompt": "Find information about dictionaries in Python",
        "expected_fields": ["title", "content"],
        "min_results": 1
    },
    # Listing site tests
    {
        "category": "listing",
        "site_name": "Hacker News",
        "start_url": "https://news.ycombinator.com/",
        "user_prompt": "Get the latest technology news articles and their scores",
        "expected_fields": ["title", "url", "score"],
        "min_results": 5
    }
]

async def run_test_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single test case
    
    Args:
        case: Test case configuration
        
    Returns:
        Test results
    """
    logger.info(f"Running test for {case['site_name']} ({case['category']})")
    start_time = time.time()
    
    # Create adaptive scraper instance
    scraper = AdaptiveScraper(
        use_ai=True,
        max_pages=10,
        max_depth=3,
        response_cache_path="integration_test_cache"
    )
    
    try:
        # Process the user request
        job_info = await scraper.process_user_request(
            user_prompt=case["user_prompt"],
            start_urls=[case["start_url"]]
        )
        
        job_id = job_info["job_id"]
        logger.info(f"Created job {job_id} for {case['site_name']}")
        
        # Poll for job completion
        max_wait_time = 300  # 5 minutes max
        start_wait = time.time()
        completed = False
        
        while time.time() - start_wait < max_wait_time:
            job_status = scraper.get_job_status(job_id)
            logger.info(f"Job status: {job_status['status']} - Progress: {job_status['progress']}%")
            
            if job_status["status"] == "completed":
                completed = True
                break
            
            # Wait before checking again
            await asyncio.sleep(5)
        
        # Get the final results
        if completed:
            job_results = scraper.get_job_results(job_id)
            results = job_results.get("results", [])
            
            # Check if we got enough results
            has_min_results = len(results) >= case["min_results"]
            
            # Check if all expected fields are present in at least one result
            field_coverage = {field: False for field in case["expected_fields"]}
            for result in results:
                for field in case["expected_fields"]:
                    if field in result and result[field]:
                        field_coverage[field] = True
            
            all_fields_present = all(field_coverage.values())
            
            success = has_min_results and all_fields_present
            
            return {
                "case": case,
                "success": success,
                "job_id": job_id,
                "results_count": len(results),
                "field_coverage": field_coverage,
                "execution_time": time.time() - start_time,
                "has_min_results": has_min_results,
                "all_fields_present": all_fields_present
            }
        else:
            logger.error(f"Test for {case['site_name']} timed out after {max_wait_time} seconds")
            return {
                "case": case,
                "success": False,
                "job_id": job_id,
                "results_count": 0,
                "field_coverage": {field: False for field in case["expected_fields"]},
                "execution_time": time.time() - start_time,
                "has_min_results": False,
                "all_fields_present": False,
                "error": "Timeout"
            }
    except Exception as e:
        logger.error(f"Error running test for {case['site_name']}: {str(e)}")
        return {
            "case": case,
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time
        }
    finally:
        # Clean up
        await scraper.cleanup()

async def run_all_tests(categories: List[str] = None) -> Dict[str, Any]:
    """
    Run all test cases
    
    Args:
        categories: List of categories to test, or None for all
        
    Returns:
        Test results summary
    """
    logger.info(f"Starting integration tests for SmartScrape")
    
    # Filter test cases by category if specified
    test_cases = TEST_CASES
    if categories:
        test_cases = [case for case in TEST_CASES if case["category"] in categories]
    
    results = []
    for case in test_cases:
        result = await run_test_case(case)
        results.append(result)
        
        # Log result
        if result.get("success", False):
            logger.info(f"✓ Test PASSED for {case['site_name']}")
        else:
            logger.error(f"✗ Test FAILED for {case['site_name']}: {result.get('error', 'No error specified')}")
    
    # Calculate summary statistics
    succeeded = sum(1 for r in results if r.get("success", False))
    total = len(results)
    success_rate = succeeded / total if total > 0 else 0
    
    summary = {
        "total_tests": total,
        "passed": succeeded,
        "failed": total - succeeded,
        "success_rate": f"{success_rate:.0%}",
        "results": results
    }
    
    # Log summary
    logger.info(f"Test Summary: {succeeded}/{total} passed ({success_rate:.0%})")
    
    # Write results to file
    with open("integration_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Run the integration tests"""
    parser = argparse.ArgumentParser(description="Run integration tests for SmartScrape")
    parser.add_argument("--categories", type=str, nargs="+", 
                      help="Categories to test (e-commerce, real_estate, content, listing)")
    args = parser.parse_args()
    
    # Create cache directory
    os.makedirs("integration_test_cache", exist_ok=True)
    
    # Run tests
    asyncio.run(run_all_tests(args.categories))

if __name__ == "__main__":
    main()