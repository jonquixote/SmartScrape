# Add these imports at the top
import os
import sys

# Before any other imports, add the SmartScrape root directory to the Python path
project_root = "/Users/johnny/Downloads/SmartScrape"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

"""
Phase 7 Integration Tests for SmartScrape

This script runs a comprehensive set of tests to evaluate the adaptive scraper's
performance across different website categories, collects metrics, and
triggers the continuous improvement system.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from controllers.adaptive_scraper import AdaptiveScraper
from utils.metrics_analyzer import ScraperMetricsAnalyzer
from utils.continuous_improvement import ContinuousImprovementSystem
from tests.test_adaptive_scraper import (
    setup_test_suite,
    run_real_estate_tests,
    run_ecommerce_tests,
    run_content_site_tests,
    run_directory_site_tests
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'phase7_tests.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Phase7Tests")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Phase 7 integration tests")
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=["all", "real_estate", "ecommerce", "content", "directory"],
        default=["all"],
        help="Website categories to test"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to store test results"
    )
    
    parser.add_argument(
        "--improve",
        action="store_true",
        help="Run continuous improvement after tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def setup_test_environment(output_dir: str) -> Dict[str, Any]:
    """Set up the test environment and return test configuration"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize test environment
    test_environment = {
        "start_time": datetime.datetime.now(),
        "output_dir": output_dir,
        "results_file": os.path.join(output_dir, f"phase7_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
        "metrics_file": os.path.join(output_dir, f"phase7_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
        "report_file": os.path.join(output_dir, f"phase7_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
        "log_file": os.path.join(output_dir, f"phase7_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        "categories": {
            "real_estate": {
                "sites": ["zillow.com", "redfin.com", "realtor.com"],
                "fields": ["price", "address", "bedrooms", "bathrooms", "sqft"]
            },
            "ecommerce": {
                "sites": ["books.toscrape.com", "amazon.com", "etsy.com"],
                "fields": ["title", "price", "image", "description", "rating"]
            },
            "content": {
                "sites": ["news.ycombinator.com", "bbc.com", "techcrunch.com"],
                "fields": ["title", "author", "date", "content", "comments"]
            },
            "directory": {
                "sites": ["yelp.com", "tripadvisor.com", "indeed.com"],
                "fields": ["name", "address", "phone", "category", "rating"]
            }
        }
    }
    
    return test_environment

def run_tests(categories: List[str], test_environment: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """Run tests for specified categories and return results"""
    # Initialize result container
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "categories_tested": categories,
        "test_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "category_results": {},
        "detailed_results": []
    }
    
    # Initialize the scraper
    scraper = AdaptiveScraper()
    
    # Set up the test suite
    test_suite = setup_test_suite()
    
    # Run tests for each category
    if "all" in categories or "real_estate" in categories:
        logger.info("Running real estate site tests...")
        real_estate_results = run_real_estate_tests(scraper, test_suite["real_estate_sites"])
        results["category_results"]["real_estate"] = summarize_category_results(real_estate_results)
        results["detailed_results"].extend(real_estate_results)
        results["test_count"] += len(real_estate_results)
        results["success_count"] += sum(1 for r in real_estate_results if r["success"])
        results["failure_count"] += sum(1 for r in real_estate_results if not r["success"])
        
        if verbose:
            print_category_results("Real Estate", real_estate_results)
    
    if "all" in categories or "ecommerce" in categories:
        logger.info("Running e-commerce site tests...")
        ecommerce_results = run_ecommerce_tests(scraper, test_suite["ecommerce_sites"])
        results["category_results"]["ecommerce"] = summarize_category_results(ecommerce_results)
        results["detailed_results"].extend(ecommerce_results)
        results["test_count"] += len(ecommerce_results)
        results["success_count"] += sum(1 for r in ecommerce_results if r["success"])
        results["failure_count"] += sum(1 for r in ecommerce_results if not r["success"])
        
        if verbose:
            print_category_results("E-commerce", ecommerce_results)
    
    if "all" in categories or "content" in categories:
        logger.info("Running content site tests...")
        content_results = run_content_site_tests(scraper, test_suite["content_sites"])
        results["category_results"]["content"] = summarize_category_results(content_results)
        results["detailed_results"].extend(content_results)
        results["test_count"] += len(content_results)
        results["success_count"] += sum(1 for r in content_results if r["success"])
        results["failure_count"] += sum(1 for r in content_results if not r["success"])
        
        if verbose:
            print_category_results("Content Sites", content_results)
    
    if "all" in categories or "directory" in categories:
        logger.info("Running directory site tests...")
        directory_results = run_directory_site_tests(scraper, test_suite["directory_sites"])
        results["category_results"]["directory"] = summarize_category_results(directory_results)
        results["detailed_results"].extend(directory_results)
        results["test_count"] += len(directory_results)
        results["success_count"] += sum(1 for r in directory_results if r["success"])
        results["failure_count"] += sum(1 for r in directory_results if not r["success"])
        
        if verbose:
            print_category_results("Directory Sites", directory_results)
    
    # Calculate overall success rate
    if results["test_count"] > 0:
        results["success_rate"] = round((results["success_count"] / results["test_count"]) * 100, 2)
    else:
        results["success_rate"] = 0
    
    # Save results to file
    with open(test_environment["results_file"], "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Test results saved to {test_environment['results_file']}")
    
    return results

def summarize_category_results(category_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize test results for a category"""
    total_tests = len(category_results)
    if total_tests == 0:
        return {
            "total_tests": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": "0%",
            "avg_processing_time": 0
        }
    
    success_count = sum(1 for r in category_results if r["success"])
    success_rate = round((success_count / total_tests) * 100, 2)
    
    # Calculate average processing time
    processing_times = [r.get("processing_time", 0) for r in category_results]
    avg_processing_time = round(sum(processing_times) / len(processing_times), 2) if processing_times else 0
    
    return {
        "total_tests": total_tests,
        "success_count": success_count,
        "failure_count": total_tests - success_count,
        "success_rate": f"{success_rate}%",
        "avg_processing_time": avg_processing_time
    }

def print_category_results(category_name: str, results: List[Dict[str, Any]]):
    """Print test results for a category"""
    print(f"\n===== {category_name} Results =====")
    
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failure = total - success
    
    print(f"Total tests: {total}")
    print(f"Successful: {success} ({round((success/total)*100, 2)}%)")
    print(f"Failed: {failure} ({round((failure/total)*100, 2)}%)")
    
    print("\nDetailed Results:")
    for i, result in enumerate(results):
        status = "✅ SUCCESS" if result["success"] else "❌ FAILURE"
        site = result.get("site", "Unknown")
        print(f"{i+1}. {site}: {status}")
        if not result["success"]:
            print(f"   - Error: {result.get('error', 'Unknown error')}")

def analyze_metrics(results: Dict[str, Any], test_environment: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze test results and generate metrics"""
    logger.info("Analyzing test metrics...")
    
    # Initialize metrics analyzer
    analyzer = ScraperMetricsAnalyzer()
    
    # Generate metrics from test results
    metrics = analyzer.analyze_test_results(results["detailed_results"])
    
    # Add overall success rate to metrics
    metrics["overall_success_rate"] = f"{results['success_rate']}%"
    
    # Add category-specific metrics
    metrics["by_category"] = {}
    for category, category_results in results["category_results"].items():
        metrics["by_category"][category] = category_results
    
    # Add extraction quality metrics
    metrics["extraction_quality"] = analyzer.calculate_extraction_quality(results["detailed_results"])
    
    # Add field coverage metrics
    metrics["field_coverage"] = analyzer.calculate_field_coverage(results["detailed_results"])
    
    # Save metrics to file
    with open(test_environment["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to {test_environment['metrics_file']}")
    
    # Generate HTML report
    report_html = analyzer.generate_html_report(
        results=results,
        metrics=metrics,
        title="Phase 7 Test Report",
        description="Comprehensive test results for the adaptive scraper"
    )
    
    with open(test_environment["report_file"], "w") as f:
        f.write(report_html)
    
    logger.info(f"HTML report saved to {test_environment['report_file']}")
    
    return metrics

def run_continuous_improvement(test_environment: Dict[str, Any], results: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Run the continuous improvement system"""
    logger.info("Running continuous improvement system...")
    
    # Initialize continuous improvement system
    improvement_system = ContinuousImprovementSystem(
        results_dir=test_environment["output_dir"]
    )
    
    # Create a metrics report file for the improvement system to analyze
    metrics_report_file = os.path.join(
        test_environment["output_dir"],
        f"metrics_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    # Combine results and metrics for the report
    metrics_report = {
        "test_results": results,
        "test_metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat(),
        "log_metrics": {
            "error_counts": {
                "connection_error": sum(1 for r in results["detailed_results"] if "connection" in str(r.get("error", "")).lower()),
                "timeout_error": sum(1 for r in results["detailed_results"] if "timeout" in str(r.get("error", "")).lower()),
                "extraction_error": sum(1 for r in results["detailed_results"] if "extraction" in str(r.get("error", "")).lower()),
                "validation_error": sum(1 for r in results["detailed_results"] if "validation" in str(r.get("error", "")).lower())
            }
        }
    }
    
    # Save metrics report
    with open(metrics_report_file, "w") as f:
        json.dump(metrics_report, f, indent=2, default=str)
    
    logger.info(f"Metrics report saved to {metrics_report_file}")
    
    # Analyze test results and generate improvement suggestions
    analysis = improvement_system.analyze_test_results()
    
    # Generate improvements
    improvements = improvement_system.generate_improvements()
    
    # Apply improvements
    improvement_results = improvement_system.apply_improvements(improvements)
    
    logger.info(f"Continuous improvement complete. Applied {len(improvement_results['applied_pattern_improvements'])} pattern improvements")
    
    return improvement_results

def main():
    """Main function to run Phase 7 tests"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up test environment
    test_environment = setup_test_environment(args.output_dir)
    
    # Display test configuration
    logger.info(f"Running Phase 7 tests for categories: {', '.join(args.categories)}")
    logger.info(f"Output directory: {test_environment['output_dir']}")
    
    # Run tests
    start_time = time.time()
    results = run_tests(args.categories, test_environment, args.verbose)
    test_duration = time.time() - start_time
    
    # Display summary results
    print("\n===== Test Summary =====")
    print(f"Total tests: {results['test_count']}")
    print(f"Successful: {results['success_count']} ({results['success_rate']}%)")
    print(f"Failed: {results['failure_count']}")
    print(f"Test duration: {round(test_duration, 2)} seconds")
    
    # Analyze metrics
    metrics = analyze_metrics(results, test_environment)
    
    # Display metrics summary
    print("\n===== Metrics Summary =====")
    print(f"Overall success rate: {metrics['overall_success_rate']}")
    print("Success rate by category:")
    for category, category_metrics in metrics["by_category"].items():
        print(f"- {category.replace('_', ' ').title()}: {category_metrics['success_rate']}")
    
    # Run continuous improvement if requested
    if args.improve:
        print("\n===== Running Continuous Improvement =====")
        improvement_results = run_continuous_improvement(test_environment, results, metrics)
        
        # Display improvement results
        print(f"Applied {len(improvement_results['applied_pattern_improvements'])} pattern improvements")
        print(f"Applied {len(improvement_results['applied_strategy_improvements'])} strategy improvements")
        print(f"Applied {len(improvement_results['applied_error_handling_improvements'])} error handling improvements")
    
    print(f"\nTest results saved to: {test_environment['results_file']}")
    print(f"Metrics saved to: {test_environment['metrics_file']}")
    print(f"HTML report saved to: {test_environment['report_file']}")

if __name__ == "__main__":
    main()