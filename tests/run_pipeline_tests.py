#!/usr/bin/env python3
"""
Pipeline Architecture Test Runner

This script runs all pipeline-related tests, including integration tests,
performance benchmarks, and resilience tests. It generates test reports,
measures code coverage, and performs performance analysis.
"""

import os
import sys
import time
import argparse
import unittest
import json
import coverage
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline_tests')


def setup_test_environment():
    """Set up the test environment."""
    logger.info("Setting up test environment")
    
    # Create necessary directories
    os.makedirs("tests/results", exist_ok=True)
    os.makedirs("tests/results/reports", exist_ok=True)
    os.makedirs("tests/results/coverage", exist_ok=True)
    os.makedirs("tests/results/benchmarks", exist_ok=True)
    
    # Initialize environment variables for testing
    os.environ["PIPELINE_TEST_MODE"] = "True"
    os.environ["PIPELINE_TEST_FIXTURES"] = str(Path("tests/fixtures/pipelines").resolve())
    
    logger.info("Test environment set up complete")


def discover_tests(test_pattern=None):
    """Discover all pipeline tests."""
    logger.info("Discovering pipeline tests")
    
    test_loader = unittest.TestLoader()
    
    if test_pattern:
        logger.info(f"Using test pattern: {test_pattern}")
        test_suite = test_loader.discover("tests/core/pipeline", pattern=test_pattern)
    else:
        # Load all pipeline tests
        test_suite = unittest.TestSuite()
        
        # Integration tests
        integration_tests = test_loader.discover("tests/core/pipeline", pattern="test_integration.py")
        test_suite.addTest(integration_tests)
        
        # Performance tests
        performance_tests = test_loader.discover("tests/core/pipeline", pattern="test_performance.py")
        test_suite.addTest(performance_tests)
        
        # Resilience tests
        resilience_tests = test_loader.discover("tests/core/pipeline", pattern="test_resilience.py")
        test_suite.addTest(resilience_tests)
        
        # Other pipeline tests
        other_tests = test_loader.discover("tests/core/pipeline", pattern="test_*.py")
        for test in other_tests:
            # Only add tests that haven't already been added
            test_name = test.id().split('.')[-1]
            if not any(test_name in t.id() for t in test_suite):
                test_suite.addTest(test)
    
    logger.info(f"Discovered {test_suite.countTestCases()} test cases")
    return test_suite


def run_tests(test_suite, xml_report=True, html_report=True):
    """Run the test suite and generate reports."""
    logger.info("Running pipeline tests")
    
    start_time = time.time()
    
    # Set up test runner
    if xml_report:
        import xmlrunner
        test_runner = xmlrunner.XMLTestRunner(output="tests/results/reports", verbosity=2)
    else:
        test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = test_runner.run(test_suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Generate summary
    summary = {
        "total": test_suite.countTestCases(),
        "run": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "skipped": len(getattr(result, 'skipped', [])),
        "success": result.wasSuccessful(),
        "execution_time": execution_time
    }
    
    # Save summary to file
    with open("tests/results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report if requested
    if html_report:
        try:
            from junit2htmlreport import runner as html_runner
            html_runner.run("tests/results/reports", "tests/results/pipeline_test_report.html")
        except ImportError:
            logger.warning("junit2htmlreport not installed. Skipping HTML report generation.")
    
    logger.info(f"Tests completed in {execution_time:.2f} seconds")
    logger.info(f"Ran {result.testsRun} tests with {len(result.errors)} errors and {len(result.failures)} failures")
    
    return result


def measure_code_coverage(test_suite):
    """Measure code coverage for the tests."""
    logger.info("Measuring code coverage")
    
    # Initialize coverage measurement
    cov = coverage.Coverage(
        source=["core/pipeline"],
        omit=["*/tests/*", "*/virtualenv/*", "*/venv/*"],
        config_file=False
    )
    
    # Start coverage measurement
    cov.start()
    
    # Run tests
    unittest.TextTestRunner(verbosity=0).run(test_suite)
    
    # Stop coverage measurement
    cov.stop()
    cov.save()
    
    # Generate coverage reports
    logger.info("Generating coverage reports")
    cov.html_report(directory="tests/results/coverage/html")
    cov.xml_report(outfile="tests/results/coverage/coverage.xml")
    
    # Get coverage percentage
    coverage_percentage = cov.report()
    logger.info(f"Overall code coverage: {coverage_percentage:.2f}%")
    
    return coverage_percentage


def run_benchmark_tests():
    """Run performance benchmark tests."""
    logger.info("Running benchmark tests")
    
    import tests.core.pipeline.test_performance as benchmark_module
    
    # Create a test suite just for benchmarks
    benchmark_suite = unittest.TestLoader().loadTestsFromModule(benchmark_module)
    
    # Run benchmarks
    start_time = time.time()
    unittest.TextTestRunner(verbosity=1).run(benchmark_suite)
    end_time = time.time()
    
    benchmark_time = end_time - start_time
    logger.info(f"Benchmark tests completed in {benchmark_time:.2f} seconds")
    
    # The actual benchmark results are output by the benchmark tests themselves
    return benchmark_time


def generate_final_report(test_results, coverage_percentage, benchmark_time):
    """Generate a final comprehensive report."""
    logger.info("Generating final report")
    
    report = {
        "report_generated": datetime.now().isoformat(),
        "test_summary": {
            "total_tests": test_results.testsRun,
            "errors": len(test_results.errors),
            "failures": len(test_results.failures),
            "skipped": len(getattr(test_results, 'skipped', [])),
            "success": test_results.wasSuccessful()
        },
        "coverage": {
            "percentage": coverage_percentage,
            "reports_location": "tests/results/coverage/"
        },
        "benchmarks": {
            "execution_time": benchmark_time,
            "detailed_results": "tests/results/benchmarks/"
        }
    }
    
    # Add error details to report
    if test_results.errors:
        report["test_summary"]["error_details"] = [
            {"test": str(test), "error": str(error)} 
            for test, error in test_results.errors
        ]
    
    if test_results.failures:
        report["test_summary"]["failure_details"] = [
            {"test": str(test), "failure": str(failure)} 
            for test, failure in test_results.failures
        ]
    
    # Save report to file
    with open("tests/results/pipeline_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Final report saved to tests/results/pipeline_test_report.json")
    
    return report


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run pipeline architecture tests')
    parser.add_argument('--pattern', help='Test file pattern to run specific tests')
    parser.add_argument('--skip-coverage', action='store_true', help='Skip code coverage measurement')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip performance benchmarks')
    parser.add_argument('--no-xml', action='store_true', help='Do not generate XML reports')
    parser.add_argument('--no-html', action='store_true', help='Do not generate HTML reports')
    
    args = parser.parse_args()
    
    # Set up test environment
    setup_test_environment()
    
    # Discover tests
    test_suite = discover_tests(args.pattern)
    
    # Run tests and generate reports
    test_results = run_tests(test_suite, not args.no_xml, not args.no_html)
    
    coverage_percentage = 0
    if not args.skip_coverage:
        # Measure code coverage
        coverage_percentage = measure_code_coverage(test_suite)
    
    benchmark_time = 0
    if not args.skip_benchmarks:
        # Run benchmark tests
        benchmark_time = run_benchmark_tests()
    
    # Generate final report
    final_report = generate_final_report(test_results, coverage_percentage, benchmark_time)
    
    # Print summary
    print("\n== Pipeline Test Suite Summary ==")
    print(f"Total tests: {final_report['test_summary']['total_tests']}")
    print(f"Errors: {final_report['test_summary']['errors']}")
    print(f"Failures: {final_report['test_summary']['failures']}")
    print(f"Success: {final_report['test_summary']['success']}")
    print(f"Code coverage: {final_report['coverage']['percentage']:.2f}%")
    print(f"Benchmark time: {final_report['benchmarks']['execution_time']:.2f}s")
    print(f"Detailed report: tests/results/pipeline_test_report.json")
    
    # Return exit code
    return 0 if test_results.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())