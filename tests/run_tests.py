"""
Test runner for SmartScrape

This module provides a convenient way to run all or selected tests.
"""

import os
import sys
import unittest
import argparse

# Add the project root directory to Python path for imports to work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # Using insert(0) to prioritize our project path

# Print the Python path for debugging
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Project root added to path: {PROJECT_ROOT}")

# Try to import key modules to verify they're accessible
try:
    import crawl4ai
    print(f"Successfully imported crawl4ai")
except ImportError as e:
    print(f"Error importing crawl4ai: {e}")

# Import BaseStrategy directly using the correct path
try:
    from strategies.base_strategy import BaseStrategy
    print(f"Successfully imported BaseStrategy")
except ImportError as e:
    print(f"Error importing BaseStrategy: {e}")
    # List available modules in strategies directory for debugging
    strategies_dir = os.path.join(PROJECT_ROOT, 'strategies')
    if os.path.exists(strategies_dir):
        print(f"Contents of strategies directory:")
        for f in os.listdir(strategies_dir):
            print(f"  - {f}")


def run_all_tests():
    """Run all test cases"""
    # Start discovery from the tests directory 
    tests = unittest.TestLoader().discover(os.path.dirname(__file__), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return result.wasSuccessful()


def run_strategy_tests():
    """Run only strategy tests"""
    # Use absolute path for strategies directory
    strategy_test_dir = os.path.join(os.path.dirname(__file__), 'strategies')
    print(f"Looking for tests in: {strategy_test_dir}")
    if not os.path.exists(strategy_test_dir):
        print(f"Error: Directory {strategy_test_dir} does not exist")
        return False
    
    files = [f for f in os.listdir(strategy_test_dir) if f.startswith('test_') and f.endswith('.py')]
    if not files:
        print(f"No test files found in {strategy_test_dir}")
    else:
        print(f"Found test files: {files}")
    
    tests = unittest.TestLoader().discover(strategy_test_dir, pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return result.wasSuccessful()


def run_single_test(test_name):
    """Run a single test file"""
    if not test_name.startswith('test_'):
        test_name = f'test_{test_name}'
    if not test_name.endswith('.py'):
        test_name = f'{test_name}.py'
    
    # Look in multiple directories for the test
    for root, _, files in os.walk(os.path.dirname(__file__)):
        if test_name in files:
            test_path = os.path.join(root, test_name)
            print(f"Running test: {test_path}")
            # Use loader to load the test module
            loader = unittest.TestLoader()
            test_suite = loader.discover(os.path.dirname(test_path), pattern=test_name)
            result = unittest.TextTestRunner(verbosity=2).run(test_suite)
            return result.wasSuccessful()
    
    print(f"Test '{test_name}' not found.")
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SmartScrape tests')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='Run all tests')
    group.add_argument('--strategies', action='store_true', help='Run strategy tests')
    group.add_argument('--test', type=str, help='Run a specific test file')
    
    args = parser.parse_args()
    
    if args.all or (not args.strategies and not args.test):
        # If no specific option is given, run all tests
        success = run_all_tests()
    elif args.strategies:
        success = run_strategy_tests()
    elif args.test:
        success = run_single_test(args.test)
    
    # Return exit code based on test success
    sys.exit(0 if success else 1)