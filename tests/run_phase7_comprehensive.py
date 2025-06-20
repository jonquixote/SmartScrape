#!/usr/bin/env python3
"""
Comprehensive test runner for SmartScrape Phase 7 testing.

Executes unit tests, integration tests, and performance tests with
detailed reporting and coverage analysis.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner for Phase 7."""
    
    def __init__(self, verbose: bool = True, generate_coverage: bool = True):
        self.verbose = verbose
        self.generate_coverage = generate_coverage
        self.test_results = {}
        self.start_time = time.time()
        
        # Test directories
        self.test_dirs = {
            'unit_components': 'tests/components/',
            'unit_controllers': 'tests/controllers/', 
            'unit_strategies': 'tests/strategies/',
            'integration': 'tests/integration/',
            'performance': 'tests/performance/'
        }
        
        # Specific test files for Phase 7
        self.test_files = {
            'unit_components': [
                'test_universal_intent_analyzer.py',
                'test_intelligent_url_generator.py',
                'test_ai_schema_generator.py', 
                'test_content_quality_scorer.py'
            ],
            'unit_controllers': [
                'test_extraction_coordinator.py'
            ],
            'unit_strategies': [
                'test_universal_crawl4ai_strategy.py',
                'test_composite_universal_strategy.py'
            ],
            'integration': [
                'test_new_component_integration.py'
            ],
            'performance': [
                'test_component_performance.py'
            ]
        }
    
    def print_header(self, text: str):
        """Print a formatted header."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  {text}")
            print(f"{'='*60}")
    
    def print_subheader(self, text: str):
        """Print a formatted subheader."""
        if self.verbose:
            print(f"\n{'-'*40}")
            print(f"  {text}")
            print(f"{'-'*40}")
    
    def run_test_suite(self, suite_name: str, test_dir: str, test_files: List[str]) -> Dict[str, Any]:
        """Run a specific test suite."""
        self.print_subheader(f"Running {suite_name} Tests")
        
        suite_results = {
            'suite_name': suite_name,
            'total_files': len(test_files),
            'passed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'execution_time': 0,
            'file_results': {}
        }
        
        suite_start_time = time.time()
        
        for test_file in test_files:
            test_path = os.path.join(test_dir, test_file)
            
            if not os.path.exists(test_path):
                print(f"  ⚠️  SKIPPED: {test_file} (file not found)")
                suite_results['skipped_files'] += 1
                suite_results['file_results'][test_file] = {
                    'status': 'SKIPPED',
                    'reason': 'File not found'
                }
                continue
            
            print(f"  🧪 Running: {test_file}")
            
            # Build pytest command
            cmd = [
                'python', '-m', 'pytest',
                test_path,
                '-v',
                '--tb=short',
                '--no-header'
            ]
            
            if self.generate_coverage:
                cmd.extend([
                    '--cov=.',
                    '--cov-report=term-missing',
                    '--cov-append'
                ])
            
            try:
                # Run the test
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per test file
                )
                
                if result.returncode == 0:
                    print(f"    ✅ PASSED: {test_file}")
                    suite_results['passed_files'] += 1
                    suite_results['file_results'][test_file] = {
                        'status': 'PASSED',
                        'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                    }
                else:
                    print(f"    ❌ FAILED: {test_file}")
                    suite_results['failed_files'] += 1
                    suite_results['file_results'][test_file] = {
                        'status': 'FAILED',
                        'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
                        'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                    }
                    
                    if self.verbose:
                        print(f"    Error output: {result.stderr[:200]}...")
            
            except subprocess.TimeoutExpired:
                print(f"    ⏰ TIMEOUT: {test_file}")
                suite_results['failed_files'] += 1
                suite_results['file_results'][test_file] = {
                    'status': 'TIMEOUT',
                    'reason': 'Test execution exceeded 5 minutes'
                }
            
            except Exception as e:
                print(f"    💥 ERROR: {test_file} - {str(e)}")
                suite_results['failed_files'] += 1
                suite_results['file_results'][test_file] = {
                    'status': 'ERROR',
                    'reason': str(e)
                }
        
        suite_end_time = time.time()
        suite_results['execution_time'] = suite_end_time - suite_start_time
        
        # Print suite summary
        total_tests = suite_results['total_files']
        passed = suite_results['passed_files']
        failed = suite_results['failed_files']
        skipped = suite_results['skipped_files']
        
        print(f"\n  📊 {suite_name} Summary:")
        print(f"    Total: {total_tests}")
        print(f"    Passed: {passed} ✅")
        print(f"    Failed: {failed} ❌")
        print(f"    Skipped: {skipped} ⚠️")
        print(f"    Time: {suite_results['execution_time']:.2f}s")
        
        return suite_results
    
    def run_all_tests(self):
        """Run all test suites."""
        self.print_header("SmartScrape Phase 7 - Comprehensive Test Suite")
        
        print(f"🚀 Starting comprehensive test execution...")
        print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run each test suite
        for suite_key, test_dir in self.test_dirs.items():
            if suite_key in self.test_files:
                suite_results = self.run_test_suite(
                    suite_key.replace('_', ' ').title(),
                    test_dir,
                    self.test_files[suite_key]
                )
                self.test_results[suite_key] = suite_results
        
        # Generate final summary
        self.generate_final_summary()
    
    def generate_final_summary(self):
        """Generate and display final test summary."""
        self.print_header("Final Test Summary")
        
        total_execution_time = time.time() - self.start_time
        
        # Aggregate results
        total_files = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for suite_name, results in self.test_results.items():
            total_files += results['total_files']
            total_passed += results['passed_files']
            total_failed += results['failed_files']
            total_skipped += results['skipped_files']
            
            print(f"📋 {results['suite_name']}:")
            print(f"   Passed: {results['passed_files']}/{results['total_files']} ✅")
            print(f"   Failed: {results['failed_files']}/{results['total_files']} ❌")
            print(f"   Time: {results['execution_time']:.2f}s")
        
        # Overall summary
        success_rate = (total_passed / total_files * 100) if total_files > 0 else 0
        
        print(f"\n🎯 Overall Results:")
        print(f"   Total Test Files: {total_files}")
        print(f"   Passed: {total_passed} ✅")
        print(f"   Failed: {total_failed} ❌")
        print(f"   Skipped: {total_skipped} ⚠️")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Total Execution Time: {total_execution_time:.2f}s")
        
        # Determine overall status
        if total_failed == 0 and total_passed > 0:
            print(f"\n🎉 ALL TESTS PASSED! Phase 7 Testing Complete! 🎉")
            exit_code = 0
        elif total_failed > 0:
            print(f"\n⚠️  SOME TESTS FAILED - Review failed tests above ⚠️")
            exit_code = 1
        else:
            print(f"\n❓ NO TESTS EXECUTED - Check test file paths ❓")
            exit_code = 2
        
        # Save detailed results to file
        self.save_detailed_results(total_execution_time, success_rate)
        
        return exit_code
    
    def save_detailed_results(self, total_time: float, success_rate: float):
        """Save detailed test results to JSON file."""
        results_data = {
            'phase': 'Phase 7 - Testing and Validation',
            'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_execution_time': total_time,
            'success_rate': success_rate,
            'test_suites': self.test_results,
            'summary': {
                'total_files': sum(r['total_files'] for r in self.test_results.values()),
                'total_passed': sum(r['passed_files'] for r in self.test_results.values()),
                'total_failed': sum(r['failed_files'] for r in self.test_results.values()),
                'total_skipped': sum(r['skipped_files'] for r in self.test_results.values())
            }
        }
        
        results_file = 'tests/phase7_test_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"\n📄 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save results file: {e}")
    
    def run_specific_suite(self, suite_name: str):
        """Run a specific test suite."""
        if suite_name not in self.test_dirs:
            print(f"❌ Unknown test suite: {suite_name}")
            print(f"Available suites: {list(self.test_dirs.keys())}")
            return 1
        
        if suite_name not in self.test_files:
            print(f"❌ No test files defined for suite: {suite_name}")
            return 1
        
        self.print_header(f"Running {suite_name} Test Suite")
        
        suite_results = self.run_test_suite(
            suite_name.replace('_', ' ').title(),
            self.test_dirs[suite_name],
            self.test_files[suite_name]
        )
        
        self.test_results[suite_name] = suite_results
        return self.generate_final_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SmartScrape Phase 7 Test Runner')
    parser.add_argument('--suite', type=str, help='Run specific test suite only')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    parser.add_argument('--no-coverage', action='store_true', help='Skip coverage reporting')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(
        verbose=not args.quiet,
        generate_coverage=not args.no_coverage
    )
    
    try:
        if args.suite:
            exit_code = runner.run_specific_suite(args.suite)
        else:
            exit_code = runner.run_all_tests()
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
