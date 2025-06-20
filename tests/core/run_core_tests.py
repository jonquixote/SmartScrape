#!/usr/bin/env python
"""
Core Services Test Runner

This script runs all tests for the core services of SmartScrape. It adds the project
root to the Python path and runs pytest on the core test directory.

Usage:
    python3 run_core_tests.py
    # Or if made executable:
    ./run_core_tests.py
"""
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", os.path.dirname(__file__)])