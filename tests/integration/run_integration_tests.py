#!/usr/bin/env python
"""
Integration test runner for SmartScrape.

This script runs all integration tests for the SmartScrape core infrastructure.
"""
import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", os.path.dirname(__file__)])