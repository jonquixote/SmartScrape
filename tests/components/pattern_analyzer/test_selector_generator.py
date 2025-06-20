"""
Test Selector Generator

This module contains comprehensive tests for the SelectorGenerator class 
that handles generating, testing, and optimizing CSS selectors for web scraping.
"""

import pytest
import os
import sys
from bs4 import BeautifulSoup, Tag
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

# Ensure the module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from components.pattern_analyzer.selector_generator import SelectorGenerator
from components.pattern_analyzer.base_analyzer import get_registry


# Sample HTML for testing
SIMPLE_HTML = """
<html>
<body>
    <div id="main">
        <div class="container">
            <div id="product-list">
                <div class="product" data-id="1">
                    <h2 class="title">Product 1</h2>
                    <span class="price">$10.00</span>
                    <div class="description">Description 1</div>
                </div>
                <div class="product" data-id="2">
                    <h2 class="title">Product 2</h2>
                    <span class="price">$20.00</span>
                    <div class="description">Description 2</div>
                </div>
                <div class="product" data-id="3">
                    <h2 class="title">Product 3</h2>
                    <span class="price">$30.00</span>
                    <div class="description">Description 3</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

COMPLEX_HTML = """
<html>
<body>
    <div id="main">
        <div class="container">
            <div id="results">
                <div class="result-item" data-testid="result-1">
                    <div class="header">
                        <h3 class="title">Result 1</h3>
                        <span class="badge">New</span>
                    </div>
                    <div class="content">
                        <p>Description for result 1</p>
                        <ul class="details">
                            <li class="detail-item">Feature 1</li>
                            <li class="detail-item">Feature 2</li>
                        </ul>
                        <div class="price-container">
                            <span class="price">$100</span>
                            <span class="original-price">$120</span>
                        </div>
                    </div>
                    <div class="footer">
                        <button class="action-btn">Add to cart</button>
                    </div>
                </div>
                <div class="result-item" data-testid="result-2">
                    <div class="header">
                        <h3 class="title">Result 2</h3>
                    </div>
                    <div class="content">
                        <p>Description for result 2</p>
                        <ul class="details">
                            <li class="detail-item">Feature A</li>
                            <li class="detail-item">Feature B</li>
                        </ul>
                        <div class="price-container">
                            <span class="price">$200</span>
                        </div>
                    </div>
                    <div class="footer">
                        <button class="action-btn">Add to cart</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

VARIED_HTML = """
<html>
<body>
    <div id="catalog">
        <section class="category">
            <h2>Category A</h2>
            <div class="items">
                <article class="item" data-sku="a1">
                    <h3>Item A1</h3>
                    <div class="price">$15.99</div>
                </article>
                <article class="item" data-sku="a2">
                    <h3>Item A2</h3>
                    <div class="price">$25.99</div>
                </article>
            </div>
        </section>
        <section class="category">
            <h2>Category B</h2>
            <div class="items">
                <article class="item" data-sku="b1">
                    <h3>Item B1</h3>
                    <div class="price">$35.99</div>
                </article>
            </div>
        </section>
    </div>
</body>
</html>
"""


@pytest.fixture
def selector_generator():
    """Create a SelectorGenerator instance for testing."""
    return SelectorGenerator()


@pytest.fixture
def simple_soup():
    """Create a BeautifulSoup object from simple HTML."""
    return BeautifulSoup(SIMPLE_HTML, 'html.parser')


@pytest.fixture
def complex_soup():
    """Create a BeautifulSoup object from complex HTML."""
    return BeautifulSoup(COMPLEX_HTML, 'html.parser')


@pytest.fixture
def varied_soup():
    """Create a BeautifulSoup object from varied HTML."""
    return BeautifulSoup(VARIED_HTML, 'html.parser')


class TestSelectorGenerator:
    """Test suite for the SelectorGenerator class."""

    def test_init(self):
        """Test that the SelectorGenerator initializes with correct default values."""
        generator = SelectorGenerator()
        assert generator.stability_threshold == 0.7
        assert len(generator.selector_strategies) == 5
        assert 'container' in generator.common_classes

    def test_generate_selector_with_id(self, selector_generator, simple_soup):
        """Test generating a selector for an element with ID."""
        product_list = simple_soup.select_one('#product-list')
        selector = selector_generator.generate_selector(product_list, SIMPLE_HTML)
        assert selector == '#product-list'
        
        # Verify the selector works
        matches = simple_soup.select(selector)
        assert len(matches) == 1
        assert matches[0] == product_list

    def test_generate_selector_with_class(self, selector_generator, simple_soup):
        """Test generating a selector for an element with class."""
        product = simple_soup.select_one('.product')
        selector = selector_generator.generate_selector(product, SIMPLE_HTML)
        
        # Should prefer data-id attribute over generic class name
        assert 'data-id' in selector or '.product' in selector
        
        # Verify the selector works
        matches = simple_soup.select(selector)
        assert len(matches) > 0
        assert product in matches

    def test_generate_selector_with_attribute(self, selector_generator, simple_soup):
        """Test generating a selector for an element with a data attribute."""
        product = simple_soup.select_one('[data-id="2"]')
        selector = selector_generator.generate_selector(product, SIMPLE_HTML)
        
        assert 'data-id="2"' in selector
        
        # Verify the selector works
        matches = simple_soup.select(selector)
        assert len(matches) == 1
        assert matches[0] == product

    def test_test_selector(self, selector_generator, complex_soup):
        """Test that test_selector correctly validates a selector."""
        result_item = complex_soup.select_one('[data-testid="result-1"]')
        
        # Should return True for a valid selector
        assert selector_generator.test_selector(
            '[data-testid="result-1"]', 
            result_item, 
            COMPLEX_HTML
        ) is True
        
        # Should return False for an invalid selector
        assert selector_generator.test_selector(
            '.wrong-class', 
            result_item, 
            COMPLEX_HTML
        ) is False

    def test_validate_selector_stability(self, selector_generator):
        """Test that validate_selector_stability calculates stability correctly."""
        # Create multiple HTML samples with consistent structure
        samples = [
            f"""<div class="container"><div class="item" id="item-{i}">Item {i}</div></div>"""
            for i in range(1, 4)
        ]
        
        # Test a stable selector
        stability = selector_generator.validate_selector_stability(".item", samples)
        assert stability > 0.7
        
        # Test an unstable selector
        stability = selector_generator.validate_selector_stability("#item-1", samples)
        assert stability < 0.5

    def test_optimize_selector(self, selector_generator):
        """Test that optimize_selector improves selector stability."""
        # Create HTML samples with varying IDs but consistent classes
        samples = [
            f"""<div class="container"><div class="product-item" id="product-{i}">Product {i}</div></div>"""
            for i in range(1, 4)
        ]
        
        # Start with a highly specific but unstable selector
        initial_selector = "#product-1.product-item"
        
        # Optimize the selector
        optimized = selector_generator.optimize_selector(initial_selector, samples)
        
        # Should prefer the more stable class-based selector
        assert optimized == ".product-item"

    def test_validate_content_match(self, selector_generator):
        """Test that _validate_content_match correctly evaluates content matches."""
        samples = [
            """<div class="price">$100</div>""",
            """<div class="price">$200</div>""",
            """<div class="price">$300</div>"""
        ]
        
        # Test with pattern that should match all price content
        match_score = selector_generator._validate_content_match(
            ".price", 
            samples, 
            r"\$\d+"
        )
        assert match_score > 0.9
        
        # Test with pattern that should only match some content
        match_score = selector_generator._validate_content_match(
            ".price", 
            samples, 
            r"\$2\d+"
        )
        assert 0 < match_score < 0.5

    def test_generate_id_selector(self, selector_generator, simple_soup):
        """Test generating an ID-based selector."""
        element = simple_soup.select_one('#product-list')
        selector = selector_generator._generate_id_selector(element)
        assert selector == '#product-list'

    def test_generate_class_selector(self, selector_generator, simple_soup):
        """Test generating a class-based selector."""
        # Get product element with multiple classes
        element = simple_soup.select_one('.product')
        selector = selector_generator._generate_class_selector(element)
        assert selector and '.product' in selector

    def test_generate_attribute_selector(self, selector_generator, simple_soup):
        """Test generating an attribute-based selector."""
        element = simple_soup.select_one('[data-id="2"]')
        selector = selector_generator._generate_attribute_selector(element)
        assert selector and 'data-id' in selector

    def test_generate_parent_child_selector(self, selector_generator, complex_soup):
        """Test generating a parent-child relationship selector."""
        title = complex_soup.select_one('.title')
        selector = selector_generator._generate_parent_child_selector(title)
        # Should include some parent relationship
        assert selector and '>' in selector

    def test_generate_positional_selector(self, selector_generator, simple_soup):
        """Test generating a position-based selector."""
        products = simple_soup.select('.product')
        second_product = products[1]
        selector = selector_generator._generate_positional_selector(second_product)
        assert selector and 'nth' in selector.lower()

    def test_generate_simplified_selectors(self, selector_generator):
        """Test generating simplified versions of a complex selector."""
        complex_selector = "#main .container > div.product-list .item[data-id='123']"
        simplified = selector_generator._generate_simplified_selectors(complex_selector)
        
        # Should generate multiple simpler alternatives
        assert len(simplified) > 1
        # At least one should be simpler than the original
        assert any(len(s) < len(complex_selector) for s in simplified)

    def test_adjust_selector_dynamically(self, selector_generator, complex_soup):
        """Test dynamic adjustment of a selector based on extraction results."""
        # Define a validation function for testing
        def is_valid_result(element):
            return element.name == 'div' and 'result-item' in element.get('class', [])
        
        # Test with a selector that needs adjustment
        adjusted = selector_generator.adjust_selector_dynamically(
            ".non-existent-class", 
            COMPLEX_HTML,
            is_valid_result
        )
        
        # Should find a valid alternative
        assert adjusted != ".non-existent-class"
        
        # Test with a selector that works fine
        adjusted = selector_generator.adjust_selector_dynamically(
            ".result-item", 
            COMPLEX_HTML,
            is_valid_result
        )
        
        # Should keep the original selector
        assert adjusted == ".result-item"

    @patch('components.pattern_analyzer.base_analyzer.get_registry')
    def test_generate_selectors_from_pattern(self, mock_get_registry, selector_generator):
        """Test generation of selectors from a registered pattern."""
        # Mock the registry and pattern
        mock_registry = MagicMock()
        mock_registry.get_pattern.return_value = {
            'selector': '.product-grid .product'
        }
        mock_get_registry.return_value = mock_registry
        
        # Generate selectors from a pattern
        selectors = selector_generator.generate_selectors_from_pattern(
            'product_listing', 
            'https://example.com/products'
        )
        
        # Should return the selector from the pattern
        assert selectors == {'product_listing': '.product-grid .product'}
        
        # Verify registry was called correctly
        mock_registry.get_pattern.assert_called_with(
            'product_listing', 
            'https://example.com/products'
        )

    def test_selector_with_nested_elements(self, selector_generator, complex_soup):
        """Test generating selectors for deeply nested elements."""
        price = complex_soup.select_one('.price')
        selector = selector_generator.generate_selector(price, COMPLEX_HTML)
        
        # Should generate a valid selector for the nested element
        assert selector and '.price' in selector
        
        # Verify the selector works
        matches = complex_soup.select(selector)
        assert len(matches) > 0
        assert price in matches

    def test_selectors_with_similar_elements(self, selector_generator, simple_soup):
        """Test generating distinct selectors for similar elements."""
        products = simple_soup.select('.product')
        
        selectors = [
            selector_generator.generate_selector(product, SIMPLE_HTML)
            for product in products
        ]
        
        # Should be able to distinguish between similar elements
        for i, selector in enumerate(selectors):
            matches = simple_soup.select(selector)
            # Either selector is unique to this product or it contains an attribute
            # that distinguishes it (like data-id)
            assert len(matches) == 1 or f"data-id=\"{i+1}\"" in selector

    def test_handle_dynamic_content(self, selector_generator):
        """Test handling of dynamic content variations."""
        # Base HTML
        base_html = """<div class="container"><div class="item">Content</div></div>"""
        
        # Create variations with dynamic classes that might be added by JavaScript
        variations = [
            """<div class="container active"><div class="item highlighted">Content</div></div>""",
            """<div class="container collapsed"><div class="item">Content</div></div>""",
            """<div class="container"><div class="item loading">Content</div></div>"""
        ]
        
        # Get a base selector
        base_soup = BeautifulSoup(base_html, 'html.parser')
        item = base_soup.select_one('.item')
        base_selector = selector_generator.generate_selector(item, base_html)
        
        # Check stability across variations
        stability = selector_generator.validate_selector_stability(base_selector, variations)
        
        # Should be reasonably stable even with dynamic classes
        assert stability > 0.5

    def test_handling_of_empty_elements(self, selector_generator):
        """Test handling of empty elements."""
        html = """
        <div class="container">
            <div class="empty"></div>
            <div class="has-content">Content</div>
        </div>
        """
        soup = BeautifulSoup(html, 'html.parser')
        empty = soup.select_one('.empty')
        
        # Should still generate a valid selector for empty elements
        selector = selector_generator.generate_selector(empty, html)
        assert selector and '.empty' in selector

    def test_multiple_strategy_fallback(self, selector_generator, varied_soup):
        """Test fallback through multiple strategies to find a working selector."""
        # Patch the first strategies to fail
        original_strategies = selector_generator.selector_strategies
        
        # Replace with strategies that return None, forcing fallback
        failing_strategy = lambda el: None
        selector_generator.selector_strategies = [
            failing_strategy, 
            failing_strategy,
            original_strategies[-1]  # Keep last strategy as backup
        ]
        
        # Should still generate a selector using the last strategy
        item = varied_soup.select_one('[data-sku="a1"]')
        selector = selector_generator.generate_selector(item, VARIED_HTML)
        
        assert selector is not None
        
        # Restore original strategies
        selector_generator.selector_strategies = original_strategies


if __name__ == "__main__":
    pytest.main(["-v", "test_selector_generator.py"])