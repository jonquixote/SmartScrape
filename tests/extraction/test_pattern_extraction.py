"""
Pattern Extraction Tests

This module provides comprehensive tests for the DOMPatternExtractor class
to ensure reliable extraction of structured data from HTML content.
"""

import unittest
import os
import json
import re
from bs4 import BeautifulSoup
from unittest.mock import patch, MagicMock

from extraction.pattern_extractor import DOMPatternExtractor
from extraction.helpers.selector_generator import SelectorGenerator
from core.html_service import HTMLService

# Sample HTML content for testing
PRODUCT_LIST_HTML = """
<html>
<body>
    <div class="products-container">
        <div class="product">
            <h3 class="title">Product 1</h3>
            <span class="price">$19.99</span>
            <img src="product1.jpg" alt="Product 1">
            <p class="description">This is product 1 description</p>
            <a href="/product1" class="link">View Details</a>
        </div>
        <div class="product">
            <h3 class="title">Product 2</h3>
            <span class="price">$29.99</span>
            <img src="product2.jpg" alt="Product 2">
            <p class="description">This is product 2 description</p>
            <a href="/product2" class="link">View Details</a>
        </div>
        <div class="product">
            <h3 class="title">Product 3</h3>
            <span class="price">$39.99</span>
            <img src="product3.jpg" alt="Product 3">
            <p class="description">This is product 3 description</p>
            <a href="/product3" class="link">View Details</a>
        </div>
    </div>
    <div class="pagination">
        <a href="/page/1" class="current">1</a>
        <a href="/page/2">2</a>
        <a href="/page/3">3</a>
        <a href="/page/2" class="next">Next</a>
    </div>
</body>
</html>
"""

ARTICLE_LIST_HTML = """
<html>
<body>
    <div class="articles">
        <article class="post">
            <h2><a href="/article1">Article 1 Title</a></h2>
            <div class="meta">
                <span class="author">By John Doe</span>
                <span class="date">January 1, 2023</span>
            </div>
            <div class="summary">Summary of article 1</div>
        </article>
        <article class="post">
            <h2><a href="/article2">Article 2 Title</a></h2>
            <div class="meta">
                <span class="author">By Jane Smith</span>
                <span class="date">January 2, 2023</span>
            </div>
            <div class="summary">Summary of article 2</div>
        </article>
        <article class="post">
            <h2><a href="/article3">Article 3 Title</a></h2>
            <div class="meta">
                <span class="author">By Bob Johnson</span>
                <span class="date">January 3, 2023</span>
            </div>
            <div class="summary">Summary of article 3</div>
        </article>
    </div>
</body>
</html>
"""

TABLE_HTML = """
<html>
<body>
    <table class="data-table">
        <tr>
            <th>Name</th>
            <th>Age</th>
            <th>Location</th>
        </tr>
        <tr>
            <td>John Doe</td>
            <td>32</td>
            <td>New York</td>
        </tr>
        <tr>
            <td>Jane Smith</td>
            <td>28</td>
            <td>San Francisco</td>
        </tr>
        <tr>
            <td>Bob Johnson</td>
            <td>45</td>
            <td>Chicago</td>
        </tr>
    </table>
</body>
</html>
"""

IRREGULAR_PATTERN_HTML = """
<html>
<body>
    <div class="mixed-container">
        <div class="item type-a">
            <h4>Item A-1</h4>
            <p>Description A-1</p>
        </div>
        <div class="item type-b">
            <h3>Item B-1</h3>
            <span>Description B-1</span>
            <img src="image-b1.jpg">
        </div>
        <div class="item type-a">
            <h4>Item A-2</h4>
            <p>Description A-2</p>
        </div>
        <div class="item type-c">
            <h5>Item C-1</h5>
            <div class="desc">Description C-1</div>
            <a href="/link-c1">Link C-1</a>
        </div>
    </div>
</body>
</html>
"""

NO_PATTERN_HTML = """
<html>
<body>
    <div class="content">
        <h1>Welcome to Our Website</h1>
        <p>This is a simple page with no repeating patterns.</p>
        <img src="banner.jpg">
        <div class="about">
            <h2>About Us</h2>
            <p>We are a company that provides services.</p>
        </div>
    </div>
</body>
</html>
"""


class TestDOMPatternExtractor(unittest.TestCase):
    """Test cases for the DOMPatternExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DOMPatternExtractor()
        self.extractor.initialize()
        
    def tearDown(self):
        """Clean up after tests."""
        self.extractor.shutdown()
    
    def test_can_handle(self):
        """Test content type detection."""
        # Should handle HTML content
        self.assertTrue(self.extractor.can_handle(PRODUCT_LIST_HTML))
        self.assertTrue(self.extractor.can_handle(PRODUCT_LIST_HTML, 'html'))
        
        # Should not handle non-HTML content
        self.assertFalse(self.extractor.can_handle('{"key": "value"}', 'json'))
        self.assertFalse(self.extractor.can_handle('plain text'))
    
    def test_identify_result_containers(self):
        """Test identification of result containers."""
        # Test with product list
        containers = self.extractor.identify_result_containers(PRODUCT_LIST_HTML)
        self.assertGreater(len(containers), 0)
        self.assertIn('selector', containers[0])
        
        # Test with article list
        containers = self.extractor.identify_result_containers(ARTICLE_LIST_HTML)
        self.assertGreater(len(containers), 0)
        
        # Test with non-pattern content
        containers = self.extractor.identify_result_containers(NO_PATTERN_HTML)
        self.assertEqual(len(containers), 0)
    
    def test_extract_with_explicit_pattern(self):
        """Test extraction with a pre-defined pattern."""
        # Extract products with explicit container selector
        result = self.extractor.extract(PRODUCT_LIST_HTML, {
            'target_pattern': '.products-container',
            'max_items': 2
        })
        
        self.assertTrue('items' in result)
        self.assertEqual(result['count'], 2)  # Should respect max_items
        self.assertTrue('container_selector' in result)
        self.assertTrue('item_selector' in result)
        
        # Check first item has expected fields
        first_item = result['items'][0]
        self.assertTrue('title' in first_item or 'name' in first_item)
        self.assertTrue('price' in first_item)
    
    def test_extract_auto_pattern_detection(self):
        """Test extraction with automatic pattern detection."""
        # Extract articles without specifying a pattern
        result = self.extractor.extract(ARTICLE_LIST_HTML)
        
        self.assertTrue('items' in result)
        self.assertGreater(result['count'], 0)
        
        # Verify some fields were extracted
        if result['items']:
            first_item = result['items'][0]
            # Check if at least one expected field exists
            self.assertTrue(any(field in first_item for field in ['title', 'author', 'date', 'summary']))
    
    def test_pagination_detection(self):
        """Test detection of pagination patterns."""
        result = self.extractor.extract(PRODUCT_LIST_HTML, {'extract_pagination': True})
        
        self.assertTrue('pagination' in result)
        pagination = result['pagination']
        self.assertTrue(pagination['has_pagination'])
        self.assertEqual(pagination['current_page'], 1)
        self.assertEqual(pagination['next_page'], '/page/2')
    
    def test_tabular_data_extraction(self):
        """Test extraction of tabular data."""
        # First find the table container
        containers = self.extractor.identify_result_containers(TABLE_HTML)
        
        # Use first container for extraction
        if containers:
            container_selector = containers[0]['selector']
            result = self.extractor.extract(TABLE_HTML, {'target_pattern': container_selector})
            
            # Check structure analysis to verify it's a table
            structure = self.extractor.analyze_result_structure(TABLE_HTML, container_selector)
            self.assertTrue(structure.get('is_table', False))
            
            # Check table data
            self.assertGreater(len(result['items']), 0)
            
    def test_field_mapping(self):
        """Test mapping of extracted fields to standard schema."""
        # Extract raw items
        raw_items = [
            {'product_name': 'Item 1', 'product_price': '$10.99', 'img': 'item1.jpg'},
            {'name': 'Item 2', 'cost': '$20.99', 'image': 'item2.jpg'},
            {'title': 'Item 3', 'price': '$30.99', 'thumbnail': 'item3.jpg'}
        ]
        
        # Map fields to standard schema
        mapped_items = self.extractor.map_item_fields(raw_items)
        
        # Check if fields were standardized
        self.assertEqual(mapped_items[0]['title'], 'Item 1')
        self.assertEqual(mapped_items[1]['title'], 'Item 2')
        self.assertEqual(mapped_items[2]['title'], 'Item 3')
        
        self.assertTrue('price' in mapped_items[0])
        self.assertTrue('price' in mapped_items[1])
        self.assertTrue('price' in mapped_items[2])
        
        self.assertTrue('image' in mapped_items[0])
        self.assertTrue('image' in mapped_items[1])
        self.assertTrue('image' in mapped_items[2])
    
    def test_irregular_pattern_handling(self):
        """Test handling of irregular patterns with mixed structures."""
        # Extract from content with irregular patterns
        result = self.extractor.extract(IRREGULAR_PATTERN_HTML)
        
        # Verify items were extracted despite irregularities
        self.assertGreater(result['count'], 0)
        
        # Check confidence score - should be lower for irregular patterns
        self.assertLess(result['confidence'], 1.0)
    
    def test_find_repeating_patterns(self):
        """Test identification of repeating DOM patterns."""
        soup = BeautifulSoup(PRODUCT_LIST_HTML, 'lxml')
        patterns = self.extractor.find_repeating_patterns(soup)
        
        self.assertGreater(len(patterns), 0)
        
        # Check that the pattern contains a container and items
        first_pattern = patterns[0]
        self.assertIn('container', first_pattern)
        self.assertIn('items', first_pattern)
        self.assertGreater(len(first_pattern['items']), 0)
        
    def test_element_density_analysis(self):
        """Test content-rich section detection based on element density."""
        soup = BeautifulSoup(PRODUCT_LIST_HTML, 'lxml')
        dense_areas = self.extractor.analyze_element_density(soup)
        
        self.assertGreater(len(dense_areas), 0)
        self.assertTrue(isinstance(dense_areas[0], Tag))
    
    def test_robust_selector_generation(self):
        """Test generation of robust selectors that handle variations."""
        # Create a sample container and items
        soup = BeautifulSoup(PRODUCT_LIST_HTML, 'lxml')
        container = soup.select_one('.products-container')
        items = soup.select('.product')
        
        # Generate selectors
        selectors = self.extractor.generate_selectors(container, items)
        
        # Verify selectors were generated
        self.assertIn('container', selectors)
        self.assertIn('item', selectors)
        
        # Verify the selectors work to find elements
        found_container = soup.select_one(selectors['container'])
        self.assertEqual(found_container, container)
        
        found_items = soup.select(selectors['item'])
        self.assertEqual(len(found_items), len(items))
    
    def test_selector_validation(self):
        """Test validation of selector reliability."""
        # Valid selector test
        valid_result = self.extractor.validate_selector('.product', PRODUCT_LIST_HTML)
        self.assertTrue(valid_result['valid'])
        self.assertGreater(valid_result['count'], 0)
        
        # Invalid selector test
        invalid_result = self.extractor.validate_selector('.nonexistent', PRODUCT_LIST_HTML)
        self.assertFalse(invalid_result['valid'])
        
        # Inconsistent selector test (selects elements with different structures)
        inconsistent_result = self.extractor.validate_selector('div', IRREGULAR_PATTERN_HTML)
        self.assertTrue(inconsistent_result['valid'])
        if 'consistent' in inconsistent_result:
            self.assertLess(inconsistent_result.get('similarity', 1.0), 1.0)
    
    def test_error_recovery(self):
        """Test error recovery strategies during extraction."""
        # Test with malformed HTML
        malformed_html = "<div><p>Unclosed paragraph<div>Next div</div>"
        result = self.extractor.extract(malformed_html)
        
        # Should return empty result but not crash
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)
        
        # Test with valid HTML but a bad selector
        with patch.object(self.extractor, 'identify_result_containers', return_value=[{'selector': 'bad::selector', 'confidence': 1.0}]):
            result = self.extractor.extract(PRODUCT_LIST_HTML)
            self.assertEqual(result['count'], 0)
            self.assertIn('error', result)
    
    def test_list_structure_analysis(self):
        """Test analysis of list-like structures."""
        # Create a list container
        soup = BeautifulSoup('<ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul>', 'lxml')
        container = soup.find('ul')
        
        # Analyze list structure
        result = self.extractor.extract_list_structure(container)
        
        self.assertTrue(result['is_list'])
        self.assertEqual(result['list_type'], 'ul')
        self.assertEqual(result['item_count'], 3)
    
    def test_no_pattern_handling(self):
        """Test behavior when no patterns are found."""
        result = self.extractor.extract(NO_PATTERN_HTML)
        
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)
        self.assertEqual(result['confidence'], 0.0)
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        result = self.extractor.extract("")
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)
        
        result = self.extractor.extract(None)
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)
    
    @unittest.skip("Integration test requiring external resources")
    def test_integration_with_real_website(self):
        """
        Integration test with a real website.
        
        This test requires internet access and is skipped by default.
        """
        from core.html_service import HTMLService
        
        # Create HTML service to fetch real content
        html_service = HTMLService()
        
        # Fetch a known listing page
        content = html_service.fetch_url("https://books.toscrape.com/")
        
        # Extract data
        result = self.extractor.extract(content)
        
        # Verify extraction worked
        self.assertGreater(result['count'], 0)
        self.assertGreater(result['confidence'], 0.5)
        
        # Check fields - book listing should have title, price
        first_item = result['items'][0] if result['items'] else {}
        self.assertTrue(any(field in first_item for field in ['title', 'name']))
        self.assertTrue('price' in first_item)
        

if __name__ == '__main__':
    unittest.main()