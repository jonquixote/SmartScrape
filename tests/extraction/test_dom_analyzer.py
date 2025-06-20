"""
Test module for the DOM Analyzer utilities.
"""

import unittest
from bs4 import BeautifulSoup
import sys
import os

# Add parent directory to path to allow imports during testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from extraction.helpers.dom_analyzer import (
    calculate_text_density,
    detect_boilerplate,
    find_headline_elements,
    analyze_element_attributes,
    detect_list_structures,
    calculate_content_similarity,
    detect_visual_sections,
    analyze_whitespace_distribution,
    detect_layout_grid,
    map_reading_flow
)

class TestDOMAnalyzer(unittest.TestCase):
    """Test cases for DOM Analyzer utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample HTML with various structures
        self.sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test page for DOM analysis">
        </head>
        <body>
            <header class="site-header">
                <nav class="main-nav">
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                        <li><a href="/products">Products</a></li>
                        <li><a href="/contact">Contact</a></li>
                    </ul>
                </nav>
            </header>
            
            <main>
                <section class="hero">
                    <h1>Welcome to Our Website</h1>
                    <p>This is a test page for structural analysis.</p>
                </section>
                
                <section class="featured-products">
                    <h2>Featured Products</h2>
                    <div class="product-grid" style="display: grid; grid-template-columns: 1fr 1fr 1fr;">
                        <div class="product">
                            <img src="product1.jpg" alt="Product 1" width="200" height="200">
                            <h3>Product 1</h3>
                            <p class="price">$19.99</p>
                            <button>Add to Cart</button>
                        </div>
                        <div class="product">
                            <img src="product2.jpg" alt="Product 2" width="200" height="200">
                            <h3>Product 2</h3>
                            <p class="price">$29.99</p>
                            <button>Add to Cart</button>
                        </div>
                        <div class="product">
                            <img src="product3.jpg" alt="Product 3" width="200" height="200">
                            <h3>Product 3</h3>
                            <p class="price">$39.99</p>
                            <button>Add to Cart</button>
                        </div>
                    </div>
                </section>
                
                <article itemscope itemtype="http://schema.org/Article">
                    <h2 itemprop="headline">Latest News</h2>
                    <div itemprop="articleBody">
                        <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
                        Nullam eget felis eget nunc lobortis mattis aliquam faucibus.</p>
                        <p>Pellentesque habitant morbi tristique senectus et netus et 
                        malesuada fames ac turpis egestas.</p>
                    </div>
                </article>
                
                <hr>
                
                <section class="features">
                    <h2>Our Features</h2>
                    <div class="feature-list" style="display: flex;">
                        <div class="feature">
                            <i class="icon-feature1"></i>
                            <h3>Feature 1</h3>
                            <p>Feature description goes here</p>
                        </div>
                        <div class="feature">
                            <i class="icon-feature2"></i>
                            <h3>Feature 2</h3>
                            <p>Feature description goes here</p>
                        </div>
                        <div class="feature">
                            <i class="icon-feature3"></i>
                            <h3>Feature 3</h3>
                            <p>Feature description goes here</p>
                        </div>
                    </div>
                </section>
            </main>
            
            <aside class="sidebar">
                <h3>Related Links</h3>
                <ul>
                    <li><a href="/related1">Related 1</a></li>
                    <li><a href="/related2">Related 2</a></li>
                </ul>
                
                <div class="banner">
                    <img src="banner.jpg" alt="Banner" width="300" height="250">
                </div>
            </aside>
            
            <footer>
                <p>&copy; 2025 Test Company</p>
                <div class="social-links">
                    <a href="https://facebook.com">Facebook</a>
                    <a href="https://twitter.com">Twitter</a>
                    <a href="https://instagram.com">Instagram</a>
                </div>
            </footer>
        </body>
        </html>
        """
        
        # Parse HTML
        self.soup = BeautifulSoup(self.sample_html, 'html.parser')

    def test_calculate_text_density(self):
        """Test text density calculation."""
        # Test full document
        density = calculate_text_density(self.soup)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)
        
        # Test article section which should have higher text density
        article = self.soup.find('article')
        article_density = calculate_text_density(article)
        self.assertGreaterEqual(article_density, 0.0)
        
        # Test navigation which should have lower text density
        nav = self.soup.find('nav')
        nav_density = calculate_text_density(nav)
        self.assertGreaterEqual(nav_density, 0.0)
        
        # Article should have higher density than navigation
        self.assertGreater(article_density, nav_density)
        
    def test_detect_boilerplate(self):
        """Test boilerplate detection."""
        # Header should be detected as boilerplate
        header = self.soup.find('header')
        self.assertTrue(detect_boilerplate(header))
        
        # Footer should be detected as boilerplate
        footer = self.soup.find('footer')
        self.assertTrue(detect_boilerplate(footer))
        
        # Navigation should be detected as boilerplate
        nav = self.soup.find('nav')
        self.assertTrue(detect_boilerplate(nav))
        
        # Main content should not be detected as boilerplate
        article = self.soup.find('article')
        self.assertFalse(detect_boilerplate(article))
        
        # Featured products section should not be boilerplate
        products = self.soup.find('section', class_='featured-products')
        self.assertFalse(detect_boilerplate(products))
    
    def test_find_headline_elements(self):
        """Test headline element detection."""
        headlines = find_headline_elements(self.soup)
        
        # Should find all heading tags
        self.assertGreaterEqual(len(headlines), 6)  # We have 6 heading tags in our sample
        
        # Check if h1 is included
        h1_tags = [h for h in headlines if h.name == 'h1']
        self.assertEqual(len(h1_tags), 1)
        
        # Check if elements with headline-related attributes are found
        schema_headlines = [h for h in headlines if h.get('itemprop') == 'headline']
        self.assertGreaterEqual(len(schema_headlines), 1)
    
    def test_analyze_element_attributes(self):
        """Test element attribute analysis."""
        # Test analyzing a link element
        link = self.soup.find('a')
        link_attrs = analyze_element_attributes(link)
        
        self.assertEqual(link_attrs['tag_name'], 'a')
        self.assertIn('href', link_attrs['link_data'])
        
        # Test analyzing an image element
        img = self.soup.find('img')
        img_attrs = analyze_element_attributes(img)
        
        self.assertEqual(img_attrs['tag_name'], 'img')
        self.assertIn('image_data', img_attrs)
        self.assertIn('src', img_attrs['image_data'])
        self.assertIn('alt', img_attrs['image_data'])
        
        # Test analyzing element with schema.org attributes
        article = self.soup.find('article')
        article_attrs = analyze_element_attributes(article)
        
        self.assertEqual(article_attrs['tag_name'], 'article')
        self.assertIn('microdata', article_attrs)
        self.assertIn('itemtype', article_attrs['microdata'])
    
    def test_detect_list_structures(self):
        """Test list structure detection."""
        list_structures = detect_list_structures(self.soup)
        
        # Should find at least 2 lists (nav menu and sidebar)
        self.assertGreaterEqual(len(list_structures), 2)
        
        # Check for different list types
        list_types = [lst['type'] for lst in list_structures]
        self.assertIn('semantic', list_types)  # Should detect semantic lists
        
        # Find a specific list
        nav_list = None
        for lst in list_structures:
            if 'nav' in lst['selector'].lower() and lst['type'] == 'semantic':
                nav_list = lst
                break
                
        self.assertIsNotNone(nav_list)
        self.assertEqual(nav_list['item_count'], 4)  # Nav has 4 items
    
    def test_calculate_content_similarity(self):
        """Test content similarity calculation."""
        # Get two product elements which should be similar
        products = self.soup.select('.product')
        self.assertGreaterEqual(len(products), 2)
        
        # Calculate similarity between two products
        similarity = calculate_content_similarity(products[0], products[1])
        
        # Products should be very similar (>0.7)
        self.assertGreaterEqual(similarity, 0.7)
        
        # Compare dissimilar elements
        article = self.soup.find('article')
        header = self.soup.find('header')
        dissimilar_score = calculate_content_similarity(article, header)
        
        # Article and header should be dissimilar (<0.5)
        self.assertLessEqual(dissimilar_score, 0.5)
        
        # Similarity should be higher for similar elements
        self.assertGreater(similarity, dissimilar_score)
    
    def test_detect_visual_sections(self):
        """Test visual section detection."""
        visual_sections = detect_visual_sections(self.soup)
        
        # Should find multiple visual sections
        self.assertGreaterEqual(len(visual_sections), 3)
        
        # Check section types
        section_types = [section['visual_type'] for section in visual_sections]
        
        # Should identify at least one main content section
        self.assertIn('main_content', section_types)
        
        # Should identify header section
        self.assertIn('header', section_types)
        
        # Should identify features section
        feature_sections = [s for s in visual_sections if 'feature' in s['selector'].lower()]
        self.assertGreaterEqual(len(feature_sections), 1)
    
    def test_analyze_whitespace_distribution(self):
        """Test whitespace distribution analysis."""
        whitespace = analyze_whitespace_distribution(self.soup)
        
        # Should detect the horizontal rule
        self.assertEqual(whitespace['horizontal_dividers'], 1)
        
        # Should detect grid/flex layouts
        self.assertGreaterEqual(whitespace['grid_flex_layouts'], 2)  # We have grid and flex in our sample
        
        # Should detect layout complexity
        self.assertIn(whitespace['layout_complexity'], ['low', 'medium', 'high'])
    
    def test_detect_layout_grid(self):
        """Test layout grid detection."""
        grids = detect_layout_grid(self.soup)
        
        # Should find at least 2 grid structures
        self.assertGreaterEqual(len(grids), 2)
        
        # Check grid types
        grid_types = [grid['type'] for grid in grids]
        self.assertIn('css_grid', grid_types)  # Explicit CSS grid
        self.assertIn('flexbox', grid_types)   # Flexbox layout
        
        # Find the product grid
        product_grid = None
        for grid in grids:
            if 'product-grid' in grid['selector']:
                product_grid = grid
                break
                
        self.assertIsNotNone(product_grid)
        self.assertEqual(product_grid['type'], 'css_grid')
    
    def test_map_reading_flow(self):
        """Test reading flow mapping."""
        reading_flow = map_reading_flow(self.soup)
        
        # Should find multiple content elements
        self.assertGreaterEqual(len(reading_flow), 5)
        
        # Check element types
        element_types = [elem['type'] for elem in reading_flow]
        self.assertIn('heading', element_types)
        self.assertIn('paragraph', element_types)
        
        # Should include images
        images = [elem for elem in reading_flow if elem['type'] == 'image']
        self.assertGreaterEqual(len(images), 1)
        
        # First heading should be h1
        headings = [elem for elem in reading_flow if elem['type'] == 'heading']
        self.assertGreaterEqual(len(headings), 1)
        first_heading = min(headings, key=lambda x: x['order'])
        self.assertEqual(first_heading['level'], 1)  # Should be h1

if __name__ == '__main__':
    unittest.main()