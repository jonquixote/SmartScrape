import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.multi_strategy import MultiStrategy, create_multi_strategy
from strategies.dfs_strategy import DFSStrategy
from strategies.bfs_strategy import BFSStrategy
from strategies.best_first import BestFirstStrategy

class TestMultiStrategy(unittest.TestCase):
    """Test cases for the Multi-Strategy Extraction System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dfs_strategy = DFSStrategy(max_depth=2, max_pages=10)
        self.bfs_strategy = BFSStrategy(max_depth=2, max_pages=10)
        self.best_first_strategy = BestFirstStrategy(max_depth=2, max_pages=10)
        
        # Create a multi-strategy with the above strategies
        self.multi_strategy = MultiStrategy(
            strategies=[self.dfs_strategy, self.bfs_strategy, self.best_first_strategy],
            fallback_threshold=0.4,
            confidence_threshold=0.6,
            max_depth=2,
            max_pages=10
        )
        
        # Sample HTML content
        self.sample_html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Product: Test Product</h1>
                <div class="price">$99.99</div>
                <div class="description">This is a test product description.</div>
                <div class="info">
                    <span>Weight: 2kg</span>
                    <span>Color: Blue</span>
                </div>
                <div class="links">
                    <a href="/product/1">Product 1</a>
                    <a href="/product/2">Product 2</a>
                    <a href="/category/electronics">Electronics</a>
                </div>
            </body>
        </html>
        """
    
    def test_create_multi_strategy(self):
        """Test creating a multi-strategy with factory function."""
        strategy = create_multi_strategy(
            strategy_types=["dfs", "bfs", "best-first"],
            max_depth=3,
            max_pages=20
        )
        
        self.assertIsInstance(strategy, MultiStrategy)
        self.assertEqual(len(strategy.strategies), 3)
        self.assertEqual(strategy.max_depth, 3)
        self.assertEqual(strategy.max_pages, 20)
    
    @patch('strategies.base_strategy.BaseStrategy.extract')
    async def test_extract_with_multi_strategy(self, mock_extract):
        """Test extraction using multiple strategies."""
        # Mock strategy extraction results
        mock_extract.side_effect = [
            # First strategy
            {
                'success': True,
                'data': {'product': 'Test Product', 'price': '$99.99'},
                'confidence': 0.8
            },
            # Second strategy
            {
                'success': True,
                'data': {'product': 'Test Product', 'description': 'This is a test product description.'},
                'confidence': 0.7
            },
            # Third strategy
            {
                'success': True,
                'data': {'product': 'Test Product', 'color': 'Blue', 'weight': '2kg'},
                'confidence': 0.6
            }
        ]
        
        # Test the multi-strategy extraction
        result, confidence, strategy = await self.multi_strategy._extract_with_multi_strategy(
            url='https://example.com',
            html=self.sample_html
        )
        
        self.assertEqual(strategy, 'combined')
        self.assertGreaterEqual(confidence, 0.6)
        self.assertEqual(result.get('product'), 'Test Product')
        self.assertEqual(result.get('price'), '$99.99')
        self.assertEqual(result.get('description'), 'This is a test product description.')
        self.assertEqual(result.get('color'), 'Blue')
        self.assertEqual(result.get('weight'), '2kg')
    
    def test_combine_results_with_voting(self):
        """Test combining results with weighted voting."""
        strategy_results = [
            (
                {'product': 'Test Product', 'price': '$99.99'}, 
                0.8, 
                'dfs'
            ),
            (
                {'product': 'Test Product', 'description': 'Test description'}, 
                0.7, 
                'bfs'
            ),
            (
                {'product': 'Different Name', 'color': 'Blue'}, 
                0.4, 
                'best-first'
            )
        ]
        
        combined_data, confidence, strategy = self.multi_strategy._combine_results_with_voting(strategy_results)
        
        self.assertEqual(strategy, 'combined')
        self.assertGreaterEqual(confidence, 0.5)
        self.assertEqual(combined_data.get('product'), 'Test Product')
        self.assertEqual(combined_data.get('price'), '$99.99')
        self.assertEqual(combined_data.get('description'), 'Test description')
        self.assertEqual(combined_data.get('color'), 'Blue')
    
    @patch('strategies.multi_strategy.MultiStrategy._extract_with_multi_strategy')
    @patch('strategies.multi_strategy.BaseStrategy.get_next_urls')
    async def test_execute_crawling(self, mock_get_next_urls, mock_extract):
        """Test the execution of multi-strategy crawling."""
        # Mock crawler
        mock_crawler = MagicMock()
        mock_crawler.fetch_url.side_effect = [
            {'html': self.sample_html, 'links': ['/page1', '/page2']},
            {'html': self.sample_html, 'links': ['/page3']},
            {'html': self.sample_html, 'links': []}
        ]
        
        # Mock extraction results
        mock_extract.side_effect = [
            ({'product': 'Test Product'}, 0.8, 'combined'),
            ({'product': 'Another Product'}, 0.7, 'combined'),
            ({'product': 'Final Product'}, 0.6, 'combined')
        ]
        
        # Mock next URLs
        mock_get_next_urls.return_value = [
            {'url': 'https://example.com/page1', 'depth': 1, 'score': 0.9},
            {'url': 'https://example.com/page2', 'depth': 1, 'score': 0.8}
        ]
        
        # Execute the strategy
        result = await self.multi_strategy.execute(
            crawler=mock_crawler,
            start_url='https://example.com'
        )
        
        # Assertions
        self.assertIn('results', result)
        self.assertIn('stats', result)
        self.assertIn('visited_urls', result)
        self.assertEqual(len(result['results']), 3)
        self.assertEqual(result['stats']['successful_extractions'], 3)

def run_async_test(coro):
    """Helper function to run coroutines in tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)

if __name__ == '__main__':
    unittest.main()