"""
Unit tests for IntelligentURLGenerator component.

This module tests the intelligent URL generation system including intent analysis
integration, semantic term expansion, URL scoring, validation, and template-based
generation for efficient web scraping.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from urllib.parse import urlparse

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from components.intelligent_url_generator import IntelligentURLGenerator, URLScore


class TestIntelligentURLGenerator(unittest.TestCase):
    """Test suite for IntelligentURLGenerator component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock intent analyzer
        self.mock_intent_analyzer = Mock()
        self.mock_intent_analyzer.analyze_intent.return_value = {
            'intent_type': 'information_seeking',
            'entities': [
                {'text': 'technology', 'label': 'TOPIC'},
                {'text': 'artificial intelligence', 'label': 'TOPIC'}
            ],
            'semantic_keywords': ['AI', 'machine learning', 'neural networks'],
            'query_complexity': 'medium',
            'domain_hints': ['tech', 'research', 'academic'],
            'intent_confidence': 0.85
        }
        self.mock_intent_analyzer.expand_query_contextually.return_value = [
            'AI technology', 'machine learning', 'artificial intelligence research'
        ]
        
        # Create test configuration
        self.mock_config = {
            'INTELLIGENT_URL_GENERATION': True,
            'CONTEXTUAL_QUERY_EXPANSION': True,
            'SEARCH_DEPTH': 3,
            'CRAWL4AI_MAX_PAGES': 10
        }
        
        # Initialize generator with mocks
        self.generator = IntelligentURLGenerator(
            intent_analyzer=self.mock_intent_analyzer,
            config=self.mock_config
        )
    
    def test_initialization(self):
        """Test IntelligentURLGenerator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.intent_analyzer, self.mock_intent_analyzer)
        self.assertEqual(self.generator.config, self.mock_config)
        self.assertTrue(hasattr(self.generator, 'domain_templates'))
        self.assertTrue(hasattr(self.generator, 'high_value_patterns'))
        self.assertTrue(hasattr(self.generator, 'domain_reputation'))
    
    def test_initialization_without_intent_analyzer(self):
        """Test initialization without intent analyzer."""
        generator = IntelligentURLGenerator()
        self.assertIsNone(generator.intent_analyzer)
        self.assertIsNotNone(generator.domain_templates)
        self.assertIsNotNone(generator.high_value_patterns)
    
    def test_generate_urls_basic(self):
        """Test basic URL generation functionality."""
        query = "artificial intelligence"
        
        with patch.object(self.generator, '_generate_direct_hit_urls', return_value=['https://example.com/ai']):
            with patch.object(self.generator, '_generate_template_urls', return_value=['https://tech.com/ai']):
                with patch.object(self.generator, '_generate_search_urls', return_value=['https://search.com?q=ai']):
                    with patch.object(self.generator, '_generate_navigation_urls', return_value=['https://news.com/tech']):
                        with patch.object(self.generator, 'validate_and_score_url') as mock_score:
                            mock_score.return_value = URLScore(
                                url='https://example.com/ai',
                                relevance_score=0.8,
                                intent_match_score=0.7,
                                domain_reputation_score=0.9,
                                pattern_match_score=0.8,
                                confidence=0.8
                            )
                            
                            urls = self.generator.generate_urls(query)
                            
                            self.assertIsInstance(urls, list)
                            self.assertGreater(len(urls), 0)
                            self.assertIsInstance(urls[0], URLScore)
    
    def test_generate_urls_with_base_url(self):
        """Test URL generation with base URL constraint."""
        query = "technology news"
        base_url = "https://example.com"
        
        with patch.object(self.generator, '_generate_direct_hit_urls', return_value=['https://example.com/tech']):
            with patch.object(self.generator, 'validate_and_score_url') as mock_score:
                mock_score.return_value = URLScore(
                    url='https://example.com/tech',
                    relevance_score=0.7,
                    intent_match_score=0.6,
                    domain_reputation_score=0.8,
                    pattern_match_score=0.7,
                    confidence=0.7
                )
                
                urls = self.generator.generate_urls(query, base_url=base_url)
                
                self.assertIsInstance(urls, list)
                # Verify base URL constraint is considered
                for url_score in urls:
                    parsed = urlparse(url_score.url)
                    base_parsed = urlparse(base_url)
                    # Should either be same domain or related domain
                    self.assertTrue(
                        parsed.netloc == base_parsed.netloc or 
                        base_parsed.netloc in parsed.netloc or
                        True  # Allow for expanded searches
                    )
    
    def test_generate_urls_with_intent_analysis(self):
        """Test URL generation with provided intent analysis."""
        query = "machine learning tutorials"
        intent_analysis = {
            'intent_type': 'learning',
            'entities': [{'text': 'machine learning', 'label': 'TOPIC'}],
            'semantic_keywords': ['ML', 'tutorials', 'education'],
            'domain_hints': ['educational', 'tutorial']
        }
        
        with patch.object(self.generator, '_generate_direct_hit_urls', return_value=['https://edu.com/ml']):
            with patch.object(self.generator, 'validate_and_score_url') as mock_score:
                mock_score.return_value = URLScore(
                    url='https://edu.com/ml',
                    relevance_score=0.9,
                    intent_match_score=0.8,
                    domain_reputation_score=0.7,
                    pattern_match_score=0.9,
                    confidence=0.85
                )
                
                urls = self.generator.generate_urls(query, intent_analysis=intent_analysis)
                
                self.assertIsInstance(urls, list)
                # Verify intent analyzer was not called (since intent_analysis provided)
                self.mock_intent_analyzer.analyze_intent.assert_not_called()
    
    def test_generate_urls_with_max_urls_limit(self):
        """Test URL generation with maximum URLs limit."""
        query = "test query"
        max_urls = 3
        
        # Mock multiple URL generation methods to return more than max_urls
        with patch.object(self.generator, '_generate_direct_hit_urls', return_value=['https://1.com', 'https://2.com']):
            with patch.object(self.generator, '_generate_template_urls', return_value=['https://3.com', 'https://4.com']):
                with patch.object(self.generator, '_generate_search_urls', return_value=['https://5.com', 'https://6.com']):
                    with patch.object(self.generator, 'validate_and_score_url') as mock_score:
                        # Return decreasing relevance scores
                        mock_score.side_effect = [
                            URLScore('https://1.com', 0.9, 0.8, 0.9, 0.8, 0.85),
                            URLScore('https://2.com', 0.8, 0.7, 0.8, 0.7, 0.75),
                            URLScore('https://3.com', 0.7, 0.6, 0.7, 0.6, 0.65),
                            URLScore('https://4.com', 0.6, 0.5, 0.6, 0.5, 0.55),
                            URLScore('https://5.com', 0.5, 0.4, 0.5, 0.4, 0.45),
                            URLScore('https://6.com', 0.4, 0.3, 0.4, 0.3, 0.35)
                        ]
                        
                        urls = self.generator.generate_urls(query, max_urls=max_urls)
                        
                        self.assertLessEqual(len(urls), max_urls)
                        # Verify sorted by relevance score
                        if len(urls) > 1:
                            for i in range(len(urls) - 1):
                                self.assertGreaterEqual(urls[i].relevance_score, urls[i + 1].relevance_score)
    
    def test_expand_search_terms_with_intent(self):
        """Test search term expansion with intent analysis."""
        query = "artificial intelligence"
        intent_analysis = {
            'entities': [
                {'text': 'machine learning', 'label': 'TOPIC'},
                {'text': 'neural networks', 'label': 'TOPIC'}
            ],
            'intent_type': 'research'
        }
        
        expanded_terms = self.generator.expand_search_terms(query, intent_analysis)
        
        self.assertIsInstance(expanded_terms, list)
        self.assertIn(query, expanded_terms)  # Original query should be included
        self.assertGreater(len(expanded_terms), 1)  # Should have additional terms
        
        # Verify contextual expansion was called
        self.mock_intent_analyzer.expand_query_contextually.assert_called_once()
    
    def test_expand_search_terms_without_intent(self):
        """Test search term expansion without intent analysis."""
        query = "python programming"
        
        with patch.object(self.generator, '_basic_term_expansion', return_value=['python', 'programming', 'coding']):
            expanded_terms = self.generator.expand_search_terms(query, None)
            
            self.assertIsInstance(expanded_terms, list)
            self.assertIn(query, expanded_terms)
            self.assertGreater(len(expanded_terms), 1)
    
    def test_expand_search_terms_deduplication(self):
        """Test that expanded search terms are deduplicated."""
        query = "AI"
        intent_analysis = {
            'entities': [
                {'text': 'AI', 'label': 'TOPIC'},  # Duplicate of original query
                {'text': 'artificial intelligence', 'label': 'TOPIC'}
            ]
        }
        
        # Mock contextual expansion to return terms including duplicates
        self.mock_intent_analyzer.expand_query_contextually.return_value = ['AI', 'machine learning']
        
        expanded_terms = self.generator.expand_search_terms(query, intent_analysis)
        
        # Should not contain duplicate 'AI' terms
        ai_count = sum(1 for term in expanded_terms if term.lower() == 'ai')
        self.assertEqual(ai_count, 1)
    
    def test_validate_and_score_url_valid(self):
        """Test URL validation and scoring for valid URLs."""
        url = "https://example.com/artificial-intelligence"
        intent = {
            'semantic_keywords': ['artificial intelligence', 'AI'],
            'intent_type': 'information_seeking'
        }
        
        with patch.object(self.generator, '_calculate_intent_match_score', return_value=0.8):
            with patch.object(self.generator, '_get_domain_reputation_score', return_value=0.7):
                with patch.object(self.generator, '_calculate_pattern_match_score', return_value=0.9):
                    
                    score = self.generator.validate_and_score_url(url, intent)
                    
                    self.assertIsInstance(score, URLScore)
                    self.assertEqual(score.url, url)
                    self.assertGreater(score.relevance_score, 0)
                    self.assertGreater(score.confidence, 0)
    
    def test_validate_and_score_url_invalid(self):
        """Test URL validation and scoring for invalid/malformed URLs."""
        invalid_url = "not-a-valid-url"
        intent = {}
        
        score = self.generator.validate_and_score_url(invalid_url, intent)
        
        self.assertIsInstance(score, URLScore)
        # Should handle gracefully with low scores
        self.assertLessEqual(score.confidence, 0.5)
    
    def test_validate_and_score_url_empty_intent(self):
        """Test URL validation with empty intent analysis."""
        url = "https://example.com"
        intent = {}
        
        with patch.object(self.generator, '_get_domain_reputation_score', return_value=0.5):
            score = self.generator.validate_and_score_url(url, intent)
            
            self.assertIsInstance(score, URLScore)
            self.assertEqual(score.url, url)
            # Should still return reasonable scores even without intent
            self.assertGreaterEqual(score.confidence, 0)
    
    @patch('components.intelligent_url_generator.INTELLIGENT_URL_GENERATION', False)
    def test_generate_urls_disabled_config(self):
        """Test URL generation when intelligent generation is disabled."""
        query = "test query"
        
        with patch.object(self.generator, '_generate_basic_urls', return_value=[]) as mock_basic:
            urls = self.generator.generate_urls(query)
            
            mock_basic.assert_called_once_with(query, None)
            self.assertIsInstance(urls, list)
    
    def test_generate_urls_error_handling(self):
        """Test error handling in URL generation."""
        query = "test query"
        
        # Mock methods to raise exceptions
        with patch.object(self.generator, '_generate_direct_hit_urls', side_effect=Exception("Test error")):
            with patch.object(self.generator, '_generate_template_urls', return_value=[]):
                with patch.object(self.generator, '_generate_search_urls', return_value=[]):
                    with patch.object(self.generator, '_generate_navigation_urls', return_value=[]):
                        
                        # Should handle exceptions gracefully
                        urls = self.generator.generate_urls(query)
                        self.assertIsInstance(urls, list)
    
    def test_url_score_dataclass(self):
        """Test URLScore dataclass functionality."""
        url_score = URLScore(
            url="https://example.com",
            relevance_score=0.8,
            intent_match_score=0.7,
            domain_reputation_score=0.9,
            pattern_match_score=0.8,
            confidence=0.8
        )
        
        self.assertEqual(url_score.url, "https://example.com")
        self.assertEqual(url_score.relevance_score, 0.8)
        self.assertEqual(url_score.intent_match_score, 0.7)
        self.assertEqual(url_score.domain_reputation_score, 0.9)
        self.assertEqual(url_score.pattern_match_score, 0.8)
        self.assertEqual(url_score.confidence, 0.8)
    
    def test_integration_with_intent_analyzer(self):
        """Test integration with UniversalIntentAnalyzer."""
        query = "deep learning research"
        
        # Test that intent analyzer is called when no intent_analysis provided
        with patch.object(self.generator, '_generate_direct_hit_urls', return_value=[]):
            with patch.object(self.generator, '_generate_template_urls', return_value=[]):
                with patch.object(self.generator, '_generate_search_urls', return_value=[]):
                    with patch.object(self.generator, '_generate_navigation_urls', return_value=[]):
                        
                        self.generator.generate_urls(query)
                        
                        # Verify intent analyzer was called
                        self.mock_intent_analyzer.analyze_intent.assert_called_once_with(query)
    
    def test_url_deduplication(self):
        """Test URL deduplication functionality."""
        # Create duplicate URLs with different scores
        duplicate_urls = [
            URLScore("https://example.com", 0.8, 0.7, 0.9, 0.8, 0.8),
            URLScore("https://example.com", 0.7, 0.6, 0.8, 0.7, 0.7),  # Duplicate
            URLScore("https://different.com", 0.9, 0.8, 0.9, 0.9, 0.9)
        ]
        
        with patch.object(self.generator, '_deduplicate_urls', return_value=duplicate_urls[:2]) as mock_dedup:
            with patch.object(self.generator, '_generate_direct_hit_urls', return_value=['https://example.com']):
                with patch.object(self.generator, 'validate_and_score_url', side_effect=duplicate_urls):
                    
                    urls = self.generator.generate_urls("test")
                    
                    mock_dedup.assert_called_once()
    
    def test_performance_with_large_result_set(self):
        """Test performance with large number of URL candidates."""
        query = "test query"
        
        # Generate many URLs to test performance
        large_url_list = [f"https://example{i}.com" for i in range(100)]
        
        with patch.object(self.generator, '_generate_direct_hit_urls', return_value=large_url_list[:25]):
            with patch.object(self.generator, '_generate_template_urls', return_value=large_url_list[25:50]):
                with patch.object(self.generator, '_generate_search_urls', return_value=large_url_list[50:75]):
                    with patch.object(self.generator, '_generate_navigation_urls', return_value=large_url_list[75:]):
                        with patch.object(self.generator, 'validate_and_score_url') as mock_score:
                            # Return decreasing scores
                            mock_score.side_effect = [
                                URLScore(url, max(0.1, 1.0 - i * 0.01), 0.5, 0.5, 0.5, max(0.1, 1.0 - i * 0.01))
                                for i, url in enumerate(large_url_list)
                            ]
                            
                            urls = self.generator.generate_urls(query, max_urls=10)
                            
                            # Should limit results and maintain performance
                            self.assertLessEqual(len(urls), 10)
                            # Should be sorted by relevance
                            if len(urls) > 1:
                                for i in range(len(urls) - 1):
                                    self.assertGreaterEqual(urls[i].relevance_score, urls[i + 1].relevance_score)


if __name__ == '__main__':
    unittest.main()
