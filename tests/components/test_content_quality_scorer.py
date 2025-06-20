"""
Unit tests for ContentQualityScorer component.

This test suite validates the content quality scoring functionality including:
- Content quality assessment and scoring
- Semantic similarity calculation
- Duplicate content detection
- Integration with spaCy and sentence transformers
- Query relevance scoring
- Error handling and edge cases
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from processors.content_quality_scorer import ContentQualityScorer


class TestContentQualityScorer(unittest.TestCase):
    """Test suite for ContentQualityScorer component."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock intent analyzer
        self.mock_intent_analyzer = Mock()
        self.mock_intent_analyzer.analyze_intent.return_value = {
            'intent_type': 'information_seeking',
            'keywords': ['test', 'content'],
            'entities': [],
            'query_complexity': 0.6
        }
        
        # Test content samples
        self.high_quality_content = """
        This is a well-structured article about machine learning and artificial intelligence.
        It contains comprehensive information about neural networks, deep learning algorithms,
        and their applications in various industries. The content is informative, well-formatted,
        and provides detailed explanations with examples. It includes proper sentence structure,
        appropriate paragraph breaks, and maintains consistent quality throughout the text.
        """
        
        self.low_quality_content = "THIS IS ALL CAPS TEXT!!! BAD QUALITY!!!"
        
        self.short_content = "Hi"
        
        self.empty_content = ""
        
        self.html_content = "<div>Some <span>HTML</span> content with <a href='#'>tags</a></div>"
        
        self.structured_content = """
        # Title
        ## Subtitle
        
        - Item 1
        - Item 2
        - Item 3
        
        Paragraph with good structure.
        """
        
    def test_initialization_success(self):
        """Test successful initialization of ContentQualityScorer."""
        # Test initialization without intent analyzer
        scorer = ContentQualityScorer()
        self.assertIsNotNone(scorer)
        self.assertIsNone(scorer.intent_analyzer)
        self.assertIsInstance(scorer.quality_weights, dict)
        
        # Test initialization with intent analyzer
        scorer_with_analyzer = ContentQualityScorer(self.mock_intent_analyzer)
        self.assertIsNotNone(scorer_with_analyzer)
        self.assertEqual(scorer_with_analyzer.intent_analyzer, self.mock_intent_analyzer)
    
    def test_initialization_with_dependencies(self):
        """Test initialization with spaCy and sentence transformers availability."""
        # Mock successful spaCy loading
        with patch('processors.content_quality_scorer.SPACY_AVAILABLE', True), \
             patch('processors.content_quality_scorer.spacy') as mock_spacy:
            
            mock_nlp = Mock()
            mock_spacy.load.return_value = mock_nlp
            
            scorer = ContentQualityScorer()
            self.assertEqual(scorer.nlp, mock_nlp)
    
    def test_initialization_spacy_failure(self):
        """Test initialization when spaCy loading fails."""
        with patch('processors.content_quality_scorer.SPACY_AVAILABLE', True), \
             patch('processors.content_quality_scorer.spacy') as mock_spacy:
            
            mock_spacy.load.side_effect = OSError("Model not found")
            
            scorer = ContentQualityScorer()
            self.assertIsNone(scorer.nlp)
    
    def test_score_content_high_quality(self):
        """Test scoring of high-quality content."""
        scorer = ContentQualityScorer()
        score = scorer.score_content(self.high_quality_content)
        
        # High-quality content should score well
        self.assertGreater(score, 0.6)
        self.assertLessEqual(score, 1.0)
    
    def test_score_content_low_quality(self):
        """Test scoring of low-quality content."""
        scorer = ContentQualityScorer()
        score = scorer.score_content(self.low_quality_content)
        
        # Low-quality content should score poorly
        self.assertLess(score, 0.6)
        self.assertGreaterEqual(score, 0.0)
    
    def test_score_content_edge_cases(self):
        """Test scoring of edge case content."""
        scorer = ContentQualityScorer()
        
        # Empty content
        score = scorer.score_content(self.empty_content)
        self.assertEqual(score, 0.0)
        
        # Very short content
        score = scorer.score_content(self.short_content)
        self.assertEqual(score, 0.0)
        
        # None content
        score = scorer.score_content(None)
        self.assertEqual(score, 0.0)
        
        # Whitespace only
        score = scorer.score_content("   \n\t   ")
        self.assertEqual(score, 0.0)
    
    def test_score_content_with_query(self):
        """Test content scoring with query relevance."""
        scorer = ContentQualityScorer()
        
        # Content relevant to query should score higher
        relevant_content = "Machine learning algorithms and neural networks for AI applications"
        query = "machine learning AI"
        
        score_with_query = scorer.score_content(relevant_content, query)
        score_without_query = scorer.score_content(relevant_content)
        
        # Scores should be valid
        self.assertGreaterEqual(score_with_query, 0.0)
        self.assertLessEqual(score_with_query, 1.0)
        self.assertGreaterEqual(score_without_query, 0.0)
        self.assertLessEqual(score_without_query, 1.0)
    
    def test_score_content_html_handling(self):
        """Test scoring of content with HTML tags."""
        scorer = ContentQualityScorer()
        score = scorer.score_content(self.html_content)
        
        # HTML content should be scored (structure metric should detect issues)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.8)  # Should be penalized for HTML tags
    
    def test_score_semantic_similarity_basic(self):
        """Test basic semantic similarity calculation."""
        scorer = ContentQualityScorer()
        
        # Similar texts
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on a mat"
        similarity = scorer.score_semantic_similarity(text1, text2)
        
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        
        # Identical texts
        similarity_identical = scorer.score_semantic_similarity(text1, text1)
        self.assertGreater(similarity_identical, 0.8)
        
        # Completely different texts
        text3 = "Quantum physics equations"
        similarity_different = scorer.score_semantic_similarity(text1, text3)
        self.assertLess(similarity_different, similarity)
    
    def test_score_semantic_similarity_edge_cases(self):
        """Test semantic similarity with edge cases."""
        scorer = ContentQualityScorer()
        
        # Empty strings
        similarity = scorer.score_semantic_similarity("", "some text")
        self.assertEqual(similarity, 0.0)
        
        similarity = scorer.score_semantic_similarity("", "")
        self.assertEqual(similarity, 0.0)
        
        # None inputs
        similarity = scorer.score_semantic_similarity(None, "some text")
        self.assertEqual(similarity, 0.0)
    
    @patch('processors.content_quality_scorer.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    def test_semantic_similarity_with_sentence_transformers(self):
        """Test semantic similarity with sentence transformers."""
        mock_model = Mock()
        mock_embeddings = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        mock_model.encode.return_value = mock_embeddings
        
        scorer = ContentQualityScorer()
        scorer.sentence_model = mock_model
        
        similarity = scorer.score_semantic_similarity("text1", "text2")
        
        mock_model.encode.assert_called_once_with(["text1", "text2"])
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    @patch('processors.content_quality_scorer.SPACY_AVAILABLE', True)
    def test_semantic_similarity_with_spacy(self):
        """Test semantic similarity with spaCy."""
        mock_nlp = Mock()
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        
        # Set up mock documents with vector properties
        mock_doc1.vector_norm = 1.0
        mock_doc2.vector_norm = 1.0
        mock_doc1.similarity.return_value = 0.7
        
        mock_nlp.side_effect = [mock_doc1, mock_doc2]
        
        scorer = ContentQualityScorer()
        scorer.nlp = mock_nlp
        
        similarity = scorer.score_semantic_similarity("text1", "text2")
        
        self.assertEqual(similarity, 0.7)
    
    def test_detect_duplicate_content_basic(self):
        """Test basic duplicate content detection."""
        scorer = ContentQualityScorer()
        
        contents = [
            "This is original content",
            "This is completely different content about different topics",
            "This is original content with minor changes",  # Similar to first
            "Unique content here"
        ]
        
        duplicates = scorer.detect_duplicate_content(contents, threshold=0.8)
        
        # Should return list of lists
        self.assertIsInstance(duplicates, list)
        
        # Each element should be a list of indices
        for duplicate_group in duplicates:
            self.assertIsInstance(duplicate_group, list)
            for index in duplicate_group:
                self.assertIsInstance(index, int)
                self.assertLess(index, len(contents))
    
    def test_detect_duplicate_content_edge_cases(self):
        """Test duplicate content detection with edge cases."""
        scorer = ContentQualityScorer()
        
        # Empty list
        duplicates = scorer.detect_duplicate_content([])
        self.assertEqual(duplicates, [])
        
        # Single item
        duplicates = scorer.detect_duplicate_content(["single content"])
        self.assertEqual(duplicates, [])
        
        # Two identical items
        duplicates = scorer.detect_duplicate_content(["same content", "same content"])
        self.assertIsInstance(duplicates, list)
    
    def test_filter_low_quality_content(self):
        """Test filtering of low-quality content."""
        scorer = ContentQualityScorer()
        
        contents = [
            self.high_quality_content,
            self.low_quality_content,
            self.short_content,
            self.structured_content
        ]
        
        filtered = scorer.filter_low_quality_content(contents, min_score=0.3)
        
        # Should return list of tuples (content, score)
        self.assertIsInstance(filtered, list)
        for content, score in filtered:
            self.assertIsInstance(content, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.3)
    
    def test_filter_low_quality_content_with_query(self):
        """Test filtering with query relevance."""
        scorer = ContentQualityScorer()
        
        contents = [
            "Machine learning and AI technologies",
            "Cooking recipes and food preparation",
            "Neural networks for artificial intelligence"
        ]
        
        filtered = scorer.filter_low_quality_content(
            contents, 
            min_score=0.2, 
            query="machine learning AI"
        )
        
        # Should filter based on both quality and relevance
        self.assertIsInstance(filtered, list)
        self.assertGreaterEqual(len(filtered), 0)
    
    def test_private_score_methods(self):
        """Test private scoring methods."""
        scorer = ContentQualityScorer()
        
        # Test length scoring
        length_score = scorer._score_content_length(self.high_quality_content)
        self.assertGreaterEqual(length_score, 0.0)
        self.assertLessEqual(length_score, 1.0)
        
        # Test readability scoring
        readability_score = scorer._score_readability(self.high_quality_content)
        self.assertGreaterEqual(readability_score, 0.0)
        self.assertLessEqual(readability_score, 1.0)
        
        # Test structure scoring
        structure_score = scorer._score_structure(self.structured_content)
        self.assertGreaterEqual(structure_score, 0.0)
        self.assertLessEqual(structure_score, 1.0)
        
        # Test informativeness scoring
        info_score = scorer._score_informativeness(self.high_quality_content)
        self.assertGreaterEqual(info_score, 0.0)
        self.assertLessEqual(info_score, 1.0)
    
    def test_lexical_similarity_fallback(self):
        """Test lexical similarity fallback method."""
        scorer = ContentQualityScorer()
        
        # Similar texts should have higher lexical similarity
        text1 = "machine learning algorithms"
        text2 = "learning algorithms machine"
        text3 = "cooking food recipes"
        
        similarity_high = scorer._lexical_similarity(text1, text2)
        similarity_low = scorer._lexical_similarity(text1, text3)
        
        self.assertGreater(similarity_high, similarity_low)
        self.assertGreaterEqual(similarity_high, 0.0)
        self.assertLessEqual(similarity_high, 1.0)
        self.assertGreaterEqual(similarity_low, 0.0)
        self.assertLessEqual(similarity_low, 1.0)
    
    def test_error_handling(self):
        """Test error handling in scoring methods."""
        scorer = ContentQualityScorer()
        
        # Mock an error in the scoring process
        with patch.object(scorer, '_score_content_length', side_effect=Exception("Test error")):
            score = scorer.score_content(self.high_quality_content)
            self.assertEqual(score, 0.0)  # Should return 0.0 on error
        
        # Test semantic similarity error handling
        with patch.object(scorer, '_score_semantic_relevance', side_effect=Exception("Test error")):
            score = scorer.score_semantic_similarity("text1", "text2")
            self.assertEqual(score, 0.0)  # Should return 0.0 on error
    
    def test_quality_weights_configuration(self):
        """Test that quality weights are properly configured."""
        scorer = ContentQualityScorer()
        
        # Check that all expected weight categories exist
        expected_weights = ['length', 'readability', 'structure', 'informativeness', 'semantic_relevance']
        for weight in expected_weights:
            self.assertIn(weight, scorer.quality_weights)
            self.assertGreaterEqual(scorer.quality_weights[weight], 0.0)
            self.assertLessEqual(scorer.quality_weights[weight], 1.0)
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(scorer.quality_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=1)
    
    def test_integration_with_intent_analyzer(self):
        """Test integration with UniversalIntentAnalyzer."""
        scorer = ContentQualityScorer(self.mock_intent_analyzer)
        
        # Verify intent analyzer is properly set
        self.assertEqual(scorer.intent_analyzer, self.mock_intent_analyzer)
        
        # Test that intent analyzer methods can be called
        if hasattr(scorer.intent_analyzer, 'analyze_intent'):
            result = scorer.intent_analyzer.analyze_intent("test query")
            self.assertIsInstance(result, dict)
    
    def test_performance_with_large_content(self):
        """Test performance with large content."""
        scorer = ContentQualityScorer()
        
        # Create large content (but not too large for testing)
        large_content = self.high_quality_content * 100  # Repeat content
        
        # Should handle large content without crashing
        score = scorer.score_content(large_content)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_content_normalization(self):
        """Test content normalization and preprocessing."""
        scorer = ContentQualityScorer()
        
        # Content with extra whitespace
        messy_content = "  \n\n  This is content with   extra    whitespace  \n\n  "
        score = scorer.score_content(messy_content)
        
        # Should handle messy content gracefully
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == '__main__':
    unittest.main()
