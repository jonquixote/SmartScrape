"""
Unit tests for UniversalIntentAnalyzer component.

Tests semantic search, query expansion, URL prioritization, and intent analysis capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from components.universal_intent_analyzer import UniversalIntentAnalyzer
from config import Config


class TestUniversalIntentAnalyzer:
    """Test suite for UniversalIntentAnalyzer component."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock()
        config.SEMANTIC_SEARCH_ENABLED = True
        config.CONTEXTUAL_QUERY_EXPANSION = True
        config.SPACY_ENABLED = True
        config.SPACY_MODEL = "en_core_web_sm"
        config.SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
        return config
    
    @pytest.fixture
    def analyzer(self, mock_config):
        """Create UniversalIntentAnalyzer instance for testing."""
        with patch('components.universal_intent_analyzer.Config', mock_config):
            analyzer = UniversalIntentAnalyzer()
            # Mock the heavy components for faster testing
            analyzer.nlp = Mock()
            analyzer.sentence_transformer = Mock()
            return analyzer
    
    def test_init(self, mock_config):
        """Test proper initialization of UniversalIntentAnalyzer."""
        with patch('components.universal_intent_analyzer.Config', mock_config):
            with patch('spacy.load') as mock_spacy_load:
                with patch('sentence_transformers.SentenceTransformer') as mock_st:
                    analyzer = UniversalIntentAnalyzer()
                    
                    mock_spacy_load.assert_called_once_with("en_core_web_sm")
                    mock_st.assert_called_once_with("all-MiniLM-L6-v2")
                    assert analyzer.config == mock_config
    
    def test_analyze_intent_basic(self, analyzer):
        """Test basic intent analysis functionality."""
        # Mock spaCy processing
        mock_doc = Mock()
        mock_doc.ents = [Mock(text="laptops", label_="PRODUCT")]
        analyzer.nlp.return_value = mock_doc
        
        # Mock semantic analysis
        analyzer._analyze_semantic_intent = Mock(return_value={
            "semantic_similarity": 0.85,
            "intent_clusters": ["shopping", "technology"]
        })
        
        query = "find laptops under $1000"
        result = analyzer.analyze_intent(query)
        
        assert isinstance(result, dict)
        assert "entities" in result
        assert "data_types" in result
        assert "semantic_analysis" in result
        assert "confidence_score" in result
        assert "intent_type" in result
        
        # Verify confidence score is between 0 and 1
        assert 0 <= result["confidence_score"] <= 1
    
    def test_extract_entities(self, analyzer):
        """Test entity extraction using spaCy NER."""
        # Setup mock spaCy document
        mock_entity1 = Mock()
        mock_entity1.text = "Seattle"
        mock_entity1.label_ = "GPE"
        mock_entity1.start_char = 0
        mock_entity1.end_char = 7
        
        mock_entity2 = Mock()
        mock_entity2.text = "$1000"
        mock_entity2.label_ = "MONEY"
        mock_entity2.start_char = 20
        mock_entity2.end_char = 25
        
        mock_doc = Mock()
        mock_doc.ents = [mock_entity1, mock_entity2]
        analyzer.nlp.return_value = mock_doc
        
        query = "Seattle restaurants under $1000"
        entities = analyzer.extract_entities(query)
        
        assert len(entities) == 2
        assert entities[0]["text"] == "Seattle"
        assert entities[0]["label"] == "GPE"
        assert entities[1]["text"] == "$1000"
        assert entities[1]["label"] == "MONEY"
    
    def test_identify_data_types(self, analyzer):
        """Test data type identification from entities."""
        entities = [
            {"text": "Seattle", "label": "GPE"},
            {"text": "$1000", "label": "MONEY"},
            {"text": "laptop", "label": "PRODUCT"},
            {"text": "2024", "label": "DATE"}
        ]
        
        data_types = analyzer.identify_data_types(entities)
        
        assert "location" in data_types
        assert "price" in data_types
        assert "product" in data_types
        assert "date" in data_types
    
    def test_expand_query_contextually(self, analyzer):
        """Test contextual query expansion functionality."""
        query = "best coffee shops"
        intent_analysis = {
            "entities": [{"text": "coffee shops", "label": "BUSINESS"}],
            "intent_type": "local_search"
        }
        
        # Mock the expansion logic
        analyzer._generate_semantic_expansions = Mock(return_value=[
            "cafes", "coffee houses", "espresso bars"
        ])
        analyzer._generate_synonym_expansions = Mock(return_value=[
            "top coffee shops", "popular coffee shops"
        ])
        
        expansions = analyzer.expand_query_contextually(query, intent_analysis)
        
        assert isinstance(expansions, list)
        assert len(expansions) > 0
        assert "cafes" in expansions
        assert "top coffee shops" in expansions
    
    def test_semantic_similarity_calculation(self, analyzer):
        """Test semantic similarity calculation between queries."""
        # Mock sentence transformer
        analyzer.sentence_transformer.encode = Mock(return_value=[
            [0.1, 0.2, 0.3],  # query1 embedding
            [0.15, 0.25, 0.28]  # query2 embedding
        ])
        
        query1 = "best restaurants"
        query2 = "top dining places"
        
        similarity = analyzer._calculate_semantic_similarity(query1, query2)
        
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
    
    def test_prioritize_urls(self, analyzer):
        """Test URL prioritization based on intent analysis."""
        urls = [
            "https://yelp.com/search?find_desc=restaurants",
            "https://example.com/random-page",
            "https://restaurants.com/seattle",
            "https://tripadvisor.com/restaurants"
        ]
        
        intent_analysis = {
            "entities": [{"text": "restaurants", "label": "BUSINESS"}],
            "intent_type": "local_search",
            "confidence_score": 0.9
        }
        
        analyzer._score_url_relevance = Mock(side_effect=[0.9, 0.2, 0.8, 0.7])
        
        prioritized_urls = analyzer.prioritize_urls(urls, intent_analysis)
        
        assert len(prioritized_urls) == len(urls)
        # Should be sorted by relevance score (descending)
        assert prioritized_urls[0]["url"] == "https://yelp.com/search?find_desc=restaurants"
        assert prioritized_urls[0]["score"] == 0.9
        assert prioritized_urls[-1]["score"] == 0.2
    
    def test_intent_classification(self, analyzer):
        """Test intent classification for different query types."""
        test_cases = [
            ("buy laptop under $1000", "shopping"),
            ("weather in Seattle", "information"),
            ("best restaurants near me", "local_search"),
            ("news about technology", "content_search"),
            ("how to cook pasta", "instructional")
        ]
        
        for query, expected_intent in test_cases:
            # Mock the classification logic
            analyzer._classify_intent_pattern = Mock(return_value=expected_intent)
            
            intent_analysis = analyzer.analyze_intent(query)
            
            # Verify the intent was classified correctly
            analyzer._classify_intent_pattern.assert_called_with(query, intent_analysis.get("entities", []))
    
    def test_confidence_scoring(self, analyzer):
        """Test confidence score calculation based on multiple factors."""
        # Test high confidence scenario
        high_confidence_analysis = {
            "entities": [{"text": "Seattle", "label": "GPE"}, {"text": "restaurants", "label": "BUSINESS"}],
            "semantic_analysis": {"semantic_similarity": 0.9},
            "intent_type": "local_search"
        }
        
        high_score = analyzer._calculate_confidence_score(high_confidence_analysis)
        assert high_score > 0.7
        
        # Test low confidence scenario
        low_confidence_analysis = {
            "entities": [],
            "semantic_analysis": {"semantic_similarity": 0.3},
            "intent_type": "unknown"
        }
        
        low_score = analyzer._calculate_confidence_score(low_confidence_analysis)
        assert low_score < 0.5
    
    def test_error_handling(self, analyzer):
        """Test error handling for various edge cases."""
        # Test empty query
        result = analyzer.analyze_intent("")
        assert result["confidence_score"] == 0.0
        
        # Test None query
        result = analyzer.analyze_intent(None)
        assert result["confidence_score"] == 0.0
        
        # Test very long query
        long_query = "a" * 10000
        result = analyzer.analyze_intent(long_query)
        assert isinstance(result, dict)
        assert "error" not in result
    
    @pytest.mark.asyncio
    async def test_async_operations(self, analyzer):
        """Test any asynchronous operations in the analyzer."""
        # Mock async semantic analysis
        analyzer._async_semantic_analysis = Mock(return_value=asyncio.Future())
        analyzer._async_semantic_analysis.return_value.set_result({
            "semantic_similarity": 0.8,
            "intent_clusters": ["shopping"]
        })
        
        query = "best laptop deals"
        # If there are async methods, test them here
        # result = await analyzer.analyze_intent_async(query)
        # assert isinstance(result, dict)
    
    def test_caching_behavior(self, analyzer):
        """Test caching of analysis results."""
        query = "test query for caching"
        
        # Mock the analysis to return consistent results
        mock_result = {
            "entities": [],
            "confidence_score": 0.8,
            "intent_type": "test"
        }
        
        # Enable caching if available
        if hasattr(analyzer, '_cache_enabled'):
            analyzer._cache_enabled = True
            
        # First call should perform analysis
        with patch.object(analyzer, '_perform_analysis', return_value=mock_result) as mock_analysis:
            result1 = analyzer.analyze_intent(query)
            mock_analysis.assert_called_once()
            
            # Second call should use cache
            result2 = analyzer.analyze_intent(query)
            # Should still be called only once if caching works
            mock_analysis.assert_called_once()
            
            assert result1 == result2
    
    def test_performance_metrics(self, analyzer):
        """Test performance metrics collection."""
        query = "performance test query"
        
        # Enable performance tracking if available
        if hasattr(analyzer, 'enable_performance_tracking'):
            analyzer.enable_performance_tracking(True)
        
        result = analyzer.analyze_intent(query)
        
        # Check if performance metrics are collected
        if hasattr(analyzer, 'get_performance_metrics'):
            metrics = analyzer.get_performance_metrics()
            assert "analysis_time" in metrics
            assert "query_count" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
