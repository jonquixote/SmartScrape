"""
Content Quality Scorer with Semantic Relevance

This module provides content quality assessment capabilities with semantic
understanding for evaluating content relevance to user queries and overall
textual quality using spaCy and other NLP techniques.
"""

import logging
import re
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

try:
    import spacy  # Import the base spacy module
    from spacy.tokens import Doc, Span  # Import specific types
    # Test if spaCy can actually load a model
    try:
        # Load spaCy with model priority
        SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
        for model_name in SPACY_MODELS:
            try:
                nlp = spacy.load(model_name)
                break
            except OSError:
                continue
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        spacy = None
        Doc = type(None)
        Span = type(None)
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None  # The module itself is not available
    Doc = type(None)  # Placeholder for type checking if needed, or ensure code checks SPACY_AVAILABLE
    Span = type(None)  # Placeholder

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from config import SPACY_MODEL_NAME, SEMANTIC_SIMILARITY_THRESHOLD

# Configure logging
logger = logging.getLogger(__name__)

if not SPACY_AVAILABLE:
    logger.info("spaCy not available. Advanced NLP features in ContentQualityScorer will be disabled.")
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    logger.info("SentenceTransformers not available. Semantic similarity features will be limited.")


class ContentQualityScorer:
    """
    Analyzes content quality with semantic relevance scoring.
    
    This class provides:
    - Content quality assessment based on linguistic features
    - Semantic similarity scoring to user queries
    - Readability and structure analysis
    - Duplicate content detection using semantic similarity
    """
    
    def __init__(self, intent_analyzer=None):
        """
        Initialize the content quality scorer.
        
        Args:
            intent_analyzer: Optional UniversalIntentAnalyzer for query context
        """
        self.intent_analyzer = intent_analyzer
        self.nlp = None
        self.sentence_model = None
        
        # Store availability flags as instance variables
        self.SENTENCE_TRANSFORMERS_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE
        self.SPACY_AVAILABLE = SPACY_AVAILABLE
        
        # Initialize spaCy if available
        if self.SPACY_AVAILABLE and spacy:
            try:
                # Try models in order of preference
                for model_name in ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']:
                    try:
                        self.nlp = spacy.load(model_name)
                        logger.info(f"spaCy model '{model_name}' loaded for ContentQualityScorer.")
                        break
                    except OSError:
                        continue
                        
                if self.nlp is None:
                    logger.error("Could not load any spaCy model. Please install at least en_core_web_sm. Disabling spaCy features.")
            except Exception as e:
                logger.error(f"Unexpected error loading spaCy model: {e}. Disabling spaCy features.")
                self.nlp = None
        
        # Initialize sentence transformer if available
        if self.SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        # Quality scoring weights
        self.quality_weights = {
            'length': 0.2,
            'readability': 0.25,
            'structure': 0.2,
            'informativeness': 0.15,
            'semantic_relevance': 0.2
        }
    
    def score_content(self, content: str, query: Optional[str] = None) -> float:
        """
        Return quality score 0-1, considering textual quality and relevance to query.
        
        Args:
            content: Text content to score
            query: Optional query for relevance scoring
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not content or len(content.strip()) < 10:
            return 0.0
        
        try:
            scores = {}
            
            # Basic content quality metrics
            scores['length'] = self._score_content_length(content)
            scores['readability'] = self._score_readability(content)
            scores['structure'] = self._score_structure(content)
            scores['informativeness'] = self._score_informativeness(content)
            
            # Semantic relevance to query if provided
            if query and (self.SPACY_AVAILABLE or self.SENTENCE_TRANSFORMERS_AVAILABLE):
                scores['semantic_relevance'] = self._score_semantic_relevance(content, query)
            else:
                scores['semantic_relevance'] = 0.5  # Neutral score if no query
            
            # Calculate weighted total
            total_score = sum(
                scores.get(metric, 0.0) * weight 
                for metric, weight in self.quality_weights.items()
            )
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            logger.error(f"Error scoring content: {e}")
            return 0.0
    
    def score_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using spaCy or sentence transformers.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Use sentence transformers if available (more accurate)
            if self.SENTENCE_TRANSFORMERS_AVAILABLE and self.sentence_model:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = float(embeddings[0] @ embeddings[1].T)
                return max(0.0, min(1.0, similarity))
            
            # Fallback to spaCy similarity
            elif self.SPACY_AVAILABLE and self.nlp:
                doc1 = self.nlp(text1[:1000])  # Limit text length for performance
                doc2 = self.nlp(text2[:1000])
                
                if doc1.vector_norm > 0 and doc2.vector_norm > 0:
                    similarity = doc1.similarity(doc2)
                    return max(0.0, min(1.0, similarity))
            
            # Basic lexical similarity fallback
            return self._lexical_similarity(text1, text2)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def detect_duplicate_content(self, contents: List[str], threshold: float = None) -> List[List[int]]:
        """
        Detect duplicate or near-duplicate content using semantic similarity.
        
        Args:
            contents: List of text contents to compare
            threshold: Similarity threshold for duplicate detection
            
        Returns:
            List of lists, where each inner list contains indices of similar contents
        """
        if threshold is None:
            threshold = SEMANTIC_SIMILARITY_THRESHOLD or 0.85
        
        if len(contents) <= 1:
            return []
        
        try:
            # Calculate similarity matrix
            similarity_matrix = []
            for i in range(len(contents)):
                row = []
                for j in range(len(contents)):
                    if i == j:
                        row.append(1.0)
                    elif j < i:
                        # Use symmetric property
                        row.append(similarity_matrix[j][i])
                    else:
                        similarity = self.score_semantic_similarity(contents[i], contents[j])
                        row.append(similarity)
                similarity_matrix.append(row)
            
            # Find groups of similar content
            duplicate_groups = []
            processed = set()
            
            for i in range(len(contents)):
                if i in processed:
                    continue
                
                group = [i]
                for j in range(i + 1, len(contents)):
                    if j not in processed and similarity_matrix[i][j] >= threshold:
                        group.append(j)
                        processed.add(j)
                
                if len(group) > 1:
                    duplicate_groups.append(group)
                    processed.update(group)
            
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error detecting duplicate content: {e}")
            return []
    
    def filter_low_quality_content(self, contents: List[str], 
                                 min_score: float = 0.3,
                                 query: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Filter out low-quality content based on quality scores.
        
        Args:
            contents: List of text contents to filter
            min_score: Minimum quality score threshold
            query: Optional query for relevance scoring
            
        Returns:
            List of tuples (content, score) for contents above threshold
        """
        try:
            scored_contents = []
            
            for content in contents:
                score = self.score_content(content, query)
                if score >= min_score:
                    scored_contents.append((content, score))
            
            # Sort by score (highest first)
            scored_contents.sort(key=lambda x: x[1], reverse=True)
            return scored_contents
            
        except Exception as e:
            logger.error(f"Error filtering content: {e}")
            return [(content, 0.0) for content in contents]
    
    def get_detailed_scores(self, content: str, query: Optional[str] = None) -> Dict[str, float]:
        """
        Get detailed quality scores for all metrics.
        
        Args:
            content: Text content to score
            query: Optional query for relevance scoring
            
        Returns:
            Dictionary with detailed scores for each metric
        """
        if not content or len(content.strip()) < 10:
            return {metric: 0.0 for metric in self.quality_weights.keys()}
        
        try:
            scores = {}
            
            # Basic content quality metrics
            scores['length'] = self._score_content_length(content)
            scores['readability'] = self._score_readability(content)
            scores['structure'] = self._score_structure(content)
            scores['informativeness'] = self._score_informativeness(content)
            
            # Semantic relevance to query if provided
            if query and (self.SPACY_AVAILABLE or self.SENTENCE_TRANSFORMERS_AVAILABLE):
                scores['semantic_relevance'] = self._score_semantic_relevance(content, query)
            else:
                scores['semantic_relevance'] = 0.5  # Neutral score if no query
            
            # Calculate overall score
            overall_score = sum(
                scores.get(metric, 0.0) * weight 
                for metric, weight in self.quality_weights.items()
            )
            scores['overall_score'] = min(1.0, max(0.0, overall_score))
            
            return scores
            
        except Exception as e:
            logger.error(f"Error getting detailed scores: {e}")
            return {metric: 0.0 for metric in self.quality_weights.keys()}
    
    def _score_content_length(self, content: str) -> float:
        """Score content based on length (sweet spot around 500-2000 characters)."""
        length = len(content.strip())
        
        if length < 50:
            return 0.1
        elif length < 200:
            return 0.5
        elif length < 500:
            return 0.7
        elif length < 2000:
            return 1.0
        elif length < 5000:
            return 0.8
        else:
            return 0.6  # Too long might be less focused
    
    def _score_readability(self, content: str) -> float:
        """Score content readability using simple metrics."""
        try:
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            if not sentences or not words:
                return 0.0
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Readability score (simple approximation of Flesch Reading Ease)
            if avg_sentence_length == 0:
                return 0.0
            
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, score / 100.0))
            
        except Exception:
            return 0.5
    
    def _score_structure(self, content: str) -> float:
        """Score content structure (paragraphs, headings, etc.)."""
        try:
            score = 0.0
            
            # Check for paragraph breaks
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                score += 0.3
            
            # Check for proper sentence structure
            sentences = re.split(r'[.!?]+', content)
            if len(sentences) > 2:
                score += 0.2
            
            # Check for variety in sentence length
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths and len(set(sentence_lengths)) > 1:
                score += 0.3
            
            # Check for proper capitalization
            if content and content[0].isupper():
                score += 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _score_informativeness(self, content: str) -> float:
        """Score content informativeness using linguistic features."""
        try:
            if not SPACY_AVAILABLE or not self.nlp:
                return self._basic_informativeness_score(content)
            
            doc = self.nlp(content[:1000])  # Limit for performance
            
            # Count informative parts of speech
            informative_pos = {'NOUN', 'VERB', 'ADJ', 'NUM'}
            total_tokens = len([token for token in doc if not token.is_punct and not token.is_space])
            informative_tokens = len([token for token in doc if token.pos_ in informative_pos])
            
            if total_tokens == 0:
                return 0.0
            
            informativeness_ratio = informative_tokens / total_tokens
            
            # Count named entities (indicates specific information)
            entity_density = len(doc.ents) / max(1, len(doc))
            
            # Combine metrics
            score = (informativeness_ratio * 0.7) + (min(entity_density * 10, 1.0) * 0.3)
            
            return min(1.0, max(0.0, score))
            
        except Exception:
            return self._basic_informativeness_score(content)
    
    def _basic_informativeness_score(self, content: str) -> float:
        """Basic informativeness scoring without spaCy."""
        try:
            words = content.lower().split()
            
            # Count unique words
            unique_ratio = len(set(words)) / max(1, len(words))
            
            # Check for numbers (often indicate specific information)
            number_ratio = len([w for w in words if any(c.isdigit() for c in w)]) / max(1, len(words))
            
            # Combine metrics
            score = (unique_ratio * 0.7) + (min(number_ratio * 5, 1.0) * 0.3)
            
            return min(1.0, max(0.0, score))
            
        except Exception:
            return 0.5
    
    def _score_semantic_relevance(self, content: str, query: str) -> float:
        """Score semantic relevance to query using available NLP tools."""
        try:
            if self.intent_analyzer:
                # Use intent analyzer for enhanced context understanding
                intent_analysis = self.intent_analyzer.analyze_intent(query)
                expanded_queries = self.intent_analyzer.expand_query_contextually(
                    query, intent_analysis
                )
                
                # Calculate relevance score for each expanded query and take the maximum
                max_relevance = 0.0
                for expanded_query in expanded_queries:
                    if expanded_query:  # Skip empty queries
                        relevance_score = self.score_semantic_similarity(content, expanded_query)
                        max_relevance = max(max_relevance, relevance_score)
                
                return max_relevance
            else:
                # Direct similarity comparison
                return self.score_semantic_similarity(content, query)
                
        except Exception as e:
            logger.error(f"Error scoring semantic relevance: {e}")
            return 0.5
    
    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """Basic lexical similarity as fallback when advanced models aren't available."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
