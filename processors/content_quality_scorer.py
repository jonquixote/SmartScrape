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
    
    # ===== Content-Type Specific Quality Validation Methods =====
    
    def validate_news_article(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate news article content with news-specific quality criteria.
        
        Args:
            content: Extracted news content with fields like title, content, author, date
            
        Returns:
            Dictionary with validation results and quality scores
        """
        validation_result = {
            'is_valid': False,
            'quality_score': 0.0,
            'validation_details': {},
            'issues': []
        }
        
        details = validation_result['validation_details']
        issues = validation_result['issues']
        
        # Check required fields for news articles
        if not content.get('title'):
            issues.append("Missing article title")
            details['has_title'] = False
        else:
            details['has_title'] = True
            details['title_length'] = len(content['title'])
            
        if not content.get('content'):
            issues.append("Missing article content")
            details['has_content'] = False
        else:
            details['has_content'] = True
            details['content_length'] = len(content['content'])
            
        # News-specific quality checks
        details['has_author'] = bool(content.get('author'))
        details['has_date'] = bool(content.get('date'))
        details['has_summary'] = bool(content.get('summary'))
        
        # Check for substantial content (news articles should be substantial)
        if content.get('content'):
            word_count = len(content['content'].split())
            details['word_count'] = word_count
            if word_count < 30:  # Reduced from 50 to be less strict
                issues.append("Article content too short (less than 30 words)")
            elif word_count < 50:  # Reduced from 80 to be less strict
                issues.append("Article content may be too brief for news")
        
        # Check for news-specific patterns
        news_patterns = [
            r'\b(said|told|stated|announced|reported|according to)\b',
            r'\b(today|yesterday|this week|last week|recently)\b',
            r'\b(source|sources|spokesperson|official)\b'
        ]
        
        if content.get('content'):
            pattern_matches = sum(1 for pattern in news_patterns 
                                if re.search(pattern, content['content'], re.I))
            details['news_pattern_matches'] = pattern_matches
            if pattern_matches == 0:
                issues.append("Content lacks typical news article patterns")
        
        # Calculate quality score
        score = 0.0
        
        # Title quality (20%)
        if details.get('has_title'):
            title_score = min(1.0, details.get('title_length', 0) / 100)  # Good titles ~50-100 chars
            score += title_score * 0.2
        
        # Content quality (40%)
        if details.get('has_content'):
            word_count = details.get('word_count', 0)
            if word_count >= 200:
                content_score = 1.0
            elif word_count >= 100:
                content_score = 0.8
            elif word_count >= 50:
                content_score = 0.5
            else:
                content_score = 0.2
            score += content_score * 0.4
        
        # Metadata quality (25%)
        metadata_score = (
            (0.4 if details.get('has_author') else 0) +
            (0.4 if details.get('has_date') else 0) +
            (0.2 if details.get('has_summary') else 0)
        )
        score += metadata_score * 0.25
        
        # News patterns (15%)
        pattern_score = min(1.0, details.get('news_pattern_matches', 0) / 3)
        score += pattern_score * 0.15
        
        validation_result['quality_score'] = score
        validation_result['is_valid'] = score >= 0.4 and len(issues) == 0  # More lenient threshold
        
        return validation_result
    
    def validate_product_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate product information with product-specific quality criteria.
        
        Args:
            content: Extracted product content with fields like name, price, description
            
        Returns:
            Dictionary with validation results and quality scores
        """
        validation_result = {
            'is_valid': False,
            'quality_score': 0.0,
            'validation_details': {},
            'issues': []
        }
        
        details = validation_result['validation_details']
        issues = validation_result['issues']
        
        # Check required fields for products
        if not content.get('name') and not content.get('title'):
            issues.append("Missing product name/title")
            details['has_name'] = False
        else:
            details['has_name'] = True
            name = content.get('name') or content.get('title')
            details['name_length'] = len(name)
            
        if not content.get('description'):
            issues.append("Missing product description")
            details['has_description'] = False
        else:
            details['has_description'] = True
            details['description_length'] = len(content['description'])
        
        # Product-specific quality checks
        details['has_price'] = bool(content.get('price'))
        details['has_images'] = bool(content.get('images'))
        details['has_specs'] = bool(content.get('specs') or content.get('specifications'))
        details['has_availability'] = bool(content.get('availability') or content.get('in_stock'))
        details['has_brand'] = bool(content.get('brand') or content.get('manufacturer'))
        
        # Price validation
        if content.get('price'):
            price_text = str(content['price'])
            price_patterns = [r'\$\d+', r'\d+\.\d+', r'USD', r'EUR', r'GBP']
            has_valid_price = any(re.search(pattern, price_text, re.I) for pattern in price_patterns)
            details['has_valid_price'] = has_valid_price
            if not has_valid_price:
                issues.append("Price format may be invalid")
        
        # Check for product-specific patterns
        product_patterns = [
            r'\b(features?|specifications?|specs)\b',
            r'\b(warranty|guarantee)\b',
            r'\b(model|version|edition)\b',
            r'\b(compatible|compatibility)\b'
        ]
        
        description = content.get('description', '')
        if description:
            pattern_matches = sum(1 for pattern in product_patterns 
                                if re.search(pattern, description, re.I))
            details['product_pattern_matches'] = pattern_matches
        
        # Calculate quality score
        score = 0.0
        
        # Product name quality (20%)
        if details.get('has_name'):
            name_score = min(1.0, details.get('name_length', 0) / 80)  # Good names ~40-80 chars
            score += name_score * 0.2
        
        # Description quality (30%)
        if details.get('has_description'):
            desc_length = details.get('description_length', 0)
            if desc_length >= 200:
                desc_score = 1.0
            elif desc_length >= 100:
                desc_score = 0.8
            elif desc_length >= 50:
                desc_score = 0.5
            else:
                desc_score = 0.3
            score += desc_score * 0.3
        
        # Essential product info (35%)
        essential_score = (
            (0.5 if details.get('has_price') else 0) +
            (0.2 if details.get('has_availability') else 0) +
            (0.15 if details.get('has_brand') else 0)
        )
        score += essential_score * 0.35
        
        # Additional product info (15%)
        additional_score = (
            (0.5 if details.get('has_specs') else 0) +
            (0.3 if details.get('has_images') else 0) +
            (0.2 if details.get('product_pattern_matches', 0) > 0 else 0)
        )
        score += additional_score * 0.15
        
        validation_result['quality_score'] = score
        validation_result['is_valid'] = score >= 0.5 and len(issues) == 0
        
        return validation_result
    
    def validate_job_listing(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate job listing content with job-specific quality criteria.
        
        Args:
            content: Extracted job content with fields like title, company, location
            
        Returns:
            Dictionary with validation results and quality scores
        """
        validation_result = {
            'is_valid': False,
            'quality_score': 0.0,
            'validation_details': {},
            'issues': []
        }
        
        details = validation_result['validation_details']
        issues = validation_result['issues']
        
        # Check required fields for job listings
        if not content.get('title'):
            issues.append("Missing job title")
            details['has_title'] = False
        else:
            details['has_title'] = True
            details['title_length'] = len(content['title'])
            
        if not content.get('company'):
            issues.append("Missing company name")
            details['has_company'] = False
        else:
            details['has_company'] = True
            
        if not content.get('location'):
            issues.append("Missing job location")
            details['has_location'] = False
        else:
            details['has_location'] = True
        
        # Job-specific quality checks
        details['has_description'] = bool(content.get('description'))
        details['has_requirements'] = bool(content.get('requirements'))
        details['has_salary'] = bool(content.get('salary'))
        details['has_type'] = bool(content.get('type'))
        details['has_apply_url'] = bool(content.get('apply_url'))
        
        # Check for job-specific patterns
        job_patterns = [
            r'\b(responsibilities|duties|requirements)\b',
            r'\b(experience|years|skills)\b',
            r'\b(qualifications|education|degree)\b',
            r'\b(benefits|salary|compensation)\b'
        ]
        
        description = content.get('description', '') + ' ' + content.get('requirements', '')
        if description.strip():
            pattern_matches = sum(1 for pattern in job_patterns 
                                if re.search(pattern, description, re.I))
            details['job_pattern_matches'] = pattern_matches
            
            # Check description length
            word_count = len(description.split())
            details['description_word_count'] = word_count
            if word_count < 30:  # Reduced threshold
                issues.append("Job description too brief")
        
        # Calculate quality score
        score = 0.0
        
        # Essential job info (60%)
        essential_score = (
            (0.25 if details.get('has_title') else 0) +
            (0.2 if details.get('has_company') else 0) +
            (0.15 if details.get('has_location') else 0)
        )
        score += essential_score * 0.6
        
        # Job description quality (25%)
        if details.get('has_description'):
            word_count = details.get('description_word_count', 0)
            if word_count >= 100:
                desc_score = 1.0
            elif word_count >= 50:
                desc_score = 0.7
            else:
                desc_score = 0.3
            score += desc_score * 0.25
        
        # Additional job info (15%)
        additional_score = (
            (0.3 if details.get('has_requirements') else 0) +
            (0.2 if details.get('has_salary') else 0) +
            (0.2 if details.get('has_type') else 0) +
            (0.3 if details.get('has_apply_url') else 0)
        )
        score += additional_score * 0.15
        
        validation_result['quality_score'] = score
        validation_result['is_valid'] = score >= 0.5 and len(issues) == 0  # Reasonable threshold
        
        return validation_result
    
    def validate_contact_information(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate contact information with contact-specific quality criteria.
        
        Args:
            content: Extracted contact content with fields like phone, email, address
            
        Returns:
            Dictionary with validation results and quality scores
        """
        validation_result = {
            'is_valid': False,
            'quality_score': 0.0,
            'validation_details': {},
            'issues': []
        }
        
        details = validation_result['validation_details']
        issues = validation_result['issues']
        
        # Validate phone numbers
        phones = content.get('phone', [])
        if isinstance(phones, str):
            phones = [phones]
        
        valid_phones = []
        for phone in phones:
            # Basic phone validation (international formats)
            phone_clean = re.sub(r'[^\d+() -]', '', phone)  # Fixed regex
            if re.match(r'^[\+]?[\d\s\-\(\)]{10,}$', phone_clean):
                valid_phones.append(phone)
        
        details['phone_count'] = len(phones)
        details['valid_phone_count'] = len(valid_phones)
        details['has_valid_phone'] = len(valid_phones) > 0
        
        if phones and not valid_phones:
            issues.append("No valid phone numbers found")
        
        # Validate email addresses
        emails = content.get('email', [])
        if isinstance(emails, str):
            emails = [emails]
        
        valid_emails = []
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        for email in emails:
            if re.match(email_pattern, email.strip()):
                valid_emails.append(email)
        
        details['email_count'] = len(emails)
        details['valid_email_count'] = len(valid_emails)
        details['has_valid_email'] = len(valid_emails) > 0
        
        if emails and not valid_emails:
            issues.append("No valid email addresses found")
        
        # Validate address
        address = content.get('address', '')
        if address:
            # Basic address validation (should have numbers and text)
            has_number = bool(re.search(r'\d+', address))
            has_text = bool(re.search(r'[a-zA-Z]{3,}', address))
            details['has_valid_address'] = has_number and has_text
            if not details['has_valid_address']:
                issues.append("Address format may be invalid")
        else:
            details['has_valid_address'] = False
        
        # Check for additional contact info
        details['has_hours'] = bool(content.get('hours'))
        details['has_website'] = bool(content.get('website'))
        details['has_company_name'] = bool(content.get('company_name'))
        
        # Calculate quality score
        score = 0.0
        
        # Primary contact methods (70%)
        contact_score = (
            (0.4 if details.get('has_valid_phone') else 0) +
            (0.3 if details.get('has_valid_email') else 0)
        )
        score += contact_score * 0.7
        
        # Address quality (20%)
        if details.get('has_valid_address'):
            score += 0.2
        
        # Additional info (10%)
        additional_score = (
            (0.4 if details.get('has_company_name') else 0) +
            (0.3 if details.get('has_hours') else 0) +
            (0.3 if details.get('has_website') else 0)
        )
        score += additional_score * 0.1
        
        validation_result['quality_score'] = score
        validation_result['is_valid'] = score >= 0.5 and len(issues) == 0
        
        return validation_result
    
    def validate_content_by_type(self, content: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """
        Route content validation to appropriate content-type-specific validator.
        
        Args:
            content: Extracted content
            content_type: Type of content (NEWS_ARTICLES, PRODUCT_INFORMATION, etc.)
            
        Returns:
            Dictionary with validation results and quality scores
        """
        content_type = content_type.upper()
        
        if content_type == 'NEWS_ARTICLES':
            return self.validate_news_article(content)
        elif content_type == 'PRODUCT_INFORMATION':
            return self.validate_product_data(content)
        elif content_type == 'JOB_LISTINGS':
            return self.validate_job_listing(content)
        elif content_type == 'CONTACT_INFORMATION':
            return self.validate_contact_information(content)
        else:
            # Generic validation for unsupported content types
            return {
                'is_valid': True,
                'quality_score': self.score_content_quality(content.get('content', ''), ''),
                'validation_details': {'content_type': content_type, 'generic_validation': True},
                'issues': []
            }
    
    def validate_relevance_comprehensive(self, content: Dict[str, Any], query: str, 
                                       content_type: str = None, 
                                       temporal_context: Dict[str, Any] = None,
                                       geographic_context: Dict[str, Any] = None,
                                       target_entities: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive relevance validation with multiple dimensions.
        
        Args:
            content: Extracted content to validate
            query: Original user query
            content_type: Expected content type (NEWS_ARTICLES, PRODUCT_INFORMATION, etc.)
            temporal_context: Temporal requirements (e.g., {"recency": "latest", "timeframe": "24h"})
            geographic_context: Geographic requirements (e.g., {"location": "US", "radius": "50km"})
            target_entities: Specific entities that should be present in content
            
        Returns:
            Dictionary with comprehensive relevance validation results
        """
        validation_result = {
            'overall_relevance_score': 0.0,
            'is_relevant': False,
            'relevance_breakdown': {},
            'validation_details': {},
            'issues': []
        }
        
        breakdown = validation_result['relevance_breakdown']
        details = validation_result['validation_details']
        issues = validation_result['issues']
        
        # 1. Content-Type Relevance (25%)
        content_type_relevance = self._validate_content_type_relevance(
            content, query, content_type
        )
        breakdown['content_type_relevance'] = content_type_relevance
        
        # 2. Semantic Relevance (30%)
        semantic_relevance = self._validate_semantic_relevance(content, query)
        breakdown['semantic_relevance'] = semantic_relevance
        
        # 3. Temporal Relevance (20%)
        temporal_relevance = self._validate_temporal_relevance(
            content, temporal_context
        )
        breakdown['temporal_relevance'] = temporal_relevance
        
        # 4. Geographic Relevance (10%)
        geographic_relevance = self._validate_geographic_relevance(
            content, geographic_context
        )
        breakdown['geographic_relevance'] = geographic_relevance
        
        # 5. Entity Matching (15%)
        entity_relevance = self._validate_entity_matching(content, target_entities)
        breakdown['entity_relevance'] = entity_relevance
        
        # Calculate overall relevance score
        overall_score = (
            content_type_relevance['score'] * 0.25 +
            semantic_relevance['score'] * 0.30 +
            temporal_relevance['score'] * 0.20 +
            geographic_relevance['score'] * 0.10 +
            entity_relevance['score'] * 0.15
        )
        
        validation_result['overall_relevance_score'] = overall_score
        validation_result['is_relevant'] = overall_score >= 0.6  # Reasonable threshold
        
        # Collect issues from all validation dimensions
        for dimension_result in breakdown.values():
            if dimension_result.get('issues'):
                issues.extend(dimension_result['issues'])
        
        # Store detailed validation information
        details['query'] = query
        details['content_type'] = content_type
        details['temporal_context'] = temporal_context
        details['geographic_context'] = geographic_context
        details['target_entities'] = target_entities
        
        return validation_result
    
    def _validate_content_type_relevance(self, content: Dict[str, Any], query: str, 
                                        expected_type: str) -> Dict[str, Any]:
        """Validate that content matches expected content type."""
        result = {
            'score': 0.0,
            'is_valid': False,
            'details': {},
            'issues': []
        }
        
        if not expected_type:
            result['score'] = 0.8  # No specific type expected
            result['is_valid'] = True
            return result
        
        expected_type = expected_type.upper()
        
        # Content type indicators
        content_indicators = {
            'NEWS_ARTICLES': {
                'required_fields': ['title', 'content', 'published_date'],
                'optional_fields': ['author', 'source', 'category'],
                'content_patterns': [r'breaking', r'news', r'reported', r'according to'],
                'structure_indicators': ['headline', 'byline', 'article_body']
            },
            'PRODUCT_INFORMATION': {
                'required_fields': ['name', 'price'],
                'optional_fields': ['description', 'specifications', 'availability'],
                'content_patterns': [r'\$[\d,]+', r'price', r'buy now', r'add to cart'],
                'structure_indicators': ['product_title', 'price_display', 'product_details']
            },
            'JOB_LISTINGS': {
                'required_fields': ['title', 'company'],
                'optional_fields': ['location', 'salary', 'requirements'],
                'content_patterns': [r'apply', r'requirements', r'experience', r'salary'],
                'structure_indicators': ['job_title', 'company_name', 'job_description']
            },
            'CONTACT_INFORMATION': {
                'required_fields': ['phone', 'email', 'address'],
                'optional_fields': ['hours', 'website'],
                'content_patterns': [r'\(\d{3}\)', r'@\w+\.\w+', r'\d+\s+\w+\s+street'],
                'structure_indicators': ['phone_number', 'email_address', 'physical_address']
            }
        }
        
        if expected_type not in content_indicators:
            result['score'] = 0.5  # Unknown type
            result['issues'].append(f"Unknown content type: {expected_type}")
            return result
        
        indicators = content_indicators[expected_type]
        score_components = []
        
        # Check required fields (50% weight)
        required_score = 0.0
        present_required = 0
        for field in indicators['required_fields']:
            if content.get(field):
                present_required += 1
        
        if indicators['required_fields']:
            required_score = present_required / len(indicators['required_fields'])
        score_components.append(('required_fields', required_score, 0.5))
        
        # Check optional fields (25% weight)
        optional_score = 0.0
        present_optional = 0
        for field in indicators['optional_fields']:
            if content.get(field):
                present_optional += 1
        
        if indicators['optional_fields']:
            optional_score = present_optional / len(indicators['optional_fields'])
        score_components.append(('optional_fields', optional_score, 0.25))
        
        # Check content patterns (25% weight)
        pattern_score = 0.0
        content_text = str(content.get('content', '')) + ' ' + str(content.get('title', ''))
        pattern_matches = 0
        
        for pattern in indicators['content_patterns']:
            if re.search(pattern, content_text, re.IGNORECASE):
                pattern_matches += 1
        
        if indicators['content_patterns']:
            pattern_score = min(1.0, pattern_matches / len(indicators['content_patterns']))
        score_components.append(('content_patterns', pattern_score, 0.25))
        
        # Calculate final score
        final_score = sum(score * weight for _, score, weight in score_components)
        
        result['score'] = final_score
        result['is_valid'] = final_score >= 0.6
        result['details'] = {
            'expected_type': expected_type,
            'required_fields_present': present_required,
            'optional_fields_present': present_optional,
            'pattern_matches': pattern_matches,
            'score_components': score_components
        }
        
        if final_score < 0.6:
            result['issues'].append(f"Content doesn't match expected type {expected_type}")
        
        return result
    
    def _validate_semantic_relevance(self, content: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate semantic relevance between content and query."""
        result = {
            'score': 0.0,
            'is_valid': False,
            'details': {},
            'issues': []
        }
        
        if not query:
            result['score'] = 0.5
            return result
        
        # Extract text content
        content_text = self._extract_text_content(content)
        
        if not content_text:
            result['issues'].append("No text content to analyze")
            return result
        
        # Use existing semantic similarity scoring
        try:
            similarity_score = self.score_semantic_similarity(content_text, query)
            
            # Enhanced scoring with multiple approaches
            keyword_score = self._calculate_keyword_overlap(content_text, query)
            entity_score = self._calculate_entity_overlap(content_text, query)
            
            # Weighted combination
            final_score = (
                similarity_score * 0.5 +
                keyword_score * 0.3 +
                entity_score * 0.2
            )
            
            result['score'] = final_score
            result['is_valid'] = final_score >= 0.6
            result['details'] = {
                'similarity_score': similarity_score,
                'keyword_score': keyword_score,
                'entity_score': entity_score,
                'content_length': len(content_text),
                'query_length': len(query)
            }
            
            if final_score < 0.6:
                result['issues'].append("Low semantic relevance to query")
                
        except Exception as e:
            logger.error(f"Error in semantic relevance validation: {e}")
            result['score'] = 0.5
            result['issues'].append("Error calculating semantic relevance")
        
        return result
    
    def _validate_temporal_relevance(self, content: Dict[str, Any], 
                                   temporal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate temporal relevance of content."""
        result = {
            'score': 1.0,  # Default to valid if no temporal requirements
            'is_valid': True,
            'details': {},
            'issues': []
        }
        
        if not temporal_context:
            return result
        
        # Extract date information from content
        content_date = self._extract_content_date(content)
        
        result['details']['content_date'] = content_date
        result['details']['temporal_requirements'] = temporal_context
        
        if not content_date:
            if temporal_context.get('recency') == 'latest':
                result['score'] = 0.3
                result['is_valid'] = False
                result['issues'].append("No date found in content for recency requirement")
            return result
        
        # Validate recency requirements
        recency = temporal_context.get('recency')
        if recency:
            age_score = self._calculate_content_age_score(content_date, recency)
            result['score'] *= age_score
            result['details']['age_score'] = age_score
            
            if age_score < 0.5:
                result['is_valid'] = False
                result['issues'].append(f"Content too old for '{recency}' requirement")
        
        # Validate specific timeframe
        timeframe = temporal_context.get('timeframe')
        if timeframe:
            timeframe_score = self._validate_timeframe(content_date, timeframe)
            result['score'] *= timeframe_score
            result['details']['timeframe_score'] = timeframe_score
            
            if timeframe_score < 0.5:
                result['is_valid'] = False
                result['issues'].append(f"Content outside specified timeframe: {timeframe}")
        
        return result
    
    def _validate_geographic_relevance(self, content: Dict[str, Any], 
                                     geographic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate geographic relevance of content."""
        result = {
            'score': 1.0,  # Default to valid if no geographic requirements
            'is_valid': True,
            'details': {},
            'issues': []
        }
        
        if not geographic_context:
            return result
        
        # Extract location information from content
        content_locations = self._extract_content_locations(content)
        
        result['details']['content_locations'] = content_locations
        result['details']['geographic_requirements'] = geographic_context
        
        target_location = geographic_context.get('location')
        if target_location and content_locations:
            location_score = self._calculate_location_relevance(
                content_locations, target_location
            )
            result['score'] = location_score
            result['details']['location_score'] = location_score
            
            if location_score < 0.5:
                result['is_valid'] = False
                result['issues'].append(f"Content not relevant to location: {target_location}")
        elif target_location and not content_locations:
            result['score'] = 0.7  # Neutral if no location info found
        
        return result
    
    def _validate_entity_matching(self, content: Dict[str, Any], 
                                target_entities: List[str]) -> Dict[str, Any]:
        """Validate that target entities are present in content."""
        result = {
            'score': 1.0,  # Default to valid if no entities specified
            'is_valid': True,
            'details': {},
            'issues': []
        }
        
        if not target_entities:
            return result
        
        content_text = self._extract_text_content(content)
        if not content_text:
            result['score'] = 0.0
            result['is_valid'] = False
            result['issues'].append("No content to check for entities")
            return result
        
        content_text_lower = content_text.lower()
        
        matched_entities = []
        partial_matches = []
        
        for entity in target_entities:
            entity_lower = entity.lower()
            
            # Exact match
            if entity_lower in content_text_lower:
                matched_entities.append(entity)
            # Partial match (for compound entities)
            elif any(word in content_text_lower for word in entity_lower.split()):
                partial_matches.append(entity)
        
        # Calculate entity matching score
        exact_score = len(matched_entities) / len(target_entities) if target_entities else 1.0
        partial_score = len(partial_matches) / len(target_entities) if target_entities else 0.0
        
        # Weighted combination (exact matches worth more)
        entity_score = exact_score * 0.8 + partial_score * 0.2
        
        result['score'] = entity_score
        result['is_valid'] = entity_score >= 0.6
        result['details'] = {
            'target_entities': target_entities,
            'matched_entities': matched_entities,
            'partial_matches': partial_matches,
            'exact_score': exact_score,
            'partial_score': partial_score
        }
        
        if entity_score < 0.6:
            missing_entities = [e for e in target_entities 
                              if e not in matched_entities and e not in partial_matches]
            result['issues'].append(f"Missing target entities: {missing_entities}")
        
        return result
    
    # Helper methods for relevance validation
    
    def _extract_text_content(self, content: Dict[str, Any]) -> str:
        """Extract all text content from a content dictionary."""
        text_parts = []
        
        # Common text fields
        text_fields = ['title', 'content', 'description', 'summary', 'text']
        
        for field in text_fields:
            if content.get(field):
                text_parts.append(str(content[field]))
        
        return ' '.join(text_parts)
    
    def _extract_content_date(self, content: Dict[str, Any]) -> Optional[str]:
        """Extract date information from content."""
        date_fields = ['published_date', 'date', 'created_date', 'updated_date', 'timestamp']
        
        for field in date_fields:
            if content.get(field):
                return str(content[field])
        
        # Try to extract date from content text
        content_text = self._extract_text_content(content)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content_text)
            if match:
                return match.group()
        
        return None
    
    def _extract_content_locations(self, content: Dict[str, Any]) -> List[str]:
        """Extract location information from content."""
        locations = []
        
        # Check specific location fields
        location_fields = ['location', 'address', 'city', 'state', 'country']
        
        for field in location_fields:
            if content.get(field):
                locations.append(str(content[field]))
        
        # Extract locations from text using basic patterns
        content_text = self._extract_text_content(content)
        
        # Simple location patterns (can be enhanced with NER)
        location_patterns = [
            r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City, ST
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+,\s*[A-Z]{2}\b',  # City Name, ST
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, content_text)
            locations.extend(matches)
        
        return list(set(locations))  # Remove duplicates
    
    def _calculate_keyword_overlap(self, content_text: str, query: str) -> float:
        """Calculate keyword overlap between content and query."""
        if not content_text or not query:
            return 0.0
        
        # Simple keyword extraction (can be enhanced with NLP)
        query_words = set(query.lower().split())
        content_words = set(content_text.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        query_words -= stop_words
        content_words -= stop_words
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def _calculate_entity_overlap(self, content_text: str, query: str) -> float:
        """Calculate entity overlap between content and query using basic NER."""
        if not self.SPACY_AVAILABLE or not self.nlp:
            return self._calculate_keyword_overlap(content_text, query)
        
        try:
            query_doc = self.nlp(query)
            content_doc = self.nlp(content_text[:1000])  # Limit text length
            
            query_entities = {ent.text.lower() for ent in query_doc.ents}
            content_entities = {ent.text.lower() for ent in content_doc.ents}
            
            if not query_entities:
                return 0.5  # No entities in query
            
            overlap = len(query_entities.intersection(content_entities))
            return overlap / len(query_entities)
            
        except Exception as e:
            logger.error(f"Error in entity overlap calculation: {e}")
            return self._calculate_keyword_overlap(content_text, query)
    
    def _calculate_content_age_score(self, content_date: str, recency_requirement: str) -> float:
        """Calculate score based on content age and recency requirement."""
        from datetime import datetime, timedelta
        
        try:
            # Parse content date (simplified - can be enhanced)
            if re.match(r'\d{4}-\d{2}-\d{2}', content_date):
                date_obj = datetime.strptime(content_date[:10], '%Y-%m-%d')
            else:
                return 0.5  # Unknown date format
            
            now = datetime.now()
            age_days = (now - date_obj).days
            
            # Score based on recency requirement
            if recency_requirement == 'latest':
                if age_days <= 1:
                    return 1.0
                elif age_days <= 7:
                    return 0.8
                elif age_days <= 30:
                    return 0.6
                else:
                    return 0.3
            elif recency_requirement == 'recent':
                if age_days <= 7:
                    return 1.0
                elif age_days <= 30:
                    return 0.8
                elif age_days <= 90:
                    return 0.6
                else:
                    return 0.4
            
            return 0.5  # Unknown recency requirement
            
        except Exception as e:
            logger.error(f"Error calculating content age score: {e}")
            return 0.5
    
    def _validate_timeframe(self, content_date: str, timeframe: str) -> float:
        """Validate content falls within specified timeframe."""
        # Simplified timeframe validation - can be enhanced
        return 1.0  # For now, assume valid
    
    def _calculate_location_relevance(self, content_locations: List[str], 
                                    target_location: str) -> float:
        """Calculate relevance score for location matching."""
        if not content_locations:
            return 0.5
        
        target_lower = target_location.lower()
        
        for location in content_locations:
            location_lower = location.lower()
            
            # Exact match
            if target_lower in location_lower or location_lower in target_lower:
                return 1.0
            
            # Partial match
            target_words = set(target_lower.split())
            location_words = set(location_lower.split())
            
            if target_words.intersection(location_words):
                return 0.7
        
        return 0.3  # No location match found
