"""
Semantic Analysis Module for SmartScrape
Enhanced with spaCy for intent analysis and semantic similarity
"""

import logging
import spacy
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SemanticMatch:
    """Represents a semantic match between two pieces of text."""
    text1: str
    text2: str
    similarity_score: float
    method: str
    confidence: float
    
@dataclass
class IntentSemantics:
    """Semantic analysis of user intent."""
    query: str
    content_type: str
    semantic_keywords: List[str]
    related_concepts: List[str]
    confidence: float
    embedding: Optional[np.ndarray] = None

class SemanticAnalyzer:
    """
    Enhanced semantic analyzer using spaCy for intent analysis and similarity computation.
    Provides semantic understanding for better orchestration and content matching.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.logger = logging.getLogger(__name__)
        self.spacy_model = spacy_model
        self.nlp = None
        self._initialize_spacy()
        
        # Content type patterns for semantic matching
        self.content_type_patterns = {
            'news': ['news', 'article', 'breaking', 'report', 'story', 'headline', 'journalism'],
            'product': ['product', 'item', 'buy', 'purchase', 'review', 'price', 'specification'],
            'research': ['research', 'study', 'paper', 'academic', 'journal', 'analysis', 'publication'],
            'event': ['event', 'conference', 'meeting', 'webinar', 'workshop', 'seminar'],
            'person': ['person', 'profile', 'biography', 'bio', 'people', 'individual'],
            'job': ['job', 'position', 'career', 'employment', 'hiring', 'vacancy', 'opportunity']
        }
        
        # Semantic expansion patterns
        self.semantic_expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ML', 'AI'],
            'tech': ['technology', 'digital', 'software', 'hardware', 'innovation', 'technical'],
            'business': ['company', 'corporate', 'enterprise', 'organization', 'firm', 'business'],
            'science': ['scientific', 'research', 'study', 'experiment', 'analysis', 'data']
        }
        
        self.logger.info(f"SemanticAnalyzer initialized with model: {spacy_model}")
    
    def _initialize_spacy(self):
        """Initialize spaCy model with error handling."""
        try:
            self.nlp = spacy.load(self.spacy_model)
            self.logger.info(f"spaCy model loaded: {self.spacy_model}")
        except OSError:
            self.logger.warning(f"spaCy model {self.spacy_model} not found, trying en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_model = "en_core_web_sm"
                self.logger.info("Loaded fallback model: en_core_web_sm")
            except OSError:
                self.logger.error("No spaCy model available. Install with: python -m spacy download en_core_web_sm")
                raise
    
    def analyze_intent_semantics(self, query: str) -> IntentSemantics:
        """
        Analyze the semantic meaning of a user query for better intent understanding.
        
        Args:
            query: User query to analyze
            
        Returns:
            IntentSemantics object with semantic analysis results
        """
        if not self.nlp:
            self.logger.warning("spaCy not available, using basic analysis")
            return self._basic_intent_analysis(query)
        
        # Process query with spaCy
        doc = self.nlp(query.lower())
        
        # Extract semantic keywords (important entities and concepts)
        semantic_keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                semantic_keywords.append(token.lemma_)
        
        # Extract named entities
        entities = [ent.text.lower() for ent in doc.ents]
        semantic_keywords.extend(entities)
        
        # Determine content type based on semantic patterns
        content_type = self._determine_content_type(query, semantic_keywords)
        
        # Generate related concepts
        related_concepts = self._generate_related_concepts(semantic_keywords)
        
        # Calculate confidence based on semantic richness
        confidence = self._calculate_semantic_confidence(doc, semantic_keywords, entities)
        
        # Create document embedding (using spaCy's doc vector)
        embedding = doc.vector if hasattr(doc, 'vector') else None
        
        return IntentSemantics(
            query=query,
            content_type=content_type,
            semantic_keywords=list(set(semantic_keywords)),
            related_concepts=related_concepts,
            confidence=confidence,
            embedding=embedding
        )
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> SemanticMatch:
        """
        Calculate semantic similarity between two texts using spaCy.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            SemanticMatch object with similarity details
        """
        if not self.nlp:
            return self._basic_similarity(text1, text2)
        
        doc1 = self.nlp(text1.lower())
        doc2 = self.nlp(text2.lower())
        
        # Calculate spaCy similarity
        similarity = doc1.similarity(doc2)
        
        # Calculate confidence based on document quality
        confidence = self._calculate_similarity_confidence(doc1, doc2)
        
        return SemanticMatch(
            text1=text1,
            text2=text2,
            similarity_score=similarity,
            method="spacy_similarity",
            confidence=confidence
        )
    
    def enhance_query_with_semantics(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Enhance a query with semantically related terms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries to generate
            
        Returns:
            List of enhanced queries including the original
        """
        intent_semantics = self.analyze_intent_semantics(query)
        
        enhanced_queries = [query]  # Start with original
        
        # Add queries with semantic keywords
        for keyword in intent_semantics.semantic_keywords[:3]:
            if keyword not in query.lower():
                enhanced_queries.append(f"{query} {keyword}")
        
        # Add queries with related concepts
        for concept in intent_semantics.related_concepts[:2]:
            if concept not in query.lower():
                enhanced_queries.append(f"{query} {concept}")
        
        # Add content-type specific enhancements
        if intent_semantics.content_type in self.content_type_patterns:
            patterns = self.content_type_patterns[intent_semantics.content_type]
            for pattern in patterns[:2]:
                if pattern not in query.lower():
                    enhanced_queries.append(f"{query} {pattern}")
        
        return enhanced_queries[:max_expansions]
    
    def rank_results_by_relevance(self, query: str, results: List[Dict[str, Any]], 
                                 content_field: str = "title") -> List[Dict[str, Any]]:
        """
        Rank results by semantic relevance to the query.
        
        Args:
            query: User query
            results: List of result dictionaries
            content_field: Field to use for relevance comparison
            
        Returns:
            Results sorted by relevance score (highest first)
        """
        if not results:
            return results
        
        # Calculate relevance scores
        scored_results = []
        for result in results:
            content = result.get(content_field, "")
            if content:
                similarity = self.calculate_semantic_similarity(query, str(content))
                result_copy = result.copy()
                result_copy["semantic_relevance"] = similarity.similarity_score
                result_copy["semantic_confidence"] = similarity.confidence
                scored_results.append(result_copy)
            else:
                result_copy = result.copy()
                result_copy["semantic_relevance"] = 0.0
                result_copy["semantic_confidence"] = 0.0
                scored_results.append(result_copy)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x["semantic_relevance"], reverse=True)
        
        return scored_results
    
    def _determine_content_type(self, query: str, keywords: List[str]) -> str:
        """Determine content type based on semantic patterns."""
        query_lower = query.lower()
        all_terms = [query_lower] + keywords
        
        # Score each content type
        type_scores = defaultdict(int)
        
        for content_type, patterns in self.content_type_patterns.items():
            for pattern in patterns:
                for term in all_terms:
                    if pattern in term:
                        type_scores[content_type] += 1
        
        # Return highest scoring type, or 'general' if none match
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _generate_related_concepts(self, keywords: List[str]) -> List[str]:
        """Generate related concepts based on semantic expansions."""
        related = []
        
        for keyword in keywords:
            for expansion_key, expansions in self.semantic_expansions.items():
                if keyword in expansions or expansion_key in keyword:
                    related.extend([exp for exp in expansions if exp != keyword])
        
        return list(set(related))[:10]  # Limit to 10 related concepts
    
    def _calculate_semantic_confidence(self, doc, keywords: List[str], entities: List[str]) -> float:
        """Calculate confidence based on semantic richness."""
        base_confidence = 0.5
        
        # Boost for entities
        if entities:
            base_confidence += 0.2
        
        # Boost for semantic keywords
        if len(keywords) > 2:
            base_confidence += 0.2
        
        # Boost for longer, more complex queries
        if len(doc) > 5:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _calculate_similarity_confidence(self, doc1, doc2) -> float:
        """Calculate confidence for similarity calculation."""
        base_confidence = 0.6
        
        # Boost for longer documents
        if len(doc1) > 3 and len(doc2) > 3:
            base_confidence += 0.2
        
        # Boost if both have entities
        if doc1.ents and doc2.ents:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _basic_intent_analysis(self, query: str) -> IntentSemantics:
        """Fallback intent analysis without spaCy."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 2]
        
        content_type = self._determine_content_type(query, keywords)
        
        return IntentSemantics(
            query=query,
            content_type=content_type,
            semantic_keywords=keywords,
            related_concepts=[],
            confidence=0.3
        )
    
    def _basic_similarity(self, text1: str, text2: str) -> SemanticMatch:
        """Fallback similarity calculation without spaCy."""
        # Simple Jaccard similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return SemanticMatch(
            text1=text1,
            text2=text2,
            similarity_score=similarity,
            method="jaccard_similarity",
            confidence=0.4
        )

# Global instance for easy access
semantic_analyzer = SemanticAnalyzer()
