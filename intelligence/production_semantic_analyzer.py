"""
Production Semantic Analyzer for SmartScrape
Advanced semantic analysis using only production-grade models (spaCy lg + sentence-transformers)
NO SMALL/MEDIUM MODEL SUPPORT - Production ready only
"""

import logging
import spacy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class AdvancedSemanticMatch:
    """Enhanced semantic match with confidence and reasoning"""
    text1: str
    text2: str
    similarity_score: float
    confidence: float
    method: str
    reasoning: str
    vector_similarity: Optional[float] = None

@dataclass
class ContentTypeResult:
    """Advanced content type detection result"""
    primary_type: str
    secondary_types: List[str]
    confidence: float
    indicators: Dict[str, float]
    reasoning: str

@dataclass
class QualityScore:
    """Content quality assessment"""
    overall_score: float
    readability: float
    information_density: float
    semantic_coherence: float
    completeness: float
    
@dataclass
class AdvancedIntentSemantics:
    """Enhanced semantic analysis of user intent"""
    query: str
    primary_intent: str
    secondary_intents: List[str]
    content_type: ContentTypeResult
    semantic_keywords: List[str]
    named_entities: List[Tuple[str, str]]
    related_concepts: List[str]
    query_variations: List[str]
    confidence: float
    embedding: np.ndarray
    quality_indicators: Dict[str, float]

class ProductionSemanticAnalyzer:
    """
    Production-grade semantic analyzer using only large models
    - spaCy en_core_web_lg (ENFORCED)
    - sentence-transformers for embeddings
    - Advanced content type detection (15+ types)
    - Multi-strategy semantic enhancement
    """
    
    # PRODUCTION MODEL REQUIREMENTS - NO FALLBACKS
    REQUIRED_SPACY_MODEL = "en_core_web_lg"
    FALLBACK_SPACY_MODEL = "en_core_web_trf"  # Optional transformer model
    SENTENCE_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(self, use_transformer_model: bool = False):
        self.logger = logging.getLogger(__name__)
        self.use_transformer_model = use_transformer_model
        
        # Initialize models (PRODUCTION ONLY)
        self.nlp = None
        self.sentence_model = None
        self._initialize_production_models()
        
        # Advanced content type patterns with semantic weights
        self.content_type_patterns = {
            'news_article': {
                'keywords': ['news', 'breaking', 'report', 'journalist', 'headline', 'story', 'article'],
                'entities': ['PERSON', 'ORG', 'GPE', 'EVENT'],
                'patterns': [r'breaking:?\s', r'reports?\s+that', r'according\s+to'],
                'weight': 1.0
            },
            'research_paper': {
                'keywords': ['research', 'study', 'analysis', 'methodology', 'findings', 'conclusion', 'abstract'],
                'entities': ['PERSON', 'ORG', 'GPE'],
                'patterns': [r'abstract:?\s', r'methodology', r'findings\s+show', r'peer.?reviewed'],
                'weight': 1.0
            },
            'product_listing': {
                'keywords': ['product', 'price', 'buy', 'purchase', 'specifications', 'features', 'review'],
                'entities': ['MONEY', 'PRODUCT', 'ORG'],
                'patterns': [r'\$\d+', r'buy\s+now', r'add\s+to\s+cart', r'specifications?'],
                'weight': 1.0
            },
            'job_posting': {
                'keywords': ['job', 'position', 'career', 'salary', 'experience', 'requirements', 'apply'],
                'entities': ['ORG', 'GPE', 'MONEY'],
                'patterns': [r'job\s+title', r'requirements?', r'apply\s+now', r'salary\s+range'],
                'weight': 1.0
            },
            'event_listing': {
                'keywords': ['event', 'conference', 'meeting', 'webinar', 'workshop', 'seminar', 'date'],
                'entities': ['EVENT', 'DATE', 'TIME', 'GPE'],
                'patterns': [r'date:?\s', r'time:?\s', r'location:?\s', r'register'],
                'weight': 1.0
            },
            'person_profile': {
                'keywords': ['biography', 'profile', 'background', 'experience', 'education', 'skills'],
                'entities': ['PERSON', 'ORG', 'GPE', 'DATE'],
                'patterns': [r'born\s+in', r'graduated\s+from', r'worked\s+at', r'experience\s+in'],
                'weight': 1.0
            },
            'company_info': {
                'keywords': ['company', 'corporation', 'business', 'founded', 'headquarters', 'revenue'],
                'entities': ['ORG', 'GPE', 'MONEY', 'DATE'],
                'patterns': [r'founded\s+in', r'headquarters?', r'revenue', r'employees?'],
                'weight': 1.0
            },
            'technical_doc': {
                'keywords': ['documentation', 'API', 'tutorial', 'guide', 'manual', 'reference'],
                'entities': ['PRODUCT'],
                'patterns': [r'function', r'parameter', r'return', r'example'],
                'weight': 1.0
            },
            'blog_post': {
                'keywords': ['blog', 'post', 'opinion', 'thoughts', 'personal', 'author'],
                'entities': ['PERSON', 'DATE'],
                'patterns': [r'posted\s+by', r'author:?', r'published\s+on'],
                'weight': 0.9
            },
            'review': {
                'keywords': ['review', 'rating', 'opinion', 'recommend', 'pros', 'cons'],
                'entities': ['PRODUCT', 'PERSON'],
                'patterns': [r'rating:?\s+\d', r'stars?', r'recommend', r'pros\s+and\s+cons'],
                'weight': 1.0
            },
            'forum_post': {
                'keywords': ['forum', 'discussion', 'thread', 'reply', 'question', 'answer'],
                'entities': ['PERSON'],
                'patterns': [r'replied\s+to', r'question:?', r'thread', r'forum'],
                'weight': 0.8
            },
            'social_media': {
                'keywords': ['tweet', 'post', 'share', 'like', 'follow', 'hashtag'],
                'entities': ['PERSON'],
                'patterns': [r'#\w+', r'@\w+', r'RT\s+@', r'shared\s+this'],
                'weight': 0.7
            },
            'legal_document': {
                'keywords': ['legal', 'contract', 'agreement', 'terms', 'conditions', 'policy'],
                'entities': ['ORG', 'LAW'],
                'patterns': [r'section\s+\d+', r'pursuant\s+to', r'hereby', r'terms\s+and\s+conditions'],
                'weight': 1.0
            },
            'financial_report': {
                'keywords': ['financial', 'earnings', 'revenue', 'profit', 'quarterly', 'annual'],
                'entities': ['ORG', 'MONEY', 'PERCENT'],
                'patterns': [r'Q[1-4]\s+\d{4}', r'earnings\s+per\s+share', r'revenue\s+of'],
                'weight': 1.0
            },
            'educational_content': {
                'keywords': ['course', 'lesson', 'tutorial', 'learning', 'education', 'student'],
                'entities': ['ORG', 'PERSON'],
                'patterns': [r'lesson\s+\d+', r'chapter\s+\d+', r'learning\s+objectives'],
                'weight': 1.0
            }
        }
        
        # Semantic expansion dictionary with embeddings
        self.semantic_expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ML', 'AI', 'automation'],
            'technology': ['tech', 'digital', 'software', 'hardware', 'innovation', 'technical', 'IT', 'computing'],
            'business': ['company', 'corporate', 'enterprise', 'organization', 'firm', 'startup', 'commerce'],
            'science': ['scientific', 'research', 'study', 'experiment', 'analysis', 'data', 'methodology'],
            'web': ['website', 'online', 'internet', 'digital', 'web-based', 'portal', 'platform'],
            'finance': ['financial', 'money', 'investment', 'banking', 'economic', 'fiscal', 'monetary']
        }
        
        self.logger.info(f"ProductionSemanticAnalyzer initialized with models: {self.REQUIRED_SPACY_MODEL}, {self.SENTENCE_MODEL}")
    
    def _initialize_production_models(self):
        """Initialize PRODUCTION models only - NO fallbacks to small/medium"""
        # Initialize spaCy (LARGE MODEL ONLY)
        try:
            if self.use_transformer_model:
                try:
                    self.nlp = spacy.load(self.FALLBACK_SPACY_MODEL)
                    self.logger.info(f"✅ Loaded transformer model: {self.FALLBACK_SPACY_MODEL}")
                except OSError:
                    self.logger.warning(f"Transformer model {self.FALLBACK_SPACY_MODEL} not available, using large model")
                    self.nlp = spacy.load(self.REQUIRED_SPACY_MODEL)
            else:
                self.nlp = spacy.load(self.REQUIRED_SPACY_MODEL)
                self.logger.info(f"✅ Loaded spaCy model: {self.REQUIRED_SPACY_MODEL}")
                
        except OSError as e:
            error_msg = f"CRITICAL: Production spaCy model {self.REQUIRED_SPACY_MODEL} not available. Install with: python -m spacy download {self.REQUIRED_SPACY_MODEL}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer(self.SENTENCE_MODEL)
            self.logger.info(f"✅ Loaded sentence transformer: {self.SENTENCE_MODEL}")
        except Exception as e:
            error_msg = f"CRITICAL: Sentence transformer {self.SENTENCE_MODEL} failed to load"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @lru_cache(maxsize=1000)
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get cached text embedding"""
        return self.sentence_model.encode([text])[0]
    
    def analyze_advanced_intent(self, query: str) -> AdvancedIntentSemantics:
        """
        Advanced intent analysis with production models
        
        Args:
            query: User query to analyze
            
        Returns:
            AdvancedIntentSemantics with comprehensive analysis
        """
        # Process with spaCy
        doc = self.nlp(query.lower())
        
        # Extract semantic features
        semantic_keywords = self._extract_semantic_keywords(doc)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Advanced content type detection
        content_type = self.detect_content_type_advanced(query)
        
        # Generate query variations
        query_variations = self.generate_semantic_variations(query, num_variations=8)
        
        # Extract related concepts
        related_concepts = self._generate_advanced_concepts(semantic_keywords, content_type.primary_type)
        
        # Determine primary and secondary intents
        primary_intent, secondary_intents = self._analyze_intent_hierarchy(doc, content_type)
        
        # Calculate quality indicators
        quality_indicators = self._calculate_query_quality(doc, semantic_keywords, named_entities)
        
        # Generate embedding
        embedding = self._get_text_embedding(query)
        
        # Calculate overall confidence
        confidence = self._calculate_intent_confidence(
            doc, semantic_keywords, named_entities, content_type, quality_indicators
        )
        
        return AdvancedIntentSemantics(
            query=query,
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            content_type=content_type,
            semantic_keywords=semantic_keywords,
            named_entities=named_entities,
            related_concepts=related_concepts,
            query_variations=query_variations,
            confidence=confidence,
            embedding=embedding,
            quality_indicators=quality_indicators
        )
    
    def detect_content_type_advanced(self, text: str) -> ContentTypeResult:
        """
        Advanced content type detection using multiple signals
        
        Args:
            text: Text to analyze for content type
            
        Returns:
            ContentTypeResult with confidence and reasoning
        """
        doc = self.nlp(text.lower())
        
        # Calculate scores for each content type
        type_scores = {}
        detailed_indicators = defaultdict(dict)
        
        for content_type, patterns in self.content_type_patterns.items():
            score = 0.0
            indicators = {}
            
            # Keyword matching with semantic similarity
            keyword_score = 0.0
            for keyword in patterns['keywords']:
                # Direct keyword match
                if keyword in text.lower():
                    keyword_score += 1.0
                else:
                    # Semantic similarity with sentence transformers
                    keyword_emb = self._get_text_embedding(keyword)
                    text_emb = self._get_text_embedding(text)
                    similarity = cosine_similarity([keyword_emb], [text_emb])[0][0]
                    if similarity > 0.7:  # High semantic similarity
                        keyword_score += similarity * 0.8
            
            keyword_score = min(keyword_score, len(patterns['keywords'])) / len(patterns['keywords'])
            indicators['keyword_match'] = keyword_score
            
            # Entity type matching
            entity_score = 0.0
            doc_entity_types = set(ent.label_ for ent in doc.ents)
            expected_entities = set(patterns['entities'])
            if expected_entities:
                entity_overlap = len(doc_entity_types & expected_entities) / len(expected_entities)
                entity_score = entity_overlap
            indicators['entity_match'] = entity_score
            
            # Pattern matching
            pattern_score = 0.0
            for pattern in patterns['patterns']:
                if re.search(pattern, text.lower()):
                    pattern_score += 1.0
            pattern_score = min(pattern_score, len(patterns['patterns'])) / len(patterns['patterns'])
            indicators['pattern_match'] = pattern_score
            
            # Combine scores with weights
            final_score = (
                keyword_score * 0.5 +
                entity_score * 0.3 +
                pattern_score * 0.2
            ) * patterns['weight']
            
            type_scores[content_type] = final_score
            detailed_indicators[content_type] = indicators
        
        # Determine primary and secondary types
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        primary_type = sorted_types[0][0] if sorted_types else 'general'
        primary_score = sorted_types[0][1] if sorted_types else 0.0
        
        # Secondary types with score > 0.3
        secondary_types = [t for t, s in sorted_types[1:] if s > 0.3][:3]
        
        # Generate reasoning
        reasoning = self._generate_content_type_reasoning(
            primary_type, primary_score, detailed_indicators[primary_type]
        )
        
        return ContentTypeResult(
            primary_type=primary_type,
            secondary_types=secondary_types,
            confidence=primary_score,
            indicators=dict(detailed_indicators[primary_type]),
            reasoning=reasoning
        )
    
    def generate_semantic_variations(self, query: str, num_variations: int = 10) -> List[str]:
        """
        Generate semantic variations of a query using advanced NLP
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of semantically enhanced query variations
        """
        variations = [query]  # Start with original
        
        # Analyze query intent
        intent = self.analyze_advanced_intent(query)
        doc = self.nlp(query.lower())
        
        # Strategy 1: Keyword substitution with synonyms
        for keyword in intent.semantic_keywords[:3]:
            # Find semantic expansions
            for category, expansions in self.semantic_expansions.items():
                if any(cosine_similarity(
                    [self._get_text_embedding(keyword)], 
                    [self._get_text_embedding(exp)]
                )[0][0] > 0.6 for exp in expansions):
                    for expansion in expansions[:2]:
                        if expansion not in query.lower():
                            variations.append(f"{query} {expansion}")
        
        # Strategy 2: Entity-based variations
        for entity_text, entity_type in intent.named_entities:
            if entity_type in ['ORG', 'PRODUCT', 'PERSON']:
                variations.append(f"{query} about {entity_text}")
                variations.append(f"{entity_text} {query}")
        
        # Strategy 3: Content-type specific enhancements
        content_patterns = self.content_type_patterns.get(intent.content_type.primary_type, {})
        if 'keywords' in content_patterns:
            for keyword in content_patterns['keywords'][:2]:
                if keyword not in query.lower():
                    variations.append(f"{query} {keyword}")
        
        # Strategy 4: Related concept integration
        for concept in intent.related_concepts[:2]:
            variations.append(f"{query} {concept}")
        
        # Strategy 5: Intent-based reformulations
        if intent.primary_intent == 'search':
            variations.append(f"find {query}")
            variations.append(f"search for {query}")
        elif intent.primary_intent == 'information':
            variations.append(f"information about {query}")
            variations.append(f"details on {query}")
        
        # Remove duplicates and limit to requested number
        unique_variations = list(dict.fromkeys(variations))  # Preserve order
        return unique_variations[:num_variations]
    
    def calculate_vector_similarity(self, text1: str, text2: str) -> AdvancedSemanticMatch:
        """
        Calculate advanced semantic similarity using multiple methods
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            AdvancedSemanticMatch with detailed similarity analysis
        """
        # Get embeddings
        emb1 = self._get_text_embedding(text1)
        emb2 = self._get_text_embedding(text2)
        
        # Calculate cosine similarity
        vector_similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        # spaCy similarity (if available)
        doc1 = self.nlp(text1.lower())
        doc2 = self.nlp(text2.lower())
        spacy_similarity = doc1.similarity(doc2) if hasattr(doc1, 'similarity') else 0.0
        
        # Combine similarities
        combined_similarity = (vector_similarity * 0.7 + spacy_similarity * 0.3)
        
        # Calculate confidence based on text quality
        confidence = self._calculate_similarity_confidence(doc1, doc2, vector_similarity)
        
        # Generate reasoning
        reasoning = self._generate_similarity_reasoning(
            text1, text2, vector_similarity, spacy_similarity, combined_similarity
        )
        
        return AdvancedSemanticMatch(
            text1=text1,
            text2=text2,
            similarity_score=combined_similarity,
            confidence=confidence,
            method="vector_spacy_hybrid",
            reasoning=reasoning,
            vector_similarity=vector_similarity
        )
    
    def semantic_content_quality_score(self, content: str) -> QualityScore:
        """
        Assess content quality using semantic analysis
        
        Args:
            content: Content to assess
            
        Returns:
            QualityScore with detailed quality metrics
        """
        doc = self.nlp(content)
        
        # Readability assessment
        readability = self._assess_readability(doc)
        
        # Information density
        information_density = self._calculate_information_density(doc)
        
        # Semantic coherence
        semantic_coherence = self._assess_semantic_coherence(content)
        
        # Completeness assessment
        completeness = self._assess_completeness(doc)
        
        # Overall score (weighted average)
        overall_score = (
            readability * 0.25 +
            information_density * 0.30 +
            semantic_coherence * 0.30 +
            completeness * 0.15
        )
        
        return QualityScore(
            overall_score=overall_score,
            readability=readability,
            information_density=information_density,
            semantic_coherence=semantic_coherence,
            completeness=completeness
        )
    
    # Helper methods
    def _extract_semantic_keywords(self, doc) -> List[str]:
        """Extract semantically important keywords"""
        keywords = []
        
        # Important POS tags
        important_pos = ['NOUN', 'PROPN', 'ADJ', 'VERB']
        
        for token in doc:
            if (token.pos_ in important_pos and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(keywords))
    
    def _generate_advanced_concepts(self, keywords: List[str], content_type: str) -> List[str]:
        """Generate related concepts using semantic analysis"""
        concepts = []
        
        # Add semantic expansions
        for keyword in keywords[:3]:
            for category, expansions in self.semantic_expansions.items():
                # Check semantic similarity
                for expansion in expansions:
                    similarity = cosine_similarity(
                        [self._get_text_embedding(keyword)],
                        [self._get_text_embedding(expansion)]
                    )[0][0]
                    
                    if similarity > 0.6 and expansion not in concepts:
                        concepts.append(expansion)
        
        # Add content-type specific concepts
        if content_type in self.content_type_patterns:
            type_keywords = self.content_type_patterns[content_type]['keywords']
            concepts.extend(type_keywords[:3])
        
        return list(dict.fromkeys(concepts))[:10]  # Top 10 unique concepts
    
    def _analyze_intent_hierarchy(self, doc, content_type: ContentTypeResult) -> Tuple[str, List[str]]:
        """Analyze intent hierarchy"""
        # Simple intent classification based on linguistic patterns
        question_words = ['what', 'where', 'when', 'why', 'how', 'who']
        action_words = ['find', 'search', 'get', 'download', 'buy', 'learn']
        
        text = doc.text.lower()
        
        if any(qw in text for qw in question_words):
            primary = 'information'
        elif any(aw in text for aw in action_words):
            primary = 'action'
        elif content_type.primary_type in ['product_listing', 'job_posting']:
            primary = 'acquisition'
        else:
            primary = 'search'
        
        # Secondary intents based on content type
        secondary = []
        if content_type.primary_type == 'research_paper':
            secondary.extend(['academic', 'detailed'])
        elif content_type.primary_type == 'news_article':
            secondary.extend(['current', 'factual'])
        elif content_type.primary_type == 'product_listing':
            secondary.extend(['commercial', 'comparison'])
        
        return primary, secondary
    
    def _calculate_query_quality(self, doc, keywords: List[str], entities: List[Tuple[str, str]]) -> Dict[str, float]:
        """Calculate query quality indicators"""
        return {
            'specificity': min(len(keywords) / 5.0, 1.0),  # More keywords = more specific
            'entity_richness': min(len(entities) / 3.0, 1.0),  # More entities = richer
            'length_adequacy': min(len(doc) / 10.0, 1.0),  # Adequate length
            'complexity': min(len([t for t in doc if t.pos_ in ['VERB', 'ADJ']]) / len(doc), 1.0)
        }
    
    def _calculate_intent_confidence(self, doc, keywords: List[str], entities: List[Tuple[str, str]], 
                                   content_type: ContentTypeResult, quality: Dict[str, float]) -> float:
        """Calculate overall intent confidence"""
        # Base confidence from content type detection
        base_confidence = content_type.confidence
        
        # Quality boost
        quality_avg = sum(quality.values()) / len(quality)
        quality_boost = quality_avg * 0.2
        
        # Entity and keyword richness
        richness_boost = min((len(keywords) + len(entities)) / 10.0, 0.3)
        
        # Document completeness
        completeness = min(len(doc) / 20.0, 0.2)
        
        return min(base_confidence + quality_boost + richness_boost + completeness, 1.0)
    
    def _generate_content_type_reasoning(self, content_type: str, score: float, indicators: Dict[str, float]) -> str:
        """Generate reasoning for content type detection"""
        reasons = []
        
        if indicators.get('keyword_match', 0) > 0.5:
            reasons.append(f"Strong keyword match ({indicators['keyword_match']:.2f})")
        
        if indicators.get('entity_match', 0) > 0.3:
            reasons.append(f"Entity types match expected pattern ({indicators['entity_match']:.2f})")
        
        if indicators.get('pattern_match', 0) > 0.2:
            reasons.append(f"Structural patterns detected ({indicators['pattern_match']:.2f})")
        
        if not reasons:
            reasons.append("Default classification based on general content analysis")
        
        return f"Classified as '{content_type}' (confidence: {score:.2f}) because: " + "; ".join(reasons)
    
    def _calculate_similarity_confidence(self, doc1, doc2, vector_sim: float) -> float:
        """Calculate confidence in similarity measurement"""
        # Base confidence from vector similarity strength
        base_conf = abs(vector_sim - 0.5) * 2  # Higher for values far from 0.5
        
        # Document quality factors
        doc1_quality = min(len(doc1) / 10.0, 1.0)
        doc2_quality = min(len(doc2) / 10.0, 1.0)
        quality_factor = (doc1_quality + doc2_quality) / 2
        
        return min(base_conf * quality_factor, 1.0)
    
    def _generate_similarity_reasoning(self, text1: str, text2: str, vector_sim: float, 
                                     spacy_sim: float, combined_sim: float) -> str:
        """Generate reasoning for similarity calculation"""
        if combined_sim > 0.8:
            return f"Very high similarity (vector: {vector_sim:.3f}, spacy: {spacy_sim:.3f})"
        elif combined_sim > 0.6:
            return f"High similarity (vector: {vector_sim:.3f}, spacy: {spacy_sim:.3f})"
        elif combined_sim > 0.4:
            return f"Moderate similarity (vector: {vector_sim:.3f}, spacy: {spacy_sim:.3f})"
        else:
            return f"Low similarity (vector: {vector_sim:.3f}, spacy: {spacy_sim:.3f})"
    
    def _assess_readability(self, doc) -> float:
        """Assess text readability"""
        # Simple readability based on sentence length and complexity
        sentences = list(doc.sents)
        if not sentences:
            return 0.0
        
        avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)
        complexity_ratio = len([t for t in doc if t.pos_ in ['VERB', 'ADJ']]) / len(doc)
        
        # Optimal sentence length is around 15-20 words
        length_score = 1.0 - abs(avg_sent_length - 17.5) / 17.5
        complexity_score = min(complexity_ratio * 2, 1.0)  # Some complexity is good
        
        return (length_score + complexity_score) / 2
    
    def _calculate_information_density(self, doc) -> float:
        """Calculate information density"""
        # Ratio of content words to total words
        content_words = len([t for t in doc if not t.is_stop and not t.is_punct])
        total_words = len([t for t in doc if not t.is_punct])
        
        if total_words == 0:
            return 0.0
        
        return content_words / total_words
    
    def _assess_semantic_coherence(self, content: str) -> float:
        """Assess semantic coherence using sentence similarity"""
        sentences = content.split('.')[:10]  # Limit to first 10 sentences
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by definition
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(sentences) - 1):
            if sentences[i].strip() and sentences[i+1].strip():
                sim = self.calculate_vector_similarity(sentences[i], sentences[i+1])
                similarities.append(sim.similarity_score)
        
        if not similarities:
            return 0.5  # Neutral coherence
        
        return sum(similarities) / len(similarities)
    
    def _assess_completeness(self, doc) -> float:
        """Assess content completeness"""
        # Based on presence of different linguistic structures
        has_entities = len(list(doc.ents)) > 0
        has_verbs = any(token.pos_ == 'VERB' for token in doc)
        has_nouns = any(token.pos_ in ['NOUN', 'PROPN'] for token in doc)
        has_adjectives = any(token.pos_ == 'ADJ' for token in doc)
        
        structure_score = sum([has_entities, has_verbs, has_nouns, has_adjectives]) / 4
        
        # Length adequacy (more content generally means more complete)
        length_score = min(len(doc) / 50.0, 1.0)  # Optimal around 50+ tokens
        
        return (structure_score + length_score) / 2
