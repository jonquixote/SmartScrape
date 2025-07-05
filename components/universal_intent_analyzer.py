"""
Universal Intent Analyzer with spaCy and Semantic Search Integration

This module provides enhanced intent analysis capabilities by combining:
- spaCy NLP for entity recognition and linguistic analysis
- Semantic search using embeddings for intent matching
- Contextual query expansion
- URL prioritization based on semantic scores
"""

import spacy
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import json
import re

# Import configuration
from config import (
    SPACY_ENABLED, SPACY_MODEL, SPACY_INTENT_ANALYSIS,
    SEMANTIC_SEARCH_ENABLED, CONTEXTUAL_QUERY_EXPANSION,
    get_config
)

# Import existing components
from ai_helpers.intent_parser import IntentParser
from ai_helpers.rule_based_extractor import RuleBasedExtractor, extract_intent

# Import semantic analyzer for enhanced analysis
from intelligence.semantic_analyzer import semantic_analyzer, SemanticAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniversalIntentAnalyzer")

class UniversalIntentAnalyzer:
    """
    Enhanced intent analyzer that combines spaCy NLP, semantic search, and contextual expansion
    for deeper understanding of user queries and intelligent URL prioritization.
    """

    def __init__(self, 
                 spacy_model: str = None,
                 use_semantic_search: bool = None):
        """
        Initialize the Universal Intent Analyzer.
        
        Args:
            spacy_model: spaCy model to use (defaults to config)
            use_semantic_search: Whether to use semantic search (defaults to config)
        """
        self.config = get_config()
        self.spacy_model_name = spacy_model or SPACY_MODEL
        self.use_semantic_search = use_semantic_search if use_semantic_search is not None else SEMANTIC_SEARCH_ENABLED
        
        # Initialize spaCy
        self.nlp = None
        if SPACY_ENABLED:
            try:
                # Try the medium model first, fallback to small
                try:
                    self.nlp = spacy.load(self.spacy_model_name)
                    logger.info(f"Loaded spaCy model: {self.spacy_model_name}")
                except OSError:
                    # Fallback to small model if medium not available
                    logger.warning(f"Medium model {self.spacy_model_name} not found, trying small model")
                # Load spaCy with model priority
                SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
                for model_name in SPACY_MODELS:
                    try:
                        self.nlp = spacy.load(model_name)
                        logger.info(f"Loaded spaCy model: {model_name}")
                        break
                    except OSError:
                        continue
            except OSError as e:
                logger.warning(f"Failed to load any spaCy model: {e}")
                logger.info("Falling back to basic intent analysis")
        
        # Initialize components
        self.intent_parser = IntentParser()
        self.rule_extractor = RuleBasedExtractor()
        
        # Initialize semantic analyzer
        self.semantic_analyzer = semantic_analyzer
        
        # Load intent patterns for semantic matching
        self.intent_patterns = self._load_intent_patterns()
        
        # Cache for embeddings (in lieu of FAISS)
        self.embedding_cache = {}
        
        logger.info("UniversalIntentAnalyzer initialized")

    def _load_intent_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load predefined intent patterns for semantic matching.
        
        Returns:
            Dictionary of intent patterns categorized by content type
        """
        patterns = {
            "news": [
                {
                    "pattern": "latest news about topic",
                    "template": {"content_type": "NEWS_ARTICLES", "temporal": "latest"},
                    "keywords": ["news", "latest", "updates", "breaking", "recent", "article", "story", "report"],
                    "indicators": ["latest", "breaking", "recent", "new", "current", "today", "yesterday"]
                },
                {
                    "pattern": "breaking news topic",
                    "template": {"content_type": "NEWS_ARTICLES", "temporal": "breaking"},
                    "keywords": ["breaking", "urgent", "alert", "developing", "just in"],
                    "indicators": ["breaking", "urgent", "alert", "developing"]
                }
            ],
            "product": [
                {
                    "pattern": "product price and specs",
                    "template": {"content_type": "PRODUCT_INFORMATION", "data_focus": "specs_pricing"},
                    "keywords": ["price", "cost", "specs", "specifications", "features", "buy", "purchase"],
                    "indicators": ["price", "cost", "$", "buy", "purchase", "specs", "features"]
                },
                {
                    "pattern": "product reviews and ratings",
                    "template": {"content_type": "REVIEWS_RATINGS", "data_focus": "user_feedback"},
                    "keywords": ["reviews", "rating", "stars", "feedback", "opinion", "experience"],
                    "indicators": ["review", "rating", "stars", "feedback", "opinion"]
                }
            ],
            "jobs": [
                {
                    "pattern": "job opportunities in location",
                    "template": {"content_type": "JOB_LISTINGS", "location_required": True},
                    "keywords": ["jobs", "careers", "employment", "hiring", "openings", "positions"],
                    "indicators": ["jobs", "careers", "hiring", "employment", "work", "position"]
                }
            ],
            "financial": [
                {
                    "pattern": "stock price and market data",
                    "template": {"content_type": "FINANCIAL_DATA", "data_focus": "market"},
                    "keywords": ["stock", "price", "market", "trading", "shares", "investment"],
                    "indicators": ["stock", "price", "market", "trading", "shares", "$"]
                }
            ],
            "contact": [
                {
                    "pattern": "contact information and location",
                    "template": {"content_type": "CONTACT_INFORMATION", "data_focus": "location_contact"},
                    "keywords": ["contact", "phone", "address", "location", "hours", "email"],
                    "indicators": ["contact", "phone", "address", "location", "hours", "email"]
                }
            ],
            "events": [
                {
                    "pattern": "upcoming events and schedules",
                    "template": {"content_type": "EVENT_LISTINGS", "temporal": "upcoming"},
                    "keywords": ["events", "calendar", "schedule", "meeting", "conference", "when"],
                    "indicators": ["event", "calendar", "schedule", "when", "date", "time"]
                }
            ],
            "search": [
                {
                    "pattern": "find products under price",
                    "template": {"entity_type": "product", "criteria": ["price"]},
                    "keywords": ["find", "search", "products", "under", "price", "cost"]
                },
                {
                    "pattern": "restaurants in location",
                    "template": {"entity_type": "restaurant", "location_required": True},
                    "keywords": ["restaurants", "food", "dining", "location", "area"]
                },
                {
                    "pattern": "homes for sale in location",
                    "template": {"entity_type": "property", "location_required": True},
                    "keywords": ["homes", "houses", "property", "real estate", "sale"]
                },
                {
                    "pattern": "latest news about topic",
                    "template": {"entity_type": "news", "topic_required": True},
                    "keywords": ["news", "latest", "updates", "information", "topic"]
                },
                {
                    "pattern": "jobs in location",
                    "template": {"entity_type": "job", "location_required": True},
                    "keywords": ["jobs", "employment", "career", "work", "position"]
                }
            ],
            "comparison": [
                {
                    "pattern": "compare products features",
                    "template": {"entity_type": "comparison", "comparison_type": "features"},
                    "keywords": ["compare", "comparison", "vs", "versus", "features"]
                }
            ],
            "information": [
                {
                    "pattern": "information about topic",
                    "template": {"entity_type": "information", "topic_required": True},
                    "keywords": ["information", "about", "details", "facts"]
                }
            ]
        }
        return patterns

    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query using spaCy NER, pattern matching, and semantic similarity.
        
        Args:
            query: User query string
            
        Returns:
            Enhanced intent analysis with semantic insights
        """
        logger.info(f"Analyzing intent for query: {query}")
        
        # Start with basic AI and rule-based analysis
        try:
            # Try to get AI intent synchronously or with a simple async wrapper
            import asyncio
            if asyncio.iscoroutinefunction(self.intent_parser.parse_query):
                # If it's a coroutine function, we need to handle it carefully
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in an async context, use asyncio.create_task
                        ai_intent = {}  # Fallback for now
                    else:
                        ai_intent = loop.run_until_complete(self.intent_parser.parse_query(query))
                except Exception as e:
                    logger.warning(f"Failed to get AI intent: {e}")
                    ai_intent = {}
            else:
                ai_intent = self.intent_parser.parse_query(query)
        except Exception as e:
            logger.warning(f"Failed to get AI intent: {e}")
            ai_intent = {}
            
        rule_intent = extract_intent(query)  # Use standalone function instead of method
        
        # ENHANCED: Use semantic analyzer for intent semantics
        semantic_intent = self.semantic_analyzer.analyze_intent_semantics(query)
        
        # Enhance with spaCy analysis if available
        spacy_analysis = self._analyze_with_spacy(query) if self.nlp else {}
        
        # Perform semantic matching if enabled
        semantic_matches = self._semantic_pattern_matching(query) if self.use_semantic_search else []
        
        # Combine all analyses
        enhanced_intent = self._combine_analyses(
            query, ai_intent, rule_intent, spacy_analysis, semantic_matches, semantic_intent
        )
        
        # ENHANCED: Add semantic insights from our analyzer
        enhanced_intent["semantic_analysis"] = {
            "content_type": semantic_intent.content_type,
            "semantic_keywords": semantic_intent.semantic_keywords,
            "related_concepts": semantic_intent.related_concepts,
            "confidence": semantic_intent.confidence
        }
        
        # Generate enhanced queries for better URL discovery
        enhanced_intent["enhanced_queries"] = self.semantic_analyzer.enhance_query_with_semantics(query)
        
        # Add semantic scores and URL prioritization data
        enhanced_intent["semantic_scores"] = self._calculate_semantic_scores(enhanced_intent)
        enhanced_intent["url_priorities"] = self._generate_url_priorities(enhanced_intent)
        
        # Add content-type classification
        enhanced_intent["content_type"] = self._classify_content_type(query, enhanced_intent)
        enhanced_intent["temporal_context"] = self._analyze_temporal_context(query)
        enhanced_intent["specificity_level"] = self._analyze_specificity(query, enhanced_intent)
        
        logger.info(f"Intent analysis complete. Content type: {enhanced_intent.get('content_type', 'unknown')}, Entity type: {enhanced_intent.get('entity_type', 'unknown')}")
        return enhanced_intent

    def _analyze_with_spacy(self, query: str) -> Dict[str, Any]:
        """
        Analyze query using spaCy for linguistic features.
        
        Args:
            query: User query string
            
        Returns:
            spaCy analysis results
        """
        try:
            doc = self.nlp(query)
            
            analysis = {
                "entities": [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": spacy.explain(ent.label_)
                    }
                    for ent in doc.ents
                ],
                "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
                "keywords": [
                    token.text for token in doc 
                    if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "ADJ"]
                ],
                "pos_tags": [(token.text, token.pos_, token.dep_) for token in doc],
                "sentiment": self._analyze_sentiment(doc)
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"spaCy analysis failed: {e}")
            return {}

    def _analyze_sentiment(self, doc) -> Dict[str, Any]:
        """
        Basic sentiment analysis using spaCy tokens.
        
        Args:
            doc: spaCy document
            
        Returns:
            Sentiment analysis results
        """
        # Simple sentiment based on token characteristics
        positive_indicators = ["best", "good", "great", "excellent", "top", "high", "quality"]
        negative_indicators = ["worst", "bad", "terrible", "poor", "low", "cheap"]
        
        tokens = [token.text.lower() for token in doc if not token.is_stop]
        
        positive_count = sum(1 for token in tokens if token in positive_indicators)
        negative_count = sum(1 for token in tokens if token in negative_indicators)
        
        if positive_count > negative_count:
            polarity = "positive"
        elif negative_count > positive_count:
            polarity = "negative"
        else:
            polarity = "neutral"
        
        return {
            "polarity": polarity,
            "positive_count": positive_count,
            "negative_count": negative_count
        }

    def _semantic_pattern_matching(self, query: str) -> List[Dict[str, Any]]:
        """
        Match query against predefined intent patterns using semantic similarity.
        
        Args:
            query: User query string
            
        Returns:
            List of matched patterns with similarity scores
        """
        matches = []
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for category, patterns in self.intent_patterns.items():
            for pattern in patterns:
                # Enhanced similarity calculation
                pattern_keywords = set(pattern["keywords"])
                pattern_indicators = set(pattern.get("indicators", []))
                
                # Calculate keyword similarity
                keyword_intersection = len(query_keywords.intersection(pattern_keywords))
                keyword_union = len(query_keywords.union(pattern_keywords))
                keyword_similarity = keyword_intersection / keyword_union if keyword_union > 0 else 0
                
                # Calculate indicator similarity (direct text matches)
                indicator_matches = sum(1 for indicator in pattern_indicators if indicator in query_lower)
                indicator_similarity = indicator_matches / max(len(pattern_indicators), 1)
                
                # Combined similarity with indicator boost
                total_similarity = (keyword_similarity * 0.6) + (indicator_similarity * 0.4)
                
                if total_similarity > 0.1:  # Threshold for matching
                    matches.append({
                        "category": category,
                        "pattern": pattern["pattern"],
                        "template": pattern["template"],
                        "similarity": total_similarity,
                        "matched_keywords": list(query_keywords.intersection(pattern_keywords)),
                        "matched_indicators": [ind for ind in pattern_indicators if ind in query_lower]
                    })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:5]  # Return top 5 matches

    def _combine_analyses(self, 
                         query: str,
                         ai_intent: Dict[str, Any],
                         rule_intent: Dict[str, Any],
                         spacy_analysis: Dict[str, Any],
                         semantic_matches: List[Dict[str, Any]],
                         semantic_intent) -> Dict[str, Any]:
        """
        Combine all analysis results into a unified intent.
        
        Args:
            query: Original query
            ai_intent: AI-based intent analysis
            rule_intent: Rule-based intent analysis  
            spacy_analysis: spaCy linguistic analysis
            semantic_matches: Semantic pattern matches
            semantic_intent: Semantic intent analysis
            
        Returns:
            Combined intent analysis
        """
        # Start with AI intent as base
        combined = ai_intent.copy()
        
        # Enhance entities with spaCy results
        if spacy_analysis and "entities" in spacy_analysis:
            combined["spacy_entities"] = spacy_analysis["entities"]
            combined["noun_phrases"] = spacy_analysis.get("noun_phrases", [])
            combined["spacy_keywords"] = spacy_analysis.get("keywords", [])
            combined["sentiment"] = spacy_analysis.get("sentiment", {})
        
        # Add semantic insights
        if semantic_matches:
            combined["semantic_matches"] = semantic_matches
            best_match = semantic_matches[0]
            
            # Use best semantic match to enhance intent
            if best_match["similarity"] > 0.3:  # High confidence threshold
                template = best_match["template"]
                combined["semantic_entity_type"] = template.get("entity_type")
                combined["semantic_category"] = best_match["category"]
                
                # Merge template requirements
                if template.get("location_required"):
                    combined["requires_location"] = True
                if template.get("topic_required"):
                    combined["requires_topic"] = True
        
        # ENHANCED: Add semantic intent insights
        if semantic_intent:
            combined["semantic_content_type"] = semantic_intent.content_type
            combined["semantic_keywords"] = semantic_intent.semantic_keywords
            combined["related_concepts"] = semantic_intent.related_concepts
            combined["semantic_confidence"] = semantic_intent.confidence
            
            # Use semantic content type to enhance existing classification
            if not combined.get("content_type") and semantic_intent.content_type != 'general':
                combined["content_type"] = semantic_intent.content_type.upper()
        
        # Enhance keywords with all sources
        all_keywords = set(combined.get("keywords", []))
        all_keywords.update(rule_intent.get("keywords", []))
        all_keywords.update(spacy_analysis.get("keywords", []))
        combined["keywords"] = list(all_keywords)
        
        # Add query expansion suggestions
        if CONTEXTUAL_QUERY_EXPANSION:
            combined["expanded_queries"] = self.expand_query_contextually(query, combined)
        
        return combined

    def expand_query_contextually(self, query: str, intent_analysis: Dict[str, Any]) -> List[str]:
        """
        Expand query with synonyms, related terms, and semantic variations.
        
        Args:
            query: Original query string
            intent_analysis: Current intent analysis
            
        Returns:
            List of expanded query variations
        """
        expansions = [query]  # Start with original
        
        # Add variations based on entity type
        entity_type = intent_analysis.get("entity_type") or intent_analysis.get("semantic_entity_type")
        
        if entity_type == "product":
            base_terms = intent_analysis.get("keywords", [])
            for term in base_terms:
                if "laptop" in term.lower():
                    expansions.extend([
                        query.replace(term, "notebook computer"),
                        query.replace(term, "portable computer")
                    ])
                elif "phone" in term.lower():
                    expansions.extend([
                        query.replace(term, "smartphone"),
                        query.replace(term, "mobile device")
                    ])
        
        elif entity_type == "restaurant":
            expansions.extend([
                query.replace("restaurant", "dining"),
                query.replace("restaurant", "food"),
                query + " reviews",
                query + " ratings"
            ])
        
        elif entity_type == "property":
            expansions.extend([
                query.replace("homes", "houses"),
                query.replace("for sale", "listings"),
                query + " real estate",
                query + " market"
            ])
        
        # Add location variations if location is detected
        if intent_analysis.get("location"):
            location = intent_analysis["location"]
            if "city" in location and location["city"]:
                city = location["city"]
                if city and isinstance(city, str):  # Ensure city is not None and is a string
                    expansions.append(query.replace(city, f"{city} area"))
                    expansions.append(query.replace(city, f"near {city}"))
        
        # Remove duplicates and empty strings
        expansions = list(set(filter(None, expansions)))
        
        logger.info(f"Generated {len(expansions)} query expansions")
        return expansions

    def _calculate_semantic_scores(self, intent_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate semantic confidence scores for different aspects of the intent.
        
        Args:
            intent_analysis: Current intent analysis
            
        Returns:
            Dictionary of semantic scores
        """
        scores = {
            "overall_confidence": 0.5,  # Default
            "entity_confidence": 0.5,
            "location_confidence": 0.5,
            "semantic_match_confidence": 0.0
        }
        
        # Boost confidence based on spaCy entities
        if intent_analysis.get("spacy_entities"):
            entity_count = len(intent_analysis["spacy_entities"])
            scores["entity_confidence"] = min(0.9, 0.3 + (entity_count * 0.2))
        
        # Boost confidence based on semantic matches
        if intent_analysis.get("semantic_matches"):
            best_match = intent_analysis["semantic_matches"][0]
            scores["semantic_match_confidence"] = best_match["similarity"]
            scores["overall_confidence"] = max(scores["overall_confidence"], best_match["similarity"])
        
        # Boost confidence if location is well-identified
        if intent_analysis.get("location") and intent_analysis["location"].get("city"):
            scores["location_confidence"] = 0.8
        
        # Calculate overall confidence
        scores["overall_confidence"] = (
            scores["entity_confidence"] * 0.4 +
            scores["semantic_match_confidence"] * 0.4 +
            scores["location_confidence"] * 0.2
        )
        
        return scores

    def _generate_url_priorities(self, intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate URL prioritization data based on semantic analysis.
        
        Args:
            intent_analysis: Current intent analysis
            
        Returns:
            List of URL priority suggestions
        """
        priorities = []
        
        entity_type = intent_analysis.get("entity_type") or intent_analysis.get("semantic_entity_type")
        location = intent_analysis.get("location", {})
        
        # Domain-specific prioritization
        if entity_type == "product":
            priorities.extend([
                {"domain_pattern": "amazon.com", "priority": 0.9, "reason": "Major e-commerce platform"},
                {"domain_pattern": "ebay.com", "priority": 0.8, "reason": "Marketplace with wide selection"},
                {"domain_pattern": "walmart.com", "priority": 0.7, "reason": "Retail chain with online presence"},
                {"url_pattern": "/product/", "priority": 0.8, "reason": "Product detail page pattern"},
                {"url_pattern": "/shop/", "priority": 0.7, "reason": "Shopping section"}
            ])
        
        elif entity_type == "restaurant":
            priorities.extend([
                {"domain_pattern": "yelp.com", "priority": 0.9, "reason": "Restaurant review platform"},
                {"domain_pattern": "tripadvisor.com", "priority": 0.8, "reason": "Travel and dining reviews"},
                {"domain_pattern": "opentable.com", "priority": 0.7, "reason": "Restaurant booking platform"},
                {"url_pattern": "/restaurant/", "priority": 0.8, "reason": "Restaurant detail page"},
                {"url_pattern": "/dining/", "priority": 0.7, "reason": "Dining section"}
            ])
        
        elif entity_type == "property":
            priorities.extend([
                {"domain_pattern": "zillow.com", "priority": 0.9, "reason": "Real estate platform"},
                {"domain_pattern": "realtor.com", "priority": 0.8, "reason": "MLS listings"},
                {"domain_pattern": "redfin.com", "priority": 0.8, "reason": "Real estate search"},
                {"url_pattern": "/homes-for-sale/", "priority": 0.9, "reason": "Property listings"}
            ])
        
        elif entity_type == "news":
            priorities.extend([
                {"domain_pattern": "cnn.com", "priority": 0.8, "reason": "Major news outlet"},
                {"domain_pattern": "reuters.com", "priority": 0.9, "reason": "News wire service"},
                {"domain_pattern": "ap.org", "priority": 0.8, "reason": "Associated Press"},
                {"url_pattern": "/news/", "priority": 0.8, "reason": "News section"},
                {"url_pattern": "/article/", "priority": 0.7, "reason": "Article page"}
            ])
        
        # Location-specific boosts
        if location.get("city") and location["city"]:
            city = location["city"].lower().replace(" ", "-")
            priorities.append({
                "url_pattern": f"/{city}/",
                "priority": 0.8,
                "reason": f"City-specific content for {location['city']}"
            })
        
        return priorities

    def suggest_urls(self, intent_analysis: Dict[str, Any], base_domains: List[str] = None) -> List[str]:
        """
        Suggest starting URLs based on intent analysis with enhanced semantic understanding.
        
        Args:
            intent_analysis: Current intent analysis
            base_domains: Optional list of base domains to prioritize
            
        Returns:
            List of suggested URLs ordered by relevance
        """
        suggestions = []
        
        entity_type = intent_analysis.get("entity_type") or intent_analysis.get("semantic_entity_type")
        keywords = intent_analysis.get("keywords", [])
        location = intent_analysis.get("location", {})
        
        # Generate search URLs based on entity type
        search_query = " ".join(keywords) if keywords else ""
        
        if entity_type == "product" and search_query:
            suggestions.extend([
                f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}",
                f"https://www.google.com/search?q={search_query.replace(' ', '+')}+buy+shop",
                f"https://shopping.google.com/search?q={search_query.replace(' ', '+')}"
            ])
        
        elif entity_type == "restaurant" and search_query:
            location_str = location.get("city", "") if location else ""
            if location_str:
                suggestions.extend([
                    f"https://www.yelp.com/search?find_desc={search_query.replace(' ', '+')}&find_loc={location_str.replace(' ', '+')}",
                    f"https://www.tripadvisor.com/Search?q={search_query.replace(' ', '+')}+{location_str.replace(' ', '+')}"
                ])
        
        elif entity_type == "property":
            location_str = location.get("city", "") if location else ""
            if location_str:
                suggestions.extend([
                    f"https://www.zillow.com/homes/{location_str.replace(' ', '-')}_rb/",
                    f"https://www.realtor.com/realestateandhomes-search/{location_str.replace(' ', '-')}"
                ])
        
        elif entity_type == "news" and keywords:
            topic = " ".join(keywords)
            suggestions.extend([
                f"https://news.google.com/search?q={topic.replace(' ', '+')}",
                f"https://www.reuters.com/search/news?blob={topic.replace(' ', '+')}"
            ])
        
        # Add general search as fallback
        if search_query:
            suggestions.append(f"https://www.google.com/search?q={search_query.replace(' ', '+')}")
        
        return suggestions

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using spaCy with enhanced recognition.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0,  # spaCy doesn't provide confidence scores by default
                "description": spacy.explain(ent.label_)
            })
        
        return entities

    def identify_data_types(self, intent_analysis: Dict[str, Any]) -> List[str]:
        """
        Identify expected data types based on intent analysis.
        
        Args:
            intent_analysis: Current intent analysis
            
        Returns:
            List of expected data types/fields
        """
        data_types = []
        
        entity_type = intent_analysis.get("entity_type") or intent_analysis.get("semantic_entity_type")
        
        if entity_type == "product":
            data_types.extend([
                "product_name", "price", "brand", "description", "rating",
                "reviews_count", "availability", "image_url", "product_url"
            ])
        
        elif entity_type == "restaurant":
            data_types.extend([
                "restaurant_name", "address", "phone", "rating", "price_range",
                "cuisine_type", "hours", "reviews", "website", "menu_url"
            ])
        
        elif entity_type == "property":
            data_types.extend([
                "address", "price", "bedrooms", "bathrooms", "square_feet",
                "lot_size", "year_built", "property_type", "listing_agent", "photos"
            ])
        
        elif entity_type == "news":
            data_types.extend([
                "headline", "summary", "publication_date", "author", "source",
                "article_url", "category", "tags"
            ])
        
        elif entity_type == "job":
            data_types.extend([
                "job_title", "company", "location", "salary", "description",
                "requirements", "benefits", "application_deadline", "job_url"
            ])
        
        # Add generic fields that are often useful
        data_types.extend(["title", "description", "url", "last_updated"])
        
        return list(set(data_types))  # Remove duplicates

    def _classify_content_type(self, query: str, intent_analysis: Dict[str, Any]) -> str:
        """
        Classify the query into specific content types for targeted extraction.
        
        Args:
            query: Original query string
            intent_analysis: Current intent analysis results
            
        Returns:
            Content type classification
        """
        query_lower = query.lower()
        
        # News content patterns
        news_patterns = [
            r'\b(news|latest|recent|breaking|update|announcement|report)\b',
            r'\b(article|story|headline|journalism|press release)\b',
            r'\b(today|yesterday|this week|current)\b.*\b(news|update)\b'
        ]
        
        # Product information patterns  
        product_patterns = [
            r'\b(price|cost|buy|purchase|sale|deal|offer)\b',
            r'\b(specs|specification|feature|model|version)\b',
            r'\b(available|availability|in stock|out of stock)\b',
            r'\b(compare|comparison|vs|versus)\b.*\b(product|model)\b'
        ]
        
        # Job listing patterns
        job_patterns = [
            r'\b(job|jobs|career|careers|hiring|employment)\b',
            r'\b(opening|openings|position|vacancy|vacancies)\b',
            r'\b(apply|application|resume|interview)\b',
            r'\b(salary|wage|compensation|benefits)\b'
        ]
        
        # Review patterns
        review_patterns = [
            r'\b(review|reviews|rating|ratings|feedback)\b',
            r'\b(opinion|experience|testimonial)\b',
            r'\b(recommend|recommendation|worth it)\b',
            r'\b(pros and cons|good|bad|quality)\b'
        ]
        
        # Financial data patterns
        financial_patterns = [
            r'\b(stock|share|stocks|shares|market)\b',
            r'\b(price|value|worth|valuation)\b.*\b(stock|share)\b',
            r'\b(earnings|revenue|profit|financial|quarterly)\b',
            r'\b(investor|investment|trading|ticker)\b'
        ]
        
        # Contact information patterns
        contact_patterns = [
            r'\b(contact|phone|email|address|location)\b',
            r'\b(hours|open|closed|schedule)\b',
            r'\b(support|customer service|help)\b',
            r'\b(office|headquarters|branch)\b'
        ]
        
        # Event patterns
        event_patterns = [
            r'\b(event|events|calendar|schedule)\b',
            r'\b(conference|meeting|webinar|seminar)\b',
            r'\b(when|date|time|location)\b.*\b(event|meeting)\b',
            r'\b(ticket|registration|rsvp)\b'
        ]
        
        # Check patterns in order of specificity
        for pattern in news_patterns:
            if re.search(pattern, query_lower):
                return "NEWS_ARTICLES"
        
        for pattern in financial_patterns:
            if re.search(pattern, query_lower):
                return "FINANCIAL_DATA"
                
        for pattern in job_patterns:
            if re.search(pattern, query_lower):
                return "JOB_LISTINGS"
                
        for pattern in product_patterns:
            if re.search(pattern, query_lower):
                return "PRODUCT_INFORMATION"
                
        for pattern in review_patterns:
            if re.search(pattern, query_lower):
                return "REVIEWS_RATINGS"
                
        for pattern in contact_patterns:
            if re.search(pattern, query_lower):
                return "CONTACT_INFORMATION"
                
        for pattern in event_patterns:
            if re.search(pattern, query_lower):
                return "EVENT_LISTINGS"
        
        # Use spaCy entities for additional context
        if self.nlp and intent_analysis.get('spacy_entities'):
            entities = intent_analysis['spacy_entities']
            entity_labels = [ent.get('label', '') for ent in entities]
            
            if 'MONEY' in entity_labels or 'PERCENT' in entity_labels:
                if any(word in query_lower for word in ['stock', 'share', 'market', 'trading']):
                    return "FINANCIAL_DATA"
                else:
                    return "PRODUCT_INFORMATION"
                    
            if 'ORG' in entity_labels and any(word in query_lower for word in ['job', 'career', 'work']):
                return "JOB_LISTINGS"
                
            if 'DATE' in entity_labels or 'TIME' in entity_labels:
                if any(word in query_lower for word in ['event', 'meeting', 'conference']):
                    return "EVENT_LISTINGS"
                elif any(word in query_lower for word in ['news', 'update', 'latest']):
                    return "NEWS_ARTICLES"
        
        # Default to general information
        return "INFORMATION"
    
    def _analyze_temporal_context(self, query: str) -> Dict[str, Any]:
        """
        Analyze temporal context of the query.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Temporal context information
        """
        query_lower = query.lower()
        
        # Temporal preference patterns
        latest_patterns = [
            r'\b(latest|newest|recent|current|new)\b',
            r'\b(today|this week|this month|now)\b',
            r'\b(updated|fresh|breaking)\b'
        ]
        
        historical_patterns = [
            r'\b(history|historical|past|archive)\b',
            r'\b(last year|previous|old|former)\b',
            r'\b(since|from|until|before)\b'
        ]
        
        specific_time_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(2020|2021|2022|2023|2024|2025)\b',
            r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        ]
        
        temporal_context = {
            "preference": "any",
            "urgency": "normal",
            "time_frame": "open"
        }
        
        # Check for latest/recent preferences
        for pattern in latest_patterns:
            if re.search(pattern, query_lower):
                temporal_context["preference"] = "latest"
                temporal_context["urgency"] = "high"
                break
        
        # Check for historical preferences  
        for pattern in historical_patterns:
            if re.search(pattern, query_lower):
                temporal_context["preference"] = "historical"
                temporal_context["time_frame"] = "past"
                break
                
        # Check for specific time references
        for pattern in specific_time_patterns:
            if re.search(pattern, query_lower):
                temporal_context["time_frame"] = "specific"
                break
        
        return temporal_context
    
    def _analyze_specificity(self, query: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the specificity level of the query.
        
        Args:
            query: Query string to analyze
            intent_analysis: Current intent analysis
            
        Returns:
            Specificity analysis
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        specificity = {
            "level": "medium",
            "scope": "broad",
            "detail_level": "standard"
        }
        
        # Determine specificity level
        if word_count <= 2:
            specificity["level"] = "low"
            specificity["scope"] = "very_broad"
        elif word_count <= 4:
            specificity["level"] = "medium"
            specificity["scope"] = "broad"
        elif word_count <= 8:
            specificity["level"] = "high" 
            specificity["scope"] = "focused"
        else:
            specificity["level"] = "very_high"
            specificity["scope"] = "very_focused"
        
        # Check for detail-level indicators
        detail_patterns = [
            r'\b(detailed|comprehensive|complete|full|in-depth)\b',
            r'\b(everything|all|total|entire)\b'
        ]
        
        summary_patterns = [
            r'\b(summary|overview|brief|quick|short)\b',
            r'\b(highlights|key points|main)\b'
        ]
        
        for pattern in detail_patterns:
            if re.search(pattern, query_lower):
                specificity["detail_level"] = "detailed"
                break
                
        for pattern in summary_patterns:
            if re.search(pattern, query_lower):
                specificity["detail_level"] = "summary"
                break
        
        # Use entity count as specificity indicator
        entities = intent_analysis.get('spacy_entities', [])
        if len(entities) >= 3:
            specificity["level"] = "high"
            specificity["scope"] = "focused"
        
        return specificity
