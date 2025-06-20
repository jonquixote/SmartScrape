"""
Search Term Generator Module

This module provides functionality to generate optimized search terms for different
website types, enhance basic search terms with variations, and normalize search input.
It uses both rule-based approaches and AI to create effective search queries.
"""

import logging
import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
import urllib.parse
from collections import defaultdict

# spaCy imports with fallback
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
    # Try to load the best available English model
    SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
    spacy_nlp = None
    SPACY_AVAILABLE = False
    
    for model_name in SPACY_MODELS:
        try:
            spacy_nlp = spacy.load(model_name)
            SPACY_AVAILABLE = True
            logger = logging.getLogger("SearchTermGenerator")
            logger.info(f"spaCy successfully loaded with model: {model_name}")
            break
        except OSError:
            continue
    
    if not SPACY_AVAILABLE:
        logger = logging.getLogger("SearchTermGenerator")
        logger.warning("No spaCy models available. Install with: python -m spacy download en_core_web_lg")
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    spacy_nlp = None
    SPACY_STOP_WORDS = set()

# Fallback NLTK imports (only when spaCy unavailable)
NLTK_AVAILABLE = False
if not SPACY_AVAILABLE:
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        NLTK_AVAILABLE = True
    except ImportError:
        NLTK_AVAILABLE = False
# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SearchTermGenerator")

import google.generativeai as genai

import config
from ai_helpers.intent_parser import get_intent_parser
from core.service_interface import BaseService

# Ensure NLTK data is downloaded (only as fallback when spaCy unavailable)
def ensure_nltk_data():
    """Download required NLTK data if not already present (used as fallback when spaCy unavailable)"""
    if not SPACY_AVAILABLE and NLTK_AVAILABLE:
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")

# Download required NLTK data only if spaCy not available
if not SPACY_AVAILABLE and NLTK_AVAILABLE:
    ensure_nltk_data()

class SearchTermGenerator(BaseService):
    """
    Generates optimized search terms based on user intent and website characteristics.
    Provides variations, expansions, and domain-specific term optimization.
    """
    
    def __init__(self, use_ai: bool = True, service_registry=None):
        """
        Initialize the search term generator.
        
        Args:
            use_ai: Whether to use AI for enhancing search terms
            service_registry: Service registry for dependency injection
        """
        self.use_ai = use_ai
        self._initialized = False
        self.service_registry = service_registry
        
        # These will be initialized in initialize() method
        self.stopwords = None
        self.intent_parser = None
        self.wordnet = None
        self.nlp = None  # Always initialize nlp attribute to prevent AttributeError
        
        # Domain-specific term patterns
        self.domain_patterns = {
            'ecommerce': {
                'keywords': ['buy', 'price', 'shop', 'product', 'purchase', 'order', 'shipping'],
                'prefixes': ['best', 'cheap', 'discount', 'sale', 'new', 'top'],
                'suffixes': ['review', 'deal', 'online', 'shipping', 'warranty'],
                'operators': ['-', '+', '"', 'OR'],
                'symbols': ['$', '%', '&']
            },
            'realestate': {
                'keywords': ['house', 'home', 'apartment', 'condo', 'property', 'rent', 'sale'],
                'prefixes': ['cheap', 'luxury', 'new', 'modern', 'spacious'],
                'suffixes': ['for sale', 'for rent', 'near me', 'square feet', 'bedroom'],
                'operators': ['-', '+', '"', 'OR'],
                'symbols': ['$', '#', '&']
            },
            'job': {
                'keywords': ['job', 'career', 'salary', 'hiring', 'remote', 'position', 'apply'],
                'prefixes': ['entry', 'senior', 'remote', 'full-time', 'part-time'],
                'suffixes': ['experience', 'degree', 'skills', 'salary', 'benefits'],
                'operators': ['-', '+', '"', 'OR'],
                'symbols': ['$', '%', '&']
            },
            'travel': {
                'keywords': ['hotel', 'flight', 'vacation', 'resort', 'booking', 'tour', 'travel'],
                'prefixes': ['cheap', 'luxury', 'best', 'all-inclusive', 'family'],
                'suffixes': ['deal', 'discount', 'package', 'review', 'rating'],
                'operators': ['-', '+', '"', 'OR'],
                'symbols': ['$', '%', '&']
            },
            'content': {
                'keywords': ['article', 'news', 'blog', 'information', 'guide', 'tutorial', 'review'],
                'prefixes': ['latest', 'best', 'comprehensive', 'ultimate', 'complete'],
                'suffixes': ['guide', 'review', 'tutorial', 'explained', 'tips'],
                'operators': ['-', '+', '"', 'OR'],
                'symbols': ['#', '@', '&']
            }
        }
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service with the given configuration."""
        if self._initialized:
            return
            
        # Apply config if provided
        if config:
            self.use_ai = config.get('use_ai', self.use_ai)
            
        # Initialize stopwords with spaCy first, fallback to NLTK
        if SPACY_AVAILABLE:
            self.stopwords = SPACY_STOP_WORDS
        elif NLTK_AVAILABLE:
            try:
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Failed to load NLTK stopwords: {e}")
                self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        else:
            # Basic fallback stopwords
            self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Initialize intent parser - try service registry first, then fallback
        self.intent_parser = None
        if self.service_registry:
            try:
                self.intent_parser = self.service_registry.get_service("intent_parser")
                if self.intent_parser is not None:
                    logger.info("Using intent_parser from service registry")
                else:
                    logger.warning("intent_parser from service registry is None")
            except Exception as e:
                logger.warning(f"Could not get intent_parser from service registry: {e}")
        
        # Ensure we always have a valid intent_parser
        if self.intent_parser is None:
            self.intent_parser = get_intent_parser(use_ai=self.use_ai)
            logger.info("Using fallback intent_parser")
        
        # Initialize spaCy for synonym generation (spaCy-first approach)
        self.nlp = None
        self.wordnet = None  # Keep for backward compatibility, but will be None
        
        if SPACY_AVAILABLE and spacy_nlp:
            try:
                self.nlp = spacy_nlp
                logger.info("spaCy model loaded successfully for synonym generation")
            except Exception as e:
                logger.warning(f"Failed to use spaCy model: {e}")
                
        # If spaCy initialization failed, ensure we still have nlp available by trying to load it again
        if self.nlp is None and SPACY_AVAILABLE:
            try:
                import spacy
                # Load spaCy with model priority
                SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
                for model_name in SPACY_MODELS:
                    try:
                        self.nlp = spacy.load(model_name)
                        break
                    except OSError:
                        continue
                logger.info("Successfully loaded spaCy model directly")
            except Exception as e:
                logger.warning(f"Could not load spaCy model directly: {e}")
        
        # Fallback: try to load wordnet if spaCy unavailable
        if not self.nlp and NLTK_AVAILABLE:
            try:
                from nltk.corpus import wordnet
                self.wordnet = wordnet
                logger.info("Falling back to NLTK wordnet for synonym generation")
            except Exception:
                logger.warning("Neither spaCy nor NLTK wordnet available, synonym generation disabled")
        
        # Initialize AI model attribute
        self.ai_model = None
        
        # Configure AI if enabled
        if self.use_ai:
            self._setup_ai()
            
        self._initialized = True
        logger.info("SearchTermGenerator initialized")
    
    def _setup_ai(self) -> None:
        """Configure AI components for advanced search term generation."""
        try:
            # Removed "from config import config" as it's already imported as a module
            if hasattr(config, 'GEMINI_API_KEY') and config.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=config.GEMINI_API_KEY)
                self.ai_model = 'gemini-2.0-flash'
                logger.info("AI model configured for search term generation")
            else:
                logger.warning("No AI API key found, using fallback methods")
                self.ai_model = None
        except Exception as e:
            logger.warning(f"Failed to setup AI: {e}, using fallback methods")
            self.ai_model = None
    
    def shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        self._initialized = False
        logger.info("SearchTermGenerator shut down")
    
    @property
    def name(self) -> str:
        """Return the name of the service."""
        return "search_term_generator"
    
    async def generate_search_terms(self, 
                                  query: str,
                                  site_type: Optional[str] = None,
                                  site_url: Optional[str] = None,
                                  num_variations: int = 3) -> Dict[str, Any]:
        """
        Generate optimized search terms from a user query.
        
        Args:
            query: The user's search query
            site_type: The type of website (e.g., 'ecommerce', 'realestate')
            site_url: The URL of the target website
            num_variations: Number of variations to generate
            
        Returns:
            Dictionary with various search term options
        """
        # Ensure intent_parser is available
        if self.intent_parser is None:
            self.intent_parser = get_intent_parser(use_ai=self.use_ai)
            logger.warning("intent_parser was None, created new instance")
        
        # Parse intent from the query
        intent = await self.intent_parser.parse_query(query)
        
        # Extract domain from URL if available
        domain = None
        if site_url:
            parsed_url = urllib.parse.urlparse(site_url)
            domain = parsed_url.netloc
        
        # Determine site type if not provided
        if not site_type:
            site_type = self._detect_site_type(intent, domain)
        
        # Generate initial search terms
        main_term = self._extract_main_term(intent, query)
        
        # Remove common stopwords and normalize
        cleaned_term = self._normalize_search_term(main_term)
        
        # Generate variations based on site type
        variations = await self._generate_variations(cleaned_term, site_type, intent, num_variations)
        
        # Create query expansion with synonyms and related terms
        expanded_terms = await self._expand_query_terms(cleaned_term, site_type, intent)
        
        # Generate domain-specific optimized terms
        optimized_term = self._optimize_for_domain(cleaned_term, site_type, intent)
        
        # Generate advanced search operators for specific domains if available
        advanced_term = self._add_search_operators(cleaned_term, site_type, intent)
        
        # Extract important keywords for the search
        keywords = self._extract_keywords(intent, query)
        
        # If AI is enabled, enhance search terms with AI suggestions
        ai_suggestions = {}
        if self.use_ai:
            ai_suggestions = await self._generate_ai_search_terms(
                query=query,
                intent=intent,
                site_type=site_type,
                domain=domain
            )
        
        # Assemble results
        result = {
            "original_query": query,
            "main_term": main_term,
            "cleaned_term": cleaned_term,
            "optimized_term": optimized_term,
            "advanced_term": advanced_term,
            "variations": variations,
            "expanded_terms": expanded_terms,
            "keywords": keywords,
            "site_type": site_type
        }
        
        # Add AI suggestions if available
        if ai_suggestions:
            result.update({
                "ai_optimized_term": ai_suggestions.get("optimized_term", ""),
                "ai_variations": ai_suggestions.get("variations", []),
                "ai_keywords": ai_suggestions.get("keywords", []),
                "ai_advanced_term": ai_suggestions.get("advanced_term", "")
            })
        
        return result
    
    def _detect_site_type(self, intent: Dict[str, Any], domain: Optional[str] = None) -> str:
        """
        Detect the website type based on intent and domain.
        
        Args:
            intent: The parsed user intent
            domain: The website domain
            
        Returns:
            Detected site type
        """
        # Check for domain-specific indicators first
        if domain:
            domain_lower = domain.lower()
            
            # E-commerce sites
            ecommerce_domains = [
                'amazon', 'ebay', 'walmart', 'etsy', 'shopify', 'aliexpress',
                'bestbuy', 'target', 'shop', 'store', 'market', 'buy'
            ]
            if any(term in domain_lower for term in ecommerce_domains):
                return 'ecommerce'
            
            # Real estate sites
            realestate_domains = [
                'zillow', 'realtor', 'redfin', 'trulia', 'homes', 'apartments',
                'property', 'house', 'real-estate', 'rent'
            ]
            if any(term in domain_lower for term in realestate_domains):
                return 'realestate'
            
            # Job sites
            job_domains = [
                'indeed', 'linkedin', 'monster', 'glassdoor', 'careers', 'jobs',
                'work', 'employment', 'career', 'job'
            ]
            if any(term in domain_lower for term in job_domains):
                return 'job'
            
            # Travel sites
            travel_domains = [
                'expedia', 'booking', 'airbnb', 'hotels', 'tripadvisor', 'kayak',
                'travel', 'tour', 'flight', 'vacation'
            ]
            if any(term in domain_lower for term in travel_domains):
                return 'travel'
        
        # Check intent if domain doesn't yield a clear type
        if intent:
            original_query = intent.get("original_query", "")
            # Ensure original_query is a string, not a dict
            if isinstance(original_query, dict):
                # If it's a dict, try to extract query text from common keys
                original_query = original_query.get("query") or original_query.get("text") or str(original_query)
            elif not isinstance(original_query, str):
                original_query = str(original_query)
            
            query = original_query.lower() if original_query else ""
            
            # Check for e-commerce terms
            ecommerce_terms = ['buy', 'price', 'shop', 'product', 'purchase', 'order']
            if any(term in query for term in ecommerce_terms):
                return 'ecommerce'
            
            # Check for real estate terms
            realestate_terms = ['house', 'home', 'apartment', 'condo', 'property', 'rent', 'sale']
            if any(term in query for term in realestate_terms):
                return 'realestate'
            
            # Check for job search terms
            job_terms = ['job', 'career', 'salary', 'hiring', 'position', 'apply']
            if any(term in query for term in job_terms):
                return 'job'
            
            # Check for travel terms
            travel_terms = ['hotel', 'flight', 'vacation', 'resort', 'booking', 'tour']
            if any(term in query for term in travel_terms):
                return 'travel'
            
            # Check entity type if available
            entity_type = intent.get("entity_type", "").lower()
            if entity_type in ["product", "item"]:
                return 'ecommerce'
            elif entity_type in ["home", "house", "apartment", "property"]:
                return 'realestate'
            elif entity_type in ["job", "position", "career"]:
                return 'job'
            elif entity_type in ["hotel", "flight", "trip"]:
                return 'travel'
        
        # Default to content sites if no specific type is detected
        return 'content'
    
    def _extract_main_term(self, intent: Dict[str, Any], query: str) -> str:
        """
        Extract the main search term from intent and query.
        
        Args:
            intent: The parsed user intent
            query: The original query string
            
        Returns:
            Main search term
        """
        # Use target_item from intent if available
        if "target_item" in intent and intent["target_item"]:
            return intent["target_item"]
        
        # Use entity_type and additional criteria if available
        if "entity_type" in intent and intent["entity_type"]:
            entity_type = intent["entity_type"]
            
            # Combine with key specific criteria
            if "specific_criteria" in intent and intent["specific_criteria"]:
                criteria_str = " ".join(intent["specific_criteria"])
                return f"{entity_type} {criteria_str}"
            
            return entity_type
        
        # Fall back to the original query with some basic cleaning
        return re.sub(r'find|search for|tell me about|show me|looking for', '', query, flags=re.IGNORECASE).strip()
    
    def _normalize_search_term(self, term: str) -> str:
        """
        Normalize a search term by removing stopwords and excess whitespace.
        
        Args:
            term: The search term to normalize
            
        Returns:
            Normalized search term
        """
        # Tokenize the term using spaCy first, then fallback
        try:
            if SPACY_AVAILABLE and spacy_nlp:
                doc = spacy_nlp(term.lower())
                tokens = [token.text for token in doc if token.is_alpha]
            else:
                # Fallback to simple regex-based tokenization
                tokens = re.findall(r'\b[a-zA-Z]+\b', term.lower())
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}, using regex fallback")
            tokens = re.findall(r'\b[a-zA-Z]+\b', term.lower())
        
        # Ensure stopwords is available, fallback to empty set if None
        if self.stopwords is None:
            logger.warning("stopwords is None, initializing basic stopwords set")
            self.stopwords = {
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
            }
        
        # Remove stopwords, but preserve some important ones that might be needed
        important_stopwords = {'for', 'with', 'without', 'near', 'in', 'at', 'by', 'on'}
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stopwords or token in important_stopwords
        ]
        
        # Remove punctuation except for specialized characters
        keep_chars = {'-', '+', '$', '%', '#', '"'}
        filtered_tokens = [
            token if token in keep_chars else re.sub(r'[^\w\s]', '', token) 
            for token in filtered_tokens
        ]
        
        # Filter out empty tokens
        filtered_tokens = [token for token in filtered_tokens if token]
        
        # Join tokens back into a string
        normalized_term = ' '.join(filtered_tokens)
        
        # Remove excess whitespace
        normalized_term = re.sub(r'\s+', ' ', normalized_term).strip()
        
        return normalized_term
    
    async def _generate_variations(self, 
                                term: str, 
                                site_type: str,
                                intent: Dict[str, Any],
                                num_variations: int) -> List[str]:
        """
        Generate variations of the search term.
        
        Args:
            term: The normalized search term
            site_type: The type of website
            intent: The parsed user intent
            num_variations: Number of variations to generate
            
        Returns:
            List of term variations
        """
        variations = []
        
        # Break the term into tokens for manipulation
        tokens = term.split()
        
        # Get domain patterns for the site type
        patterns = self.domain_patterns.get(site_type, self.domain_patterns['content'])
        
        # 1. Variation by adding a prefix
        if tokens:
            for prefix in patterns['prefixes'][:2]:  # Limit to 2 prefixes
                if prefix not in term.lower():
                    variations.append(f"{prefix} {term}")
        
        # 2. Variation by adding a suffix
        for suffix in patterns['suffixes'][:2]:  # Limit to 2 suffixes
            if suffix not in term.lower():
                variations.append(f"{term} {suffix}")
        
        # 3. Variation by reordering tokens (if more than one token)
        if len(tokens) > 1:
            reordered = ' '.join(reversed(tokens))
            variations.append(reordered)
        
        # 4. Variation by adding a domain-specific keyword
        for keyword in patterns['keywords'][:2]:  # Limit to 2 keywords
            if keyword not in term.lower():
                variations.append(f"{term} {keyword}")
        
        # 5. Variation using synonyms for key terms
        synonym_variations = await self._generate_synonym_variations(term)
        variations.extend(synonym_variations[:2])  # Limit to 2 synonym variations
        
        # 6. Variation by adding location if present in intent
        if "location" in intent and intent["location"]:
            location_data = intent["location"]
            
            # Handle location as either dict or string
            if isinstance(location_data, dict):
                # Extract location string from dict
                location_parts = []
                for field in ['city', 'state', 'zip_code']:
                    if field in location_data and location_data[field]:
                        location_parts.append(str(location_data[field]))
                
                if location_parts:
                    location = ' '.join(location_parts)
                else:
                    location = None
            elif isinstance(location_data, str):
                location = location_data
            else:
                location = None
            
            if location and location.lower() not in term.lower():
                variations.append(f"{term} {location}")
                variations.append(f"{term} in {location}")
        
        # 7. Variation by adding questions for content sites
        if site_type == 'content':
            variations.append(f"how to {term}")
            variations.append(f"what is {term}")
            variations.append(f"guide to {term}")
        
        # 8. Variation for real estate with property type
        if site_type == 'realestate':
            property_types = ['house', 'apartment', 'condo', 'townhouse']
            for prop_type in property_types:
                if prop_type not in term.lower():
                    variations.append(f"{prop_type} {term}")
                    break  # Just add one property type
        
        # 9. Variation for e-commerce with price indications
        if site_type == 'ecommerce':
            variations.append(f"best {term}")
            variations.append(f"cheap {term}")
            variations.append(f"{term} discount")
        
        # Deduplicate and limit variations
        unique_variations = []
        for v in variations:
            normalized_v = self._normalize_search_term(v)
            if normalized_v and normalized_v not in unique_variations and normalized_v != term:
                unique_variations.append(normalized_v)
        
        # Return limited number of variations
        return unique_variations[:num_variations]
    
    async def _generate_synonym_variations(self, term: str) -> List[str]:
        """
        Generate variations of the term using synonyms.
        
        Args:
            term: The normalized search term
            
        Returns:
            List of term variations using synonyms
        """
        variations = []
        tokens = term.split()
        
        # Use spaCy for synonym generation (preferred) or fall back to wordnet
        if self.nlp is None and self.wordnet is None:
            logger.warning("Neither spaCy nor wordnet available, cannot generate synonym variations")
            return variations
        
        # Generate synonym for each significant word
        for i, token in enumerate(tokens):
            # Skip very short words or numbers
            if len(token) <= 3 or token.isdigit():
                continue
            
            # Find synonyms using spaCy or wordnet
            synonyms = set()
            try:
                if self.nlp is not None:
                    # Use spaCy word vectors for similarity-based synonyms
                    doc = self.nlp(token)
                    if doc and doc[0].has_vector:
                        # Get similar words from spaCy's vocabulary (simplified approach)
                        # Note: This is a basic implementation. Production code might use word2vec or other embeddings
                        word_token = doc[0]
                        # Find similar words in vocabulary (limited for performance)
                        vocab_words = [w for w in self.nlp.vocab if w.has_vector and w.is_alpha and len(w.text) > 3][:1000]
                        similarities = [(w.text, word_token.similarity(w)) for w in vocab_words if w != word_token]
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        
                        # Take top 3 similar words with similarity > 0.6
                        for word, sim in similarities[:3]:
                            if sim > 0.6 and word != token and len(word) > 1:
                                synonyms.add(word)
                elif self.wordnet is not None:
                    # Fallback to wordnet
                    for syn in self.wordnet.synsets(token):
                        for lemma in syn.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if synonym != token and len(synonym) > 1:
                                synonyms.add(synonym)
            except Exception as e:
                logger.warning(f"Error getting synonyms for token '{token}': {e}")
                continue
            
            # Create variations by replacing the token with its synonyms
            for synonym in list(synonyms)[:2]:  # Limit to 2 synonyms per token
                new_tokens = tokens.copy()
                new_tokens[i] = synonym
                variations.append(' '.join(new_tokens))
                
                if len(variations) >= 3:  # Limit total variations
                    break
            
            if len(variations) >= 3:
                break
        
        return variations
    
    async def _expand_query_terms(self, 
                               term: str, 
                               site_type: str, 
                               intent: Dict[str, Any]) -> List[str]:
        """
        Expand the query with related terms for better coverage.
        
        Args:
            term: The normalized search term
            site_type: The type of website
            intent: The parsed user intent
            
        Returns:
            List of expanded query terms
        """
        expanded = []
        tokens = term.split()
        
        # 1. Add related terms based on spaCy or WordNet hypernyms (more general terms)
        for token in tokens:
            if len(token) <= 3:
                continue
                
            hypernyms = set()
            
            # Try spaCy approach first
            if self.nlp is not None:
                try:
                    doc = self.nlp(token)
                    if doc and doc[0].has_vector:
                        # Find similar tokens using spaCy's word vectors
                        similar_tokens = self.nlp.vocab.vectors.most_similar(
                            doc[0].vector.reshape(1, -1), n=5
                        )
                        if similar_tokens[0].size > 0:
                            for idx in similar_tokens[0][:2]:  # Top 2 similar
                                similar_word = self.nlp.vocab[idx].text
                                if similar_word != token.lower() and len(similar_word) > 1:
                                    hypernyms.add(similar_word)
                except Exception as e:
                    pass  # Silent fallback to wordnet
            
            # Fallback to wordnet if spaCy didn't work
            if not hypernyms and self.wordnet is not None:
                try:
                    for synset in self.wordnet.synsets(token)[:2]:  # Limit to 2 synsets
                        for hypernym in synset.hypernyms()[:1]:  # Limit to 1 hypernym per synset
                            for lemma in hypernym.lemmas():
                                hypernym_name = lemma.name().replace('_', ' ')
                                if hypernym_name != token and len(hypernym_name) > 1:
                                    hypernyms.add(hypernym_name)
                except Exception as e:
                    self.logger.warning(f"Error accessing wordnet for token '{token}': {e}")
            
            # Add expanded terms combining the original term with hypernyms
            for hypernym in list(hypernyms)[:2]:  # Limit to 2 hypernyms
                expanded.append(f"{term} {hypernym}")
        
        # 2. Add related terms based on domain patterns
        patterns = self.domain_patterns.get(site_type, self.domain_patterns['content'])
        for keyword in patterns['keywords'][:3]:  # Limit to 3 keywords
            if keyword not in term.lower():
                expanded.append(f"{term} {keyword}")
        
        # 3. Add variations with modifiers based on site type
        if site_type == 'ecommerce':
            expanded.append(f"{term} review")
            expanded.append(f"{term} best price")
            expanded.append(f"{term} discount")
            
        elif site_type == 'realestate':
            expanded.append(f"{term} for rent")
            expanded.append(f"{term} for sale")
            expanded.append(f"{term} near me")
            
        elif site_type == 'job':
            expanded.append(f"{term} job")
            expanded.append(f"{term} career")
            expanded.append(f"entry level {term}")
            expanded.append(f"remote {term}")
            
        elif site_type == 'travel':
            expanded.append(f"{term} hotel")
            expanded.append(f"cheap {term}")
            expanded.append(f"{term} vacation")
            expanded.append(f"{term} deals")
            
        elif site_type == 'content':
            expanded.append(f"{term} guide")
            expanded.append(f"{term} tutorial")
            expanded.append(f"how to {term}")
            expanded.append(f"{term} examples")
        
        # 4. Add terms from original query that might have been filtered out
        original_query = intent.get("original_query", "")
        if original_query:
            # Extract quoted phrases from original query
            quoted_phrases = re.findall(r'"([^"]*)"', original_query)
            for phrase in quoted_phrases:
                if phrase.lower() not in term.lower():
                    expanded.append(f"{term} \"{phrase}\"")
        
        # Deduplicate and filter
        unique_expanded = []
        for exp in expanded:
            normalized_exp = self._normalize_search_term(exp)
            if normalized_exp and normalized_exp not in unique_expanded and normalized_exp != term:
                unique_expanded.append(normalized_exp)
        
        return unique_expanded[:5]  # Limit to 5 expanded terms
    
    def _optimize_for_domain(self, term: str, site_type: str, intent: Dict[str, Any]) -> str:
        """
        Optimize the search term for a specific domain/site type.
        
        Args:
            term: The normalized search term
            site_type: The type of website
            intent: The parsed user intent
            
        Returns:
            Domain-optimized search term
        """
        # Get domain patterns
        patterns = self.domain_patterns.get(site_type, self.domain_patterns['content'])
        
        # Start with the base term
        optimized = term
        
        # Apply domain-specific optimizations
        if site_type == 'ecommerce':
            # Add product qualifiers if not present
            if not any(kw in term.lower() for kw in ['price', 'buy', 'cheap', 'best']):
                prefix = patterns['prefixes'][0]  # Use first prefix
                optimized = f"{prefix} {optimized}"
            
            # Add brand if specified in intent
            if 'specific_criteria' in intent:
                for criterion in intent.get('specific_criteria', []):
                    if 'brand:' in criterion.lower():
                        brand = criterion.split(':', 1)[1].strip()
                        if brand not in optimized.lower():
                            optimized = f"{brand} {optimized}"
                        break
        
        elif site_type == 'realestate':
            # Check if property type is specified
            property_types = ['house', 'apartment', 'condo', 'townhouse']
            has_property_type = any(prop in term.lower() for prop in property_types)
            
            if not has_property_type:
                # Default to house if not specified
                optimized = f"house {optimized}"
            
            # Add rental/purchase intent if not specified
            if not any(intent in term.lower() for intent in ['rent', 'sale', 'buy']):
                # Check intent specific_criteria for rent/buy indication
                if 'specific_criteria' in intent:
                    criteria = ' '.join(intent.get('specific_criteria', []))
                    if 'rent' in criteria.lower():
                        optimized = f"{optimized} for rent"
                    elif any(term in criteria.lower() for term in ['buy', 'sale', 'purchase']):
                        optimized = f"{optimized} for sale"
                    else:
                        # Default to "for sale" if not specified
                        optimized = f"{optimized} for sale"
                else:
                    # Default to "for sale" if not specified
                    optimized = f"{optimized} for sale"
        
        elif site_type == 'job':
            # Add job indicator if not present
            if not any(kw in term.lower() for kw in ['job', 'position', 'career']):
                optimized = f"{optimized} job"
            
            # Add location if specified in intent but not in term
            if 'location' in intent and intent['location']:
                location = intent['location']
                if location.lower() not in optimized.lower():
                    optimized = f"{optimized} in {location}"
            
            # Add job level if specified in specific_criteria
            if 'specific_criteria' in intent:
                criteria = ' '.join(intent.get('specific_criteria', []))
                job_levels = ['entry', 'junior', 'senior', 'executive', 'lead', 'manager']
                for level in job_levels:
                    if level in criteria.lower() and level not in optimized.lower():
                        optimized = f"{level} {optimized}"
                        break
        
        elif site_type == 'travel':
            # Add travel indicator if not present
            if not any(kw in term.lower() for kw in ['hotel', 'flight', 'vacation', 'resort']):
                # Default to hotel
                optimized = f"{optimized} hotel"
            
            # Add location if specified in intent but not in term
            if 'location' in intent and intent['location']:
                location = intent['location']
                if isinstance(location, str) and location.lower() not in optimized.lower():
                    optimized = f"{optimized} in {location}"
        
        elif site_type == 'content':
            # Add content indicator if not present
            if not any(kw in term.lower() for kw in ['guide', 'tutorial', 'review', 'article']):
                # Default to guide
                optimized = f"{optimized} guide"
        
        # Normalize the result
        optimized = self._normalize_search_term(optimized)
        
        return optimized
    
    def _add_search_operators(self, term: str, site_type: str, intent: Dict[str, Any]) -> str:
        """
        Add search operators to create advanced search queries.
        
        Args:
            term: The normalized search term
            site_type: The type of website
            intent: The parsed user intent
            
        Returns:
            Advanced search term with operators
        """
        # Get domain patterns for operator styles
        patterns = self.domain_patterns.get(site_type, self.domain_patterns['content'])
        
        # Start with the base term
        advanced = term
        
        # Split into tokens for manipulation
        tokens = term.split()
        
        # 1. Add quotation marks around multi-word key terms
        if len(tokens) > 1:
            # Identify key phrases (typically noun phrases)
            for i in range(len(tokens) - 1):
                if len(tokens[i]) > 3 and len(tokens[i+1]) > 3:  # Only consider substantive words
                    phrase = f"{tokens[i]} {tokens[i+1]}"
                    advanced = advanced.replace(phrase, f'"{phrase}"')
                    break  # Only replace one phrase to avoid over-complicating
        
        # 2. Add explicit inclusion operators
        if 'specific_criteria' in intent:
            required_terms = []
            for criterion in intent.get('specific_criteria', []):
                # Parse key:value criteria
                if ':' in criterion:
                    key, value = criterion.split(':', 1)
                    if value.strip() and value.strip().lower() not in advanced.lower():
                        required_terms.append(value.strip())
            
            # Add up to 2 required terms with + operator
            for term in required_terms[:2]:
                if term not in advanced:
                    advanced = f"{advanced} +{term}"
        
        # 3. Add explicit exclusion operators if negative criteria exist
        if 'negative_criteria' in intent:
            excluded_terms = intent.get('negative_criteria', [])
            for term in excluded_terms[:2]:  # Limit to 2 exclusions
                if term not in advanced:
                    advanced = f"{advanced} -{term}"
        
        # 4. Add OR operators for alternatives
        if 'alternatives' in intent:
            alternatives = intent.get('alternatives', [])
            if len(alternatives) > 1:
                alt_phrase = ' OR '.join(alternatives[:2])  # Limit to 2 alternatives
                advanced = f"{advanced} ({alt_phrase})"
        
        # Ensure proper spacing
        advanced = re.sub(r'\s+', ' ', advanced).strip()
        
        return advanced
    
    def _extract_keywords(self, intent: Dict[str, Any], query: str) -> List[str]:
        """
        Extract important keywords for the search.
        
        Args:
            intent: The parsed user intent
            query: The original query string
            
        Returns:
            List of important keywords
        """
        keywords = set()
        
        # Add keywords from intent if available
        if 'keywords' in intent:
            keywords.update(intent.get('keywords', []))
        
        # Add target_item if available
        if 'target_item' in intent and intent['target_item']:
            target_item = intent['target_item']
            # Split multi-word target items
            for word in target_item.split():
                if len(word) > 3 and word.lower() not in self.stopwords:
                    keywords.add(word.lower())
        
        # Add entity_type if available
        if 'entity_type' in intent and intent['entity_type']:
            keywords.add(intent['entity_type'].lower())
        
        # Add location if available
        if 'location' in intent and intent['location']:
            location = intent['location']
            if isinstance(location, str):
                keywords.add(location.lower())
            elif isinstance(location, dict):
                # If location is a dict, extract string values
                for key, value in location.items():
                    if isinstance(value, str):
                        keywords.add(value.lower())
            elif isinstance(location, list):
                # If location is a list, add all string items
                for item in location:
                    if isinstance(item, str):
                        keywords.add(item.lower())
        
        # Add specific criteria
        if 'specific_criteria' in intent:
            for criterion in intent.get('specific_criteria', []):
                # Extract values from key:value criteria
                if ':' in criterion:
                    _, value = criterion.split(':', 1)
                    if value.strip():
                        keywords.add(value.strip().lower())
                else:
                    # Split by spaces and add individual words
                    for word in criterion.split():
                        if len(word) > 3 and word.lower() not in self.stopwords:
                            keywords.add(word.lower())
        
        # Extract additional keywords from the query using NLP
        if SPACY_AVAILABLE and spacy_nlp:
            doc = spacy_nlp(query.lower())
            # Extract nouns and adjectives using spaCy POS tags
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ'] and 
                    len(token.text) > 3 and 
                    not token.is_stop and 
                    token.is_alpha):
                    keywords.add(token.lemma_.lower())
        else:
            # Fallback to NLTK if available
            if NLTK_AVAILABLE:
                try:
                    from nltk.tokenize import word_tokenize
                    import nltk
                    tokens = word_tokenize(query.lower())
                    pos_tags = nltk.pos_tag(tokens)
                    # Extract nouns and adjectives
                    for word, pos in pos_tags:
                        if pos.startswith('NN') or pos.startswith('JJ'):  # Nouns and adjectives
                            if len(word) > 3 and word.lower() not in self.stopwords:
                                keywords.add(word.lower())
                except Exception as e:
                    logger.warning(f"NLTK tokenization failed: {e}")
                    # Basic regex fallback
                    words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
                    for word in words:
                        if word not in self.stopwords:
                            keywords.add(word)
            else:
                # Basic regex fallback when neither spaCy nor NLTK available
                words = re.findall(r'\b[a-zA-Z]{4,}\b', query.lower())
                for word in words:
                    if word not in self.stopwords:
                        keywords.add(word)
        
        return list(keywords)
    
    async def _generate_ai_search_terms(self, 
                                      query: str,
                                      intent: Dict[str, Any],
                                      site_type: str,
                                      domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate search terms using AI.
        
        Args:
            query: The original query string
            intent: The parsed user intent
            site_type: The type of website
            domain: The website domain
            
        Returns:
            Dictionary with AI-generated search terms
        """
        if not self.use_ai or not hasattr(self, 'ai_model') or self.ai_model is None:
            return {}
            
        try:
            model = genai.GenerativeModel(self.ai_model)
            
            site_info = ""
            if domain:
                site_info = f"\nTarget website domain: {domain}"
            
            prompt = f"""
            I need to generate optimized search terms for a website's search function.

            Original user query: "{query}"
            Website category: {site_type}{site_info}
            
            User intent details: {json.dumps(intent, indent=2)}
            
            Please generate the following:
            1. A single optimized search term that would work best for this site type
            2. 3 variations of the search term for different aspects of the query
            3. A list of 5-10 important keywords that should be in the search results
            4. An advanced search query using operators like quotes, plus, minus, OR, etc.
            
            Format your response as a JSON object with these keys:
            - optimized_term: the single best search term
            - variations: array of alternative search terms
            - keywords: array of important keywords
            - advanced_term: search term with operators
            """
            
            response = model.generate_content(prompt)
            
            # Parse the response
            response_text = response.text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
                
            ai_terms = json.loads(json_str)
            
            return ai_terms
                
        except Exception as e:
            logger.error(f"Error generating AI search terms: {str(e)}")
            return {}
    
    async def relax_search_term(self, 
                              term: str, 
                              site_type: str, 
                              failed_terms: Optional[List[str]] = None) -> List[str]:
        """
        Generate progressively more relaxed search terms when initial searches fail.
        
        Args:
            term: The search term that failed
            site_type: The type of website
            failed_terms: List of previously failed terms to avoid
            
        Returns:
            List of relaxed search terms to try
        """
        failed_terms = failed_terms or []
        relaxed_terms = []
        
        # 1. Remove quotes if present
        if '"' in term:
            relaxed = term.replace('"', '')
            if relaxed not in failed_terms:
                relaxed_terms.append(relaxed)
        
        # 2. Remove operators if present
        if any(op in term for op in ['+', '-', 'OR']):
            relaxed = re.sub(r'[+\-]', '', term)
            relaxed = re.sub(r'\sOR\s', ' ', relaxed)
            relaxed = self._normalize_search_term(relaxed)
            if relaxed not in failed_terms:
                relaxed_terms.append(relaxed)
        
        # 3. Remove domain-specific terms
        patterns = self.domain_patterns.get(site_type, self.domain_patterns['content'])
        
        # Remove prefixes
        for prefix in patterns['prefixes']:
            if term.lower().startswith(prefix.lower() + ' '):
                relaxed = term[len(prefix)+1:]
                relaxed = self._normalize_search_term(relaxed)
                if relaxed and relaxed not in failed_terms:
                    relaxed_terms.append(relaxed)
        
        # Remove suffixes
        for suffix in patterns['suffixes']:
            if term.lower().endswith(' ' + suffix.lower()):
                relaxed = term[:-len(suffix)-1]
                relaxed = self._normalize_search_term(relaxed)
                if relaxed and relaxed not in failed_terms:
                    relaxed_terms.append(relaxed)
        
        # 4. Simplify by reducing to key tokens
        tokens = term.split()
        if len(tokens) > 2:
            # Keep only substantive words (longer than 3 chars)
            key_tokens = [token for token in tokens if len(token) > 3]
            
            # If we have at least 2 key tokens
            if len(key_tokens) >= 2:
                # Try with just the first two key tokens
                relaxed = ' '.join(key_tokens[:2])
                if relaxed not in failed_terms:
                    relaxed_terms.append(relaxed)
                
                # Try with just the first key token
                if key_tokens[0] not in failed_terms:
                    relaxed_terms.append(key_tokens[0])
        
        # 5. Use synonyms for key words if still failing
        if len(tokens) >= 1:
            main_token = max(tokens, key=len)  # Use the longest token
            
            # Find synonyms
            synonyms = set()
            for syn in self.wordnet.synsets(main_token):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != main_token and len(synonym) > 1:
                        synonyms.add(synonym)
            
            # Add top synonyms as alternatives
            for synonym in list(synonyms)[:2]:  # Limit to 2 synonyms
                if synonym not in failed_terms:
                    relaxed_terms.append(synonym)
        
        # 6. Use AI to suggest relaxed terms if enabled and previous methods didn't yield enough options
        if self.use_ai and len(relaxed_terms) < 3 and hasattr(self, 'ai_model') and self.ai_model is not None:
            try:
                model = genai.GenerativeModel(self.ai_model)
                
                prompt = f"""
                I need to generate more relaxed/broader search terms for a failed search.
                
                Original search term that failed: "{term}"
                Website category: {site_type}
                Previously failed terms: {failed_terms}
                
                Please generate 3 progressively more relaxed/broader search terms that might yield results.
                Ensure these terms are not in the list of previously failed terms.
                Order them from most specific (but still more general than original) to most general.
                
                Format as a simple JSON array of strings.
                """
                
                response = model.generate_content(prompt)
                
                # Parse the response
                response_text = response.text
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].strip()
                else:
                    json_str = response_text.strip()
                    
                ai_terms = json.loads(json_str)
                
                for term in ai_terms:
                    if term not in failed_terms and term not in relaxed_terms:
                        relaxed_terms.append(term)
                    
            except Exception as e:
                logger.error(f"Error generating AI relaxed search terms: {str(e)}")
        
        # Return unique relaxed terms, limited to 5
        unique_terms = []
        for term in relaxed_terms:
            normalized = self._normalize_search_term(term)
            if normalized and normalized not in unique_terms and normalized not in failed_terms:
                unique_terms.append(normalized)
        
        return unique_terms[:5]  # Limit to 5 relaxed terms

# Function to get a singleton instance
_search_term_generator_instance = None

def get_search_term_generator(use_ai: bool = True) -> SearchTermGenerator:
    """
    Get the singleton search term generator instance.
    
    Args:
        use_ai: Whether to use AI for enhancing search terms
        
    Returns:
        SearchTermGenerator instance
    """
    global _search_term_generator_instance
    if _search_term_generator_instance is None:
        _search_term_generator_instance = SearchTermGenerator(use_ai=use_ai)
    return _search_term_generator_instance