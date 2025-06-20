"""
Intelligent URL Generator with Intent Analysis Integration

This component generates intelligent URLs based on user queries and intent analysis
results from the UniversalIntentAnalyzer. It leverages semantic understanding to
create targeted, high-value URLs for efficient web scraping.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from dataclasses import dataclass
import tldextract

from config import (
    INTELLIGENT_URL_GENERATION, CONTEXTUAL_QUERY_EXPANSION,
    SEARCH_DEPTH, CRAWL4AI_MAX_PAGES
)


@dataclass
class URLScore:
    """Represents a scored URL with relevance metrics"""
    url: str
    relevance_score: float
    intent_match_score: float
    domain_reputation_score: float
    pattern_match_score: float
    confidence: float


class IntelligentURLGenerator:
    """
    Advanced URL generation component that leverages intent analysis results
    to create highly targeted URLs for efficient web scraping.
    """
    
    def __init__(self, intent_analyzer=None, config=None):
        """
        Initialize the IntelligentURLGenerator
        
        Args:
            intent_analyzer: UniversalIntentAnalyzer instance for semantic analysis
            config: Configuration object (defaults to global config)
        """
        self.intent_analyzer = intent_analyzer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Domain-specific URL templates
        self.domain_templates = self._init_domain_templates()
        
        # High-value URL patterns
        self.high_value_patterns = self._init_high_value_patterns()
        
        # Domain reputation scores (can be enhanced with external data)
        self.domain_reputation = self._init_domain_reputation()
        
        self.logger.info("IntelligentURLGenerator initialized successfully")
    
    def generate_urls(self, query: str, base_url: str = None, 
                     intent_analysis: Dict = None, max_urls: int = None) -> List[URLScore]:
        """
        Generate intelligent URLs based on comprehensive query and intent analysis
        
        Args:
            query: User search query
            base_url: Optional base URL to constrain search
            intent_analysis: Results from UniversalIntentAnalyzer
            max_urls: Maximum number of URLs to generate
            
        Returns:
            List of scored URLs ordered by relevance
        """
        if not INTELLIGENT_URL_GENERATION:
            self.logger.info("Intelligent URL generation disabled, falling back to basic generation")
            return self._generate_basic_urls(query, base_url)
        
        max_urls = max_urls or CRAWL4AI_MAX_PAGES
        self.logger.info(f"Generating intelligent URLs for query: '{query}'")
        
        # Get intent analysis if not provided
        if intent_analysis is None and self.intent_analyzer:
            intent_analysis = self.intent_analyzer.analyze_intent(query)
        
        # Expand search terms using intent analysis
        expanded_terms = self.expand_search_terms(query, intent_analysis)
        
        # Generate URLs using multiple strategies
        url_candidates = []
        
        # 1. Direct hit URLs (high-confidence specific patterns)
        direct_hits = self._generate_direct_hit_urls(query, intent_analysis, base_url)
        url_candidates.extend(direct_hits)
        
        # 2. Template-based URLs for known domains
        template_urls = self._generate_template_urls(expanded_terms, intent_analysis, base_url)
        url_candidates.extend(template_urls)
        
        # 3. Content-focused URLs (educational, news, reference sites)
        content_urls = self._generate_search_urls(expanded_terms, intent_analysis, base_url)
        url_candidates.extend(content_urls)
        
        # 4. Domain-specific navigation URLs
        nav_urls = self._generate_navigation_urls(query, intent_analysis, base_url)
        url_candidates.extend(nav_urls)
        
        # Validate, score, and rank all candidates
        scored_urls = []
        for url in url_candidates:
            score = self.validate_and_score_url(url, intent_analysis or {})
            if score.confidence > 0.1:  # Filter out very low confidence URLs
                scored_urls.append(score)
        
        # Remove duplicates and sort by relevance
        unique_urls = self._deduplicate_urls(scored_urls)
        sorted_urls = sorted(unique_urls, key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        final_urls = sorted_urls[:max_urls]
        
        self.logger.info(f"Generated {len(final_urls)} intelligent URLs with scores: "
                        f"{[f'{url.url[:50]}... ({url.relevance_score:.3f})' for url in final_urls[:3]]}")
        
        return final_urls
    
    def expand_search_terms(self, query: str, intent_analysis: Dict = None) -> List[str]:
        """
        Expand query using synonyms and related terms from intent analysis
        
        Args:
            query: Original search query
            intent_analysis: Intent analysis results
            
        Returns:
            List of expanded search terms
        """
        expanded_terms = [query]
        
        if intent_analysis and CONTEXTUAL_QUERY_EXPANSION:
            # Use results from UniversalIntentAnalyzer.expand_query_contextually
            if self.intent_analyzer:
                try:
                    contextual_expansions = self.intent_analyzer.expand_query_contextually(
                        query, intent_analysis
                    )
                    expanded_terms.extend(contextual_expansions)
                except Exception as e:
                    self.logger.warning(f"Failed to get contextual expansions: {e}")
            
            # Add entity-based expansions
            entities = intent_analysis.get('entities', [])
            for entity in entities:
                if entity.get('text') and entity['text'].lower() not in query.lower():
                    expanded_terms.append(entity['text'])
            
            # Add intent-specific terms
            intent_type = intent_analysis.get('intent_type', '')
            if intent_type in self._get_intent_expansion_terms():
                expanded_terms.extend(self._get_intent_expansion_terms()[intent_type])
        
        # Fallback to basic expansion if no intent analysis
        if len(expanded_terms) == 1:
            expanded_terms.extend(self._basic_term_expansion(query))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        self.logger.debug(f"Expanded '{query}' to {len(unique_terms)} terms: {unique_terms[:5]}")
        return unique_terms
    
    def validate_and_score_url(self, url: str, intent: Dict) -> URLScore:
        """
        Return relevance score 0-1 for URL based on pattern matching and domain reputation
        
        Args:
            url: URL to validate and score
            intent: Intent analysis results
            
        Returns:
            URLScore object with detailed scoring
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            query_params = parse_qs(parsed.query)
            
            # Initialize scores
            intent_match_score = 0.0
            domain_reputation_score = self._get_domain_reputation_score(domain)
            pattern_match_score = 0.0
            
            # Intent matching score
            intent_match_score = self._calculate_intent_match_score(url, intent)
            
            # Pattern matching score
            pattern_match_score = self._calculate_pattern_match_score(url, intent)
            
            # Calculate overall relevance score
            relevance_score = (
                intent_match_score * 0.4 +
                domain_reputation_score * 0.3 +
                pattern_match_score * 0.3
            )
            
            # Calculate confidence based on various factors
            confidence = self._calculate_confidence(
                url, intent, relevance_score, domain_reputation_score
            )
            
            return URLScore(
                url=url,
                relevance_score=relevance_score,
                intent_match_score=intent_match_score,
                domain_reputation_score=domain_reputation_score,
                pattern_match_score=pattern_match_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.warning(f"Error scoring URL {url}: {e}")
            return URLScore(url=url, relevance_score=0.0, intent_match_score=0.0,
                          domain_reputation_score=0.0, pattern_match_score=0.0, confidence=0.0)
    
    def _generate_direct_hit_urls(self, query: str, intent_analysis: Dict, base_url: str = None) -> List[str]:
        """Generate high-confidence, specific URL patterns - avoid search engines"""
        direct_urls = []
        
        if not intent_analysis:
            return direct_urls
        
        intent_type = intent_analysis.get('intent_type', '')
        entities = intent_analysis.get('entities', [])
        target_item = intent_analysis.get('target_item', '')
        keywords = intent_analysis.get('keywords', [])
        
        # Check for programming tutorials specifically
        programming_keywords = ['python', 'programming', 'tutorial', 'tutorials', 'learn', 'guide']
        is_programming_query = any(keyword.lower() in query.lower() for keyword in programming_keywords)
        
        if is_programming_query and ('python' in query.lower() or 'tutorial' in query.lower()):
            # Target legitimate educational websites directly
            programming_sites = [
                "https://docs.python.org/3/tutorial/",  # Official Python tutorial
                "https://realpython.com/tutorials/",     # Real Python tutorials
                "https://www.w3schools.com/python/",     # W3Schools Python
                "https://www.codecademy.com/learn/learn-python-3",  # Codecademy
                "https://www.freecodecamp.org/learn/scientific-computing-with-python/",  # FreeCodeCamp
                "https://www.tutorialspoint.com/python/index.htm",  # TutorialsPoint
                "https://www.geeksforgeeks.org/python-programming-language/",  # GeeksforGeeks
                "https://www.edx.org/learn/python",      # edX Python courses
                "https://www.coursera.org/courses?query=python%20programming",  # Coursera
                "https://www.udemy.com/topic/python/",   # Udemy Python courses
            ]
            direct_urls.extend(programming_sites)
        
        # Product search direct hits - avoid search engines
        elif intent_type == 'product' and entities:
            product_names = [e['text'] for e in entities if e.get('label') == 'PRODUCT']
            for product in product_names:
                # Direct e-commerce sites
                amazon_url = f"https://www.amazon.com/s?k={product.replace(' ', '+')}"
                direct_urls.append(amazon_url)
                
                ebay_url = f"https://www.ebay.com/sch/i.html?_nkw={product.replace(' ', '+')}"
                direct_urls.append(ebay_url)
        
        # Restaurant search direct hits
        elif intent_type == 'restaurant' and entities:
            locations = [e['text'] for e in entities if e.get('label') in ['GPE', 'LOCATION']]
            for location in locations:
                # Direct restaurant discovery sites
                yelp_url = f"https://www.yelp.com/search?find_desc=restaurants&find_loc={location.replace(' ', '+')}"
                direct_urls.append(yelp_url)
                
                ta_url = f"https://www.tripadvisor.com/Restaurants-{location.replace(' ', '_')}"
                direct_urls.append(ta_url)
        
        # News search direct hits - avoid search engines
        elif intent_type == 'news' and entities:
            topics = [e['text'] for e in entities if e.get('label') in ['PERSON', 'ORG', 'EVENT']]
            # Direct news sites instead of search engines
            news_sites = [
                "https://www.reuters.com",
                "https://www.bbc.com/news",
                "https://www.cnn.com",
                "https://www.npr.org",
                "https://www.apnews.com"
            ]
            direct_urls.extend(news_sites)
        
        # Real estate direct hits
        elif intent_type == 'property' and entities:
            locations = [e['text'] for e in entities if e.get('label') in ['GPE', 'LOCATION']]
            for location in locations:
                zillow_url = f"https://www.zillow.com/homes/{location.replace(' ', '-')}_rb/"
                direct_urls.append(zillow_url)
                
                realtor_url = f"https://www.realtor.com/realestateandhomes-search/{location.replace(' ', '-')}"
                direct_urls.append(realtor_url)
        
        self.logger.debug(f"Generated {len(direct_urls)} direct hit URLs for intent: {intent_type}")
        return direct_urls
    
    def _generate_template_urls(self, expanded_terms: List[str], intent_analysis: Dict, base_url: str = None) -> List[str]:
        """Generate URLs using domain-specific templates"""
        template_urls = []
        
        intent_type = intent_analysis.get('intent_type', '') if intent_analysis else ''
        
        # Get relevant templates for the intent type
        relevant_templates = self.domain_templates.get(intent_type, {})
        
        for domain, templates in relevant_templates.items():
            for template in templates:
                for term in expanded_terms[:5]:  # Limit to avoid too many URLs
                    try:
                        formatted_url = template.format(query=term.replace(' ', '+'))
                        template_urls.append(formatted_url)
                    except (KeyError, ValueError) as e:
                        self.logger.debug(f"Failed to format template {template}: {e}")
        
        # If base_url is provided, prioritize templates from the same domain
        if base_url:
            base_domain = urlparse(base_url).netloc
            domain_specific_urls = [url for url in template_urls if base_domain in url]
            if domain_specific_urls:
                template_urls = domain_specific_urls + template_urls
        
        self.logger.debug(f"Generated {len(template_urls)} template URLs")
        return template_urls
    
    def _generate_search_urls(self, expanded_terms: List[str], intent_analysis: Dict, base_url: str = None) -> List[str]:
        """Generate content-focused URLs instead of search engine URLs"""
        content_urls = []
        
        # Limit terms to avoid too many URLs
        primary_terms = expanded_terms[:3]
        
        for term in primary_terms:
            encoded_term = term.replace(' ', '+')
            encoded_term_dash = term.replace(' ', '-')
            
            # Educational content sites
            if any(keyword in term.lower() for keyword in ['python', 'programming', 'tutorial', 'learn', 'code']):
                content_urls.extend([
                    f"https://www.tutorialspoint.com/python/{encoded_term_dash}",
                    f"https://realpython.com/search/?q={encoded_term}",
                    f"https://www.w3schools.com/python/{encoded_term_dash}.asp",
                    f"https://docs.python.org/3/search.html?q={encoded_term}",
                ])
            
            # Academic and reference sites
            elif any(keyword in term.lower() for keyword in ['research', 'study', 'academic', 'science']):
                content_urls.extend([
                    f"https://www.ncbi.nlm.nih.gov/pubmed/?term={encoded_term}",
                    f"https://www.jstor.org/action/doBasicSearch?Query={encoded_term}",
                    f"https://arxiv.org/search/?query={encoded_term}",
                    f"https://www.researchgate.net/search?q={encoded_term}",
                ])
            
            # News and current events
            elif any(keyword in term.lower() for keyword in ['news', 'current', 'latest', 'breaking']):
                content_urls.extend([
                    f"https://www.reuters.com/search/news?blob={encoded_term}",
                    f"https://www.bbc.com/search?q={encoded_term}",
                    f"https://www.npr.org/search?query={encoded_term}",
                ])
            
            # General content sites
            else:
                content_urls.extend([
                    f"https://en.wikipedia.org/wiki/Special:Search?search={encoded_term}",
                    f"https://www.reddit.com/search/?q={encoded_term}",
                    f"https://stackoverflow.com/search?q={encoded_term}",
                ])
        
        self.logger.debug(f"Generated {len(content_urls)} content-focused URLs")
        return content_urls
    
    def _generate_navigation_urls(self, query: str, intent_analysis: Dict, base_url: str = None) -> List[str]:
        """Generate domain-specific navigation URLs"""
        nav_urls = []
        
        if not base_url:
            return nav_urls
        
        try:
            parsed = urlparse(base_url)
            domain = parsed.netloc.lower()
            
            # Generate common navigation patterns
            base_patterns = [
                f"{parsed.scheme}://{domain}/search?q={query.replace(' ', '+')}",
                f"{parsed.scheme}://{domain}/find?query={query.replace(' ', '+')}",
                f"{parsed.scheme}://{domain}/s?k={query.replace(' ', '+')}",
                f"{parsed.scheme}://{domain}/products?search={query.replace(' ', '+')}",
                f"{parsed.scheme}://{domain}/category/{query.replace(' ', '-')}",
            ]
            
            nav_urls.extend(base_patterns)
            
        except Exception as e:
            self.logger.debug(f"Failed to generate navigation URLs for {base_url}: {e}")
        
        self.logger.debug(f"Generated {len(nav_urls)} navigation URLs")
        return nav_urls
    
    def _generate_basic_urls(self, query: str, base_url: str = None) -> List[URLScore]:
        """Fallback basic URL generation when intelligent generation is disabled"""
        basic_urls = []
        encoded_query = query.replace(' ', '+')
        
        # Instead of search engines, use educational and content sites directly
        content_sites = []
        
        # Check for programming/educational content
        if any(keyword in query.lower() for keyword in ['python', 'programming', 'tutorial', 'learn', 'code']):
            content_sites = [
                "https://docs.python.org/3/tutorial/",
                "https://realpython.com/tutorials/",
                "https://www.w3schools.com/python/",
                "https://www.tutorialspoint.com/python/index.htm",
                "https://www.geeksforgeeks.org/python-programming-language/"
            ]
        # Check for news content
        elif any(keyword in query.lower() for keyword in ['news', 'latest', 'current', 'today']):
            content_sites = [
                "https://www.reuters.com",
                "https://www.bbc.com/news",
                "https://www.npr.org",
                "https://www.apnews.com"
            ]
        # Check for shopping content
        elif any(keyword in query.lower() for keyword in ['buy', 'purchase', 'price', 'shop']):
            content_sites = [
                f"https://www.amazon.com/s?k={encoded_query}",
                f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}"
            ]
        # Default to educational and reference sites
        else:
            content_sites = [
                "https://en.wikipedia.org/wiki/Main_Page",
                "https://www.britannica.com/",
                "https://www.reddit.com/",
                "https://stackoverflow.com/",
                "https://www.youtube.com/"
            ]
        
        # Convert to URLScore objects with basic scoring
        scored_urls = []
        for url in content_sites:
            scored_urls.append(URLScore(
                url=url,
                relevance_score=0.6,
                intent_match_score=0.5,
                domain_reputation_score=0.8,
                pattern_match_score=0.4,
                confidence=0.5
            ))
        
        return scored_urls
    
    def _calculate_intent_match_score(self, url: str, intent: Dict) -> float:
        """Calculate how well URL matches the intent"""
        if not intent:
            return 0.5
        
        score = 0.0
        url_lower = url.lower()
        
        # Check intent type relevance
        intent_type = intent.get('intent_type', '')
        intent_keywords = {
            'product': ['shop', 'buy', 'product', 'store', 'amazon', 'ebay'],
            'restaurant': ['restaurant', 'food', 'dining', 'yelp', 'menu'],
            'news': ['news', 'article', 'blog', 'press', 'media'],
            'property': ['real', 'estate', 'property', 'home', 'zillow'],
            'job': ['job', 'career', 'work', 'employment', 'linkedin']
        }
        
        if intent_type in intent_keywords:
            matches = sum(1 for keyword in intent_keywords[intent_type] if keyword in url_lower)
            score += matches * 0.2
        
        # Check entity relevance
        entities = intent.get('entities', [])
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            if entity_text and entity_text in url_lower:
                score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_pattern_match_score(self, url: str, intent: Dict) -> float:
        """Calculate score based on high-value URL patterns"""
        score = 0.0
        
        for pattern_group in self.high_value_patterns.values():
            for pattern in pattern_group:
                if re.search(pattern, url, re.IGNORECASE):
                    score += 0.2
        
        return min(score, 1.0)
    
    def _get_domain_reputation_score(self, domain: str) -> float:
        """Get reputation score for domain"""
        return self.domain_reputation.get(domain, 0.5)
    
    def _calculate_confidence(self, url: str, intent: Dict, relevance_score: float, domain_score: float) -> float:
        """Calculate overall confidence in URL"""
        base_confidence = relevance_score
        
        # Boost confidence for well-known domains
        if domain_score > 0.8:
            base_confidence += 0.1
        
        # Boost confidence for secure URLs
        if url.startswith('https://'):
            base_confidence += 0.05
        
        # Reduce confidence for very long URLs (might be spam)
        if len(url) > 200:
            base_confidence -= 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _deduplicate_urls(self, scored_urls: List[URLScore]) -> List[URLScore]:
        """Remove duplicate URLs, keeping the highest scored version"""
        seen_urls = {}
        
        for scored_url in scored_urls:
            url = scored_url.url
            if url not in seen_urls or scored_url.relevance_score > seen_urls[url].relevance_score:
                seen_urls[url] = scored_url
        
        return list(seen_urls.values())
    
    def _init_domain_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize domain-specific URL templates"""
        return {
            'product': {
                'amazon.com': [
                    'https://www.amazon.com/s?k={query}',
                    'https://www.amazon.com/s?k={query}&ref=nb_sb_noss'
                ],
                'ebay.com': [
                    'https://www.ebay.com/sch/i.html?_nkw={query}',
                    'https://www.ebay.com/sch/i.html?_nkw={query}&_sacat=0'
                ],
                'walmart.com': [
                    'https://www.walmart.com/search/?query={query}'
                ]
            },
            'restaurant': {
                'yelp.com': [
                    'https://www.yelp.com/search?find_desc={query}',
                    'https://www.yelp.com/search?find_desc=restaurants&find_loc={query}'
                ],
                'tripadvisor.com': [
                    'https://www.tripadvisor.com/Search?q={query}&searchSessionId=000'
                ]
            },
            'news': {
                'reuters.com': [
                    'https://www.reuters.com/news/archive'
                ],
                'bbc.com': [
                    'https://www.bbc.com/news'
                ],
                'npr.org': [
                    'https://www.npr.org/sections/news/'
                ],
                'apnews.com': [
                    'https://apnews.com/'
                ],
                'reddit.com': [
                    'https://www.reddit.com/search/?q={query}&type=link'
                ]
            },
            'tutorial': {
                'docs.python.org': [
                    'https://docs.python.org/3/tutorial/'
                ],
                'realpython.com': [
                    'https://realpython.com/tutorials/',
                    'https://realpython.com/search?q={query}'
                ],
                'w3schools.com': [
                    'https://www.w3schools.com/python/',
                    'https://www.w3schools.com/python/python_intro.asp'
                ],
                'geeksforgeeks.org': [
                    'https://www.geeksforgeeks.org/python-programming-language/',
                    'https://www.geeksforgeeks.org/{query}/'
                ],
                'tutorialspoint.com': [
                    'https://www.tutorialspoint.com/python/index.htm'
                ],
                'codecademy.com': [
                    'https://www.codecademy.com/learn/learn-python-3'
                ],
                'freecodecamp.org': [
                    'https://www.freecodecamp.org/learn/scientific-computing-with-python/'
                ]
            },
            'property': {
                'zillow.com': [
                    'https://www.zillow.com/homes/{query}_rb/'
                ],
                'realtor.com': [
                    'https://www.realtor.com/realestateandhomes-search/{query}'
                ]
            }
        }
    
    def _init_high_value_patterns(self) -> Dict[str, List[str]]:
        """Initialize high-value URL patterns"""
        return {
            'product_pages': [
                r'/product/',
                r'/item/',
                r'/p/',
                r'/dp/',
                r'products?.*id='
            ],
            'search_results': [
                r'/search\?',
                r'/s\?',
                r'/find\?',
                r'query='
            ],
            'category_pages': [
                r'/category/',
                r'/categories/',
                r'/c/',
                r'/dept/'
            ],
            'location_pages': [
                r'/location/',
                r'/city/',
                r'/area/',
                r'/region/'
            ]
        }
    
    def _init_domain_reputation(self) -> Dict[str, float]:
        """Initialize domain reputation scores"""
        return {
            # High reputation domains
            'amazon.com': 0.95,
            'google.com': 0.95,
            'wikipedia.org': 0.95,
            'github.com': 0.9,
            'stackoverflow.com': 0.9,
            'reddit.com': 0.85,
            'yelp.com': 0.85,
            'tripadvisor.com': 0.85,
            'zillow.com': 0.85,
            'realtor.com': 0.85,
            'ebay.com': 0.8,
            'walmart.com': 0.8,
            'target.com': 0.8,
            'bestbuy.com': 0.8,
            'linkedin.com': 0.8,
            'indeed.com': 0.8,
            # Medium reputation domains
            'bing.com': 0.7,
            'yahoo.com': 0.7,
            'craigslist.org': 0.6,
            # Default for unknown domains
            'default': 0.5
        }
    
    def _get_intent_expansion_terms(self) -> Dict[str, List[str]]:
        """Get intent-specific expansion terms"""
        return {
            'product': ['buy', 'purchase', 'shop', 'store', 'price', 'review'],
            'restaurant': ['food', 'dining', 'menu', 'cuisine', 'reservation'],
            'news': ['latest', 'breaking', 'update', 'report', 'article'],
            'property': ['home', 'house', 'apartment', 'real estate', 'for sale'],
            'job': ['career', 'employment', 'position', 'hiring', 'work']
        }
    
    def _basic_term_expansion(self, query: str) -> List[str]:
        """Basic term expansion without intent analysis"""
        expansions = []
        
        # Add plural/singular forms
        if query.endswith('s') and len(query) > 3:
            expansions.append(query[:-1])  # Remove 's'
        else:
            expansions.append(query + 's')  # Add 's'
        
        # Add common synonyms
        synonyms = {
            'best': ['top', 'great', 'excellent'],
            'cheap': ['affordable', 'budget', 'low cost'],
            'restaurant': ['dining', 'food', 'eatery'],
            'house': ['home', 'property', 'residence'],
            'job': ['career', 'position', 'employment']
        }
        
        for word, syns in synonyms.items():
            if word in query.lower():
                for syn in syns:
                    expansions.append(query.lower().replace(word, syn))
        
        return expansions[:5]  # Limit to avoid too many terms
