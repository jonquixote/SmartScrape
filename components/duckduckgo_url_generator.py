"""
DuckDuckGo URL Generator

A URL generator that uses DuckDuckGo search to discover relevant URLs,
then ranks and filters them for intelligent web scraping. This replaces
AI/LLM-based URL generation with a more reliable search-based approach.

Compatible with the existing IntelligentURLGenerator interface.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import the existing URLScore class for compatibility
from components.intelligent_url_generator import URLScore
from config import CRAWL4AI_MAX_PAGES

try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False


@dataclass
class SearchResult:
    """Represents a search result from DuckDuckGo"""
    title: str
    url: str
    body: str
    ranking: int = 0


class DuckDuckGoURLGenerator:
    """
    URL generator that uses DuckDuckGo search to discover and rank URLs.
    
    This class provides a drop-in replacement for IntelligentURLGenerator
    that uses real search results instead of AI-generated URLs.
    """
    
    def __init__(self, intent_analyzer=None, config=None):
        """
        Initialize the DuckDuckGoURLGenerator
        
        Args:
            intent_analyzer: UniversalIntentAnalyzer instance (optional, for compatibility)
            config: Configuration object (defaults to global config)
        """
        self.intent_analyzer = intent_analyzer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if not DUCKDUCKGO_AVAILABLE:
            raise ImportError("duckduckgo-search package is required. Install with: pip install duckduckgo-search")
        
        # Initialize DuckDuckGo search client
        self.ddgs = DDGS()
        
        # Initialize HTTP session with retries for URL validation
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Domain reputation scores (can be enhanced with external data)
        self.domain_reputation = self._init_domain_reputation()
        
        # Search result filters
        self.excluded_domains = {
            'google.com', 'bing.com', 'yahoo.com', 'search.yahoo.com',
            'duckduckgo.com', 'baidu.com', 'yandex.com', 'ask.com'
        }
        
        self.logger.info("DuckDuckGoURLGenerator initialized successfully")
    
    def generate_urls(self, query: str, base_url: str = None, 
                     intent_analysis: Dict = None, max_urls: int = None) -> List[URLScore]:
        """
        Generate URLs using DuckDuckGo search and intelligent ranking
        
        Args:
            query: User search query
            base_url: Optional base URL to constrain search (not used in DDG search)
            intent_analysis: Results from UniversalIntentAnalyzer (optional)
            max_urls: Maximum number of URLs to generate
            
        Returns:
            List of scored URLs ordered by relevance
        """
        max_urls = max_urls or CRAWL4AI_MAX_PAGES
        self.logger.info(f"Generating URLs using DuckDuckGo search for query: '{query}'")
        
        try:
            # Get search results from DuckDuckGo
            search_results = self._search_duckduckgo(query, max_results=max_urls * 2)
            
            # Filter out search engines and unwanted domains
            filtered_results = self._filter_search_results(search_results)
            
            # Score and rank the URLs
            scored_urls = []
            for i, result in enumerate(filtered_results):
                score = self._score_search_result(result, query, intent_analysis or {}, ranking=i)
                if score.confidence > 0.1:  # Filter out very low confidence URLs
                    scored_urls.append(score)
            
            # Sort by relevance score and limit results
            sorted_urls = sorted(scored_urls, key=lambda x: x.relevance_score, reverse=True)
            final_urls = sorted_urls[:max_urls]
            
            self.logger.info(f"Generated {len(final_urls)} URLs from DuckDuckGo search")
            return final_urls
            
        except Exception as e:
            self.logger.error(f"Error generating URLs with DuckDuckGo: {e}")
            return []
    
    def expand_search_terms(self, query: str, intent_analysis: Dict = None) -> List[str]:
        """
        Expand query using related search suggestions from DuckDuckGo
        
        Args:
            query: Original search query
            intent_analysis: Intent analysis results (optional)
            
        Returns:
            List of expanded search terms
        """
        expanded_terms = [query]
        
        try:
            # Get search suggestions from DuckDuckGo
            suggestions = self.ddgs.suggestions(query, region='wt-wt')
            
            # Add suggestions as expanded terms
            for suggestion in suggestions[:5]:  # Limit to 5 suggestions
                if suggestion['phrase'] and suggestion['phrase'] not in expanded_terms:
                    expanded_terms.append(suggestion['phrase'])
                    
        except Exception as e:
            self.logger.warning(f"Failed to get DuckDuckGo suggestions: {e}")
        
        # If intent analysis is provided, add entity-based expansions
        if intent_analysis:
            entities = intent_analysis.get('entities', [])
            for entity in entities:
                if entity.get('text') and entity['text'].lower() not in query.lower():
                    expanded_terms.append(entity['text'])
        
        self.logger.debug(f"Expanded '{query}' to {len(expanded_terms)} terms")
        return expanded_terms
    
    def validate_and_score_url(self, url: str, intent: Dict) -> URLScore:
        """
        Validate and score a URL for relevance
        
        Args:
            url: URL to validate and score
            intent: Intent analysis results
            
        Returns:
            URLScore object with detailed scoring
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Basic validation
            if not domain or domain in self.excluded_domains:
                return URLScore(url, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Domain reputation score
            domain_score = self._get_domain_reputation_score(domain)
            
            # Intent matching score (simplified for compatibility)
            intent_score = 0.5  # Default moderate score
            
            # Pattern matching score (based on URL structure)
            pattern_score = self._calculate_pattern_score(url)
            
            # Overall relevance score
            relevance_score = (domain_score * 0.4 + intent_score * 0.3 + pattern_score * 0.3)
            
            # Confidence based on overall quality
            confidence = min(1.0, relevance_score + 0.1)
            
            return URLScore(
                url=url,
                relevance_score=relevance_score,
                intent_match_score=intent_score,
                domain_reputation_score=domain_score,
                pattern_match_score=pattern_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.warning(f"Error scoring URL {url}: {e}")
            return URLScore(url, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _search_duckduckgo(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """
        Perform DuckDuckGo search and return results
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        try:
            # Search using DuckDuckGo - ensure we get REAL URLs
            search_results = self.ddgs.text(
                keywords=query,
                region='wt-wt',
                safesearch='moderate',
                timelimit=None,
                max_results=max_results
            )
            
            self.logger.info(f"DuckDuckGo search returned raw results")
            
            # Debug: Track fake vs real URLs
            total_results = 0
            fake_filtered = 0
            
            for i, result in enumerate(search_results):
                total_results += 1
                url = result.get('href', '')
                
                # Log each URL for debugging
                self.logger.info(f"Processing result {i}: {url}")
                
                # Validate URL is real and not a fake DuckDuckGo URL
                if self._is_valid_real_url(url):
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        url=url,
                        body=result.get('body', ''),
                        ranking=i
                    )
                    results.append(search_result)
                    self.logger.info(f"✅ Added real URL #{len(results)}: {url}")
                else:
                    fake_filtered += 1
                    self.logger.warning(f"❌ Filtered out fake URL #{fake_filtered}: {url}")
            
            self.logger.info(f"DuckDuckGo filtering summary: {len(results)} real URLs, {fake_filtered} fake URLs filtered from {total_results} total")
                
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
        
        self.logger.info(f"Returning {len(results)} valid URLs")
        return results
    
    def _is_valid_real_url(self, url: str) -> bool:
        """Validate that URL is real and not fake"""
        if not url or not isinstance(url, str):
            return False
            
        # Filter out known fake patterns
        fake_patterns = [
            'lite.duckduckgo.com',
            'html.duckduckgo.com',
            'duckduckgo.com/lite',
            'duckduckgo.com/html',
            'javascript:',
            'mailto:',
            'tel:',
            '#',
        ]
        
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in fake_patterns):
            self.logger.warning(f"Filtered fake URL pattern: {url}")
            return False
            
        # Must be a proper HTTP URL
        if not url.startswith(('http://', 'https://')):
            self.logger.warning(f"Filtered non-HTTP URL: {url}")
            return False
            
        return True
    
    def _filter_search_results(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter out unwanted domains and low-quality results
        
        Args:
            search_results: List of search results
            
        Returns:
            Filtered list of search results
        """
        filtered = []
        
        for result in search_results:
            try:
                parsed = urlparse(result.url)
                domain = parsed.netloc.lower()
                
                # Skip excluded domains
                if any(excluded in domain for excluded in self.excluded_domains):
                    continue
                
                # Skip if missing essential data
                if not result.title or not result.url:
                    continue
                
                # Skip very short URLs (likely redirects or low quality)
                if len(result.url) < 20:
                    continue
                
                filtered.append(result)
                
            except Exception as e:
                self.logger.debug(f"Error filtering result {result.url}: {e}")
                continue
        
        return filtered
    
    def _score_search_result(self, result: SearchResult, query: str, 
                           intent_analysis: Dict, ranking: int) -> URLScore:
        """
        Score a search result based on multiple factors
        
        Args:
            result: SearchResult object
            query: Original search query
            intent_analysis: Intent analysis results
            ranking: Search result ranking (0-based)
            
        Returns:
            URLScore object
        """
        try:
            parsed = urlparse(result.url)
            domain = parsed.netloc.lower()
            
            # Base score from DuckDuckGo ranking (higher ranking = lower score)
            ranking_score = max(0.1, 1.0 - (ranking / 20.0))
            
            # Domain reputation score
            domain_score = self._get_domain_reputation_score(domain)
            
            # Content relevance score (based on title and body)
            content_score = self._calculate_content_relevance(result, query)
            
            # Intent matching score
            intent_score = self._calculate_intent_match(result, intent_analysis)
            
            # Pattern matching score
            pattern_score = self._calculate_pattern_score(result.url)
            
            # Weighted relevance score
            relevance_score = (
                ranking_score * 0.3 +
                domain_score * 0.2 +
                content_score * 0.3 +
                intent_score * 0.1 +
                pattern_score * 0.1
            )
            
            # Confidence based on multiple factors
            confidence = min(1.0, (relevance_score + content_score) / 2.0)
            
            return URLScore(
                url=result.url,
                relevance_score=relevance_score,
                intent_match_score=intent_score,
                domain_reputation_score=domain_score,
                pattern_match_score=pattern_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.warning(f"Error scoring search result: {e}")
            return URLScore(result.url, 0.1, 0.1, 0.1, 0.1, 0.1)
    
    def _calculate_content_relevance(self, result: SearchResult, query: str) -> float:
        """
        Calculate content relevance score based on title and body text
        
        Args:
            result: SearchResult object
            query: Search query
            
        Returns:
            Relevance score between 0 and 1
        """
        query_words = set(query.lower().split())
        title_words = set(result.title.lower().split())
        body_words = set(result.body.lower().split())
        
        # Calculate word overlap
        title_overlap = len(query_words & title_words) / max(len(query_words), 1)
        body_overlap = len(query_words & body_words) / max(len(query_words), 1)
        
        # Weight title more heavily than body
        content_score = title_overlap * 0.7 + body_overlap * 0.3
        
        return min(1.0, content_score)
    
    def _calculate_intent_match(self, result: SearchResult, intent_analysis: Dict) -> float:
        """
        Calculate how well the result matches the user's intent
        
        Args:
            result: SearchResult object
            intent_analysis: Intent analysis results
            
        Returns:
            Intent match score between 0 and 1
        """
        if not intent_analysis:
            return 0.5  # Default moderate score
        
        score = 0.5
        
        # Check for entity matches
        entities = intent_analysis.get('entities', [])
        for entity in entities:
            entity_text = entity.get('text', '').lower()
            if entity_text in result.title.lower() or entity_text in result.body.lower():
                score += 0.1
        
        # Check for semantic keyword matches
        semantic_keywords = intent_analysis.get('semantic_keywords', [])
        for keyword in semantic_keywords:
            if keyword.lower() in result.title.lower() or keyword.lower() in result.body.lower():
                score += 0.05
        
        return min(1.0, score)
    
    def _calculate_pattern_score(self, url: str) -> float:
        """
        Calculate URL pattern score based on structure quality
        
        Args:
            url: URL to analyze
            
        Returns:
            Pattern score between 0 and 1
        """
        try:
            parsed = urlparse(url)
            score = 0.5  # Base score
            
            # Bonus for HTTPS
            if parsed.scheme == 'https':
                score += 0.1
            
            # Bonus for clean path structure
            if parsed.path and not parsed.path.endswith(('.php', '.asp', '.jsp')):
                score += 0.1
            
            # Penalty for excessive query parameters
            if parsed.query and len(parse_qs(parsed.query)) > 5:
                score -= 0.1
            
            # Bonus for reasonable path depth
            path_depth = len([p for p in parsed.path.split('/') if p])
            if 1 <= path_depth <= 4:
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.3  # Default score for malformed URLs
    
    def _get_domain_reputation_score(self, domain: str) -> float:
        """
        Get domain reputation score
        
        Args:
            domain: Domain name
            
        Returns:
            Reputation score between 0 and 1
        """
        return self.domain_reputation.get(domain, 0.5)  # Default moderate score
    
    def _init_domain_reputation(self) -> Dict[str, float]:
        """
        Initialize domain reputation scores
        
        Returns:
            Dictionary mapping domains to reputation scores
        """
        return {
            # High-reputation domains
            'wikipedia.org': 0.95,
            'github.com': 0.9,
            'stackoverflow.com': 0.9,
            'medium.com': 0.8,
            'reddit.com': 0.75,
            'youtube.com': 0.8,
            'docs.python.org': 0.95,
            'developer.mozilla.org': 0.9,
            'w3schools.com': 0.8,
            'geeksforgeeks.org': 0.8,
            'tutorialspoint.com': 0.75,
            'realpython.com': 0.85,
            'towardsdatascience.com': 0.8,
            'machinelearningmastery.com': 0.8,
            'kaggle.com': 0.85,
            'coursera.org': 0.8,
            'edx.org': 0.8,
            'udemy.com': 0.75,
            'arxiv.org': 0.9,
            'scholar.google.com': 0.85,
            
            # News and media
            'bbc.com': 0.85,
            'cnn.com': 0.8,
            'reuters.com': 0.85,
            'techcrunch.com': 0.8,
            'wired.com': 0.8,
            'arstechnica.com': 0.8,
            'theverge.com': 0.8,
            
            # Government and academic
            '.gov': 0.9,
            '.edu': 0.85,
            '.ac.uk': 0.85,
            
            # Low-reputation patterns
            'blogspot.com': 0.4,
            'wordpress.com': 0.5,
            'tumblr.com': 0.4,
        }


# Compatibility function for easy integration
def get_duckduckgo_url_generator(intent_analyzer=None, config=None) -> DuckDuckGoURLGenerator:
    """
    Factory function to create a DuckDuckGoURLGenerator instance
    
    Args:
        intent_analyzer: UniversalIntentAnalyzer instance
        config: Configuration object
        
    Returns:
        DuckDuckGoURLGenerator instance
    """
    return DuckDuckGoURLGenerator(intent_analyzer, config)

# Test function to verify real URLs
def test_real_duckduckgo():
    """Test that we get real URLs"""
    generator = DuckDuckGoURLGenerator()
    results = generator.generate_urls("Tesla news", max_urls=5)
    
    print("=== REAL DUCKDUCKGO TEST ===")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.url}")
        print(f"   Title: {getattr(result, 'title', 'N/A')}")
        print(f"   Valid: {generator._is_valid_real_url(result.url)}")
        print(f"   Type: {type(result)}")
        print()
        
if __name__ == "__main__":
    test_real_duckduckgo()
