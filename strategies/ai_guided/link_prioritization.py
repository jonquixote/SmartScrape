"""
Link Prioritization Module

Provides functionality to score and prioritize links based on their relevance
to the user's intent and the current crawling strategy.
"""
from typing import Dict, List, Any, Tuple, Set, Optional
from urllib.parse import urlparse, parse_qs
import re
import logging
import json

from strategies.ai_guided.ai_cache import AIResponseCache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinkPrioritizer")

class LinkPrioritizer:
    """
    Handles the scoring and prioritization of links for the AI-guided crawler.
    """
    
    def __init__(self, 
                response_cache: Optional[AIResponseCache] = None,
                crawl_intent: Dict[str, Any] = None,
                content_memory: Dict[str, Any] = None,
                page_types_found: Set[str] = None,
                performance_stats: Dict[str, Any] = None,
                url_scores: Dict[str, float] = None,
                main_domain: str = None,
                use_ai: bool = True):
        """
        Initialize the link prioritizer.
        
        Args:
            response_cache: Cache for AI API responses
            crawl_intent: Dict containing user intent information
            content_memory: Dict of page content summaries keyed by URL
            page_types_found: Set of page types discovered so far
            performance_stats: Dict tracking performance metrics
            url_scores: Dict of URL scores for tracking
            main_domain: Primary domain being crawled
            use_ai: Whether to use AI for prioritization
        """
        self.response_cache = response_cache
        self.crawl_intent = crawl_intent or {}
        self.content_memory = content_memory or {}
        self.page_types_found = page_types_found or set()
        self.performance_stats = performance_stats or {"ai_calls": 0, "cache_hits": 0}
        self.url_scores = url_scores or {}
        self.main_domain = main_domain
        self.use_ai = use_ai
        
        # Metadata for links
        self.link_metadata = {}
        
        # Content type patterns for recognizing different page types from URLs
        self.content_patterns = {
            "product": ['/product', '/item', '/detail', '/p/', 'pdp'],
            "category": ['/category', '/collection', '/c/', '/department', '/browse'],
            "listing": ['/listing', '/search', '/results', '/properties', '/find'],
            "article": ['/article', '/post', '/blog', '/news', '/story'],
            "profile": ['/profile', '/user', '/account', '/member'],
            "contact": ['/contact', '/support', '/help', '/about']
        }
        
        # Terms indicating high-value content
        self.high_value_terms = {
            "real_estate": ["property", "house", "home", "apartment", "condo", "real estate", 
                          "listing", "rent", "sale", "bedroom", "bathroom"],
            "ecommerce": ["product", "price", "shop", "buy", "order", "cart", "checkout",
                        "shipping", "discount", "offer", "sale", "item"],
            "news": ["article", "news", "story", "update", "headline", "press", "report"],
            "jobs": ["job", "career", "position", "employment", "vacancy", "hiring",
                   "salary", "apply", "resume", "cv"]
        }
        
    async def prioritize_links(self, 
                              links: List[Tuple[str, str, str]], 
                              parent_url: str,
                              current_depth: int,
                              results: List[Dict[str, Any]],
                              max_depth: int) -> List[Tuple[str, float, str]]:
        """
        Score and prioritize links for further crawling with enhanced intelligence.
        
        Args:
            links: List of tuples (URL, anchor_text, context)
            parent_url: URL of the parent page
            current_depth: Current crawl depth
            results: List of results collected so far
            max_depth: Maximum crawl depth
            
        Returns:
            List of tuples (URL, relevance_score, explanation)
        """
        if not links:
            return []
        
        # Enhanced exploration vs exploitation balancing based on current progress
        exploration_factor = self._calculate_exploration_factor(results)
        
        # For very deep levels, limit the number of links to consider
        if current_depth >= max_depth - 1:
            # At deep levels, be more selective
            links = self._select_most_promising_links(links, max_links=5)
        
        content_quality = self._assess_content_quality(results)
        
        # Check if we have existing content analysis and intent
        if self.use_ai and self.content_memory and len(links) > 0 and len(results) > 0:
            # Get prior content and intent information for context
            prior_content_info = self._get_content_context(results)
            
            # Determine navigation intention based on current progress
            navigation_intention = self._determine_navigation_intention(
                current_depth, max_depth, results, content_quality
            )
            
            # For larger link sets, use more selective filtering
            if len(links) > 20:
                # For larger sets, pre-filter using heuristics and only use detailed AI analysis
                # on the most promising candidates
                candidate_links = self._prefilter_links(links, parent_url, current_depth, results, max_links=15)
                scored_links = await self._score_links_with_content_awareness(
                    candidate_links, parent_url, prior_content_info, 
                    navigation_intention, exploration_factor, results
                )
            else:
                # For smaller sets, analyze all links
                scored_links = await self._score_links_with_content_awareness(
                    links, parent_url, prior_content_info, 
                    navigation_intention, exploration_factor, results
                )
        else:
            # Enhanced heuristic scoring for cases where we can't use AI
            scored_links = []
            for url, text, context in links:
                parsed_url = urlparse(url)
                url_features = self._extract_url_features(url, parsed_url)
                content_type, type_confidence = self._detect_content_type(url, text)
                url_score = self._calculate_heuristic_score(
                    url, text, context, url_features, 
                    content_type, current_depth, max_depth, 
                    parent_url, results
                )
                explanation = f"Heuristic score for {content_type} content"
                scored_links.append((url, url_score, explanation))
        
        # Final adjustment based on navigation strategy
        final_links = self._adjust_scores_for_strategy(scored_links, current_depth, max_depth)
        
        # Sort by score descending
        final_links.sort(key=lambda x: x[1], reverse=True)
        
        # Store scores for future reference
        for url, score, _ in final_links:
            self.url_scores[url] = score
            
        return final_links
    
    async def _score_links_with_content_awareness(self, 
                                                links: List[Tuple[str, str, str]], 
                                                parent_url: str,
                                                prior_content_info: Dict[str, Any],
                                                navigation_intention: str,
                                                exploration_factor: float,
                                                results: List[Dict[str, Any]]) -> List[Tuple[str, float, str]]:
        """
        Score links using content awareness and relevance to current navigation goals.
        
        Args:
            links: List of tuples (URL, anchor_text, context)
            parent_url: URL of parent page
            prior_content_info: Summary of content collected so far
            navigation_intention: Current navigation goal (explore/exploit/diversify)
            exploration_factor: Factor for exploration vs exploitation
            results: Current results
            
        Returns:
            List of scored links
        """
        scored_links = []
        
        for url, anchor_text, context in links:
            # Extract content-relevant features
            parsed_url = urlparse(url)
            url_features = self._extract_url_features(url, parsed_url)
            content_type, type_confidence = self._detect_content_type(url, anchor_text)
            
            # Basic relevance score from heuristic analysis
            base_score = self._calculate_relevance_score(url, anchor_text, url_features)
            
            # Analyze URL path components for content indicators
            path_analysis = self._analyze_url_path(url_features["path_segments"], content_type)
            
            # Content-aware adjustments based on what we've found so far
            adjusted_score = self._adjust_link_score(
                base_score,
                content_type,
                navigation_intention,
                exploration_factor,
                results
            )
            
            # For detail pages, check if this is likely to be a high-value target
            if content_type in ["product", "listing", "article"] and type_confidence > 0.7:
                # Give boost to detail pages that match our target content
                high_value_boost = self._calculate_high_value_content_boost(
                    url, anchor_text, context, content_type
                )
                adjusted_score += high_value_boost
                
            # Calculate keyword match score based on crawl intent
            keyword_match_score = self._calculate_keyword_match_score(
                url, anchor_text, context
            )
            
            # Combine scores with appropriate weighting
            final_score = (
                adjusted_score * 0.5 +
                path_analysis["relevance"] * 0.2 +
                keyword_match_score * 0.3
            )
            
            # Cap final score at 1.0
            final_score = min(1.0, final_score)
            
            # Build explanation for the score
            explanation = f"{content_type.capitalize()} page ({type_confidence:.2f})"
            if path_analysis["indicators"]:
                explanation += f" | Path indicators: {', '.join(path_analysis['indicators'][:2])}"
            if keyword_match_score > 0:
                explanation += f" | Keyword relevance: {keyword_match_score:.2f}"
                
            scored_links.append((url, final_score, explanation))
            
        return scored_links
    
    def _extract_url_features(self, url: str, parsed_url=None) -> Dict[str, Any]:
        """
        Extract features from a URL for relevance analysis.
        
        Args:
            url: URL to analyze
            parsed_url: Optional pre-parsed URL
            
        Returns:
            Dictionary of URL features
        """
        if parsed_url is None:
            parsed_url = urlparse(url)
            
        # Get domain
        domain = parsed_url.netloc
        
        # Check if internal to main domain
        is_internal = False
        if self.main_domain and domain == self.main_domain:
            is_internal = True
        
        # Parse path components
        path = parsed_url.path
        path_segments = [p for p in path.split('/') if p]
        
        # Extract query parameters
        query_params = {}
        if parsed_url.query:
            try:
                query_params = parse_qs(parsed_url.query)
            except:
                pass
                
        # Check for common page identifiers
        has_id = False
        id_value = None
        id_patterns = [
            r'/(\d+)$',  # ID at end of URL
            r'/i/(\d+)',  # ID format like /i/12345
            r'[\?&]id=(\d+)',  # URL param like ?id=12345
            r'/p/(\d+)'  # Product ID format
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, url)
            if match:
                has_id = True
                id_value = match.group(1)
                break
        
        # Check if URL appears to be a detail page
        detail_indicators = ['detail', 'product', 'item', 'page', 'view', 'show']
        is_detail_page = has_id or any(indicator in url.lower() for indicator in detail_indicators)
        
        return {
            "domain": domain,
            "is_internal": is_internal,
            "path": path,
            "path_segments": path_segments,
            "path_length": len(path_segments),
            "query_params": query_params,
            "has_id": has_id, 
            "id_value": id_value,
            "is_detail_page": is_detail_page,
            "has_extension": "." in path_segments[-1] if path_segments else False,
            "depth": len([s for s in path_segments if s])
        }
    
    def _detect_content_type(self, url: str, anchor_text: str) -> Tuple[str, float]:
        """
        Detect likely content type from URL and anchor text.
        
        Args:
            url: URL string
            anchor_text: Anchor text for the link
            
        Returns:
            Tuple of (content_type, confidence)
        """
        url_lower = url.lower()
        text_lower = anchor_text.lower()
        
        # Check against content patterns
        max_confidence = 0.0
        detected_type = "unknown"
        
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if pattern in url_lower:
                    confidence = 0.8
                    return content_type, confidence
        
        # Check anchor text for clues if URL didn't match
        type_indicators = {
            "product": ["product", "item", "buy", "purchase", "shop", "order"],
            "category": ["category", "collection", "department", "shop", "browse"],
            "listing": ["listings", "results", "search", "find", "properties"],
            "article": ["article", "post", "blog", "read", "news"],
            "profile": ["profile", "account", "login", "sign in", "member"],
            "contact": ["contact", "about", "help", "support"]
        }
        
        for content_type, indicators in type_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return content_type, 0.6
        
        # Check for likely detail pages based on URL structure
        if re.search(r'/[a-z0-9-]+/\d+$', url_lower):
            return "detail", 0.7
            
        if re.search(r'/[a-z0-9-]+-[a-z0-9-]+$', url_lower):
            # URLs with slugs like "/blue-t-shirt" are likely detail pages
            return "detail", 0.6
            
        # Default to navigation type with low confidence
        return "navigation", 0.3
    
    def _calculate_exploration_factor(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate exploration factor based on results collected so far.
        
        Args:
            results: List of results collected so far
            
        Returns:
            Exploration factor between 0.0 and 1.0
        """
        if not results:
            # If no results yet, full exploration
            return 1.0
            
        # Calculate average relevance of results
        relevance_values = [r.get("score", 0) for r in results if "score" in r]
        
        if not relevance_values:
            return 0.8  # Default to high exploration if no scores
            
        avg_relevance = sum(relevance_values) / len(relevance_values)
        
        # Adjust exploration based on relevance
        if avg_relevance > 0.8:
            # If we have very relevant content, reduce exploration
            return 0.3
        elif avg_relevance > 0.6:
            # Moderate exploration for somewhat relevant content
            return 0.5
        else:
            # More exploration for less relevant results
            return 0.8
            
    def _assess_content_quality(self, results: List[Dict[str, Any]]) -> str:
        """
        Assess the quality of content collected so far.
        
        Args:
            results: List of results collected
            
        Returns:
            Content quality assessment ("poor", "moderate", "good")
        """
        if not results:
            return "unknown"
            
        # Count results with different relevance levels
        good_results = 0
        moderate_results = 0
        
        for result in results:
            score = result.get("score", 0)
            
            if score > 0.7:
                good_results += 1
            elif score > 0.4:
                moderate_results += 1
                
        # Assess overall quality
        if good_results > len(results) * 0.5:
            return "good"
        elif moderate_results + good_results > len(results) * 0.5:
            return "moderate"
        else:
            return "poor"
    
    def _adjust_link_score(self, 
                          base_score: float, 
                          content_type: str, 
                          navigation_intention: str,
                          exploration_factor: float,
                          results: List[Dict[str, Any]]) -> float:
        """
        Adjust link score based on content type and navigation intention.
        
        Args:
            base_score: Initial relevance score
            content_type: Type of content the link points to
            navigation_intention: Current navigation goal
            exploration_factor: Factor for exploration vs exploitation
            results: Current results
            
        Returns:
            Adjusted score
        """
        # Count the types of pages we've already visited
        content_type_counts = {}
        
        for result in results:
            result_type = result.get("content_type", "unknown")
            content_type_counts[result_type] = content_type_counts.get(result_type, 0) + 1
        
        adjustment = 0
        
        # Adjust based on navigation intention
        if navigation_intention == "explore":
            # For exploration, give boost to types we haven't seen much
            if content_type_counts.get(content_type, 0) <= 2:
                adjustment += 0.2 * exploration_factor
                
        elif navigation_intention == "exploit":
            # For exploitation, focus on types that have given good results
            good_types = self._get_highest_value_content_types(results)
            if content_type in good_types:
                adjustment += 0.25 * (1 - exploration_factor)
                
        elif navigation_intention == "diversify":
            # For diversification, balance with slight preference for new types
            if content_type_counts.get(content_type, 0) <= 1:
                adjustment += 0.15
            elif content_type_counts.get(content_type, 0) >= 5:
                adjustment -= 0.1
        
        # Special case for detail pages
        if content_type in ["product", "detail", "article"] and self._is_primary_target(content_type):
            adjustment += 0.2
            
        # Adjust based on page type counts
        if content_type_counts.get(content_type, 0) > 10:
            # If we've seen too many of this type, reduce score
            adjustment -= 0.15
            
        return base_score + adjustment
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        if not text:
            return []
            
        # Convert to lowercase and split
        words = text.lower().split()
        
        # Remove stop words
        stop_words = ['the', 'and', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 
                     'with', 'by', 'about', 'like', 'was', 'were', 'is', 'are']
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count word frequency
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
            
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:max_keywords]]
    
    def _analyze_url_path(self, path_segments: List[str], content_type: str) -> Dict[str, Any]:
        """
        Analyze URL path segments for content indicators.
        
        Args:
            path_segments: List of path segments
            content_type: Detected content type
            
        Returns:
            Dictionary with path analysis results
        """
        result = {
            "indicators": [],
            "relevance": 0.0
        }
        
        if not path_segments:
            return result
            
        # Check for patterns that indicate high-value content
        relevant_patterns = []
        
        # For different content types, look for relevant indicators
        if content_type == "product":
            relevant_patterns = ["product", "item", "detail", "view", "p"]
        elif content_type == "listing":
            relevant_patterns = ["search", "listing", "results", "properties", "find"]
        elif content_type == "article":
            relevant_patterns = ["article", "post", "blog", "news", "story"]
            
        # Look for matches
        for segment in path_segments:
            segment_lower = segment.lower()
            for pattern in relevant_patterns:
                if pattern in segment_lower:
                    result["indicators"].append(pattern)
                    result["relevance"] += 0.1
                    
        # Check if the last segment has an ID pattern
        if path_segments:
            last_segment = path_segments[-1]
            if re.match(r'^\d+$', last_segment):
                result["indicators"].append("numeric_id")
                result["relevance"] += 0.15
            elif re.match(r'^[a-z0-9-]+-\d+$', last_segment.lower()):
                result["indicators"].append("slug_with_id")
                result["relevance"] += 0.2
        
        # Cap relevance at 0.5
        result["relevance"] = min(0.5, result["relevance"])
        return result
    
    def _calculate_relevance_score(self, url: str, anchor_text: str, url_features: Dict[str, Any]) -> float:
        """
        Calculate relevance score based on URL and anchor text.
        
        Args:
            url: URL string
            anchor_text: Anchor text
            url_features: Dictionary of URL features
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Base score
        score = 0.4
        
        # Internal links are preferred
        if url_features["is_internal"]:
            score += 0.1
            
        # Adjust based on path depth
        depth = url_features["depth"]
        if depth == 0:
            # Homepage
            score += 0.05
        elif 1 <= depth <= 2:
            # Category pages
            score += 0.1
        elif 3 <= depth <= 4:
            # Detail pages
            score += 0.15
            
        # Detail pages with IDs are valuable
        if url_features["has_id"]:
            score += 0.15
            
        # Check if anchor text is meaningful
        if len(anchor_text) > 3:
            score += 0.05
            
        # Check for keywords in anchor text
        if self.crawl_intent and "keywords" in self.crawl_intent:
            for keyword in self.crawl_intent["keywords"]:
                if keyword.lower() in anchor_text.lower():
                    score += 0.2
                    break
                    
        return min(1.0, score)
    
    def _calculate_keyword_match_score(self, url: str, anchor_text: str, context: str) -> float:
        """
        Calculate keyword match score based on crawl intent.
        
        Args:
            url: URL string
            anchor_text: Anchor text
            context: Link context
            
        Returns:
            Keyword match score between 0.0 and 1.0
        """
        if not self.crawl_intent or "keywords" not in self.crawl_intent:
            return 0.0
            
        keywords = self.crawl_intent["keywords"]
        if not keywords:
            return 0.0
            
        # Calculate matches
        matches = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in url.lower():
                matches += 2  # URL matches are more valuable
            elif keyword_lower in anchor_text.lower():
                matches += 1
            elif context and keyword_lower in context.lower():
                matches += 0.5  # Context matches are less valuable
                
        # Calculate score based on number of matches
        return min(1.0, matches / (len(keywords) * 2))
    
    def _calculate_high_value_content_boost(self, url: str, anchor_text: str, context: str, content_type: str) -> float:
        """
        Calculate boost for high-value content based on domain and interest area.
        
        Args:
            url: URL string
            anchor_text: Anchor text
            context: Link context
            content_type: Detected content type
            
        Returns:
            Boost value between 0.0 and 0.3
        """
        boost = 0.0
        
        # Determine the domain interest area from crawl intent
        domain_area = self._determine_domain_area()
        
        # If we have a domain area, check for relevant terms
        if domain_area and domain_area in self.high_value_terms:
            relevant_terms = self.high_value_terms[domain_area]
            
            # Check URL, anchor text, and context for matches
            text_to_check = f"{url} {anchor_text} {context}".lower()
            matches = sum(1 for term in relevant_terms if term in text_to_check)
            
            if matches >= 3:
                boost = 0.3
            elif matches >= 1:
                boost = 0.2
        
        # For general detail pages that look promising
        if content_type in ["product", "article", "detail"] and ("detail" in url.lower() or "view" in url.lower()):
            boost += 0.1
            
        return min(0.3, boost)
    
    def _determine_domain_area(self) -> Optional[str]:
        """
        Determine domain area based on crawl intent.
        
        Returns:
            Domain area or None if not determinable
        """
        if not self.crawl_intent:
            return None
            
        # Check for explicit domain area
        if "domain_area" in self.crawl_intent:
            return self.crawl_intent["domain_area"]
            
        # Check keywords for domain indicators
        if "keywords" in self.crawl_intent:
            # Count matches for each domain area
            area_scores = {area: 0 for area in self.high_value_terms.keys()}
            
            for keyword in self.crawl_intent["keywords"]:
                keyword_lower = keyword.lower()
                for area, terms in self.high_value_terms.items():
                    if any(term in keyword_lower for term in terms):
                        area_scores[area] += 1
            
            # Find the best match
            best_area = None
            best_score = 0
            for area, score in area_scores.items():
                if score > best_score:
                    best_score = score
                    best_area = area
            
            # Only return if we have at least one match
            if best_score > 0:
                return best_area
        
        return None
    
    def _get_highest_value_content_types(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Get the content types that have yielded the highest value results.
        
        Args:
            results: List of results
            
        Returns:
            List of highest value content types
        """
        if not results:
            return []
            
        # Calculate average score for each content type
        type_scores = {}
        type_counts = {}
        
        for result in results:
            content_type = result.get("content_type", "unknown")
            score = result.get("score", 0)
            
            if content_type not in type_scores:
                type_scores[content_type] = 0
                type_counts[content_type] = 0
                
            type_scores[content_type] += score
            type_counts[content_type] += 1
        
        # Calculate averages
        avg_scores = {}
        for ctype in type_scores:
            if type_counts[ctype] > 0:
                avg_scores[ctype] = type_scores[ctype] / type_counts[ctype]
        
        # Sort by average score
        sorted_types = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top types with scores above 0.6
        return [ctype for ctype, score in sorted_types if score >= 0.6]
    
    def _is_primary_target(self, content_type: str) -> bool:
        """
        Check if the content type is a primary target based on crawl intent.
        
        Args:
            content_type: Content type to check
            
        Returns:
            True if primary target, False otherwise
        """
        if not self.crawl_intent:
            return False
            
        # Check explicit target type if specified
        if "target_content_type" in self.crawl_intent:
            return self.crawl_intent["target_content_type"] == content_type
            
        # For different extraction tasks, infer appropriate targets
        if "extraction_task" in self.crawl_intent:
            task = self.crawl_intent["extraction_task"].lower()
            
            if "product" in task or "item" in task:
                return content_type in ["product", "detail"]
                
            if "article" in task or "blog" in task or "news" in task:
                return content_type == "article"
                
            if "list" in task or "catalog" in task or "search" in task:
                return content_type in ["listing", "category", "search_results"]
                
        return False
        
    def _calculate_heuristic_score(self, 
                                  url: str, 
                                  anchor_text: str, 
                                  context: str,
                                  url_features: Dict[str, Any],
                                  content_type: str,
                                  current_depth: int,
                                  max_depth: int,
                                  parent_url: str,
                                  results: List[Dict[str, Any]]) -> float:
        """
        Calculate a heuristic score for a link when AI is not available.
        
        Args:
            url: URL string
            anchor_text: Anchor text
            context: Link context
            url_features: URL features
            content_type: Detected content type
            current_depth: Current depth
            max_depth: Maximum depth
            parent_url: Parent URL
            results: Results collected so far
            
        Returns:
            Heuristic score between 0.0 and 1.0
        """
        # Base score
        score = 0.3
        
        # For deep levels, be more selective
        depth_factor = max(0, 1 - (current_depth / max_depth))
        
        # Internal links are preferred
        if url_features["is_internal"]:
            score += 0.2 * depth_factor
        
        # Detail pages with IDs are valuable
        if url_features["has_id"] and content_type in ["product", "article", "detail"]:
            score += 0.25
            
        # Check for keywords in link text or URL
        if self.crawl_intent and "keywords" in self.crawl_intent:
            keyword_matches = 0
            for keyword in self.crawl_intent["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower in url.lower():
                    keyword_matches += 2  # URL matches are more valuable
                elif keyword_lower in anchor_text.lower():
                    keyword_matches += 1
                elif context and keyword_lower in context.lower():
                    keyword_matches += 0.5
            
            # Boost score based on keyword matches
            if keyword_matches > 0:
                score += min(0.3, keyword_matches * 0.1)
                
        # Adjust based on content type and depth relationship
        if content_type in ["product", "article", "detail"] and current_depth < max_depth - 1:
            score += 0.15
        elif content_type in ["listing", "category"] and current_depth < 2:
            score += 0.1
        elif content_type in ["search_results", "search_form"] and current_depth == 0:
            score += 0.2
            
        # Consider URL structure quality
        if '-' in url_features["path"]:  # Well-formed URLs often use hyphens
            score += 0.05
            
        # Avoid extremely long URLs
        if len(url) > 150:
            score -= 0.1
            
        # Consider anchor text quality
        if len(anchor_text) > 3:
            # Longer anchor text is often more descriptive
            score += min(0.1, len(anchor_text) / 100)
            
        # If we've seen similar content types with high scores
        content_type_scores = {}
        for result in results:
            result_type = result.get("content_type", "unknown")
            result_score = result.get("score", 0)
            if result_type not in content_type_scores or result_score > content_type_scores[result_type]:
                content_type_scores[result_type] = result_score
                
        if content_type in content_type_scores:
            # Boost based on previous success with this content type
            score += content_type_scores[content_type] * 0.15
            
        # Ensure score is within bounds
        return max(0.1, min(1.0, score))
    
    def _select_most_promising_links(self, links: List[Tuple[str, str, str]], max_links: int = 5) -> List[Tuple[str, str, str]]:
        """
        Select the most promising links based on heuristics.
        
        Args:
            links: List of tuples (URL, anchor_text, context)
            max_links: Maximum number of links to select
            
        Returns:
            Filtered list of links
        """
        scored_links = []
        # Calculate quick scores
        for url, text, context in links:
            # Quick score calculation
            quick_score = 0.0
            
            # Parse URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Internal links preferred
            if self.main_domain and domain == self.main_domain:
                quick_score += 0.3
            
            # Check for keywords
            if self.crawl_intent and "keywords" in self.crawl_intent:
                for keyword in self.crawl_intent["keywords"]:
                    if keyword.lower() in url.lower() or keyword.lower() in text.lower():
                        quick_score += 0.4
                        break
            
            # Prefer detail pages at deeper levels
            path_segments = parsed_url.path.split('/')
            if len(path_segments) >= 3 and current_depth > 0:
                quick_score += 0.2
            
            # Detect likely content type
            content_type, _ = self._detect_content_type(url, text)
            
            # Adjust for content type
            if content_type in ["product", "article", "detail"]:
                quick_score += 0.3
            elif content_type in ["listing", "category"]:
                quick_score += 0.2
            
            scored_links.append((url, text, context, quick_score))
        
        # Sort by score and take top max_links
        scored_links.sort(key=lambda x: x[3], reverse=True)
        return [(url, text, context) for url, text, context, _ in scored_links[:max_links]]
    
    def _prefilter_links(self, 
                        links: List[Tuple[str, str, str]], 
                        parent_url: str,
                        current_depth: int,
                        results: List[Dict[str, Any]],
                        max_links: int = 15) -> List[Tuple[str, str, str]]:
        """
        Pre-filter links using simple heuristics before detailed analysis.
        
        Args:
            links: List of links to filter
            parent_url: Parent URL
            current_depth: Current depth
            results: Results collected so far
            max_links: Maximum links to return
            
        Returns:
            Filtered list of links
        """
        scored_links = []
        # Score links using quick heuristics
        for url, text, context in links:
            # Simple scoring based on URL and text features
            score = 0.0
            
            # Check if we have previous score for this URL
            if url in self.url_scores:
                score += self.url_scores[url] * 0.5
            
            # Give higher score to detail pages based on URL features
            if re.search(r'/[a-z0-9-]+-[a-z0-9-]+$', url.lower()):
                score += 0.2
            if re.search(r'/\d+$', url.lower()) or '/p/' in url.lower():
                score += 0.3
            
            # Check keywords in crawl intent
            if self.crawl_intent and "keywords" in self.crawl_intent:
                for keyword in self.crawl_intent["keywords"]:
                    if keyword.lower() in text.lower() or keyword.lower() in url.lower():
                        score += 0.3
                        break
            
            scored_links.append((url, text, context, score))
        
        # Sort by score descending and take top max_links
        scored_links.sort(key=lambda x: x[3], reverse=True)
        return [(url, text, context) for url, text, context, _ in scored_links[:max_links]]
    
    def _determine_navigation_intention(self, 
                                       current_depth: int, 
                                       max_depth: int,
                                       results: List[Dict[str, Any]],
                                       content_quality: str) -> str:
        """
        Determine navigation intention based on current state.
        
        Args:
            current_depth: Current crawl depth
            max_depth: Maximum crawl depth
            results: List of results
            content_quality: Quality assessment of content
            
        Returns:
            Navigation intention ("explore", "exploit", "diversify")
        """
        # If we're just starting out, focus on exploration
        if current_depth == 0:
            return "explore"
            
        # If we're at max depth - 1, we should be going for the highest value
        if current_depth >= max_depth - 1:
            return "exploit"
            
        # If we've already found good content, focus on exploiting
        if content_quality == "good" and len(results) > 5:
            return "exploit"
            
        # If we've found moderate content but need more variety
        if content_quality == "moderate" and len(results) > 10:
            return "diversify"
            
        # Default to exploration if we haven't found much good content
        return "explore"
    
    def _get_content_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get content context from results.
        
        Args:
            results: List of results
            
        Returns:
            Dictionary with content context information
        """
        if not results:
            return {}
            
        context = {
            "content_types": {},
            "keywords": set(),
            "best_score": 0.0,
            "best_content": None,
        }
        
        for result in results:
            content_type = result.get("content_type", "unknown")
            score = result.get("score", 0.0)
            
            # Count content types
            context["content_types"][content_type] = context["content_types"].get(content_type, 0) + 1
            
            # Extract keywords from content
            if "data" in result:
                data = result["data"]
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            words = self._extract_keywords(value, max_keywords=5)
                            context["keywords"].update(words)
            
            # Track best content
            if score > context["best_score"]:
                context["best_score"] = score
                context["best_content"] = result
                
        return context
    
    def _adjust_scores_for_strategy(self, 
                                   scored_links: List[Tuple[str, float, str]], 
                                   current_depth: int, 
                                   max_depth: int) -> List[Tuple[str, float, str]]:
        """
        Adjust scores based on current strategy and depth.
        
        Args:
            scored_links: List of scored links
            current_depth: Current depth
            max_depth: Maximum depth
            
        Returns:
            List of links with adjusted scores
        """
        if not scored_links:
            return []
            
        link_types = {}
        for url, score, explanation in scored_links:
            content_type, _ = self._detect_content_type(url, "")
            link_types[content_type] = link_types.get(content_type, 0) + 1
        
        adjusted_links = []
        for url, score, explanation in scored_links:
            content_type, _ = self._detect_content_type(url, "")
            adjusted_score = score
            
            # At deeper levels, focus more on detail pages
            if current_depth >= max_depth - 2:
                if content_type in ["product", "article", "detail"]:
                    adjusted_score += 0.15
            
            # Ensure diversity if we have many of one type
            if link_types.get(content_type, 0) > 5:
                adjusted_score -= 0.05
            
            adjusted_links.append((url, min(1.0, adjusted_score), explanation))
        
        return adjusted_links
    
    def _get_content_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get content context from results.
        
        Args:
            results: List of results
            
        Returns:
            Dictionary with content context information
        """
        if not results:
            return {}
            
        context = {
            "content_types": {},
            "keywords": set(),
            "best_score": 0.0,
            "best_content": None,
        }
        
        for result in results:
            content_type = result.get("content_type", "unknown")
            score = result.get("score", 0.0)
            
            # Count content types
            context["content_types"][content_type] = context["content_types"].get(content_type, 0) + 1
            
            # Extract keywords from content
            if "data" in result:
                data = result["data"]
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            words = self._extract_keywords(value, max_keywords=5)
                            context["keywords"].update(words)
            
            # Track best content
            if score > context["best_score"]:
                context["best_score"] = score
                context["best_content"] = result
                
        return context
    
    def get_link_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get metadata for a link.
        """
        return self.link_metadata.get(url, {})
    
    def _store_link_metadata(self, url: str, metadata: Dict[str, Any]) -> None:
        """
        Store metadata for a link.
        """
        self.link_metadata[url] = metadata