"""
Navigation Planning Module

Handles all navigation-related functionality for the AI-guided strategy,
including path planning, navigation history, and link selection intelligence.
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin, parse_qs, urlunparse

from bs4 import BeautifulSoup
import google.generativeai as genai
import networkx as nx

from strategies.ai_guided.ai_cache import AIResponseCache
from strategies.ai_guided.link_prioritization import LinkPrioritizer
from utils.url_filters import is_same_domain, get_domain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NavigationPlanning")

class NavigationPlanner:
    """
    Handles all navigation-related functionality for the AI-guided strategy,
    including path planning, navigation history, and decisions about which links to follow.
    """
    
    def __init__(self, 
                response_cache: AIResponseCache,
                link_prioritizer: LinkPrioritizer,
                crawl_intent: Dict[str, Any] = None,
                settings: Dict[str, Any] = None):
        """
        Initialize the navigation planning module.
        
        Args:
            response_cache: Cache for AI API responses
            link_prioritizer: Module for scoring links by relevance
            crawl_intent: Dict containing user intent information
            settings: Dict with configuration settings
        """
        self.response_cache = response_cache
        self.link_prioritizer = link_prioritizer
        self.crawl_intent = crawl_intent or {}
        self.settings = settings or {}
        
        # Performance tracking
        self.performance_stats = {"ai_calls": 0, "cache_hits": 0}
        
        # Navigation state
        self.visited_pages = set()
        self.rejected_links = set()
        self.navigation_graph = nx.DiGraph()
        self.page_scores = {}  # Stores relevance scores for visited pages
        self.link_scores = {}  # Stores relevance scores for encountered links
        self.navigation_patterns = {}  # Recognizes patterns in navigation
        self.current_path = []  # Track current navigation path
        
        # Path optimization settings
        self.max_path_length = self.settings.get("max_path_length", 10)
        self.exploration_factor = self.settings.get("exploration_factor", 0.5)  # 0-1, higher means more exploration
        self.minimum_page_relevance = self.settings.get("min_page_relevance", 0.2)  # 0-1, minimum page relevance to continue
        
        # URL normalization regex patterns
        self.query_params_to_ignore = [
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'ref', 'source', 'affiliate'
        ]
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL for consistent comparison.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL string
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            # Convert domain to lowercase
            netloc = parsed.netloc.lower()
            
            # Remove unnecessary query parameters
            if parsed.query:
                query_dict = parse_qs(parsed.query)
                # Filter out tracking parameters
                filtered_query = {k: v for k, v in query_dict.items() 
                                if k.lower() not in self.query_params_to_ignore}
                # Reconstruct query string
                query = "&".join(f"{k}={'&'.join(v)}" for k, v in sorted(filtered_query.items()))
            else:
                query = ""
            
            # Remove trailing slash from path
            path = parsed.path
            if path.endswith('/') and path != '/':
                path = path[:-1]
                
            # Reconstruct URL
            cleaned = urlunparse((parsed.scheme, 
                                netloc, 
                                path, 
                                parsed.params, 
                                query, 
                                ''))  # No fragment
                                
            return cleaned
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {str(e)}")
            return url
            
    def update_navigation_graph(self, current_url: str, linked_urls: List[str], 
                              relevance_scores: Dict[str, float] = None) -> None:
        """
        Update the navigation graph with new URLs and their relationships.
        
        Args:
            current_url: Current page URL
            linked_urls: List of URLs linked from the current page
            relevance_scores: Optional dictionary mapping URLs to relevance scores
        """
        try:
            # Normalize URLs
            norm_current = self.normalize_url(current_url)
            
            # Add current page if not already in graph
            if not self.navigation_graph.has_node(norm_current):
                self.navigation_graph.add_node(norm_current)
            
            # Add linked pages and edges
            for url in linked_urls:
                norm_url = self.normalize_url(url)
                
                # Skip self-links
                if norm_url == norm_current:
                    continue
                    
                # Add node and edge
                if not self.navigation_graph.has_node(norm_url):
                    self.navigation_graph.add_node(norm_url)
                    
                    # Add relevance score if available
                    if relevance_scores and url in relevance_scores:
                        self.navigation_graph.nodes[norm_url]['relevance'] = relevance_scores[url]
                
                # Add edge from current to linked URL if not exists
                if not self.navigation_graph.has_edge(norm_current, norm_url):
                    self.navigation_graph.add_edge(norm_current, norm_url)
            
        except Exception as e:
            logger.error(f"Error updating navigation graph: {str(e)}")
    
    def update_path_memory(self, url: str, page_score: float = 0.0, 
                         content_type: str = None) -> None:
        """
        Update path history with information about a visited page.
        
        Args:
            url: URL of the visited page
            page_score: Relevance score of the page (0-1)
            content_type: Type of content found on the page
        """
        normalized_url = self.normalize_url(url)
        
        # Add to visited pages
        self.visited_pages.add(normalized_url)
        
        # Update page score
        self.page_scores[normalized_url] = page_score
        
        # Update current path
        self.current_path.append((normalized_url, page_score, content_type))
        
        # Limit path length for memory efficiency
        if len(self.current_path) > self.max_path_length:
            self.current_path.pop(0)
    
    def mark_link_rejected(self, url: str, reason: str = None) -> None:
        """
        Mark a link as rejected to avoid re-evaluating it.
        
        Args:
            url: URL to mark as rejected
            reason: Optional reason for rejection
        """
        normalized_url = self.normalize_url(url)
        self.rejected_links.add(normalized_url)
        
        if reason and normalized_url in self.link_scores:
            # Add rejection reason to link metadata
            if 'metadata' not in self.link_scores[normalized_url]:
                self.link_scores[normalized_url]['metadata'] = {}
            
            self.link_scores[normalized_url]['metadata']['rejection_reason'] = reason
    
    def is_link_rejected(self, url: str) -> bool:
        """
        Check if a link has been previously rejected.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if link was rejected
        """
        return self.normalize_url(url) in self.rejected_links
    
    def is_page_visited(self, url: str) -> bool:
        """
        Check if a page has been previously visited.
        
        Args:
            url: URL to check
            
        Returns:
            Boolean indicating if page was visited
        """
        return self.normalize_url(url) in self.visited_pages
    
    def detect_navigation_patterns(self) -> Dict[str, Any]:
        """
        Analyze the navigation history to detect patterns.
        
        Returns:
            Dictionary with detected navigation patterns
        """
        patterns = {}
        
        if len(self.current_path) < 3:
            return patterns
            
        # Extract domains from current path
        domains = [get_domain(url) for url, _, _ in self.current_path]
        
        # Check for domain switching patterns
        domain_switches = []
        current_domain = domains[0]
        for i, domain in enumerate(domains[1:], 1):
            if domain != current_domain:
                domain_switches.append((i, current_domain, domain))
                current_domain = domain
                
        if domain_switches:
            patterns["domain_switches"] = domain_switches
            
        # Check for repetitive content patterns (similar page types)
        content_types = [content_type for _, _, content_type in self.current_path if content_type]
        if len(content_types) >= 3:
            # Count consecutive occurrences of the same content type
            repeat_patterns = []
            current_type = content_types[0]
            count = 1
            
            for ct in content_types[1:]:
                if ct == current_type:
                    count += 1
                else:
                    if count >= 3:
                        repeat_patterns.append((current_type, count))
                    current_type = ct
                    count = 1
                    
            # Check the last sequence
            if count >= 3:
                repeat_patterns.append((current_type, count))
                
            if repeat_patterns:
                patterns["content_repetition"] = repeat_patterns
        
        # Check for score progression (increasing or decreasing relevance)
        scores = [score for _, score, _ in self.current_path]
        if len(scores) >= 3:
            increasing = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
            decreasing = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            
            if increasing:
                patterns["score_progression"] = "increasing"
            elif decreasing:
                patterns["score_progression"] = "decreasing"
        
        return patterns
    
    def calculate_exploration_priority(self, 
                                     url: str, 
                                     content_scores: Dict[str, float],
                                     navigation_patterns: Dict[str, Any] = None) -> float:
        """
        Calculate an exploration priority score for a URL based on navigation history.
        
        Args:
            url: URL to calculate priority for
            content_scores: Relevance scores for content
            navigation_patterns: Optional patterns detected in navigation
            
        Returns:
            Priority score for exploration (0-1)
        """
        base_score = content_scores.get(url, 0.5)
        normalized_url = self.normalize_url(url)
        
        # Adjust score based on navigation patterns
        adjustment = 0.0
        patterns = navigation_patterns or self.detect_navigation_patterns()
        
        # If we're getting decreasing scores, encourage more exploration
        if patterns.get("score_progression") == "decreasing":
            adjustment += 0.2 * self.exploration_factor
        
        # If we're seeing content repetition, encourage breaking the pattern
        if "content_repetition" in patterns:
            adjustment += 0.15 * self.exploration_factor
            
        # If URL is in a domain we haven't visited much, increase priority
        domain = get_domain(url)
        visited_domains = [get_domain(u) for u, _, _ in self.current_path]
        domain_frequency = visited_domains.count(domain) / max(1, len(visited_domains))
        
        # Lower frequency means higher adjustment (more exploration)
        if domain_frequency < 0.3:
            adjustment += 0.25 * self.exploration_factor
            
        # Check link structure for indicators of listing/index pages
        # These often have higher exploration value
        is_index_page = any(pattern in normalized_url for pattern in 
                           ['/index', '/listing', '/category', '/products', '/search'])
        if is_index_page:
            adjustment += 0.1
            
        # Final score is a weighted combination of content relevance and exploration adjustment
        # The weights are determined by the exploration factor setting
        final_score = (base_score * (1 - self.exploration_factor) + 
                      adjustment * self.exploration_factor)
                      
        # Ensure score is within range
        return max(0.0, min(1.0, final_score))
    
    def find_navigation_path(self, start_url: str, target_url: str) -> List[str]:
        """
        Find a navigation path between two URLs in the graph.
        
        Args:
            start_url: Starting URL
            target_url: Target URL
            
        Returns:
            List of URLs forming a path from start to target, or empty list if no path
        """
        norm_start = self.normalize_url(start_url)
        norm_target = self.normalize_url(target_url)
        
        if norm_start not in self.navigation_graph or norm_target not in self.navigation_graph:
            return []
            
        try:
            path = nx.shortest_path(self.navigation_graph, source=norm_start, target=norm_target)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    async def plan_next_steps(self, 
                            current_url: str, 
                            links_with_text: List[Tuple[str, str]], 
                            content_context: Dict[str, Any],
                            user_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Plan the next navigation steps based on the current state.
        
        Args:
            current_url: Current page URL
            links_with_text: List of (url, text) tuples for links on current page
            content_context: Dict with content information
            user_intent: Dict with user intent information
            
        Returns:
            Prioritized list of next URLs to visit with metadata
        """
        # Get links with scores from the link prioritizer
        links_with_scores = await self.link_prioritizer.prioritize_links(
            links_with_text, 
            current_url, 
            content_context, 
            user_intent
        )
        
        # Extract just the URLs and scores
        all_urls = [url for url, _, _ in links_with_text]
        content_scores = {url: score for url, score in links_with_scores}
        
        # Update navigation graph
        self.update_navigation_graph(current_url, all_urls, content_scores)
        
        # Detect navigation patterns
        patterns = self.detect_navigation_patterns()
        
        # Calculate exploration priorities for each link
        exploration_scores = {}
        for url in all_urls:
            if not self.is_page_visited(url) and not self.is_link_rejected(url):
                exploration_scores[url] = self.calculate_exploration_priority(
                    url, 
                    content_scores,
                    patterns
                )
        
        # Combine content relevance and exploration scores
        combined_scores = {}
        for url in all_urls:
            if url in content_scores and url in exploration_scores:
                # Weight relevance vs. exploration based on settings
                content_weight = 1 - self.exploration_factor
                exploration_weight = self.exploration_factor
                
                combined_scores[url] = (
                    content_scores[url] * content_weight +
                    exploration_scores[url] * exploration_weight
                )
        
        # Create prioritized list of next steps
        prioritized_steps = []
        for url, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            # Find link text for this URL
            link_text = next((text for link_url, text in links_with_text if link_url == url), "")
            
            prioritized_steps.append({
                "url": url,
                "priority_score": score,
                "content_relevance": content_scores.get(url, 0),
                "exploration_value": exploration_scores.get(url, 0),
                "link_text": link_text,
                "is_visited": self.is_page_visited(url),
                "is_rejected": self.is_link_rejected(url)
            })
        
        return prioritized_steps
    
    async def analyze_search_for_navigation(self, 
                                         search_results: Dict[str, Any],
                                         user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results to plan optimal navigation.
        
        Args:
            search_results: Dict with search results information
            user_intent: Dict with user intent information
            
        Returns:
            Dict with navigation recommendations
        """
        if not search_results or not search_results.get("success", False):
            return {
                "should_use_results": False,
                "reason": "Search failed or returned no results"
            }
        
        html_content = search_results.get("html", "")
        if not html_content:
            return {
                "should_use_results": False,
                "reason": "No HTML content in search results"
            }
        
        # Extract links from search results
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            url = a.get('href')
            text = a.get_text(strip=True)
            if url and text:
                # Ensure absolute URL
                if not url.startswith(('http://', 'https://')):
                    base_url = search_results.get("url", "")
                    url = urljoin(base_url, url)
                
                links.append((url, text))
        
        # If no links found, search results are not useful
        if not links:
            return {
                "should_use_results": False,
                "reason": "No links found in search results"
            }
        
        # Get prioritized links
        links_with_scores = await self.link_prioritizer.prioritize_links(
            links, 
            search_results.get("url", ""), 
            {"search_results": True}, 
            user_intent
        )
        
        # Check if we have any high-relevance links
        high_relevance_links = [url for url, score in links_with_scores if score > 0.7]
        
        if high_relevance_links:
            return {
                "should_use_results": True,
                "reason": f"Found {len(high_relevance_links)} highly relevant links in search results",
                "prioritized_links": links_with_scores[:10]  # Return top 10
            }
        elif len(links_with_scores) > 0:
            # If we have any links at all, still worth exploring
            return {
                "should_use_results": True,
                "reason": "Found some potentially relevant links in search results",
                "prioritized_links": links_with_scores[:10]  # Return top 10
            }
        else:
            return {
                "should_use_results": False,
                "reason": "No relevant links found in search results"
            }
    
    async def detect_site_structure(self, links_by_url: Dict[str, List[Tuple[str, str]]]) -> Dict[str, Any]:
        """
        Detect the structure of the site based on links.
        
        Args:
            links_by_url: Dictionary mapping URLs to lists of (link_url, link_text) tuples
            
        Returns:
            Dict with detected site structure information
        """
        structure = {
            "site_type": "unknown",
            "navigation_depth": 0,
            "has_catalog": False,
            "has_categories": False,
            "has_pagination": False,
            "identified_patterns": []
        }
        
        # Not enough data to detect structure
        if not links_by_url or len(links_by_url) < 2:
            return structure
        
        # Count links per page
        links_per_page = {url: len(links) for url, links in links_by_url.items()}
        
        # Calculate average links per page
        avg_links_per_page = sum(links_per_page.values()) / len(links_per_page)
        structure["avg_links_per_page"] = avg_links_per_page
        
        # Detect pagination patterns
        pagination_patterns = [
            r'page=\d+', r'p=\d+', r'/page/\d+', r'/p/\d+',
            r'offset=\d+', r'start=\d+', r'from=\d+',
            r'next', r'prev', r'previous', r'\bnext\b', r'\bprev\b'
        ]
        
        # Detect category patterns
        category_patterns = [
            r'/category/', r'/categories/', r'/catalog/', 
            r'/department/', r'/section/', r'/topics/'
        ]
        
        # Check for pagination and category links
        pagination_links = 0
        category_links = 0
        
        for links in links_by_url.values():
            for url, text in links:
                # Check URL for pagination patterns
                if any(re.search(pattern, url, re.IGNORECASE) for pattern in pagination_patterns):
                    pagination_links += 1
                    
                # Check URL for category patterns
                if any(re.search(pattern, url, re.IGNORECASE) for pattern in category_patterns):
                    category_links += 1
                    
                # Check link text for pagination indicators
                if re.search(r'next|previous|prev|\d+', text, re.IGNORECASE):
                    pagination_links += 1
        
        # Determine if pagination and categories exist
        structure["has_pagination"] = pagination_links >= 3
        structure["has_categories"] = category_links >= 3
        
        # If we have both pagination and categories, likely a catalog site
        if structure["has_pagination"] and structure["has_categories"]:
            structure["has_catalog"] = True
            structure["site_type"] = "catalog"
        
        # Estimate navigation depth based on URL structure
        all_urls = list(links_by_url.keys())
        depths = [len(urlparse(url).path.split('/')) - 1 for url in all_urls]
        structure["navigation_depth"] = max(depths) if depths else 0
        
        # Identify common path patterns
        paths = [urlparse(url).path for url in all_urls]
        path_segments = {}
        
        for path in paths:
            segments = path.strip('/').split('/')
            for i, segment in enumerate(segments):
                if i not in path_segments:
                    path_segments[i] = []
                path_segments[i].append(segment)
        
        # Find common segments at each depth
        for depth, segments in path_segments.items():
            # Count occurrences of each segment
            segment_counts = {}
            for segment in segments:
                if segment not in segment_counts:
                    segment_counts[segment] = 0
                segment_counts[segment] += 1
            
            # Get segments that appear frequently
            common_segments = [(s, c) for s, c in segment_counts.items() 
                              if c > len(segments) * 0.3]  # At least 30% occurrence
            
            # Add to identified patterns
            if common_segments:
                structure["identified_patterns"].append({
                    "depth": depth,
                    "common_segments": common_segments
                })
        
        return structure
    
    async def optimize_navigation_strategy(self, 
                                        site_structure: Dict[str, Any], 
                                        crawl_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize navigation strategy based on site structure and user intent.
        
        Args:
            site_structure: Dict with detected site structure
            crawl_intent: Dict with user intent information
            
        Returns:
            Dict with optimized navigation strategy
        """
        strategy = {
            "exploration_factor": self.exploration_factor,
            "max_depth": 5,  # Default
            "use_breadth_first": False,
            "follow_pagination": True,
            "focus_on_categories": False,
            "priority_patterns": [],
            "avoid_patterns": []
        }
        
        # Adjust strategy based on site structure
        if site_structure.get("site_type") == "catalog":
            # For catalog sites, focus on categories and pagination
            strategy["focus_on_categories"] = True
            strategy["max_depth"] = max(3, site_structure.get("navigation_depth", 3))
            
            # If we know the site has deep navigation, increase max depth
            if site_structure.get("navigation_depth", 0) > 5:
                strategy["max_depth"] = site_structure.get("navigation_depth")
                
            # For catalog sites with many links, breadth-first is often better
            if site_structure.get("avg_links_per_page", 0) > 30:
                strategy["use_breadth_first"] = True
        
        # Adjust based on intent
        intent_type = crawl_intent.get("intent_type", "").lower() if crawl_intent else ""
        
        if intent_type == "find_specific":
            # For specific item search, reduce exploration, increase focus
            strategy["exploration_factor"] = max(0.2, self.exploration_factor - 0.2)
            strategy["max_depth"] += 2  # Allow deeper exploration to find specific item
        
        elif intent_type == "collect_multiple":
            # For collection tasks, increase breadth exploration
            strategy["use_breadth_first"] = True
            strategy["exploration_factor"] = min(0.8, self.exploration_factor + 0.2)
        
        elif intent_type == "learn_about":
            # For information gathering, balance depth and breadth
            strategy["max_depth"] += 1
            
        # Add priority patterns based on intent keywords
        if crawl_intent and "keywords" in crawl_intent:
            for keyword in crawl_intent["keywords"]:
                # Create a regex pattern that looks for the keyword in URLs
                pattern = r'/' + re.escape(keyword.lower()) + r'(/|$)'
                strategy["priority_patterns"].append(pattern)
        
        return strategy
                
    def generate_navigation_report(self) -> Dict[str, Any]:
        """
        Generate a report on navigation performance and patterns.
        
        Returns:
            Dict with navigation statistics and insights
        """
        # Calculate basic statistics
        report = {
            "pages_visited": len(self.visited_pages),
            "links_rejected": len(self.rejected_links),
            "average_page_score": 0,
            "path_length": len(self.current_path),
            "navigation_patterns": {}
        }
        
        # Calculate average page relevance
        if self.page_scores:
            report["average_page_score"] = sum(self.page_scores.values()) / len(self.page_scores)
        
        # Get navigation patterns
        report["navigation_patterns"] = self.detect_navigation_patterns()
        
        # Calculate domain statistics
        domains = {}
        for url in self.visited_pages:
            domain = get_domain(url)
            if domain not in domains:
                domains[domain] = 0
            domains[domain] += 1
            
        report["domains_visited"] = len(domains)
        report["pages_per_domain"] = domains
        
        # Calculate graph statistics if we have a non-empty graph
        if self.navigation_graph:
            report["graph_stats"] = {
                "nodes": len(self.navigation_graph.nodes),
                "edges": len(self.navigation_graph.edges),
                "connected_components": nx.number_connected_components(
                    self.navigation_graph.to_undirected()
                )
            }
            
            # Find most central pages
            try:
                centrality = nx.eigenvector_centrality(self.navigation_graph)
                most_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                report["most_central_pages"] = most_central
            except:
                # Eigenvector centrality can fail on some graph structures
                report["most_central_pages"] = []
            
        return report