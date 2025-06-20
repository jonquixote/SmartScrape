"""
Best-First Search Strategy

This strategy prioritizes URLs based on relevance to the user's query.
It uses a scoring system to determine which pages are most likely to contain
the desired information and visits them first.
"""

import heapq
import re
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from strategies.base_strategy import BaseStrategy
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig


class BestFirstStrategy(BaseStrategy):
    """
    Best-First Search strategy that prioritizes URLs based on relevance.
    
    Features:
    - Scores URLs based on relevance to the user's search query
    - Visits the most promising pages first
    - Uses a priority queue to manage the crawl frontier
    """
    
    def __init__(self, 
                max_depth: int = 3, 
                max_pages: int = 100,
                include_external: bool = False,
                user_prompt: str = "",
                relevance_threshold: float = 0.2,
                filter_chain: Optional[Any] = None,
                **kwargs):
        """
        Initialize the Best-First Search strategy.
        
        Args:
            max_depth: Maximum crawling depth
            max_pages: Maximum number of pages to crawl
            include_external: Whether to include external links
            user_prompt: The user's original query for relevance scoring
            relevance_threshold: Minimum relevance score for a URL to be crawled
            filter_chain: Filter chain to apply to URLs
        """
        super().__init__(
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            user_prompt=user_prompt,
            filter_chain=filter_chain,
            **kwargs
        )
        
        self.relevance_threshold = relevance_threshold
        self.keywords = self._extract_keywords(user_prompt)
        self.url_scores = {}  # Cache for URL scores
        self.priority_queue = []  # Min-heap (we'll use negative scores for max-heap behavior)
        
    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract keywords from the user prompt for relevance scoring."""
        if not prompt:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', prompt.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords

    @property
    def name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            str: Strategy name
        """
        return "best-first"
    
    def execute(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Execute the best-first search strategy for the given URL.
        
        Args:
            url: The URL to crawl
            **kwargs: Additional arguments including crawler, extraction_config
            
        Returns:
            Dictionary containing the results
        """
        import asyncio
        
        # Extract parameters from kwargs
        crawler = kwargs.get('crawler')
        extraction_config = kwargs.get('extraction_config')
        
        if not crawler:
            raise ValueError("Crawler is required in kwargs for BestFirstStrategy execution")
        
        # Run the async execute method
        try:
            if asyncio.get_event_loop().is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._execute_async(crawler, url, extraction_config))
                    return future.result()
            else:
                return asyncio.run(self._execute_async(crawler, url, extraction_config))
        except Exception as e:
            print(f"Error executing best-first strategy for {url}: {e}")
            return None
    
    async def _execute_async(self, 
                     crawler: AsyncWebCrawler, 
                     start_url: str,
                     extraction_config: Optional[CrawlerRunConfig] = None) -> Dict[str, Any]:
        """
        Execute the Best-First Search crawling strategy.
        
        Args:
            crawler: The AsyncWebCrawler instance
            start_url: The starting URL
            extraction_config: Optional extraction configuration
            
        Returns:
            Dictionary containing the results
        """
        print(f"Starting best-first search from {start_url}")
        print(f"Keywords: {self.keywords}")
        
        # Initialize the priority queue with the starting URL
        start_score = self._calculate_url_score(start_url)
        heapq.heappush(self.priority_queue, (-start_score, 0, start_url))  # negative for max-heap
        
        visited_count = 0
        
        while self.priority_queue and visited_count < self.max_pages:
            # Get the highest priority URL
            neg_score, depth, url = heapq.heappop(self.priority_queue)
            score = -neg_score
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
                
            # Skip if depth exceeds maximum
            if depth > self.max_depth:
                continue
                
            # Skip if score is below threshold
            if score < self.relevance_threshold:
                print(f"Skipping URL with low relevance score {score:.3f}: {url}")
                continue
            
            print(f"Crawling URL (score: {score:.3f}, depth: {depth}): {url}")
            
            # Mark as visited
            self.visited_urls.add(url)
            visited_count += 1
            
            try:
                # Fetch the page
                result = await crawler.arun(url, config=extraction_config)
                
                if result.success and result.html:
                    # Extract data from the page
                    extracted_data = await self.extract(url, result.html)
                    
                    # Add to results
                    self.add_result(
                        url=url,
                        content=extracted_data,
                        depth=depth,
                        relevance=score
                    )
                    
                    # Extract and score new URLs for the queue
                    if depth < self.max_depth:
                        new_urls = self._extract_links(result.html, url)
                        
                        for new_url in new_urls:
                            if self.should_visit(new_url, url):
                                new_score = self._calculate_url_score(new_url)
                                heapq.heappush(self.priority_queue, (-new_score, depth + 1, new_url))
                    
                else:
                    print(f"Failed to fetch {url}")
                    
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
        
        print(f"Best-first search completed. Visited {visited_count} pages.")
        
        return {
            "results": self.results,
            "visited_count": visited_count,
            "strategy": "best-first"
        }
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)
                
                # Basic URL validation
                if absolute_url.startswith(('http://', 'https://')):
                    links.append(absolute_url)
            
            return links
        except Exception as e:
            print(f"Error extracting links: {str(e)}")
            return []
    
    def _calculate_url_score(self, url: str) -> float:
        """
        Calculate a relevance score for a URL based on keywords.
        
        Args:
            url: The URL to score
            
        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        if url in self.url_scores:
            return self.url_scores[url]
        
        if not self.keywords:
            return 0.5  # Default score when no keywords
        
        # Score based on URL path and keywords
        url_lower = url.lower()
        score = 0.0
        
        # Check if keywords appear in the URL
        for keyword in self.keywords:
            if keyword in url_lower:
                score += 0.3  # Base score for keyword match
        
        # Bonus for certain URL patterns that typically contain content
        content_indicators = ['about', 'info', 'details', 'article', 'blog', 'news', 'page']
        for indicator in content_indicators:
            if indicator in url_lower:
                score += 0.1
        
        # Penalty for certain URL patterns that are less likely to contain useful content
        penalty_patterns = ['login', 'register', 'cart', 'checkout', 'admin', 'api', '.css', '.js', '.png', '.jpg']
        for pattern in penalty_patterns:
            if pattern in url_lower:
                score -= 0.2
        
        # Normalize score
        score = max(0.0, min(1.0, score))
        
        # Cache the score
        self.url_scores[url] = score
        
        return score

    async def extract(self, url: str, html: str, config=None) -> Dict[str, Any]:
        """
        Extract data from HTML based on best-first strategy approach.
        
        Args:
            url: The URL being processed
            html: The HTML content to extract from
            config: Optional extraction configuration
            
        Returns:
            Dictionary containing extracted data
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else ""
            
            # Count keyword matches in content
            text_content = soup.get_text().lower()
            keyword_matches = []
            for keyword in self.keywords:
                if keyword in text_content:
                    keyword_matches.append(keyword)
            
            # Simple extraction result
            data = {
                "title": title,
                "url": url,
                "keyword_matches": keyword_matches
            }
            
            # Calculate confidence based on keyword matches
            confidence = min(0.8, 0.4 + (0.1 * len(keyword_matches)))
            
            return {
                "data": data,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Error extracting data from {url}: {str(e)}")
            return {"data": {}, "confidence": 0.0}

    def crawl(self, start_url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Crawl from a starting URL using this strategy.
        
        Args:
            start_url: The URL to start crawling from
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional dictionary with crawl results, or None if crawl failed
        """
        # For BestFirstStrategy, crawl is the same as execute
        return self.execute(start_url, **kwargs)

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all results collected by this strategy.
        
        Returns:
            List of dictionaries containing the collected results
        """
        return self.results