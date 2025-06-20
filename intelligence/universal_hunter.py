"""
Universal Intelligent Hunter System

This module implements a site-agnostic, intelligent hunting system that:
1. Understands user query intent (what are they looking for?)
2. Classifies page types (index/listing vs content/article)
3. Navigates adaptively to find actual content
4. Extracts targeted, high-quality content matching user intent
5. Validates and scores results for relevance and quality

This replaces basic scraping with intelligent, targeted hunting.
"""

import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import hashlib

from bs4 import BeautifulSoup
import spacy
from spacy.tokens import Doc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UniversalHunter")

@dataclass
class HuntingIntent:
    """Structured representation of what the user is hunting for"""
    target_type: str  # news, products, services, people, events, etc.
    content_category: str  # tech, automotive, finance, etc.
    temporal_preference: str  # latest, recent, historical, etc.
    specificity: str  # specific, general, comparative
    entities: List[str]  # key entities mentioned
    keywords: List[str]  # important keywords
    confidence: float  # confidence in intent analysis

@dataclass
class PageClassification:
    """Classification of a page's purpose and structure"""
    page_type: str  # index, article, product, profile, search_results, etc.
    navigation_depth: int  # how deep in site hierarchy
    content_richness: float  # 0-1 score of content density
    link_density: float  # 0-1 score of outbound link density  
    has_listings: bool  # contains list of items/articles
    main_content_selector: str  # CSS selector for main content
    article_link_selectors: List[str]  # selectors for finding article links
    confidence: float  # confidence in classification

@dataclass
class HuntingTarget:
    """A specific target found during hunting"""
    url: str
    title: str
    content_preview: str
    relevance_score: float
    quality_score: float
    content_type: str
    extraction_method: str
    metadata: Dict[str, Any]

class UniversalIntentAnalyzer:
    """Analyzes user queries to understand hunting intent"""
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model
        if not self.nlp:
            try:
                self.nlp = spacy.load("en_core_web_lg")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.warning("Using small spaCy model - intent analysis may be limited")
    
    def analyze_intent(self, query: str) -> HuntingIntent:
        """Analyze query to extract structured hunting intent"""
        doc = self.nlp(query.lower())
        
        # Extract entities
        entities = [ent.text for ent in doc.ents]
        
        # Extract keywords (important nouns and adjectives)
        keywords = [token.lemma_ for token in doc 
                   if token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and not token.is_stop]
        
        # Determine target type based on query patterns
        target_type = self._classify_target_type(query, doc, entities)
        
        # Determine content category
        content_category = self._classify_content_category(query, entities, keywords)
        
        # Determine temporal preference
        temporal_preference = self._extract_temporal_preference(query, doc)
        
        # Determine specificity
        specificity = self._assess_specificity(query, entities, keywords)
        
        # Calculate confidence
        confidence = self._calculate_intent_confidence(doc, entities, keywords)
        
        return HuntingIntent(
            target_type=target_type,
            content_category=content_category,
            temporal_preference=temporal_preference,
            specificity=specificity,
            entities=entities,
            keywords=keywords,
            confidence=confidence
        )
    
    def _classify_target_type(self, query: str, doc: Doc, entities: List[str]) -> str:
        """Classify what type of content the user is hunting for"""
        query_lower = query.lower()
        
        # News/articles indicators
        news_indicators = ['news', 'article', 'story', 'report', 'update', 'latest', 'breaking']
        if any(indicator in query_lower for indicator in news_indicators):
            return 'news'
        
        # Product indicators
        product_indicators = ['buy', 'purchase', 'price', 'cost', 'sale', 'deal', 'review', 'specs']
        if any(indicator in query_lower for indicator in product_indicators):
            return 'product'
        
        # Service indicators
        service_indicators = ['service', 'provider', 'company', 'business', 'hire', 'contact']
        if any(indicator in query_lower for indicator in service_indicators):
            return 'service'
        
        # Event indicators
        event_indicators = ['event', 'conference', 'meeting', 'schedule', 'calendar', 'when']
        if any(indicator in query_lower for indicator in event_indicators):
            return 'event'
        
        # Person/profile indicators
        person_indicators = ['who is', 'profile', 'biography', 'about', 'background']
        if any(indicator in query_lower for indicator in person_indicators):
            return 'person'
        
        # Research/information indicators
        research_indicators = ['how to', 'tutorial', 'guide', 'learn', 'information', 'explain']
        if any(indicator in query_lower for indicator in research_indicators):
            return 'information'
        
        # Default to news if temporal words present, otherwise information
        temporal_words = ['latest', 'recent', 'new', 'current', 'today', 'yesterday']
        if any(word in query_lower for word in temporal_words):
            return 'news'
        
        return 'information'
    
    def _classify_content_category(self, query: str, entities: List[str], keywords: List[str]) -> str:
        """Classify the content category/domain"""
        query_lower = query.lower()
        all_terms = query_lower + ' ' + ' '.join(entities) + ' ' + ' '.join(keywords)
        
        # Technology
        tech_terms = ['tesla', 'apple', 'google', 'microsoft', 'ai', 'tech', 'software', 'app', 'internet']
        if any(term in all_terms for term in tech_terms):
            return 'technology'
        
        # Automotive
        auto_terms = ['car', 'vehicle', 'automotive', 'tesla', 'bmw', 'mercedes', 'toyota', 'model']
        if any(term in all_terms for term in auto_terms):
            return 'automotive'
        
        # Finance
        finance_terms = ['stock', 'market', 'finance', 'investment', 'money', 'bank', 'economy']
        if any(term in all_terms for term in finance_terms):
            return 'finance'
        
        # Health
        health_terms = ['health', 'medical', 'doctor', 'medicine', 'treatment', 'disease', 'wellness']
        if any(term in all_terms for term in health_terms):
            return 'health'
        
        # Sports
        sports_terms = ['sport', 'game', 'team', 'player', 'match', 'score', 'championship']
        if any(term in all_terms for term in sports_terms):
            return 'sports'
        
        return 'general'
    
    def _extract_temporal_preference(self, query: str, doc: Doc) -> str:
        """Extract temporal preference from query"""
        query_lower = query.lower()
        
        # Latest/breaking
        if any(word in query_lower for word in ['latest', 'breaking', 'recent', 'new', 'current']):
            return 'latest'
        
        # Today/yesterday
        if any(word in query_lower for word in ['today', 'yesterday', 'this week']):
            return 'recent'
        
        # Historical
        if any(word in query_lower for word in ['history', 'past', 'historical', 'old', 'archive']):
            return 'historical'
        
        return 'any'
    
    def _assess_specificity(self, query: str, entities: List[str], keywords: List[str]) -> str:
        """Assess how specific vs general the query is"""
        # Count specific indicators
        specificity_score = 0
        
        # Named entities increase specificity
        specificity_score += len(entities) * 2
        
        # Specific keywords
        specificity_score += len([k for k in keywords if len(k) > 4])
        
        # Specific phrases
        specific_phrases = ['model 3', 'iphone 15', 'version 2.0', 'series x']
        if any(phrase in query.lower() for phrase in specific_phrases):
            specificity_score += 3
        
        if specificity_score >= 5:
            return 'specific'
        elif specificity_score >= 2:
            return 'moderate'
        else:
            return 'general'
    
    def _calculate_intent_confidence(self, doc: Doc, entities: List[str], keywords: List[str]) -> float:
        """Calculate confidence in intent analysis"""
        confidence = 0.5  # Base confidence
        
        # More entities = higher confidence
        confidence += min(0.3, len(entities) * 0.1)
        
        # More meaningful keywords = higher confidence
        confidence += min(0.2, len(keywords) * 0.02)
        
        # Clear sentence structure = higher confidence
        if len(doc) > 3 and any(token.dep_ == 'ROOT' for token in doc):
            confidence += 0.1
        
        return min(1.0, confidence)

class UniversalPageClassifier:
    """Classifies pages to understand their structure and purpose"""
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model
    
    def classify_page(self, html: str, url: str) -> PageClassification:
        """Classify a page's type and structure"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Basic metrics
        text_content = soup.get_text()
        links = soup.find_all('a', href=True)
        
        # Calculate metrics
        content_length = len(text_content.strip())
        link_count = len(links)
        link_density = link_count / max(content_length / 100, 1)
        
        # Determine page type
        page_type = self._determine_page_type(soup, url, text_content, links)
        
        # Calculate navigation depth
        navigation_depth = self._calculate_navigation_depth(url)
        
        # Calculate content richness
        content_richness = self._calculate_content_richness(soup, text_content)
        
        # Check for listings
        has_listings = self._detect_listings(soup)
        
        # Find main content selector
        main_content_selector = self._find_main_content_selector(soup)
        
        # Find article link selectors
        article_link_selectors = self._find_article_link_selectors(soup, page_type)
        
        # Calculate confidence
        confidence = self._calculate_classification_confidence(
            page_type, content_richness, has_listings, len(article_link_selectors)
        )
        
        return PageClassification(
            page_type=page_type,
            navigation_depth=navigation_depth,
            content_richness=content_richness,
            link_density=link_density,
            has_listings=has_listings,
            main_content_selector=main_content_selector,
            article_link_selectors=article_link_selectors,
            confidence=confidence
        )
    
    def _determine_page_type(self, soup: BeautifulSoup, url: str, text_content: str, links: List) -> str:
        """Determine the primary type of the page"""
        url_lower = url.lower()
        text_lower = text_content.lower()
        
        # Check URL patterns first
        if any(pattern in url_lower for pattern in ['/category/', '/tag/', '/archive/', '/list']):
            return 'index'
        
        if any(pattern in url_lower for pattern in ['/article/', '/post/', '/news/', '/story/']):
            return 'article'
        
        if any(pattern in url_lower for pattern in ['/product/', '/item/', '/p/']):
            return 'product'
        
        # Check content patterns
        # High link density suggests index page
        if len(links) > 20 and len(text_content) < 5000:
            return 'index'
        
        # Look for article indicators
        article_indicators = soup.find_all(['article', 'main', '.post', '.article-content'])
        if article_indicators and len(text_content) > 1000:
            return 'article'
        
        # Look for product indicators
        product_indicators = soup.find_all(attrs={'class': re.compile(r'price|buy|cart|product')})
        if product_indicators:
            return 'product'
        
        # Look for profile indicators
        if any(indicator in text_lower for indicator in ['biography', 'about me', 'profile', 'background']):
            return 'profile'
        
        # Look for search results
        if any(indicator in text_lower for indicator in ['search results', 'results for', 'found']):
            return 'search_results'
        
        # Default classification based on content length and structure
        if len(text_content) > 2000 and len(links) < 10:
            return 'article'
        elif len(links) > 10:
            return 'index'
        else:
            return 'general'
    
    def _calculate_navigation_depth(self, url: str) -> int:
        """Calculate how deep in the site hierarchy this page is"""
        path = urlparse(url).path
        return len([p for p in path.split('/') if p])
    
    def _calculate_content_richness(self, soup: BeautifulSoup, text_content: str) -> float:
        """Calculate how content-rich the page is (0-1)"""
        # Base score from text length
        text_score = min(1.0, len(text_content) / 5000)
        
        # Bonus for structured content
        structure_bonus = 0
        if soup.find_all(['h1', 'h2', 'h3']):
            structure_bonus += 0.1
        if soup.find_all(['p']) and len(soup.find_all(['p'])) > 3:
            structure_bonus += 0.1
        if soup.find_all(['img']):
            structure_bonus += 0.05
        
        return min(1.0, text_score + structure_bonus)
    
    def _detect_listings(self, soup: BeautifulSoup) -> bool:
        """Detect if page contains listings/repeated content"""
        # Look for repeated structures
        potential_containers = soup.find_all(['div', 'article', 'section', 'li'])
        
        # Group by similar class names
        class_groups = {}
        for element in potential_containers:
            classes = element.get('class', [])
            if classes:
                class_key = ' '.join(sorted(classes))
                class_groups[class_key] = class_groups.get(class_key, 0) + 1
        
        # If we have multiple elements with same classes, likely a listing
        for count in class_groups.values():
            if count >= 3:
                return True
        
        # Look for semantic listing indicators
        listing_selectors = [
            '.post', '.article', '.item', '.product', '.entry',
            '[class*="post"]', '[class*="article"]', '[class*="item"]'
        ]
        
        for selector in listing_selectors:
            elements = soup.select(selector)
            if len(elements) >= 3:
                return True
        
        return False
    
    def _find_main_content_selector(self, soup: BeautifulSoup) -> str:
        """Find the CSS selector for the main content area"""
        # Try semantic selectors first
        semantic_selectors = ['main', 'article', '[role="main"]', '#main', '#content', '.content']
        
        for selector in semantic_selectors:
            element = soup.select_one(selector)
            if element and len(element.get_text().strip()) > 200:
                return selector
        
        # Try to find the largest content block
        content_candidates = soup.find_all(['div', 'section'])
        if content_candidates:
            largest = max(content_candidates, key=lambda x: len(x.get_text()))
            if largest.get('id'):
                return f"#{largest['id']}"
            elif largest.get('class'):
                return f".{largest['class'][0]}"
        
        return 'body'  # Fallback
    
    def _find_article_link_selectors(self, soup: BeautifulSoup, page_type: str) -> List[str]:
        """Find selectors for article/content links on index pages"""
        if page_type not in ['index', 'search_results']:
            return []
        
        selectors = []
        
        # Look for links within likely article containers
        article_containers = soup.select('.post, .article, .item, .entry, [class*="post"], [class*="article"]')
        
        for container in article_containers[:5]:  # Check first 5
            # Find title links
            title_links = container.find_all('a', href=True)
            for link in title_links:
                if link.find(['h1', 'h2', 'h3', 'h4']) or 'title' in ' '.join(link.get('class', [])).lower():
                    # Generate selector for this type of link
                    if container.get('class'):
                        container_class = container['class'][0]
                        selectors.append(f".{container_class} a")
                        break
        
        # Common article link patterns
        common_patterns = [
            'h2 a', 'h3 a', '.title a', '.headline a',
            'article a', '.post-title a', '.entry-title a'
        ]
        
        for pattern in common_patterns:
            if soup.select(pattern):
                selectors.append(pattern)
        
        return list(set(selectors))  # Remove duplicates
    
    def _calculate_classification_confidence(self, page_type: str, content_richness: float, 
                                          has_listings: bool, article_link_count: int) -> float:
        """Calculate confidence in page classification"""
        confidence = 0.6  # Base confidence
        
        # Strong indicators boost confidence
        if page_type == 'index' and has_listings and article_link_count > 0:
            confidence += 0.3
        elif page_type == 'article' and content_richness > 0.5:
            confidence += 0.3
        elif page_type == 'product' and content_richness > 0.3:
            confidence += 0.2
        
        return min(1.0, confidence)

class UniversalHunter:
    """Main hunting system that coordinates intent analysis, page classification, and content extraction"""
    
    def __init__(self, session=None, nlp_model=None):
        self.session = session
        self.intent_analyzer = UniversalIntentAnalyzer(nlp_model)
        self.page_classifier = UniversalPageClassifier(nlp_model)
        self.nlp = nlp_model or self.intent_analyzer.nlp
        
        # Hunting statistics
        self.pages_analyzed = 0
        self.targets_found = 0
        self.navigation_hops = 0
    
    async def hunt(self, query: str, urls: List[str], max_targets: int = 5, direct_urls: bool = True) -> List[HuntingTarget]:
        """
        Main hunting method that intelligently searches for content matching user intent
        
        Args:
            query: User's hunting query
            urls: Starting URLs to hunt from
            max_targets: Maximum number of targets to return
            direct_urls: If True, treat URLs as user-provided direct URLs (more lenient extraction)
            
        Returns:
            List of HuntingTarget objects ranked by relevance and quality
        """
        logger.info(f"🎯 Starting intelligent hunt for: '{query}'")
        
        # Step 1: Analyze user intent
        intent = self.intent_analyzer.analyze_intent(query)
        logger.info(f"🧠 Intent analysis: {intent.target_type} in {intent.content_category} domain")
        logger.info(f"   Temporal: {intent.temporal_preference}, Specificity: {intent.specificity}")
        logger.info(f"   Entities: {intent.entities[:3]}...")
        
        # Step 2: Hunt through URLs
        targets = []
        for url in urls[:10]:  # Limit initial URLs
            page_targets = await self._hunt_page(url, intent, query, is_direct_url=direct_urls)
            targets.extend(page_targets)
            
            if len(targets) >= max_targets * 2:  # Get extra for ranking
                break
        
        # Step 3: Rank and filter targets
        ranked_targets = self._rank_targets(targets, intent, query)
        
        logger.info(f"🏆 Hunt completed: Found {len(ranked_targets)} high-quality targets")
        logger.info(f"📊 Stats: {self.pages_analyzed} pages analyzed, {self.navigation_hops} navigation hops")
        
        return ranked_targets[:max_targets]
    
    async def _hunt_page(self, url: str, intent: HuntingIntent, query: str, is_direct_url: bool = False) -> List[HuntingTarget]:
        """Hunt a specific page for targets"""
        self.pages_analyzed += 1
        targets = []
        
        try:
            # Get page content
            if self.session:
                async with self.session.get(url, timeout=10) as response:
                    if response.status != 200:
                        return targets
                    html = await response.text()
            else:
                # Fallback - in real implementation would use proper HTTP client
                return targets
            
            # Classify the page
            classification = self.page_classifier.classify_page(html, url)
            logger.info(f"📄 Page classified: {classification.page_type} (confidence: {classification.confidence:.2f})")
            
            # Hunt based on page type
            if classification.page_type == 'index' and classification.has_listings:
                targets = await self._hunt_index_page(html, url, classification, intent, query)
            elif classification.page_type in ['article', 'product', 'profile']:
                target = await self._extract_content_page(html, url, classification, intent, query, is_direct_url)
                if target:
                    targets = [target]
            else:
                # Try both approaches for ambiguous pages
                target = await self._extract_content_page(html, url, classification, intent, query, is_direct_url)
                if target and target.quality_score > 0.3:
                    targets = [target]
                else:
                    targets = await self._hunt_index_page(html, url, classification, intent, query)
            
        except Exception as e:
            logger.error(f"❌ Error hunting page {url}: {e}")
        
        return targets
    
    async def _hunt_index_page(self, html: str, base_url: str, classification: PageClassification,
                              intent: HuntingIntent, query: str) -> List[HuntingTarget]:
        """Hunt an index/listing page by following links to content"""
        targets = []
        soup = BeautifulSoup(html, 'html.parser')
        
        logger.info(f"🔍 Hunting index page with {len(classification.article_link_selectors)} link patterns")
        
        # Extract article links
        article_links = []
        for selector in classification.article_link_selectors:
            links = soup.select(selector)
            for link in links[:5]:  # Limit per selector
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    title = link.get_text().strip() or link.get('title', '')
                    
                    # Filter links based on intent relevance
                    if self._is_link_relevant(title, full_url, intent):
                        article_links.append((full_url, title))
        
        logger.info(f"🔗 Found {len(article_links)} relevant article links")
        
        # Hunt the most promising article links
        for article_url, link_title in article_links[:8]:  # Limit navigation
            self.navigation_hops += 1
            
            try:
                if self.session:
                    async with self.session.get(article_url, timeout=8) as response:
                        if response.status == 200:
                            article_html = await response.text()
                            article_classification = self.page_classifier.classify_page(article_html, article_url)
                            
                            target = await self._extract_content_page(
                                article_html, article_url, article_classification, intent, query, False  # Links from index pages are not direct URLs
                            )
                            
                            if target and target.quality_score > 0.4:
                                targets.append(target)
                                self.targets_found += 1
                                
                                if len(targets) >= 5:  # Limit targets per index page
                                    break
                
            except Exception as e:
                logger.debug(f"Error hunting article {article_url}: {e}")
                continue
        
        return targets
    
    async def _extract_content_page(self, html: str, url: str, classification: PageClassification,
                                   intent: HuntingIntent, query: str, is_direct_url: bool = False) -> Optional[HuntingTarget]:
        """Extract content from a content page (article, product, etc.)
        
        Args:
            is_direct_url: If True, use more lenient thresholds for user-provided URLs
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove navigation/noise
        for element in soup(['nav', 'footer', 'header', 'aside', 'script', 'style']):
            element.decompose()
        
        # Extract main content
        main_content = soup.select_one(classification.main_content_selector)
        if not main_content:
            main_content = soup
        
        # Extract text content
        raw_text = main_content.get_text().strip()
        text_content = self._clean_extracted_text(raw_text)
        
        if len(text_content) < 100:  # Too little content after cleaning
            return None
        
        # Extract title
        title = self._extract_title(soup)
        
        # Extract metadata
        metadata = self._extract_metadata(soup, classification.page_type)
        
        # Score relevance and quality
        relevance_score = self._calculate_relevance_score(text_content, title, intent, query)
        quality_score = self._calculate_quality_score(text_content, title, metadata, classification)
        
        # Only return high-quality, relevant content
        # Use more lenient thresholds for direct URLs from user
        if is_direct_url:
            # For direct URLs, be more lenient - user explicitly provided this URL
            relevance_threshold = 0.1  # Much lower relevance requirement
            quality_threshold = 0.1    # Lower quality requirement
        else:
            # For discovered URLs, maintain strict thresholds
            relevance_threshold = 0.3
            quality_threshold = 0.2
            
        if relevance_score < relevance_threshold or quality_score < quality_threshold:
            return None
        
        return HuntingTarget(
            url=url,
            title=title,
            content_preview=text_content[:500],
            relevance_score=relevance_score,
            quality_score=quality_score,
            content_type=classification.page_type,
            extraction_method='intelligent_hunting',
            metadata=metadata
        )
    
    def _is_link_relevant(self, link_text: str, link_url: str, intent: HuntingIntent) -> bool:
        """Check if a link is relevant to the hunting intent"""
        text_lower = link_text.lower()
        url_lower = link_url.lower()
        combined = text_lower + ' ' + url_lower
        
        # Check for entity matches
        entity_matches = sum(1 for entity in intent.entities 
                           if entity.lower() in combined)
        
        # Check for keyword matches
        keyword_matches = sum(1 for keyword in intent.keywords 
                            if keyword.lower() in combined)
        
        # Calculate relevance score
        relevance = (entity_matches * 0.4 + keyword_matches * 0.2) / max(1, len(intent.entities) + len(intent.keywords))
        
        # Temporal filtering
        if intent.temporal_preference == 'latest':
            temporal_words = ['latest', 'new', 'recent', '2024', '2025']
            if any(word in combined for word in temporal_words):
                relevance += 0.2
        
        return relevance > 0.3
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the best title from the page"""
        # Try different title sources in order of preference
        title_selectors = ['h1', 'title', '.title', '.headline', '.post-title', '.entry-title']
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                raw_title = element.get_text().strip()
                if raw_title and len(raw_title) > 5:
                    # Clean the title
                    title = self._clean_extracted_text(raw_title)
                    # For titles, we want single line
                    title = title.replace('\n', ' ').strip()
                    if title and len(title) > 5:
                        return title
        
        return "Untitled Content"
    
    def _extract_metadata(self, soup: BeautifulSoup, content_type: str) -> Dict[str, Any]:
        """Extract metadata appropriate for the content type"""
        metadata = {}
        
        # Date/time
        date_selectors = ['time', '.date', '.published', '[datetime]']
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                metadata['date'] = element.get('datetime') or element.get_text().strip()
                break
        
        # Author
        author_selectors = ['.author', '[rel="author"]', '.byline']
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                metadata['author'] = element.get_text().strip()
                break
        
        # Content-specific metadata
        if content_type == 'product':
            # Price
            price_selectors = ['.price', '[data-price]', '.cost']
            for selector in price_selectors:
                element = soup.select_one(selector)
                if element:
                    metadata['price'] = element.get_text().strip()
                    break
        
        return metadata
    
    def _calculate_relevance_score(self, content: str, title: str, intent: HuntingIntent, query: str) -> float:
        """Calculate how relevant content is to the hunting intent"""
        if not self.nlp:
            return 0.5
        
        # Create document from content
        content_doc = self.nlp(content[:1000])  # Limit for performance
        title_doc = self.nlp(title)
        query_doc = self.nlp(query)
        
        # Calculate semantic similarity
        content_similarity = content_doc.similarity(query_doc) if hasattr(content_doc, 'similarity') else 0.3
        title_similarity = title_doc.similarity(query_doc) if hasattr(title_doc, 'similarity') else 0.3
        
        # Entity matching
        content_lower = content.lower()
        entity_score = sum(1 for entity in intent.entities if entity.lower() in content_lower)
        entity_score = min(1.0, entity_score / max(1, len(intent.entities)))
        
        # Keyword matching
        keyword_score = sum(1 for keyword in intent.keywords if keyword.lower() in content_lower)
        keyword_score = min(1.0, keyword_score / max(1, len(intent.keywords)))
        
        # Combine scores
        relevance = (
            content_similarity * 0.4 +
            title_similarity * 0.3 +
            entity_score * 0.2 +
            keyword_score * 0.1
        )
        
        return min(1.0, relevance)
    
    def _calculate_quality_score(self, content: str, title: str, metadata: Dict[str, Any], 
                                classification: PageClassification) -> float:
        """Calculate content quality score"""
        quality = 0.0
        
        # Content length (sweet spot around 1000-5000 chars)
        content_length = len(content)
        if content_length < 200:
            length_score = content_length / 200
        elif content_length > 5000:
            length_score = 0.8
        else:
            length_score = min(1.0, content_length / 1000)
        
        quality += length_score * 0.3
        
        # Title quality
        title_score = min(1.0, len(title) / 50) if title and title != "Untitled Content" else 0.1
        quality += title_score * 0.2
        
        # Metadata richness
        metadata_score = min(1.0, len(metadata) / 3)
        quality += metadata_score * 0.2
        
        # Classification confidence
        quality += classification.confidence * 0.2
        
        # Content richness from classification
        quality += classification.content_richness * 0.1
        
        return min(1.0, quality)
    
    def _rank_targets(self, targets: List[HuntingTarget], intent: HuntingIntent, query: str) -> List[HuntingTarget]:
        """Rank targets by combined relevance and quality scores"""
        # Calculate combined scores
        for target in targets:
            # Weight relevance higher for specific queries, quality higher for general queries
            if intent.specificity == 'specific':
                target.combined_score = target.relevance_score * 0.7 + target.quality_score * 0.3
            else:
                target.combined_score = target.relevance_score * 0.5 + target.quality_score * 0.5
        
        # Sort by combined score
        ranked = sorted(targets, key=lambda t: t.combined_score, reverse=True)
        
        # Remove near-duplicates
        deduplicated = []
        seen_content_hashes = set()
        
        for target in ranked:
            # Create content hash for deduplication
            content_hash = hashlib.md5(
                (target.title + target.content_preview[:200]).encode()
            ).hexdigest()
            
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                deduplicated.append(target)
        
        return deduplicated

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text content"""
        import re
        
        if not text:
            return ""
        
        # Remove common HTML entities and artifacts
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Remove multiple consecutive whitespace/newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)   # More than 2 newlines to 2 newlines
        
        # Remove common web artifacts
        patterns_to_remove = [
            r'Share\s*Tweet\s*',
            r'(Image credit:.*?)\n',
            r'(Photo credit:.*?)\n',
            r'(Credit:.*?)\n\n\n+',
            r'Subscribe\s*',
            r'Sign up\s*',
            r'Newsletter\s*',
            r'Follow us\s*',
            r'Read more\s*',
            r'Loading\.\.\.\s*',
            r'Advertisement\s*',
            r'Sponsored\s*',
            r'Click here.*?\s*',
            r'\s*\|\s*[A-Z][a-z]{2}\s+\d{1,2}\s+\d{4}\s*-\s*\d{1,2}:\d{2}\s+[ap]m\s+[A-Z]{2,3}\s*',  # Date patterns
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up navigation and menu artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip very short lines that are likely navigation
            if len(line) < 3:
                continue
                
            # Skip lines that are just navigation words
            nav_words = ['home', 'about', 'contact', 'login', 'register', 'menu', 'search']
            if line.lower() in nav_words:
                continue
                
            # Skip lines with only symbols or numbers
            if re.match(r'^[\s\-_=\+\*\.#]+$', line):
                continue
                
            # Skip very repetitive lines (likely formatting artifacts)
            if len(set(line.replace(' ', ''))) < 3 and len(line) > 10:
                continue
                
            cleaned_lines.append(line)
        
        # Reconstruct text
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        # If the text starts with a title-like pattern, clean it up
        lines = text.split('\n', 2)
        if len(lines) >= 2 and len(lines[0]) > 0:
            # Remove redundant title repetition
            if lines[0].lower() in text[len(lines[0]):].lower():
                text = '\n'.join(lines[1:])
        
        return text
