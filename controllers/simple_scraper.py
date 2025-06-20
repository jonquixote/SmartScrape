"""
Simple single-pass scraper that eliminates recursive loops
Enhanced with hybrid deep extraction using ExtractionCoordinator and Crawl4AI
Universal Intelligent Content Extraction using Advanced Content Analysis + spaCy
"""
import asyncio
import re
import hashlib
import dateutil.parser
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from duckduckgo_search import DDGS
import aiohttp
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

# Import deep extraction components
try:
    from controllers.extraction_coordinator import get_extraction_coordinator
    EXTRACTION_COORDINATOR_AVAILABLE = True
except ImportError:
    EXTRACTION_COORDINATOR_AVAILABLE = False
    logger.warning("ExtractionCoordinator not available")

try:
    from strategies.universal_crawl4ai_strategy import UniversalCrawl4AIStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("UniversalCrawl4AIStrategy not available")

# Import advanced content analysis components
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("spaCy available for intelligent content analysis")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, intelligent analysis will be limited")

try:
    from extraction.quality_evaluator import ExtractedDataQualityEvaluator
    QUALITY_EVALUATOR_AVAILABLE = True
except ImportError:
    QUALITY_EVALUATOR_AVAILABLE = False
    logger.warning("ExtractedDataQualityEvaluator not available")

try:
    from processors.content_quality_scorer import ContentQualityScorer
    CONTENT_QUALITY_SCORER_AVAILABLE = True
except ImportError:
    CONTENT_QUALITY_SCORER_AVAILABLE = False
    logger.warning("ContentQualityScorer not available")

@dataclass
class ScrapingContext:
    """Single context object passed through the pipeline"""
    query: str
    target_urls: List[str] = None
    extracted_items: List[Dict] = None
    ai_service: Any = None

class SimpleScraper:
    """Single-responsibility scraper with universal intelligent content extraction"""
    
    def __init__(self):
        self.session = None
        self._last_target_urls = []  # Track URLs for web handler
        
        # Initialize deep extraction components
        self.extraction_coordinator = None
        self.crawl4ai_strategy = None
        
        # Initialize spaCy model for intelligent analysis
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_lg")
                logger.info("spaCy large model loaded for intelligent analysis")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("spaCy medium model loaded for intelligent analysis")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                        logger.info("spaCy small model loaded as fallback")
                    except OSError:
                        logger.warning("No spaCy model available")
                        self.nlp = None
        
        # Initialize quality evaluator with proper error handling
        self.quality_evaluator = None
        if QUALITY_EVALUATOR_AVAILABLE:
            try:
                self.quality_evaluator = ExtractedDataQualityEvaluator()
                logger.info("Quality evaluator initialized successfully")
            except Exception as e:
                logger.warning(f"Quality evaluator initialization failed: {e}")
                self.quality_evaluator = None
        
        # Initialize content quality scorer with proper error handling
        self.content_quality_scorer = None
        if CONTENT_QUALITY_SCORER_AVAILABLE:
            try:
                self.content_quality_scorer = ContentQualityScorer()
                logger.info("Content quality scorer initialized successfully")
            except Exception as e:
                logger.warning(f"Content quality scorer initialization failed: {e}")
                self.content_quality_scorer = None
        
        # Initialize extraction coordinator if available
        if EXTRACTION_COORDINATOR_AVAILABLE:
            try:
                self.extraction_coordinator = get_extraction_coordinator()
                logger.info("ExtractionCoordinator initialized for deep extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize ExtractionCoordinator: {e}")
                self.extraction_coordinator = None
        
        # Initialize Crawl4AI strategy if available
        if CRAWL4AI_AVAILABLE:
            try:
                self.crawl4ai_strategy = UniversalCrawl4AIStrategy()
                logger.info("UniversalCrawl4AIStrategy initialized as fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize UniversalCrawl4AIStrategy: {e}")
                self.crawl4ai_strategy = None
        
        # Log capabilities
        capabilities = []
        if self.nlp: capabilities.append("spaCy NLP")
        if self.quality_evaluator: capabilities.append("Quality Evaluation")
        if self.content_quality_scorer: capabilities.append("Content Quality Scoring")
        if self.extraction_coordinator: capabilities.append("Deep Extraction")
        if self.crawl4ai_strategy: capabilities.append("Crawl4AI")
        
        logger.info(f"SimpleScraper initialized with capabilities: {', '.join(capabilities) if capabilities else 'Basic HTML only'}")
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def scrape_query(self, query: str) -> Dict:
        """Main entry point - single pass through pipeline"""
        context = ScrapingContext(query=query)
        
        # Step 1: URL Discovery (ONCE)
        logger.info(f"Step 1: Discovering URLs for '{query}'")
        context.target_urls = await self._discover_urls(query)
        logger.info(f"Found {len(context.target_urls)} URLs")
        
        # Store for reference in handler
        self._last_target_urls = context.target_urls.copy()
        
        # Step 2: Content Extraction (ONCE per URL)
        logger.info("Step 2: Extracting content from URLs")
        context.extracted_items = []
        for url in context.target_urls[:5]:  # Limit to 5 URLs
            try:
                items = await self._extract_from_url(url, query)
                context.extracted_items.extend(items)
                logger.info(f"Extracted {len(items)} items from {url}")
            except Exception as e:
                logger.error(f"Failed to extract from {url}: {e}")
                
        # Step 3: Result Processing (ONCE)
        logger.info("Step 3: Processing and consolidating results")
        return self._consolidate_results(context)
        
    async def _discover_urls(self, query: str) -> List[str]:
        """URL discovery - NO recursive calls"""
        urls = []
        try:
            # Use real DuckDuckGo search
            ddgs = DDGS()
            results = ddgs.text(query, max_results=10)
            
            for result in results:
                url = result.get('href', '')
                if url and self._is_valid_url(url):
                    urls.append(url)
                    
        except Exception as e:
            logger.error(f"URL discovery failed: {e}")
            # Fallback to predefined Tesla news sites
            urls = [
                "https://www.teslarati.com/",
                "https://electrek.co/",
                "https://www.reuters.com/business/autos-transportation/",
            ]
            
        return urls
        
    def _is_valid_url(self, url: str) -> bool:
        """Filter out fake/invalid URLs"""
        invalid_patterns = [
            'duckduckgo.com',
            'lite.duckduckgo.com',
            'html.duckduckgo.com',
            'javascript:',
            'mailto:',
        ]
        return not any(pattern in url.lower() for pattern in invalid_patterns)
        
    async def _extract_from_url(self, url: str, query: str) -> List[Dict]:
        """Universal intelligent content extraction using Advanced Content Analysis + spaCy"""
        items = []
        
        # Step 1: Try Universal Intelligent Content Analysis (PRIMARY)
        intelligent_content = await self._universal_intelligent_extraction(url, query)
        if intelligent_content:
            items.extend(intelligent_content)
            logger.info(f"Universal intelligent analysis extracted {len(intelligent_content)} items from {url}")
            return items
        
        # Step 2: Try deep extraction with ExtractionCoordinator (fallback)
        deep_content = await self._deep_extract_with_coordinator(url, query)
        if deep_content:
            items.extend(deep_content)
            logger.info(f"ExtractionCoordinator extracted {len(deep_content)} items from {url}")
            return items
        
        # Step 3: Try deep extraction with Crawl4AI (fallback)
        crawl4ai_content = await self._deep_extract_with_crawl4ai(url, query)
        if crawl4ai_content:
            items.extend(crawl4ai_content)
            logger.info(f"Crawl4AI extracted {len(crawl4ai_content)} items from {url}")
            return items
        
        # Step 4: Fallback to basic HTML scraping (last resort)
        logger.info(f"Using basic HTML scraping for {url}")
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract content using basic methods
                    articles = self._extract_articles(soup, query)
                    items.extend(articles)
                    
        except Exception as e:
            logger.error(f"Basic extraction from {url} failed: {e}")
            
        return items
    
    async def _universal_intelligent_extraction(self, url: str, query: str) -> List[Dict]:
        """Universal intelligent content extraction using Advanced Content Analysis + spaCy"""
        try:
            # Check if we have the required components
            if not self.nlp:
                logger.warning("spaCy not available, falling back to simpler extraction")
                return []
            
            # Get HTML content
            async with self.session.get(url, timeout=15) as response:
                if response.status != 200:
                    return []
                    
                html = await response.text()
                
                # Step 1: Analyze site structure (universal)
                site_analysis = await self._analyze_site_structure(html, url)
                logger.info(f"Site analysis: {site_analysis.get('site_type', 'unknown')} type detected")
                
                # Step 2: Extract and score content blocks universally
                content_blocks = await self._extract_universal_content_blocks(
                    html, self.nlp, query, site_analysis
                )
                
                # Step 3: Quality evaluation and filtering (if available)
                validated_items = []
                for block in content_blocks:
                    # Calculate quality scores
                    completeness = 1.0
                    relevance = 0.5
                    confidence = 0.5
                    text_quality = 0.5
                    
                    # Use quality evaluator if available
                    if self.quality_evaluator:
                        try:
                            completeness = self.quality_evaluator.calculate_completeness_score(block['data'])
                            relevance = self.quality_evaluator.calculate_relevance_score(block['data'], query)
                            confidence = self.quality_evaluator.calculate_confidence_score(block['data'])
                        except Exception as e:
                            logger.warning(f"Quality evaluation failed: {e}")
                    
                    # Use content quality scorer if available
                    semantic_relevance = 0.5  # Default relevance score
                    if self.content_quality_scorer:
                        try:
                            text_quality = self.content_quality_scorer.score_content(block['content'], query)
                            # Calculate semantic relevance between query and content
                            semantic_relevance = self.content_quality_scorer.score_semantic_similarity(
                                query, block['content']
                            )
                        except Exception as e:
                            logger.warning(f"Content quality scoring failed: {e}")
                    
                    # Enhanced quality score with semantic relevance
                    overall_quality = (completeness * 0.25 + relevance * 0.25 + 
                                     confidence * 0.15 + text_quality * 0.15 + 
                                     semantic_relevance * 0.2)
                    
                    # Filter content by semantic relevance and overall quality
                    relevance_threshold = 0.4  # Minimum semantic relevance
                    quality_threshold = 0.35   # Minimum overall quality
                    
                    if semantic_relevance >= relevance_threshold and overall_quality >= quality_threshold:
                        validated_items.append({
                            'title': block['data'].get('title', 'Content Block'),
                            'content': block['content'][:2000],
                            'url': url,
                            'timestamp': block['data'].get('timestamp', 'Unknown date'),
                            'extraction_method': 'universal_intelligent_analysis',
                            'site_type': site_analysis.get('site_type', 'unknown'),
                            'content_type': block['data'].get('content_type', 'general'),
                            'quality_score': overall_quality,
                            'completeness': completeness,
                            'relevance': relevance,
                            'semantic_relevance': semantic_relevance,
                            'confidence': confidence,
                            'text_quality': text_quality,
                            'entities': block['data'].get('entities', []),
                            'structured_data': self._make_serializable(block['data'])
                        })
                
                # Sort by quality score and return top results
                validated_items.sort(key=lambda x: x['quality_score'], reverse=True)
                logger.info(f"Universal intelligent extraction found {len(validated_items)} quality items")
                return validated_items[:5]  # Top 5 quality items
                
        except Exception as e:
            logger.error(f"Universal intelligent extraction failed for {url}: {e}")
            return []
    
    async def _analyze_site_structure(self, html: str, url: str) -> Dict:
        """Analyze site structure to determine extraction strategy"""
        try:
            # Try to use existing site structure analysis
            from extraction.content_analysis import analyze_site_structure
            return await analyze_site_structure(html, url)
        except ImportError:
            # Fallback to basic site type detection
            return self._basic_site_type_detection(html, url)
        except Exception as e:
            logger.warning(f"Site structure analysis failed: {e}")
            return self._basic_site_type_detection(html, url)
    
    def _basic_site_type_detection(self, html: str, url: str) -> Dict:
        """Basic site type detection based on URL patterns and HTML content"""
        html_lower = html.lower()
        url_lower = url.lower()
        
        # E-commerce indicators
        if any(pattern in url_lower for pattern in ['shop', 'store', 'buy', 'cart', 'product']):
            return {'site_type': 'ecommerce', 'confidence': 0.8}
        
        if any(pattern in html_lower for pattern in ['add to cart', 'buy now', 'price', '$']):
            return {'site_type': 'ecommerce', 'confidence': 0.7}
        
        # News/Blog indicators
        if any(pattern in url_lower for pattern in ['news', 'blog', 'article', 'post']):
            return {'site_type': 'news', 'confidence': 0.8}
        
        # Social media indicators
        if any(pattern in url_lower for pattern in ['facebook', 'twitter', 'instagram', 'linkedin']):
            return {'site_type': 'social', 'confidence': 0.9}
        
        # Directory/Business listing indicators
        if any(pattern in url_lower for pattern in ['directory', 'listing', 'business', 'local']):
            return {'site_type': 'directory', 'confidence': 0.8}
        
        # Default to generic
        return {'site_type': 'generic', 'confidence': 0.5}
    
    async def _extract_universal_content_blocks(self, html: str, nlp, query: str, 
                                              site_analysis: Dict) -> List[Dict]:
        """Extract content blocks universally based on site type and semantic analysis with duplicate detection"""
        soup = BeautifulSoup(html, 'html.parser')
        content_blocks = []
        seen_content = set()  # Track seen content to avoid duplicates
        
        # Remove navigation and unwanted elements
        for element in soup(['nav', 'footer', 'header', 'aside', 'script', 'style']):
            element.decompose()
        
        site_type = site_analysis.get('site_type', 'generic')
        
        # Define universal selectors based on site type
        selectors = self._get_universal_selectors(site_type)
        
        # Process query with spaCy for semantic matching
        query_doc = nlp(query.lower()) if query else None
        
        # Extract content blocks
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements[:10]:  # Limit to prevent memory issues
                
                # Extract text content
                text_content = element.get_text().strip()
                if len(text_content) < 50:  # Skip very short content
                    continue
                
                # Create content hash for duplicate detection
                content_hash = hashlib.md5(text_content.encode()).hexdigest()
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                # Use spaCy to analyze content
                doc = nlp(text_content)
                
                # Calculate content quality score using spaCy features
                quality_score = self._calculate_spacy_content_quality(
                    doc, element, query_doc, site_type
                )
                
                # Only process high-quality content
                if quality_score > 0.4:
                    # Extract clean structured data
                    structured_data = self._extract_clean_structured_data(element, doc, site_type)
                    
                    # Create clean content block
                    content_block = {
                        'title': structured_data.get('title', self._extract_title_from_element(element)),
                        'content': self._clean_content(text_content[:1500]),  # Limit and clean
                        'author': structured_data.get('author', 'Unknown Author'),
                        'url': self._extract_url_from_element(element, soup),
                        'timestamp': self._format_timestamp(structured_data.get('timestamp')),
                        'quality_score': round(quality_score, 2),
                        'content_type': self._identify_content_type(doc, element),
                        'extraction_method': 'universal_intelligent_analysis',
                        'site_type': site_type,
                        'word_count': len(doc),
                        'language': 'en'
                    }
                    
                    content_blocks.append(content_block)
        
        # Remove duplicates and sort by quality
        unique_blocks = self._remove_duplicate_blocks(content_blocks)
        unique_blocks.sort(key=lambda x: x['quality_score'], reverse=True)
        return unique_blocks[:3]  # Return top 3 unique results
    
    def _get_universal_selectors(self, site_type: str) -> List[str]:
        """Get CSS selectors based on site type for universal content extraction"""
        
        base_selectors = [
            'article', 'main', '.content', '.main-content', 
            '[role="main"]', '[data-content]'
        ]
        
        if site_type == 'ecommerce':
            return base_selectors + [
                '.product', '.item', '.listing', '.product-item', 
                '.product-card', '[data-product]', '.product-info',
                '.product-details', '.item-details'
            ]
        elif site_type == 'news' or site_type == 'blog':
            return base_selectors + [
                '.post', '.entry', '.story', '.news-item', 
                '.article-content', '.blog-post', '.entry-content',
                '.post-content', '.article-body'
            ]
        elif site_type == 'social':
            return base_selectors + [
                '.post', '.tweet', '.update', '.status', 
                '[data-post]', '.feed-item', '.social-post'
            ]
        elif site_type == 'directory':
            return base_selectors + [
                '.listing', '.business', '.entry', '.directory-item',
                '.business-card', '.location', '.place'
            ]
        else:
            # Generic approach - comprehensive selectors
            return base_selectors + [
                '.content-block', '.text-content', '.description',
                '.summary', '.details', '.info', '.card', '.tile',
                'section', '.section', 'div[class*="content"]'
            ]
    
    def _calculate_spacy_content_quality(self, doc, element, query_doc, site_type: str) -> float:
        """Calculate content quality using spaCy NLP features"""
        score = 0.0
        
        # 1. Semantic similarity to query (if available)
        if query_doc and doc.vector_norm > 0 and query_doc.vector_norm > 0:
            try:
                similarity = doc.similarity(query_doc)
                score += similarity * 0.35
            except:
                pass  # Handle cases where similarity fails
        
        # 2. Entity density (content with entities is usually more valuable)
        entity_density = len(doc.ents) / max(len(doc), 1)
        score += min(entity_density * 0.20, 0.20)
        
        # 3. Sentence structure quality
        sentences = list(doc.sents)
        if sentences:
            avg_sentence_length = sum(len(sent) for sent in sentences) / len(sentences)
            sentence_quality = min(avg_sentence_length / 50, 1.0)  # Normalize
            score += sentence_quality * 0.15
        
        # 4. POS tag diversity (good content has varied grammar)
        pos_tags = set(token.pos_ for token in doc)
        pos_diversity = len(pos_tags) / 15  # Normalize by typical POS count
        score += min(pos_diversity * 0.10, 0.10)
        
        # 5. Content length appropriateness
        text_length = len(doc.text)
        if 100 <= text_length <= 3000:  # Sweet spot for content
            score += 0.10
        elif text_length > 50:  # At least some substance
            score += 0.05
        
        # 6. Anti-navigation/chrome detection
        navigation_penalties = [
            'click', 'subscribe', 'follow', 'share', 'menu', 'navigation',
            'copyright', 'privacy', 'terms', 'cookies', 'advertisement'
        ]
        
        element_text_lower = doc.text.lower()
        element_classes = ' '.join(element.get('class', [])).lower()
        element_id = element.get('id', '').lower()
        
        penalty_count = 0
        for penalty_term in navigation_penalties:
            if (penalty_term in element_text_lower or 
                penalty_term in element_classes or 
                penalty_term in element_id):
                penalty_count += 1
        
        score -= penalty_count * 0.05
        
        # 7. Site-type specific bonuses
        if site_type == 'ecommerce':
            # Look for product-specific entities
            product_entities = ['MONEY', 'PRODUCT', 'ORG']
            if any(ent.label_ in product_entities for ent in doc.ents):
                score += 0.10
                
        elif site_type == 'news':
            # Look for news-specific entities
            news_entities = ['PERSON', 'ORG', 'DATE', 'GPE']
            if any(ent.label_ in news_entities for ent in doc.ents):
                score += 0.10
        
        return max(0.0, min(1.0, score))
    
    def _extract_structured_data_universal(self, element, doc, site_type: str, url=None) -> Dict:
        """Extract structured data universally based on site type and spaCy analysis"""
        structured_data = {}
        
        # Universal title extraction
        title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if title_elem:
            structured_data['title'] = title_elem.get_text().strip()
        else:
            # Use first sentence or significant noun phrase as title
            sentences = list(doc.sents)
            if sentences:
                first_sentence = sentences[0].text.strip()
                if len(first_sentence) < 200:  # Reasonable title length
                    structured_data['title'] = first_sentence
                else:
                    # Extract significant noun phrases as title
                    noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 10]
                    if noun_phrases:
                        structured_data['title'] = noun_phrases[0]
        
        # Universal date/time extraction using spaCy NER
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                structured_data['timestamp'] = ent.text
                break
        
        # Universal entity extraction
        structured_data['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Universal content classification
        structured_data['content_type'] = self._classify_content_type(doc, element)
        
        # Site-type specific extractions
        if site_type == 'ecommerce':
            structured_data.update(self._extract_ecommerce_data(element, doc))
        elif site_type == 'news' or site_type == 'blog':
            structured_data.update(self._extract_news_data(element, doc))
        elif site_type == 'social':
            structured_data.update(self._extract_social_data(element, doc))
        elif site_type == 'directory':
            structured_data.update(self._extract_directory_data(element, doc))
        
        # Universal metadata
        structured_data.update({
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'entity_count': len(doc.ents),
            'language': doc.lang_ if hasattr(doc, 'lang_') else 'en'
        })
        
        return structured_data
    
    def _classify_content_type(self, doc, element) -> str:
        """Classify content type using spaCy NLP analysis"""
        # Analyze entities to determine content type
        entity_labels = [ent.label_ for ent in doc.ents]
        entity_counts = {label: entity_labels.count(label) for label in set(entity_labels)}
        
        # Product content
        if 'MONEY' in entity_counts or 'PRODUCT' in entity_counts:
            return 'product'
        
        # News/article content
        if ('PERSON' in entity_counts and 'DATE' in entity_counts and 
            entity_counts.get('PERSON', 0) > 1):
            return 'article'
        
        # Business/organization content
        if ('ORG' in entity_counts and 'GPE' in entity_counts and
            entity_counts.get('ORG', 0) > 0):
            return 'business'
        
        # Event content
        if 'EVENT' in entity_counts:
            return 'event'
        
        # Review content (look for opinion words)
        opinion_words = ['good', 'bad', 'great', 'terrible', 'amazing', 'awful', 'love', 'hate']
        if any(token.text.lower() in opinion_words for token in doc):
            return 'review'
        
        return 'general_content'
    
    def _identify_content_type(self, doc, element) -> str:
        """Identify content type - wrapper for _classify_content_type"""
        return self._classify_content_type(doc, element)
    
    def _extract_ecommerce_data(self, element, doc) -> Dict:
        """Extract e-commerce specific data"""
        data = {}
        
        # Price extraction
        import re
        price_patterns = [
            r'\$[\d,]+\.?\d*',
            r'USD\s*[\d,]+\.?\d*',
            r'Price:?\s*\$?[\d,]+\.?\d*'
        ]
        
        for pattern in price_patterns:
            price_match = re.search(pattern, element.get_text(), re.IGNORECASE)
            if price_match:
                data['price'] = price_match.group()
                break
        
        # Rating extraction
        rating_elem = element.find(class_=re.compile(r'rating|stars|score'))
        if rating_elem:
            data['rating'] = rating_elem.get_text().strip()
        
        # Brand extraction using NER
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                data['brand'] = ent.text
                break
        
        return data
    
    def _extract_news_data(self, element, doc) -> Dict:
        """Extract news/blog specific data"""
        data = {}
        
        # Author extraction
        author_elem = element.find(class_=re.compile(r'author|byline|writer'))
        if author_elem:
            data['author'] = author_elem.get_text().strip()
        else:
            # Use spaCy NER for author
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    data['author'] = ent.text
                    break
        
        # Category extraction
        category_elem = element.find(class_=re.compile(r'category|section|topic'))
        if category_elem:
            data['category'] = category_elem.get_text().strip()
        
        return data
    
    def _extract_social_data(self, element, doc) -> Dict:
        """Extract social media specific data"""
        data = {}
        
        # User/author extraction
        user_elem = element.find(class_=re.compile(r'user|username|handle'))
        if user_elem:
            data['user'] = user_elem.get_text().strip()
        
        # Engagement metrics
        likes_elem = element.find(class_=re.compile(r'likes|hearts|reactions'))
        if likes_elem:
            data['likes'] = likes_elem.get_text().strip()
        
        return data
    
    def _extract_directory_data(self, element, doc) -> Dict:
        """Extract directory/business listing specific data"""
        data = {}
        
        # Contact information
        contact_elem = element.find(class_=re.compile(r'contact|phone|email'))
        if contact_elem:
            data['contact'] = contact_elem.get_text().strip()
        
        # Address using NER
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                data['location'] = ent.text
                break
        
        return data
    
    async def _deep_extract_with_coordinator(self, url: str, query: str) -> List[Dict]:
        """Deep extraction using ExtractionCoordinator"""
        if not self.extraction_coordinator:
            return []
        
        try:
            logger.info(f"Attempting deep extraction with coordinator for {url}")
            result = await self.extraction_coordinator.extract_with_intelligent_selection(url, query=query)
            
            if result and result.get('items'):
                # Convert to our format
                converted_items = []
                for item in result['items'][:5]:  # Limit to 5 items
                    converted_items.append({
                        'title': item.get('title', 'Deep Extracted Content'),
                        'content': item.get('content', '')[:2000],
                        'url': url,
                        'timestamp': item.get('timestamp', 'Unknown date'),
                        'extraction_method': 'extraction_coordinator',
                        'structured_data': self._make_serializable(item)
                    })
                return converted_items
        except Exception as e:
            logger.error(f"ExtractionCoordinator failed for {url}: {e}")
            
        return []
    
    async def _deep_extract_with_crawl4ai(self, url: str, query: str) -> List[Dict]:
        """Deep extraction using Crawl4AI"""
        if not self.crawl4ai_strategy:
            return []
        
        try:
            logger.info(f"Attempting deep extraction with Crawl4AI for {url}")
            
            # Get HTML content first
            async with self.session.get(url, timeout=15) as response:
                if response.status != 200:
                    return []
                html = await response.text()
            
            # Use Crawl4AI's extract method
            result = self.crawl4ai_strategy.extract(html, url, query=query)
            
            if result and result.get('items'):
                # Convert to our format
                converted_items = []
                for item in result['items'][:5]:  # Limit to 5 items
                    converted_items.append({
                        'title': item.get('title', 'Crawl4AI Extracted Content'),
                        'content': item.get('content', '')[:2000],
                        'url': url,
                        'timestamp': item.get('timestamp', 'Unknown date'),
                        'extraction_method': 'crawl4ai',
                        'structured_data': self._make_serializable(item)
                    })
                return converted_items
        except Exception as e:
            logger.error(f"Crawl4AI failed for {url}: {e}")
            
        return []
    
    def _make_serializable(self, obj) -> Dict:
        """Convert complex objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _extract_articles(self, soup: BeautifulSoup, query: str) -> List[Dict]:
        """Basic article extraction for fallback"""
        articles = []
        
        # Common article selectors
        selectors = [
            'article',
            '.article',
            '.post',
            '.entry',
            '.content',
            '.main-content',
            'main',
            '[role="main"]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements[:3]:  # Limit to 3 elements
                text = element.get_text().strip()
                if len(text) > 100:  # Minimum content length
                    
                    # Try to extract title
                    title_elem = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    title = title_elem.get_text().strip() if title_elem else 'Basic Extracted Content'
                    
                    articles.append({
                        'title': title,
                        'content': text[:2000],
                        'url': soup.find('link', {'rel': 'canonical'})['href'] if soup.find('link', {'rel': 'canonical'}) else 'Unknown URL',
                        'timestamp': 'Unknown date',
                        'extraction_method': 'basic_html',
                        'structured_data': {'title': title, 'content_length': len(text)}
                    })
                    
        return articles
    
    def _consolidate_results(self, context: ScrapingContext) -> Dict:
        """Consolidate and format final results"""
        if not context.extracted_items:
            return {
                'query': context.query,
                'status': 'no_results',
                'items': [],
                'total_items': 0,
                'extraction_methods': [],
                'summary': f'No relevant content found for query: {context.query}'
            }
        
        # Remove duplicate content using semantic similarity
        if self.content_quality_scorer and len(context.extracted_items) > 1:
            try:
                contents = [item.get('content', '') for item in context.extracted_items]
                duplicate_groups = self.content_quality_scorer.detect_duplicate_content(contents, threshold=0.8)
                
                # Keep only the highest quality item from each duplicate group
                items_to_remove = set()
                for group in duplicate_groups:
                    # Sort group by quality score and keep the best one
                    group_items = [context.extracted_items[i] for i in group]
                    group_items.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
                    # Mark all but the best for removal
                    for i in group[1:]:  # Skip the first (best) item
                        items_to_remove.add(i)
                
                # Filter out duplicates
                context.extracted_items = [
                    item for i, item in enumerate(context.extracted_items) 
                    if i not in items_to_remove
                ]
                
                if duplicate_groups:
                    logger.info(f"Removed {len(items_to_remove)} duplicate items from {len(duplicate_groups)} groups")
            except Exception as e:
                logger.warning(f"Duplicate detection failed: {e}")
        
        # Group by extraction method
        method_counts = {}
        for item in context.extracted_items:
            method = item.get('extraction_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Sort items by quality if available
        sorted_items = sorted(
            context.extracted_items,
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )
        
        return {
            'query': context.query,
            'status': 'success',
            'items': sorted_items,
            'total_items': len(sorted_items),
            'extraction_methods': list(method_counts.keys()),
            'method_counts': method_counts,
            'summary': f'Found {len(sorted_items)} relevant items for query: {context.query}',
            'top_quality_score': sorted_items[0].get('quality_score', 0) if sorted_items else 0
        }
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content for better readability"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove navigation-like patterns
        nav_patterns = [
            r'(?i)(subscribe|follow us|share|click here|read more)',
            r'(?i)(menu|navigation|home|about|contact)',
            r'(?i)(facebook|twitter|instagram|linkedin)',
            r'(?i)(newsletter|email|sign up|join)',
        ]
        
        for pattern in nav_patterns:
            content = re.sub(pattern, '', content)
        
        # Clean up the result
        content = content.strip()
        
        # Ensure it ends properly
        if content and not content.endswith(('.', '!', '?')):
            last_sentence_end = max(
                content.rfind('.'), 
                content.rfind('!'), 
                content.rfind('?')
            )
            if last_sentence_end > len(content) * 0.8:  # If within last 20%
                content = content[:last_sentence_end + 1]
        
        return content

    def _extract_clean_structured_data(self, element, doc, site_type: str) -> Dict:
        """Extract clean structured data without noise"""
        structured_data = {}
        
        # Clean title extraction
        title_elem = element.find(['h1', 'h2', 'h3'])
        if title_elem:
            title = title_elem.get_text().strip()
            # Clean title of navigation elements
            title = re.sub(r'(?i)(home|menu|nav)', '', title).strip()
            if title and len(title) > 5:  # Ensure meaningful title
                structured_data['title'] = title
        
        # Clean author extraction using spaCy NER
        author = self._extract_clean_author(doc)
        if author:
            structured_data['author'] = author
        
        # Clean timestamp extraction
        timestamp = self._extract_clean_timestamp(element, doc)
        if timestamp:
            structured_data['timestamp'] = timestamp
        
        return structured_data

    def _extract_clean_author(self, doc) -> str:
        """Extract author name cleanly using spaCy"""
        # Look for PERSON entities near author indicators
        author_indicators = ['by ', 'author:', 'written by', 'reporter:', '@']
        
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:  # Reasonable name length
                context = doc.text[max(0, ent.start_char-30):ent.end_char+10].lower()
                if any(indicator in context for indicator in author_indicators):
                    # Validate it's a real name (not navigation text)
                    if not any(nav in ent.text.lower() for nav in ['menu', 'nav', 'home', 'click']):
                        return ent.text
        
        return None

    def _extract_clean_timestamp(self, element, doc) -> str:
        """Extract clean timestamp from element or content"""
        # Look for time elements first
        time_elem = element.find('time')
        if time_elem:
            datetime_attr = time_elem.get('datetime')
            if datetime_attr:
                return datetime_attr
            time_text = time_elem.get_text().strip()
            if time_text:
                return time_text
        
        # Look for date patterns in the text using spaCy
        for ent in doc.ents:
            if ent.label_ == "DATE" and len(ent.text) > 3:
                return ent.text
        
        return None

    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp consistently"""
        if not timestamp or timestamp == "Unknown date":
            return "Unknown date"
        
        # Try to parse and format common date patterns
        try:
            parsed_date = dateutil.parser.parse(timestamp)
            return parsed_date.strftime("%Y-%m-%d")
        except:
            return timestamp

    def _extract_title_from_element(self, element) -> str:
        """Extract title from element"""
        title_elem = element.find(['h1', 'h2', 'h3'])
        if title_elem:
            title = title_elem.get_text().strip()
            if title and len(title) > 5:
                return title
        return "Content Block"

    def _extract_url_from_element(self, element, soup) -> str:
        """Extract URL from element or use current page URL"""
        link_elem = element.find('a')
        if link_elem and link_elem.get('href'):
            href = link_elem.get('href')
            if href.startswith('http'):
                return href
        return "Current page"

    def _remove_duplicate_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Remove duplicate content blocks"""
        unique_blocks = []
        seen_titles = set()
        seen_content_hashes = set()
        
        for block in blocks:
            title = block.get('title', '')
            content = block.get('content', '')
            
            # Create content similarity hash
            content_words = set(content.lower().split())
            content_hash = hashlib.md5(''.join(sorted(content_words)).encode()).hexdigest()
            
            # Skip if we've seen very similar content
            if title in seen_titles or content_hash in seen_content_hashes:
                continue
                
            seen_titles.add(title)
            seen_content_hashes.add(content_hash)
            unique_blocks.append(block)
        
        return unique_blocks

# Handler function for web routes
async def simple_scrape_handler(query: str) -> Dict:
    """
    Simple handler function for web routes that uses the SimpleScraper
    
    Args:
        query: Search query string
        
    Returns:
        Dictionary with extraction results in expected format
    """
    try:
        async with SimpleScraper() as scraper:
            result = await scraper.scrape_query(query)
            
            # Add urls_processed field if not present
            if 'urls_processed' not in result:
                # Estimate from target_urls or items
                if hasattr(scraper, '_last_target_urls'):
                    result['urls_processed'] = scraper._last_target_urls
                else:
                    # Estimate from unique URLs in items
                    urls = set()
                    for item in result.get('items', []):
                        if 'url' in item:
                            urls.add(item['url'])
                    result['urls_processed'] = list(urls)
            
            return result
            
    except Exception as e:
        logger.error(f"simple_scrape_handler failed: {e}")
        return {
            'status': 'error',
            'items': [],
            'total_items': 0,
            'extraction_methods': [],
            'urls_processed': [],
            'summary': f'Error processing query: {query}',
            'error': str(e)
        }
