"""
Enhanced Content Extraction Module

Provides advanced content extraction capabilities using multiple libraries:
- BeautifulSoup with enhanced parsing options
- readability-lxml for main content extraction
- trafilatura for article extraction
- playwright-stealth for handling complex JavaScript sites
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urljoin
import re
import json
import logging
import lxml.html
import trafilatura
import justext
import readability
from bs4 import BeautifulSoup, SoupStrainer
from goose3 import Goose
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from collections import Counter
import logging # Added import

# Try to import spacy, but don't fail if it's not available
try:
    import spacy
    # Test if spaCy can actually load a model with fallback chain
    SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
    nlp = None
    SPACY_AVAILABLE = False
    
    for model_name in SPACY_MODELS:
        try:
            nlp = spacy.load(model_name)
            SPACY_AVAILABLE = True
            logger = logging.getLogger("ContentExtraction")
            logger.info(f"spaCy successfully loaded with model: {model_name}")
            break
        except OSError:
            continue
    
    if not SPACY_AVAILABLE:
        logger = logging.getLogger("ContentExtraction") 
        logger.error("No spaCy models available. Install with: python -m spacy download en_core_web_lg")
        
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None # Set spacy to None if import fails

# Configure logging
logger = logging.getLogger("ContentExtraction")

if not SPACY_AVAILABLE:
    logger.info("spaCy not available. Semantic features requiring spaCy will be limited.")

# Add the missing function that's causing the import error
async def extract_content_with_ai(
    html_content: str,
    url: str = None,
    user_intent: Dict[str, Any] = None,
    page_structure: Dict = None,
    desired_properties: List[str] = None,
    entity_type: str = "item"
) -> List[Dict[str, Any]]:
    """
    Use AI-guided extraction to pull structured data from HTML content.
    
    Args:
        html_content: The HTML content to extract data from
        url: The URL the content was downloaded from
        user_intent: Dictionary containing user's search/extraction intent
        page_structure: Optional pre-analyzed page structure information
        desired_properties: List of properties to extract
        entity_type: Type of entity to extract ("product", "article", "listing" etc.)
        
    Returns:
        List of extracted entities with their properties
    """
    try:
        # Initialize the content extractor
        extractor = ContentExtractor()
        
        # Extract main content first to reduce noise
        content_result = await extractor.extract_content(
            html_content, 
            url=url,
            content_type="listing" if entity_type in ["product", "item", "listing"] else "article"
        )
        
        # For now, implement a simple extraction based on the ContentExtractor
        # In a real implementation, this would connect to an AI service
        
        if content_result["content_type"] == "listing" and "items" in content_result:
            # We already have structured data from the extractor
            return content_result.get("items", [])
        
        # Fall back to basic extraction
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Use different extraction strategies based on entity_type
        if entity_type in ["product", "item", "listing"]:
            containers = extractor._find_listing_containers(soup)
            results = []
            
            if containers:
                for container in containers[:20]:  # Limit to 20 items
                    item = extractor._extract_listing_item(container, url)
                    if item:
                        results.append(item)
            
            # If we found results, return them
            if results:
                return results
        
        # Default fallback: extract data based on common patterns
        results = []
        
        # Look for titles as anchors for items
        potential_items = []
        
        # Find all headings with links that might be item titles
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            item_container = heading.parent
            
            # Basic extraction for title and URL
            link = heading.find('a')
            title = heading.get_text(strip=True)
            
            if link and title and len(title) > 3:
                item = {
                    'title': title,
                    'url': link.get('href') if link else None
                }
                
                # Look for nearby elements with desired properties
                if desired_properties:
                    for prop in desired_properties:
                        # Find elements that might contain this property
                        prop_elements = item_container.find_all(
                            string=re.compile(f"{prop}|{prop.replace('_', ' ')}", re.I)
                        )
                        
                        for el in prop_elements:
                            parent = el.parent
                            if parent and parent.name in ['div', 'span', 'p']:
                                # Get text that might be the property value
                                value = parent.get_text(strip=True)
                                if value and value != prop:
                                    # Clean up the value
                                    value = re.sub(f"{prop}:|{prop.replace('_', ' ')}:", "", value, flags=re.I).strip()
                                    item[prop] = value
                                    break
                
                potential_items.append(item)
        
        # Return what we've found, or empty list if nothing
        return potential_items if potential_items else []
        
    except Exception as e:
        logger.error(f"Error in AI-guided extraction: {str(e)}")
        return []

class SemanticContentExtractor:
    """
    Enhanced semantic content extraction using NLP and linguistic analysis.
    
    This class provides advanced semantic extraction capabilities:
    - Text hierarchy analysis
    - Semantic relationship recognition
    - Entity and concept extraction
    - Contextual content importance scoring
    - Cross-document relationship identification
    """
    
    def __init__(self, use_spacy: bool = False):
        """
        Initialize the semantic content extractor.
        
        Args:
            use_spacy: Whether to use spaCy for enhanced NLP (if available)
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE # Only use if explicitly requested AND available
        self.nlp = None
        if self.use_spacy and spacy: # Check if spacy module is not None
            try:
                from config import SPACY_MODEL_NAME 
                self.nlp = spacy.load(SPACY_MODEL_NAME)
                logger.info(f"spaCy model '{SPACY_MODEL_NAME}' loaded for SemanticContentExtractor.")
            except ImportError:
                logger.error("config.py or SPACY_MODEL_NAME not found. Cannot load spaCy model.")
                self.nlp = None
                self.use_spacy = False
            except Exception as e:
                logger.error(f"Failed to load spaCy model '{SPACY_MODEL_NAME}': {e}")
                self.nlp = None
                self.use_spacy = False # Disable spacy use if model loading fails
        

    async def extract_semantic_content(self, 
                                  html_content: str, 
                                  url: str = None,
                                  content_type: str = None,
                                  extraction_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content with semantic understanding.
        
        Args:
            html_content: HTML content to extract from
            url: URL of the content (for context)
            content_type: Type of content (article, listing, etc.) if known
            extraction_params: Additional parameters for extraction
            
        Returns:
            Dictionary with semantically structured content
        """
        if not html_content:
            return {"success": False, "error": "No HTML content provided"}
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Prepare the result structure
        result = {
            "success": False,
            "url": url,
            "content_type": content_type or "unknown",
            "title": self._extract_title(soup),
            "semantic_structure": {},
            "entities": {},
            "keywords": [],
            "summary": "",
            "language": "en"  # Default to English
        }
        
        try:
            # Extract clean text content
            clean_text = self._extract_clean_text(soup)
            
            if not clean_text or len(clean_text) < 100:
                logger.warning("Insufficient text content for semantic analysis")
                return {"success": False, "error": "Insufficient text content"}
            
            # Analyze text structure
            result["semantic_structure"] = self._analyze_text_structure(clean_text)
            
            # Extract entities
            result["entities"] = self._extract_entities(clean_text)
            
            # Extract keywords
            result["keywords"] = self._extract_keywords(clean_text)
            
            # Generate summary
            result["summary"] = self._generate_summary(clean_text, 
                                                    result["semantic_structure"].get("sections", []))
            
            # Detect language (simplified)
            result["language"] = self._detect_language(clean_text)
            
            # Update success flag and stats
            result["success"] = True
            self.stats["documents_processed"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in semantic content extraction: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """
        Extract the title using multiple methods.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Page title
        """
        # Try schema.org markup first
        schema_title = soup.find("meta", {"property": "og:title"})
        if schema_title and schema_title.get("content"):
            return schema_title["content"]
        
        # Try h1 tags next
        h1_title = soup.find("h1")
        if h1_title:
            return h1_title.get_text(strip=True)
        
        # Fall back to title tag
        title_tag = soup.title
        if title_tag:
            return title_tag.get_text(strip=True)
        
        return ""
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text for semantic analysis.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Clean text content
        """
        # Remove script, style and navigation elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Extract text from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'})
        
        if main_content:
            # Process paragraphs to maintain structure
            paragraphs = []
            for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = p.get_text(strip=True)
                if text and len(text) > 10:  # Skip very short paragraphs
                    paragraphs.append(text)
            
            return "\n\n".join(paragraphs)
        
        # Fallback to all paragraph elements
        paragraphs = []
        for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = p.get_text(strip=True)
            if text and len(text) > 10:
                paragraphs.append(text)
        
        return "\n\n".join(paragraphs)
    
    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the semantic structure of text.
        
        Args:
            text: Clean text content
            
        Returns:
            Dictionary with text structure analysis
        """
        # Split text into paragraphs
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        
        # Split paragraphs into sentences
        sentences = []
        for p in paragraphs:
            if SPACY_AVAILABLE and nlp:
                doc = nlp(p)
                sentences.extend([sent.text.strip() for sent in doc.sents])
            else:
                # Fallback to regex-based sentence splitting
                import re
                sentence_endings = re.split(r'[.!?]+\s+', p)
                sentences.extend([s.strip() for s in sentence_endings if s.strip()])
        
        self.stats["sentences_analyzed"] += len(sentences)
        
        # Group into sections based on semantic relatedness
        sections = self._identify_sections(paragraphs)
        
        # Analyze overall structure
        structure = {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / max(1, len(sentences)),
            "sections": sections,
            "readability_score": self._calculate_readability(text)
        }
        
        return structure
    
    def _identify_sections(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """
        Identify semantic sections within paragraphs.
        
        Args:
            paragraphs: List of text paragraphs
            
        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = {"title": "", "paragraphs": [], "keywords": []}
        
        for i, paragraph in enumerate(paragraphs):
            # Check if paragraph looks like a heading
            if len(paragraph) < 100 and (paragraph.endswith(':') or paragraph.isupper() 
                                       or (i > 0 and len(paragraph) < len(paragraphs[i-1]) * 0.5)):
                # Finalize previous section
                if current_section["paragraphs"]:
                    current_section["keywords"] = self._extract_keywords(
                        "\n".join(current_section["paragraphs"]), 3)
                    sections.append(current_section)
                
                # Start a new section
                current_section = {"title": paragraph, "paragraphs": [], "keywords": []}
            else:
                current_section["paragraphs"].append(paragraph)
        
        # Add the final section
        if current_section["paragraphs"]:
            current_section["keywords"] = self._extract_keywords(
                "\n".join(current_section["paragraphs"]), 3)
            sections.append(current_section)
        
        return sections
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text using NLP techniques.
        
        Args:
            text: Clean text content
            
        Returns:
            Dictionary with categorized entities
        """
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "other": []
        }
        
        if self.use_spacy and self.nlp:
            # Use spaCy for advanced entity recognition
            doc = self.nlp(text[:10000])  # Limit to 10k chars for performance
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["persons"].append(ent.text)
                elif ent.label_ in ["ORG", "NORP"]:
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ in ["DATE", "TIME"]:
                    entities["dates"].append(ent.text)
                else:
                    entities["other"].append(ent.text)
        else:
            # Simple regex-based entity extraction as fallback
            # Extract capitalized phrases as potential named entities
            cap_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text)
            
            # Filter by frequency and length
            entity_counter = Counter(cap_phrases)
            
            for entity, count in entity_counter.most_common(30):
                if len(entity.split()) == 1:
                    if count >= 2:  # Require higher frequency for single words
                        entities["other"].append(entity)
                else:
                    entities["other"].append(entity)
            
            # Extract dates with regex
            date_patterns = [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
                r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),\s+\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b'
            ]
            
            for pattern in date_patterns:
                entities["dates"].extend(re.findall(pattern, text))
        
        # Deduplicate entities
        for category in entities:
            entities[category] = list(set(entities[category]))
        
        self.stats["entities_extracted"] += sum(len(v) for v in entities.values())
        
        return entities
    
    def _extract_keywords(self, text: str, limit: int = 10) -> List[str]:
        """
        Extract keywords from text using spaCy-first approach.
        
        Args:
            text: Text to extract keywords from
            limit: Maximum number of keywords to return
            
        Returns:
            List of keyword strings
        """
        if self.use_spacy and self.nlp:
            # Use spaCy for advanced keyword extraction
            doc = self.nlp(text.lower())
            
            # Extract meaningful tokens, excluding stopwords, punctuation, and short words
            keywords = []
            for token in doc:
                if (not token.is_stop and not token.is_punct and not token.is_space 
                    and token.is_alpha and len(token.text) > 3
                    and token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                    # Use lemmatized form for better grouping
                    keywords.append(token.lemma_)
            
            # Count frequencies and return top keywords
            word_freq = Counter(keywords)
            return [word for word, _ in word_freq.most_common(limit)]
        else:
            # Fallback to basic tokenization if spaCy not available
            words = text.lower().split()
            
            # Remove stopwords and non-alphabetic tokens
            filtered_words = [word for word in words if word.isalpha() and word not in self.stopwords
                          and len(word) > 3]
            
            # Count word frequencies
            word_freq = Counter(filtered_words)
            
            # Return most common words
            return [word for word, _ in word_freq.most_common(limit)]
    
    def _generate_summary(self, text: str, sections: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the content.
        
        Args:
            text: Full text content
            sections: Analyzed text sections
            
        Returns:
            Summary string
        """
        # If we have sections with titles, use them to generate a structured summary
        if sections and any(s.get("title") for s in sections):
            summary_parts = []
            
            for section in sections:
                if section.get("title") and section.get("paragraphs"):
                    # Add section title
                    summary_parts.append(section["title"])
                    
                    # Add first sentence of first paragraph
                    first_para = section["paragraphs"][0]
                    if first_para:
                        # Use spaCy first, fallback to basic sentence splitting
                        if SPACY_AVAILABLE and nlp:
                            doc = nlp(first_para)
                            sentences = [sent.text for sent in doc.sents]
                            first_sentence = sentences[0] if sentences else ""
                        else:
                            # Fallback to basic sentence splitting
                            sentences = re.split(r'[.!?]+', first_para.strip())
                            first_sentence = sentences[0].strip() + '.' if sentences and sentences[0].strip() else ""
                    else:
                        first_sentence = ""
                    
                    if first_sentence and first_sentence != section["title"]:
                        summary_parts.append(first_sentence)
            
            if summary_parts:
                return " ".join(summary_parts)
        
        # Fallback: Use first few sentences
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Fallback to basic sentence splitting
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Calculate target length (about 10% of original or 3 sentences, whichever is larger)
        target_sentence_count = max(3, int(len(sentences) * 0.1))
        
        return " ".join(sentences[:target_sentence_count])
    
    def _calculate_readability(self, text: str) -> float:
        """
        Calculate a readability score for the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability score (0-100)
        """
        # Simple implementation of Flesch Reading Ease score
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            words = [token.text for token in doc if token.is_alpha]
        else:
            # Fallback to regex-based tokenization
            import re
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        if not sentences or not words:
            return 0
        
        # Count syllables (approximation)
        syllable_count = 0
        for word in words:
            word = word.lower()
            if word.isalpha():
                if len(word) <= 3:
                    syllable_count += 1
                else:
                    # Count vowel groups as syllables
                    syllable_count += len(re.findall(r'[aeiouy]+', word))
        
        # Calculate averages
        words_per_sentence = len(words) / len(sentences)
        syllables_per_word = syllable_count / len(words)
        
        # Flesch Reading Ease formula
        score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
        
        # Clamp to range 0-100
        return max(0, min(100, score))
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (ISO 639-1)
        """
        # This is a simplified implementation
        # In a production system, use a dedicated language detection library
        return "en"  # Default to English
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on the semantic extraction process."""
        return self.stats

class ContentExtractor:
    """
    Enhanced content extraction using multiple extraction engines.
    
    This class provides:
    - Multiple extraction backends with automatic fallback
    - Smart content type detection
    - Special handlers for articles, listings, and data tables
    - JavaScript-rendered content extraction
    """
    
    def __init__(self, use_stealth_browser: bool = False):
        """
        Initialize the content extractor.
        
        Args:
            use_stealth_browser: Whether to use Playwright with stealth mode
        """
        self.use_stealth_browser = use_stealth_browser
        self.goose_extractor = Goose()
        self.extraction_stats = {
            "calls": 0,
            "readability_success": 0,
            "trafilatura_success": 0,
            "goose_success": 0,
            "justext_success": 0,
            "fallback_used": 0
        }
        
        # Add semantic extractor for enhanced text analysis capabilities
        self.semantic_extractor = SemanticContentExtractor()
        
    async def extract_with_schema(self, 
                              soup: BeautifulSoup, 
                              url: str, 
                              extraction_schema: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Extract data from HTML based on extraction schema.
        
        Args:
            soup: BeautifulSoup object of the page
            url: URL of the page
            extraction_schema: Schema defining what to extract and how
            
        Returns:
            Dictionary with extracted data based on schema
        """
        try:
            if not soup or not extraction_schema:
                return {"error": "Missing required parameters"}
            
            results = {}
            
            # Handle nested items extraction
            if "items" in extraction_schema:
                results["items"] = []
                item_selectors = extraction_schema["items"]
                
                # Find all item containers
                containers = []
                for selector in item_selectors:
                    try:
                        found_containers = soup.select(selector)
                        if found_containers and len(found_containers) > 0:
                            containers = found_containers
                            results["items_selector_used"] = selector
                            break
                    except Exception as e:
                        logger.warning(f"Error with selector '{selector}': {str(e)}")
                
                # Define item extraction schema by removing the "items" key
                item_extraction_schema = {k: v for k, v in extraction_schema.items() if k != "items"}
                
                # Extract data from each container
                for container in containers[:20]:  # Limit to 20 items to prevent overload
                    item_data = await self._extract_data_with_schema(container, url, item_extraction_schema)
                    if item_data and not all(v is None for v in item_data.values()):
                        results["items"].append(item_data)
                
                results["item_count"] = len(results["items"])
                
            # If not extracting items or if items extraction failed, extract from the whole page
            if "items" not in extraction_schema or not results.get("items"):
                direct_extraction = await self._extract_data_with_schema(soup, url, extraction_schema)
                
                # If extracting items failed, return direct extraction
                if "items" not in extraction_schema:
                    results = direct_extraction
                # Otherwise, add direct extraction results alongside any items
                else:
                    results.update({k: v for k, v in direct_extraction.items() if k != "items"})
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting with schema: {str(e)}")
            return {"error": str(e)}
    
    async def _extract_data_with_schema(self, 
                                   soup: BeautifulSoup, 
                                   url: str, 
                                   extraction_schema: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Extract data from an element based on schema.
        
        Args:
            soup: BeautifulSoup element to extract from
            url: URL of the page
            extraction_schema: Schema defining what to extract and how
            
        Returns:
            Dictionary with extracted data
        """
        extracted_data = {}
        
        # Process each field in the schema
        for field_name, selectors in extraction_schema.items():
            # Skip the "items" field as it's handled separately
            if field_name == "items":
                continue
                
            # Try each selector in order until one works
            for selector in selectors:
                try:
                    # Handle attribute extraction with special syntax
                    if '::attr(' in selector:
                        # Format: "css_selector::attr(attribute_name)"
                        css_part, attr_part = selector.split('::attr(')
                        attr_name = attr_part.rstrip(')')
                        
                        elements = soup.select(css_part)
                        if elements:
                            value = elements[0].get(attr_name, None)
                            if value:
                                extracted_data[field_name] = value
                                break
                    # Handle text extraction (default)
                    else:
                        elements = soup.select(selector)
                        if elements:
                            # For images, extract the src attribute
                            if field_name == "image" or field_name.endswith("_image"):
                                value = elements[0].get('src')
                                if not value:
                                    value = elements[0].get('data-src')  # Try data-src for lazy loading
                                
                                # Handle relative URLs
                                if value and url and value.startswith('/'):
                                    from urllib.parse import urljoin
                                    value = urljoin(url, value)
                                    
                                extracted_data[field_name] = value
                                break
                            # For links, extract href attribute
                            elif field_name == "link" or field_name == "url" or field_name.endswith("_link") or field_name.endswith("_url"):
                                value = elements[0].get('href')
                                
                                # Handle relative URLs
                                if value and url and value.startswith('/'):
                                    from urllib.parse import urljoin
                                    value = urljoin(url, value)
                                    
                                extracted_data[field_name] = value
                                break
                            # For prices, try to extract and clean price information
                            elif field_name == "price" or field_name.endswith("_price"):
                                price_text = elements[0].get_text(strip=True)
                                # Extract price using regex
                                price_match = re.search(r'(\$|€|£|¥)?\s*[\d,]+(\.\d{1,2})?', price_text)
                                if price_match:
                                    extracted_data[field_name] = price_match.group(0).strip()
                                else:
                                    extracted_data[field_name] = price_text
                                break
                            # For dates, try to extract and standardize date formats
                            elif field_name == "date" or field_name.endswith("_date"):
                                date_text = elements[0].get_text(strip=True)
                                # Attempt to parse various date formats
                                try:
                                    import dateutil.parser
                                    parsed_date = dateutil.parser.parse(date_text)
                                    extracted_data[field_name] = parsed_date.isoformat()
                                except:
                                    # If parsing fails, just use the text
                                    extracted_data[field_name] = date_text
                                break
                            # For all other fields, extract text content
                            else:
                                value = elements[0].get_text(strip=True)
                                if value:
                                    extracted_data[field_name] = value
                                    break
                except Exception as e:
                    logger.debug(f"Selector '{selector}' failed: {str(e)}")
                    continue
            
            # If field wasn't found with any selector, set it to None
            if field_name not in extracted_data:
                extracted_data[field_name] = None
        
        return extracted_data
    
    async def extract_content(self, 
                           html_content: str, 
                           url: str = None,
                           content_type: str = None,
                           extraction_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content from HTML using the best method for the content type.
        
        Args:
            html_content: HTML content to extract from
            url: URL of the content (for context)
            content_type: Type of content (article, listing, etc.) if known
            extraction_params: Additional parameters for extraction
            
        Returns:
            Dictionary with extracted content and metadata
        """
        self.extraction_stats["calls"] += 1
        
        if not html_content:
            return {"success": False, "error": "No HTML content provided"}
        
        # Default response structure
        result = {
            "success": False,
            "content_type": content_type or "unknown",
            "url": url,
            "title": "",
            "text": "",
            "html": "",
            "metadata": {},
            "extraction_method": ""
        }
        
        # Detect content type if not provided
        if not content_type:
            content_type = self._detect_content_type(html_content, url)
            result["content_type"] = content_type
        
        # Handle JavaScript content if needed
        if self._needs_javascript_processing(html_content, content_type):
            if self.use_stealth_browser:
                # Extract using Playwright with stealth mode
                js_result = await self._extract_with_playwright(url, extraction_params)
                if js_result and js_result.get("success", False):
                    html_content = js_result.get("html", html_content)
                    result["js_rendered"] = True
        
        # Try semantic extraction for enhanced content understanding
        if content_type in ["article", "blog_post", "news"] or extraction_params and extraction_params.get("use_semantic", False):
            try:
                semantic_result = await self.semantic_extractor.extract_semantic_content(
                    html_content, url, content_type, extraction_params
                )
                
                if semantic_result.get("success", False):
                    # Use semantic extraction results
                    result.update({
                        "success": True,
                        "extraction_method": "semantic",
                        "semantic_structure": semantic_result.get("semantic_structure", {}),
                        "entities": semantic_result.get("entities", {}),
                        "keywords": semantic_result.get("keywords", []),
                        "summary": semantic_result.get("summary", ""),
                        "title": semantic_result.get("title", result["title"]),
                        "text": "\n\n".join([
                            section.get("title", "") + "\n" + 
                            "\n".join(section.get("paragraphs", []))
                            for section in semantic_result.get("semantic_structure", {}).get("sections", [])
                        ]),
                        "language": semantic_result.get("language", "en")
                    })
                    
                    return result
            except Exception as e:
                logger.warning(f"Semantic extraction failed, falling back to standard: {str(e)}")
        
        # Use specialized extraction based on content type
        if content_type == "article":
            extracted = await self._extract_article(html_content, url, extraction_params)
        elif content_type == "listing":
            extracted = await self._extract_listing(html_content, url, extraction_params)
        elif content_type == "data_table":
            extracted = await self._extract_table(html_content, url, extraction_params)
        else:
            # Generic extraction
            extracted = await self._extract_generic(html_content, url, extraction_params)
        
        # Merge extracted results with our response structure
        if extracted and extracted.get("success", False):
            result.update(extracted)
            result["success"] = True
        else:
            # Fallback to simple extraction if all specialized methods failed
            fallback = self._fallback_extraction(html_content, url)
            result.update(fallback)
            result["extraction_method"] = "fallback"
            self.extraction_stats["fallback_used"] += 1
        
        return result
    
    async def _extract_article(self, 
                             html_content: str, 
                             url: str = None,
                             params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract article content using multiple engines with fallback.
        
        Args:
            html_content: HTML content
            url: Article URL
            params: Extraction parameters
            
        Returns:
            Extracted article content and metadata
        """
        # Try trafilatura first (best for articles)
        trafilatura_result = self._extract_with_trafilatura(html_content, url)
        if trafilatura_result.get("success", False):
            self.extraction_stats["trafilatura_success"] += 1
            return trafilatura_result
        
        # Try readability-lxml
        readability_result = self._extract_with_readability(html_content, url)
        if readability_result.get("success", False):
            self.extraction_stats["readability_success"] += 1
            return readability_result
        
        # Try goose3
        goose_result = self._extract_with_goose(html_content, url)
        if goose_result.get("success", False):
            self.extraction_stats["goose_success"] += 1
            return goose_result
        
        # Try justext
        justext_result = self._extract_with_justext(html_content, url)
        if justext_result.get("success", False):
            self.extraction_stats["justext_success"] += 1
            return justext_result
        
        # All methods failed
        return {"success": False, "error": "All article extraction methods failed"}
    
    async def _extract_listing(self, 
                              html_content: str, 
                              url: str = None,
                              params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract listing content (products, items, etc.).
        
        Args:
            html_content: HTML content
            url: Page URL
            params: Extraction parameters
            
        Returns:
            Extracted listing items
        """
        items = []
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Look for common item containers
            containers = self._find_listing_containers(soup)
            
            if containers:
                for container in containers[:20]:  # Limit to 20 items
                    item = self._extract_listing_item(container, url)
                    if item:
                        items.append(item)
                
                return {
                    "success": True,
                    "content_type": "listing",
                    "items": items,
                    "count": len(items),
                    "extraction_method": "pattern_based"
                }
            
            # If we didn't find items, try generic content extraction
            # and then attempt to find structured data
            generic_result = await self._extract_generic(html_content, url, params)
            
            # Check for schema.org/Product or other structured data
            ld_json = soup.find_all('script', {'type': 'application/ld+json'})
            structured_items = []
            
            for script in ld_json:
                try:
                    data = json.loads(script.string)
                    # Handle array of items
                    if isinstance(data, list):
                        structured_items.extend(data)
                    # Handle single item
                    else:
                        structured_items.append(data)
                except:
                    pass
            
            if structured_items:
                return {
                    "success": True,
                    "content_type": "listing",
                    "items": structured_items,
                    "count": len(structured_items),
                    "extraction_method": "structured_data"
                }
            # Return generic result if we didn't find structured data
            return generic_result
                
        except Exception as e:
            logger.error(f"Error in listing extraction: {str(e)}")
            return {"success": False, "error": f"Listing extraction failed: {str(e)}"}
    
    async def _extract_table(self,
                           html_content: str,
                           url: str = None,
                           params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract data tables from HTML content.
        
        Args:
            html_content: HTML content
            url: Page URL
            params: Extraction parameters
            
        Returns:
            Extracted table data
        """
        tables = []
        
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            page_tables = soup.find_all('table')
            
            for i, table in enumerate(page_tables):
                rows = []
                
                # Get headers first
                headers = []
                header_row = table.find('thead')
                
                if header_row:
                    header_cells = header_row.find_all(['th', 'td'])
                    headers = [cell.get_text(strip=True) for cell in header_cells]
                
                # Process table body
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        # Skip rows that are likely headers if we already have headers
                        if headers and row_data == headers:
                            continue
                        rows.append(row_data)
                
                # If we didn't extract headers before but have rows
                if not headers and rows:
                    # Use first row as headers
                    headers = rows[0]
                    rows = rows[1:]
                
                # Only add tables with actual data
                if rows and any(row for row in rows if any(cell for cell in row)):
                    tables.append({
                        "id": f"table_{i}",
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows)
                    })
            
            # Only return success if we found tables
            if tables:
                return {
                    "success": True,
                    "content_type": "data_table",
                    "tables": tables,
                    "count": len(tables),
                    "extraction_method": "table_parser"
                }
            
            # If no tables found, fall back to generic extraction
            return await self._extract_generic(html_content, url, params)
                
        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}")
            return {"success": False, "error": f"Table extraction failed: {str(e)}"}
    
    async def _extract_generic(self,
                              html_content: str,
                              url: str = None,
                              params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generic content extraction for when type is unknown.
        
        Args:
            html_content: HTML content
            url: Page URL
            params: Extraction parameters
            
        Returns:
            Extracted content
        """
        # Try each extraction method in order and use the first successful one
        
        # First try trafilatura for best generic content
        trafilatura_result = self._extract_with_trafilatura(html_content, url)
        if trafilatura_result.get("success", False):
            self.extraction_stats["trafilatura_success"] += 1
            return trafilatura_result
        
        # Then try readability
        readability_result = self._extract_with_readability(html_content, url)
        if readability_result.get("success", False):
            self.extraction_stats["readability_success"] += 1
            return readability_result
        
        # Then try goose
        goose_result = self._extract_with_goose(html_content, url)
        if goose_result.get("success", False):
            self.extraction_stats["goose_success"] += 1
            return goose_result
        
        # Then try justext
        justext_result = self._extract_with_justext(html_content, url)
        if justext_result.get("success", False):
            self.extraction_stats["justext_success"] += 1
            return justext_result
        
        # All methods failed, use raw extraction
        return self._fallback_extraction(html_content, url)
    
    def _extract_with_trafilatura(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract content using Trafilatura.
        
        Args:
            html_content: HTML content
            url: Page URL
            
        Returns:
            Extracted content and metadata
        """
        try:
            # Extract article text
            extracted_text = trafilatura.extract(
                html_content,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                output_format='text'
            )
            
            # Extract article HTML
            extracted_html = trafilatura.extract(
                html_content,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                output_format='html'
            )
            
            # Extract metadata
            metadata = trafilatura.metadata.extract_metadata(
                html_content,
                url=url
            )
            
            if not extracted_text or len(extracted_text.strip()) < 50:
                return {"success": False, "error": "Trafilatura extraction returned minimal content"}
            
            # Format metadata
            formatted_metadata = {}
            if metadata:
                for key, value in metadata.__dict__.items():
                    if not key.startswith('_') and value:
                        formatted_metadata[key] = value
            
            return {
                "success": True,
                "title": formatted_metadata.get('title', ''),
                "text": extracted_text,
                "html": extracted_html,
                "metadata": formatted_metadata,
                "extraction_method": "trafilatura"
            }
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {str(e)}")
            return {"success": False, "error": f"Trafilatura extraction failed: {str(e)}"}
    
    def _extract_with_readability(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract content using readability-lxml.
        
        Args:
            html_content: HTML content
            url: Page URL
            
        Returns:
            Extracted content and metadata
        """
        try:
            # Create a document
            doc = readability.Document(html_content)
            
            # Extract title
            title = doc.title()
            
            # Extract main content
            article = doc.summary(html_partial=True)
            
            # Extract text from HTML
            article_soup = BeautifulSoup(article, 'lxml')
            text_content = article_soup.get_text(separator=' ', strip=True)
            
            # Create metadata
            metadata = {
                "title": title,
                "excerpt": text_content[:150] + "..." if len(text_content) > 150 else text_content
            }
            
            # Enhance metadata with opengraph tags
            soup = BeautifulSoup(html_content, 'lxml')
            og_tags = {
                tag.get('property', ''): tag.get('content', '')
                for tag in soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            }
            
            # Add opengraph data to metadata
            for key, value in og_tags.items():
                if value:
                    clean_key = key.replace('og:', '')
                    metadata[clean_key] = value
            
            # Check if extraction has enough content
            if not text_content or len(text_content.strip()) < 50:
                return {"success": False, "error": "Readability extraction returned minimal content"}
            
            return {
                "success": True,
                "title": title,
                "text": text_content,
                "html": article,
                "metadata": metadata,
                "extraction_method": "readability"
            }
        except Exception as e:
            logger.warning(f"Readability extraction failed: {str(e)}")
            return {"success": False, "error": f"Readability extraction failed: {str(e)}"}
    
    def _extract_with_goose(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract content using Goose3.
        
        Args:
            html_content: HTML content
            url: Page URL
            
        Returns:
            Extracted content and metadata
        """
        try:
            # Extract article
            article = self.goose_extractor.extract(raw_html=html_content, url=url)
            
            # Get article text
            text_content = article.cleaned_text
            
            # Get article HTML if available
            html_content = article.cleaned_article_html or ""
            
            # Format metadata
            metadata = {
                "title": article.title,
                "meta_description": article.meta_description,
                "meta_keywords": article.meta_keywords,
                "canonical_link": article.canonical_link,
                "top_image": article.top_image.src if article.top_image else None,
                "publish_date": article.publish_date_str if hasattr(article, 'publish_date_str') else None,
                "meta_lang": article.meta_lang,
                "authors": article.authors
            }
            
            # Clean metadata to remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Check if extraction has enough content
            if not text_content or len(text_content.strip()) < 50:
                return {"success": False, "error": "Goose extraction returned minimal content"}
            
            return {
                "success": True,
                "title": article.title,
                "text": text_content,
                "html": html_content,
                "metadata": metadata,
                "extraction_method": "goose"
            }
        except Exception as e:
            logger.warning(f"Goose extraction failed: {str(e)}")
            return {"success": False, "error": f"Goose extraction failed: {str(e)}"}
    
    def _extract_with_justext(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract content using JusText.
        
        Args:
            html_content: HTML content
            url: Page URL
            
        Returns:
            Extracted content and metadata
        """
        try:
            paragraphs = justext.justext(
                html_content.encode('utf-8', errors='replace'), 
                justext.get_stoplist('English')
            )
            
            # Extract good paragraphs
            content_paragraphs = [p.text for p in paragraphs if not p.is_boilerplate]
            
            # Combine paragraphs into text content
            text_content = "\n\n".join(content_paragraphs)
            
            # Extract title from HTML
            soup = BeautifulSoup(html_content, 'lxml')
            title = soup.title.get_text(strip=True) if soup.title else ""
            
            # Create basic metadata
            metadata = {"title": title}
            
            # Enhance metadata with opengraph tags
            og_tags = {
                tag.get('property', ''): tag.get('content', '')
                for tag in soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            }
            
            # Add opengraph data to metadata
            for key, value in og_tags.items():
                if value:
                    clean_key = key.replace('og:', '')
                    metadata[clean_key] = value
            
            # Check if extraction has enough content
            if not text_content or len(text_content.strip()) < 50:
                return {"success": False, "error": "JusText extraction returned minimal content"}
            
            return {
                "success": True,
                "title": title,
                "text": text_content,
                "html": "",  # JusText doesn't preserve HTML
                "metadata": metadata,
                "extraction_method": "justext"
            }
        except Exception as e:
            logger.warning(f"JusText extraction failed: {str(e)}")
            return {"success": False, "error": f"JusText extraction failed: {str(e)}"}
    
    def _fallback_extraction(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Basic fallback extraction using BeautifulSoup.
        
        Args:
            html_content: HTML content
            url: Page URL
            
        Returns:
            Extracted content and metadata in the format expected by the system
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
                element.decompose()
            
            # Extract title
            title = soup.title.get_text(strip=True) if soup.title else ""
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Clean up text content
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Create basic metadata
            metadata = {"title": title}
            
            # Extract meta tags for additional metadata
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                
                if name and content:
                    metadata[name] = content
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'id': 'content'})
            
            if main_content:
                main_text = main_content.get_text(separator=' ', strip=True)
                main_text = re.sub(r'\s+', ' ', main_text).strip()
                
                # If main_text has enough content, use it instead of full page text
                if len(main_text) > 200:
                    text_content = main_text
            
            # Create extracted data item in the format expected by the system
            extracted_item = {
                "title": title,
                "content": text_content[:1000],  # Limit content for display
                "description": metadata.get("description", ""),
                "url": url if url else "",
                "source_url": url if url else "",
                "extraction_method": "basic_soup_fallback",
                "content_type": "article",
                "tags": ["fallback_extraction"]
            }
            
            # Add more metadata if available
            if "keywords" in metadata:
                extracted_item["keywords"] = metadata["keywords"]
            
            return {
                "success": True,
                "title": title,
                "text": text_content,
                "html": str(soup),
                "metadata": metadata,
                "extraction_method": "basic_soup",
                # Return data in array format as expected by the system
                "data": [extracted_item]
            }
        except Exception as e:
            logger.error(f"Basic extraction failed: {str(e)}")
            # Last resort: just extract what we can without formatting
            basic_text = re.sub(r'<[^>]+>', '', html_content)
            
            # Create minimal extracted data item even for failed extraction
            extracted_item = {
                "title": "Extraction failed",
                "content": basic_text[:500] if basic_text else "No content extracted",
                "description": "",
                "url": url if url else "",
                "source_url": url if url else "",
                "extraction_method": "raw_text_fallback",
                "content_type": "error",
                "tags": ["fallback_extraction", "error"]
            }
            
            return {
                "success": True,
                "title": "",
                "text": basic_text,
                "html": html_content,
                "metadata": {},
                "extraction_method": "raw_text",
                # Return data in array format as expected by the system
                "data": [extracted_item]
            }
    
    async def _extract_with_playwright(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content using Playwright with stealth mode.
        
        Args:
            url: Page URL
            params: Extraction parameters
            
        Returns:
            Extracted content
        """
        if not url:
            return {"success": False, "error": "URL required for Playwright extraction"}
        
        wait_for = params.get("wait_for", 5000) if params else 5000
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 800}
            )
            page = await context.new_page()
            
            # Apply stealth mode to avoid detection
            await stealth_async(page)
            
            try:
                # Navigate to the URL
                await page.goto(url, wait_until="domcontentloaded")
                
                # Wait for content to load
                await page.wait_for_timeout(wait_for)
                
                # Get the HTML content
                html_content = await page.content()
                
                # Extract any useful data from the page
                metadata = {}
                
                # Get page title
                title = await page.title()
                metadata["title"] = title
                
                # Check for lazy-loaded images and scroll down if needed
                await self._handle_lazy_loading(page)
                
                # Get the final HTML after all processing
                final_html = await page.content()
                
                return {
                    "success": True,
                    "html": final_html,
                    "title": title,
                    "metadata": metadata,
                    "extraction_method": "playwright"
                }
            except Exception as e:
                logger.error(f"Playwright extraction failed: {str(e)}")
                return {"success": False, "error": f"Playwright extraction failed: {str(e)}"}
            finally:
                await browser.close()
    
    async def _handle_lazy_loading(self, page):
        """Scroll to handle lazy-loaded content."""
        try:
            # Get the page height
            height = await page.evaluate('document.body.scrollHeight')
            
            # Scroll down in steps
            for i in range(0, height, 300):
                await page.evaluate(f'window.scrollTo(0, {i})')
                await page.wait_for_timeout(100)  # Short wait between scrolls
            
            # Scroll back to top
            await page.evaluate('window.scrollTo(0, 0)')
            
            # Wait for any lazy-loaded elements to appear
            await page.wait_for_timeout(500)
        except Exception as e:
            logger.warning(f"Error during lazy loading: {str(e)}")
    
    def _detect_content_type(self, html_content: str, url: str = None) -> str:
        """
        Detect the type of content in the HTML.
        
        Args:
            html_content: HTML content
            url: URL of the content
            
        Returns:
            Content type string
        """
        # Try a quick check using URL patterns
        if url:
            url_lower = url.lower()
            
            # Check for article patterns
            if re.search(r'/(article|post|blog)s?/', url_lower) or '/news/' in url_lower:
                return "article"
            
            # Check for product patterns
            if '/product/' in url_lower or '/item/' in url_lower or '/p/' in url_lower:
                return "product"
            
            # Check for listing patterns
            if ('/category/' in url_lower or '/collection/' in url_lower or 
                '/search?' in url_lower or '/listing' in url_lower):
                return "listing"
        
        # Analyze HTML structure
        soup = BeautifulSoup(html_content, 'lxml', parse_only=SoupStrainer(['article', 'table', 'div', 'meta']))
        
        # Check for article indicators
        if soup.find('article') or soup.find('meta', {'property': 'article:published_time'}):
            return "article"
        
        # Check for product indicators
        if (soup.find('meta', {'property': 'product:price:amount'}) or 
            soup.find('div', {'itemtype': 'http://schema.org/Product'})):
            return "product"
        
        # Check for data tables
        tables = soup.find_all('table')
        if tables and len(tables) > 0:
            # Check if any table has enough data cells to be considered a data table
            for table in tables:
                data_cells = table.find_all(['td', 'th'])
                if len(data_cells) > 20:  # Arbitrary threshold for a data table
                    return "data_table"
        
        # Check for listing indicators
        if self._find_listing_containers(soup):
            return "listing"
        
        # Default to generic if no specific type detected
        return "generic"
    
    def _needs_javascript_processing(self, html_content: str, content_type: str) -> bool:
        """
        Determine if the content needs JavaScript processing.
        
        Args:
            html_content: HTML content
            content_type: Content type
            
        Returns:
            Boolean indicating if JavaScript processing is needed
        """
        # If content is already substantial, no need for JS processing
        if len(html_content) > 100000:  # Arbitrary threshold for substantial content
            return False
        
        # Check for lazy loading patterns
        lazy_loading_patterns = [
            'lazy-load', 'lazyload', 'data-src=', 'data-lazy', 
            'loading="lazy"', "class='lazy'", 'class="lazy"'
        ]
        
        if any(pattern in html_content for pattern in lazy_loading_patterns):
            return True
        
        # Check for no-js class on html or body
        no_js_pattern = re.search(r'<html[^>]*class=["\'][^"\']*no-?js[^"\']*["\']', html_content)
        if no_js_pattern:
            return True
        
        # Check for too little content which might indicate JS dependency
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text(strip=True)
        
        # Check for minimal content with script tags
        if len(text_content) < 500 and len(soup.find_all('script')) > 5:
            return True
        
        # Check for React/Angular/Vue patterns
        js_framework_patterns = [
            'ng-app', 'ng-controller', 'v-for', 'v-if', 'v-bind', 'data-reactid',
            'react-root', 'ember-view'
        ]
        
        for pattern in js_framework_patterns:
            if pattern in html_content:
                return True
        
        return False
    
    def _find_listing_containers(self, soup) -> List:
        """
        Find containers that might contain listing items.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of container elements
        """
        # Common selectors for listing containers
        container_selectors = [
            # Product grids
            'div.products', 'ul.products', 'div.product-grid', 'div.product-list',
            # Search results
            'div.search-results', 'div.listings', 'div.results',
            # Article listings
            'div.articles', 'div.posts'
        ]
        
        containers = []
        
        # Try direct selectors first
        for selector in container_selectors:
            tag, class_name = selector.split('.')
            found = soup.find_all(tag, class_=class_name)
            if found:
                containers.extend(found)
        
        # If no containers found, look for repeating patterns
        if not containers:
            # Look for divs/lis with certain classes
            item_class_patterns = ['item', 'product', 'card', 'cell', 'result', 'listing']
            
            for pattern in item_class_patterns:
                items = soup.find_all(['div', 'li'], class_=lambda c: c and pattern in c.lower())
                if len(items) >= 3:  # At least 3 items to consider a listing
                    # Group by parent to find container
                    parent_count = {}
                    for item in items:
                        parent = item.parent
                        parent_key = f"{parent.name}#{parent.get('id', '')}#{' '.join(parent.get('class', []))}"
                        parent_count[parent_key] = parent_count.get(parent_key, 0) + 1
                    
                    # Find parent with the most items
                    if parent_count:
                        max_parent_key = max(parent_count, key=parent_count.get)
                        
                        # Only use if parent has multiple items
                        if parent_count[max_parent_key] >= 3:
                            # Extract parts from key
                            parts = max_parent_key.split('#')
                            tag = parts[0]
                            id_value = parts[1]
                            class_names = parts[2].split()
                            
                            # Find the parent element
                            if id_value:
                                container = soup.find(tag, id=id_value)
                                if container:
                                    containers.append(container)
                            elif class_names:
                                container = soup.find(tag, class_=class_names)
                                if container:
                                    containers.append(container)
        
        return containers
    
    def _extract_listing_item(self, container, base_url=None) -> Dict[str, Any]:
        """
        Extract data from a listing item.
        
        Args:
            container: Container element
            base_url: Base URL for resolving relative links
            
        Returns:
            Extracted item data
        """
        item = {}
        
        # Extract title and link
        title_elem = container.find(['h2', 'h3', 'h4', 'h5', 'a'])
        if title_elem:
            item['title'] = title_elem.get_text(strip=True)
            
            # Extract link
            link = title_elem if title_elem.name == 'a' else title_elem.find('a')
            if link and link.has_attr('href'):
                href = link['href']
                # Resolve relative URLs
                if base_url and href.startswith('/'):
                    item['url'] = urljoin(base_url, href)
                else:
                    item['url'] = href
        
        # Extract image
        img = container.find('img')
        if img:
            # Check for lazy loading attributes
            src = img.get('data-src') or img.get('data-lazy-src') or img.get('src')
            if src:
                # Resolve relative URLs
                if base_url and src.startswith('/'):
                    item['image'] = urljoin(base_url, src)
                else:
                    item['image'] = src
        
        # Extract price
        price_elem = container.find(string=re.compile(r'(\$|€|£|)\s*\d+(\.\d{2})?'))
        if price_elem:
            item['price'] = price_elem.strip()
        else:
            # Try to find elements that might contain price
            price_classes = ['price', 'cost', 'amount']
            for cls in price_classes:
                price_container = container.find(class_=lambda c: c and cls.lower() in c.lower())
                if price_container:
                    price_text = price_container.get_text(strip=True)
                    price_match = re.search(r'(\$|€|£|)\s*\d+(\.\d{2})?', price_text)
                    if price_match:
                        item['price'] = price_match.group(0)
                        break
        
        # Extract description
        desc_elem = container.find(['p', 'div'], class_=lambda c: c and 'desc' in c.lower())
        if desc_elem:
            item['description'] = desc_elem.get_text(strip=True)
        
        # Only return item if we have at least title or URL
        return item if 'title' in item or 'url' in item else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        if self.extraction_stats["calls"] > 0:
            success_rate = (self.extraction_stats["calls"] - self.extraction_stats["fallback_used"]) / self.extraction_stats["calls"] * 100
            self.extraction_stats["success_rate"] = round(success_rate, 2)
            
            # Calculate method distribution
            total_successful = (
                self.extraction_stats["readability_success"] +
                self.extraction_stats["trafilatura_success"] +
                self.extraction_stats["goose_success"] + 
                self.extraction_stats["justext_success"]
            )
            
            if total_successful > 0:
                self.extraction_stats["method_distribution"] = {
                    "readability": round(self.extraction_stats["readability_success"] / total_successful * 100, 2),
                    "trafilatura": round(self.extraction_stats["trafilatura_success"] / total_successful * 100, 2),
                    "goose": round(self.extraction_stats["goose_success"] / total_successful * 100, 2),
                    "justext": round(self.extraction_stats["justext_success"] / total_successful * 100, 2)
                }
        
        return self.extraction_stats

class MultiStrategyExtractor:
    """
    Implements a cascading extraction engine that prioritizes and applies multiple extraction
    strategies based on content type, confidence, and context.
    
    This orchestrates different extraction approaches:
    1. CSS Selector extraction (most precise)
    2. XPath extraction (for complex selections)
    3. Content heuristics (readability, trafilatura for articles)
    4. AI-guided extraction (most flexible, but resource intensive)
    """
    
    def __init__(self, use_ai: bool = True):
        """Initialize the multi-strategy extractor"""
        self.content_extractor = ContentExtractor(use_stealth_browser=False)
        self.use_ai = use_ai
        self.strategies = {
            "css_selector": self._extract_with_css_selectors,
            "xpath": self._extract_with_xpath,
            "content_heuristics": self._extract_with_content_heuristics,
            "ai_guided": self._extract_with_ai if use_ai else None
        }
        
    async def extract(self, 
                  html_content: str, 
                  url: str = None,
                  user_intent: Dict[str, Any] = None,
                  content_type: str = None,
                  extraction_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content using the best strategy based on context
        
        Args:
            html_content: The HTML content to extract from
            url: The URL of the content
            user_intent: User intent dictionary
            content_type: Type of content if known (article, listing, etc.)
            extraction_schema: Schema defining what to extract
            
        Returns:
            Dictionary with extracted content
        """
        if not html_content:
            return {"success": False, "error": "No HTML content provided"}
            
        # Determine content type if not provided
        if not content_type:
            content_type = self._determine_content_type(html_content, user_intent)
            
        # Get strategy order based on content type and context
        strategy_order = self._get_strategy_order(content_type, user_intent)
        
        # Initialize BeautifulSoup for extraction
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try each strategy in order until one succeeds
        extraction_results = {"success": False, "attempts": []}
        
        for strategy_name in strategy_order:
            if strategy_name not in self.strategies or self.strategies[strategy_name] is None:
                continue
                
            strategy_func = self.strategies[strategy_name]
            try:
                result = await strategy_func(
                    soup=soup,
                    html_content=html_content,
                    url=url,
                    user_intent=user_intent,
                    extraction_schema=extraction_schema
                )
                
                # Record the attempt
                attempt_result = {
                    "strategy": strategy_name,
                    "success": result.get("success", False)
                }
                extraction_results["attempts"].append(attempt_result)
                
                # If successful, merge results and return
                if result.get("success", False):
                    extraction_results.update(result)
                    extraction_results["strategy_used"] = strategy_name
                    extraction_results["success"] = True
                    return extraction_results
                    
            except Exception as e:
                logging.error(f"Error with {strategy_name} strategy: {str(e)}")
                extraction_results["attempts"].append({
                    "strategy": strategy_name,
                    "success": False,
                    "error": str(e)
                })
        
        # If we get here, all strategies failed
        extraction_results["error"] = "All extraction strategies failed"
        return extraction_results
        
    def _determine_content_type(self, 
                              html_content: str, 
                              user_intent: Dict[str, Any] = None) -> str:
        """
        Determine the content type from HTML and user intent
        
        Args:
            html_content: HTML content
            user_intent: User intent dictionary
            
        Returns:
            Content type (article, listing, detail, etc.)
        """
        # Default to generic content type
        content_type = "generic"
        
        # Create soup for analysis
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Check for article indicators
        article_indicators = [
            soup.find('article'),
            soup.find('div', class_=lambda c: c and ('article' in c or 'post' in c)),
            soup.find(attrs={"property": "article:published_time"}),
            len(soup.find_all('p')) > 5 and len(''.join([p.get_text() for p in soup.find_all('p')])) > 1000
        ]
        
        # Check for listing indicators
        listing_indicators = [
            len(soup.find_all('li')) > 10,
            len(soup.find_all('div', class_=lambda c: c and ('item' in c or 'product' in c or 'listing' in c or 'result' in c))) > 3,
            soup.find('ul', class_=lambda c: c and ('list' in c or 'results' in c)),
            soup.find('div', class_=lambda c: c and ('grid' in c or 'results' in c))
        ]
        
        # Check for table indicators
        table_indicators = [
            len(soup.find_all('table')) > 0,
            len(soup.find_all('tr')) > 3
        ]
        
        # Determine content type based on indicators
        if any(article_indicators):
            content_type = "article"
        elif any(table_indicators):
            content_type = "data_table"
        elif any(listing_indicators):
            content_type = "listing"
            
        # If user intent has an entity_type, use that to refine the content type
        if user_intent and 'entity_type' in user_intent:
            entity_type = user_intent['entity_type'].lower()
            if entity_type in ['article', 'blog', 'news']:
                content_type = "article"
            elif entity_type in ['product', 'listing', 'item', 'property', 'home', 'house', 'apartment']:
                content_type = "listing"
            elif entity_type in ['table', 'data', 'statistics']:
                content_type = "data_table"
                
        return content_type
        
    def _get_strategy_order(self, 
                          content_type: str, 
                          user_intent: Dict[str, Any] = None) -> List[str]:
        """
        Get the optimal strategy order based on content type and user intent
        
        Args:
            content_type: Type of content
            user_intent: User intent dictionary
            
        Returns:
            List of strategy names in order of priority
        """
        # Default strategy order
        default_order = ["css_selector", "xpath", "content_heuristics"]
        if self.use_ai:
            default_order.append("ai_guided")
            
        # Customize order based on content type
        if content_type == "article":
            return ["content_heuristics", "css_selector", "xpath"] + (["ai_guided"] if self.use_ai else [])
        elif content_type == "listing":
            return ["css_selector", "xpath", "content_heuristics"] + (["ai_guided"] if self.use_ai else [])
        elif content_type == "data_table":
            return ["css_selector", "xpath"] + (["ai_guided"] if self.use_ai else []) + ["content_heuristics"]
            
        # Add AI first if explicitly requested
        if user_intent and user_intent.get("use_ai_extraction", False):
            return ["ai_guided"] + [s for s in default_order if s != "ai_guided"] 
            
        return default_order
    
    async def _extract_with_css_selectors(self, 
                                      soup: BeautifulSoup, 
                                      html_content: str = None,
                                      url: str = None,
                                      user_intent: Dict[str, Any] = None,
                                      extraction_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content using CSS selectors
        
        Args:
            soup: BeautifulSoup object
            html_content: Original HTML
            url: URL of the content
            user_intent: User intent dictionary
            extraction_schema: Schema defining selectors
            
        Returns:
            Dictionary with extracted content
        """
        try:
            if not extraction_schema:
                # Generate schema from user intent if not provided
                extraction_schema = self._generate_extraction_schema(soup, user_intent)
                
            # Use the ContentExtractor's schema extraction method
            result = await self.content_extractor.extract_with_schema(soup, url, extraction_schema)
            
            # Validate the result
            if not result or result.get("error") or (
                "items" in result and not result["items"]
            ):
                return {"success": False, "error": "CSS selector extraction failed"}
                
            return {
                "success": True,
                "extraction_method": "css_selector",
                "content_type": "listing" if "items" in result else "detail",
                "data": result
            }
            
        except Exception as e:
            logging.error(f"CSS selector extraction error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _extract_with_xpath(self, 
                              soup: BeautifulSoup, 
                              html_content: str = None,
                              url: str = None,
                              user_intent: Dict[str, Any] = None,
                              extraction_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content using XPath
        
        Args:
            soup: BeautifulSoup object
            html_content: Original HTML
            url: URL of the content
            user_intent: User intent dictionary
            extraction_schema: Schema defining what to extract
            
        Returns:
            Dictionary with extracted content
        """
        try:
            # Convert BeautifulSoup to lxml Element for XPath
            dom = lxml.html.fromstring(str(soup))
            
            # Generate XPath extraction schema if not provided
            if not extraction_schema or not isinstance(extraction_schema, dict):
                return {"success": False, "error": "No valid extraction schema for XPath"}
                
            # Convert CSS selectors to XPath if needed
            xpath_schema = {}
            for field, selectors in extraction_schema.items():
                if field == "items":
                    # Handle item containers differently
                    xpath_schema[field] = [self._css_to_xpath(s) for s in selectors]
                    continue
                    
                xpath_schema[field] = []
                for selector in selectors:
                    if "::attr(" in selector:
                        # Handle attribute extraction with special XPath syntax
                        css_part, attr_part = selector.split("::attr(")
                        attr_name = attr_part.rstrip(")")
                        xpath = f"{self._css_to_xpath(css_part)}/@{attr_name}"
                    else:
                        xpath = self._css_to_xpath(selector)
                        
                    xpath_schema[field].append(xpath)
                    
            # Extract data using XPath
            result = {"success": True, "extraction_method": "xpath"}
            
            # Handle item extraction
            if "items" in xpath_schema:
                result["items"] = []
                
                # Try each item selector
                for item_xpath in xpath_schema["items"]:
                    try:
                        containers = dom.xpath(item_xpath)
                        if containers:
                            # Found containers, extract items
                            for container in containers[:20]:  # Limit to prevent overload
                                item_data = {}
                                for field, xpaths in {k: v for k, v in xpath_schema.items() if k != "items"}.items():
                                    for xpath in xpaths:
                                        try:
                                            elements = container.xpath(xpath)
                                            if elements:
                                                if field.endswith("_image") or field == "image":
                                                    # Handle image src extraction
                                                    if isinstance(elements[0], str):
                                                        item_data[field] = elements[0]
                                                    else:
                                                        src = elements[0].get("src")
                                                        if not src:
                                                            src = elements[0].get("data-src")
                                                        if src:
                                                            item_data[field] = src
                                                    break
                                                elif field.endswith("_url") or field.endswith("_link") or field == "url" or field == "link":
                                                    # Handle link href extraction
                                                    if isinstance(elements[0], str):
                                                        item_data[field] = elements[0]
                                                    else:
                                                        href = elements[0].get("href")
                                                        if href:
                                                            item_data[field] = href
                                                    break
                                                else:
                                                    # Text content extraction
                                                    text = elements[0].text_content().strip() if hasattr(elements[0], "text_content") else str(elements[0]).strip()
                                                    if text:
                                                        item_data[field] = text
                                                        break
                                        except Exception as e:
                                            continue
                                
                                if item_data:
                                    result["items"].append(item_data)
                                    
                            result["items_selector_used"] = item_xpath
                            result["content_type"] = "listing"
                            break  # Found working selector
                    except Exception as e:
                        continue
            
            # If no items or item extraction failed, try direct extraction
            if "items" not in xpath_schema or not result.get("items"):
                direct_data = {}
                for field, xpaths in {k: v for k, v in xpath_schema.items() if k != "items"}.items():
                    for xpath in xpaths:
                        try:
                            elements = dom.xpath(xpath)
                            if elements:
                                if field.endswith("_image") or field == "image":
                                    # Handle image src extraction
                                    if isinstance(elements[0], str):
                                        direct_data[field] = elements[0]
                                    else:
                                        src = elements[0].get("src")
                                        if not src:
                                            src = elements[0].get("data-src")
                                        if src:
                                            direct_data[field] = src
                                    break
                                elif field.endswith("_url") or field.endswith("_link") or field == "url" or field == "link":
                                    # Handle link href extraction
                                    if isinstance(elements[0], str):
                                        direct_data[field] = elements[0]
                                    else:
                                        href = elements[0].get("href")
                                        if href:
                                            direct_data[field] = href
                                    break
                                else:
                                    # Text content extraction
                                    text = elements[0].text_content().strip() if hasattr(elements[0], "text_content") else str(elements[0]).strip()
                                    if text:
                                        direct_data[field] = text
                                        break
                        except Exception as e:
                            continue
                
                if direct_data:
                    if "items" in result:
                        result.update({k: v for k, v in direct_data.items() if k != "items"})
                    else:
                        result.update(direct_data)
                        result["content_type"] = "detail"
            
            # Check if we extracted anything useful
            if "items" in result and result["items"]:
                return result
            elif {k: v for k, v in result.items() if k not in ["success", "extraction_method", "content_type"]}:
                return result
            else:
                return {"success": False, "error": "XPath extraction yielded no results"}
                
        except Exception as e:
            logging.error(f"XPath extraction error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _css_to_xpath(self, css_selector: str) -> str:
        """Convert CSS selector to XPath"""
        try:
            import cssselect
            translator = cssselect.GenericTranslator()
            return translator.css_to_xpath(css_selector)
        except ImportError:
            # Fallback with basic conversion for common selectors
            if css_selector.startswith('#'):
                return f"//*[@id='{css_selector[1:]}']"
            elif css_selector.startswith('.'):
                return f"//*[contains(@class, '{css_selector[1:]}')]"
            elif ' > ' in css_selector:
                parts = css_selector.split(' > ')
                xpath = ""
                for part in parts:
                    if part.startswith('.'):
                        xpath += f"//*[contains(@class, '{part[1:]}')]"
                    elif part.startswith('#'):
                        xpath += f"//*[@id='{part[1:]}']"
                    else:
                        xpath += f"//{part}"
                return xpath
            else:
                return f"//{css_selector}"
    
    async def _extract_with_content_heuristics(self, 
                                           soup: BeautifulSoup, 
                                           html_content: str = None,
                                           url: str = None,
                                           user_intent: Dict[str, Any] = None,
                                           extraction_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content using content heuristics libraries
        
        Args:
            soup: BeautifulSoup object
            html_content: Original HTML (required for this strategy)
            url: URL of the content
            user_intent: User intent dictionary
            extraction_schema: Schema defining what to extract
            
        Returns:
            Dictionary with extracted content
        """
        if not html_content:
            return {"success": False, "error": "HTML content required for heuristic extraction"}
            
        try:
            # Use ContentExtractor's article extraction which tries multiple libraries
            content_result = await self.content_extractor.extract_content(
                html_content,
                url=url,
                content_type="article"
            )
            
            if not content_result.get("success", False):
                return {"success": False, "error": "Heuristic extraction failed"}
                
            # Convert extracted article to structured data based on user intent
            result = {
                "success": True,
                "extraction_method": content_result.get("extraction_method", "content_heuristics"),
                "content_type": "article",
                "title": content_result.get("title", ""),
                "text": content_result.get("text", ""),
                "html": content_result.get("html", ""),
                "metadata": content_result.get("metadata", {})
            }
            
            # If user intent or extraction schema indicates specific fields,
            # try to extract them from the article content
            if user_intent and "properties" in user_intent or extraction_schema:
                properties = user_intent.get("properties", []) if user_intent else []
                
                # Add properties from extraction schema
                if extraction_schema:
                    for field in extraction_schema:
                        if field != "items" and field not in properties:
                            properties.append(field)
                
                structured_data = {}
                
                # Extract properties from article using basic heuristics
                for prop in properties:
                    # Title is already extracted
                    if prop in ["title", "name", "heading"]:
                        structured_data[prop] = result.get("title", "")
                        continue
                        
                    # Try to find property in content
                    pattern = fr"(?i)(?:{prop}|{prop.replace('_', ' ')})[:\s]+([^\.]+)(?:\.|\n|$)"
                    match = re.search(pattern, result.get("text", ""))
                    if match:
                        structured_data[prop] = match.group(1).strip()
                        
                # Add the structured data to the result
                if structured_data:
                    result["structured_data"] = structured_data
                    
            return result
            
        except Exception as e:
            logging.error(f"Content heuristics extraction error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _extract_with_ai(self, 
                           soup: BeautifulSoup, 
                           html_content: str = None,
                           url: str = None,
                           user_intent: Dict[str, Any] = None,
                           extraction_schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract content using AI
        
        Args:
            soup: BeautifulSoup object
            html_content: Original HTML
            url: URL of the content
            user_intent: User intent dictionary
            extraction_schema: Schema defining what to extract
            
        Returns:
            Dictionary with extracted content
        """
        if not html_content:
            return {"success": False, "error": "HTML content required for AI extraction"}
            
        if not user_intent:
            return {"success": False, "error": "User intent required for AI extraction"}
            
        try:
            # Get desired properties from user intent or extraction schema
            desired_properties = user_intent.get("properties", []) if user_intent else []
            
            # Add properties from extraction schema
            if extraction_schema:
                for field in extraction_schema:
                    if field != "items" and field not in desired_properties:
                        desired_properties.append(field)
                        
            # Get entity type
            entity_type = user_intent.get("entity_type", "item") if user_intent else "item"
            
            # Call the AI extraction function
            items = await extract_content_with_ai(
                html_content=html_content,
                url=url,
                user_intent=user_intent,
                desired_properties=desired_properties,
                entity_type=entity_type
            )
            
            if not items:
                return {"success": False, "error": "AI extraction yielded no results"}
                
            return {
                "success": True,
                "extraction_method": "ai_guided",
                "content_type": "listing" if len(items) > 1 else "detail",
                "items": items if len(items) > 1 else None,
                "data": items[0] if len(items) == 1 else None
            }
            
        except Exception as e:
            logging.error(f"AI extraction error: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _generate_extraction_schema(self, 
                                 soup: BeautifulSoup, 
                                 user_intent: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate extraction schema based on user intent and page structure
        
        Args:
            soup: BeautifulSoup object
            user_intent: User intent dictionary
            
        Returns:
            Extraction schema with selectors
        """
        schema = {}
        
        # Generate item containers selectors if this looks like a listing
        if self._looks_like_listing(soup):
            schema["items"] = self._generate_item_selectors(soup)
            
        # Add selectors for fields based on user intent
        if user_intent and "properties" in user_intent:
            for prop in user_intent["properties"]:
                schema[prop] = self._generate_field_selectors(soup, prop)
                
        return schema
        
    def _looks_like_listing(self, soup: BeautifulSoup) -> bool:
        """Check if the page looks like a listing of items"""
        # Look for common patterns that indicate listings
        repeated_elements = [
            soup.find_all('div', class_=lambda c: c and ('item' in c or 'product' in c or 'card' in c or 'listing' in c or 'result' in c)),
            soup.find_all('li', class_=lambda c: c and ('item' in c or 'product' in c or 'card' in c or 'listing' in c or 'result' in c)),
            soup.find_all('article'),
            soup.find_all('div', attrs={'data-testid': lambda v: v and ('item' in v or 'product' in v or 'card' in v or 'listing' in v or 'result' in v)})
        ]
        
        return any(elements and len(elements) > 2 for elements in repeated_elements)
        
    def _generate_item_selectors(self, soup: BeautifulSoup) -> List[str]:
        """Generate selectors for item containers"""
        selectors = []
        
        # Look for div containers with item-like classes
        item_classes = ['item', 'product', 'card', 'listing', 'result']
        for item_class in item_classes:
            items = soup.find_all('div', class_=lambda c: c and item_class in c)
            if items and len(items) > 2:
                # If we found repeated elements, get their class and create a selector
                classes = items[0].get('class', [])
                if classes:
                    for cls in classes:
                        if item_class in cls:
                            selectors.append(f"div.{cls}")
                            break
        
        # Look for list items with item-like classes
        for item_class in item_classes:
            items = soup.find_all('li', class_=lambda c: c and item_class in c)
            if items and len(items) > 2:
                classes = items[0].get('class', [])
                if classes:
                    for cls in classes:
                        if item_class in cls:
                            selectors.append(f"li.{cls}")
                            break
        
        # Look for articles (common for blog posts, news items)
        articles = soup.find_all('article')
        if articles and len(articles) > 2:
            classes = articles[0].get('class', [])
            if classes:
                selectors.append(f"article.{classes[0]}")
            else:
                selectors.append("article")
        
        # Add generic selectors as fallbacks
        generic_selectors = [
            "div.item", "div.product", "div.card", "div.result", "div.listing",
            "li.item", "li.product", "li.card", "li.result", "li.listing",
            ".product-item", ".search-result", ".list-item", ".grid-item",
            "ul.products > li", "ul.results > li", "div.products > div", "div.results > div"
        ]
        
        # Add generic selectors that aren't already in our list
        for selector in generic_selectors:
            if selector not in selectors:
                selectors.append(selector)
                
        return selectors
        
    def _generate_field_selectors(self, soup: BeautifulSoup, field_name: str) -> List[str]:
        """Generate CSS selectors for a specific field"""
        selectors = []
        
        # Handle common fields with typical selectors
        if field_name in ["title", "name", "heading"] or field_name.endswith("_title") or field_name.endswith("_name"):
            selectors = [
                f"h1.{field_name}", f"h2.{field_name}", f"h3.{field_name}",
                f".{field_name}", f"[data-testid='{field_name}']", f"[data-test='{field_name}']",
                f"[itemprop='{field_name}']", f".product-{field_name}", f".item-{field_name}",
                "h1", "h2", "h3", ".title", ".name", ".product-title", ".item-title"
            ]
        elif field_name in ["price"] or field_name.endswith("_price"):
            selectors = [
                f".{field_name}", f"[data-testid='{field_name}']", f"[data-test='{field_name}']",
                f"[itemprop='{field_name}']", f".product-{field_name}", f".item-{field_name}",
                ".price", ".current-price", ".product-price", "span.price", "div.price"
            ]
        elif field_name in ["image"] or field_name.endswith("_image"):
            selectors = [
                f"img.{field_name}", f"[data-testid='{field_name}']", f"[data-test='{field_name}']",
                f"[itemprop='{field_name}']", f".product-{field_name}", f".item-{field_name}",
                "img.product-image", "img.main-image", "img.primary-image",
                "img::attr(src)", ".product-image img::attr(src)", ".item-image img::attr(src)"
            ]
        elif field_name in ["description"] or field_name.endswith("_description"):
            selectors = [
                f"p.{field_name}", f"div.{field_name}", f"[data-testid='{field_name}']",
                f"[data-test='{field_name}']", f"[itemprop='{field_name}']",
                f".product-{field_name}", f".item-{field_name}",
                ".description", "p.description", "div.description",
                ".product-description", ".short-description"
            ]
        elif field_name in ["url", "link"] or field_name.endswith("_url") or field_name.endswith("_link"):
            selectors = [
                f"a.{field_name}::attr(href)", f"[data-testid='{field_name}']::attr(href)",
                f"[itemprop='{field_name}']::attr(href)", f".product-{field_name}::attr(href)",
                "a.product-link::attr(href)", "a.item-link::attr(href)", 
                "a.title::attr(href)", "h2 a::attr(href)", "h3 a::attr(href)"
            ]
        else:
            # For other fields, create generic selectors based on field name
            # Convert snake_case or camelCase to dash-case and spaces
            field_dash = field_name.replace('_', '-')
            field_space = field_name.replace('_', ' ')
            
            selectors = [
                f".{field_name}", f".{field_dash}", f"[data-testid='{field_name}']",
                f"[data-test='{field_name}']", f"[itemprop='{field_name}']",
                f"div.{field_name}", f"span.{field_name}", f"p.{field_name}",
                f".product-{field_dash}", f".item-{field_dash}"
            ]
            
            # Add selectors that look for elements containing the field name text
            selectors.append(f"dt:contains('{field_space}') + dd")
            selectors.append(f"th:contains('{field_space}') + td")
            selectors.append(f"div:contains('{field_space}') span")
            
        return selectors
# =================================================================
    # SPACY-BASED ENHANCEMENT METHODS FOR TASK 5.1
    # =================================================================
    
    def apply_spacy_post_extraction_filtering(self, 
                                            extracted_content: Dict[str, Any], 
                                            filter_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply spaCy-based post-extraction filtering to improve content quality.
        
        Args:
            extracted_content: Content extracted by primary extraction methods
            filter_config: Configuration for filtering (min_sentence_length, pos_tags_to_remove, etc.)
        
        Returns:
            Filtered and enhanced content dictionary
        """
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, skipping post-extraction filtering")
            return extracted_content
        
        filter_config = filter_config or {}
        min_sentence_length = filter_config.get("min_sentence_length", 10)
        pos_tags_to_remove = filter_config.get("pos_tags_to_remove", ["PUNCT", "SPACE", "SYM"])
        min_words_per_sentence = filter_config.get("min_words_per_sentence", 5)
        
        try:
            # Load spaCy model with fallback chain
            SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
            nlp = None
            for model_name in SPACY_MODELS:
                try:
                    nlp = spacy.load(model_name)
                    break
                except OSError:
                    continue
            
            if nlp is None:
                try:
                    nlp = spacy.load("en")
                except OSError:
                    logger.warning("No spaCy model available, skipping linguistic filtering")
                    return extracted_content
            
            # Create enhanced content copy
            enhanced_content = extracted_content.copy()
            
            # Filter text content
            if "text" in extracted_content and extracted_content["text"]:
                enhanced_content["text"] = self._filter_text_with_spacy(
                    extracted_content["text"], nlp, min_sentence_length, 
                    pos_tags_to_remove, min_words_per_sentence
                )
            
            # Filter title if present
            if "title" in extracted_content and extracted_content["title"]:
                enhanced_content["title"] = self._filter_title_with_spacy(
                    extracted_content["title"], nlp
                )
            
            # Filter items if present (for listings)
            if "items" in extracted_content and isinstance(extracted_content["items"], list):
                enhanced_content["items"] = self._filter_items_with_spacy(
                    extracted_content["items"], nlp, filter_config
                )
            
            # Add linguistic metadata
            enhanced_content["linguistic_metadata"] = self._extract_linguistic_metadata(
                enhanced_content.get("text", ""), nlp
            )
            
            # Add filtering statistics
            enhanced_content["spacy_filtering_applied"] = True
            enhanced_content["spacy_filtering_stats"] = {
                "original_text_length": len(extracted_content.get("text", "")),
                "filtered_text_length": len(enhanced_content.get("text", "")),
                "filter_config": filter_config
            }
            
            return enhanced_content
            
        except Exception as e:
            logger.error(f"Error in spaCy post-extraction filtering: {str(e)}")
            return extracted_content
    
    def _filter_text_with_spacy(self, text: str, nlp, min_sentence_length: int, 
                              pos_tags_to_remove: List[str], min_words_per_sentence: int) -> str:
        """Filter text content using spaCy linguistic analysis."""
        try:
            doc = nlp(text)
            filtered_sentences = []
            
            for sent in doc.sents:
                # Skip very short sentences
                if len(sent.text.strip()) < min_sentence_length:
                    continue
                
                # Filter tokens by POS tags
                filtered_tokens = []
                for token in sent:
                    if token.pos_ not in pos_tags_to_remove and not token.is_stop:
                        if token.text.strip() and len(token.text.strip()) > 1:
                            filtered_tokens.append(token.text)
                
                # Keep sentence if it has enough meaningful words
                if len(filtered_tokens) >= min_words_per_sentence:
                    # Reconstruct sentence from filtered tokens
                    filtered_sentence = " ".join(filtered_tokens)
                    filtered_sentences.append(filtered_sentence)
            
            return ". ".join(filtered_sentences) + "." if filtered_sentences else text
            
        except Exception as e:
            logger.warning(f"Error filtering text with spaCy: {str(e)}")
            return text
    
    def _filter_title_with_spacy(self, title: str, nlp) -> str:
        """Filter and clean title using spaCy."""
        try:
            doc = nlp(title)
            
            # Extract meaningful tokens (remove stop words, punctuation)
            filtered_tokens = []
            for token in doc:
                if not token.is_stop and not token.is_punct and token.text.strip():
                    if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB", "NUM"]:
                        filtered_tokens.append(token.text)
            
            # Rebuild title if we have meaningful tokens
            if filtered_tokens:
                return " ".join(filtered_tokens)
            else:
                return title
                
        except Exception as e:
            logger.warning(f"Error filtering title with spaCy: {str(e)}")
            return title
    
    def _filter_items_with_spacy(self, items: List[Dict], nlp, filter_config: Dict) -> List[Dict]:
        """Filter items in listings using spaCy."""
        filtered_items = []
        
        for item in items:
            try:
                filtered_item = item.copy()
                
                # Filter text fields in the item
                for field_name, field_value in item.items():
                    if isinstance(field_value, str) and field_value.strip():
                        if field_name in ["title", "name", "description"]:
                            if field_name == "title" or field_name == "name":
                                filtered_item[field_name] = self._filter_title_with_spacy(field_value, nlp)
                            else:
                                filtered_item[field_name] = self._filter_text_with_spacy(
                                    field_value, nlp, 
                                    filter_config.get("min_sentence_length", 10),
                                    filter_config.get("pos_tags_to_remove", ["PUNCT", "SPACE", "SYM"]),
                                    filter_config.get("min_words_per_sentence", 3)
                                )
                
                filtered_items.append(filtered_item)
                
            except Exception as e:
                logger.warning(f"Error filtering item with spaCy: {str(e)}")
                filtered_items.append(item)
        
        return filtered_items
    
    def _extract_linguistic_metadata(self, text: str, nlp) -> Dict[str, Any]:
        """Extract linguistic metadata using spaCy."""
        try:
            if not text.strip():
                return {}
            
            doc = nlp(text)
            
            # Count POS tags
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            
            # Extract entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Calculate linguistic complexity
            avg_sentence_length = sum(len(sent) for sent in doc.sents) / len(list(doc.sents)) if list(doc.sents) else 0
            
            return {
                "sentence_count": len(list(doc.sents)),
                "token_count": len(doc),
                "pos_tag_distribution": pos_counts,
                "named_entities": entities[:10],  # Limit to first 10
                "avg_sentence_length": avg_sentence_length,
                "language_detected": doc.lang_ if hasattr(doc, 'lang_') else "unknown"
            }
            
        except Exception as e:
            logger.warning(f"Error extracting linguistic metadata: {str(e)}")
            return {}
    
    def detect_duplicate_content_advanced(self, 
                                        content_list: List[Dict[str, Any]], 
                                        similarity_threshold: float = 0.8,
                                        fields_to_compare: List[str] = None) -> Dict[str, Any]:
        """
        Detect duplicate content using advanced semantic similarity with spaCy.
        
        Args:
            content_list: List of content dictionaries to check for duplicates
            similarity_threshold: Threshold for considering content as duplicate (0.0-1.0)
            fields_to_compare: Fields to compare for similarity (default: ["title", "text", "description"])
        
        Returns:
            Dictionary with duplicate detection results and similarity matrix
        """
        if not content_list or len(content_list) < 2:
            return {
                "duplicates_found": False,
                "duplicate_groups": [],
                "similarity_matrix": [],
                "stats": {"total_items": len(content_list), "duplicate_count": 0}
            }
        
        fields_to_compare = fields_to_compare or ["title", "text", "description"]
        
        # If spaCy is not available, fall back to simple text comparison
        if not SPACY_AVAILABLE:
            return self._detect_duplicates_simple(content_list, similarity_threshold, fields_to_compare)
        
        try:
            # Load spaCy model with word vectors - prioritize large model
            SPACY_MODELS = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
            nlp = None
            for model_name in SPACY_MODELS:
                try:
                    nlp = spacy.load(model_name)
                    break
                except OSError:
                    continue
                    logger.warning("Using small spaCy model without word vectors for duplicate detection")
                except OSError:
                    logger.warning("No spaCy model available, using simple duplicate detection")
                    return self._detect_duplicates_simple(content_list, similarity_threshold, fields_to_compare)
            
            # Extract text representations for comparison
            text_representations = []
            for item in content_list:
                combined_text = ""
                for field in fields_to_compare:
                    if field in item and item[field]:
                        combined_text += str(item[field]) + " "
                text_representations.append(combined_text.strip())
            
            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(text_representations, nlp)
            
            # Find duplicate groups
            duplicate_groups = self._find_duplicate_groups(similarity_matrix, similarity_threshold)
            
            # Calculate statistics
            duplicate_count = sum(len(group) for group in duplicate_groups if len(group) > 1)
            
            return {
                "duplicates_found": len(duplicate_groups) > 0,
                "duplicate_groups": duplicate_groups,
                "similarity_matrix": similarity_matrix,
                "stats": {
                    "total_items": len(content_list),
                    "duplicate_count": duplicate_count,
                    "unique_groups": len(duplicate_groups),
                    "similarity_threshold": similarity_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced duplicate detection: {str(e)}")
            return self._detect_duplicates_simple(content_list, similarity_threshold, fields_to_compare)
    
    def _calculate_similarity_matrix(self, text_list: List[str], nlp) -> List[List[float]]:
        """Calculate semantic similarity matrix using spaCy."""
        docs = [nlp(text) for text in text_list]
        n = len(docs)
        similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    try:
                        # Use spaCy's built-in similarity if available
                        if hasattr(docs[i], 'similarity'):
                            sim = docs[i].similarity(docs[j])
                        else:
                            # Fallback to simple token overlap
                            sim = self._calculate_token_similarity(docs[i], docs[j])
                        
                        similarity_matrix[i][j] = sim
                        similarity_matrix[j][i] = sim
                    except Exception as e:
                        logger.warning(f"Error calculating similarity between items {i} and {j}: {str(e)}")
                        similarity_matrix[i][j] = 0.0
                        similarity_matrix[j][i] = 0.0
        
        return similarity_matrix
    
    def _calculate_token_similarity(self, doc1, doc2) -> float:
        """Calculate simple token-based similarity when word vectors are not available."""
        tokens1 = set(token.lemma_.lower() for token in doc1 if not token.is_stop and not token.is_punct)
        tokens2 = set(token.lemma_.lower() for token in doc2 if not token.is_stop and not token.is_punct)
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_duplicate_groups(self, similarity_matrix: List[List[float]], 
                              threshold: float) -> List[List[int]]:
        """Find groups of duplicate items based on similarity matrix."""
        n = len(similarity_matrix)
        visited = [False] * n
        duplicate_groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Find all items similar to item i
            group = [i]
            visited[i] = True
            
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i][j] >= threshold:
                    group.append(j)
                    visited[j] = True
            
            # Only add groups with duplicates
            if len(group) > 1:
                duplicate_groups.append(group)
        return duplicate_groups
    
    def _detect_duplicates_simple(self, content_list: List[Dict[str, Any]], 
                                 similarity_threshold: float, 
                                 fields_to_compare: List[str]) -> Dict[str, Any]:
        """Simple duplicate detection using basic string similarity."""
        duplicate_groups = []
        used_indices = set()
        
        for i in range(len(content_list)):
            if i in used_indices:
                continue
            
            group = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(content_list)):
                if j in used_indices:
                    continue
                
                # Calculate simple similarity
                similarity = self._calculate_simple_similarity(
                    content_list[i], content_list[j], fields_to_compare
                )
                
                if similarity >= similarity_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        duplicate_count = sum(len(group) for group in duplicate_groups)
        
        return {
            "duplicates_found": len(duplicate_groups) > 0,
            "duplicate_groups": duplicate_groups,
            "similarity_matrix": [],  # Not calculated for simple method
            "stats": {
                "total_items": len(content_list),
                "duplicate_count": duplicate_count,
                "unique_groups": len(duplicate_groups),
                "similarity_threshold": similarity_threshold
            }
        }
    
    def _calculate_simple_similarity(self, item1: Dict, item2: Dict, 
                                   fields_to_compare: List[str]) -> float:
        """Calculate simple string-based similarity between two items."""
        similarities = []
        
        for field in fields_to_compare:
            if field in item1 and field in item2:
                text1 = str(item1[field]).lower().strip()
                text2 = str(item2[field]).lower().strip()
                
                if text1 and text2:
                    # Simple Jaccard similarity on words
                    words1 = set(text1.split())
                    words2 = set(text2.split())
                    
                    if words1 or words2:
                        intersection = words1.intersection(words2)
                        union = words1.union(words2)
                        similarity = len(intersection) / len(union) if union else 0.0
                        similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def score_content_semantic_relevance(self, 
                                      content: Dict[str, Any], 
                                      query_context: Dict[str, Any] = None,
                                      user_intent: str = None) -> Dict[str, Any]:
        """
        Score content semantic relevance using spaCy and advanced NLP techniques.
        
        Args:
            content: Content dictionary to score
            query_context: Dictionary containing query/search context
            user_intent: User's search intent or query string
        
        Returns:
            Dictionary with relevance scores and detailed analysis
        """
        try:
            # Initialize from our ContentQualityScorer
            from processors.content_quality_scorer import ContentQualityScorer
            quality_scorer = ContentQualityScorer()
            
            # Use the comprehensive scoring from ContentQualityScorer
            quality_result = quality_scorer.score_content(content, query_context, user_intent)
            
            # Add ContentExtractor-specific enhancements
            enhanced_result = quality_result.copy()
            
            # Add extraction-specific scoring
            if "extraction_method" in content:
                extraction_method_score = self._score_extraction_method(content["extraction_method"])
                enhanced_result["extraction_method_score"] = extraction_method_score
                enhanced_result["overall_score"] = (
                    enhanced_result["overall_score"] * 0.8 + extraction_method_score * 0.2
                )
            
            # Add linguistic complexity scoring if spaCy filtering was applied
            if content.get("spacy_filtering_applied", False) and "linguistic_metadata" in content:
                complexity_score = self._score_linguistic_complexity(content["linguistic_metadata"])
                enhanced_result["linguistic_complexity_score"] = complexity_score
                enhanced_result["overall_score"] = (
                    enhanced_result["overall_score"] * 0.9 + complexity_score * 0.1
                )
            
            # Add metadata scoring
            metadata_score = self._score_metadata_quality(content.get("metadata", {}))
            enhanced_result["metadata_quality_score"] = metadata_score
            enhanced_result["overall_score"] = (
                enhanced_result["overall_score"] * 0.95 + metadata_score * 0.05
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in semantic relevance scoring: {str(e)}")
            return {
                "overall_score": 0.5,
                "error": str(e),
                "scores": {},
                "analysis": "Error occurred during scoring"
            }
    
    def _score_extraction_method(self, extraction_method: str) -> float:
        """Score content based on the extraction method used."""
        method_scores = {
            "semantic": 0.95,
            "trafilatura": 0.90,
            "readability": 0.85,
            "goose": 0.80,
            "justext": 0.75,
            "fallback": 0.60,
            "heuristic": 0.70,
            "pattern": 0.75
        }
        return method_scores.get(extraction_method, 0.65)
    
    def _score_linguistic_complexity(self, linguistic_metadata: Dict[str, Any]) -> float:
        """Score content based on linguistic complexity indicators."""
        try:
            score = 0.5  # Base score
            
            # Sentence count scoring
            sentence_count = linguistic_metadata.get("sentence_count", 0)
            if sentence_count > 5:
                score += 0.1
            if sentence_count > 15:
                score += 0.1
            
            # Average sentence length scoring
            avg_length = linguistic_metadata.get("avg_sentence_length", 0)
            if 10 <= avg_length <= 25:  # Optimal range
                score += 0.15
            elif avg_length > 25:
                score += 0.05  # Too long sentences
            
            # POS tag diversity scoring
            pos_distribution = linguistic_metadata.get("pos_tag_distribution", {})
            if pos_distribution:
                diversity = len(pos_distribution)
                if diversity >= 8:  # Good diversity
                    score += 0.1
                if diversity >= 12:  # Excellent diversity
                    score += 0.1
            
            # Named entities scoring
            entities = linguistic_metadata.get("named_entities", [])
            if entities:
                score += min(0.15, len(entities) * 0.02)  # Up to 0.15 for entities
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Error scoring linguistic complexity: {str(e)}")
            return 0.5
    
    def _score_metadata_quality(self, metadata: Dict[str, Any]) -> float:
        """Score content based on metadata quality and completeness."""
        try:
            score = 0.0
            total_possible = 0.0
            
            # Check for common metadata fields
            metadata_fields = {
                "title": 0.2,
                "description": 0.15,
                "keywords": 0.1,
                "author": 0.1,
                "published_date": 0.1,
                "url": 0.05,
                "language": 0.05,
                "content_type": 0.05,
                "image": 0.05,
                "canonical_url": 0.05,
                "extraction_time": 0.05,
                "word_count": 0.05
            }
            
            for field, weight in metadata_fields.items():
                total_possible += weight
                if field in metadata and metadata[field]:
                    score += weight
            
            return score / total_possible if total_possible > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error scoring metadata quality: {str(e)}")
            return 0.0
    
    def enhance_extraction_with_semantic_analysis(self, 
                                                  extraction_result: Dict[str, Any],
                                                  enhancement_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance extraction results with comprehensive semantic analysis.
        
        Args:
            extraction_result: Initial extraction result
            enhancement_config: Configuration for enhancement options
        
        Returns:
            Enhanced extraction result with semantic analysis
        """
        enhancement_config = enhancement_config or {}
        
        try:
            # Create enhanced result copy
            enhanced_result = extraction_result.copy()
            
            # Apply spaCy-based filtering if enabled
            if enhancement_config.get("apply_spacy_filtering", True):
                enhanced_result = self.apply_spacy_post_extraction_filtering(
                    enhanced_result, 
                    enhancement_config.get("filter_config", {})
                )
            
            # Detect and handle duplicates if working with lists
            if "items" in enhanced_result and isinstance(enhanced_result["items"], list):
                if enhancement_config.get("detect_duplicates", True):
                    duplicate_result = self.detect_duplicate_content_advanced(
                        enhanced_result["items"],
                        enhancement_config.get("similarity_threshold", 0.8)
                    )
                    
                    # Remove duplicates if requested
                    if duplicate_result["duplicates_found"] and enhancement_config.get("remove_duplicates", True):
                        enhanced_result["items"] = self._remove_duplicate_items(
                            enhanced_result["items"], duplicate_result["duplicate_groups"]
                        )
                    
                    enhanced_result["duplicate_analysis"] = duplicate_result
            
            # Score content relevance
            if enhancement_config.get("score_relevance", True):
                relevance_score = self.score_content_semantic_relevance(
                    enhanced_result,
                    enhancement_config.get("query_context"),
                    enhancement_config.get("user_intent")
                )
                enhanced_result["relevance_analysis"] = relevance_score
            
            # Add enhancement metadata
            enhanced_result["semantic_enhancement_applied"] = True
            enhanced_result["enhancement_config"] = enhancement_config
            enhanced_result["enhancement_timestamp"] = self._get_timestamp()
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in semantic enhancement: {str(e)}")
            # Return original result with error info
            extraction_result["semantic_enhancement_error"] = str(e)
            return extraction_result
    
    def _remove_duplicate_items(self, items: List[Dict], duplicate_groups: List[List[int]]) -> List[Dict]:
        """Remove duplicate items, keeping the first item from each duplicate group."""
        indices_to_remove = set()
        
        for group in duplicate_groups:
            # Keep the first item, remove the rest
            for index in group[1:]:
                indices_to_remove.add(index)
        
        # Create new list without duplicates
        return [item for i, item in enumerate(items) if i not in indices_to_remove]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()

    # =================================================================
    # END OF SPACY-BASED ENHANCEMENT METHODS
    # =================================================================