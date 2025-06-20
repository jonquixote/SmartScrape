import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from bs4 import BeautifulSoup, Tag, NavigableString
import html2text
# spaCy imports with fallback
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    # Try to load the English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        nlp = None
        logger = logging.getLogger("ContentProcessor")
        logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    nlp = None
    STOP_WORDS = set()
import trafilatura

class ContentProcessor:
    """Service for preprocessing content to reduce token usage and improve AI response quality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("content_processor")
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.ignore_tables = False
        self.h2t.body_width = 0  # No wrapping
        
        # Initialize NLP components with spaCy first, fallback to basic methods
        if SPACY_AVAILABLE:
            self.nlp = nlp
            self.stop_words = STOP_WORDS
        else:
            self.nlp = None
            # Basic fallback stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                'further', 'then', 'once'
            }
    
    def preprocess_html(self, html_content: str, extract_main: bool = True, 
                       max_tokens: Optional[int] = None) -> str:
        """Process HTML content to reduce token usage while preserving essential information."""
        if not html_content:
            return ""
            
        try:
            if extract_main:
                # Extract main content using trafilatura
                extracted_text = trafilatura.extract(
                    html_content,
                    include_links=True,
                    include_images=False,
                    include_tables=True,
                    no_fallback=False
                )
                
                # If trafilatura extraction fails, try fallback methods
                if not extracted_text:
                    extracted_text = self._extract_main_fallback(html_content)
            else:
                # Simple HTML to text conversion
                extracted_text = self.h2t.handle(html_content)
                
            # Clean up the extracted text
            cleaned_text = self._clean_text(extracted_text)
            
            # Truncate to max tokens if specified
            if max_tokens:
                cleaned_text = self._truncate_to_max_tokens(cleaned_text, max_tokens)
                
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error preprocessing HTML: {str(e)}")
            # Fall back to simple conversion
            return self.h2t.handle(html_content)
    
    def _extract_main_fallback(self, html_content: str) -> str:
        """Fallback method to extract main content from HTML when trafilatura fails."""
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove navigation, header, footer, sidebars, ads
            for tag_name in ['nav', 'header', 'footer', 'aside', 'script', 'style', 'noscript']:
                for tag in soup.find_all(tag_name):
                    tag.decompose()
                    
            # Remove by common navigation/header/footer/ad class and id patterns
            nav_patterns = [
                'nav', 'navigation', 'menu', 'header', 'footer', 'sidebar', 
                'widget', 'banner', 'ad-', 'ad_', 'advert', 'cookie', 'popup'
            ]
            
            for pattern in nav_patterns:
                for tag in soup.find_all(class_=re.compile(pattern, re.IGNORECASE)):
                    tag.decompose()
                for tag in soup.find_all(id=re.compile(pattern, re.IGNORECASE)):
                    tag.decompose()
            
            # Try to find the main content
            main_content = None
            
            # Common content containers in priority order
            content_selectors = [
                "article", "main", "#content", "#main", ".content", ".main", ".post",
                ".article", ".entry", ".entry-content", ".post-content", "div.content", 
                "div.main", "[role=main]"
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content and len(main_content.get_text(strip=True)) > 200:
                    break
            
            # If no suitable container found, use body with cleaned content
            if not main_content or len(main_content.get_text(strip=True)) < 200:
                main_content = soup.body
            
            if (main_content):
                return self.h2t.handle(str(main_content))
            else:
                return self.h2t.handle(html_content)
                
        except Exception as e:
            self.logger.error(f"Error in main content extraction fallback: {str(e)}")
            return self.h2t.handle(html_content)
    
    def _clean_text(self, text: str) -> str:
        """Clean up extracted text to improve quality."""
        if not text:
            return ""
            
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Remove repeated newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Clean up list formatting
        cleaned = re.sub(r'\n\s*\*\s+', '\n* ', cleaned)
        
        # Remove HTML entities
        cleaned = re.sub(r'&[a-zA-Z]+;', ' ', cleaned)
        
        # Remove URLs if they're not part of a markdown link
        cleaned = re.sub(r'(?<!\]\()https?://\S+', '', cleaned)
        
        return cleaned
    
    def _truncate_to_max_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens while preserving sentence boundaries."""
        # Rough approximation: 1 token ≈ 4 characters
        if len(text) <= max_tokens * 4:
            return text
            
        # Try to truncate at sentence boundaries using spaCy first, fallback to basic methods
        if SPACY_AVAILABLE and self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Fallback to basic sentence splitting
            sentences = re.split(r'[.!?]+', text.strip())
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        result = []
        current_length = 0
        
        for sentence in sentences:
            # Approximately calculate tokens in this sentence
            sentence_tokens = len(sentence) // 4
            
            if current_length + sentence_tokens <= max_tokens:
                result.append(sentence)
                current_length += sentence_tokens
            else:
                # If the first sentence is already too long, truncate it
                if not result:
                    return sentence[:max_tokens * 4] + "..."
                break
                
        return ' '.join(result)
    
    def chunk_content(self, content: str, max_tokens: int = 4000, 
                     overlap: int = 200) -> List[str]:
        """Split content into chunks of approximately max_tokens with optional overlap."""
        if not content:
            return []
            
        # If content is short enough, return it as a single chunk
        if len(content) // 4 <= max_tokens:  # Rough token estimate
            return [content]
            
        # Split into sentences using spaCy first, fallback to basic methods
        if SPACY_AVAILABLE and self.nlp:
            doc = self.nlp(content)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Fallback to basic sentence splitting
            sentences = re.split(r'[.!?]+', content.strip())
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            # Approximately calculate tokens in this sentence
            sentence_tokens = len(sentence) // 4
            
            # If adding this sentence would exceed max_tokens, start a new chunk
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlapping content from end of previous chunk
                if overlap > 0 and current_chunk:
                    # Get overlap from previous chunk if possible
                    overlap_start = max(0, len(current_chunk) - (overlap // 4))
                    current_chunk = current_chunk[overlap_start:]
                    current_tokens = sum(len(s) // 4 for s in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def summarize_content(self, content: str, ratio: float = 0.3, 
                         min_length: int = 100, max_length: int = 1000) -> str:
        """Create an extractive summary of the content."""
        if not content or len(content) <= min_length:
            return content
            
        try:
            # Tokenize content into sentences using spaCy first, fallback to basic methods
            if SPACY_AVAILABLE and self.nlp:
                doc = self.nlp(content)
                sentences = [sent.text for sent in doc.sents]
            else:
                # Fallback to basic sentence splitting
                sentences = re.split(r'[.!?]+', content.strip())
                sentences = [s.strip() + '.' for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                return content
                
            # Calculate word frequency using spaCy first, fallback to basic methods
            word_freq = {}
            
            if SPACY_AVAILABLE and self.nlp:
                # Use spaCy for advanced tokenization and stopword detection
                for sentence in sentences:
                    doc = self.nlp(sentence.lower())
                    for token in doc:
                        if not token.is_stop and token.is_alpha and len(token.text) > 2:
                            lemma = token.lemma_.lower()
                            word_freq[lemma] = word_freq.get(lemma, 0) + 1
            else:
                # Fallback to basic tokenization
                for sentence in sentences:
                    words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
                    for word in words:
                        if word not in self.stop_words and word.isalnum() and len(word) > 2:
                            word_freq[word] = word_freq.get(word, 0) + 1
            
            # Calculate sentence scores based on word frequency
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                score = 0
                word_count = 0
                
                if SPACY_AVAILABLE and self.nlp:
                    doc = self.nlp(sentence.lower())
                    for token in doc:
                        if token.is_alpha:
                            word_count += 1
                            lemma = token.lemma_.lower()
                            if lemma in word_freq:
                                score += word_freq[lemma]
                else:
                    words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
                    word_count = len(words)
                    for word in words:
                        if word in word_freq:
                            score += word_freq[word]
                            
                sentence_scores[i] = score / max(1, word_count)
            
            # Determine number of sentences for summary
            target_sent_count = max(3, int(len(sentences) * ratio))
            
            # Get top sentences
            top_indices = sorted(sentence_scores.keys(), 
                              key=lambda i: sentence_scores[i], 
                              reverse=True)[:target_sent_count]
            
            # Sort indices to maintain original order
            top_indices = sorted(top_indices)
            
            # Construct summary
            summary = ' '.join(sentences[i] for i in top_indices)
            
            # Truncate if too long
            if max_length and len(summary) > max_length:
                summary = self._truncate_to_max_tokens(summary, max_length // 4)
                
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing content: {str(e)}")
            # Fallback to simple truncation
            return content[:max_length] if max_length else content
    
    def extract_keywords(self, content: str, top_n: int = 10) -> List[str]:
        """Extract the most important keywords from the content."""
        if not content:
            return []
            
        try:
            # Extract keywords using spaCy first, fallback to basic methods
            if SPACY_AVAILABLE and self.nlp:
                doc = self.nlp(content.lower())
                words = [token.lemma_.lower() for token in doc 
                        if token.is_alpha and not token.is_stop and len(token.text) > 2]
            else:
                # Fallback to basic tokenization
                words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
                words = [word for word in words 
                        if word.isalnum() and word not in self.stop_words and len(word) > 2]
            
            # Calculate word frequency
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N keywords
            return [word for word, _ in sorted_words[:top_n]]
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def format_structured_content(self, structured_data: Dict[str, Any]) -> str:
        """Convert structured data to a formatted string for AI processing."""
        if not structured_data:
            return ""
            
        # Format as Markdown
        lines = []
        
        for key, value in structured_data.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"## {key}")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._format_dict_as_list(item, indent=1))
                    else:
                        lines.append(f"- {item}")
            elif isinstance(value, dict):
                lines.append(f"## {key}")
                lines.append(self._format_dict_as_list(value, indent=1))
            else:
                lines.append(f"## {key}")
                lines.append(str(value))
                
            lines.append("")  # Add blank line between sections
        
        return "\n".join(lines)
    
    def _format_dict_as_list(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Format a dictionary as a nested list."""
        lines = []
        prefix = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(self._format_dict_as_list(value, indent + 1))
            elif isinstance(value, (list, tuple)):
                lines.append(f"{prefix}- **{key}**:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._format_dict_as_list(item, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}- **{key}**: {value}")
                
        return "\n".join(lines)