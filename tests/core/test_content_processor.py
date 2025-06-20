import pytest
from core.content_processor import ContentProcessor

class TestContentProcessor:
    def setup_method(self):
        self.processor = ContentProcessor()
    
    def test_preprocess_html(self):
        # Test with simple HTML
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <header>Site Header</header>
                <nav>Navigation</nav>
                <div id="content">
                    <h1>Test Content</h1>
                    <p>This is the main content.</p>
                </div>
                <footer>Copyright 2023</footer>
            </body>
        </html>
        """
        
        processed = self.processor.preprocess_html(html)
        
        # Header, nav, and footer should be removed
        assert "Site Header" not in processed.lower()
        assert "Navigation" not in processed
        assert "Copyright" not in processed
        
        # Main content should be preserved
        assert "Test Content" in processed
        assert "This is the main content" in processed
        
        # Test with empty input
        assert self.processor.preprocess_html("") == ""
        
        # Test with non-HTML
        plain_text = "Just a plain text without any HTML"
        processed = self.processor.preprocess_html(plain_text)
        assert plain_text in processed
    
    def test_chunk_content(self):
        # Create a long text with multiple sentences
        sentences = ["This is sentence {}. ".format(i) * 5 for i in range(50)]
        long_text = " ".join(sentences)
        
        # Chunk with max tokens that should create multiple chunks
        max_tokens = 500  # Approximately 2000 characters
        chunks = self.processor.chunk_content(long_text, max_tokens)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be smaller than max_tokens (with some margin for approximation)
        for chunk in chunks:
            assert len(chunk) // 4 <= max_tokens * 1.1  # Allow 10% margin
        
        # Test with small text that fits in one chunk
        small_text = "This is a short text that should fit in one chunk."
        chunks = self.processor.chunk_content(small_text, max_tokens)
        assert len(chunks) == 1
        assert chunks[0] == small_text
        
        # Test with empty input
        assert self.processor.chunk_content("") == []
        
        # Test overlap feature
        overlap_chunks = self.processor.chunk_content(long_text, max_tokens, overlap=100)
        assert len(overlap_chunks) > 1
    
    def test_summarize_content(self):
        # Create a text with some clear important sentences
        text = """
        Artificial intelligence (AI) is revolutionizing various industries.
        The concept of machine learning is a subset of AI.
        Deep learning is a type of machine learning that uses neural networks.
        Neural networks are inspired by the human brain.
        There are many applications of AI in healthcare.
        AI can help diagnose diseases and develop new treatments.
        Some people are concerned about the ethical implications of AI.
        Privacy concerns are one of the main ethical issues.
        The future of AI is bright but requires careful consideration.
        """
        
        # Summarize to about 30% of original length
        summary = self.processor.summarize_content(text, ratio=0.3)
        
        # Summary should be shorter than original but contain key information
        assert len(summary) < len(text)
        assert "artificial intelligence" in summary.lower() or "ai" in summary.lower()
        
        # Test with very short text that shouldn't be summarized
        short_text = "This is a very short text."
        summary = self.processor.summarize_content(short_text)
        assert summary == short_text
        
        # Test with empty input
        assert self.processor.summarize_content("") == ""
        
        # Test with custom ratio
        custom_summary = self.processor.summarize_content(text, ratio=0.1)
        assert len(custom_summary) < len(summary)
    
    def test_extract_keywords(self):
        text = """
        Machine learning is a field of artificial intelligence that uses statistical techniques
        to give computer systems the ability to learn from data, without being explicitly programmed.
        The name machine learning was coined in 1959 by Arthur Samuel.
        """
        
        keywords = self.processor.extract_keywords(text, top_n=5)
        
        # Should contain relevant keywords
        assert len(keywords) <= 5
        expected_keywords = ["machine", "learning", "artificial", "intelligence", "data"]
        assert any(keyword in expected_keywords for keyword in keywords)
        
        # Common stopwords should be excluded
        assert "the" not in keywords
        assert "from" not in keywords
        
        # Test with empty text
        assert self.processor.extract_keywords("") == []
        
        # Test with custom top_n
        all_keywords = self.processor.extract_keywords(text, top_n=20)
        assert len(all_keywords) > len(keywords)
    
    def test_format_structured_content(self):
        # Test with a nested structure
        data = {
            "product": "Smartphone XYZ",
            "specifications": {
                "screen": "6.5 inch OLED",
                "processor": "Snapdragon 8 Gen 1",
                "memory": "8GB RAM",
                "storage": ["128GB", "256GB", "512GB"]
            },
            "reviews": [
                {"user": "Alice", "rating": 5, "comment": "Excellent product!"},
                {"user": "Bob", "rating": 4, "comment": "Good but expensive"}
            ]
        }
        
        formatted = self.processor.format_structured_content(data)
        
        # Check that all values are included
        assert "Smartphone XYZ" in formatted
        assert "6.5 inch OLED" in formatted
        assert "128GB" in formatted
        assert "Alice" in formatted
        assert "Excellent product!" in formatted
        
        # Check formatting
        assert "## product" in formatted
        assert "## specifications" in formatted
        assert "## reviews" in formatted
        assert "**screen**" in formatted
        assert "**user**" in formatted
        
        # Test with empty input
        assert self.processor.format_structured_content({}) == ""
        
        # Test with simple flat structure
        flat_data = {"name": "John", "age": 30, "occupation": "Developer"}
        flat_formatted = self.processor.format_structured_content(flat_data)
        assert "## name" in flat_formatted
        assert "John" in flat_formatted
        assert "## age" in flat_formatted
        assert "30" in flat_formatted
    
    def test_clean_text(self):
        # Test with text containing excessive whitespace
        text = "  This  has   too    much     whitespace.  \n\n\n  And too many newlines.  "
        cleaned = self.processor._clean_text(text)
        assert "  This  has   too    much     whitespace." not in cleaned
        assert "This has too much whitespace." in cleaned
        assert "\n\n\n" not in cleaned
        
        # Test with HTML entities
        text_with_entities = "Text with &nbsp; and &quot;entities&quot;"
        cleaned = self.processor._clean_text(text_with_entities)
        assert "&nbsp;" not in cleaned
        assert "&quot;" not in cleaned
        
        # Test with URLs
        text_with_urls = "Check out https://example.com for more info"
        cleaned = self.processor._clean_text(text_with_urls)
        assert "https://example.com" not in cleaned
        
        # Test with markdown link (should be preserved)
        text_with_md_link = "Check out [example](https://example.com) for more info"
        cleaned = self.processor._clean_text(text_with_md_link)
        assert "https://example.com" in cleaned
        
        # Test with empty input
        assert self.processor._clean_text("") == ""
    
    def test_truncate_to_max_tokens(self):
        # Create a long text
        long_text = "This is sentence one. This is sentence two. " * 50
        
        # Truncate to a specific token limit
        max_tokens = 20  # About 80 characters
        truncated = self.processor._truncate_to_max_tokens(long_text, max_tokens)
        
        # Should be approximately the right length
        assert len(truncated) <= max_tokens * 5  # Allow some margin
        
        # Should end with a complete sentence (no truncated sentence)
        assert truncated.endswith("one.") or truncated.endswith("two.")
        
        # Test with very small token limit (smaller than first sentence)
        very_small = self.processor._truncate_to_max_tokens(long_text, 5)
        assert len(very_small) <= 5 * 5  # Should be very short
        assert "..." in very_small  # Should include ellipsis
        
        # Test with text that's already under the limit
        short_text = "Just a short text."
        truncated = self.processor._truncate_to_max_tokens(short_text, 20)
        assert truncated == short_text  # Should be unchanged