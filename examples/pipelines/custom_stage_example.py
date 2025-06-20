#!/usr/bin/env python3
"""
Custom Stage Example

This example demonstrates how to create custom pipeline stages by implementing the
PipelineStage interface. It shows how to handle:
- Stage configuration
- Context data interaction
- Input validation
- Error handling
- Lifecycle hooks
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Pattern, Set, Tuple

from core.pipeline.pipeline import Pipeline
from core.pipeline.stage import PipelineStage
from core.pipeline.context import PipelineContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('custom_stage_example')


class KeywordExtractionStage(PipelineStage):
    """A custom stage that extracts keywords from text content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the keyword extraction stage.
        
        Args:
            config: Configuration for the stage
        """
        super().__init__(config or {})
        self.min_word_length = self.config.get("min_word_length", 4)
        self.max_keywords = self.config.get("max_keywords", 10)
        self.input_field = self.config.get("input_field", "text")
        self.output_field = self.config.get("output_field", "keywords")
        self.stopwords = set(self.config.get("stopwords", [
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "why", "which", "who", "whom", "this", "that",
            "these", "those", "then", "just", "so", "than", "such", "both", "through",
            "about", "for", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing",
            "would", "should", "could", "ought", "i'm", "you're", "he's", "she's",
            "it's", "we're", "they're", "i've", "you've", "we've", "they've",
            "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll",
            "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't",
            "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
            "can't", "couldn't", "shouldn't", "wouldn't", "won't", "not"
        ]))
        
        # Prepare regex patterns for tokenization
        self.word_pattern = re.compile(r'\b[a-z]{%d,}\b' % self.min_word_length)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return the JSON Schema for this stage's configuration.
        
        Returns:
            A dictionary representing the JSON schema
        """
        return {
            "type": "object",
            "properties": {
                "min_word_length": {
                    "type": "integer",
                    "minimum": 2,
                    "description": "Minimum length of words to consider as keywords"
                },
                "max_keywords": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of keywords to extract"
                },
                "input_field": {
                    "type": "string",
                    "description": "Field in the context containing the text to analyze"
                },
                "output_field": {
                    "type": "string",
                    "description": "Field in the context where keywords will be stored"
                },
                "stopwords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of words to exclude from keyword extraction"
                }
            },
            "required": ["input_field", "output_field"]
        }
        
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains the required input.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if input is valid, False otherwise
        """
        if not context.get(self.input_field):
            context.add_error(self.name, f"Missing required input field: {self.input_field}")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Extract keywords from the text in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if keywords were extracted successfully, False otherwise
        """
        # Get the text from the context
        text = context.get(self.input_field, "")
        
        # Process the text to extract keywords
        try:
            keywords = await self._extract_keywords(text)
            context.set(self.output_field, keywords)
            context.set(f"{self.output_field}_count", len(keywords))
            return True
        except Exception as e:
            context.add_error(self.name, f"Keyword extraction failed: {str(e)}")
            return False
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the text.
        
        Args:
            text: The text to extract keywords from
            
        Returns:
            A list of extracted keywords
        """
        # For simulation purposes, add a small delay
        await asyncio.sleep(0.1)
        
        # Convert to lowercase
        text = text.lower()
        
        # Find all words matching the pattern
        words = self.word_pattern.findall(text)
        
        # Remove stopwords
        filtered_words = [word for word in words if word not in self.stopwords]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, count in sorted_words[:self.max_keywords]]


class SentimentAnalysisStage(PipelineStage):
    """A custom stage that performs simple sentiment analysis on text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the sentiment analysis stage.
        
        Args:
            config: Configuration for the stage
        """
        super().__init__(config or {})
        self.input_field = self.config.get("input_field", "text")
        self.output_field = self.config.get("output_field", "sentiment")
        
        # Load sentiment lexicons (simplified for example)
        self.positive_words = set(self.config.get("positive_words", [
            "good", "great", "excellent", "positive", "nice", "wonderful", "amazing",
            "fantastic", "terrific", "happy", "joy", "love", "best", "better",
            "awesome", "superb", "outstanding", "remarkable", "exceptional"
        ]))
        
        self.negative_words = set(self.config.get("negative_words", [
            "bad", "terrible", "horrible", "negative", "awful", "poor", "worst",
            "worse", "disappointing", "sad", "hate", "dislike", "failure", "fail",
            "inferior", "mediocre", "inadequate", "unacceptable", "substandard"
        ]))
        
        # Word tokenization pattern
        self.word_pattern = re.compile(r'\b[a-z]+\b')
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains the required input.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if input is valid, False otherwise
        """
        if not context.get(self.input_field):
            context.add_error(self.name, f"Missing required input field: {self.input_field}")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Analyze sentiment of the text in the context.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if sentiment was analyzed successfully, False otherwise
        """
        # Get the text from the context
        text = context.get(self.input_field, "")
        
        try:
            # Analyze sentiment
            sentiment_score, sentiment_label = await self._analyze_sentiment(text)
            
            # Store results in context
            context.set(self.output_field, {
                "score": sentiment_score,
                "label": sentiment_label,
                "positive_words": list(self._find_sentiment_words(text, self.positive_words)),
                "negative_words": list(self._find_sentiment_words(text, self.negative_words))
            })
            return True
        except Exception as e:
            context.add_error(self.name, f"Sentiment analysis failed: {str(e)}")
            return False
    
    async def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze the sentiment of the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            A tuple of (sentiment_score, sentiment_label)
        """
        # For simulation purposes, add a small delay
        await asyncio.sleep(0.1)
        
        # Convert to lowercase
        text = text.lower()
        
        # Find all words
        words = set(self.word_pattern.findall(text))
        
        # Count positive and negative words
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        # Calculate sentiment score (-1.0 to 1.0)
        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total
        
        # Determine sentiment label
        if sentiment_score > 0.25:
            sentiment_label = "positive"
        elif sentiment_score < -0.25:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
            
        return sentiment_score, sentiment_label
    
    def _find_sentiment_words(self, text: str, sentiment_lexicon: Set[str]) -> Set[str]:
        """Find words in the text that match the sentiment lexicon.
        
        Args:
            text: The text to analyze
            sentiment_lexicon: Set of sentiment words to match
            
        Returns:
            A set of matched sentiment words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Find all words
        words = set(self.word_pattern.findall(text))
        
        # Return intersection with sentiment lexicon
        return words.intersection(sentiment_lexicon)


class TextEnrichmentStage(PipelineStage):
    """A custom stage that combines multiple text analysis results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the text enrichment stage.
        
        Args:
            config: Configuration for the stage
        """
        super().__init__(config or {})
        self.text_field = self.config.get("text_field", "text")
        self.keywords_field = self.config.get("keywords_field", "keywords")
        self.sentiment_field = self.config.get("sentiment_field", "sentiment")
        self.output_field = self.config.get("output_field", "enriched_text")
        self.include_summary = self.config.get("include_summary", True)
        self.max_summary_length = self.config.get("max_summary_length", 100)
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains the required inputs.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if inputs are valid, False otherwise
        """
        missing_fields = []
        
        if not context.get(self.text_field):
            missing_fields.append(self.text_field)
            
        if not context.get(self.keywords_field):
            missing_fields.append(self.keywords_field)
            
        if not context.get(self.sentiment_field):
            missing_fields.append(self.sentiment_field)
            
        if missing_fields:
            context.add_error(
                self.name, 
                f"Missing required input fields: {', '.join(missing_fields)}"
            )
            return False
            
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Enrich the text with analysis results.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if text was enriched successfully, False otherwise
        """
        try:
            # Get inputs from context
            text = context.get(self.text_field, "")
            keywords = context.get(self.keywords_field, [])
            sentiment = context.get(self.sentiment_field, {})
            
            # Create enriched text object
            enriched_text = {
                "original_text": text,
                "analysis": {
                    "keywords": keywords,
                    "sentiment": sentiment,
                    "word_count": len(text.split()),
                    "char_count": len(text)
                }
            }
            
            # Generate summary if configured
            if self.include_summary:
                enriched_text["summary"] = await self._generate_summary(
                    text, keywords, sentiment, self.max_summary_length
                )
            
            # Store in context
            context.set(self.output_field, enriched_text)
            return True
        except Exception as e:
            context.add_error(self.name, f"Text enrichment failed: {str(e)}")
            return False
    
    async def _generate_summary(
        self, 
        text: str, 
        keywords: List[str], 
        sentiment: Dict[str, Any], 
        max_length: int
    ) -> str:
        """Generate a simple summary of the text.
        
        Args:
            text: The original text
            keywords: List of extracted keywords
            sentiment: Sentiment analysis results
            max_length: Maximum summary length
            
        Returns:
            A generated summary
        """
        # For simulation purposes, add a small delay
        await asyncio.sleep(0.1)
        
        # Very simplified summary - in real implementation, use more sophisticated techniques
        if len(text) <= max_length:
            return text
            
        # Just take the first part of the text, trying to break at a sentence boundary
        summary = text[:max_length]
        
        # Try to find the last sentence end
        last_period = summary.rfind('.')
        if last_period > max_length // 2:
            summary = summary[:last_period + 1]
        
        return summary


class JsonFormatterStage(PipelineStage):
    """A custom stage that formats results as JSON."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the JSON formatter stage.
        
        Args:
            config: Configuration for the stage
        """
        super().__init__(config or {})
        self.input_field = self.config.get("input_field", "enriched_text")
        self.output_field = self.config.get("output_field", "json_output")
        self.pretty_print = self.config.get("pretty_print", True)
        self.include_metadata = self.config.get("include_metadata", False)
    
    def validate_input(self, context: PipelineContext) -> bool:
        """Validate that the context contains the required input.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if input is valid, False otherwise
        """
        if not context.get(self.input_field):
            context.add_error(self.name, f"Missing required input field: {self.input_field}")
            return False
        return True
    
    async def process(self, context: PipelineContext) -> bool:
        """Format the data as JSON.
        
        Args:
            context: The pipeline context
            
        Returns:
            True if formatting was successful, False otherwise
        """
        try:
            # Get input data
            data = context.get(self.input_field)
            
            # Add metadata if configured
            output_data = data.copy() if isinstance(data, dict) else data
            
            if self.include_metadata and isinstance(output_data, dict):
                import datetime
                output_data["metadata"] = {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "version": "1.0"
                }
            
            # Convert to JSON
            indent = 2 if self.pretty_print else None
            json_output = json.dumps(output_data, indent=indent)
            
            # Store in context
            context.set(self.output_field, json_output)
            return True
        except Exception as e:
            context.add_error(self.name, f"JSON formatting failed: {str(e)}")
            return False


async def run_custom_pipeline(text: str) -> Dict[str, Any]:
    """Run a pipeline with custom stages.
    
    Args:
        text: The text to analyze
        
    Returns:
        The analysis results
    """
    # Create a pipeline
    pipeline = Pipeline("text_analysis_pipeline")
    
    # Add custom stages
    pipeline.add_stage(KeywordExtractionStage({
        "input_field": "text",
        "output_field": "keywords",
        "min_word_length": 4,
        "max_keywords": 5
    }))
    
    pipeline.add_stage(SentimentAnalysisStage({
        "input_field": "text",
        "output_field": "sentiment"
    }))
    
    pipeline.add_stage(TextEnrichmentStage({
        "text_field": "text",
        "keywords_field": "keywords",
        "sentiment_field": "sentiment",
        "output_field": "enriched_text",
        "include_summary": True
    }))
    
    pipeline.add_stage(JsonFormatterStage({
        "input_field": "enriched_text",
        "output_field": "json_output",
        "pretty_print": True,
        "include_metadata": True
    }))
    
    # Set up initial context
    initial_data = {"text": text}
    
    # Execute pipeline
    context = await pipeline.execute(initial_data)
    
    # Prepare result
    result = {
        "success": not context.has_errors(),
        "original_text": text
    }
    
    # Add output if available
    if context.get("json_output"):
        result["output"] = json.loads(context.get("json_output"))
    
    # Add errors if any
    if context.has_errors():
        result["errors"] = context.metadata["errors"]
    
    # Add metrics
    result["metrics"] = context.get_metrics()
    
    return result


async def main():
    """Main function to run the example."""
    logger.info("=== Running Custom Stage Example ===")
    
    # Sample text to analyze
    text = """
    SmartScrape is an amazing tool for web extraction. It makes gathering data from websites
    incredibly easy and efficient. The pipeline architecture is flexible and powerful,
    allowing users to customize their extraction processes. I'm very happy with how it handles
    complex websites and delivers clean, structured data. Highly recommended for anyone
    needing to extract web content programmatically!
    """
    
    # Run the custom pipeline
    result = await run_custom_pipeline(text)
    
    # Print the result
    print("\nText Analysis Result:")
    print(json.dumps(result, indent=2))
    
    logger.info("=== Example completed ===")


if __name__ == "__main__":
    asyncio.run(main())