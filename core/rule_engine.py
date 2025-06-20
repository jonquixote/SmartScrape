import re
import logging
import json
import time
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Pattern, Set, Callable, TypeVar, Generic, Tuple
from datetime import datetime
import importlib
import inspect

# Type for the result of rule execution
T = TypeVar('T')

class Rule(Generic[T], ABC):
    """Base class for all rules that can be applied to content."""
    
    def __init__(self, name: str, description: str = "", priority: int = 0):
        self.name = name
        self.description = description
        self.priority = priority
        self.logger = logging.getLogger(f"rule_engine.{name}")
        
    @abstractmethod
    def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if this rule matches the content.
        
        Args:
            content: The content to check
            context: Optional additional context
            
        Returns:
            True if the rule matches, False otherwise
        """
        pass
        
    @abstractmethod
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[T]:
        """
        Execute the rule on the content.
        
        Args:
            content: The content to process
            context: Optional additional context
            
        Returns:
            The result of applying the rule, or None if rule couldn't be applied
        """
        pass
        
    @property
    def confidence(self) -> float:
        """
        Get the confidence level of this rule (0.0 to 1.0).
        Override in derived classes to provide more sophisticated confidence estimation.
        """
        return 0.8  # Default confidence
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "type": self.__class__.__name__
        }


class RegexRule(Rule[str]):
    """Rule based on regular expression pattern matching."""
    
    def __init__(self, 
                name: str, 
                pattern: Union[str, Pattern], 
                template: Optional[str] = None,
                group: int = 0,
                flags: int = 0,
                description: str = "",
                priority: int = 0):
        """
        Initialize a regex rule.
        
        Args:
            name: Rule name
            pattern: Regex pattern to match
            template: Optional template for formatting the result with capture groups
            group: Group to extract (0 = entire match)
            flags: Regex flags (re.IGNORECASE, etc.)
            description: Rule description
            priority: Rule priority (higher = tried first)
        """
        super().__init__(name, description, priority)
        self.pattern = re.compile(pattern, flags) if isinstance(pattern, str) else pattern
        self.template = template
        self.group = group
        
    def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the pattern matches the content."""
        if not content:
            return False
        return bool(self.pattern.search(content))
        
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract data using the pattern."""
        if not content:
            return None
            
        match = self.pattern.search(content)
        if not match:
            return None
            
        # If a template is provided, use it for formatting
        if self.template:
            try:
                return self.template.format(*match.groups(), **match.groupdict())
            except Exception as e:
                self.logger.error(f"Error formatting template: {e}")
                return match.group(self.group)
        else:
            # Otherwise return the specified group
            return match.group(self.group)
            
    def find_all(self, content: str) -> List[str]:
        """Find all matches in the content."""
        if not content:
            return []
            
        matches = self.pattern.finditer(content)
        if self.template:
            results = []
            for match in matches:
                try:
                    results.append(self.template.format(*match.groups(), **match.groupdict()))
                except Exception:
                    results.append(match.group(self.group))
            return results
        else:
            return [match.group(self.group) for match in matches]
            
    @property
    def confidence(self) -> float:
        """Estimate confidence based on the specificity of the regex pattern."""
        pattern_str = self.pattern.pattern
        
        # More specific patterns have higher confidence
        if len(pattern_str) > 50:
            return 0.9
        elif len(pattern_str) > 20:
            return 0.8
        else:
            return 0.7


class MultiPatternRule(Rule[List[str]]):
    """Rule for extracting multiple patterns from content."""
    
    def __init__(self, 
                name: str, 
                patterns: List[Union[str, Pattern]],
                combine_with_and: bool = True,
                description: str = "",
                priority: int = 0):
        """
        Initialize a multi-pattern rule.
        
        Args:
            name: Rule name
            patterns: List of regex patterns to match
            combine_with_and: If True, all patterns must match; if False, any pattern can match
            description: Rule description
            priority: Rule priority
        """
        super().__init__(name, description, priority)
        self.patterns = [re.compile(p) if isinstance(p, str) else p for p in patterns]
        self.combine_with_and = combine_with_and
        
    def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the patterns match the content according to combination rule."""
        if not content:
            return False
            
        if self.combine_with_and:
            return all(pattern.search(content) for pattern in self.patterns)
        else:
            return any(pattern.search(content) for pattern in self.patterns)
        
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
        """Extract all matching patterns."""
        if not content:
            return None
            
        results = []
        for pattern in self.patterns:
            match = pattern.search(content)
            if match:
                results.append(match.group(0))
                
        return results if results else None


class FunctionRule(Rule[Any]):
    """Rule that uses a custom function for matching and extraction."""
    
    def __init__(self, 
                name: str, 
                match_fn: Callable[[str, Optional[Dict[str, Any]]], bool],
                extract_fn: Callable[[str, Optional[Dict[str, Any]]], Any],
                description: str = "",
                priority: int = 0,
                confidence_value: float = 0.8):
        """
        Initialize a function rule.
        
        Args:
            name: Rule name
            match_fn: Function to determine if the rule matches
            extract_fn: Function to extract data
            description: Rule description
            priority: Rule priority
            confidence_value: Confidence value for this rule
        """
        super().__init__(name, description, priority)
        self.match_fn = match_fn
        self.extract_fn = extract_fn
        self._confidence = confidence_value
        
    def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the content matches using the provided function."""
        if not content:
            return False
        return self.match_fn(content, context)
        
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Extract data using the provided function."""
        if not content:
            return None
        return self.extract_fn(content, context)
        
    @property
    def confidence(self) -> float:
        """Return the configured confidence value for this rule."""
        return self._confidence


class RuleSet(Generic[T]):
    """A collection of related rules that can be applied to content."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.rules: List[Rule[T]] = []
        self.logger = logging.getLogger(f"rule_engine.ruleset.{name}")
        
    def add_rule(self, rule: Rule[T]) -> None:
        """Add a rule to the rule set."""
        self.rules.append(rule)
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: -r.priority)
        
    def add_rules(self, rules: List[Rule[T]]) -> None:
        """Add multiple rules to the rule set."""
        self.rules.extend(rules)
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: -r.priority)
        
    def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if any rule in the set matches the content."""
        return any(rule.matches(content, context) for rule in self.rules)
        
    def execute_first_match(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[T]:
        """Execute the first matching rule in the set."""
        for rule in self.rules:
            if rule.matches(content, context):
                try:
                    result = rule.execute(content, context)
                    if result is not None:
                        return result
                except Exception as e:
                    self.logger.error(f"Error executing rule {rule.name}: {e}")
        return None
        
    def execute_all_matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[T]:
        """Execute all matching rules in the set and return their results."""
        results = []
        for rule in self.rules:
            if rule.matches(content, context):
                try:
                    result = rule.execute(content, context)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error executing rule {rule.name}: {e}")
        return results
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the rule set to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "rules": [rule.to_dict() for rule in self.rules]
        }


class RuleExecutor:
    """Applies rules to content and returns the results."""
    
    def __init__(self, default_context: Optional[Dict[str, Any]] = None):
        self.default_context = default_context or {}
        self.logger = logging.getLogger("rule_engine.executor")
        
    def execute_rule(self, rule: Rule[T], content: str, 
                    context: Optional[Dict[str, Any]] = None) -> Optional[T]:
        """Execute a single rule on the content."""
        merged_context = {**self.default_context, **(context or {})}
        
        try:
            if rule.matches(content, merged_context):
                return rule.execute(content, merged_context)
        except Exception as e:
            self.logger.error(f"Error executing rule {rule.name}: {e}")
            
        return None
        
    def execute_ruleset(self, ruleset: RuleSet[T], content: str,
                       context: Optional[Dict[str, Any]] = None,
                       execute_all: bool = False) -> Union[Optional[T], List[T]]:
        """Execute a rule set on the content."""
        merged_context = {**self.default_context, **(context or {})}
        
        if execute_all:
            return ruleset.execute_all_matches(content, merged_context)
        else:
            return ruleset.execute_first_match(content, merged_context)
        
    def execute_multiple_rulesets(self, rulesets: List[RuleSet[T]], content: str,
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute multiple rule sets and combine their results."""
        results = {}
        merged_context = {**self.default_context, **(context or {})}
        
        for ruleset in rulesets:
            try:
                result = ruleset.execute_first_match(content, merged_context)
                if result is not None:
                    results[ruleset.name] = result
            except Exception as e:
                self.logger.error(f"Error executing ruleset {ruleset.name}: {e}")
                
        return results


class RuleRegistry:
    """Registry for managing and retrieving rules and rulesets."""
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.rulesets: Dict[str, RuleSet] = {}
        self.logger = logging.getLogger("rule_engine.registry")
        
    def register_rule(self, rule: Rule) -> None:
        """Register a rule in the registry."""
        self.rules[rule.name] = rule
        self.logger.debug(f"Registered rule: {rule.name}")
        
    def register_ruleset(self, ruleset: RuleSet) -> None:
        """Register a ruleset in the registry."""
        self.rulesets[ruleset.name] = ruleset
        self.logger.debug(f"Registered ruleset: {ruleset.name}")
        
    def get_rule(self, name: str) -> Optional[Rule]:
        """Get a rule by name."""
        return self.rules.get(name)
        
    def get_ruleset(self, name: str) -> Optional[RuleSet]:
        """Get a ruleset by name."""
        return self.rulesets.get(name)
        
    def get_rulesets_for_task(self, task_type: str) -> List[RuleSet]:
        """Get all rulesets relevant for a specific task type."""
        relevant_rulesets = []
        for ruleset in self.rulesets.values():
            # Look for rulesets with naming convention or metadata indicating task relevance
            if (task_type.lower() in ruleset.name.lower() or 
                task_type.lower() in ruleset.description.lower()):
                relevant_rulesets.append(ruleset)
        return relevant_rulesets
        
    def load_rules_from_directory(self, directory_path: str) -> None:
        """Load rules from Python modules in a directory."""
        try:
            # Add directory to path if not already there
            if directory_path not in sys.path:
                sys.path.append(directory_path)
                
            # Find all Python files in the directory
            for filename in os.listdir(directory_path):
                if filename.endswith('.py'):
                    module_name = filename[:-3]  # Remove .py extension
                    try:
                        # Import the module
                        spec = importlib.util.spec_from_file_location(
                            module_name, 
                            os.path.join(directory_path, filename)
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find all Rule subclasses in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, Rule) and 
                                obj != Rule and 
                                obj != RegexRule):
                                # Instantiate and register the rule
                                try:
                                    rule = obj()
                                    self.register_rule(rule)
                                except Exception as e:
                                    self.logger.error(f"Error instantiating rule {name}: {e}")
                    except Exception as e:
                        self.logger.error(f"Error loading module {module_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error loading rules from directory: {e}")


# Specialized extractors (rule implementations)

class ProductPriceExtractor(RegexRule):
    """Extracts product prices with currency handling."""
    
    def __init__(self, 
                name: str = "product_price", 
                priority: int = 10,
                currencies: Optional[List[str]] = None):
        """
        Initialize a product price extractor.
        
        Args:
            name: Rule name
            priority: Rule priority
            currencies: List of currency symbols to detect (defaults to common ones)
        """
        currencies = currencies or ['$', '€', '£', '¥']
        currency_pattern = '|'.join(re.escape(c) for c in currencies)
        
        # Pattern matches currency symbols followed by numbers, or numbers followed by currency codes
        pattern = fr'(?:({currency_pattern})\s*(\d+(?:,\d+)*(?:\.\d+)?))|(?:(\d+(?:,\d+)*(?:\.\d+)?)\s*({currency_pattern}|USD|EUR|GBP|JPY))'
        
        description = "Extracts product prices with various currency formats"
        super().__init__(name, pattern, description=description, priority=priority)
        
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract price from content."""
        if not content:
            return None
            
        match = self.pattern.search(content)
        if not match:
            return None
            
        # Process the match to handle different formats
        if match.group(1) and match.group(2):  # Currency symbol before number
            currency = match.group(1)
            price = match.group(2)
            return f"{currency}{price}"
        elif match.group(3) and match.group(4):  # Number before currency
            price = match.group(3)
            currency = match.group(4)
            return f"{currency}{price}"
            
        # Fallback to the standard behavior
        return match.group(0)


class DateExtractor(RegexRule):
    """Extracts dates in various formats and normalizes them."""
    
    def __init__(self, name: str = "date_extractor", priority: int = 10, output_format: str = "%Y-%m-%d"):
        """
        Initialize a date extractor.
        
        Args:
            name: Rule name
            priority: Rule priority
            output_format: The format to use for output dates
        """
        # Pattern covers common date formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, Month DD, YYYY, etc.
        pattern = r'(?:\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b)|(?:\b\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4}\b)|(?:\b\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2}\b)'
        
        description = "Extracts and normalizes dates in various formats"
        super().__init__(name, pattern, description=description, priority=priority)
        self.output_format = output_format
        
        # Format patterns for parsing
        self.format_patterns = [
            # Month name formats
            ("%B %d, %Y", r'January|February|March|April|May|June|July|August|September|October|November|December'),
            ("%B %d %Y", r'January|February|March|April|May|June|July|August|September|October|November|December'),
            ("%b %d, %Y", r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'),
            ("%b %d %Y", r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'),
            
            # Numeric formats
            ("%m/%d/%Y", r'\d{1,2}/\d{1,2}/\d{4}'),
            ("%d/%m/%Y", r'\d{1,2}/\d{1,2}/\d{4}'),
            ("%Y-%m-%d", r'\d{4}-\d{1,2}-\d{1,2}'),
            ("%m-%d-%Y", r'\d{1,2}-\d{1,2}-\d{4}'),
            ("%d-%m-%Y", r'\d{1,2}-\d{1,2}-\d{4}'),
            ("%m.%d.%Y", r'\d{1,2}\.\d{1,2}\.\d{4}'),
            ("%d.%m.%Y", r'\d{1,2}\.\d{1,2}\.\d{4}'),
            
            # Two-digit year formats
            ("%m/%d/%y", r'\d{1,2}/\d{1,2}/\d{2}'),
            ("%d/%m/%y", r'\d{1,2}/\d{1,2}/\d{2}'),
            ("%y-%m-%d", r'\d{2}-\d{1,2}-\d{1,2}'),
            ("%m-%d-%y", r'\d{1,2}-\d{1,2}-\d{2}'),
            ("%d-%m-%y", r'\d{1,2}-\d{1,2}-\d{2}')
        ]
        
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract and normalize date from content."""
        if not content:
            return None
            
        match = self.pattern.search(content)
        if not match:
            return None
            
        date_str = match.group(0)
        
        # Try to parse the date with each format
        parsed_date = None
        for fmt, pattern in self.format_patterns:
            if re.search(pattern, date_str):
                try:
                    # Special handling for day suffixes like "1st", "2nd", etc.
                    clean_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                    parsed_date = datetime.strptime(clean_date, fmt)
                    break
                except ValueError:
                    continue
        
        if parsed_date:
            return parsed_date.strftime(self.output_format)
        
        # If parsing failed, return the original string
        return date_str


class ContactInfoExtractor(RuleSet[str]):
    """Extracts contact information like emails, phone numbers, etc."""
    
    def __init__(self, name: str = "contact_info", include_social: bool = True):
        """
        Initialize a contact info extractor set.
        
        Args:
            name: Rule set name
            include_social: Whether to include social media handles
        """
        super().__init__(name, "Extracts various types of contact information")
        
        # Add email rule
        email_rule = RegexRule(
            name="email",
            pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            description="Extracts email addresses",
            priority=20
        )
        self.add_rule(email_rule)
        
        # Add phone number rule (handles various formats)
        phone_rule = RegexRule(
            name="phone",
            pattern=r'(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            description="Extracts phone numbers in various formats",
            priority=15
        )
        self.add_rule(phone_rule)
        
        # Add address rule (simple pattern, can be improved)
        address_rule = RegexRule(
            name="address",
            pattern=r'\d+\s+[A-Za-z0-9\s,]+(?:Avenue|Lane|Road|Boulevard|Drive|Street|Ave|Dr|Rd|Blvd|Ln|St)\.?(?:\s+[A-Za-z]+)?(?:\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?)?',
            description="Extracts physical addresses",
            priority=10
        )
        self.add_rule(address_rule)
        
        if include_social:
            # Add social media handle rules
            twitter_rule = RegexRule(
                name="twitter",
                pattern=r'(?:@)([A-Za-z0-9_]+)',
                description="Extracts Twitter handles",
                priority=5
            )
            self.add_rule(twitter_rule)
            
            # URL-based social patterns
            social_rule = RegexRule(
                name="social_url",
                pattern=r'https?://(?:www\.)?(twitter\.com|facebook\.com|linkedin\.com|instagram\.com)/[A-Za-z0-9_\.-]+',
                description="Extracts social media profile URLs",
                priority=5
            )
            self.add_rule(social_rule)


class MetadataExtractor(RuleSet[Dict[str, str]]):
    """Extracts metadata from HTML content."""
    
    def __init__(self, name: str = "metadata_extractor"):
        """Initialize a metadata extractor set."""
        super().__init__(name, "Extracts metadata from HTML content")
        
        # Add meta tag rule
        class MetaTagRule(Rule[Dict[str, str]]):
            def __init__(self):
                super().__init__("meta_tags", "Extracts meta tags from HTML", priority=10)
                
            def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
                return '<meta' in content
                
            def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
                result = {}
                
                # Extract standard meta tags
                meta_pattern = re.compile(r'<meta(?:\s+[^>]*?name=[\'"]([^\'"]*)[\'"][^>]*?content=[\'"]([^\'"]*)[\'"]|\s+[^>]*?content=[\'"]([^\'"]*)[\'"][^>]*?name=[\'"]([^\'"]*)[\'"])[^>]*?>', re.IGNORECASE)
                for match in meta_pattern.finditer(content):
                    name = match.group(1) or match.group(4)
                    content_val = match.group(2) or match.group(3)
                    if name:
                        result[name.lower()] = content_val
                
                # Extract Open Graph meta tags
                og_pattern = re.compile(r'<meta(?:\s+[^>]*?property=[\'"]og:([^\'"]*)[\'"][^>]*?content=[\'"]([^\'"]*)[\'"]|\s+[^>]*?content=[\'"]([^\'"]*)[\'"][^>]*?property=[\'"]og:([^\'"]*)[\'"])[^>]*?>', re.IGNORECASE)
                for match in og_pattern.finditer(content):
                    name = match.group(1) or match.group(4)
                    content_val = match.group(2) or match.group(3)
                    if name:
                        result[f'og:{name}'] = content_val
                
                return result if result else None
        
        # Add title rule
        class TitleRule(Rule[Dict[str, str]]):
            def __init__(self):
                super().__init__("title", "Extracts page title", priority=5)
                self.pattern = re.compile(r'<title[^>]*>([^<]+)</title>', re.IGNORECASE)
                
            def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
                return '<title' in content and '</title>' in content
                
            def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
                match = self.pattern.search(content)
                if match:
                    return {"title": match.group(1).strip()}
                return None
        
        # Add the rules to the ruleset
        self.add_rule(MetaTagRule())
        self.add_rule(TitleRule())


class ListingExtractor(Rule[List[Dict[str, str]]]):
    """Extracts listings (products, articles, etc.) from structured content."""
    
    def __init__(self, name: str = "listing_extractor", 
                item_selector: str = None,
                field_selectors: Dict[str, str] = None,
                description: str = "Extracts listings from structured content",
                priority: int = 10):
        """
        Initialize a listing extractor.
        
        Args:
            name: Rule name
            item_selector: CSS-like selector pattern for items (will be converted to regex)
            field_selectors: Dict mapping field names to CSS-like selector patterns
            description: Rule description
            priority: Rule priority
        """
        super().__init__(name, description, priority)
        
        self.item_pattern = self._selector_to_regex(item_selector) if item_selector else None
        self.field_patterns = {}
        
        if field_selectors:
            for field, selector in field_selectors.items():
                self.field_patterns[field] = self._selector_to_regex(selector)
    
    def _selector_to_regex(self, selector: str) -> Pattern:
        """Convert a simplified CSS-like selector to regex pattern."""
        # This is a basic implementation - a real one would be more sophisticated
        if selector.startswith('.'):
            # Class selector
            class_name = selector[1:]
            return re.compile(fr'class=[\'"][^\'"]*{re.escape(class_name)}[^\'"]*[\'"]', re.IGNORECASE)
        elif selector.startswith('#'):
            # ID selector
            id_name = selector[1:]
            return re.compile(fr'id=[\'"]?{re.escape(id_name)}[\'"]?', re.IGNORECASE)
        elif '=' in selector:
            # Attribute selector
            attr, value = selector.split('=', 1)
            attr = attr.strip('[]')
            value = value.strip('\'"')
            return re.compile(fr'{re.escape(attr)}=[\'"]?{re.escape(value)}[\'"]?', re.IGNORECASE)
        else:
            # Tag selector
            return re.compile(fr'<{re.escape(selector)}[^>]*>.*?</{re.escape(selector)}>', re.DOTALL | re.IGNORECASE)
    
    def matches(self, content: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if the content likely contains listings."""
        if not content:
            return False
            
        # Check for specific HTML structures that indicate lists
        if '<ul' in content and '<li' in content:
            return True
        if '<table' in content and '<tr' in content:
            return True
        if '<div' in content and 'class=' in content and ('list' in content.lower() or 'grid' in content.lower()):
            return True
            
        # If we have a custom item pattern, use it
        if self.item_pattern and self.item_pattern.search(content):
            return True
            
        return False
    
    def execute(self, content: str, context: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, str]]]:
        """Extract listings from the content."""
        if not content or not self.matches(content, context):
            return None
            
        results = []
        context = context or {}
        
        try:
            # Use libraries like BeautifulSoup if available in context
            if 'soup' in context:
                # This would be handled by actual HTML parsing
                items = context['soup'].select(context.get('item_selector', 'li'))
                for item in items:
                    result = {}
                    for field, selector in context.get('field_selectors', {}).items():
                        field_elem = item.select_one(selector)
                        if field_elem:
                            result[field] = field_elem.text.strip()
                    if result:
                        results.append(result)
                return results if results else None
            
            # Fallback to regex-based extraction
            # For simplicity, we'll focus on common patterns like list items or table rows
            if '<ul' in content and '<li' in content:
                pattern = re.compile(r'<li[^>]*>(.*?)</li>', re.DOTALL | re.IGNORECASE)
                for match in pattern.finditer(content):
                    item_html = match.group(1)
                    item_data = self._extract_item_fields(item_html)
                    if item_data:
                        results.append(item_data)
            elif '<table' in content and '<tr' in content:
                # Extract table headers first
                header_pattern = re.compile(r'<thead[^>]*>.*?<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
                header_match = header_pattern.search(content)
                headers = []
                if header_match:
                    th_pattern = re.compile(r'<th[^>]*>(.*?)</th>', re.DOTALL | re.IGNORECASE)
                    headers = [m.group(1).strip() for m in th_pattern.finditer(header_match.group(1))]
                
                # Extract rows
                row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
                for match in row_pattern.finditer(content):
                    row_html = match.group(1)
                    if '<th' in row_html:
                        continue  # Skip header rows
                    
                    cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)
                    cells = [m.group(1).strip() for m in cell_pattern.finditer(row_html)]
                    
                    if headers and len(cells) == len(headers):
                        results.append(dict(zip(headers, cells)))
                    elif cells:
                        # If no headers, just use index as keys
                        results.append({f'col{i}': cell for i, cell in enumerate(cells)})
            
            # If we have custom patterns, use them
            elif self.item_pattern:
                # Custom extraction based on configured patterns
                pass
                
            return results if results else None
            
        except Exception as e:
            self.logger.error(f"Error extracting listings: {e}")
            return None
    
    def _extract_item_fields(self, item_html: str) -> Dict[str, str]:
        """Extract fields from an item's HTML."""
        item_data = {}
        
        # Extract text content
        clean_html = re.sub(r'<[^>]*>', ' ', item_html)
        clean_html = re.sub(r'\s+', ' ', clean_html).strip()
        if clean_html:
            item_data['text'] = clean_html
        
        # Extract links
        link_pattern = re.compile(r'<a[^>]*href=[\'"]([^\'"]*)[\'"][^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
        links = [(m.group(1), m.group(2).strip()) for m in link_pattern.finditer(item_html)]
        if links:
            item_data['links'] = links
            
            # Also store the first link separately as it's often the main one
            item_data['url'] = links[0][0]
            item_data['title'] = links[0][1]
        
        # Extract images
        img_pattern = re.compile(r'<img[^>]*src=[\'"]([^\'"]*)[\'"][^>]*>', re.IGNORECASE)
        images = [m.group(1) for m in img_pattern.finditer(item_html)]
        if images:
            item_data['images'] = images
            item_data['image'] = images[0]  # The first image
            
        # Custom field extraction based on configured patterns
        for field, pattern in self.field_patterns.items():
            match = pattern.search(item_html)
            if match:
                # Try to extract text from the match
                field_html = match.group(0)
                clean_field = re.sub(r'<[^>]*>', ' ', field_html)
                clean_field = re.sub(r'\s+', ' ', clean_field).strip()
                item_data[field] = clean_field
                
        return item_data


# Decision framework functions

def is_suitable_for_rules(content: str, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if a task can be handled by rules rather than AI.
    
    Args:
        content: The content to process
        task: The task description
        context: Optional additional context
        
    Returns:
        True if the task is suitable for rule-based processing
    """
    context = context or {}
    
    # Task suitability checks
    task_lower = task.lower()
    
    # Specific task types that are suitable for rules
    suitable_tasks = [
        "extract", "find", "get", "identify", "parse", "scrape"
    ]
    
    # Common entities that are suitable for rule-based extraction
    suitable_entities = [
        "price", "prices", "date", "dates", "email", "emails", 
        "phone", "phone number", "address", "link", "url", "product",
        "metadata", "contact", "listing"
    ]
    
    # Check if task matches suitable patterns
    is_suitable_task = any(term in task_lower for term in suitable_tasks)
    is_suitable_entity = any(entity in task_lower for entity in suitable_entities)
    
    # Check content characteristics
    has_structured_content = False
    
    # HTML content typically has good structure for rules
    if '<html' in content or '<body' in content or '<div' in content:
        has_structured_content = True
    
    # Tabular data is good for rules
    if '<table' in content and '<tr' in content:
        has_structured_content = True
        
    # Lists are good for rules
    if '<ul' in content and '<li' in content:
        has_structured_content = True
        
    # Clearly formatted text (product listings, contact pages, etc.)
    if (re.search(r'\$\s*\d+(?:\.\d+)?', content) or  # Prices
        re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content) or  # Emails
        re.search(r'(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', content)):  # Phone numbers
        has_structured_content = True
        
    # Consider context hints
    if context.get('prefer_rules', False):
        return True
        
    if context.get('content_type') == 'html':
        has_structured_content = True
        
    # Make the final decision
    return (is_suitable_task and is_suitable_entity) or (is_suitable_task and has_structured_content)


def estimate_rule_confidence(content: str, ruleset: RuleSet, context: Optional[Dict[str, Any]] = None) -> float:
    """
    Estimate the confidence level for rule-based extraction on this content.
    
    Args:
        content: The content to process
        ruleset: The ruleset to evaluate
        context: Optional additional context
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    context = context or {}
    
    # Base confidence starts with the ruleset's matching ability
    if not ruleset.matches(content, context):
        return 0.0
        
    # Start with a moderate confidence
    confidence = 0.5
    
    # Factor 1: Number of matching rules
    matching_rules = [rule for rule in ruleset.rules if rule.matches(content, context)]
    rule_ratio = len(matching_rules) / len(ruleset.rules) if ruleset.rules else 0
    confidence += rule_ratio * 0.2  # Up to 0.2 boost for high rule match ratio
    
    # Factor 2: Average confidence of matching rules
    if matching_rules:
        avg_rule_confidence = sum(rule.confidence for rule in matching_rules) / len(matching_rules)
        confidence += avg_rule_confidence * 0.2  # Up to 0.2 boost for high rule confidence
    
    # Factor 3: Content structure
    structure_score = 0.0
    
    # Check for well-structured content
    if '<table' in content and '<tr' in content:
        structure_score += 0.1  # Tables are usually well-structured
    if '<ul' in content and '<li' in content:
        structure_score += 0.05  # Lists are somewhat structured
    if '<div' in content and 'class=' in content:
        structure_score += 0.05  # Classed divs indicate some structure
        
    # Check for clean text
    if context.get('content_type') == 'clean_text':
        structure_score += 0.1
        
    confidence += structure_score
    
    # Factor 4: Context clues
    if context.get('prefer_rules', False):
        confidence += 0.1
    if context.get('prior_success_rate', 0) > 0.8:  # If rules have worked well before
        confidence += 0.1
        
    # Enforce bounds
    return max(0.0, min(1.0, confidence))


# Factory for creating common rule sets

def create_common_rulesets() -> List[RuleSet]:
    """Create a collection of common rule sets for standard extraction tasks."""
    rulesets = []
    
    # Prices ruleset
    price_set = RuleSet("prices", "Extract product prices")
    price_set.add_rule(ProductPriceExtractor())
    rulesets.append(price_set)
    
    # Dates ruleset
    date_set = RuleSet("dates", "Extract dates")
    date_set.add_rule(DateExtractor())
    rulesets.append(date_set)
    
    # Contact info ruleset
    contact_set = ContactInfoExtractor()
    rulesets.append(contact_set)
    
    # Metadata ruleset
    metadata_set = MetadataExtractor()
    rulesets.append(metadata_set)
    
    # Create a product listing ruleset
    product_set = RuleSet("product_listings", "Extract product listings")
    product_set.add_rule(ListingExtractor(
        name="product_items",
        item_selector=".product",
        field_selectors={
            "title": ".product-title",
            "price": ".product-price",
            "description": ".product-description"
        }
    ))
    rulesets.append(product_set)
    
    return rulesets


class RuleEngine:
    """Engine that applies rules to content based on context."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the rule engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = RuleRegistry()
        self.executor = RuleExecutor()
        self.logger = logging.getLogger("rule_engine")
        
    async def process(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process content using rules based on the context.
        
        Args:
            content: The content to process
            context: Optional context that might influence rule selection
            
        Returns:
            Dictionary with processing results
        """
        context = context or {}
        start_time = time.time()
        
        try:
            # Determine the task type
            task_type = context.get("task_type", "unknown")
            
            # Check if this task is suitable for rule-based processing
            if not is_suitable_for_rules(content, task_type, context):
                return {
                    "success": False,
                    "reason": "Task not suitable for rule-based processing",
                    "content": ""
                }
            
            # Get relevant rule sets for this task
            rulesets = self.registry.get_rulesets_for_task(task_type)
            
            if not rulesets:
                return {
                    "success": False,
                    "reason": f"No rulesets found for task type: {task_type}",
                    "content": ""
                }
            
            # Execute rulesets
            results = self.executor.execute_multiple_rulesets(rulesets, content, context)
            
            if not results:
                return {
                    "success": False,
                    "reason": "No rule matches found",
                    "content": ""
                }
            
            # Format results
            output = {
                "success": True,
                "rule_id": ",".join(results.keys()),
                "content": str(results),
                "processing_time": time.time() - start_time
            }
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error in rule processing: {str(e)}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}",
                "content": "",
                "processing_time": time.time() - start_time
            }


# SimpleRuleEngine for basic rule operations without the complexity of the main engine
class SimpleRuleEngine:
    """Provides rule-based alternatives for common extraction tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the simple rule engine."""
        self.config = config or {}
        self.logger = logging.getLogger("simple_rule_engine")
        
    def extract_emails(self, content: str) -> List[str]:
        """Extract email addresses from text."""
        if not content:
            return []
            
        # Email regex pattern
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        emails = re.findall(pattern, content)
        
        # Remove duplicates
        return list(set(emails))
        
    def extract_urls(self, content: str) -> List[str]:
        """Extract URLs from text."""
        if not content:
            return []
            
        # URL regex pattern
        pattern = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        urls = re.findall(pattern, content)
        
        # Remove duplicates
        return list(set(urls))
        
    def extract_phones(self, content: str) -> List[str]:
        """Extract phone numbers from text."""
        if not content:
            return []
            
        # Various phone patterns
        patterns = [
            r"\+\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US/Canada
            r"\d{5}[-.\s]?\d{6}",  # Some European countries
        ]
        
        phones = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            phones.extend(matches)
            
        # Remove duplicates
        return list(set(phones))
        
    def detect_content_type(self, content: str) -> str:
        """Detect the type of content."""
        if not content:
            return "unknown"
            
        # Check if it's likely HTML
        if re.search(r"<html|<body|<div|<p>", content):
            return "html"
            
        # Check if it's likely JSON
        if content.strip().startswith("{") and content.strip().endswith("}"):
            try:
                json.loads(content)
                return "json"
            except:
                pass
                
        # Check if it's likely CSV
        if "," in content and "\n" in content:
            lines = content.split("\n")
            if len(lines) > 1:
                first_line_commas = lines[0].count(",")
                if first_line_commas > 0:
                    return "csv"
                    
        # Check if it's likely XML
        if re.search(r"<\?xml|<[a-zA-Z]+>.*</[a-zA-Z]+>", content, re.DOTALL):
            return "xml"
            
        # Default to plain text
        return "text"
        
    def process(self, content: str, task_type: str) -> Dict[str, Any]:
        """
        Process content based on the task type.
        
        Args:
            content: The content to process
            task_type: The type of task to perform
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "success": False,
            "content": "",
            "reason": "Unknown task type"
        }
        
        try:
            if task_type == "extract_emails":
                emails = self.extract_emails(content)
                result = {
                    "success": True,
                    "content": emails,
                    "count": len(emails)
                }
            elif task_type == "extract_urls":
                urls = self.extract_urls(content)
                result = {
                    "success": True,
                    "content": urls,
                    "count": len(urls)
                }
            elif task_type == "extract_phones":
                phones = self.extract_phones(content)
                result = {
                    "success": True,
                    "content": phones,
                    "count": len(phones)
                }
            elif task_type == "detect_content_type":
                content_type = self.detect_content_type(content)
                result = {
                    "success": True,
                    "content": content_type
                }
            else:
                result["reason"] = f"Task type '{task_type}' not supported"
                
        except Exception as e:
            result["reason"] = f"Error: {str(e)}"
            self.logger.error(f"Error processing {task_type}: {e}")
            
        return result