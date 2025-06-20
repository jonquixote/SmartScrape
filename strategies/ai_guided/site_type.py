"""
Site Type Settings Module

This module provides configuration and settings for different website types.
Different site types require different crawling strategies, depths, and extraction methods.
"""
from typing import Dict, Any, Optional, List, Tuple
import logging
import re
from urllib.parse import urlparse
import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Default settings for different site types
SITE_TYPE_SETTINGS = {
    # Documentation sites: Deep crawl with high depth for comprehensive coverage
    "docs": {
        "max_depth": 4,
        "max_pages": 50,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": False,
        "exploration_ratio": 0.8,  # Higher ratio for comprehensive documentation coverage
        "content_extraction_mode": "structured",
    },
    
    # E-commerce sites: Medium depth, focus on products
    "ecommerce": {
        "max_depth": 3,
        "max_pages": 30,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": True,
        "exploration_ratio": 0.6,  # Balanced for products and categories
        "content_extraction_mode": "product",
    },
    
    # Blog sites: Medium depth, focus on content
    "blog": {
        "max_depth": 2,
        "max_pages": 20,
        "use_search": False,
        "use_sitemap": True,
        "follow_pagination": True,
        "exploration_ratio": 0.7,  # Focus on content discoverability
        "content_extraction_mode": "article",
    },
    
    # Forum/Community sites: Medium depth, focus on threads
    "forum": {
        "max_depth": 2,
        "max_pages": 15,
        "use_search": True,
        "use_sitemap": False,
        "follow_pagination": True,
        "exploration_ratio": 0.5,  # Only moderate exploration
        "content_extraction_mode": "thread",
    },
    
    # Real estate sites: Focused crawl for listings
    "real_estate": {
        "max_depth": 3,
        "max_pages": 25,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": True,
        "exploration_ratio": 0.7,  # Focus on property listings
        "content_extraction_mode": "listing",
        "priority_paths": ["listings", "homes", "properties", "search", "for-sale", "rent"],
        "extraction_fields": ["price", "address", "bedrooms", "bathrooms", "sqft", "description", "features"],
    },
    
    # News sites: Shallow crawl with high page count for latest articles
    "news": {
        "max_depth": 2,
        "max_pages": 40,
        "use_search": True,
        "use_sitemap": True, 
        "follow_pagination": True,
        "exploration_ratio": 0.6,  # Balance between categories and articles
        "content_extraction_mode": "article",
    },
    
    # Job sites: Medium depth, focus on job listings
    "jobs": {
        "max_depth": 3,
        "max_pages": 25,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": True,
        "exploration_ratio": 0.7,  # Focus on job listings
        "content_extraction_mode": "listing",
        "priority_paths": ["jobs", "careers", "positions", "employment", "job-search"],
        "extraction_fields": ["title", "company", "location", "salary", "description", "requirements"],
    },
    
    # Academic/Research sites: Deep crawl for comprehensive coverage
    "academic": {
        "max_depth": 4,
        "max_pages": 40,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": True,
        "exploration_ratio": 0.8,  # Higher ratio for comprehensive research coverage
        "content_extraction_mode": "article",
        "priority_paths": ["papers", "research", "publications", "journals", "articles"],
        "extraction_fields": ["title", "authors", "abstract", "publication_date", "keywords", "content"],
    },
    
    # Travel sites: Medium depth, focus on destinations and accommodations
    "travel": {
        "max_depth": 3,
        "max_pages": 30,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": True,
        "exploration_ratio": 0.6,  # Balanced for destinations and bookings
        "content_extraction_mode": "listing",
        "priority_paths": ["destinations", "hotels", "flights", "packages", "vacation"],
        "extraction_fields": ["name", "location", "price", "rating", "amenities", "description", "availability"],
    },
    
    # Government sites: Deep crawl with focus on official information
    "government": {
        "max_depth": 4,
        "max_pages": 50,
        "use_search": True,
        "use_sitemap": True,
        "follow_pagination": False,
        "exploration_ratio": 0.7,  # Focus on official information
        "content_extraction_mode": "structured",
        "priority_paths": ["services", "departments", "documents", "forms", "regulations"],
        "extraction_fields": ["title", "agency", "description", "publication_date", "document_type", "content"],
    },
    
    # Unknown (default): Conservative settings
    "unknown": {
        "max_depth": 2,
        "max_pages": 10,
        "use_search": False, 
        "use_sitemap": True,
        "follow_pagination": False,
        "exploration_ratio": 0.5,  # Neutral exploration
        "content_extraction_mode": "generic",
    }
}

# TLD-specific indications of site types
TLD_SITE_TYPE_HINTS = {
    "gov": "government",
    "edu": "academic",
    "ac.uk": "academic",
    "edu.au": "academic",
    "org": None,  # Needs more context
    "io": "docs",  # Often tech/documentation
    "dev": "docs"  # Often tech/documentation
}

def get_site_settings(site_type: str) -> Dict[str, Any]:
    """
    Get the settings for a specific site type.
    
    Args:
        site_type: The type of website (docs, ecommerce, blog, etc.)
        
    Returns:
        Dictionary with site-specific settings
    """
    # Return the settings for the specified site type, or default to "unknown"
    if site_type.lower() in SITE_TYPE_SETTINGS:
        logger.info(f"Using settings for site type: {site_type}")
        return SITE_TYPE_SETTINGS[site_type.lower()]
    else:
        logger.info(f"Site type '{site_type}' not found, using default settings")
        return SITE_TYPE_SETTINGS["unknown"]

def detect_site_type(url: str, content: Optional[str] = None) -> str:
    """
    Detect the type of website based on URL and content.
    
    Args:
        url: The URL of the website
        content: Optional HTML content of the website
        
    Returns:
        Detected site type as a string
    """
    # Parse the URL to get the domain and path
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    
    # First check for TLD-specific site types
    tld_match = _check_tld_for_site_type(domain)
    if tld_match:
        return tld_match
    
    # Check for documentation sites
    if any(x in domain for x in ['docs', 'documentation', 'developer', 'api']):
        return "docs"
    if path.startswith('/docs') or '/documentation/' in path:
        return "docs"
    
    # Check for e-commerce indicators
    ecommerce_terms = [
        'shop', 'store', 'buy', 'cart', 'checkout', 
        'product', 'catalog', 'item', 'purchase'
    ]
    if any(term in domain for term in ecommerce_terms):
        return "ecommerce"
    if any(f"/{term}" in path for term in ecommerce_terms):
        return "ecommerce"
    
    # Check for real estate sites
    real_estate_terms = ['realty', 'property', 'home', 'house', 'real-estate', 'apartment', 'housing', 'realtor']
    if any(term in domain for term in real_estate_terms):
        return "real_estate"
    if any(term in path for term in ['property', 'properties', 'for-sale', 'rent', 'homes', 'housing']):
        return "real_estate"
    
    # Check for job sites
    job_terms = ['job', 'career', 'employ', 'hire', 'recruit', 'work', 'hiring']
    if any(term in domain for term in job_terms):
        return "jobs"
    if any(term in path for term in ['job', 'career', 'position', 'employment', 'vacancy']):
        return "jobs"
    
    # Check for academic/research sites
    academic_terms = ['research', 'academic', 'university', 'college', 'institute', 'journal', 'science']
    if any(term in domain for term in academic_terms):
        return "academic"
    if any(term in path for term in ['research', 'publication', 'paper', 'journal', 'study']):
        return "academic"
    
    # Check for blog sites
    if 'blog' in domain or '/blog/' in path:
        return "blog"
    
    # Check for forum/community sites
    forum_terms = ['forum', 'community', 'discuss', 'board']
    if any(term in domain for term in forum_terms):
        return "forum"
    
    # Check for news sites
    news_terms = ['news', 'article', 'post', 'times', 'daily', 'tribune']
    if any(term in domain for term in news_terms):
        return "news"
    
    # Check for travel sites
    travel_terms = ['travel', 'tour', 'vacation', 'hotel', 'flight', 'trip']
    if any(term in domain for term in travel_terms):
        return "travel"
    if any(term in path for term in ['travel', 'destination', 'hotel', 'vacation', 'flight']):
        return "travel"
    
    # If content is available, use it for additional checks
    if content:
        detected_type = _detect_site_type_from_content(content, domain, path)
        if detected_type:
            return detected_type
    
    # Default to unknown if no specific type is detected
    logger.info(f"Could not determine site type for {url}, defaulting to 'unknown'")
    return "unknown"

def _check_tld_for_site_type(domain: str) -> Optional[str]:
    """
    Check if the TLD indicates a specific site type.
    
    Args:
        domain: The domain to check
        
    Returns:
        Site type if detected from TLD, None otherwise
    """
    for tld, site_type in TLD_SITE_TYPE_HINTS.items():
        if domain.endswith(f".{tld}"):
            if site_type:
                logger.info(f"Detected site type '{site_type}' based on TLD '{tld}'")
                return site_type
            # If TLD is recognized but needs more context, we'll return None to continue checking
    return None

def _detect_site_type_from_content(content: str, domain: str, path: str) -> Optional[str]:
    """
    Detect site type from content with enhanced pattern recognition.
    
    Args:
        content: HTML content
        domain: Website domain
        path: URL path
        
    Returns:
        Detected site type or None if no definitive match
    """
    # Convert content to lowercase for case-insensitive matching
    content_lower = content.lower()
    
    # Look for meta tags that indicate site type
    meta_types = _extract_site_type_from_meta_tags(content)
    if meta_types:
        return meta_types
    
    # Check for e-commerce content indicators
    ecommerce_indicators = [
        'add to cart', 'buy now', 'shopping cart', 'checkout', 'product catalog',
        'price', 'shipping', 'add to wishlist', 'payment method', 'stock'
    ]
    if _count_terms_in_content(content_lower, ecommerce_indicators) >= 3:
        return "ecommerce"
    
    # Check for real estate indicators
    real_estate_indicators = [
        'property', 'bedroom', 'bathroom', 'square feet', 'sqft', 'real estate', 
        'for sale', 'for rent', 'mortgage', 'listing', 'mls', 'agent'
    ]
    if _count_terms_in_content(content_lower, real_estate_indicators) >= 3:
        return "real_estate"
    
    # Check for documentation indicators
    docs_indicators = [
        'documentation', 'api reference', 'developer guide', 'function', 'method', 
        'parameter', 'return value', 'example code', 'getting started', 'installation'
    ]
    if _count_terms_in_content(content_lower, docs_indicators) >= 3:
        return "docs"
    
    # Check for blog indicators
    blog_indicators = [
        'blog post', 'author', 'posted on', 'comments', 'read more', 
        'recent posts', 'categories', 'tags', 'archive'
    ]
    if _count_terms_in_content(content_lower, blog_indicators) >= 3:
        return "blog"
    
    # Check for forum indicators
    forum_indicators = [
        'thread', 'post reply', 'discussion', 'forum', 'topic', 
        'member', 'join date', 'post count', 'moderator'
    ]
    if _count_terms_in_content(content_lower, forum_indicators) >= 3:
        return "forum"
    
    # Check for news indicators
    news_indicators = [
        'breaking news', 'latest news', 'published on', 'editor', 'journalist',
        'article', 'headline', 'press release', 'news article'
    ]
    if _count_terms_in_content(content_lower, news_indicators) >= 3:
        return "news"
    
    # Check for job site indicators
    job_indicators = [
        'job description', 'apply now', 'job requirements', 'qualifications', 
        'experience required', 'job title', 'salary', 'benefits', 'employer'
    ]
    if _count_terms_in_content(content_lower, job_indicators) >= 3:
        return "jobs"
    
    # Check for academic site indicators
    academic_indicators = [
        'research paper', 'journal', 'abstract', 'publication', 'citation',
        'doi', 'peer review', 'academic', 'faculty', 'student', 'professor'
    ]
    if _count_terms_in_content(content_lower, academic_indicators) >= 3:
        return "academic"
    
    # Check for travel site indicators
    travel_indicators = [
        'destination', 'hotel', 'flight', 'booking', 'accommodation', 'tour',
        'vacation', 'trip', 'travel guide', 'itinerary', 'resort'
    ]
    if _count_terms_in_content(content_lower, travel_indicators) >= 3:
        return "travel"
    
    # Check for government site indicators
    government_indicators = [
        'government', 'official', 'department', 'agency', 'public service', 
        'legislation', 'regulation', 'policy', 'federal', 'state', 'municipal'
    ]
    if _count_terms_in_content(content_lower, government_indicators) >= 3:
        return "government"
    
    # No definitive match
    return None

def _extract_site_type_from_meta_tags(content: str) -> Optional[str]:
    """
    Extract site type hints from meta tags.
    
    Args:
        content: HTML content
        
    Returns:
        Detected site type or None
    """
    meta_patterns = [
        (r'<meta\s+[^>]*name\s*=\s*["\']description["\'][^>]*content\s*=\s*["\'](.*?)["\']', 1),
        (r'<meta\s+[^>]*content\s*=\s*["\'](.*?)["\'][^>]*name\s*=\s*["\']description["\']', 1),
        (r'<meta\s+[^>]*name\s*=\s*["\']keywords["\'][^>]*content\s*=\s*["\'](.*?)["\']', 1),
        (r'<meta\s+[^>]*content\s*=\s*["\'](.*?)["\'][^>]*name\s*=\s*["\']keywords["\']', 1),
        (r'<meta\s+[^>]*property\s*=\\s*["\'](og:type)["\'][^>]*content\s*=\s*["\'](.*?)["\']', 2),
        (r'<meta\s+[^>]*content\s*=\s*["\'](.*?)["\'][^>]*property\s*=\s*["\'](og:type)["\']', 1)
    ]
    
    meta_content = []
    for pattern, group in meta_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                meta_content.append(matches[0][group-1])
            else:
                meta_content.append(matches[0])
    
    meta_text = ' '.join(meta_content).lower()
    
    # Map meta content to site types
    type_mapping = [
        (r'\bshop\b|\becommerce\b|\bproduct\b|\bstore\b|\bcart\b', "ecommerce"),
        (r'\breal\s*estate\b|\bproperty\b|\bhome\s*for\s*sale\b|\bhousing\b', "real_estate"),
        (r'\bjob\b|\bcareer\b|\bemployment\b|\bhiring\b', "jobs"),
        (r'\bblog\b|\barticle\b|\bpost\b', "blog"),
        (r'\bnews\b|\bheadline\b|\blatest\b', "news"),
        (r'\bdocumentation\b|\bapi\b|\bdeveloper\b', "docs"),
        (r'\bforum\b|\bcommunity\b|\bdiscussion\b', "forum"),
        (r'\bacademic\b|\bresearch\b|\bstudy\b|\buniversity\b|\beducation\b', "academic"),
        (r'\btravel\b|\bhotel\b|\bvacation\b|\bdestination\b', "travel"),
        (r'\bgovernment\b|\bofficial\b|\bagency\b|\bdepartment\b', "government")
    ]
    
    for pattern, site_type in type_mapping:
        if re.search(pattern, meta_text):
            logger.info(f"Detected site type '{site_type}' from meta tags")
            return site_type
    
    # Check for specific Open Graph types
    og_type_mapping = {
        'product': 'ecommerce',
        'article': 'blog',
        'news.article': 'news',
        'website': None,  # Too generic
        'business.business': None  # Too generic
    }
    
    og_type_pattern = r'<meta\s+[^>]*property\s*=\s*["\']og:type["\'][^>]*content\s*=\s*["\'](.*?)["\']'
    og_matches = re.findall(og_type_pattern, content, re.IGNORECASE)
    
    if og_matches:
        og_type = og_matches[0].lower()
        if og_type in og_type_mapping and og_type_mapping[og_type]:
            logger.info(f"Detected site type '{og_type_mapping[og_type]}' from og:type tag")
            return og_type_mapping[og_type]
    
    return None

def _count_terms_in_content(content: str, terms: List[str]) -> int:
    """
    Count how many terms from a list appear in the content.
    
    Args:
        content: Text content to search in
        terms: List of terms to look for
        
    Returns:
        Number of terms found in the content
    """
    count = 0
    for term in terms:
        if term in content:
            count += 1
    return count

def get_depth_settings_for_site(site_type: str, url: str) -> Dict[str, Any]:
    """
    Get depth-related settings for a specific site and URL.
    
    Args:
        site_type: The detected site type
        url: The URL being crawled
        
    Returns:
        Dictionary with depth-related settings
    """
    # Get the base settings for the site type
    settings = get_site_settings(site_type)
    
    # Special case adjustments based on URL patterns
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    
    # Adjust depth for deep documentation sites
    if site_type == "docs" and (domain.startswith("docs.") or "/reference/" in path):
        settings["max_depth"] += 1
        settings["max_pages"] += 20
    
    # Adjust depth for product category pages in e-commerce sites
    elif site_type == "ecommerce" and any(pattern in path for pattern in ['/category/', '/categories/', '/products/']):
        settings["max_depth"] += 1
        settings["follow_pagination"] = True
        
    # Adjust for real estate listing pages
    elif site_type == "real_estate" and any(pattern in path for pattern in ['/listing/', '/search/', '/properties/']):
        settings["follow_pagination"] = True
        settings["max_pages"] += 10
        
    # Adjust for blog archives
    elif site_type == "blog" and any(pattern in path for pattern in ['/archive/', '/category/', '/tag/']):
        settings["follow_pagination"] = True
        
    # Adjust for job search results
    elif site_type == "jobs" and any(pattern in path for pattern in ['/search/', '/jobs/', '/careers/']):
        settings["follow_pagination"] = True
        settings["max_pages"] += 5
        
    # Adjust for academic search or publication listings
    elif site_type == "academic" and any(pattern in path for pattern in ['/search/', '/publications/', '/papers/']):
        settings["follow_pagination"] = True
        settings["max_pages"] += 10
        
    # Adjust for travel search results or listings
    elif site_type == "travel" and any(pattern in path for pattern in ['/search/', '/hotels/', '/flights/']):
        settings["follow_pagination"] = True
        settings["max_pages"] += 5
        
    # For unknown sites with promising URLs, be more thorough
    elif site_type == "unknown" and any(term in url for term in ['search', 'find', 'listing', 'result']):
        settings["max_depth"] += 1
        settings["follow_pagination"] = True
        
    return settings

def get_site_extraction_selectors(site_type: str) -> Dict[str, List[str]]:
    """
    Get CSS selectors for different content elements based on site type.
    
    Args:
        site_type: The type of website
        
    Returns:
        Dictionary with CSS selectors for different content elements
    """
    # Common selectors that work across many sites
    common_selectors = {
        "title": ["h1", "h1.main-title", ".title", ".main-title", "[itemprop='name']"],
        "content": ["article", ".content", ".main-content", "[itemprop='articleBody']", ".body"],
        "image": ["img.main-image", "[itemprop='image']", ".featured-image"]
    }
    
    # Site type specific selectors
    site_specific_selectors = {
        "ecommerce": {
            "title": [".product-title", "[itemprop='name']", ".product-name", "h1"],
            "price": [".price", "[itemprop='price']", ".product-price", ".price-box"],
            "description": [".product-description", "[itemprop='description']", ".description", "#description"],
            "image": [".product-image", "[itemprop='image']", ".main-image", ".gallery img"],
            "options": [".product-options", ".options", ".variants"],
            "stock": [".stock", ".inventory", ".availability"]
        },
        "real_estate": {
            "title": [".listing-title", ".property-title", "h1"],
            "price": [".listing-price", ".price", "[itemprop='price']"],
            "description": [".listing-description", ".property-description", "[itemprop='description']"],
            "image": [".listing-image", ".property-image", ".main-image", ".gallery img"],
            "details": [".listing-details", ".property-details", ".details"],
            "features": [".listing-features", ".property-features", ".features", "ul.features"],
            "address": [".listing-address", ".property-address", "[itemprop='address']"],
            "agent": [".listing-agent", ".agent-info", ".broker"]
        },
        "blog": {
            "title": [".post-title", ".article-title", "h1", "[itemprop='headline']"],
            "date": [".post-date", ".published", "[itemprop='datePublished']"],
            "author": [".post-author", "[itemprop='author']", ".author"],
            "content": [".post-content", ".article-content", "[itemprop='articleBody']"],
            "image": [".post-image", ".featured-image", "[itemprop='image']"],
            "tags": [".post-tags", ".tags", ".categories"]
        },
        "news": {
            "title": [".article-title", ".headline", "h1", "[itemprop='headline']"],
            "date": [".article-date", ".published-date", "[itemprop='datePublished']"],
            "author": [".article-author", ".byline", "[itemprop='author']"],
            "content": [".article-content", ".story-body", "[itemprop='articleBody']"],
            "image": [".article-image", ".main-image", "[itemprop='image']"],
            "category": [".article-category", ".section"]
        },
        "jobs": {
            "title": [".job-title", ".position-title", "h1"],
            "company": [".company-name", ".employer", "[itemprop='hiringOrganization']"],
            "location": [".job-location", ".location", "[itemprop='jobLocation']"],
            "description": [".job-description", ".description", "[itemprop='description']"],
            "requirements": [".job-requirements", ".requirements", ".qualifications"],
            "salary": [".salary", ".compensation", "[itemprop='baseSalary']"],
            "apply": [".apply-button", ".application-link", ".job-apply"]
        },
        "academic": {
            "title": [".paper-title", ".article-title", "h1"],
            "authors": [".authors", ".author-list", "[itemprop='author']"],
            "abstract": [".abstract", "#abstract", "[itemprop='abstract']"],
            "publication_date": [".publication-date", ".published-date", "[itemprop='datePublished']"],
            "journal": [".journal", ".publication", "[itemprop='isPartOf']"],
            "keywords": [".keywords", ".tags", "[itemprop='keywords']"],
            "content": [".paper-content", ".article-content", "[itemprop='articleBody']"]
        },
        "forum": {
            "thread_title": [".thread-title", ".topic-title", "h1"],
            "posts": [".post", ".message", ".comment"],
            "post_author": [".post-author", ".author", ".username"],
            "post_date": [".post-date", ".date", ".timestamp"],
            "post_content": [".post-content", ".message-content", ".content"]
        },
        "docs": {
            "title": [".document-title", ".page-title", "h1"],
            "content": [".document-content", ".content", "article", ".main"],
            "navigation": [".navigation", ".sidebar", ".toc", ".menu"],
            "code_sample": ["pre", "code", ".code-sample", ".highlight"]
        },
        "travel": {
            "title": [".hotel-name", ".destination-name", ".tour-name", "h1"],
            "price": [".price", ".rate", ".cost"],
            "description": [".description", ".details", "[itemprop='description']"],
            "images": [".gallery", ".photos", ".images", "[itemprop='image']"],
            "rating": [".rating", ".stars", "[itemprop='ratingValue']"],
            "amenities": [".amenities", ".facilities", ".features"],
            "location": [".location", ".address", "[itemprop='address']"]
        },
        "government": {
            "title": [".page-title", ".content-title", "h1"],
            "department": [".department", ".agency", ".organization"],
            "content": [".page-content", ".content", "article", ".main"],
            "date": [".date", ".published", ".updated"],
            "contact": [".contact", ".contact-info", ".contact-details"]
        }
    }
    
    # Get the specific selectors for the site type, or fall back to common selectors
    if site_type in site_specific_selectors:
        return {**common_selectors, **site_specific_selectors[site_type]}
    else:
        return common_selectors

def should_index_as_listings_page(url: str, site_type: str, content: Optional[str] = None) -> bool:
    """
    Determine if a page should be indexed as a listings page.
    
    Args:
        url: The URL of the page
        site_type: The detected site type
        content: Optional HTML content
        
    Returns:
        True if the page should be indexed as a listings page
    """
    # Parse the URL
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    
    # Check common listing page URL patterns
    common_listing_patterns = ['/search', '/listings', '/list', '/results', '/find']
    if any(pattern in path for pattern in common_listing_patterns):
        return True
    
    # Check site-specific listing patterns
    site_specific_patterns = {
        "real_estate": ['/properties', '/homes', '/for-sale', '/rent', '/listings'],
        "ecommerce": ['/products', '/shop', '/catalog', '/category', '/categories'],
        "jobs": ['/jobs', '/careers', '/positions', '/vacancies'],
        "travel": ['/hotels', '/flights', '/tours', '/destinations', '/packages'],
        "academic": ['/papers', '/publications', '/research', '/journals'],
        "news": ['/articles', '/news', '/stories', '/press-releases']
    }
    
    if site_type in site_specific_patterns:
        if any(pattern in path for pattern in site_specific_patterns[site_type]):
            return True
    
    # Check for listings in content
    if content:
        # Common listing container CSS classes
        listing_containers = [
            'listing-results', 'search-results', 'product-list', 'product-grid',
            'job-listings', 'property-list', 'results-list', 'hotel-list',
            'article-list', 'publication-list'
        ]
        
        # Check if any of these appear in the content
        for container in listing_containers:
            if f'class="{container}"' in content or f"class='{container}'" in content:
                return True
        
        # Check for multiple similar elements which is typical for listings
        listing_item_patterns = [
            (r'<div\s+class="[^"]*(?:item|listing|result|product|card)[^"]*"', 3),  # 3 or more similar divs
            (r'<li\s+class="[^"]*(?:item|listing|result|product|card)[^"]*"', 3),  # 3 or more similar list items
            (r'<article\s+class="[^"]*(?:item|listing|result|post|product)[^"]*"', 3)  # 3 or more similar articles
        ]
        
        for pattern, min_count in listing_item_patterns:
            matches = re.findall(pattern, content)
            if len(matches) >= min_count:
                return True
    
    return False

def get_site_specific_search_terms(site_type: str, search_term: str) -> List[str]:
    """
    Get site-specific search terms based on the detected site type.
    
    Args:
        site_type: The type of website
        search_term: The original search term
        
    Returns:
        List of modified search terms optimized for the site type
    """
    search_terms = [search_term]  # Start with the original term
    
    if site_type == "real_estate":
        # Add real estate specific terms
        if re.search(r'\b\d+\s+bed', search_term, re.IGNORECASE):
            # Already has bedroom info, no need to add
            pass
        else:
            # Add common property searches
            search_terms.append(f"{search_term} homes for sale")
            search_terms.append(f"{search_term} houses")
            search_terms.append(f"property {search_term}")
    
    elif site_type == "ecommerce":
        # Add e-commerce specific terms
        search_terms.append(f"buy {search_term}")
        search_terms.append(f"{search_term} price")
        search_terms.append(f"{search_term} product")
    
    elif site_type == "jobs":
        # Add job search specific terms
        search_terms.append(f"{search_term} jobs")
        search_terms.append(f"{search_term} careers")
        search_terms.append(f"hiring {search_term}")
    
    elif site_type == "travel":
        # Add travel specific terms
        search_terms.append(f"{search_term} hotel")
        search_terms.append(f"visit {search_term}")
        search_terms.append(f"{search_term} vacation")
    
    elif site_type == "academic":
        # Add academic specific terms
        search_terms.append(f"{search_term} research")
        search_terms.append(f"{search_term} paper")
        search_terms.append(f"{search_term} study")
    
    elif site_type == "news":
        # Add news specific terms
        search_terms.append(f"{search_term} news")
        search_terms.append(f"latest {search_term}")
        search_terms.append(f"{search_term} article")
    
    elif site_type == "docs":
        # Add documentation specific terms
        search_terms.append(f"{search_term} api")
        search_terms.append(f"{search_term} documentation")
        search_terms.append(f"{search_term} example")
    
    return search_terms

def detect_listing_type_from_content(content: str, site_type: str) -> Optional[str]:
    """
    Detect the specific listing type from page content.
    
    Args:
        content: HTML content
        site_type: The detected site type
        
    Returns:
        Specific listing type or None
    """
    content_lower = content.lower()
    
    # Site-specific listings detection
    if site_type == "real_estate":
        # Check for residential properties
        residential_terms = ['bedroom', 'bathroom', 'home', 'house', 'apartment', 'condo']
        residential_count = _count_terms_in_content(content_lower, residential_terms)
        
        # Check for commercial properties
        commercial_terms = ['office', 'retail', 'commercial property', 'business', 'industrial']
        commercial_count = _count_terms_in_content(content_lower, commercial_terms)
        
        # Check for land/lots
        land_terms = ['land', 'lot', 'acre', 'development', 'vacant']
        land_count = _count_terms_in_content(content_lower, land_terms)
        
        # Determine type based on count
        if residential_count > commercial_count and residential_count > land_count:
            return "residential"
        elif commercial_count > residential_count and commercial_count > land_count:
            return "commercial"
        elif land_count > residential_count and land_count > commercial_count:
            return "land"
    
    elif site_type == "ecommerce":
        # Check for product categories
        product_categories = {
            "electronics": ['electronics', 'device', 'gadget', 'computer', 'phone', 'tv'],
            "clothing": ['clothing', 'apparel', 'fashion', 'wear', 'dress', 'shirt', 'pants'],
            "home": ['furniture', 'home', 'decor', 'kitchen', 'appliance', 'garden'],
            "beauty": ['beauty', 'cosmetic', 'makeup', 'skin care', 'perfume'],
            "toys": ['toy', 'game', 'kids', 'children', 'play']
        }
        
        # Find the category with the most matches
        max_count = 0
        best_category = None
        
        for category, terms in product_categories.items():
            count = _count_terms_in_content(content_lower, terms)
            if count > max_count:
                max_count = count
                best_category = category
                
        if best_category and max_count >= 2:
            return best_category
    
    elif site_type == "jobs":
        # Check for job industry
        industries = {
            "tech": ['software', 'developer', 'it', 'tech', 'programmer', 'engineer'],
            "healthcare": ['healthcare', 'medical', 'nurse', 'doctor', 'clinical', 'hospital'],
            "finance": ['finance', 'banking', 'accounting', 'financial', 'investment'],
            "education": ['education', 'teaching', 'school', 'academic', 'teacher', 'professor'],
            "sales": ['sales', 'marketing', 'retail', 'customer', 'representative']
        }
        
        # Find the industry with the most matches
        max_count = 0
        best_industry = None
        
        for industry, terms in industries.items():
            count = _count_terms_in_content(content_lower, terms)
            if count > max_count:
                max_count = count
                best_industry = industry
                
        if best_industry and max_count >= 2:
            return best_industry
    
    return None