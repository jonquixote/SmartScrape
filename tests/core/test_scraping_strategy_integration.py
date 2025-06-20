"""
Tests for content extraction and cleaning across different scraping strategies.

This module tests how HTML cleaning integrates with different scraping strategies
and validates that content extraction works correctly for various use cases.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock

from core.pipeline.context import PipelineContext
from core.pipeline.dto import PipelineRequest, PipelineResponse, ResponseStatus
from core.pipeline.stages.processing.html_processing import (
    HTMLCleaningStage,
    ContentExtractionStage,
    CleaningStrategy,
    ExtractionAlgorithm
)


class TestScrapingStrategyIntegration:
    """Test HTML cleaning integration with different scraping strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.context = PipelineContext()
    
    # ======================== DOM STRATEGY TESTS ========================
    
    @pytest.mark.asyncio
    async def test_dom_strategy_news_article(self):
        """Test DOM strategy with news article content."""
        news_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breaking: Major Economic Changes</title>
            <script>trackPageView('news_article');</script>
            <style>.article { font-family: Georgia; }</style>
        </head>
        <body>
            <nav class="site-nav">
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/business">Business</a></li>
                    <li><a href="/politics">Politics</a></li>
                </ul>
            </nav>
            
            <article class="news-article">
                <header>
                    <h1>Breaking: Major Economic Changes Announced</h1>
                    <div class="article-meta">
                        <time datetime="2023-12-01T10:30:00Z">December 1, 2023 10:30 AM</time>
                        <span class="author">by Economic Reporter Jane Smith</span>
                        <span class="category">Business</span>
                    </div>
                </header>
                
                <div class="article-body">
                    <p class="lead">Government officials announced sweeping economic reforms that will impact businesses and consumers nationwide.</p>
                    
                    <p>The new policies, effective January 1st, include changes to tax structures, business regulations, and consumer protection laws. Key highlights include:</p>
                    
                    <ul class="key-points">
                        <li>Corporate tax rate reduction from 21% to 18%</li>
                        <li>Increased small business tax credits</li>
                        <li>Enhanced consumer privacy protections</li>
                        <li>Streamlined regulatory approval processes</li>
                    </ul>
                    
                    <blockquote class="official-statement">
                        "These reforms represent a balanced approach to economic growth while protecting consumer interests," said Treasury Secretary John Anderson.
                    </blockquote>
                    
                    <h2>Impact on Small Businesses</h2>
                    <p>Small business owners are expected to benefit significantly from the new tax credit structure. The reforms include:</p>
                    
                    <div class="info-box">
                        <h3>Small Business Benefits</h3>
                        <ul>
                            <li>Up to $50,000 in additional tax credits</li>
                            <li>Simplified filing procedures</li>
                            <li>Reduced compliance costs</li>
                        </ul>
                    </div>
                    
                    <h2>Consumer Protection Measures</h2>
                    <p>New consumer protection laws will strengthen privacy rights and increase transparency in financial services.</p>
                </div>
            </article>
            
            <aside class="related-content">
                <h3>Related Articles</h3>
                <ul>
                    <li><a href="/article1">Previous Economic Policy Changes</a></li>
                    <li><a href="/article2">Business Community Reactions</a></li>
                </ul>
            </aside>
            
            <script>initArticleTracking();</script>
        </body>
        </html>
        """
        
        # Test with DOM strategy configuration
        cleaning_stage = HTMLCleaningStage(config={
            "strategy": "MODERATE",
            "preserve_structure": True
        })
        
        extraction_stage = ContentExtractionStage(config={
            "algorithm": "DENSITY_BASED",
            "focus_selectors": ["article", ".news-article", ".article-body"]
        })
        
        request = PipelineRequest(
            source="news_site",
            data={"html": news_html},
            metadata={
                "url": "https://news.example.com/breaking-economic-changes",
                "extraction_strategy": "dom_strategy",
                "content_type": "news_article"
            }
        )
        
        # Clean HTML first
        cleaned_response = await cleaning_stage.process_request(request, self.context)
        assert cleaned_response.status == ResponseStatus.SUCCESS
        
        cleaned_html = cleaned_response.data["html"]
        
        # Verify script removal
        assert 'trackPageView(' not in cleaned_html
        assert 'initArticleTracking()' not in cleaned_html
        
        # Verify content preservation
        assert 'Breaking: Major Economic Changes Announced' in cleaned_html
        assert 'Government officials announced sweeping economic reforms' in cleaned_html
        assert 'Corporate tax rate reduction from 21% to 18%' in cleaned_html
        assert 'Treasury Secretary John Anderson' in cleaned_html
        assert 'Up to $50,000 in additional tax credits' in cleaned_html
        
        # Extract content using cleaned HTML
        extraction_request = PipelineRequest(
            source="news_site",
            data={"html": cleaned_html},
            metadata=request.metadata
        )
        
        extracted_response = await extraction_stage.process_request(extraction_request, self.context)
        assert extracted_response.status == ResponseStatus.SUCCESS
        
        # Verify extracted content contains key information
        extracted_data = extracted_response.data
        assert "extracted_content" in extracted_data

    @pytest.mark.asyncio
    async def test_dom_strategy_ecommerce_product(self):
        """Test DOM strategy with e-commerce product page."""
        product_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Premium Wireless Headphones - TechStore</title>
            <script>
                dataLayer.push({
                    'event': 'product_view',
                    'product_id': 'TWH-001',
                    'product_name': 'Premium Wireless Headphones',
                    'price': 199.99
                });
            </script>
            <style>
                .product-container { max-width: 1200px; margin: 0 auto; }
                .price { color: #e74c3c; font-size: 28px; font-weight: bold; }
                .discounted { text-decoration: line-through; color: #999; }
            </style>
        </head>
        <body>
            <header class="site-header">
                <div class="logo">TechStore</div>
                <nav>
                    <ul>
                        <li><a href="/electronics">Electronics</a></li>
                        <li><a href="/audio">Audio</a></li>
                        <li><a href="/headphones">Headphones</a></li>
                    </ul>
                </nav>
                <div class="cart-icon">
                    <span id="cart-count">0</span>
                </div>
            </header>
            
            <main class="product-container">
                <div class="product-gallery">
                    <img src="/images/headphones-main.jpg" alt="Premium Wireless Headphones">
                    <div class="thumbnail-list">
                        <img src="/images/headphones-side.jpg" alt="Side view">
                        <img src="/images/headphones-back.jpg" alt="Back view">
                    </div>
                </div>
                
                <div class="product-info">
                    <h1 class="product-title">Premium Wireless Headphones</h1>
                    <div class="product-rating">
                        <span class="stars">★★★★★</span>
                        <span class="rating-count">(247 reviews)</span>
                    </div>
                    
                    <div class="pricing">
                        <span class="price">$199.99</span>
                        <span class="discounted">$249.99</span>
                        <span class="savings">Save $50.00 (20% off)</span>
                    </div>
                    
                    <div class="product-highlights">
                        <h2>Key Features</h2>
                        <ul>
                            <li>Active Noise Cancellation (ANC)</li>
                            <li>30-hour battery life</li>
                            <li>Premium leather ear cushions</li>
                            <li>Bluetooth 5.0 with aptX HD</li>
                            <li>Quick charge: 15 minutes = 3 hours playback</li>
                        </ul>
                    </div>
                    
                    <div class="product-options">
                        <div class="color-selection">
                            <label>Color:</label>
                            <select id="color-select">
                                <option value="black">Matte Black</option>
                                <option value="white">Pearl White</option>
                                <option value="brown">Cognac Brown</option>
                            </select>
                        </div>
                        
                        <div class="warranty-options">
                            <label>
                                <input type="checkbox" id="extended-warranty">
                                Extended 3-year warranty (+$29.99)
                            </label>
                        </div>
                    </div>
                    
                    <div class="purchase-section">
                        <button onclick="addToCart('TWH-001')" class="add-to-cart-btn">Add to Cart</button>
                        <button onclick="buyNow('TWH-001')" class="buy-now-btn">Buy Now</button>
                        <div class="shipping-info">
                            <p>✓ Free shipping on orders over $100</p>
                            <p>✓ 30-day return policy</p>
                            <p>✓ 2-year manufacturer warranty</p>
                        </div>
                    </div>
                </div>
                
                <section class="product-description">
                    <h2>Product Description</h2>
                    <p>Experience premium audio quality with our flagship wireless headphones. Engineered with precision and crafted with premium materials, these headphones deliver exceptional sound clarity and comfort for extended listening sessions.</p>
                    
                    <h3>Technical Specifications</h3>
                    <table class="specs-table">
                        <tr><td>Driver Size</td><td>40mm dynamic drivers</td></tr>
                        <tr><td>Frequency Response</td><td>20Hz - 20kHz</td></tr>
                        <tr><td>Impedance</td><td>32 ohms</td></tr>
                        <tr><td>Weight</td><td>285g</td></tr>
                        <tr><td>Battery Type</td><td>Lithium-ion rechargeable</td></tr>
                        <tr><td>Charging Port</td><td>USB-C</td></tr>
                    </table>
                    
                    <h3>What's in the Box</h3>
                    <ul>
                        <li>Premium Wireless Headphones</li>
                        <li>USB-C charging cable</li>
                        <li>3.5mm audio cable</li>
                        <li>Premium carrying case</li>
                        <li>Quick start guide</li>
                        <li>Warranty card</li>
                    </ul>
                </section>
            </main>
            
            <aside class="recommendations">
                <h3>Customers also viewed</h3>
                <div class="product-card" onclick="viewProduct('TWE-002')">
                    <h4>Wireless Earbuds Pro</h4>
                    <p>$149.99</p>
                </div>
                <div class="product-card" onclick="viewProduct('TWS-003')">
                    <h4>Bluetooth Speaker</h4>
                    <p>$89.99</p>
                </div>
            </aside>
            
            <script>
                // Initialize product page functionality
                document.addEventListener('DOMContentLoaded', function() {
                    initProductGallery();
                    setupColorSelection();
                    loadRecommendations();
                    trackProductView();
                });
                
                function addToCart(productId) {
                    // Add to cart logic
                    gtag('event', 'add_to_cart', {
                        'currency': 'USD',
                        'value': 199.99,
                        'items': [{
                            'item_id': productId,
                            'item_name': 'Premium Wireless Headphones',
                            'price': 199.99,
                            'quantity': 1
                        }]
                    });
                }
            </script>
        </body>
        </html>
        """
        
        # Test with e-commerce focused configuration
        cleaning_stage = HTMLCleaningStage(config={
            "strategy": "MODERATE",
            "preserve_product_info": True
        })
        
        extraction_stage = ContentExtractionStage(config={
            "algorithm": "SELECTOR_BASED",
            "focus_selectors": [".product-info", ".product-description", ".specs-table"]
        })
        
        request = PipelineRequest(
            source="ecommerce_site",
            data={"html": product_html},
            metadata={
                "url": "https://techstore.com/products/premium-wireless-headphones",
                "extraction_strategy": "dom_strategy",
                "content_type": "product_page"
            }
        )
        
        # Clean HTML
        cleaned_response = await cleaning_stage.process_request(request, self.context)
        assert cleaned_response.status == ResponseStatus.SUCCESS
        
        cleaned_html = cleaned_response.data["html"]
        
        # Verify script and tracking removal
        assert 'dataLayer.push' not in cleaned_html
        assert 'addToCart(' not in cleaned_html
        assert 'buyNow(' not in cleaned_html
        assert 'viewProduct(' not in cleaned_html
        assert 'initProductGallery()' not in cleaned_html
        assert 'gtag(' not in cleaned_html
        
        # Verify product information preservation
        assert 'Premium Wireless Headphones' in cleaned_html
        assert '$199.99' in cleaned_html
        assert '$249.99' in cleaned_html
        assert 'Save $50.00 (20% off)' in cleaned_html
        assert 'Active Noise Cancellation (ANC)' in cleaned_html
        assert '30-hour battery life' in cleaned_html
        assert 'Bluetooth 5.0 with aptX HD' in cleaned_html
        assert '40mm dynamic drivers' in cleaned_html
        assert '20Hz - 20kHz' in cleaned_html
        assert 'Free shipping on orders over $100' in cleaned_html
        assert '30-day return policy' in cleaned_html
        
        # Extract content
        extraction_request = PipelineRequest(
            source="ecommerce_site",
            data={"html": cleaned_html},
            metadata=request.metadata
        )
        
        extracted_response = await extraction_stage.process_request(extraction_request, self.context)
        assert extracted_response.status == ResponseStatus.SUCCESS

    # ======================== AI-GUIDED STRATEGY TESTS ========================
    
    @pytest.mark.asyncio
    async def test_ai_guided_strategy_research_paper(self):
        """Test AI-guided strategy with academic research paper."""
        research_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Machine Learning in Web Scraping: A Comprehensive Analysis</title>
            <meta name="description" content="Research paper on ML applications in web scraping">
            <script>trackAcademicPaper('ml-web-scraping-2023');</script>
            <style>
                .paper-container { max-width: 800px; font-family: 'Times New Roman'; }
                .abstract { background: #f9f9f9; padding: 20px; margin: 20px 0; }
                .citation { font-style: italic; margin: 10px 0; }
            </style>
        </head>
        <body>
            <header class="journal-header">
                <div class="journal-info">
                    <h1>Journal of Advanced Computing</h1>
                    <p>Volume 45, Issue 3, December 2023</p>
                </div>
            </header>
            
            <main class="paper-container">
                <article class="research-paper">
                    <header class="paper-header">
                        <h1>Machine Learning in Web Scraping: A Comprehensive Analysis</h1>
                        <div class="authors">
                            <span class="author">Dr. Sarah Chen<sup>1</sup></span>,
                            <span class="author">Prof. Michael Rodriguez<sup>2</sup></span>,
                            <span class="author">Dr. Emily Watson<sup>1</sup></span>
                        </div>
                        <div class="affiliations">
                            <p><sup>1</sup>Department of Computer Science, Tech University</p>
                            <p><sup>2</sup>Institute of Data Science, Research Institute</p>
                        </div>
                        <div class="paper-meta">
                            <p>Received: October 15, 2023 | Accepted: November 20, 2023 | Published: December 1, 2023</p>
                            <p>DOI: 10.1234/jac.2023.45.3.001</p>
                        </div>
                    </header>
                    
                    <section class="abstract">
                        <h2>Abstract</h2>
                        <p><strong>Background:</strong> Web scraping has evolved significantly with the integration of machine learning techniques. This study examines the current state and future directions of ML-enhanced web scraping systems.</p>
                        
                        <p><strong>Methods:</strong> We conducted a systematic review of 150 research papers published between 2018-2023, analyzing ML applications in content extraction, pattern recognition, and adaptive scraping strategies.</p>
                        
                        <p><strong>Results:</strong> Our analysis reveals that ML-enhanced scrapers achieve 23% higher accuracy in content extraction and 35% better adaptation to website changes compared to traditional rule-based systems.</p>
                        
                        <p><strong>Conclusions:</strong> Machine learning significantly improves web scraping effectiveness, particularly in handling dynamic content and evolving website structures. Future research should focus on few-shot learning and transfer learning applications.</p>
                        
                        <p><strong>Keywords:</strong> web scraping, machine learning, content extraction, adaptive systems, natural language processing</p>
                    </section>
                    
                    <section class="introduction">
                        <h2>1. Introduction</h2>
                        <p>Web scraping, the automated extraction of data from websites, has become increasingly complex as web technologies evolve. Traditional rule-based scraping systems struggle with dynamic content, anti-bot measures, and frequent layout changes [1, 2].</p>
                        
                        <p>The integration of machine learning (ML) techniques into web scraping systems offers promising solutions to these challenges. ML-enhanced scrapers can adapt to changes, learn from patterns, and improve extraction accuracy over time [3, 4].</p>
                        
                        <p>This paper provides a comprehensive analysis of current ML applications in web scraping, evaluating their effectiveness and identifying future research directions.</p>
                    </section>
                    
                    <section class="methodology">
                        <h2>2. Methodology</h2>
                        <h3>2.1 Literature Review Process</h3>
                        <p>We conducted a systematic literature review following PRISMA guidelines. Our search strategy included the following databases:</p>
                        <ul>
                            <li>IEEE Xplore Digital Library</li>
                            <li>ACM Digital Library</li>
                            <li>Google Scholar</li>
                            <li>arXiv preprint repository</li>
                        </ul>
                        
                        <h3>2.2 Selection Criteria</h3>
                        <p>Papers were included if they met the following criteria:</p>
                        <ol>
                            <li>Published between January 2018 and October 2023</li>
                            <li>Focus on ML applications in web data extraction</li>
                            <li>Empirical evaluation of proposed methods</li>
                            <li>Written in English</li>
                        </ol>
                        
                        <h3>2.3 Data Extraction</h3>
                        <p>From each selected paper, we extracted information on:</p>
                        <ul>
                            <li>ML techniques used (supervised, unsupervised, reinforcement learning)</li>
                            <li>Application domains (e-commerce, news, social media, academic)</li>
                            <li>Performance metrics and evaluation methods</li>
                            <li>Limitations and future work suggestions</li>
                        </ul>
                    </section>
                    
                    <section class="results">
                        <h2>3. Results</h2>
                        <h3>3.1 Overview of Selected Studies</h3>
                        <p>Our systematic review identified 150 relevant papers. Figure 1 shows the distribution of publications by year, indicating increasing interest in ML-enhanced web scraping.</p>
                        
                        <div class="figure">
                            <img src="/images/publication-timeline.png" alt="Publication timeline graph">
                            <p class="caption"><strong>Figure 1:</strong> Number of publications on ML-enhanced web scraping by year (2018-2023)</p>
                        </div>
                        
                        <h3>3.2 ML Techniques in Web Scraping</h3>
                        <p>The most commonly used ML techniques include:</p>
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Technique</th>
                                    <th>Number of Papers</th>
                                    <th>Primary Application</th>
                                    <th>Average Accuracy</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Random Forest</td>
                                    <td>45</td>
                                    <td>Element classification</td>
                                    <td>87.3%</td>
                                </tr>
                                <tr>
                                    <td>Neural Networks</td>
                                    <td>38</td>
                                    <td>Content extraction</td>
                                    <td>91.2%</td>
                                </tr>
                                <tr>
                                    <td>SVM</td>
                                    <td>32</td>
                                    <td>Pattern recognition</td>
                                    <td>84.7%</td>
                                </tr>
                                <tr>
                                    <td>Deep Learning</td>
                                    <td>29</td>
                                    <td>Text understanding</td>
                                    <td>93.8%</td>
                                </tr>
                                <tr>
                                    <td>Reinforcement Learning</td>
                                    <td>6</td>
                                    <td>Adaptive strategies</td>
                                    <td>89.1%</td>
                                </tr>
                            </tbody>
                        </table>
                    </section>
                    
                    <section class="discussion">
                        <h2>4. Discussion</h2>
                        <h3>4.1 Key Findings</h3>
                        <p>Our analysis reveals several important trends in ML-enhanced web scraping:</p>
                        
                        <p><strong>Improved Accuracy:</strong> ML-based systems consistently outperform traditional rule-based approaches. Deep learning models show particular promise for complex content understanding tasks.</p>
                        
                        <p><strong>Adaptability:</strong> ML systems demonstrate superior ability to adapt to website changes without manual intervention, reducing maintenance costs by an average of 40%.</p>
                        
                        <p><strong>Scalability:</strong> Transfer learning techniques enable trained models to work across different websites and domains with minimal retraining.</p>
                        
                        <h3>4.2 Limitations and Challenges</h3>
                        <p>Despite promising results, several challenges remain:</p>
                        <ul>
                            <li>Training data requirements for supervised approaches</li>
                            <li>Computational overhead of complex models</li>
                            <li>Interpretability of ML decisions</li>
                            <li>Handling of adversarial anti-scraping measures</li>
                        </ul>
                    </section>
                    
                    <section class="conclusion">
                        <h2>5. Conclusion</h2>
                        <p>Machine learning has significantly advanced the field of web scraping, offering solutions to long-standing challenges in content extraction and system adaptability. Our systematic review demonstrates clear benefits in accuracy, robustness, and maintenance efficiency.</p>
                        
                        <p>Future research should focus on developing more efficient training methods, improving model interpretability, and addressing ethical considerations in automated web data collection.</p>
                    </section>
                    
                    <section class="references">
                        <h2>References</h2>
                        <ol class="reference-list">
                            <li class="citation">Smith, J., & Johnson, A. (2021). Adaptive web scraping using machine learning. <em>Journal of Web Technologies</em>, 15(3), 45-62.</li>
                            <li class="citation">Brown, M., et al. (2020). Deep learning for content extraction in e-commerce websites. <em>Proceedings of WWW 2020</em>, 123-134.</li>
                            <li class="citation">Davis, K. (2022). Transfer learning applications in web data extraction. <em>AI & Web Mining Conference</em>, 78-89.</li>
                            <li class="citation">Wilson, P., & Lee, S. (2023). Reinforcement learning for adaptive scraping strategies. <em>Machine Learning Journal</em>, 42(7), 156-171.</li>
                        </ol>
                    </section>
                </article>
            </main>
            
            <aside class="related-papers">
                <h3>Related Articles</h3>
                <ul>
                    <li><a href="/paper/ai-web-mining-2023">AI Techniques in Web Mining</a></li>
                    <li><a href="/paper/scraping-ethics-2023">Ethical Considerations in Web Scraping</a></li>
                </ul>
            </aside>
            
            <script>
                // Track paper reading behavior
                window.addEventListener('scroll', function() {
                    trackReadingProgress();
                });
                
                // Initialize citation tracking
                document.querySelectorAll('.citation').forEach(function(citation, index) {
                    citation.addEventListener('click', function() {
                        trackCitationClick(index + 1);
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Test with AI-guided strategy - should preserve more content for AI analysis
        cleaning_stage = HTMLCleaningStage(config={
            "strategy": "LIGHT",  # Less aggressive cleaning for AI processing
            "preserve_structure": True,
            "preserve_academic_content": True
        })
        
        extraction_stage = ContentExtractionStage(config={
            "algorithm": "AI_GUIDED",
            "focus_content": ["abstract", "methodology", "results", "conclusion"]
        })
        
        request = PipelineRequest(
            source="academic_journal",
            data={"html": research_html},
            metadata={
                "url": "https://journal.example.com/ml-web-scraping-analysis",
                "extraction_strategy": "ai_guided",
                "content_type": "research_paper",
                "intent": "extract academic paper content including title, authors, abstract, methodology, results, and conclusion"
            }
        )
        
        # Clean HTML with light strategy
        cleaned_response = await cleaning_stage.process_request(request, self.context)
        assert cleaned_response.status == ResponseStatus.SUCCESS
        
        cleaned_html = cleaned_response.data["html"]
        
        # Verify script removal but content preservation
        assert 'trackAcademicPaper(' not in cleaned_html
        assert 'trackReadingProgress()' not in cleaned_html
        assert 'trackCitationClick(' not in cleaned_html
        
        # Verify academic content preservation
        assert 'Machine Learning in Web Scraping: A Comprehensive Analysis' in cleaned_html
        assert 'Dr. Sarah Chen' in cleaned_html
        assert 'Prof. Michael Rodriguez' in cleaned_html
        assert 'Department of Computer Science, Tech University' in cleaned_html
        assert 'DOI: 10.1234/jac.2023.45.3.001' in cleaned_html
        
        # Abstract content
        assert 'Web scraping has evolved significantly' in cleaned_html
        assert 'systematic review of 150 research papers' in cleaned_html
        assert '23% higher accuracy in content extraction' in cleaned_html
        assert '35% better adaptation to website changes' in cleaned_html
        
        # Methodology content
        assert 'Literature Review Process' in cleaned_html
        assert 'PRISMA guidelines' in cleaned_html
        assert 'IEEE Xplore Digital Library' in cleaned_html
        
        # Results content
        assert 'Overview of Selected Studies' in cleaned_html
        assert 'Random Forest' in cleaned_html
        assert 'Neural Networks' in cleaned_html
        assert 'Deep Learning' in cleaned_html
        
        # References
        assert 'Smith, J., & Johnson, A. (2021)' in cleaned_html
        assert 'Journal of Web Technologies' in cleaned_html

    # ======================== MIXED CONTENT STRATEGY TESTS ========================
    
    @pytest.mark.asyncio
    async def test_mixed_content_extraction(self):
        """Test extraction of mixed content types on a single page."""
        mixed_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Technology Blog - Latest Updates</title>
            <script>initAnalytics('tech-blog-2023');</script>
            <style>
                .blog-post { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .product-review { background: #f0f8ff; }
                .news-update { background: #fff8f0; }
            </style>
        </head>
        <body>
            <header class="site-header">
                <h1>TechBlog Daily</h1>
                <nav>
                    <ul>
                        <li><a href="/reviews">Reviews</a></li>
                        <li><a href="/news">News</a></li>
                        <li><a href="/tutorials">Tutorials</a></li>
                    </ul>
                </nav>
            </header>
            
            <main class="content-area">
                <!-- Product Review Section -->
                <article class="blog-post product-review">
                    <h2>iPhone 15 Pro Review: A Photographer's Dream</h2>
                    <div class="post-meta">
                        <span class="author">by Tech Reviewer Mike</span>
                        <time datetime="2023-12-01">December 1, 2023</time>
                        <span class="category">Product Review</span>
                    </div>
                    
                    <div class="review-summary">
                        <div class="rating">
                            <span class="score">4.5/5 stars</span>
                            <div class="pros-cons">
                                <div class="pros">
                                    <h3>Pros:</h3>
                                    <ul>
                                        <li>Exceptional camera quality</li>
                                        <li>Improved battery life</li>
                                        <li>Premium build quality</li>
                                    </ul>
                                </div>
                                <div class="cons">
                                    <h3>Cons:</h3>
                                    <ul>
                                        <li>High price point</li>
                                        <li>Limited storage options</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="review-content">
                        <p>The iPhone 15 Pro represents Apple's most significant camera upgrade in years. With its new titanium design and advanced computational photography, it's clearly targeting professional photographers and content creators.</p>
                        
                        <h3>Camera Performance</h3>
                        <p>The new 48MP main sensor delivers stunning detail and improved low-light performance. The Action Button replaces the traditional mute switch, providing quick access to camera functions.</p>
                        
                        <div class="tech-specs">
                            <h3>Key Specifications</h3>
                            <table>
                                <tr><td>Display</td><td>6.1" Super Retina XDR</td></tr>
                                <tr><td>Processor</td><td>A17 Pro chip</td></tr>
                                <tr><td>Camera</td><td>48MP main, 12MP ultra-wide, 12MP telephoto</td></tr>
                                <tr><td>Storage</td><td>128GB, 256GB, 512GB, 1TB</td></tr>
                                <tr><td>Price</td><td>Starting at $999</td></tr>
                            </table>
                        </div>
                    </div>
                </article>
                
                <!-- News Update Section -->
                <article class="blog-post news-update">
                    <h2>Microsoft Announces New AI Features for Office 365</h2>
                    <div class="post-meta">
                        <span class="author">by News Reporter Sarah</span>
                        <time datetime="2023-11-30">November 30, 2023</time>
                        <span class="category">Tech News</span>
                    </div>
                    
                    <div class="news-content">
                        <p class="lead">Microsoft unveiled a suite of AI-powered features for Office 365, bringing advanced automation and intelligent assistance to Word, Excel, and PowerPoint.</p>
                        
                        <h3>New Features Include:</h3>
                        <ul>
                            <li><strong>Smart Writing Assistant:</strong> AI-powered grammar and style suggestions</li>
                            <li><strong>Data Insights:</strong> Automatic chart and trend analysis in Excel</li>
                            <li><strong>Presentation Builder:</strong> AI-generated slide layouts and content</li>
                            <li><strong>Meeting Summarizer:</strong> Automatic transcription and action item extraction</li>
                        </ul>
                        
                        <blockquote>
                            "These AI features will transform how people work with documents and data," said Microsoft CEO Satya Nadella during the announcement.
                        </blockquote>
                        
                        <p>The features will be rolled out to Office 365 subscribers starting in January 2024, with enterprise customers getting early access in December.</p>
                    </div>
                </article>
                
                <!-- Tutorial Section -->
                <article class="blog-post tutorial">
                    <h2>How to Build a Simple Web Scraper with Python</h2>
                    <div class="post-meta">
                        <span class="author">by Tutorial Writer Alex</span>
                        <time datetime="2023-11-29">November 29, 2023</time>
                        <span class="category">Tutorial</span>
                    </div>
                    
                    <div class="tutorial-content">
                        <p>Learn how to create a basic web scraper using Python and BeautifulSoup. This tutorial covers the fundamentals of web scraping for beginners.</p>
                        
                        <h3>Prerequisites</h3>
                        <ul>
                            <li>Python 3.6 or higher</li>
                            <li>Basic knowledge of HTML</li>
                            <li>Familiarity with Python syntax</li>
                        </ul>
                        
                        <h3>Step 1: Install Required Libraries</h3>
                        <div class="code-block">
                            <pre><code>
pip install requests beautifulsoup4 lxml
                            </code></pre>
                        </div>
                        
                        <h3>Step 2: Basic Scraper Code</h3>
                        <div class="code-block">
                            <pre><code>
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract title
    title = soup.find('title').text
    
    # Extract all paragraphs
    paragraphs = [p.text for p in soup.find_all('p')]
    
    return {
        'title': title,
        'content': paragraphs
    }

# Example usage
result = scrape_website('https://example.com')
print(result)
                            </code></pre>
                        </div>
                        
                        <h3>Best Practices</h3>
                        <ol>
                            <li>Always check robots.txt before scraping</li>
                            <li>Add delays between requests to be respectful</li>
                            <li>Handle errors gracefully</li>
                            <li>Use proper headers to identify your scraper</li>
                        </ol>
                    </div>
                </article>
            </main>
            
            <aside class="sidebar">
                <div class="popular-posts">
                    <h3>Popular This Week</h3>
                    <ul>
                        <li><a href="/post1">Top 10 Programming Languages 2023</a></li>
                        <li><a href="/post2">AI Revolution in Software Development</a></li>
                        <li><a href="/post3">Cybersecurity Trends to Watch</a></li>
                    </ul>
                </div>
                
                <div class="newsletter-signup">
                    <h3>Subscribe to Our Newsletter</h3>
                    <form onsubmit="submitNewsletter(event)">
                        <input type="email" placeholder="Enter your email">
                        <button type="submit">Subscribe</button>
                    </form>
                </div>
            </aside>
            
            <script>
                // Track different content types
                document.addEventListener('DOMContentLoaded', function() {
                    trackContentTypes();
                    initSocialSharing();
                    setupCommentSystem();
                });
                
                function submitNewsletter(event) {
                    event.preventDefault();
                    // Newsletter signup logic
                    gtag('event', 'newsletter_signup');
                }
            </script>
        </body>
        </html>
        """
        
        # Test with adaptive cleaning strategy
        cleaning_stage = HTMLCleaningStage(config={
            "strategy": "MODERATE",
            "preserve_code_blocks": True,
            "preserve_tables": True
        })
        
        request = PipelineRequest(
            source="tech_blog",
            data={"html": mixed_html},
            metadata={
                "url": "https://techblog.example.com/latest-updates",
                "extraction_strategy": "mixed_content",
                "content_types": ["product_review", "news_article", "tutorial"]
            }
        )
        
        # Clean the mixed content
        cleaned_response = await cleaning_stage.process_request(request, self.context)
        assert cleaned_response.status == ResponseStatus.SUCCESS
        
        cleaned_html = cleaned_response.data["html"]
        
        # Verify script removal
        assert 'initAnalytics(' not in cleaned_html
        assert 'trackContentTypes()' not in cleaned_html
        assert 'submitNewsletter(' not in cleaned_html
        assert 'gtag(' not in cleaned_html
        
        # Verify product review content preservation
        assert 'iPhone 15 Pro Review: A Photographer\'s Dream' in cleaned_html
        assert '4.5/5 stars' in cleaned_html
        assert 'Exceptional camera quality' in cleaned_html
        assert 'High price point' in cleaned_html
        assert '48MP main sensor' in cleaned_html
        assert 'A17 Pro chip' in cleaned_html
        assert 'Starting at $999' in cleaned_html
        
        # Verify news content preservation
        assert 'Microsoft Announces New AI Features for Office 365' in cleaned_html
        assert 'Smart Writing Assistant' in cleaned_html
        assert 'Data Insights' in cleaned_html
        assert 'Satya Nadella' in cleaned_html
        assert 'January 2024' in cleaned_html
        
        # Verify tutorial content preservation
        assert 'How to Build a Simple Web Scraper with Python' in cleaned_html
        assert 'pip install requests beautifulsoup4 lxml' in cleaned_html
        assert 'import requests' in cleaned_html
        assert 'from bs4 import BeautifulSoup' in cleaned_html
        assert 'Always check robots.txt' in cleaned_html
        
        # Verify metadata preservation
        assert 'by Tech Reviewer Mike' in cleaned_html
        assert 'by News Reporter Sarah' in cleaned_html
        assert 'by Tutorial Writer Alex' in cleaned_html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
