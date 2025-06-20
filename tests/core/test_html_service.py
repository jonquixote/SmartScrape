import pytest
from bs4 import BeautifulSoup, Comment
from core.html_service import HTMLService

class TestHTMLService:
    def setup_method(self):
        self.html_service = HTMLService()
        self.html_service.initialize()
    
    def teardown_method(self):
        self.html_service.shutdown()
    
    def test_clean_html(self):
        # Test basic cleaning
        html = '''
        <html>
            <head>
                <script>alert("test");</script>
                <style>.test{color:red;}</style>
            </head>
            <body>
                <div style="display:none;">Hidden</div>
                <div><!-- Comment -->Visible</div>
                <div hidden>Also hidden</div>
            </body>
        </html>
        '''
        cleaned = self.html_service.clean_html(html)
    
        # Script, style, hidden div, and comments should be removed
        assert 'script' not in cleaned
        assert 'style' not in cleaned
        assert 'Hidden' not in cleaned
        assert 'Comment' not in cleaned
        assert 'Visible' in cleaned
        assert 'Also hidden' not in cleaned
    
    def test_clean_html_options(self):
        # Test with different options
        html = '''
        <html>
            <head>
                <script>alert("test");</script>
                <style>.test{color:red;}</style>
            </head>
            <body>
                <div style="color: blue;">Styled</div>
                <div><!-- Comment -->Content</div>
            </body>
        </html>
        '''
        
        # Keep JavaScript
        js_result = self.html_service.clean_html(html, remove_js=False, remove_css=True)
        assert 'script' in js_result
        assert 'style' not in js_result
        
        # Keep CSS
        css_result = self.html_service.clean_html(html, remove_js=True, remove_css=False)
        assert 'script' not in css_result
        assert 'style' in css_result
        assert 'color: blue' in css_result
        
        # Keep comments
        comment_result = self.html_service.clean_html(html, remove_comments=False)
        assert 'Comment' in comment_result
    
    def test_clean_html_empty_input(self):
        # Test with empty input
        assert self.html_service.clean_html("") == ""
        assert self.html_service.clean_html(None) == ""
    
    def test_clean_html_malformed(self):
        # Test with malformed HTML
        malformed = "<div>Unclosed div <p>Paragraph</div>"
        cleaned = self.html_service.clean_html(malformed)
        assert "Unclosed div" in cleaned
        assert "Paragraph" in cleaned
    
    def test_generate_selector(self):
        html = '<html><body><div id="main"><ul class="list"><li class="item">Item 1</li><li class="item">Item 2</li></ul></div></body></html>'
        soup = BeautifulSoup(html, 'lxml')
        
        # Test optimized selector with id
        div = soup.find('div')
        selector = self.html_service.generate_selector(div)
        assert selector == '#main'
        
        # Test selector with class
        ul = soup.find('ul')
        selector = self.html_service.generate_selector(ul)
        assert selector == 'ul.list'
        
        # Test selector for list item
        li = soup.find_all('li')[1]
        selector = self.html_service.generate_selector(li, optimized=False)
        assert 'li' in selector
        assert 'item' in selector
    
    def test_generate_selector_edge_cases(self):
        # Test with different element types and options
        html = '<div><span>Test</span><p><a href="#">Link</a></p></div>'
        soup = BeautifulSoup(html, 'lxml')
        
        # Test XPath selector
        a_tag = soup.find('a')
        xpath = self.html_service.generate_selector(a_tag, method='xpath')
        assert '//' in xpath
        assert 'a' in xpath
        
        # Test with string input
        string_selector = self.html_service.generate_selector('div > p > a', html=html)
        assert 'a' in string_selector
        
        # Test with invalid input
        assert self.html_service.generate_selector(None) == ""
        assert self.html_service.generate_selector("invalid", html="<div>Test</div>") == ""
    
    def test_compare_elements(self):
        html1 = '<div class="item"><span>Test</span></div>'
        html2 = '<div class="item"><span>Different</span></div>'
        html3 = '<div class="different"><span>Test</span></div>'
        html4 = '<p class="item"><span>Test</span></p>'
        
        soup1 = BeautifulSoup(html1, 'lxml')
        soup2 = BeautifulSoup(html2, 'lxml')
        soup3 = BeautifulSoup(html3, 'lxml')
        soup4 = BeautifulSoup(html4, 'lxml')
        
        # Same structure, different text - should be similar
        similarity = self.html_service.compare_elements(soup1.div, soup2.div)
        assert similarity > 0.8
        
        # Same tag, different class - should be moderately similar
        similarity = self.html_service.compare_elements(soup1.div, soup3.div)
        assert 0.3 < similarity < 0.9
        
        # Different tag - should have low similarity
        similarity = self.html_service.compare_elements(soup1.div, soup4.p)
        assert similarity == 0.0
    
    def test_extract_main_content(self):
        html = '''
        <html>
            <header>Header</header>
            <nav>Navigation</nav>
            <div id="content">
                <h1>Main Content</h1>
                <p>This is the main content.</p>
            </div>
            <footer>Footer</footer>
        </html>
        '''
        
        main_content = self.html_service.extract_main_content(html)
        assert 'Header' not in main_content
        assert 'Navigation' not in main_content
        assert 'Footer' not in main_content
        assert 'Main Content' in main_content
        assert 'This is the main content' in main_content
    
    def test_extract_main_content_with_article(self):
        html = '''
        <html>
            <header>Header</header>
            <article>
                <h1>Article Title</h1>
                <p>Article content.</p>
            </article>
            <footer>Footer</footer>
        </html>
        '''
        
        main_content = self.html_service.extract_main_content(html)
        assert 'Header' not in main_content
        assert 'Footer' not in main_content
        assert 'Article Title' in main_content
        assert 'Article content' in main_content
    
    def test_extract_main_content_empty(self):
        # Test with empty input
        assert self.html_service.extract_main_content("") == ""
        assert self.html_service.extract_main_content(None) == ""
    
    def test_extract_tables(self):
        html = '''
        <table>
            <caption>Test Table</caption>
            <thead>
                <tr>
                    <th>Header 1</th>
                    <th>Header 2</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Row 1, Cell 1</td>
                    <td>Row 1, Cell 2</td>
                </tr>
                <tr>
                    <td>Row 2, Cell 1</td>
                    <td>Row 2, Cell 2</td>
                </tr>
            </tbody>
        </table>
        '''
        
        tables = self.html_service.extract_tables(html)
        assert len(tables) == 1
        assert tables[0]['caption'] == 'Test Table'
        assert tables[0]['headers'] == ['Header 1', 'Header 2']
        assert len(tables[0]['rows']) == 2
        assert tables[0]['rows'][0] == ['Row 1, Cell 1', 'Row 1, Cell 2']
        assert tables[0]['data'][1]['Header 2'] == 'Row 2, Cell 2'
    
    def test_extract_tables_without_thead(self):
        html = '''
        <table>
            <tr>
                <th>Header A</th>
                <th>Header B</th>
            </tr>
            <tr>
                <td>Data A</td>
                <td>Data B</td>
            </tr>
        </table>
        '''
        
        tables = self.html_service.extract_tables(html)
        assert len(tables) == 1
        assert tables[0]['headers'] == ['Header A', 'Header B']
        assert tables[0]['rows'][0] == ['Data A', 'Data B']
    
    def test_extract_tables_multiple(self):
        html = '''
        <div>
            <table id="table1">
                <tr><th>T1 Header</th></tr>
                <tr><td>T1 Data</td></tr>
            </table>
            <table id="table2">
                <tr><th>T2 Header</th></tr>
                <tr><td>T2 Data</td></tr>
            </table>
        </div>
        '''
        
        tables = self.html_service.extract_tables(html)
        assert len(tables) == 2
        assert tables[0]['headers'] == ['T1 Header']
        assert tables[1]['headers'] == ['T2 Header']
    
    def test_extract_tables_empty(self):
        # Test with empty input
        assert self.html_service.extract_tables("") == []
        assert self.html_service.extract_tables(None) == []
        
        # Test with no tables
        html = '<div>No tables here</div>'
        assert self.html_service.extract_tables(html) == []
    
    def test_extract_links(self):
        html = '''
        <div>
            <a href="https://example.com" title="Example">External Link</a>
            <a href="/relative" rel="nofollow">Relative Link</a>
            <a href="mailto:test@example.com">Email Link</a>
        </div>
        '''
        
        links = self.html_service.extract_links(html)
        assert len(links) == 3
        
        # Check external link
        assert links[0]['url'] == 'https://example.com'
        assert links[0]['text'] == 'External Link'
        assert links[0]['title'] == 'Example'
        assert links[0]['is_internal'] == False
        
        # Check relative link
        assert links[1]['url'] == '/relative'
        assert links[1]['text'] == 'Relative Link'
        assert 'nofollow' in links[1]['rel']
        assert links[1]['is_internal'] == True
        
        # Check email link
        assert links[2]['url'] == 'mailto:test@example.com'
        assert links[2]['text'] == 'Email Link'
        assert links[2]['is_internal'] == False
    
    def test_extract_links_with_base_url(self):
        html = '''
        <div>
            <a href="/page1">Page 1</a>
            <a href="page2">Page 2</a>
            <a href="../page3">Page 3</a>
        </div>
        '''
        
        links = self.html_service.extract_links(html, base_url='https://test.com/subdir/')
        
        # Check absolute URL resolution
        assert links[0]['url'] == 'https://test.com/page1'
        assert links[1]['url'] == 'https://test.com/subdir/page2'
        assert links[2]['url'] == 'https://test.com/page3'
        
        # All should be marked as internal
        assert all(link['is_internal'] for link in links)
    
    def test_extract_links_edge_cases(self):
        # Test with empty input
        assert self.html_service.extract_links("") == []
        assert self.html_service.extract_links(None) == []
        
        # Test with no links
        html = '<div>No links here</div>'
        assert self.html_service.extract_links(html) == []
        
        # Test with empty href
        html = '<a href="">Empty Link</a>'
        links = self.html_service.extract_links(html)
        assert len(links) == 1
        assert links[0]['url'] == ''