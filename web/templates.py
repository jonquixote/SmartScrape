def get_frontend_html():
    """Return the HTML for the HAL 9000 frontend page with API configuration"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartScrape - HAL 9000 Interface</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Orbitron', monospace;
            background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #0c0c0c 100%);
            color: #00ff00;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow-x: hidden;
        }
        
        .container {
            text-align: center;
            max-width: 900px;
            padding: 40px;
            width: 100%;
        }
        
        .title {
            font-size: 3.5rem;
            font-weight: 900;
            letter-spacing: 0.2em;
            margin-bottom: 20px;
            text-shadow: 0 0 20px #00ff00, 0 0 40px #00ff00;
            animation: pulse 2s infinite ease-in-out;
        }
        
        .subtitle {
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 60px;
            opacity: 0.8;
            letter-spacing: 0.1em;
        }
        
        .query-interface {
            display: none;
            margin-top: 40px;
            background: rgba(0, 0, 0, 0.9);
            border: 2px solid #00ff00;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
            animation: slideDown 0.5s ease-out;
        }
        
        .query-section {
            margin-bottom: 25px;
            text-align: left;
        }
        
        .query-textarea {
            width: 100%;
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            color: #00ff00;
            padding: 15px;
            font-family: 'Orbitron', monospace;
            font-size: 1rem;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
            transition: all 0.3s ease;
            min-height: 120px;
            resize: vertical;
        }
        
        .query-textarea:focus {
            outline: none;
            box-shadow: 0 0 25px rgba(0, 255, 0, 0.6);
            border-color: #00ffff;
        }
        
        .hal-button {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 0 auto 40px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #ff0000, #cc0000, #800000);
            border: 4px solid #ff0000;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 
                0 0 50px rgba(255, 0, 0, 0.5),
                inset 0 0 50px rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 10px #ffffff;
            animation: breathe 3s infinite ease-in-out;
        }
        
        .hal-button:hover {
            transform: scale(1.1);
            box-shadow: 
                0 0 80px rgba(255, 0, 0, 0.8),
                inset 0 0 50px rgba(255, 255, 255, 0.2);
        }
        
        .hal-eye {
            position: absolute;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #ffffff, #cccccc, #666666);
            border: 2px solid #333;
            animation: blink 4s infinite;
        }
        
        .status-text {
            font-size: 1.1rem;
            margin-top: 30px;
            color: #00ff00;
            opacity: 0.7;
        }
        
        .config-panel {
            display: none;
            margin-top: 40px;
            background: rgba(0, 0, 0, 0.9);
            border: 2px solid #00ff00;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
            animation: slideDown 0.5s ease-out;
        }
        
        .config-section {
            margin-bottom: 25px;
            text-align: left;
        }
        
        .config-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 700;
            color: #00ff00;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
        }
        
        .config-input, .config-select {
            width: 100%;
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            color: #00ff00;
            padding: 12px 15px;
            font-family: 'Orbitron', monospace;
            font-size: 0.9rem;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
            transition: all 0.3s ease;
        }
        
        .config-input:focus, .config-select:focus {
            outline: none;
            box-shadow: 0 0 25px rgba(0, 255, 0, 0.6);
            border-color: #00ffff;
        }
        
        .config-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .url-section {
            grid-column: 1 / -1;
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }
        
        .btn {
            padding: 15px 30px;
            font-family: 'Orbitron', monospace;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .btn-analyze {
            background: linear-gradient(45deg, #ff0000, #cc0000);
            border-color: #ff0000;
            color: #ffffff;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
        }
        
        .btn-analyze:hover {
            background: linear-gradient(45deg, #cc0000, #990000);
            box-shadow: 0 0 30px rgba(255, 0, 0, 0.6);
            transform: translateY(-2px);
        }
        
        .btn-reset {
            background: transparent;
            border-color: #00ff00;
            color: #00ff00;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }
        
        .btn-reset:hover {
            background: rgba(0, 255, 0, 0.1);
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.6);
            transform: translateY(-2px);
        }
        
        .results-container {
            margin-top: 40px;
            background: rgba(0, 0, 0, 0.9);
            border: 2px solid #00ff00;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
            display: none;
            text-align: left;
        }
        
        .result-header {
            color: #00ffff;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .result-item {
            margin-bottom: 15px;
            padding: 15px;
            background: rgba(0, 255, 0, 0.05);
            border: 1px solid rgba(0, 255, 0, 0.3);
            border-radius: 8px;
        }
        
        .result-label {
            font-weight: 700;
            color: #00ffff;
            margin-right: 10px;
        }
        
        .grid-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                linear-gradient(rgba(0, 255, 0, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 0, 0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
            animation: scroll 20s linear infinite;
        }
        
        .loading-indicator {
            display: none;
            text-align: center;
            padding: 20px;
            color: #00ffff;
        }
        
        .spinner {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid rgba(0, 255, 255, 0.3);
            border-top: 3px solid #00ffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        @keyframes breathe {
            0%, 100% { 
                box-shadow: 
                    0 0 50px rgba(255, 0, 0, 0.5),
                    inset 0 0 50px rgba(255, 255, 255, 0.1);
            }
            50% { 
                box-shadow: 
                    0 0 80px rgba(255, 0, 0, 0.8),
                    inset 0 0 50px rgba(255, 255, 255, 0.2);
            }
        }
        
        @keyframes blink {
            0%, 90%, 100% { opacity: 1; }
            95% { opacity: 0.3; }
        }
        
        @keyframes scroll {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }
        
        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .title {
                font-size: 2.5rem;
            }
            
            .config-grid {
                grid-template-columns: 1fr;
            }
            
            .action-buttons {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="grid-bg"></div>
    
    <div class="container">
        <h1 class="title">SMARTSCRAPE</h1>
        <p class="subtitle">HAL 9000 INTELLIGENCE INTERFACE</p>
        
        <div class="hal-button" onclick="activateHAL()">
            <div class="hal-eye"></div>
            <span style="position: absolute; bottom: 20px; font-size: 0.9rem;">ACTIVATE</span>
        </div>
        
        <div class="status-text" id="statusText">
            SYSTEM READY - CLICK TO INITIALIZE
        </div>
        
        <div class="config-panel" id="configPanel">
            <div class="config-grid">
                <div class="config-section">
                    <label class="config-label">AI PROVIDER</label>
                    <select class="config-select" id="providerSelect">
                        <option value="openai">OpenAI</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="google">Google Gemini</option>
                        <option value="ollama">Ollama (Local)</option>
                    </select>
                </div>
                
                <div class="config-section">
                    <label class="config-label">MODEL</label>
                    <select class="config-select" id="modelSelect">
                        <option value="gpt-4o">GPT-4o</option>
                        <option value="gpt-4">GPT-4</option>
                        <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                        <option value="gemini-pro">Gemini Pro</option>
                    </select>
                </div>
            </div>
            
            <div class="config-section">
                <label class="config-label">API KEY</label>
                <input type="password" class="config-input" id="apiKeyInput" placeholder="Enter your API key..." />
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-analyze" onclick="validateAPI()">
                    ✓ VALIDATE API KEY
                </button>
                <button class="btn btn-reset" onclick="resetHAL()">
                    🔄 RESET SYSTEM
                </button>
            </div>
        </div>
        
        <div class="query-interface" id="queryInterface">
            <div class="query-section">
                <label class="config-label">SCRAPING QUERY / INTENT</label>
                <textarea class="query-textarea" id="queryInput" placeholder="Enter your scraping query or intent here...

Examples:
• Extract all product information from e-commerce pages
• Find contact information and company details 
• Scrape job listings with salary and requirements
• Get article content and metadata from news sites"></textarea>
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-analyze" onclick="executeQuery()">
                    � EXECUTE QUERY
                </button>
                <button class="btn btn-reset" onclick="showConfig()">
                    ⚙️ CHANGE CONFIG
                </button>
            </div>
        </div>
        
        <div class="results-container" id="resultsContainer">
            <div class="result-header">API VALIDATION / QUERY RESULTS</div>
            <div class="loading-indicator" id="loadingIndicator">
                <div class="spinner"></div>
                HAL 9000 IS ANALYZING TARGET...
            </div>
            <div id="results"></div>
        </div>
    </div>
    
    <script>
        console.log('SmartScrape HAL 9000 Interface loading...');
        
        let halActivated = false;
        
        // Check if all required elements exist
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM loaded, checking elements...');
            const statusText = document.getElementById('statusText');
            const configPanel = document.getElementById('configPanel');
            const halButton = document.querySelector('.hal-button');
            
            console.log('statusText:', statusText);
            console.log('configPanel:', configPanel);
            console.log('halButton:', halButton);
            
            if (!statusText || !configPanel || !halButton) {
                console.error('Missing required elements!');
            } else {
                console.log('All elements found, interface ready');
            }
        });
        
        function activateHAL() {
            console.log('activateHAL() called');
            if (!halActivated) {
                console.log('Activating HAL...');
                halActivated = true;
                document.getElementById('statusText').textContent = 'HAL 9000 ONLINE - CONFIGURE PARAMETERS';
                document.getElementById('configPanel').style.display = 'block';
                
                // Update HAL button
                const button = document.querySelector('.hal-button');
                button.innerHTML = '<div class="hal-eye"></div><span style="position: absolute; bottom: 20px; font-size: 0.9rem;">ONLINE</span>';
                button.style.background = 'radial-gradient(circle at 30% 30%, #00ff00, #00cc00, #008000)';
                button.style.borderColor = '#00ff00';
                button.onclick = null; // Disable further clicks
                
                // Play activation sound effect
                try {
                    playActivationSound();
                } catch (e) {
                    console.log('Sound error:', e);
                }
                console.log('HAL activated successfully');
            } else {
                console.log('HAL already activated');
            }
        }
        
        function resetHAL() {
            halActivated = false;
            document.getElementById('statusText').textContent = 'SYSTEM READY - CLICK TO INITIALIZE';
            document.getElementById('configPanel').style.display = 'none';
            document.getElementById('queryInterface').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'none';
            
            // Reset form
            document.getElementById('providerSelect').value = 'openai';
            document.getElementById('modelSelect').value = 'gpt-4o';
            document.getElementById('apiKeyInput').value = '';
            
            // Reset HAL button
            const button = document.querySelector('.hal-button');
            button.innerHTML = '<div class="hal-eye"></div><span style="position: absolute; bottom: 20px; font-size: 0.9rem;">ACTIVATE</span>';
            button.style.background = 'radial-gradient(circle at 30% 30%, #ff0000, #cc0000, #800000)';
            button.style.borderColor = '#ff0000';
            button.onclick = activateHAL;
        }
        
        function showConfig() {
            document.getElementById('configPanel').style.display = 'block';
            document.getElementById('queryInterface').style.display = 'none';
            document.getElementById('statusText').textContent = 'HAL 9000 ONLINE - CONFIGURE PARAMETERS';
        }
        
        async function validateAPI() {
            const provider = document.getElementById('providerSelect').value;
            const model = document.getElementById('modelSelect').value;
            const apiKey = document.getElementById('apiKeyInput').value.trim();
            
            // Validation
            if (!apiKey) {
                alert('Please enter your API key');
                return;
            }
            
            // Show loading
            document.getElementById('statusText').textContent = 'HAL 9000 IS VALIDATING API KEY...';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('loadingIndicator').style.display = 'block';
            document.getElementById('loadingIndicator').innerHTML = '<div class="spinner"></div>VALIDATING API CREDENTIALS...';
            document.getElementById('results').innerHTML = '';
            
            try {
                // Simple API validation test
                const testResponse = await fetch('/api/ai-config/test-key', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        provider: provider,
                        model: model,
                        api_key: apiKey
                    })
                });
                
                document.getElementById('loadingIndicator').style.display = 'none';
                
                if (testResponse.ok) {
                    const validationData = await testResponse.json();
                    displayValidationSuccess(validationData);
                    
                    // Store API config for later use
                    window.apiConfig = { provider, model, apiKey };
                    
                    // Transition to query interface after short delay
                    setTimeout(() => {
                        document.getElementById('configPanel').style.display = 'none';
                        document.getElementById('queryInterface').style.display = 'block';
                        document.getElementById('resultsContainer').style.display = 'none';
                        document.getElementById('statusText').textContent = 'API VALIDATED - ENTER YOUR QUERY';
                    }, 2000);
                } else {
                    const errorData = await testResponse.json().catch(() => ({ error: 'API validation failed' }));
                    throw new Error(errorData.error || 'Invalid API key or configuration');
                }
            } catch (error) {
                document.getElementById('loadingIndicator').style.display = 'none';
                document.getElementById('results').innerHTML = `
                    <div class="result-item" style="border-color: #ff0000; background: rgba(255, 0, 0, 0.1);">
                        <span class="result-label" style="color: #ff0000;">VALIDATION FAILED:</span> ${error.message}
                    </div>
                `;
                document.getElementById('statusText').textContent = 'API VALIDATION FAILED - CHECK CREDENTIALS';
            }
        }
        
        async function pollJobStatus(jobId) {
            const maxAttempts = 60; // Poll for up to 5 minutes
            let attempts = 0;
            
            const poll = async () => {
                try {
                    attempts++;
                    const response = await fetch(`/status/${jobId}`);
                    
                    if (response.ok) {
                        const jobData = await response.json();
                        
                        // Update loading message based on status
                        const loadingMessages = {
                            'pending_intelligent_analysis': 'ANALYZING QUERY INTENT...',
                            'processing': 'SCRAPING TARGET WEBSITES...',
                            'completed': 'PROCESSING COMPLETE',
                            'failed': 'PROCESSING FAILED'
                        };
                        
                        const message = loadingMessages[jobData.status] || 'PROCESSING...';
                        document.getElementById('loadingIndicator').innerHTML = `<div class="spinner"></div>${message}`;
                        
                        if (jobData.status === 'completed') {
                            document.getElementById('loadingIndicator').style.display = 'none';
                            displayQueryResults(jobData);
                            document.getElementById('statusText').textContent = 'QUERY COMPLETED - HAL 9000 READY';
                            return;
                        } else if (jobData.status === 'failed') {
                            document.getElementById('loadingIndicator').style.display = 'none';
                            document.getElementById('results').innerHTML = `
                                <div class="result-item" style="border-color: #ff0000; background: rgba(255, 0, 0, 0.1);">
                                    <span class="result-label" style="color: #ff0000;">PROCESSING FAILED:</span> ${jobData.error || 'Unknown error'}
                                </div>
                            `;
                            document.getElementById('statusText').textContent = 'PROCESSING FAILED - PLEASE RETRY';
                            return;
                        }
                        
                        // Continue polling if still processing
                        if (attempts < maxAttempts) {
                            setTimeout(poll, 2000); // Poll every 2 seconds
                        } else {
                            throw new Error('Processing timeout - job took too long');
                        }
                    } else {
                        throw new Error('Failed to get job status');
                    }
                } catch (error) {
                    document.getElementById('loadingIndicator').style.display = 'none';
                    document.getElementById('results').innerHTML = `
                        <div class="result-item" style="border-color: #ff0000; background: rgba(255, 0, 0, 0.1);">
                            <span class="result-label" style="color: #ff0000;">STATUS CHECK FAILED:</span> ${error.message}
                        </div>
                    `;
                    document.getElementById('statusText').textContent = 'STATUS CHECK FAILED - PLEASE RETRY';
                }
            };
            
            // Start polling
            poll();
        }
        
        async function executeQuery() {
            const query = document.getElementById('queryInput').value.trim();
            
            if (!query) {
                alert('Please enter a scraping query or intent');
                return;
            }
            
            if (!window.apiConfig) {
                alert('API configuration lost. Please reconfigure.');
                showConfig();
                return;
            }
            
            // Show loading
            document.getElementById('statusText').textContent = 'HAL 9000 IS PROCESSING QUERY...';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('loadingIndicator').style.display = 'block';
            document.getElementById('loadingIndicator').innerHTML = '<div class="spinner"></div>INITIATING INTELLIGENT SCRAPE...';
            document.getElementById('results').innerHTML = '';
            
            try {
                // Start the scraping job
                const response = await fetch('/scrape-intelligent', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-KEY': 'smartscrape-api-key-2024'
                    },
                    body: JSON.stringify({
                        query: query,
                        start_url: null,
                        options: {
                            ai_provider: window.apiConfig.provider,
                            ai_model: window.apiConfig.model,
                            api_key: window.apiConfig.apiKey
                        }
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    const jobId = data.job_id;
                    
                    // Start polling for results
                    await pollJobStatus(jobId);
                } else {
                    const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                    throw new Error(errorData.error || errorData.detail || 'Query processing failed');
                }
            } catch (error) {
                document.getElementById('loadingIndicator').style.display = 'none';
                document.getElementById('results').innerHTML = `
                    <div class="result-item" style="border-color: #ff0000; background: rgba(255, 0, 0, 0.1);">
                        <span class="result-label" style="color: #ff0000;">ERROR:</span> ${error.message}
                    </div>
                `;
                document.getElementById('statusText').textContent = 'ERROR OCCURRED - PLEASE RETRY';
            }
        }
        
        function displayValidationSuccess(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="result-item" style="border-color: #00ff00; background: rgba(0, 255, 0, 0.1);">
                    <span class="result-label" style="color: #00ff00;">✓ API VALIDATION SUCCESS</span>
                </div>
                <div class="result-item">
                    <span class="result-label">PROVIDER:</span> ${data.provider || window.apiConfig?.provider || 'N/A'}
                </div>
                <div class="result-item">
                    <span class="result-label">MODEL:</span> ${data.model || window.apiConfig?.model || 'N/A'}
                </div>
                <div class="result-item">
                    <span class="result-label">STATUS:</span> Ready for queries
                </div>
                <div class="result-item">
                    <span class="result-label">TIMESTAMP:</span> ${new Date().toLocaleString()}
                </div>
            `;
        }
        
        function displayQueryResults(jobData) {
            const resultsDiv = document.getElementById('results');
            
            if (jobData.result && jobData.result.data) {
                // Display the actual scraping results
                const results = jobData.result.data;
                let resultHtml = `
                    <div class="result-item">
                        <span class="result-label">JOB ID:</span> ${jobData.job_id || 'N/A'}
                    </div>
                    <div class="result-item">
                        <span class="result-label">STATUS:</span> ${jobData.status || 'Completed'}
                    </div>
                    <div class="result-item">
                        <span class="result-label">QUERY:</span> ${document.getElementById('queryInput').value}
                    </div>
                    <div class="result-item">
                        <span class="result-label">COMPLETED:</span> ${jobData.completed_at || new Date().toLocaleString()}
                    </div>
                `;
                
                if (Array.isArray(results) && results.length > 0) {
                    resultHtml += `
                        <div class="result-item">
                            <span class="result-label">EXTRACTED DATA (${results.length} items):</span>
                        </div>
                    `;
                    
                    results.slice(0, 10).forEach((item, index) => { // Show first 10 items
                        resultHtml += `
                            <div class="result-item" style="margin-left: 20px; background: rgba(0, 255, 0, 0.05);">
                                <strong>Item ${index + 1}:</strong><br>
                                <pre style="white-space: pre-wrap; color: #00ff00; font-size: 0.9rem;">${JSON.stringify(item, null, 2)}</pre>
                            </div>
                        `;
                    });
                    
                    if (results.length > 10) {
                        resultHtml += `
                            <div class="result-item" style="margin-left: 20px; color: #00ffff;">
                                ... and ${results.length - 10} more items
                            </div>
                        `;
                    }
                } else if (jobData.result.summary) {
                    resultHtml += `
                        <div class="result-item">
                            <span class="result-label">SUMMARY:</span>
                            <pre style="white-space: pre-wrap; color: #00ff00; margin-top: 10px;">${jobData.result.summary}</pre>
                        </div>
                    `;
                } else {
                    resultHtml += `
                        <div class="result-item">
                            <span class="result-label">RESULT:</span>
                            <pre style="white-space: pre-wrap; color: #00ff00; margin-top: 10px;">${JSON.stringify(jobData.result, null, 2)}</pre>
                        </div>
                    `;
                }
                
                resultsDiv.innerHTML = resultHtml;
            } else {
                // Fallback for jobs without detailed results
                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <span class="result-label">JOB ID:</span> ${jobData.job_id || 'N/A'}
                    </div>
                    <div class="result-item">
                        <span class="result-label">STATUS:</span> ${jobData.status || 'Processing'}
                    </div>
                    <div class="result-item">
                        <span class="result-label">MESSAGE:</span> ${jobData.message || 'Query processing initiated'}
                    </div>
                    <div class="result-item">
                        <span class="result-label">QUERY:</span> ${document.getElementById('queryInput').value}
                    </div>
                    <div class="result-item">
                        <span class="result-label">TIMESTAMP:</span> ${new Date().toLocaleString()}
                    </div>
                `;
            }
        }
        
        function playActivationSound() {
            // Create a simple beep sound using Web Audio API
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.3);
                
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.3);
            } catch (e) {
                console.log('Audio not available');
            }
        }
        
        // Provider/Model mapping - Comprehensive lists
        const modelsByProvider = {
            openai: [
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
                'gpt-4',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k'
            ],
            anthropic: [
                'claude-3-5-sonnet-20241022',
                'claude-3-5-haiku-20241022',
                'claude-3-opus-20240229',
                'claude-3-sonnet-20240229',
                'claude-3-haiku-20240307',
                'claude-2.1',
                'claude-2.0'
            ],
            google: [
                'gemini-2.0-flash-exp',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-pro',
                'gemini-pro-vision'
            ],
            ollama: [
                'llama3.2',
                'llama3.1',
                'llama2',
                'codellama',
                'mistral',
                'mixtral',
                'qwen2.5',
                'phi3'
            ]
        };
        
        // Update models when provider changes
        document.getElementById('providerSelect').addEventListener('change', async function() {
            const provider = this.value;
            const modelSelect = document.getElementById('modelSelect');
            
            // Clear current options
            modelSelect.innerHTML = '<option value="">Loading models...</option>';
            
            try {
                const response = await fetch(`/api/models/${provider}`);
                if (response.ok) {
                    const data = await response.json();
                    modelSelect.innerHTML = '';
                    
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.name || model.id;
                        modelSelect.appendChild(option);
                    });
                } else {
                    // Fallback to static list if API fails
                    const models = modelsByProvider[provider] || [];
                    modelSelect.innerHTML = '';
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model.toUpperCase();
                        modelSelect.appendChild(option);
                    });
                }
            } catch (error) {
                console.error('Error fetching models:', error);
                // Fallback to static list
                const models = modelsByProvider[provider] || [];
                modelSelect.innerHTML = '';
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model.toUpperCase();
                    modelSelect.appendChild(option);
                });
            }
        });
        
        // Initialize models on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Trigger initial model population
            document.getElementById('providerSelect').dispatchEvent(new Event('change'));
        });
        
        // Enter key support
        document.getElementById('apiKeyInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                validateAPI();
            }
        });
        
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                executeQuery();
            }
        });
    </script>
</body>
</html>"""
    return html
