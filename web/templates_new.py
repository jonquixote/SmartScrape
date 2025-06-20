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
            
            <div class="config-section url-section">
                <label class="config-label">TARGET URL</label>
                <input type="text" class="config-input" id="urlInput" placeholder="https://example.com" />
            </div>
            
            <div class="action-buttons">
                <button class="btn btn-analyze" onclick="analyzeTarget()">
                    🔍 ANALYZE TARGET
                </button>
                <button class="btn btn-reset" onclick="resetHAL()">
                    🔄 RESET SYSTEM
                </button>
            </div>
        </div>
        
        <div class="results-container" id="resultsContainer">
            <div class="result-header">ANALYSIS RESULTS</div>
            <div class="loading-indicator" id="loadingIndicator">
                <div class="spinner"></div>
                HAL 9000 IS ANALYZING TARGET...
            </div>
            <div id="results"></div>
        </div>
    </div>
    
    <script>
        let halActivated = false;
        
        function activateHAL() {
            if (!halActivated) {
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
                playActivationSound();
            }
        }
        
        function resetHAL() {
            halActivated = false;
            document.getElementById('statusText').textContent = 'SYSTEM READY - CLICK TO INITIALIZE';
            document.getElementById('configPanel').style.display = 'none';
            document.getElementById('resultsContainer').style.display = 'none';
            
            // Reset form
            document.getElementById('providerSelect').value = 'openai';
            document.getElementById('modelSelect').value = 'gpt-4o';
            document.getElementById('apiKeyInput').value = '';
            document.getElementById('urlInput').value = '';
            
            // Reset HAL button
            const button = document.querySelector('.hal-button');
            button.innerHTML = '<div class="hal-eye"></div><span style="position: absolute; bottom: 20px; font-size: 0.9rem;">ACTIVATE</span>';
            button.style.background = 'radial-gradient(circle at 30% 30%, #ff0000, #cc0000, #800000)';
            button.style.borderColor = '#ff0000';
            button.onclick = activateHAL;
        }
        
        async function analyzeTarget() {
            const provider = document.getElementById('providerSelect').value;
            const model = document.getElementById('modelSelect').value;
            const apiKey = document.getElementById('apiKeyInput').value.trim();
            const url = document.getElementById('urlInput').value.trim();
            
            // Validation
            if (!apiKey) {
                alert('Please enter your API key');
                return;
            }
            
            if (!url) {
                alert('Please enter a target URL');
                return;
            }
            
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                alert('Please enter a valid URL (must start with http:// or https://)');
                return;
            }
            
            // Show loading
            document.getElementById('statusText').textContent = 'HAL 9000 IS ANALYZING TARGET...';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('loadingIndicator').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            
            try {
                const response = await fetch('/scrape', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-KEY': 'smartscrape-api-key-2024'
                    },
                    body: JSON.stringify({
                        url: url,
                        user_intent: 'Extract all relevant data from this website using HAL 9000 intelligence',
                        extraction_rules: {
                            ai_provider: provider,
                            ai_model: model,
                            api_key: apiKey
                        }
                    })
                });
                
                document.getElementById('loadingIndicator').style.display = 'none';
                
                if (response.ok) {
                    const data = await response.json();
                    displayResults(data);
                    document.getElementById('statusText').textContent = 'ANALYSIS COMPLETE - HAL 9000 READY';
                } else {
                    const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                    throw new Error(errorData.error || 'Analysis failed');
                }
            } catch (error) {
                document.getElementById('loadingIndicator').style.display = 'none';
                document.getElementById('results').innerHTML = `
                    <div class="result-item" style="border-color: #ff0000; background: rgba(255, 0, 0, 0.1);">
                        <span class="result-label" style="color: #ff0000;">ERROR:</span> ${error.message}
                    </div>
                `;
                document.getElementById('statusText').textContent = 'ERROR OCCURRED - CHECK CONFIGURATION';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="result-item">
                    <span class="result-label">JOB ID:</span> ${data.job_id || 'N/A'}
                </div>
                <div class="result-item">
                    <span class="result-label">STATUS:</span> ${data.status || 'Processing'}
                </div>
                <div class="result-item">
                    <span class="result-label">MESSAGE:</span> ${data.message || 'Analysis initiated successfully'}
                </div>
                <div class="result-item">
                    <span class="result-label">TIMESTAMP:</span> ${new Date().toLocaleString()}
                </div>
            `;
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
        
        // Provider/Model mapping
        const modelsByProvider = {
            openai: ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo'],
            anthropic: ['claude-3-sonnet', 'claude-3-haiku', 'claude-2'],
            google: ['gemini-pro', 'gemini-1.5-pro'],
            ollama: ['llama2', 'codellama', 'mistral']
        };
        
        // Update models when provider changes
        document.getElementById('providerSelect').addEventListener('change', function() {
            const provider = this.value;
            const modelSelect = document.getElementById('modelSelect');
            const models = modelsByProvider[provider] || [];
            
            modelSelect.innerHTML = '';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model.toUpperCase();
                modelSelect.appendChild(option);
            });
        });
        
        // Enter key support
        document.getElementById('urlInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeTarget();
            }
        });
        
        document.getElementById('apiKeyInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeTarget();
            }
        });
    </script>
</body>
</html>"""
    return html
