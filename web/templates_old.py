def get_frontend_html():
    """Return the HTML for the main frontend page"""
    html = """
<!DOCTYPE html>
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
            overflow: hidden;
        }
        
        .container {
            text-align: center;
            max-width: 800px;
            padding: 40px;
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
        
        .hal-button:active {
            transform: scale(0.95);
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
        
        .input-container {
            margin-top: 40px;
            opacity: 0;
            animation: fadeIn 1s ease-in-out 2s forwards;
        }
        
        .input-group {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-top: 20px;
        }
        
        .hal-input {
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            color: #00ff00;
            padding: 15px 20px;
            font-family: 'Orbitron', monospace;
            font-size: 1rem;
            border-radius: 5px;
            min-width: 300px;
            box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
        }
        
        .hal-input:focus {
            outline: none;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.6);
        }
        
        .analyze-btn {
            background: linear-gradient(45deg, #ff0000, #cc0000);
            border: 2px solid #ff0000;
            color: #ffffff;
            padding: 15px 30px;
            font-family: 'Orbitron', monospace;
            font-size: 1rem;
            font-weight: 700;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
        }
        
        .analyze-btn:hover {
            background: linear-gradient(45deg, #cc0000, #990000);
            box-shadow: 0 0 30px rgba(255, 0, 0, 0.6);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .results-container {
            margin-top: 40px;
            background: rgba(0, 0, 0, 0.8);
            border: 2px solid #00ff00;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
            display: none;
        }
        
        .result-item {
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(0, 255, 0, 0.1);
            border-radius: 5px;
            text-align: left;
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
        
        <div class="input-container" id="inputContainer">
            <div class="input-group">
                <input type="text" class="hal-input" id="urlInput" placeholder="ENTER TARGET URL FOR ANALYSIS..." />
                <button class="analyze-btn" onclick="analyzeTarget()">ANALYZE</button>
            </div>
        </div>
        
        <div class="results-container" id="resultsContainer">
            <div id="results"></div>
        </div>
    </div>
    
    <script>
        let halActivated = false;
        
        function activateHAL() {
            if (!halActivated) {
                halActivated = true;
                document.getElementById('statusText').textContent = 'HAL 9000 ONLINE - READY FOR COMMANDS';
                document.getElementById('inputContainer').style.display = 'block';
                
                // Add some dramatic effect
                const button = document.querySelector('.hal-button');
                button.style.animation = 'breathe 1s infinite ease-in-out';
                
                // Play activation sound effect (if available)
                playSound('activation');
            }
        }
        
        async function analyzeTarget() {
            const url = document.getElementById('urlInput').value.trim();
            if (!url) {
                alert('Please enter a target URL');
                return;
            }
            
            document.getElementById('statusText').textContent = 'ANALYZING TARGET... PLEASE WAIT';
            document.getElementById('resultsContainer').style.display = 'block';
            document.getElementById('results').innerHTML = '<div style="text-align: center; padding: 20px;">🔍 SCANNING IN PROGRESS...</div>';
            
            try {
                const response = await fetch('/scrape', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-KEY': 'your-api-key-here'  // You might want to make this configurable
                    },
                    body: JSON.stringify({
                        url: url,
                        user_intent: 'Extract all relevant data from this website',
                        extraction_rules: {}
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    displayResults(data);
                    document.getElementById('statusText').textContent = 'ANALYSIS COMPLETE';
                } else {
                    throw new Error('Analysis failed');
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `<div style="color: #ff0000; text-align: center; padding: 20px;">ERROR: ${error.message}</div>`;
                document.getElementById('statusText').textContent = 'ERROR OCCURRED - PLEASE RETRY';
            }
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="result-item">
                    <strong>JOB ID:</strong> ${data.job_id || 'N/A'}
                </div>
                <div class="result-item">
                    <strong>STATUS:</strong> ${data.status || 'Processing'}
                </div>
                <div class="result-item">
                    <strong>MESSAGE:</strong> ${data.message || 'Analysis initiated successfully'}
                </div>
            `;
        }
        
        function playSound(type) {
            // Placeholder for sound effects
            console.log(`Playing ${type} sound`);
        }
        
        // Enter key support
        document.getElementById('urlInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeTarget();
            }
        });
    </script>
</body>
</html>
    """
    return html