// Results page JavaScript
document.addEventListener('DOMContentLoaded', () => {
    // Get job ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const jobId = urlParams.get('id');
    const userQuery = urlParams.get('query');
    
    // Display the user query
    if (userQuery) {
        document.getElementById('user-query').textContent = userQuery;
    }
    
    // View switching functionality
    setupViewSwitching();
    
    // Export button functionality
    setupExportButtons(jobId);
    
    // If no job ID, show error
    if (!jobId) {
        showError('No job ID provided. Please try again with a valid job.');
        return;
    }
    
    // Fetch results
    fetchJobResults(jobId);
});

function setupViewSwitching() {
    const viewButtons = document.querySelectorAll('.view-btn');
    const viewContents = document.querySelectorAll('.view-content');
    
    viewButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetView = button.getAttribute('data-view');
            
            // Update active button
            viewButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show selected view, hide others
            viewContents.forEach(content => {
                if (content.id === `${targetView}-view`) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
        });
    });
}

function setupExportButtons(jobId) {
    // Set up export buttons with the job ID
    if (!jobId) return;
    
    document.getElementById('export-json').addEventListener('click', (e) => {
        e.preventDefault();
        window.open(`/api/export/${jobId}/json`, '_blank');
    });
    
    document.getElementById('export-csv').addEventListener('click', (e) => {
        e.preventDefault();
        window.open(`/api/export/${jobId}/csv`, '_blank');
    });
    
    document.getElementById('export-excel').addEventListener('click', (e) => {
        e.preventDefault();
        window.open(`/api/export/${jobId}/excel`, '_blank');
    });
}

async function fetchJobResults(jobId) {
    try {
        // First check job status to make sure it's completed
        const statusResponse = await fetch(`/api/status/${jobId}`);
        if (!statusResponse.ok) {
            throw new Error(`Failed to fetch job status: ${statusResponse.status}`);
        }
        
        const statusData = await statusResponse.json();
        
        // If job is still running, show polling UI and check again
        if (statusData.status === 'initializing' || statusData.status === 'processing') {
            setupPolling(jobId, statusData.progress);
            return;
        }
        
        // If job failed, show error
        if (statusData.status === 'error' || statusData.status === 'failed') {
            showError(`Job failed: ${statusData.error || 'Unknown error'}`);
            return;
        }
        
        // Fetch the actual results
        const resultsResponse = await fetch(`/api/results/${jobId}`);
        if (!resultsResponse.ok) {
            throw new Error(`Failed to fetch results: ${resultsResponse.status}`);
        }
        
        const resultsData = await resultsResponse.json();
        renderResults(resultsData);
        
    } catch (error) {
        showError(`Error: ${error.message}`);
    }
}

function setupPolling(jobId, initialProgress) {
    const loadingIndicator = document.getElementById('loading-indicator');
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.innerHTML = `
        <div class="progress-indicator" style="width: ${initialProgress * 100}%;"></div>
        <div class="progress-text">${Math.round(initialProgress * 100)}%</div>
    `;
    loadingIndicator.innerHTML = 'Processing your scraping request...<br>This may take a minute.';
    loadingIndicator.appendChild(progressBar);
    
    // Poll every 2 seconds
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/status/${jobId}`);
            if (!response.ok) throw new Error('Failed to get status');
            
            const data = await response.json();
            
            // Update progress
            progressBar.querySelector('.progress-indicator').style.width = `${data.progress * 100}%`;
            progressBar.querySelector('.progress-text').textContent = `${Math.round(data.progress * 100)}%`;
            
            // Check if completed
            if (data.status === 'completed') {
                clearInterval(pollInterval);
                fetchJobResults(jobId); // Re-fetch to get the results
            } 
            // Check if failed
            else if (data.status === 'error' || data.status === 'failed') {
                clearInterval(pollInterval);
                showError(`Job failed: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            clearInterval(pollInterval);
            showError(`Polling error: ${error.message}`);
        }
    }, 2000);
}

function renderResults(resultsData) {
    // Hide loading indicator
    document.getElementById('loading-indicator').style.display = 'none';
    
    // Extract data based on the structure we expect from the API
    let dataToDisplay;
    
    // Handle different result structures
    if (Array.isArray(resultsData)) {
        dataToDisplay = resultsData;
    } else if (resultsData.results && Array.isArray(resultsData.results)) {
        dataToDisplay = resultsData.results;
    } else if (resultsData.data && Array.isArray(resultsData.data)) {
        dataToDisplay = resultsData.data;
    } else {
        // If structure is unknown, use the raw data
        dataToDisplay = [resultsData];
    }
    
    // Render the text view (most user-friendly)
    renderTextView(dataToDisplay);
    
    // Render the table view
    renderTableView(dataToDisplay);
    
    // Render raw data view
    renderRawView(resultsData);
    
    // Render source list
    renderSourceList(dataToDisplay);
}

function renderTextView(data) {
    const textContent = document.getElementById('text-content');
    
    if (!data || data.length === 0) {
        textContent.innerHTML = '<p>No results found.</p>';
        return;
    }
    
    // Create structured HTML content
    let html = '';
    
    // Try to intelligently format the content based on the data structure
    data.forEach((item, index) => {
        if (typeof item === 'string') {
            // Simple string data
            html += `<p>${item}</p>`;
        } else if (item.data && typeof item.data === 'string') {
            // Common structure with string data
            html += `<p>${item.data}</p>`;
        } else if (item.data && Array.isArray(item.data)) {
            // List data
            html += '<ul>';
            item.data.forEach(listItem => {
                if (typeof listItem === 'string') {
                    html += `<li>${listItem}</li>`;
                } else if (typeof listItem === 'object') {
                    // Object in a list - create a structured view
                    html += '<li>';
                    for (const [key, value] of Object.entries(listItem)) {
                        html += `<strong>${key}:</strong> ${formatValue(value)}<br>`;
                    }
                    html += '</li>';
                }
            });
            html += '</ul>';
        } else if (item.data && typeof item.data === 'object') {
            // Object data - create a structured view
            html += '<div class="data-object">';
            for (const [key, value] of Object.entries(item.data)) {
                html += `<div><strong>${key}:</strong> ${formatValue(value)}</div>`;
            }
            html += '</div>';
        } else if (typeof item === 'object') {
            // Plain object - create a structured view
            html += '<div class="data-object">';
            for (const [key, value] of Object.entries(item)) {
                if (key !== 'source_url' && key !== 'score' && key !== 'depth') {
                    html += `<div><strong>${key}:</strong> ${formatValue(value)}</div>`;
                }
            }
            html += '</div>';
        }
        
        // Add separator between items
        if (index < data.length - 1) {
            html += '<hr>';
        }
    });
    
    textContent.innerHTML = html;
}

function renderTableView(data) {
    const tableHeaders = document.getElementById('table-headers');
    const tableBody = document.getElementById('table-body');
    
    if (!data || data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="3">No results found.</td></tr>';
        return;
    }
    
    // Extract all possible headers from the data
    const headers = new Set();
    data.forEach(item => {
        // Add standard columns
        headers.add('source_url');
        
        // Add data fields
        if (item.data && typeof item.data === 'object' && !Array.isArray(item.data)) {
            Object.keys(item.data).forEach(key => headers.add(key));
        } else {
            headers.add('data');
        }
        
        // Add any top-level fields
        Object.keys(item).forEach(key => {
            if (key !== 'data' && key !== 'source_url') {
                headers.add(key);
            }
        });
    });
    
    // Create table header
    tableHeaders.innerHTML = '';
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        tableHeaders.appendChild(th);
    });
    
    // Create table rows
    tableBody.innerHTML = '';
    data.forEach(item => {
        const row = document.createElement('tr');
        
        headers.forEach(header => {
            const cell = document.createElement('td');
            
            if (header === 'source_url') {
                cell.innerHTML = item.source_url ? `<a href="${item.source_url}" target="_blank">${item.source_url}</a>` : '';
            } else if (header === 'data' && item.data) {
                cell.textContent = formatValue(item.data);
            } else if (header !== 'data' && item[header]) {
                cell.textContent = formatValue(item[header]);
            } else if (item.data && typeof item.data === 'object' && item.data[header]) {
                cell.textContent = formatValue(item.data[header]);
            } else {
                cell.textContent = '';
            }
            
            row.appendChild(cell);
        });
        
        tableBody.appendChild(row);
    });
}

function renderRawView(data) {
    const rawContent = document.getElementById('raw-content');
    rawContent.textContent = JSON.stringify(data, null, 2);
}

function renderSourceList(data) {
    const sourceList = document.getElementById('source-list');
    const sources = new Set();
    
    // Extract unique sources
    data.forEach(item => {
        if (item.source_url) {
            sources.add(item.source_url);
        }
    });
    
    // Render sources
    if (sources.size === 0) {
        sourceList.innerHTML = '<li>No sources available</li>';
        return;
    }
    
    sourceList.innerHTML = '';
    sources.forEach(url => {
        const li = document.createElement('li');
        li.innerHTML = `<a href="${url}" target="_blank">${url}</a>`;
        sourceList.appendChild(li);
    });
}

function formatValue(value) {
    if (value === null || value === undefined) {
        return '';
    }
    if (typeof value === 'object') {
        return JSON.stringify(value);
    }
    return String(value);
}

function showError(message) {
    const loadingIndicator = document.getElementById('loading-indicator');
    loadingIndicator.style.display = 'none';
    
    const errorElement = document.getElementById('error-message');
    errorElement.style.display = 'block';
    errorElement.textContent = message;
}
