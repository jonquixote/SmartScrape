document.addEventListener('DOMContentLoaded', () => {
    // AI Configuration State
    let selectedProvider = null; // This might need to be context-specific (modal vs popover)
    let configuredProviders = {}; // Stores { providerId: { configured: true, apiKey: '...', model: '...' } }
    let availableProviders = {}; // From API: { providerId: { name: '..', models: [], configured: false } }
    let modelDiscoveryCache = {}; // Cache for dynamically discovered models
    
    const userInput = document.getElementById('user-query-input');
    const scrapeButton = document.getElementById('scrape-button');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessage = document.getElementById('error-message');

    // New Homepage Controls Elements
    const settingsButton = document.getElementById('settings-button');
    const enhancedDashboardModal = document.getElementById('enhanced-dashboard-modal');
    const dashboardModalCloseBtn = document.getElementById('dashboard-modal-close-btn');
    const aiStatusLightButton = document.getElementById('ai-status-light-button');
    const halLight = document.querySelector('.hal-light'); // More specific if multiple: document.getElementById('ai-status-light-button').querySelector('.hal-light');
    const aiStatusText = document.getElementById('ai-status-text');
    const aiStatusPopover = document.getElementById('ai-status-popover');
    // Assuming a close button for popover, if not, rely on outside click
    const popoverCloseBtn = document.getElementById('popover-close-btn');

    // Dashboard Tabs
    const featureControlsTabLink = document.getElementById('feature-controls-tab-link');
    const aiSettingsTabLink = document.getElementById('ai-settings-tab-link');
    const featureControlsTabContent = document.getElementById('feature-controls-tab');
    const aiSettingsTabContent = document.getElementById('ai-settings-tab'); // Content area for AI settings

    // AI Configuration Elements (now inside #ai-settings-tab of enhancedDashboardModal)
    // These keep their original IDs for easier refactoring of existing functions
    const providerCards = document.querySelectorAll('#ai-settings-tab .provider-card'); // Scoped to new modal
    const apiKeySection = document.getElementById('api-key-section'); // Assumed to be in #ai-settings-tab
    const apiKeyInput = document.getElementById('api-key-input');
    const modelSelect = document.getElementById('model-select');
    const testKeyBtn = document.getElementById('test-key-btn');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const showHideKeyBtn = document.getElementById('show-hide-key');
    const testResult = document.getElementById('test-result');

    // AI Configuration Elements for Popover (New IDs)
    const popoverProviderSelect = document.getElementById('popover-provider-select');
    const popoverApiKeyInput = document.getElementById('popover-api-key-input');
    const popoverModelSelect = document.getElementById('popover-model-select');
    const popoverSaveTestBtn = document.getElementById('popover-save-test-btn');
    const popoverShowHideKeyBtn = document.getElementById('popover-show-hide-key');
    const popoverTestResult = document.getElementById('popover-test-result');
    
    // API Key Required Modal (existing)
    const apiKeyRequiredModal = document.getElementById('api-key-required-modal');
    const requiredModalCloseBtn = document.getElementById('required-modal-close-btn');
    const configureNowBtn = document.getElementById('configure-now-btn');
    const continueWithoutBtn = document.getElementById('continue-without-btn');

    // --- Helper Functions ---
    function showGenericModal(modalElement) {
        if (modalElement) {
            modalElement.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }

    function hideGenericModal(modalElement) {
        if (modalElement) {
            modalElement.style.display = 'none';
            document.body.style.overflow = '';
        }
    }

    function updateAIStatusDisplay() {
        console.log('🔄 updateAIStatusDisplay called');
        console.log('availableProviders:', availableProviders);
        console.log('halLight element:', halLight);
        console.log('aiStatusText element:', aiStatusText);
        
        const isConfigured = Object.values(availableProviders).some(p => p.configured);
        console.log('isConfigured:', isConfigured);
        
        // More sophisticated check might involve testing the primary configured key
        if (isConfigured) {
            if (halLight) {
                halLight.classList.add('active'); // Green
                console.log('✅ HAL light set to GREEN (active)');
            } else {
                console.log('❌ HAL light element not found!');
            }
            if (aiStatusText) {
                aiStatusText.textContent = 'AI Activated';
                console.log('✅ Status text set to "AI Activated"');
            } else {
                console.log('❌ AI status text element not found!');
            }
        } else {
            if (halLight) {
                halLight.classList.remove('active'); // Red
                console.log('🔴 HAL light set to RED (inactive)');
            } else {
                console.log('❌ HAL light element not found!');
            }
            if (aiStatusText) {
                aiStatusText.textContent = 'AI Inactive';
                console.log('🔴 Status text set to "AI Inactive"');
            } else {
                console.log('❌ AI status text element not found!');
            }
        }
    }

    function positionPopover() {
        if (!aiStatusPopover || !aiStatusLightButton) return;
        const rect = aiStatusLightButton.getBoundingClientRect();
        aiStatusPopover.style.left = `${rect.left + window.scrollX}px`;
        aiStatusPopover.style.top = `${rect.bottom + window.scrollY + 5}px`; // 5px offset
    }

    function showPopover(popoverElement) {
        if (popoverElement) {
            positionPopover();
            popoverElement.style.display = 'block'; // Or 'flex' if styled that way
            // Populate popover fields with current config
            populatePopoverAIFields();
        }
    }

    function hidePopover(popoverElement) {
        if (popoverElement) {
            popoverElement.style.display = 'none';
        }
    }
    
    function switchTab(targetTabContentId, activeTabLink) {
        // Hide all tab contents
        [featureControlsTabContent, aiSettingsTabContent].forEach(tab => tab.style.display = 'none');
        // Deactivate all tab links
        [featureControlsTabLink, aiSettingsTabLink].forEach(link => link.classList.remove('active'));

        // Show target tab content and activate link
        const targetTab = document.getElementById(targetTabContentId);
        if (targetTab) targetTab.style.display = 'block';
        if (activeTabLink) activeTabLink.classList.add('active');
    }

    // --- AI Configuration Functions (Adapted) ---
    async function loadAvailableProviders() {
        try {
            const response = await fetch('/api/app-config');
            if (response.ok) {
                const data = await response.json();
                
                // Convert app-config response to expected format
                availableProviders = {};
                
                // Add hardcoded provider definitions with basic info
                const providerDefinitions = {
                    'openai': { name: 'OpenAI', configured: false },
                    'google': { name: 'Google Gemini', configured: false },
                    'anthropic': { name: 'Anthropic', configured: false }
                };
                
                // Initialize all providers
                Object.keys(providerDefinitions).forEach(providerId => {
                    availableProviders[providerId] = {
                        ...providerDefinitions[providerId],
                        models: [],
                        api_key: null,
                        model: null
                    };
                });
                
                // Update with configured providers from API response
                if (data.ai_settings && data.ai_settings.settings) {
                    console.log('🔍 DETAILED DEBUG: Processing AI settings from API');
                    console.log('🔍 DETAILED DEBUG: data.ai_settings.settings:', data.ai_settings.settings);
                    
                    data.ai_settings.settings.forEach(setting => {
                        console.log('🔍 DETAILED DEBUG: Processing setting for provider:', setting.provider);
                        console.log('🔍 DETAILED DEBUG: Setting details:', setting);
                        
                        if (availableProviders[setting.provider]) {
                            // If a provider has a setting returned from API, it's configured
                            // (API only returns configured providers)
                            console.log('🔍 DETAILED DEBUG: Found provider in availableProviders, marking as configured');
                            availableProviders[setting.provider].configured = true;
                            availableProviders[setting.provider].api_key = setting.api_key;
                            availableProviders[setting.provider].model = setting.model;
                            console.log('🔍 DETAILED DEBUG: Updated provider:', availableProviders[setting.provider]);
                        } else {
                            console.log('🔍 DETAILED DEBUG: Provider not found in availableProviders');
                        }
                    });
                    
                    console.log('🔍 DETAILED DEBUG: Final availableProviders state:', availableProviders);
                } else {
                    console.log('🔍 DETAILED DEBUG: No ai_settings or settings found in API response');
                }
                
                updateProviderStatusOnCards(); // For main modal cards
                updateAIStatusDisplay();
                populateProviderSelects(); // For popover and potentially other dropdowns
            } else {
                console.error('Failed to load AI providers:', response.status);
                updateAIStatusDisplay(); // Show inactive if failed
            }
        } catch (error) {
            console.error('Error loading providers:', error);
            updateAIStatusDisplay(); // Show inactive on error
        }
    }

    function populateProviderSelects() {
        if (popoverProviderSelect) {
            popoverProviderSelect.innerHTML = '<option value="">Select Provider...</option>';
            Object.keys(availableProviders).forEach(providerId => {
                const option = document.createElement('option');
                option.value = providerId;
                option.textContent = availableProviders[providerId].name || providerId;
                popoverProviderSelect.appendChild(option);
            });
        }
        // Could also populate a similar select in the main modal if cards are replaced
    }
    
    function updateProviderStatusOnCards() { // Specifically for the cards in the main modal
        Object.keys(availableProviders).forEach(providerId => {
            const card = document.querySelector(`#ai-settings-tab .provider-card[data-provider="${providerId}"]`);
            const statusEl = document.getElementById(`${providerId}-status`); // Assumes these status elements exist near cards
            if (card && statusEl) {
                const isConfigured = availableProviders[providerId].configured;
                statusEl.textContent = isConfigured ? 'Configured' : 'Not configured';
                statusEl.className = `provider-status ${isConfigured ? 'configured' : ''}`;
                if (isConfigured) card.classList.add('configured'); else card.classList.remove('configured');
            }
        });
    }

    // `selectProvider` now needs to know which context (modal cards or popover select)
    // For now, existing `selectProvider` works for cards in modal.
    // A new one will be needed for popover's select.
    async function selectProviderOnCard(providerId) {
        providerCards.forEach(card => card.classList.remove('selected'));
        const selectedCard = document.querySelector(`#ai-settings-tab .provider-card[data-provider="${providerId}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
            selectedProvider = providerId; // This global `selectedProvider` is now for the main modal context
            
            // Load models asynchronously
            await updateModelOptionsForElement(modelSelect, providerId);
            
            if(apiKeySection) apiKeySection.style.display = 'block';
            // Load existing API key if available for this provider
            if (availableProviders[providerId] && availableProviders[providerId].api_key) {
                apiKeyInput.value = availableProviders[providerId].api_key;
            } else {
                apiKeyInput.value = '';
            }
            if (availableProviders[providerId] && availableProviders[providerId].model) {
                modelSelect.value = availableProviders[providerId].model;
            }
        }
    }

    async function updateModelOptionsForElement(selectElement, providerId) {
        if (!selectElement || !providerId) return;
        
        // Show loading state
        selectElement.innerHTML = '<option value="">Loading models...</option>';
        selectElement.disabled = true;
        
        try {
            // First check if we have cached models in availableProviders
            const provider = availableProviders[providerId];
            if (provider && provider.models && provider.models.length > 0) {
                populateModelSelect(selectElement, provider.models);
                return;
            }
            
            // Fetch models dynamically from the model discovery service
            const response = await fetch(`/api/ai/models/${providerId}`);
            if (response.ok) {
                const data = await response.json();
                
                // Handle the API response structure
                if (data.models && data.models.length > 0) {
                    // Update the cached models in availableProviders
                    if (availableProviders[providerId]) {
                        availableProviders[providerId].models = data.models;
                    }
                    populateModelSelect(selectElement, data.models);
                } else {
                    const message = data.message || 'No models available';
                    populateModelSelect(selectElement, [], message);
                }
            } else {
                console.error(`Failed to fetch models for ${providerId}:`, response.status);
                populateModelSelect(selectElement, [], 'Error loading models');
            }
        } catch (error) {
            console.error(`Error fetching models for ${providerId}:`, error);
            populateModelSelect(selectElement, [], 'Error loading models');
        } finally {
            selectElement.disabled = false;
        }
    }
    
    function populateModelSelect(selectElement, models, emptyMessage = 'Select a model...') {
        selectElement.innerHTML = `<option value="">${emptyMessage}</option>`;
        if (models && models.length > 0) {
            models.forEach(model => {
                const option = document.createElement('option');
                
                // Handle both string format and object format
                if (typeof model === 'string') {
                    option.value = model;
                    option.textContent = model;
                } else if (model && typeof model === 'object') {
                    // Use the proper model object structure
                    option.value = model.model_id || model.name || model;
                    
                    // Create a descriptive display name
                    let displayName = model.name || model.model_id || model;
                    if (model.cost_tier) {
                        displayName += ` (${model.cost_tier})`;
                    }
                    option.textContent = displayName;
                    
                    // Add description as title for tooltip
                    if (model.description) {
                        option.title = model.description;
                    }
                } else {
                    option.value = model;
                    option.textContent = model;
                }
                
                selectElement.appendChild(option);
            });
        }
    }
    
    async function refreshModelsForProvider(providerId) {
        try {
            const response = await fetch('/api/ai/models/refresh', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: providerId })
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.status === 'success') {
                    // Update cached models
                    if (availableProviders[providerId]) {
                        availableProviders[providerId].models = data.discovered_models[providerId] || [];
                    }
                    modelDiscoveryCache[providerId] = data.discovered_models[providerId] || [];
                    return data.discovered_models[providerId] || [];
                }
            }
            console.error(`Failed to refresh models for ${providerId}`);
            return null;
        } catch (error) {
            console.error(`Error refreshing models for ${providerId}:`, error);
            return null;
        }
    }
    
    async function getCacheStatus() {
        try {
            const response = await fetch('/api/ai/models/cache/status');
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Error getting cache status:', error);
        }
        return null;
    }
    
    // Generic test/save result display
    function showAITestResult(resultElement, message, type, details = null) {
        if (!resultElement) return;
        resultElement.innerHTML = ''; // Clear previous
        resultElement.className = `test-result ${type}`; // Apply type class (success, error, info)
        resultElement.style.display = 'block';
        
        const messageEl = document.createElement('div');
        messageEl.textContent = message;
        resultElement.appendChild(messageEl);
        
        if (details) { /* ... (existing detail display logic can be moved here) ... */ }
    }

    async function testApiKeyForContext(context) { // context: { provider, apiKey, model, testBtnEl, resultEl }
        if (!context.provider || !context.apiKey.trim()) {
            showAITestResult(context.resultEl, 'Please select a provider and enter an API key.', 'error');
            return;
        }
        context.testBtnEl.disabled = true;
        context.testBtnEl.textContent = 'Testing...';
        showAITestResult(context.resultEl, 'Testing API key...', 'info');

        try {
            const response = await fetch('/api/ai/test-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: context.provider, api_key: context.apiKey, model: context.model })
            });
            const result = await response.json();
            if (result.status === 'success') {
                showAITestResult(context.resultEl, 'API key test successful!', 'success', result.test_details);
            } else {
                showAITestResult(context.resultEl, result.message || 'Test failed.', 'error', result.test_details);
            }
        } catch (error) {
            showAITestResult(context.resultEl, 'Network error during API key test.', 'error');
            console.error('API key test error:', error);
        } finally {
            context.testBtnEl.disabled = false;
            context.testBtnEl.textContent = context.isPopover ? 'Test' : 'Test Key'; // Adjust text based on context
        }
    }

    async function saveConfigurationForContext(context) { // context: { provider, apiKey, model, saveBtnEl, resultEl }
        if (!context.provider || !context.apiKey.trim()) {
            showAITestResult(context.resultEl, 'Please select a provider and enter an API key.', 'error');
            return;
        }
        context.saveBtnEl.disabled = true;
        context.saveBtnEl.textContent = 'Saving...';

        try {
            const response = await fetch('/api/ai-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    settings: [{ provider: context.provider, api_key: context.apiKey, model: context.model }],
                    default_provider: context.provider
                })
            });
            const result = await response.json();
            if (result.status === 'success') {
                showAITestResult(context.resultEl, 'Configuration saved successfully!', 'success');
                
                // Update the local state immediately to mark provider as configured
                if (availableProviders[context.provider]) {
                    availableProviders[context.provider].configured = true;
                    availableProviders[context.provider].api_key = context.apiKey;
                    availableProviders[context.provider].model = context.model;
                }
                
                // Update the AI status display immediately
                updateAIStatusDisplay();
                updateProviderStatusOnCards();
                
                // Also reload from server to ensure consistency
                await loadAvailableProviders();
                
                // If in modal, maybe close it after a delay
                if (!context.isPopover) {
                    setTimeout(() => {
                        hideGenericModal(enhancedDashboardModal);
                        resetModalAIForm();
                    }, 1500);
                } else {
                     // For popover, close after a short delay to show success
                    setTimeout(() => {
                        hidePopover(aiStatusPopover);
                    }, 1000);
                }
            } else {
                showAITestResult(context.resultEl, result.message || 'Error saving configuration.', 'error');
            }
        } catch (error) {
            showAITestResult(context.resultEl, 'Error saving configuration.', 'error');
            console.error('Save configuration error:', error);
        } finally {
            context.saveBtnEl.disabled = false;
            context.saveBtnEl.textContent = context.isPopover ? 'Save & Test' : 'Save Configuration';
        }
    }

    function resetModalAIForm() {
        selectedProvider = null; // Reset for modal context
        if (apiKeyInput) apiKeyInput.value = '';
        if (modelSelect) modelSelect.innerHTML = '<option value="">Select a model...</option>';
        if (apiKeySection) apiKeySection.style.display = 'none';
        if (testResult) testResult.style.display = 'none';
        providerCards.forEach(card => card.classList.remove('selected'));
    }
    
    function resetPopoverAIForm() {
        if(popoverProviderSelect) popoverProviderSelect.value = '';
        if(popoverApiKeyInput) popoverApiKeyInput.value = '';
        if(popoverModelSelect) popoverModelSelect.innerHTML = '<option value="">Select a model...</option>';
        if(popoverTestResult) popoverTestResult.style.display = 'none';
        // Disable model select until provider is chosen
        if(popoverModelSelect) popoverModelSelect.disabled = true;
        if(popoverApiKeyInput) popoverApiKeyInput.disabled = true;

    }

    function togglePasswordVisibilityForElement(inputElement, buttonElement) {
        if (!inputElement || !buttonElement) return;
        const type = inputElement.getAttribute('type') === 'password' ? 'text' : 'password';
        inputElement.setAttribute('type', type);
        buttonElement.textContent = type === 'password' ? '👁️' : '🙈';
    }

    function populatePopoverAIFields() {
        // Find the first configured provider, or default to empty
        const configured = Object.values(availableProviders).find(p => p.configured && p.id); // Ensure provider has an ID
        if (configured && popoverProviderSelect) {
            popoverProviderSelect.value = configured.id; // Assumes provider.id is the value for select
            updateModelOptionsForElement(popoverModelSelect, configured.id);
            if (popoverApiKeyInput) popoverApiKeyInput.value = configured.api_key || '';
            if (popoverModelSelect) popoverModelSelect.value = configured.model || '';
            if(popoverModelSelect) popoverModelSelect.disabled = false;
            if(popoverApiKeyInput) popoverApiKeyInput.disabled = false;
        } else {
            resetPopoverAIForm();
        }
    }


    // --- Existing Helper functions (showError, hideError, setLoading, isValidUrl) ---
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    function hideError() {
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';
    }

    function setLoading(isLoading) {
        if (isLoading) {
            scrapeButton.disabled = true;
            loadingSpinner.style.display = 'block';
            hideError();
        } else {
            scrapeButton.disabled = false;
            loadingSpinner.style.display = 'none';
        }
    }
    function isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (e) {
            return false;
        }
    }
    async function checkApiKeyRequired() {
        // Check if any provider is configured
        const hasConfiguredProvider = Object.values(availableProviders).some(p => p.configured);
        return !hasConfiguredProvider; // Needs key if no provider is configured
    }


    // --- Main Scrape Logic (Adapted to check API key status) ---
    async function handleScrape() {
        const query = userInput.value.trim();
        if (!query) {
            showError('Please enter what you want to scrape or a URL.');
            return;
        }

        const needsApiKey = await checkApiKeyRequired();
        if (needsApiKey) {
            showGenericModal(apiKeyRequiredModal);
            return;
        }

        setLoading(true);
        hideError();

        const payload = { query: query };
        if (isValidUrl(query)) {
            payload.start_url = query;
        }

        try {
            const response = await fetch('/scrape-intelligent', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Scrape initiated:', result);
                window.location.href = `/results?id=${result.job_id}&query=${encodeURIComponent(query)}`;
            } else {
                const errorData = await response.json();
                showError(`Error: ${errorData.detail || 'Something went wrong.'}`);
            }
        } catch (error) {
            console.error('Network or unexpected error:', error);
            showError('An unexpected error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    }
    async function handleScrapeWithoutAI() {
        const query = userInput.value.trim();
        if (!query) {
            showError('Please enter what you want to scrape or a URL.');
            return;
        }

        setLoading(true);
        hideError();

        const payload = { 
            query: query,
            options: { use_ai: false } 
        };
        if (isValidUrl(query)) {
            payload.start_url = query;
        }

        try {
            const response = await fetch('/scrape-intelligent', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Scrape initiated (without AI):', result);
                window.location.href = `/results?id=${result.job_id}&query=${encodeURIComponent(query)}`;
            } else {
                const errorData = await response.json();
                showError(`Error: ${errorData.detail || 'Something went wrong.'}`);
            }
        } catch (error) {
            console.error('Network or unexpected error:', error);
            showError('An unexpected error occurred. Please try again.');
        } finally {
            setLoading(false);
        }
    }


    // --- Event Listeners ---
    if (scrapeButton) scrapeButton.addEventListener('click', handleScrape);
    if (userInput) userInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            handleScrape();
        }
    });

    // New Homepage Controls Event Listeners
    if (settingsButton) {
        console.log('Settings button found, attaching event listener');
        settingsButton.addEventListener('click', () => {
            console.log('Settings button clicked!');
            console.log('Enhanced dashboard modal:', enhancedDashboardModal);
            showGenericModal(enhancedDashboardModal);
            // Optionally switch to a default tab
            switchTab('feature-controls-tab', featureControlsTabLink); 
        });
    } else {
        console.log('Settings button NOT found!');
    }

    if (dashboardModalCloseBtn) {
        console.log('Dashboard modal close button found');
        dashboardModalCloseBtn.addEventListener('click', () => {
            console.log('Dashboard modal close button clicked!');
            hideGenericModal(enhancedDashboardModal);
        });
    } else {
        console.log('Dashboard modal close button NOT found!');
    }
    
    if (enhancedDashboardModal) {
        console.log('Enhanced dashboard modal found');
        enhancedDashboardModal.addEventListener('click', (e) => { // Close on backdrop click
            if (e.target === enhancedDashboardModal) {
                console.log('Modal backdrop clicked, closing modal');
                hideGenericModal(enhancedDashboardModal);
            }
        });
    } else {
        console.log('Enhanced dashboard modal NOT found!');
    }

    if (aiStatusLightButton) {
        console.log('AI status light button found, attaching event listener');
        aiStatusLightButton.addEventListener('click', (e) => {
            console.log('AI status light button clicked!');
            console.log('AI status popover:', aiStatusPopover);
            e.stopPropagation(); // Prevent body click listener from hiding it immediately
            showPopover(aiStatusPopover);
        });
    } else {
        console.log('AI status light button NOT found!');
    }

    if (popoverCloseBtn) popoverCloseBtn.addEventListener('click', () => { // If a dedicated close button exists
        hidePopover(aiStatusPopover);
    });

    // Close popover on outside click
    document.addEventListener('click', (event) => {
        if (aiStatusPopover && aiStatusPopover.style.display !== 'none') {
            if (!aiStatusPopover.contains(event.target) && event.target !== aiStatusLightButton && !aiStatusLightButton.contains(event.target)) {
                hidePopover(aiStatusPopover);
            }
        }
    });


    // Dashboard Tabs
    if (featureControlsTabLink) featureControlsTabLink.addEventListener('click', (e) => {
        e.preventDefault();
        switchTab('feature-controls-tab', featureControlsTabLink);
    });
    if (aiSettingsTabLink) aiSettingsTabLink.addEventListener('click', (e) => {
        e.preventDefault();
        switchTab('ai-settings-tab', aiSettingsTabLink);
    });

    // AI Configuration Event Listeners (for main modal, using original element IDs)
    providerCards.forEach(card => { // These are cards within #ai-settings-tab
        card.addEventListener('click', () => {
            const providerId = card.getAttribute('data-provider');
            selectProviderOnCard(providerId);
        });
    });

    if (testKeyBtn) testKeyBtn.addEventListener('click', () => {
        testApiKeyForContext({
            provider: selectedProvider, // Uses the global selectedProvider from card clicks
            apiKey: apiKeyInput.value,
            model: modelSelect.value,
            testBtnEl: testKeyBtn,
            resultEl: testResult,
            isPopover: false
        });
    });
    if (saveConfigBtn) saveConfigBtn.addEventListener('click', () => {
        saveConfigurationForContext({
            provider: selectedProvider,
            apiKey: apiKeyInput.value,
            model: modelSelect.value,
            saveBtnEl: saveConfigBtn,
            resultEl: testResult,
            isPopover: false
        });
    });
    if (showHideKeyBtn) showHideKeyBtn.addEventListener('click', () => {
        togglePasswordVisibilityForElement(apiKeyInput, showHideKeyBtn);
    });

    // AI Configuration Event Listeners for Popover
    if (popoverProviderSelect) popoverProviderSelect.addEventListener('change', async () => {
        const providerId = popoverProviderSelect.value;
        if (providerId) {
            // Enable controls and show loading
            if(popoverModelSelect) popoverModelSelect.disabled = false;
            if(popoverApiKeyInput) popoverApiKeyInput.disabled = false;
            
            // Load models for the selected provider
            await updateModelOptionsForElement(popoverModelSelect, providerId);
            
            // Load existing API key if available for this provider
            if (availableProviders[providerId] && availableProviders[providerId].api_key) {
                if(popoverApiKeyInput) popoverApiKeyInput.value = availableProviders[providerId].api_key;
            } else {
                if(popoverApiKeyInput) popoverApiKeyInput.value = '';
            }
            
            // Set the model if previously configured
            if (availableProviders[providerId] && availableProviders[providerId].model) {
                if(popoverModelSelect) popoverModelSelect.value = availableProviders[providerId].model;
            }
        } else {
            // Reset to default state
            if(popoverModelSelect) popoverModelSelect.innerHTML = '<option value="">Select model...</option>';
            if(popoverModelSelect) popoverModelSelect.disabled = true;
            if(popoverApiKeyInput) popoverApiKeyInput.disabled = true;
            if(popoverApiKeyInput) popoverApiKeyInput.value = '';
        }
    });

    if (popoverSaveTestBtn) popoverSaveTestBtn.addEventListener('click', () => {
        const provider = popoverProviderSelect.value;
        const apiKey = popoverApiKeyInput.value;
        const model = popoverModelSelect.value;
        // Could do a test then save, or just save. For now, direct save.
        saveConfigurationForContext({
            provider: provider,
            apiKey: apiKey,
            model: model,
            saveBtnEl: popoverSaveTestBtn,
            resultEl: popoverTestResult,
            isPopover: true
        });
    });
    if (popoverShowHideKeyBtn) popoverShowHideKeyBtn.addEventListener('click', () => {
        togglePasswordVisibilityForElement(popoverApiKeyInput, popoverShowHideKeyBtn);
    });


    // API Key Required Modal Listeners (existing)
    if (requiredModalCloseBtn) requiredModalCloseBtn.addEventListener('click', () => {
        hideGenericModal(apiKeyRequiredModal);
    });
    if (configureNowBtn) configureNowBtn.addEventListener('click', () => {
        hideGenericModal(apiKeyRequiredModal);
        showGenericModal(enhancedDashboardModal);
        switchTab('ai-settings-tab', aiSettingsTabLink); // Go to AI settings tab
    });
    if (continueWithoutBtn) continueWithoutBtn.addEventListener('click', () => {
        hideGenericModal(apiKeyRequiredModal);
        handleScrapeWithoutAI(); // Allow scraping without AI
    });
    if (apiKeyRequiredModal) apiKeyRequiredModal.addEventListener('click', (e) => { // Close on backdrop click
        if (e.target === apiKeyRequiredModal) {
            hideGenericModal(apiKeyRequiredModal);
        }
    });
    
    // Feature Controls Tab - Placeholder Listeners
    const semanticSearchToggle = document.getElementById('semantic-search-toggle');
    if (semanticSearchToggle) semanticSearchToggle.addEventListener('click', () => {
        semanticSearchToggle.classList.toggle('active');
        console.log('Semantic Search Toggled:', semanticSearchToggle.classList.contains('active'));
    });
    // Add more for other feature controls as needed...


    // --- Initializations ---
    // Debug element availability
    console.log('🔍 Debug - Element availability at initialization:');
    console.log('halLight:', halLight);
    console.log('aiStatusText:', aiStatusText);
    console.log('availableProviders (initial):', availableProviders);
    
    loadAvailableProviders(); // Load AI provider info and update status display
    
    // Set initial tab for dashboard
    if (featureControlsTabLink) { // Ensure it's the default
        switchTab('feature-controls-tab', featureControlsTabLink);
    } else if (aiSettingsTabLink) { // Fallback if feature tab doesn't exist for some reason
         switchTab('ai-settings-tab', aiSettingsTabLink);
    }

    // Hide modals/popover by default
    if(enhancedDashboardModal) enhancedDashboardModal.style.display = 'none';
    if(aiStatusPopover) aiStatusPopover.style.display = 'none';
    if(apiKeyRequiredModal) apiKeyRequiredModal.style.display = 'none';
    resetPopoverAIForm(); // Ensure popover starts in a clean state

    // Remove obsolete `aiConfigButton` and its modal logic if any remnants
    const oldAiConfigButton = document.getElementById('ai-config-button'); // Old button ID
    if (oldAiConfigButton) {
        // If there were direct listeners on oldAiConfigButton, they are implicitly removed
        // as we are not adding them here.
        // The old `aiConfigModal` element itself is also not being used.
    }
    
    // Smooth scrolling for navigation links (if any)
    document.querySelectorAll('nav-links a').forEach(anchor => { // Make sure 'nav-links' is correct selector
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetElement = document.querySelector(this.getAttribute('href'));
            if (targetElement) {
                targetElement.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });

});
