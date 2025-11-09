/**
 * preview_page.js
 * ===============
 * 
 * Main controller for the Data Preview Page
 * 
 * Purpose:
 *  - Initialize and coordinate all preview page functionality
 *  - Handle page state management
 *  - Coordinate between different modules
 *  - Handle navigation and routing
 * 
 * Features:
 *  - Tab switching
 *  - Data loading coordination
 *  - Error handling
 *  - Loading states
 *  - Navigation controls
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.previewPage = global.DI.previewPage || {};
    
    // Page state
    let currentSourceId = null;
    let currentPreviewData = null;
    let currentQualityReport = null;
    let loadingStates = {
        quality: false,
        text: false,
        recommendations: false,
        preview: false
    };

    const PREVIEW_CACHE_MAX_AGE = 60 * 1000; // 60 seconds
    const PREVIEW_PREFETCH_DELAY = 180; // ms before prefetch kicks in
    const PREVIEW_CACHE_FALLBACK_KEY = '__default__';
    const previewDataCache = new Map();
    let previewPrefetchPromise = null;
    let previewPrefetchScheduled = false;
    let previewPageInitialized = false;
    let recommendationsRetryCount = 0;
    const MAX_RECOMMENDATIONS_RETRIES = 12;
    const RECOMMENDATIONS_RETRY_DELAY = 250;
    let eventListenersInitialized = false;
    
    /**
     * Initialize the data preview page
     */
    function initializeDataPreviewPage() {
        initializeEventListeners();

        if (previewPageInitialized) {
            if (window.DATA_PREVIEW_CONFIG?.sourceId && window.DATA_PREVIEW_CONFIG.sourceId !== currentSourceId) {
                currentSourceId = window.DATA_PREVIEW_CONFIG.sourceId;
                schedulePreviewPrefetch({ force: true, immediate: true });
            }
            return;
        }

        console.log('Initializing Data Preview Page...');
        
        // Get source ID from page config
        if (window.DATA_PREVIEW_CONFIG && window.DATA_PREVIEW_CONFIG.sourceId) {
            currentSourceId = window.DATA_PREVIEW_CONFIG.sourceId;
            console.log('Source ID:', currentSourceId);
            
            // Load initial data
            loadDataForCurrentTab();
            // Kick off a background prefetch so heavy preview work happens off the critical path
            schedulePreviewPrefetch({ delay: PREVIEW_PREFETCH_DELAY });
        } else {
            console.error('No source ID provided');
            showError('No data source specified');
            return;
        }
        
        console.log('Data Preview Page initialized successfully');
        previewPageInitialized = true;
    }

    function resolvePreviewCacheKey(sourceId) {
        if (sourceId) {
            return sourceId;
        }
        if (window.DATA_PREVIEW_CONFIG?.sourceId) {
            return window.DATA_PREVIEW_CONFIG.sourceId;
        }
        return PREVIEW_CACHE_FALLBACK_KEY;
    }

    function cachePreviewData(previewData, sourceId) {
        if (!previewData || typeof previewData !== 'object') {
            return;
        }
        const key = resolvePreviewCacheKey(sourceId);
        previewDataCache.set(key, {
            data: previewData,
            timestamp: Date.now()
        });
    }

    function getCachedPreviewData(sourceId, maxAge = PREVIEW_CACHE_MAX_AGE) {
        const key = resolvePreviewCacheKey(sourceId);
        const cached = previewDataCache.get(key);
        if (!cached) {
            return null;
        }
        if (maxAge && maxAge > 0) {
            const age = Date.now() - cached.timestamp;
            if (age > maxAge) {
                return null;
            }
        }
        return cached.data;
    }

    function invalidatePreviewCache(sourceId) {
        if (sourceId) {
            previewDataCache.delete(resolvePreviewCacheKey(sourceId));
        } else {
            previewDataCache.clear();
        }
    }

    function dispatchPreviewUpdated(previewData) {
        try {
            const event = new CustomEvent('previewDataUpdated', {
                detail: {
                    sourceId: currentSourceId,
                    preview: previewData
                }
            });
            document.dispatchEvent(event);
        } catch (e) {
            console.warn('Failed to broadcast previewDataUpdated event', e);
        }
    }

    function showPreviewLoadingState() {
        const tabContent = document.getElementById('previewTabContent');
        if (!tabContent) {
            return;
        }
        tabContent.innerHTML = `
            <div class="preview-loading-state text-center py-5">
                <i class="fas fa-spinner fa-spin fa-3x mb-3" aria-hidden="true"></i>
                <p class="mt-2 mb-1 fw-semibold">Preparing your dataset preview…</p>
                <small class="text-muted">Fetching a lightweight sample so the rest of the page stays responsive.</small>
            </div>
        `;
    }

    function renderPreviewDataset(previewData) {
        if (!previewData) {
            renderBasicDataPreview(previewData);
            return;
        }
        if (window.DI?.dataTable?.renderDataTable) {
            try {
                window.DI.dataTable.renderDataTable(previewData, 'previewTabContent');
                return;
            } catch (e) {
                console.warn('Enhanced data table render failed, falling back to basic renderer:', e);
            }
        }
        renderBasicDataPreview(previewData);
    }

    function fetchPreviewData(options = {}) {
        const { force = false } = options;

        if (!currentSourceId) {
            return Promise.reject(new Error('No data source selected.'));
        }

        if (!window.DATA_PREVIEW_CONFIG || !window.DATA_PREVIEW_CONFIG.apiBasePath) {
            return Promise.reject(new Error('Configuration error: API base path not found.'));
        }

        if (!force) {
            const cached = getCachedPreviewData(currentSourceId, PREVIEW_CACHE_MAX_AGE);
            if (cached) {
                currentPreviewData = cached;
                return Promise.resolve(cached);
            }
        }

        const sampleSize = window.DATA_PREVIEW_CONFIG.previewSampleSize || 100;
        const previewMode = window.DATA_PREVIEW_CONFIG.previewMode || 'first_last';
        let apiUrl = `${window.DATA_PREVIEW_CONFIG.apiBasePath}/${currentSourceId}/preview?sample_size=${sampleSize}&mode=${previewMode}`;
        if (force) {
            apiUrl += '&force_refresh=true';
        }

        console.log('Fetching preview dataset from URL:', apiUrl);

        return fetch(apiUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                if (!data?.success || !data.preview) {
                    throw new Error(data?.error || 'Preview data unavailable');
                }

                currentPreviewData = data.preview;
                cachePreviewData(data.preview, currentSourceId);
                dispatchPreviewUpdated(data.preview);
                return data.preview;
            });
    }

    function schedulePreviewPrefetch(options = {}) {
        const { force = false, delay = PREVIEW_PREFETCH_DELAY, immediate = false } = options;

        if (!currentSourceId || !window.DATA_PREVIEW_CONFIG?.apiBasePath) {
            return Promise.resolve(null);
        }

        if (!force) {
            const cached = getCachedPreviewData(currentSourceId, PREVIEW_CACHE_MAX_AGE);
            if (cached) {
                return Promise.resolve(cached);
            }

            if (previewPrefetchPromise) {
                return previewPrefetchPromise;
            }

            if (previewPrefetchScheduled && !immediate) {
                return previewPrefetchPromise || Promise.resolve(null);
            }
        }

        const runFetch = () => {
            previewPrefetchScheduled = false;
            previewPrefetchPromise = fetchPreviewData({ force })
                .catch(error => {
                    // Surface errors to caller but keep promise reusable
                    console.warn('Preview prefetch failed:', error);
                    throw error;
                })
                .finally(() => {
                    previewPrefetchPromise = null;
                });
            return previewPrefetchPromise;
        };

        if (immediate) {
            return runFetch();
        }

        previewPrefetchScheduled = true;

        return new Promise((resolve, reject) => {
            const execute = () => {
                runFetch().then(resolve).catch(reject);
            };

            if (typeof window.requestIdleCallback === 'function') {
                window.requestIdleCallback(() => execute(), { timeout: Math.max(delay * 5, 1200) });
            } else {
                setTimeout(execute, delay);
            }
        });
    }

    // Small helpers for showing/hiding the page-level loading overlay
    function showPageLoading(title, message) {
        try {
            const overlay = document.getElementById('loadingOverlay');
            if (!overlay) return;
            const titleEl = document.getElementById('loadingTitle');
            const msgEl = document.getElementById('loadingMessage');
            if (titleEl && title) titleEl.textContent = title;
            if (msgEl && message) msgEl.textContent = message;
            overlay.style.display = 'flex';
        } catch (e) {
            console.warn('Failed to show loading overlay', e);
        }
    }

    function hidePageLoading() {
        try {
            const overlay = document.getElementById('loadingOverlay');
            if (!overlay) return;
            overlay.style.display = 'none';
        } catch (e) {
            console.warn('Failed to hide loading overlay', e);
        }
    }
    
    /**
     * Initialize event listeners
     */
    function initializeEventListeners() {
        if (eventListenersInitialized) {
            return;
        }
        eventListenersInitialized = true;

        // Handle browser back/forward buttons
        window.addEventListener('popstate', function(event) {
            // Handle navigation state if needed
        });
        
        // Handle keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                // Could be used to close modals or go back
            }
            
            // Toggle dark mode with Ctrl+D
            if (event.ctrlKey && event.key === 'd') {
                event.preventDefault();
                toggleDarkMode();
            }
        });
        
        // Initialize dark mode toggle
        initializeDarkModeToggle();
        
        // Initialize go to top functionality
        initializeGoToTop();
        
        // Initialize scroll animations
        initializeScrollAnimations();
    }
    
    /**
     * Initialize dark mode toggle functionality
     */
    function initializeDarkModeToggle() {
        console.log('Initializing dark mode toggle...');
        
    // Check for saved theme preference (consistent with main app)
    // Prefer 'eda-theme' (new unified key), fallback to old 'darkMode' for compatibility
    const storedEda = localStorage.getItem('eda-theme');
    const storedLegacy = localStorage.getItem('darkMode');
    const isDark = (storedEda === 'dark') || (storedEda == null && storedLegacy === 'true');
        console.log('Current dark mode state:', isDark);
        
        if (isDark) {
            document.body.classList.add('dark-mode');
        }
        
        updateDarkModeIcon(isDark);
        
        // Add event listener for navbar toggle button
        const navToggleBtn = document.querySelector('.theme-toggle-nav');
        console.log('Found navbar toggle button:', !!navToggleBtn);
        if (navToggleBtn) {
            if (navToggleBtn.tagName === 'BUTTON') {
                navToggleBtn.type = 'button';
            }
            // Remove any existing listeners to prevent duplicates
            navToggleBtn.removeEventListener('click', toggleDarkMode);
            navToggleBtn.addEventListener('click', toggleDarkMode);
            console.log('Added click listener to navbar toggle');
        }
        
        // Fallback for old toggle button
        const toggleBtn = document.getElementById('darkModeToggle');
        if (toggleBtn) {
            toggleBtn.removeEventListener('click', toggleDarkMode);
            toggleBtn.addEventListener('click', toggleDarkMode);
            console.log('Added click listener to fallback toggle');
        }
        
        // Listen for system theme changes if no saved preference
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem('darkMode')) {
                if (e.matches) {
                    document.body.classList.add('dark-mode');
                    updateDarkModeIcon(true);
                } else {
                    document.body.classList.remove('dark-mode');
                    updateDarkModeIcon(false);
                }
            }
        });
    }
    
    /**
     * Toggle dark mode
     */
    function toggleDarkMode() {
        // Toggle dark mode using rAF to batch class changes and avoid layout thrashing
        console.log('toggleDarkMode called');
        document.body.classList.add('theme-transition');

        // Use requestAnimationFrame to ensure the class flip is scheduled without forcing sync layout
        requestAnimationFrame(() => {
            const isDark = document.body.classList.toggle('dark-mode');
            console.log('New dark mode state:', isDark);
            updateDarkModeIcon(isDark);
            // Save preference using unified key
            try {
                localStorage.setItem('eda-theme', isDark ? 'dark' : 'light');
                // Keep legacy key for backward compatibility (optional)
                localStorage.setItem('darkMode', isDark.toString());
            } catch (e) {
                console.warn('Unable to persist theme preference', e);
            }

            // Remove transition helper after the transition finishes
            window.setTimeout(() => {
                document.body.classList.remove('theme-transition');
            }, 320);
        });
    }
    
    /**
     * Update dark mode icon
     */
    function updateDarkModeIcon(isDark) {
        let toggleBtn = document.querySelector('.theme-toggle-nav');
        let icon = document.getElementById('theme-icon')
            || document.getElementById('themeToggleNavIcon')
            || toggleBtn?.querySelector('i');

        if (!icon) {
            const legacyToggle = document.getElementById('darkModeToggle');
            if (legacyToggle) {
                toggleBtn = legacyToggle;
                icon = legacyToggle.querySelector('i');
            }
        }

        if (toggleBtn) {
            toggleBtn.title = isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        }

        if (icon) {
            if (icon.classList.contains('bi')) {
                icon.classList.remove('bi-moon-stars', 'bi-brightness-high', 'bi-moon', 'bi-sun');
                icon.classList.add(isDark ? 'bi-brightness-high' : 'bi-moon-stars');
            } else {
                icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
    }
    
    /**
     * Go to top function (for navbar button)
     */
    function goToTop() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }
    
    /**
     * Initialize go to top functionality
     */
    function initializeGoToTop() {
        // Try navbar go-to-top button first (new structure)
        let goToTopBtn = document.querySelector('.go-to-top');
        
        // Fallback to old element ID
        if (!goToTopBtn) {
            goToTopBtn = document.getElementById('goToTop');
        }
        
        if (!goToTopBtn) return;
        
        // Show/hide button based on scroll position
        const toggleGoToTop = () => {
            if (window.pageYOffset > 300) {
                goToTopBtn.classList.add('visible');
            } else {
                goToTopBtn.classList.remove('visible');
            }
        };
        
        // Add scroll listener
        window.addEventListener('scroll', toggleGoToTop);
        
        // Add click listener (if not already handled by onclick in HTML)
        if (!goToTopBtn.getAttribute('onclick')) {
            goToTopBtn.addEventListener('click', () => {
                window.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            });
        }
        
        // Initial check
        toggleGoToTop();
    }
    
    /**
     * Initialize scroll animations
     */
    function initializeScrollAnimations() {
        // Add intersection observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);
        
        // Observe quality sections
        const sections = document.querySelectorAll('.quality-section');
        sections.forEach(section => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(section);
        });
    }
    
    /**
     * Switch between preview tabs
     */
    function switchPreviewTab(tabName) {
        console.log('Switching to tab:', tabName);
        
        // Update tab buttons
        document.querySelectorAll('.preview-tab-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.getAttribute('data-tab') === tabName) {
                btn.classList.add('active');
            }
        });

        // Update tab content
        document.querySelectorAll('.preview-tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        const targetContent = document.getElementById(tabName + 'TabContent');
        if (targetContent) {
            targetContent.classList.add('active');
            
            // Load data for this tab if not already loaded
            loadDataForTab(tabName);
        }
    }
    
    /**
     * Load data for the currently active tab
     */
    function loadDataForCurrentTab() {
        const activeTab = document.querySelector('.preview-tab-btn.active');
        if (activeTab) {
            const tabName = activeTab.getAttribute('data-tab');
            loadDataForTab(tabName);
        }
    }
    
    /**
     * Load data for a specific tab
     */
    function loadDataForTab(tabName) {
        if (!currentSourceId) {
            console.error('No source ID available');
            return;
        }
        
        if (loadingStates[tabName]) {
            console.log(`Tab ${tabName} is already loading`);
            return;
        }
        
        console.log(`Loading data for tab: ${tabName}`);
        loadingStates[tabName] = true;
        
        switch (tabName) {
            case 'quality':
                loadQualityReport();
                break;
            case 'text':
                loadTextAnalysis();
                break;
            case 'recommendations':
                loadRecommendations();
                break;
            case 'preview':
                loadDataPreview();
                break;
            default:
                console.warn('Unknown tab:', tabName);
                loadingStates[tabName] = false;
        }
    }
    
    /**
     * Load quality report data
     */
    function loadQualityReport() {
        console.log('Loading quality report...');
        
        // Use current sample size if available, otherwise default to 500
        const sampleSize = window.currentSampleSize || 500;
        const apiUrl = `${window.DATA_PREVIEW_CONFIG.apiBasePath}/${currentSourceId}/quality-report?sample_size=${sampleSize}`;
        console.log('API URL:', apiUrl);
        
        fetch(apiUrl)
            .then(response => {
                console.log('Response status:', response.status);
                console.log('Response headers:', response.headers);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                loadingStates.quality = false;
                console.log('Quality report data received:', data);
                
                if (data.success) {
                    currentQualityReport = data.quality_report;

                    if (window.DI?.recommendations?.hydrateExistingRecommendations) {
                        window.DI.recommendations.hydrateExistingRecommendations('recommendationsTabContent');
                    }
                    
                    // Check if the quality report module is available
                    console.log('DI object:', window.DI);
                    console.log('qualityReport module:', window.DI && window.DI.qualityReport);
                    console.log('renderQualityReport function:', window.DI && window.DI.qualityReport && window.DI.qualityReport.renderQualityReport);
                    
                    // Use the quality report module to render the content
                    if (window.DI && window.DI.qualityReport && window.DI.qualityReport.renderQualityReport) {
                        try {
                            window.DI.qualityReport.renderQualityReport(currentQualityReport, 'qualityTabContent');
                        } catch (renderError) {
                            console.error('Error rendering quality report:', renderError);
                            showTabError('qualityTabContent', 'Error rendering quality report: ' + renderError.message);
                        }
                    } else {
                        console.warn('Quality report module not available, using fallback');
                        // Fallback rendering
                        renderBasicQualityReport(currentQualityReport);
                    }
                } else {
                    showTabError('qualityTabContent', 'Failed to load quality report: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                loadingStates.quality = false;
                console.error('Error loading quality report:', error);
                console.error('Error details:', error.message, error.stack);
                showTabError('qualityTabContent', 'Network error loading quality report: ' + error.message);
            });
    }
    
    /**
     * Load text analysis data
     */
    function loadTextAnalysis() {
        console.log('Loading text analysis...');
        
        // If we already have quality report data, use it
        if (currentQualityReport) {
            if (window.DI && window.DI.textAnalysis && window.DI.textAnalysis.renderTextAnalysis) {
                window.DI.textAnalysis.renderTextAnalysis(currentQualityReport, 'textTabContent');
            } else {
                renderBasicTextAnalysis(currentQualityReport);
            }
            loadingStates.text = false;
            return;
        }
        
        // Otherwise, load quality report first
        const sampleSize = window.currentSampleSize || 500;
        const apiUrl = `${window.DATA_PREVIEW_CONFIG.apiBasePath}/${currentSourceId}/quality-report?sample_size=${sampleSize}`;
        
        fetch(apiUrl)
            .then(response => response.json())
            .then(data => {
                loadingStates.text = false;
                
                if (data.success) {
                    currentQualityReport = data.quality_report;
                    
                    if (window.DI && window.DI.textAnalysis && window.DI.textAnalysis.renderTextAnalysis) {
                        window.DI.textAnalysis.renderTextAnalysis(currentQualityReport, 'textTabContent');
                    } else {
                        renderBasicTextAnalysis(currentQualityReport);
                    }
                } else {
                    showTabError('textTabContent', 'Failed to load text analysis: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                loadingStates.text = false;
                console.error('Error loading text analysis:', error);
                showTabError('textTabContent', 'Network error loading text analysis');
            });
    }
    
    /**
     * Load recommendations data
     */
    function loadRecommendations() {
        console.log('Loading recommendations...');
        
        // If we already have quality report data, use it
        if (currentQualityReport) {
            if (useEnhancedRecommendations()) {
                recommendationsRetryCount = 0;
                window.DI.recommendations.renderRecommendations(currentQualityReport, 'recommendationsTabContent');
            } else if (scheduleEnhancedRetry()) {
                return;
            } else {
                renderBasicRecommendations(currentQualityReport);
            }
            loadingStates.recommendations = false;
            return;
        }
        
        // Otherwise, load quality report first
        const sampleSize = window.currentSampleSize || 500;
        const apiUrl = `${window.DATA_PREVIEW_CONFIG.apiBasePath}/${currentSourceId}/quality-report?sample_size=${sampleSize}`;
        
        fetch(apiUrl)
            .then(response => response.json())
            .then(data => {
                loadingStates.recommendations = false;
                
                if (data.success) {
                    currentQualityReport = data.quality_report;
                    
                    if (useEnhancedRecommendations()) {
                        recommendationsRetryCount = 0;
                        window.DI.recommendations.renderRecommendations(currentQualityReport, 'recommendationsTabContent');
                    } else if (!scheduleEnhancedRetry()) {
                        renderBasicRecommendations(currentQualityReport);
                    }
                } else {
                    showTabError('recommendationsTabContent', 'Failed to load recommendations: ' + (data.error || 'Unknown error'));
                }
            })
            .catch(error => {
                loadingStates.recommendations = false;
                console.error('Error loading recommendations:', error);
                showTabError('recommendationsTabContent', 'Network error loading recommendations');
            });
    }

    function useEnhancedRecommendations() {
        return Boolean(window.DI && window.DI.recommendations && typeof window.DI.recommendations.renderRecommendations === 'function');
    }

    function scheduleEnhancedRetry() {
        if (recommendationsRetryCount >= MAX_RECOMMENDATIONS_RETRIES) {
            return false;
        }

        recommendationsRetryCount += 1;
        console.log(`Enhanced recommendations not ready (attempt ${recommendationsRetryCount}); retrying...`);
        loadingStates.recommendations = false;
        setTimeout(() => loadRecommendations(), RECOMMENDATIONS_RETRY_DELAY);
        return true;
    }
    
    /**
     * Load data preview
     */
    function loadDataPreview(options = {}) {
        const { force = false } = options;

        console.log('Hydrating data preview (deferred)...');
        console.log('Current source ID:', currentSourceId);

        if (!currentSourceId) {
            console.error('No current source ID available');
            showTabError('previewTabContent', 'No data source selected');
            loadingStates.preview = false;
            return;
        }

        if (!window.DATA_PREVIEW_CONFIG || !window.DATA_PREVIEW_CONFIG.apiBasePath) {
            console.error('DATA_PREVIEW_CONFIG not available or missing apiBasePath');
            showTabError('previewTabContent', 'Configuration error: API base path not found');
            loadingStates.preview = false;
            return;
        }

        if (!force) {
            const cached = getCachedPreviewData(currentSourceId, PREVIEW_CACHE_MAX_AGE);
            if (cached) {
                console.log('Using cached preview dataset.');
                currentPreviewData = cached;
                loadingStates.preview = false;
                renderPreviewDataset(cached);
                // Refresh quietly in the background so the cache stays warm
                schedulePreviewPrefetch({ force: true, delay: PREVIEW_PREFETCH_DELAY * 2 });
                return;
            }
        }

        showPreviewLoadingState();
        loadingStates.preview = true;

        schedulePreviewPrefetch({
            force,
            delay: PREVIEW_PREFETCH_DELAY,
            immediate: options.immediate === true
        })
            .then(preview => {
                loadingStates.preview = false;
                const dataToRender = preview || currentPreviewData || getCachedPreviewData(currentSourceId, 0);
                if (dataToRender) {
                    currentPreviewData = dataToRender;
                    renderPreviewDataset(dataToRender);
                } else {
                    showTabError('previewTabContent', 'Preview data is still loading. Please try again.');
                }
            })
            .catch(error => {
                loadingStates.preview = false;
                console.error('Error loading data preview:', error);
                let errorMessage = error?.message || 'Network error loading data preview';
                if (errorMessage.includes('Failed to fetch')) {
                    errorMessage = 'Unable to connect to server. Please check your network connection.';
                }
                showTabError('previewTabContent', errorMessage);
            });
    }
    
    /**
     * Show error in a tab
     */
    function showTabError(tabId, message) {
        const tabContent = document.getElementById(tabId);
        if (tabContent) {
            tabContent.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Error:</strong> ${message}
                    <div class="mt-3">
                        <button class="btn btn-outline-danger btn-sm" onclick="retryLoadData('${tabId.replace('TabContent', '')}')">
                            <i class="fas fa-redo"></i> Retry
                        </button>
                        <button class="btn btn-outline-secondary btn-sm ms-2" onclick="showDebugInfo()">
                            <i class="fas fa-info-circle"></i> Debug Info
                        </button>
                    </div>
                </div>
            `;
        }
    }
    
    /**
     * Retry loading data for a specific tab
     */
    function retryLoadData(tabName) {
        console.log('Retrying load for tab:', tabName);
        loadingStates[tabName] = false; // Reset loading state
        loadDataForTab(tabName);
    }
    
    /**
     * Show debug information
     */
    function showDebugInfo() {
        const debugInfo = {
            currentSourceId: currentSourceId,
            apiConfig: window.DATA_PREVIEW_CONFIG,
            loadingStates: loadingStates,
            userAgent: navigator.userAgent,
            url: window.location.href
        };
        
        console.log('Debug Information:', debugInfo);
        
        alert(`Debug Information:
        
Source ID: ${currentSourceId || 'Not set'}
API Base Path: ${window.DATA_PREVIEW_CONFIG?.apiBasePath || 'Not configured'}
Current URL: ${window.location.href}

Check the browser console for more detailed information.`);
    }
    
    /**
     * Show general error
     */
    function showError(message) {
        if (window.DI && window.DI.utilities && window.DI.utilities.notifications) {
            window.DI.utilities.notifications.showNotification(message, 'error');
        } else {
            alert('Error: ' + message);
        }
    }
    
    /**
     * Basic fallback rendering functions
     */
    function renderBasicQualityReport(qualityReport) {
        const content = document.getElementById('qualityTabContent');
        if (!content) return;
        
        const metadata = qualityReport.basic_metadata;
        const quality = qualityReport.quality_metrics;
        
        content.innerHTML = `
            <div class="row">
                <div class="col-md-12">
                    <h4><i class="fas fa-chart-bar"></i> Data Quality Overview</h4>
                    <div class="row mb-4">
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${metadata.sample_rows.toLocaleString()}</h5>
                                    <p class="card-text">Sample Rows</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${metadata.total_columns}</h5>
                                    <p class="card-text">Columns</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${quality.overall_completeness.toFixed(1)}%</h5>
                                    <p class="card-text">Data Completeness</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${quality.columns_with_missing}</h5>
                                    <p class="card-text">Columns with Missing Values</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    function renderBasicTextAnalysis(qualityReport) {
        const content = document.getElementById('textTabContent');
        if (!content) return;
        
        const textSummary = qualityReport.text_analysis_summary || {};
        
        content.innerHTML = `
            <div class="row">
                <div class="col-md-12">
                    <h4><i class="fas fa-font"></i> Text Analysis Overview</h4>
                    <div class="row mb-4">
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${textSummary.total_text_columns || 0}</h5>
                                    <p class="card-text">Text Columns</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${textSummary.free_text_columns || 0}</h5>
                                    <p class="card-text">NLP Candidates</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card text-center">
                                <div class="card-body">
                                    <h5 class="card-title">${textSummary.categorical_text_columns || 0}</h5>
                                    <p class="card-text">Categorical Text</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    function renderBasicRecommendations(qualityReport) {
        const content = document.getElementById('recommendationsTabContent');
        if (!content) return;

        if (window.DI && window.DI.recommendations && typeof window.DI.recommendations.renderRecommendations === 'function') {
            window.DI.recommendations.renderRecommendations(qualityReport, 'recommendationsTabContent');
            return;
        }
        const contentRegion = content.querySelector('[data-role="recommendations-content"]');
        const root = content.querySelector('[data-role="recommendations-root"]') || content;

        const recommendations = qualityReport.recommendations || [];
        
        let html = `
            <div class="row">
                <div class="col-md-12">
                    <h4><i class="fas fa-lightbulb"></i> Data Analysis Recommendations</h4>
        `;
        
        if (recommendations.length > 0) {
            html += '<div class="row">';
            recommendations.forEach((rec, index) => {
                const title = rec.title || `Recommendation ${index + 1}`;
                const description = rec.description || rec;
                
                html += `
                    <div class="col-md-6 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title"><i class="fas fa-lightbulb"></i> ${title}</h6>
                                <p class="card-text">${description}</p>
                            </div>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
        } else {
            html += `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i>
                    No specific recommendations available for this dataset.
                </div>
            `;
        }
        
        html += '</div></div>';

        if (contentRegion) {
            contentRegion.innerHTML = html;
        } else {
            content.innerHTML = html;
        }

        // Basic stat hydration for fallback mode
        const metadata = qualityReport.basic_metadata || {};
        const quality = qualityReport.quality_metrics || {};
        const textSummary = qualityReport.text_analysis_summary || {};

        const totalColsElement = root.querySelector('#total-columns');
        const sampleRowsElement = root.querySelector('#sample-rows');
        const completenessElement = root.querySelector('#completeness');
        const textColumnsElement = root.querySelector('#text-columns');

        if (totalColsElement) totalColsElement.textContent = metadata.total_columns || '-';
        if (sampleRowsElement) sampleRowsElement.textContent = metadata.sample_rows ? metadata.sample_rows.toLocaleString() : '-';
        if (completenessElement) {
            completenessElement.textContent = quality.overall_completeness != null
                ? `${Number(quality.overall_completeness).toFixed(1)}%`
                : '-';
        }
        if (textColumnsElement) textColumnsElement.textContent = textSummary.total_text_columns || '-';
    }
    
    function renderBasicDataPreview(previewData) {
        const content = document.getElementById('previewTabContent');
        if (!content) return;
        
        if (!previewData || !previewData.columns) {
            content.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    No preview data available.
                </div>
            `;
            return;
        }
        
        const tableId = 'previewTable_main';
        const sampleRows = previewData.sample_data ? previewData.sample_data.length : 0;
        const totalCols = previewData.columns ? previewData.columns.length : 0;
        
        let html = `
            <div class="row">
                <div class="col-md-12">
                    <div class="data-preview-header">
                        <h4><i class="fas fa-table"></i> Data Sample</h4>
                        <div class="data-preview-stats">
                            <div class="data-stats-inline">
                                <div class="stat-item">
                                    <span class="stat-value">${sampleRows}</span>
                                    <span class="stat-label">Sample Rows</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-value">${totalCols}</span>
                                    <span class="stat-label">Columns</span>
                                </div>
                            </div>
                        </div>
                        <div class="data-preview-actions-below-stats">
                            <button class="btn column-info-btn" onclick="showColumnInfo()" aria-label="Column information">
                                <span class="column-info-icon"><i class="fas fa-info-circle"></i></span>
                                <span class="column-info-text">
                                    <strong>Columns</strong>
                                    <small class="text-muted d-block">Details & Null Counts</small>
                                </span>
                            </button>
                        </div>
                    </div>
                    <div class="table-responsive">
                        <table id="${tableId}" class="table table-striped table-bordered">
                            <thead>
                                <tr>
        `;
        
        previewData.columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        
        html += `
                                </tr>
                            </thead>
                            <tbody>
        `;
        
        (previewData.sample_data || []).forEach(row => {
            html += '<tr>';
            previewData.columns.forEach(col => {
                const value = row[col];
                html += `<td>${value !== null && value !== undefined ? value : '<em>null</em>'}</td>`;
            });
            html += '</tr>';
        });
        
        html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
        
        content.innerHTML = html;
        
        // Initialize DataTable if available
        if (window.$ && window.$.fn.DataTable) {
            // If a DataTable already exists, destroy it first (safe re-init)
            try {
                const existing = $(`#${tableId}`).DataTable();
                if (existing && existing.destroy) {
                    existing.destroy();
                    $(`#${tableId}`).empty();
                }
            } catch (e) {
                // ignore - likely no existing instance
            }

            $(`#${tableId}`).DataTable({
                pageLength: 25,
                scrollX: true,
                order: [],
                deferRender: true, // help with large samples
                responsive: true
            });
        }
    }
    
    /**
     * Navigation functions
     */
    function goBackToDataSources() {
        window.location.href = '/data/';
    }
    
    /**
     * Action functions (delegated to existing modules)
     */
    function moveToEDA() {
        console.log('Data preview moveToEDA called');
        console.log('Current source ID:', currentSourceId);
        
        let sourceId = currentSourceId;
        
        // If no source ID available, try to get it from URL parameters or window config
        if (!sourceId) {
            console.warn('moveToEDA: No currentSourceId available in data preview');
            
            // Try to get from URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            sourceId = urlParams.get('source_id');
            
            // Try to get from window config
            if (!sourceId && window.DATA_PREVIEW_CONFIG && window.DATA_PREVIEW_CONFIG.sourceId) {
                sourceId = window.DATA_PREVIEW_CONFIG.sourceId;
            }
            
            // As a last resort, ask the user
            if (!sourceId) {
                sourceId = prompt('Please enter the dataset ID to open in ML Workflow:');
            }
            
            if (!sourceId) {
                showError('No dataset ID available for ML Workflow navigation');
                return;
            }
        }

        // Show confirmation modal with ML workflow information
        const confirmed = confirm(
            `Open this dataset in the ML Workflow canvas?\n\n` +
            `This will:\n` +
            `• Launch the ML Workflow experience\n` +
            `• Surface quick data quality insights\n` +
            `• Recommend feature engineering steps\n` +
            `• Prepare the dataset for downstream modeling\n\n` +
            `Continue to ML Workflow?`
        );

        if (!confirmed) {
            console.log('moveToEDA: User cancelled');
            return;
        }

        console.log('moveToEDA: Proceeding with ML Workflow navigation for source:', sourceId);
        
        // Show loading indicator or simple notification
        if (window.DI && window.DI.utilities && window.DI.utilities.notifications) {
            window.DI.utilities.notifications.showLoading(true, 'Navigating to ML Workflow...', 'Opening ML Workflow canvas...');
        } else {
            console.log('Navigating to ML Workflow interface...');
        }
        
        // Update global state if available
        if (window.DI && window.DI.state) {
            window.DI.state.currentSourceId = sourceId;
        }
        
    // Navigate to ML workflow canvas
    const workflowUrl = `/ml-workflow?source_id=${sourceId}`;
    console.log('Navigating to ML Workflow URL:', workflowUrl);
    window.location.href = workflowUrl;
    }
    
    function loadFullDataset() {
        if (window.loadFullDataset) {
            // Set the current source ID for the existing function
            if (window.DI && window.DI.state) {
                window.DI.state.currentSourceId = currentSourceId;
            }
            window.loadFullDataset();
        } else {
            showError('Dataset loading not available');
        }
    }
    
    /**
     * Show column information (fallback if data_table module not available)
     */
    function showColumnInfo() {
        const dataTableApi = window.DI && window.DI.dataTable;
        if (dataTableApi && typeof dataTableApi.showColumnInfo === 'function') {
            dataTableApi.showColumnInfo();
            return;
        }

        // Simple fallback if the data table module isn't available yet
        if (currentPreviewData && Array.isArray(currentPreviewData.columns)) {
            const columnList = currentPreviewData.columns.join(', ');
            alert(`Columns (${currentPreviewData.columns.length}): ${columnList}`);
            return;
        }

        alert('Column information not available');
    }
    
    // Add a simple test function for debugging EDA navigation
    function testEDANavigation() {
        console.log('Testing ML Workflow navigation from data preview...');
        const testSourceId = 'test123';
        const workflowUrl = `/ml-workflow?source_id=${testSourceId}`;
        console.log('Test ML Workflow URL:', workflowUrl);
        if (confirm('Navigate to ML Workflow test page from data preview?')) {
            window.location.href = workflowUrl;
        }
    }
    
    // Export functions to global scope
    global.initializeDataPreviewPage = initializeDataPreviewPage;
    global.switchPreviewTab = switchPreviewTab;
    global.goBackToDataSources = goBackToDataSources;
    global.moveToEDA = moveToEDA;
    global.loadFullDataset = loadFullDataset;
    global.showColumnInfo = showColumnInfo;
    global.toggleDarkMode = toggleDarkMode;
    global.goToTop = goToTop;
    global.retryLoadData = retryLoadData;
    global.showDebugInfo = showDebugInfo;
    global.testEDANavigation = testEDANavigation;
    
    // Store state globally
    global.DI.previewPage.getCurrentSourceId = () => currentSourceId;
    global.DI.previewPage.getCurrentPreviewData = () => currentPreviewData;
    global.DI.previewPage.getCurrentQualityReport = () => currentQualityReport;
    global.DI.previewPage.setCurrentQualityReport = (report) => { currentQualityReport = report; };
    global.DI.previewPage.invalidatePreviewCache = invalidatePreviewCache;
    global.DI.previewPage.isInitialized = () => previewPageInitialized;
    global.DI.previewPage.loadDataForTab = (tabName) => {
        try {
            return loadDataForTab(tabName);
        } catch (error) {
            console.warn('previewPage.loadDataForTab failed:', error);
            throw error;
        }
    };
    global.DI.previewPage.loadDataForCurrentTab = () => {
        try {
            return loadDataForCurrentTab();
        } catch (error) {
            console.warn('previewPage.loadDataForCurrentTab failed:', error);
            throw error;
        }
    };
    global.DI.previewPage.initialize = () => {
        if (typeof initializeDataPreviewPage === 'function') {
            initializeDataPreviewPage();
        }
    };
    global.DI.previewPage.refreshPreviewData = (force = false) => schedulePreviewPrefetch({ force: !!force, immediate: true });
    
})(window);

// Auto-initialize when DOM is ready to avoid requiring inline init
document.addEventListener('DOMContentLoaded', function() {
    if (typeof initializeDataPreviewPage === 'function') {
        try { initializeDataPreviewPage(); } catch (e) { console.error('Failed to init preview page', e); }
    }
});
