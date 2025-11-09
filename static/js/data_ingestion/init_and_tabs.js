/**
 * init_and_tabs.js
 * ================
 * 
 * Purpose: Initialization, tab navigation, theme handling, and global listeners
 * 
 * Features:
 *  - Application initialization and setup
 *  - Tab navigation management
 *  - Theme switching functionality
 *  - Global event listeners setup
 *  - Initial data loading
 * 
 * Dependencies:
 *  - refreshDataSources (data_sources.js)
 *  - applyOrderingToCurrentSources (utilities_formatting.js)
 *  - showNotification (shared/notifications.js)
 *  - toggleTheme (shared/notifications.js)
 * 
 * Load Order: Should load AFTER utilities modules and BEFORE modules that rely on initialized tabs
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.init = global.DI.init || {};

    /**
     * Initialize the data ingestion application
     * Sets up all core functionality and event listeners
     */
    function initializeDataIngestion() {
        try {
            // Attach core initializers
            setupTabNavigation();
            setupFileUpload();
            setupDataSourceSearch();

            // Initialize web scraping functionality
            if (typeof setupWebScraping === 'function') {
                try { setupWebScraping(); } catch (e) { console.warn('setupWebScraping failed', e); }
            }

            // Initialize theme using unified key
            try {
                const savedTheme = localStorage.getItem('eda-theme');
                const isDark = savedTheme === 'dark' || (savedTheme === null && localStorage.getItem('darkMode') === 'true');
                
                if (isDark) {
                    document.body.classList.add('dark-mode');
                    const themeIcon = document.getElementById('theme-icon');
                    if (themeIcon) {
                        themeIcon.className = 'fas fa-sun';
                    }
                }
            } catch (e) {}

            // Initialize empty data sources list - user will click refresh to load
            let sourcesPromise = Promise.resolve();
            try {
                // Show initial empty state instead of auto-loading
                const container = document.getElementById('dataSourcesList');
                if (container) {
                    container.innerHTML = `
                        <div class="text-center py-4">
                            <i class="fas fa-database" style="font-size: 3rem; opacity: 0.3; margin-bottom: 1rem;"></i>
                            <h5>Ready to Load Data Sources</h5>
                            <p class="text-muted">Click the refresh button to load your data sources.</p>
                            <button class="btn btn-primary" onclick="refreshDataSources()">
                                <i class="fas fa-sync-alt"></i> Load Data Sources
                            </button>
                        </div>
                    `;
                }
            } catch (e) { console.warn('Failed to set initial state', e); }

            // Setup ordering control
            try {
                const orderSel = document.getElementById('dsOrderSelect');
                if (orderSel) {
                    orderSel.addEventListener('change', function() {
                        // If we have cached sources, reorder locally for immediate feedback
                        if (window.currentDataSources && Array.isArray(window.currentDataSources)) {
                            applyOrderingToCurrentSources(window.currentDataSources);
                            displayDataSourcesList(window.currentDataSources);
                        } else {
                            // Fallback: refresh from server
                            refreshDataSources();
                        }
                    });
                }
            } catch (e) {}

            global.DI.utilities.notifications.showNotification('Data ingestion application initialized successfully', 'success');

        } catch (error) {
            console.error('Error initializing data ingestion:', error);
            try {
                global.DI.utilities.notifications.showNotification('Error initializing application: ' + (error && error.message ? error.message : String(error)), 'error');
            } catch (e) {}
        } finally {
            // Wait for initial sources to finish (or fail),
            // then reveal the UI. This avoids the need to rely on a fixed timeout.
            try {
                Promise.allSettled([sourcesPromise])
                    .then(() => {
                        // Small extra delay to let layout settle (tunable)
                        const settleMs = 120;
                        setTimeout(() => {
                            try {
                                const loadingOverlay = document.getElementById('loading-overlay');
                                if (loadingOverlay) loadingOverlay.style.display = 'none';
                            } catch (e) {}

                            try {
                                const dash = document.getElementById('dashboardContainer');
                                if (dash) dash.style.display = '';
                            } catch (e) {}

                            try { document.body.classList.remove('pre-init'); } catch (e) {}

                            try {
                                const initOverlay = document.getElementById('initLoadingOverlay');
                                if (initOverlay) {
                                    if (typeof initOverlay.remove === 'function') {
                                        initOverlay.remove();
                                    } else if (initOverlay.parentNode) {
                                        initOverlay.parentNode.removeChild(initOverlay);
                                    } else {
                                        initOverlay.style.display = 'none';
                                    }
                                }
                            } catch (e) {}
                        }, settleMs);
                    })
                    .catch(() => {
                        // If Promise.allSettled itself errors unexpectedly, fallback to immediate reveal
                        try { document.body.classList.remove('pre-init'); } catch (e) {}
                        try { const initOverlay = document.getElementById('initLoadingOverlay'); if (initOverlay) initOverlay.style.display = 'none'; } catch (e) {}
                    });
            } catch (e) {
                // Fallback to immediate removal if something goes wrong
                try {
                    const loadingOverlay = document.getElementById('loading-overlay');
                    if (loadingOverlay) loadingOverlay.style.display = 'none';
                } catch (e) {}
                try {
                    const dash = document.getElementById('dashboardContainer');
                    if (dash) dash.style.display = '';
                } catch (e) {}
                try {
                    const initOverlay = document.getElementById('initLoadingOverlay');
                    if (initOverlay) initOverlay.style.display = 'none';
                } catch (e) {}
            }
        }
    }

    /**
     * Setup tab navigation event listeners
     * Handles switching between different ingestion methods
     */
    function setupTabNavigation() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const targetTab = this.dataset.tab;
                switchTab(targetTab);
            });
        });
    }

// [MODULE init_and_tabs] switchTab -> target: init_and_tabs.js
function switchTab(tabId) {
    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');
    
    // Update content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
    
    // Clear newly uploaded highlighting when leaving data sources tab
    if (tabId !== 'sources') {
        sessionStorage.removeItem('newly_uploaded_source');
    }
    
    // Refresh data sources when switching to sources tab
    if (tabId === 'sources') {
        refreshDataSources(); // ensure latest
    }
}

    /**
     * Get the currently active tab
     * @returns {string|null} The active tab name or null if none found
     */
    function getCurrentActiveTab() {
        const activeTab = document.querySelector('.tab-btn.active');
        return activeTab ? activeTab.getAttribute('data-tab') : null;
    }

    // Export functions to global scope for HTML and other modules
    global.DI.init.initializeDataIngestion = initializeDataIngestion;
    global.DI.init.setupTabNavigation = setupTabNavigation;
    global.DI.init.switchTab = switchTab;
    global.DI.init.getCurrentActiveTab = getCurrentActiveTab;
    
    // Also export to global for backwards compatibility
    global.initializeDataIngestion = initializeDataIngestion;
    global.setupTabNavigation = setupTabNavigation;
    global.switchTab = switchTab;
    global.getCurrentActiveTab = getCurrentActiveTab;

})(window);