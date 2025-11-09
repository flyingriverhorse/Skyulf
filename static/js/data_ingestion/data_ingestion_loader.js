/**
 * data_ingestion_loader.js
 * ========================
 * 
 * Main loader script for the Data Ingestion Module System
 * 
 * This file loads all necessary modules in the correct order and initializes
 * the data ingestion application. Include ONLY this file in your HTML.
 * 
 * Load Order:
 * 1. Global state and utilities
 * 2. Core functionality modules
 * 3. Specialized feature modules
 * 4. Application initialization
 * 
 * Usage in HTML:
 * <script src="/static/js/data_ingestion/data_ingestion_loader.js"></script>
 */

(function() {
    'use strict';
    
    // Shared modules reused across multiple experiences
    const SHARED_MODULES = [
        '/static/js/shared/notifications.js'
    ];

    // Base path for all data ingestion modules
    const BASE_PATH = '/static/js/data_ingestion/';
    
    // Module load order (CRITICAL - do not change order without understanding dependencies)
    const MODULES = [
        // 1. Global state (must be first)
        'data_ingestion_globals.js',
        
        // 2. Utility modules (foundation for other modules)
        'utilities_formatting.js',
        
        // 3. Core functionality modules
        'init_and_tabs.js',
        'upload_module.js',
        'data_sources.js',
        'preview_eda.js',
        
        // 4. Export functionality
        'export_data.js'
    ];
    
    /**
     * Load a single JavaScript module
     * @param {string} src - The source path of the module
     * @returns {Promise} Promise that resolves when module is loaded
     */
    function loadModule(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.async = false; // Preserve load order
            script.onload = () => {
                console.log(`✓ Loaded module: ${src}`);
                resolve();
            };
            script.onerror = () => {
                console.error(`✗ Failed to load module: ${src}`);
                reject(new Error(`Failed to load module: ${src}`));
            };
            document.head.appendChild(script);
        });
    }
    
    /**
     * Load all modules sequentially
     * @returns {Promise} Promise that resolves when all modules are loaded
     */
    async function loadAllModules() {
        console.log('🔄 Starting Data Ingestion Module System...');
        
        try {
            // Load shared modules first so utilities are ready for ingestion scripts
            for (const script of SHARED_MODULES) {
                await loadModule(script);
            }

            // Load ingestion modules sequentially to maintain dependencies
            for (const module of MODULES) {
                await loadModule(BASE_PATH + module);
            }
            
            console.log('✅ All Data Ingestion modules loaded successfully');
            return true;
            
        } catch (error) {
            console.error('❌ Error loading Data Ingestion modules:', error);
            throw error;
        }
    }
    
    /**
     * Initialize the application after all modules are loaded
     */
    function initializeApplication() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', startApplication);
        } else {
            startApplication();
        }
    }
    
    /**
     * Start the application
     */
    function startApplication() {
        try {
            // Check if initializeDataIngestion is available
            if (typeof window.initializeDataIngestion === 'function') {
                window.initializeDataIngestion();
                console.log('🚀 Data Ingestion application started successfully');
            } else {
                throw new Error('initializeDataIngestion function not found');
            }
        } catch (error) {
            console.error('❌ Error starting Data Ingestion application:', error);
            
            // Show user-friendly error message
            if (typeof window.showNotification === 'function') {
                window.showNotification('Failed to start application: ' + error.message, 'error');
            } else {
                alert('Failed to start Data Ingestion application. Please check the console for details.');
            }
        }
    }
    
    // Start the loading process
    loadAllModules()
        .then(() => {
            initializeApplication();
        })
        .catch((error) => {
            console.error('❌ Critical error loading Data Ingestion system:', error);
            alert('Failed to load Data Ingestion system. Please refresh the page and try again.');
        });
    
})();
