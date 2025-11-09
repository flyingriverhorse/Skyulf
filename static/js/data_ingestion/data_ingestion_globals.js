/**
 * data_ingestion_globals.js
 * =========================
 * 
 * Global shared state for the Data Ingestion Module System
 * 
 * Purpose:
 *  - Contains variables that need to be shared across multiple modules
 *  - Manages global application state
 *  - Handles browser navigation and scroll behavior
 * 
 * Important: This file should be loaded FIRST before all other data ingestion modules
 * 
 * Global Variables:
 *  - currentFile: Currently selected file for processing
 *  - currentPreviewData: Data from file preview operations
 *  - queryTimer: Timer for debounced query operations
 *  - startTime: Timestamp for performance tracking
 *  - currentSourceId: ID of the currently selected data source
 *  - Event handlers: Stable references to prevent duplicate attachments
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    
    // ===================================================================
    // GLOBAL STATE VARIABLES - Shared across all modules
    // ===================================================================
    
    // File and data management
    let currentFile = null;
    let currentPreviewData = null;
    let currentSourceId = null;
    
    // Performance and timing
    let queryTimer = null;
    let startTime = null;
    
    // Event handler references (prevents duplicate attachments)
    let dropZoneClickHandler = null;
    let dropZoneDragOverHandler = null;
    let dropZoneDragLeaveHandler = null;
    let dropZoneDropHandler = null;
    
    // ===================================================================
    // BROWSER BEHAVIOR MANAGEMENT
    // ===================================================================
    
    // Prevent scroll restoration on page refresh for better UX
    if ('scrollRestoration' in history) {
        history.scrollRestoration = 'manual';
    }
    
    // Ensure page starts at top on refresh
    window.addEventListener('beforeunload', function() {
        window.scrollTo(0, 0);
    });
    
    // Handle back/forward navigation scroll behavior
    window.addEventListener('pageshow', function() {
        window.scrollTo(0, 0);
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
    });
    
    // ===================================================================
    // GLOBAL STATE ACCESS (for other modules)
    // ===================================================================
    
    // Export state variables to global namespace for module access
    global.DI.state = {
        get currentFile() { return currentFile; },
        set currentFile(value) { currentFile = value; },
        
        get currentPreviewData() { return currentPreviewData; },
        set currentPreviewData(value) { currentPreviewData = value; },
        
        get currentSourceId() { return currentSourceId; },
        set currentSourceId(value) { currentSourceId = value; },
        
        get queryTimer() { return queryTimer; },
        set queryTimer(value) { queryTimer = value; },
        
        get startTime() { return startTime; },
        set startTime(value) { startTime = value; },
        
        // Event handler getters/setters
        get dropZoneClickHandler() { return dropZoneClickHandler; },
        set dropZoneClickHandler(value) { dropZoneClickHandler = value; },
        
        get dropZoneDragOverHandler() { return dropZoneDragOverHandler; },
        set dropZoneDragOverHandler(value) { dropZoneDragOverHandler = value; },
        
        get dropZoneDragLeaveHandler() { return dropZoneDragLeaveHandler; },
        set dropZoneDragLeaveHandler(value) { dropZoneDragLeaveHandler = value; },
        
        get dropZoneDropHandler() { return dropZoneDropHandler; },
        set dropZoneDropHandler(value) { dropZoneDropHandler = value; }
    };
    
    console.log('✓ Data Ingestion globals initialized');
    
})(window);




    

        

        


        

