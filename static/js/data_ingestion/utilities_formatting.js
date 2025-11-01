/**
 * utilities_formatting.js
 * =======================
 * 
 * Data Formatting and Display Utilities Module
 * 
 * Purpose:
 *  - Data formatting and presentation utilities
 *  - Preview table generation and styling
 *  - File size and data formatting
 *  - Loading indicators and timers
 *  - Data quality report generation
 *  - Metadata display formatting
 * 
 * Features:
 *  - Human-readable file size formatting
 *  - Interactive preview table generation
 *  - Data quality visualization
 *  - Performance timing utilities
 *  - Loading overlay management
 *  - Responsive data presentation
 *  - Statistical summary formatting
 * 
 * Dependencies: Minimal (only DOM manipulation and basic utilities)
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.utilities = global.DI.utilities || {};
    global.DI.utilities.formatting = global.DI.utilities.formatting || {};
    
    // Note: Timing state is managed through global.DI.state from globals module
    
    /**
     * Format file size in human-readable format
     * @param {number} bytes - Size in bytes
     * @returns {string} Formatted file size (e.g., "1.5 MB")
     */
    function formatFileSize(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // [MODULE utilities_formatting] formatBytes -> target: utilities_formatting.js
        function formatBytes(bytes) {
            if (!bytes || bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function parseTimestampInput(value) {
            if (value instanceof Date) {
                return Number.isNaN(value.getTime()) ? null : value;
            }

            if (typeof value === 'number') {
                const fromNumber = new Date(value);
                return Number.isNaN(fromNumber.getTime()) ? null : fromNumber;
            }

            if (typeof value === 'string') {
                let normalized = value.trim();
                if (!normalized) return null;

                // Numeric strings (epoch milliseconds/seconds)
                if (/^-?\d+$/.test(normalized)) {
                    const numeric = Number(normalized);
                    const msValue = normalized.length >= 13 ? numeric : numeric * 1000;
                    const numericDate = new Date(msValue);
                    return Number.isNaN(numericDate.getTime()) ? null : numericDate;
                }

                if (normalized.includes(' ') && !normalized.includes('T')) {
                    normalized = normalized.replace(' ', 'T');
                }

                const hasTimeComponent = normalized.includes('T');
                const hasTimezone = /(Z|z|[+-]\d{2}:?\d{2})$/.test(normalized);

                if (!hasTimezone) {
                    if (hasTimeComponent) {
                        normalized = normalized + 'Z';
                    } else {
                        normalized = normalized + 'T00:00:00Z';
                    }
                }

                let parsed = new Date(normalized);
                if (Number.isNaN(parsed.getTime())) {
                    parsed = new Date(value);
                }

                return Number.isNaN(parsed.getTime()) ? null : parsed;
            }

            return null;
        }

        function formatTimestampForDisplay(value, options = {}) {
            const {
                includeTimeZone = false,
                fallback = 'N/A',
                hour12 = false
            } = options;

            const date = parseTimestampInput(value);
            if (!date) return fallback;

            const formatOptions = {
                year: 'numeric',
                month: 'short',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12
            };

            if (includeTimeZone) {
                formatOptions.timeZoneName = 'short';
            }

            try {
                return new Intl.DateTimeFormat(undefined, formatOptions).format(date);
            } catch (err) {
                console.warn('Failed to format timestamp', err);
                return date.toLocaleString();
            }
        }

        // ---------- Data Preview helpers ----------
        // [MODULE utilities_formatting] initializePreviewTables -> target: utilities_formatting.js
        function initializePreviewTables() {
            if (typeof $ === 'undefined' || !$.fn.DataTable) {
                return; // DataTables not loaded
            }
            document.querySelectorAll('table[data-preview-table]')
                .forEach(tbl => {
                    if (tbl.dataset.enhanced) return;
                    $(tbl).DataTable({
                        pageLength: 10,
                        lengthMenu: [5,10,25,50,100],
                        ordering: true,
                        searching: true,
                        responsive: true
                    });
                    tbl.dataset.enhanced = '1';
                });
        }

    
        // [MODULE utilities_formatting] showLoading -> target: utilities_formatting.js
        function showLoading(show, title = 'Processing...', message = 'Please wait...', showTimer = true) {
            const overlay = document.getElementById('loading-overlay');
            const titleEl = document.getElementById('loading-title');
            const messageEl = document.getElementById('loading-message');
            const timerDisplay = document.getElementById('timer-display');
            
            if (show) {
                // Stop any existing timer first
                if (showTimer) {
                    stopTimer();
                }
                
                if (titleEl) titleEl.textContent = title;
                if (messageEl) messageEl.textContent = message;
                if (overlay) overlay.style.display = 'flex';
                
                // Handle timer display
                if (timerDisplay) {
                    if (showTimer) {
                        timerDisplay.style.display = 'inline';
                        timerDisplay.textContent = '00:00';
                        startTimer();
                    } else {
                        timerDisplay.style.display = 'none';
                    }
                }
            } else {
                if (showTimer) {
                    // Always show timer for at least 1 second to ensure visibility
                    const minDisplayTime = 1000; // 1 second minimum
                    const elapsedTime = global.DI.state.startTime ? (Date.now() - global.DI.state.startTime) : minDisplayTime;
                    
                    if (elapsedTime < minDisplayTime) {
                        // Wait until minimum time has passed
                        setTimeout(() => {
                            stopTimer();
                            if (overlay) overlay.style.display = 'none';
                        }, minDisplayTime - elapsedTime);
                    } else {
                        stopTimer();
                        if (overlay) overlay.style.display = 'none';
                    }
                } else {
                    if (overlay) overlay.style.display = 'none';
                }
            }
        }

        function startTimer() {
            global.DI.state.startTime = Date.now();
            
            // Clear any existing timer first
            if (global.DI.state.queryTimer) {
                clearInterval(global.DI.state.queryTimer);
            }
            
            // Update immediately first, then set interval for every 100ms for smoother updates
            updateTimer();
            global.DI.state.queryTimer = setInterval(updateTimer, 100);
        }

        function stopTimer() {
            if (global.DI.state.queryTimer) {
                clearInterval(global.DI.state.queryTimer);
                global.DI.state.queryTimer = null;
            }
            global.DI.state.startTime = null;
        }

        function updateTimer() {
            if (global.DI.state.startTime) {
                const elapsed = Math.floor((Date.now() - global.DI.state.startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                const display = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                
                const timerDisplay = document.getElementById('timer-display');
                if (timerDisplay) {
                    timerDisplay.textContent = display;
                }
            }
        }

// Export functions to global scope for HTML onclick handlers
global.DI.utilities.formatting.formatFileSize = formatFileSize;
global.DI.utilities.formatting.formatBytes = formatBytes;
global.DI.utilities.formatting.initializePreviewTables = initializePreviewTables;
global.DI.utilities.formatting.formatTimestampForDisplay = formatTimestampForDisplay;
global.DI.utilities.formatting.parseTimestampInput = parseTimestampInput;
global.DI.utilities.formatting.startTimer = startTimer;
global.DI.utilities.formatting.stopTimer = stopTimer;
global.DI.utilities.formatting.updateTimer = updateTimer;

// Also export to global for backward compatibility
global.formatFileSize = formatFileSize;
global.formatBytes = formatBytes;
global.initializePreviewTables = initializePreviewTables;
global.formatTimestampForDisplay = formatTimestampForDisplay;
global.parseTimestampInput = parseTimestampInput;
global.startTimer = startTimer;
global.stopTimer = stopTimer;
global.updateTimer = updateTimer;

})(window);
