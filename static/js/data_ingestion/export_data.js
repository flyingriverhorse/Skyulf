/**
 * export_data.js
 * ==============
 * 
 * Data Export and Download Management Module
 * 
 * Purpose:
 *  - Universal data export functionality
 *  - Export format selection and configuration
 *  - Download progress tracking
 *  - Export preview and validation
 *  - Multi-format support (CSV, JSON, Excel, etc.)
 *  - Export history and management
 * 
 * Features:
 *  - Dynamic export modal generation
 *  - Real-time export preview
 *  - Multiple output formats
 *  - Configurable export options
 *  - Progress tracking and status
 *  - Error handling and recovery
 *  - Export history tracking
 * 
 * Dependencies:
 *  - showLoading (shared/notifications.js)
 *  - showNotification (shared/notifications.js)
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.exporter = global.DI.exporter || {};
    
    /**
     * Show export modal for specified data
     * @param {string} dataId - ID of the data to export
     * @param {string} dataType - Type of data (scraped, source, etc.)
     */
    function showExportModal(dataId, dataType) {
        console.log('showExportModal called with dataId:', dataId, 'dataType:', dataType);
        
        try {
            // First, load available export formats
            fetch('/data/api/export/formats')
                .then(response => {
                    console.log('Formats API response status:', response.status);
                    return response.json();
                })
                .then(result => {
                    console.log('Formats API result:', result);
                    if (result.success) {
                    createExportModal(dataId, dataType, result.formats);
                } else {
                    console.error('Formats API failed:', result);
                    global.DI.utilities.notifications.showNotification('Failed to load export formats', 'error');
                }
            })
            .catch(error => {
                console.error('Error loading export formats:', error);
                global.DI.utilities.notifications.showNotification('Error loading export options: ' + error.message, 'error');
            });
    } catch (error) {
        console.error('Error in showExportModal:', error);
        global.DI.utilities.notifications.showNotification('Error showing export modal: ' + error.message, 'error');
    }
}

// [MODULE export_data] createExportModal -> target: export_data.js
function createExportModal(dataId, dataType, formats) {
    console.log('createExportModal called with:', { dataId, dataType, formats });
    
    try {
        // Create modal HTML
        const modalId = 'exportModal';
        
        // Remove existing modal if present
        const existingModal = document.getElementById(modalId);
        if (existingModal) {
            console.log('Removing existing modal');
            existingModal.remove();
        }
    
        const formatOptions = Object.keys(formats).map(key => {
            const format = formats[key];
            return `
                <div class="format-option" data-format="${key}">
                    <div class="format-header">
                        <i class="fas ${format.icon}"></i>
                        <strong>${format.name}</strong>
                        <span class="format-extension">.${format.extension}</span>
                    </div>
                    <p class="format-description">${format.description}</p>
                </div>
            `;
        }).join('');
        
        // Build export options based on data type
        let exportOptionsHtml = '';
        if (dataType === 'scraped') {
            exportOptionsHtml = `
                <div class="export-options">
                    <h6><i class="fas fa-cog"></i> Export Options</h6>
                    <div class="form-group">
                        <label class="form-label">Export Type</label>
                        <select id="exportType" class="form-control-enhanced">
                            <option value="all">All Tables (Multi-file/Multi-sheet)</option>
                            <option value="combined">Combined Tables</option>
                            <option value="specific">Specific Table</option>
                        </select>
                    </div>
                    
                    <div id="specificTableGroup" style="display: none;">
                        <div class="form-group">
                            <label class="form-label">Select Table</label>
                            <select id="specificTable" class="form-control-enhanced">
                                <!-- Options will be loaded dynamically -->
                            </select>
                        </div>
                    </div>
                </div>
            `;
        } else {
            exportOptionsHtml = `
                <div class="export-options">
                    <h6><i class="fas fa-cog"></i> Export Options</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="form-group">
                                <label class="form-label">Export Size</label>
                                <select id="exportSize" class="form-control-enhanced">
                                    <option value="full">Full Dataset</option>
                                    <option value="sample">Sample (1,000 rows)</option>
                                    <option value="custom">Custom Limit</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-6" id="customLimitGroup" style="display: none;">
                            <div class="form-group">
                                <label class="form-label">Max Rows</label>
                                <input type="number" id="customLimit" class="form-control-enhanced" 
                                       value="10000" min="1" max="100000">
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        const modalHtml = `
            <div id="${modalId}" class="xml-modal">
                <div class="xml-modal-content" style="max-width: 600px;">
                    <div class="xml-modal-header">
                        <h3><i class="fas fa-download"></i> Export Data</h3>
                        <button class="xml-modal-close" onclick="closeExportModal()">&times;</button>
                    </div>
                    <div class="xml-modal-body">
                        <div class="export-formats">
                            <h6><i class="fas fa-file-export"></i> Choose Export Format</h6>
                            <div class="formats-grid">
                                ${formatOptions}
                            </div>
                        </div>
                        
                        ${exportOptionsHtml}
                        
                        <div class="export-preview" id="exportPreview" style="display: none;">
                            <div class="alert alert-info">
                                <i class="fas fa-info-circle"></i>
                                <span id="exportPreviewText">Select a format to see export details</span>
                            </div>
                        </div>
                    </div>
                    <div class="xml-modal-footer">
                        <button class="btn-modern btn-primary-modern" id="startExportBtn" onclick="startExport('${dataId}', '${dataType}')" disabled>
                            <i class="fas fa-download"></i> Start Export
                        </button>
                        <button class="btn-modern btn-secondary-modern" onclick="closeExportModal()">
                            <i class="fas fa-times"></i> Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        console.log('Modal HTML added to body');
        
        // Check if modal is in DOM
        const addedModal = document.getElementById(modalId);
        if (addedModal) {
            console.log('Modal found in DOM and ready to display');
            // Show the modal
            addedModal.style.display = 'flex';
        } else {
            console.error('Modal not found in DOM after insertion');
        }
        
        // Setup event listeners
        setupExportModalListeners(dataId, dataType);
        console.log('Event listeners setup completed');
        
        // If scraped data, load table options
        if (dataType === 'scraped') {
            loadScrapedTableOptions(dataId);
        }
        
        console.log('Export modal created successfully');
        
    } catch (error) {
        console.error('Error in createExportModal:', error);
        showNotification('Error creating export dialog: ' + error.message, 'error');
    }
}

// [MODULE export_data] setupExportModalListeners -> target: export_data.js
function setupExportModalListeners(dataId, dataType) {
    // Format selection
    document.querySelectorAll('.format-option').forEach(option => {
        option.addEventListener('click', () => {
            // Remove previous selection
            document.querySelectorAll('.format-option').forEach(opt => opt.classList.remove('selected'));
            
            // Select this format
            option.classList.add('selected');
            
            // Enable export button
            document.getElementById('startExportBtn').disabled = false;
            
            // Update preview
            const format = option.dataset.format;
            updateExportPreview(format, dataType);
        });
    });
    
    // Export type change (for scraped data)
    const exportType = document.getElementById('exportType');
    if (exportType) {
        exportType.addEventListener('change', () => {
            const specificGroup = document.getElementById('specificTableGroup');
            if (exportType.value === 'specific') {
                specificGroup.style.display = 'block';
            } else {
                specificGroup.style.display = 'none';
            }
        });
    }
    
    // Export size change (for regular data)
    const exportSize = document.getElementById('exportSize');
    if (exportSize) {
        exportSize.addEventListener('change', () => {
            const customGroup = document.getElementById('customLimitGroup');
            if (exportSize.value === 'custom') {
                customGroup.style.display = 'block';
            } else {
                customGroup.style.display = 'none';
            }
        });
    }
}

// [MODULE export_data] loadScrapedTableOptions -> target: export_data.js
function loadScrapedTableOptions(scrapingId) {
    // Load available tables for the scraping ID
    fetch(`/data/api/scraping-history`)
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                const scraping = result.history.find(h => h.scraping_id == scrapingId || h.id == scrapingId);
                if (scraping && scraping.preview_data) {
                    const tableSelect = document.getElementById('specificTable');
                    if (tableSelect) {
                        tableSelect.innerHTML = '';
                        Object.keys(scraping.preview_data).forEach(tableName => {
                            const option = document.createElement('option');
                            option.value = tableName;
                            option.textContent = tableName;
                            tableSelect.appendChild(option);
                        });
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error loading table options:', error);
        });
}

// [MODULE export_data] updateExportPreview -> target: export_data.js
function updateExportPreview(format, dataType) {
    const preview = document.getElementById('exportPreview');
    const previewText = document.getElementById('exportPreviewText');
    
    let message = '';
    switch (format) {
        case 'csv':
            message = 'CSV format: Universal compatibility, great for Excel and data analysis tools';
            break;
        case 'xlsx':
            message = dataType === 'scraped' ? 
                'Excel format: Multiple tables will be exported as separate sheets' :
                'Excel format: Includes formatting and is perfect for business reports';
            break;
        case 'json':
            message = 'JSON format: Ideal for APIs, web applications, and programmatic access';
            break;
        case 'parquet':
            message = 'Parquet format: Optimized for analytics, works great with Python pandas and big data tools';
            break;
    }
    
    previewText.textContent = message;
    preview.style.display = 'block';
}

// [MODULE export_data] startExport -> target: export_data.js
function startExport(dataId, dataType) {
    console.log('startExport called with dataId:', dataId, 'dataType:', dataType);
    console.log('Type of dataId:', typeof dataId);
    
    const selectedFormat = document.querySelector('.format-option.selected');
    if (!selectedFormat) {
        showNotification('Please select an export format', 'warning');
        return;
    }
    
    const format = selectedFormat.dataset.format;
    let exportUrl = '';
    let params = new URLSearchParams();
    
    console.log('Selected format:', format);
    
    // Build URL and parameters based on data type
    if (dataType === 'scraped') {
        exportUrl = `/data/api/export/scraped/${dataId}`;
        params.append('format', format);
        
        console.log('Export URL for scraped data:', exportUrl);
        
        const exportType = document.getElementById('exportType').value;
        if (exportType === 'specific') {
            const tableName = document.getElementById('specificTable').value;
            if (tableName) {
                params.append('table_name', tableName);
                console.log('Exporting specific table:', tableName);
            }
        } else if (exportType === 'combined') {
            params.append('combine_tables', 'true');
            console.log('Exporting combined tables');
        }
        
    } else {
        exportUrl = `/data/api/export/source/${dataId}`;
        params.append('format', format);
        
        const exportSize = document.getElementById('exportSize').value;
        if (exportSize === 'sample') {
            params.append('sample_only', 'true');
        } else if (exportSize === 'custom') {
            const customLimit = document.getElementById('customLimit').value;
            if (customLimit) {
                params.append('max_rows', customLimit);
            }
        }
    }
    
    console.log('Final export URL with params:', `${exportUrl}?${params.toString()}`);
    
    // Show loading
    showNotification('Preparing export...', 'info');
    const exportBtn = document.getElementById('startExportBtn');
    exportBtn.disabled = true;
    exportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Exporting...';
    
    // Create download link
    const downloadUrl = `${exportUrl}?${params.toString()}`;
    
    // Create invisible download link
    const downloadLink = document.createElement('a');
    downloadLink.href = downloadUrl;
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
    
    // Trigger download
    downloadLink.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(downloadLink);
        exportBtn.disabled = false;
        exportBtn.innerHTML = '<i class="fas fa-download"></i> Start Export';
        closeExportModal();
        showNotification('Export started! File will download automatically.', 'success');
    }, 1000);
}

// [MODULE export_data] closeExportModal -> target: export_data.js
function closeExportModal() {
    const modal = document.getElementById('exportModal');
    if (modal) {
        modal.remove();
    }
}

// === END DATA EXPORT FUNCTIONALITY ===

// Export functions to global scope for HTML onclick handlers
global.DI.export = global.DI.export || {};
global.DI.export.showExportModal = showExportModal;
global.DI.export.startExport = startExport;
global.DI.export.closeExportModal = closeExportModal;

// Also export to global for backward compatibility
global.showExportModal = showExportModal;
global.startExport = startExport;
global.closeExportModal = closeExportModal;

})(window);
