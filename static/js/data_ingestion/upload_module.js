/**
 * upload_module.js
 * ================
 * 
 * File Upload Handling Module
 * 
 * Purpose:
 *  - Drag and drop file upload functionality
 *  - File selection and validation
 *  - Upload progress tracking
 *  - File information display
 *  - Upload form management
 * 
 * Features:
 *  - Visual drag/drop zones with feedback
 *  - Multiple file format support
 *  - File size validation
 *  - Upload progress indicators
 *  - Error handling and user feedback
 * 
 * Dependencies:
 *  - showNotification (shared/notifications.js)
 *  - showLoading (shared/notifications.js)
 *  - formatFileSize (utilities_formatting.js)
 *  - goToDataSources (data_sources.js)
 *  - previewDataSource (preview_eda.js)
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.upload = global.DI.upload || {};

    const formatTimestamp = (value, options = {}) => {
        const formatter = global.DI?.utilities?.formatting?.formatTimestampForDisplay;
        if (typeof formatter === 'function') {
            return formatter(value, options);
        }

        const fallback = options.fallback !== undefined ? options.fallback : '';
        if (!value) return fallback;

        const date = value instanceof Date ? value : new Date(value);
        if (Number.isNaN(date.getTime())) return fallback;

        return date.toLocaleString();
    };
    
    /**
     * Setup file upload functionality
     * Initializes drag/drop zones and file input handlers
     */
    function setupFileUpload() {
        const dropZone = document.getElementById('uploadDropZone');
        const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');

    // Ensure we attach a single set of stable handlers (prevents duplicates)
    ensureDropZoneHandlers(dropZone);

    // Ensure file input has exactly one change listener
    try { fileInput.removeEventListener('change', handleFileSelect); } catch (e) {}
    fileInput.addEventListener('change', handleFileSelect);

    // Ensure upload button has a single click listener
    try { uploadBtn.removeEventListener('click', uploadFile); } catch (e) {}
    uploadBtn.addEventListener('click', uploadFile);
}

    /**
     * Ensure drop zone event handlers are properly attached
     * Uses global state to maintain stable handler references
     * @param {Element} dropZone - The drop zone element
     */
    function ensureDropZoneHandlers(dropZone) {
        // Create stable handler references once and reuse them
        if (!global.DI.state.dropZoneClickHandler) {
            global.DI.state.dropZoneClickHandler = function(e) { 
                const fi = document.getElementById('fileInput'); 
                if (fi) fi.click(); 
            };
            
            global.DI.state.dropZoneDragOverHandler = function(e) { 
                e.preventDefault(); 
                if (e.currentTarget) e.currentTarget.classList.add('dragover'); 
            };
            
            global.DI.state.dropZoneDragLeaveHandler = function(e) { 
                if (e.currentTarget) e.currentTarget.classList.remove('dragover'); 
            };
            
            global.DI.state.dropZoneDropHandler = function(e) {
                e.preventDefault();
                if (e.currentTarget) e.currentTarget.classList.remove('dragover');
                const files = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files : [];
                if (files.length > 0) {
                    const fileInput = document.getElementById('fileInput');
                    if (fileInput) {
                        try { 
                            fileInput.files = files; 
                        } catch (err) {
                            // Fallback: create a DataTransfer if direct assignment fails (older browsers)
                            try {
                                const dt = new DataTransfer();
                                for (let i = 0; i < files.length; i++) dt.items.add(files[i]);
                                fileInput.files = dt.files;
                            } catch (err2) {}
                        }
                        handleFileSelect();
                    }
                }
            };
        }

        // Re-attach handlers (safe: remove first then add)
        try { dropZone.removeEventListener('click', global.DI.state.dropZoneClickHandler); } catch (e) {}
        try { dropZone.removeEventListener('dragover', global.DI.state.dropZoneDragOverHandler); } catch (e) {}
        try { dropZone.removeEventListener('dragleave', global.DI.state.dropZoneDragLeaveHandler); } catch (e) {}
        try { dropZone.removeEventListener('drop', global.DI.state.dropZoneDropHandler); } catch (e) {}

        dropZone.addEventListener('click', global.DI.state.dropZoneClickHandler);
        dropZone.addEventListener('dragover', global.DI.state.dropZoneDragOverHandler);
        dropZone.addEventListener('dragleave', global.DI.state.dropZoneDragLeaveHandler);
        dropZone.addEventListener('drop', global.DI.state.dropZoneDropHandler);
    }

// FIXED: Handle file selection with better UX
// [MODULE upload_module] handleFileSelect -> target: upload_module.js
function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    
    if (fileInput.files.length > 0) {
        global.DI.state.currentFile = fileInput.files[0];
        uploadBtn.disabled = false;
        
        // Update drop zone to show selected file WITH option to select a different file
        const dropZone = document.getElementById('uploadDropZone');
        dropZone.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-file-alt"></i>
            </div>
            <h5>${global.DI.state.currentFile.name}</h5>
            <p class="text-muted">Size: ${(global.DI.state.currentFile.size / 1024 / 1024).toFixed(2)} MB</p>
            <div class="file-actions mt-2">
                <button type="button" class="btn btn-sm btn-outline-secondary" id="selectDifferentFile">
                    <i class="fas fa-edit"></i> Select Different File
                </button>
            </div>
        `;

        // Ensure a hidden file input exists inside the drop zone (so select button can target it)
        let localInput = document.getElementById('fileInput');
        if (!localInput) {
            localInput = document.createElement('input');
            localInput.type = 'file';
            localInput.id = 'fileInput';
            localInput.style.display = 'none';
            localInput.accept = '.csv,.xlsx,.xls,.parquet,.json,.txt';
            dropZone.appendChild(localInput);
            localInput.addEventListener('change', handleFileSelect);
        }

        // Attach handler for select different file (idempotent)
        const selectBtn = document.getElementById('selectDifferentFile');
        if (selectBtn) {
            selectBtn.onclick = function(evt) {
                evt.stopPropagation();
                evt.preventDefault();
                const fi = document.getElementById('fileInput');
                if (fi) { fi.value = ''; fi.click(); }
            };
        }
        
    } else {
        // No file selected, reset to default state
        resetDropZoneToDefault();
    }
}

// HELPER: Reset drop zone to default state with proper event handling
// [MODULE upload_module] resetDropZoneToDefault -> target: upload_module.js
function resetDropZoneToDefault() {
    const dropZone = document.getElementById('uploadDropZone');
    
    // Reset HTML to default
    dropZone.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
        </div>
        <h5>Drop files here or click to browse</h5>
        <p class="text-muted">Supports CSV, Excel, Parquet, and JSON files</p>
    `;
    
    // Re-attach stable drop zone handlers
    // Ensure a hidden file input exists (some patches replace it)
    if (!document.getElementById('fileInput')) {
        const input = document.createElement('input');
        input.type = 'file';
        input.id = 'fileInput';
        input.style.display = 'none';
        input.accept = '.csv,.xlsx,.xls,.parquet,.json,.txt';
        dropZone.appendChild(input);
        input.addEventListener('change', handleFileSelect);
    }

    ensureDropZoneHandlers(dropZone);
}

        // [MODULE upload_module] uploadFile -> target: upload_module.js
        function uploadFile() {
            if (!global.DI.state.currentFile) {
                global.DI.utilities.notifications.showNotification('Please select a file first', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', global.DI.state.currentFile);
            
            const customName = document.getElementById('customName').value.trim();
            if (customName) {
                formData.append('custom_name', customName);
            }

            global.DI.utilities.notifications.showLoading(true, 'Uploading File...', 'Analyzing file structure and generating preview...');

            fetch('data/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                global.DI.utilities.notifications.showLoading(false);
                
                if (result.success) {
                    // Show completion status like cloud connectors
                    const fileInfo = result.file_info;
                    
                    // Debug: log fileInfo to see available fields
                    console.log('File upload result.file_info:', fileInfo);
                    
                    const uploadedStamp = formatTimestamp(fileInfo.upload_timestamp || fileInfo.uploaded_at, {
                        fallback: '',
                        includeTimeZone: true
                    });

                    const completionMessage = `
                        <div class="ingestion-complete">
                            <h4><i class="fas fa-check-circle"></i> Ingestion Complete</h4>
                            <div class="completion-details">
                                <p><strong>Dataset:</strong> ${fileInfo.display_name || fileInfo.original_filename}</p>
                                <p><strong>Rows:</strong> ${fileInfo.estimated_rows ? fileInfo.estimated_rows.toLocaleString() : (fileInfo.preview_rows ? fileInfo.preview_rows.toLocaleString() : (fileInfo.row_count ? fileInfo.row_count.toLocaleString() : 'N/A'))}</p>
                                <p><strong>Size:</strong> ${fileInfo.file_size_bytes ? formatFileSize(fileInfo.file_size_bytes) : (fileInfo.file_size_mb ? fileInfo.file_size_mb + ' MB' : (fileInfo.file_size ? formatFileSize(fileInfo.file_size) : 'N/A'))}</p>
                                <p><strong>Source ID:</strong> ${fileInfo.source_id}</p>
                                ${uploadedStamp ? `<p><strong>Uploaded:</strong> ${uploadedStamp}</p>` : ''}
                            </div>
                            <div class="completion-actions">
                                <button class="btn-modern btn-primary-modern" onclick="goToDataSources('${fileInfo.source_id}')">
                                    <i class="fas fa-database"></i> View in Data Sources
                                </button>
                                <button class="btn-modern btn-secondary-modern" onclick="window.location.href='/ml-workflow?source_id=${fileInfo.source_id}'">
                                    <i class="fas fa-project-diagram"></i> Open ML Workflow
                                </button>
                            </div>
                        </div>
                    `;
                    
                    const fileInfoContainer = document.getElementById('fileInfoDisplay');
                    fileInfoContainer.innerHTML = completionMessage;
                    fileInfoContainer.style.display = 'block';
                    
                    // Show appropriate notification
                    const notificationType = result.is_duplicate ? 'warning' : 'success';
                    global.DI.utilities.notifications.showNotification(result.message, notificationType);
                    
                    refreshDataSources();
                    
                    // Mark this source as newly uploaded for highlighting
                    sessionStorage.setItem('newly_uploaded_source', fileInfo.source_id);
                } else {
                    global.DI.utilities.notifications.showNotification(result.error || 'Upload failed', 'error');
                }
            })
            .catch(error => {
                global.DI.utilities.notifications.showLoading(false);
                global.DI.utilities.notifications.showNotification('Network error: ' + error.message, 'error');
                console.error('Upload error:', error);
            });
        }

        // [MODULE upload_module] displayFileInfo -> target: upload_module.js
        function displayFileInfo(fileInfo) {
            const display = document.getElementById('fileInfoDisplay');
            
            let html = `
                <div class="file-info-card">
                    <h5><i class="fas fa-info-circle"></i> File Information</h5>
                    <div class="data-stats">
                        <div class="data-stat">
                            <div class="data-stat-value">${fileInfo.file_size_mb}</div>
                            <div class="data-stat-label">MB</div>
                        </div>`;
            
            if (fileInfo.column_count) {
                html += `
                        <div class="data-stat">
                            <div class="data-stat-value">${fileInfo.column_count}</div>
                            <div class="data-stat-label">Columns</div>
                        </div>`;
            }
            
            if (fileInfo.estimated_rows) {
                html += `
                        <div class="data-stat">
                            <div class="data-stat-value">${fileInfo.estimated_rows.toLocaleString()}</div>
                            <div class="data-stat-label">Est. Rows</div>
                        </div>`;
            }
            
            html += `
                    </div>
                    <p><strong>Source ID:</strong> ${fileInfo.source_id}</p>
                    <p><strong>Original Name:</strong> ${fileInfo.original_filename}</p>
                </div>`;
            
            if (fileInfo.preview_data && fileInfo.preview_data.length > 0) {
                html += `
                    <div class="upload-preview-table">
                        <h6>Data Preview (first 100 rows)</h6>
                        <div class="table-responsive">
                            <table id="uploadPreviewTable" class="table table-sm table-striped table-bordered" data-upload-preview="1">
                                <thead>
                                    <tr>`;
                
                fileInfo.columns.forEach(col => {
                    html += `<th>${col}</th>`;
                });
                
                html += `</tr></thead><tbody>`;
                
                fileInfo.preview_data.forEach(row => {
                    html += '<tr>';
                    fileInfo.columns.forEach(col => {
                        const value = row[col];
                        html += `<td>${value !== null && value !== undefined ? value : '<em>null</em>'}</td>`;
                    });
                    html += '</tr>';
                });
                
                html += '</tbody></table></div></div>';
            }
            
            display.innerHTML = html;
            display.style.display = 'block';

            // Initialize DataTables for upload preview (search + pagination)
            if (typeof $ !== 'undefined' && $.fn.DataTable) {
                const tbl = $('#uploadPreviewTable');
                if (tbl.length) {
                    tbl.DataTable({
                        pageLength: 10,
                        lengthMenu: [5,10,25,50,100],
                        ordering: true,
                        searching: true,
                        responsive: true
                    });
                }
            }
        }

// === MODULE: FILE UPLOAD (target: upload_module.js) [END] ===

// [MODULE upload_module] clearUploadForm -> target: upload_module.js
function clearUploadForm() {
    // Replace file input element to ensure browsers clear selection
    const oldInput = document.getElementById('fileInput');
    if (oldInput && oldInput.parentNode) {
        const newInput = oldInput.cloneNode();
        newInput.type = 'file';
        newInput.id = 'fileInput';
        newInput.accept = oldInput.accept || '.csv,.xlsx,.xls,.parquet,.json,.txt';
        newInput.style.display = 'none';
        oldInput.parentNode.replaceChild(newInput, oldInput);
        // Attach change handler to the new input
        newInput.addEventListener('change', handleFileSelect);
    }

    // Reset other form fields and UI
    const custom = document.getElementById('customName'); if (custom) custom.value = '';
    const uploadBtn = document.getElementById('uploadBtn'); if (uploadBtn) uploadBtn.disabled = true;
    const fileInfo = document.getElementById('fileInfoDisplay'); if (fileInfo) fileInfo.style.display = 'none';
    global.DI.state.currentFile = null;

    // Reset drop zone UI and reattach handlers
    resetDropZoneToDefault();

    // Show confirmation
    global.DI.utilities.notifications.showNotification('Upload form cleared', 'info');
}

// Export functions to global scope for HTML onclick handlers
global.setupFileUpload = setupFileUpload;
global.handleFileSelect = handleFileSelect;
global.uploadFile = uploadFile;
global.clearUploadForm = clearUploadForm;

})(window);
