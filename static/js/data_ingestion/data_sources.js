/**
 * data_sources.js
 * ===============
 * 
 * Data Source Management Module
 * 
 * Purpose:
 *  - Data source listing and organization
 *  - Search and filtering capabilities
 *  - Bulk operations and selection
 *  - Data source metadata display
 *  - Navigation and routing
 *  - Source lifecycle management
 * 
 * Features:
 *  - Real-time search and filtering
 *  - Multiple sorting options
 *  - Bulk selection and deletion
 *  - Source preview and loading
 *  - Metadata visualization
 *  - Keyboard shortcuts for power users
 *  - Responsive data source cards
 * 
 * Dependencies:
 *  - showNotification (shared/notifications.js)
 *  - showLoading (shared/notifications.js)
 *  - displayDataPreview (preview_eda.js)
 *  - moveToEDAFromList (preview_eda.js)
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.sources = global.DI.sources || {};

    const formatTimestamp = (value, options = {}) => {
        const formatter = global.DI?.utilities?.formatting?.formatTimestampForDisplay;
        if (typeof formatter === 'function') {
            return formatter(value, options);
        }

        const fallback = options.fallback !== undefined ? options.fallback : 'N/A';
        if (!value) return fallback;

        const date = value instanceof Date ? value : new Date(value);
        if (Number.isNaN(date.getTime())) return fallback;

        return date.toLocaleString();
    };

    // Helper: attach auth headers if token present, otherwise include cookies
    function getAuthFetchOptions(method = 'GET', body = undefined, extraHeaders = {}) {
        const headers = Object.assign({}, extraHeaders);
        try {
            // Prefer explicit bearer token from localStorage (if your auth flow stores it there)
            const token = localStorage.getItem('access_token') || localStorage.getItem('token');
            if (token) {
                headers['Authorization'] = token.toLowerCase().startsWith('bearer ') ? token : ('Bearer ' + token);
            }
        } catch (e) {
            // localStorage may be unavailable in some contexts
        }

        const opts = { method: method, headers: headers };
        // If a body is supplied, add it (assume caller provides appropriate Content-Type)
        if (body !== undefined) opts.body = body;

        // If no Authorization header provided, include cookies (for cookie-based auth)
        if (!headers['Authorization']) {
            opts.credentials = 'include';
        }

        return opts;
    }
    
    /**
     * Setup search and filter functionality for data sources
     * Initializes event listeners for search input, filters, and controls
     */
    function setupDataSourceSearch() {
        const searchBtn = document.getElementById('dsSearchBtn');
        const clearBtn = document.getElementById('dsClearSearchBtn');
        const searchInput = document.getElementById('dsSearchInput');
        const categorySel = document.getElementById('dsCategoryFilter');
        const limitInput = document.getElementById('dsLimitInput');
        const orderSelect = document.getElementById('dsOrderSelect');
        if (!searchBtn || !searchInput) return; // Ensure search button and input are available

        let dsSearchTimer = null;

        function performSearch() {
            const q = searchInput.value.trim();
            const category = categorySel ? categorySel.value.trim() : '';
            const limit = limitInput ? limitInput.value : 50;
            const order = orderSelect ? orderSelect.value : 'created_desc';
            const params = new URLSearchParams();
            if (q) params.append('q', q);
            if (category) params.append('category', category);
            if (limit) params.append('limit', limit);
            if (order) params.append('order', order);
            
            fetch('/data/api/sources/search?' + params.toString(), getAuthFetchOptions())
                .then(r => r.json())
                .then(res => {
                    if (res.success) {
                        // Store filtered results as the current view so ordering operates on the active set
                        const results = res.results || [];
                        window.currentDataSources = results;
                        displayDataSourcesList(results);
                        const countEl = document.getElementById('dsResultCount');
                        if (countEl) {
                            countEl.style.display = 'inline-block';
                            countEl.textContent = (res.count || 0) + ' result' + (res.count === 1 ? '' : 's');
                        }
                        if (clearBtn) clearBtn.style.display = (q || category) ? 'inline-block' : 'none';
                    } else {
                        global.DI.utilities.notifications.showNotification('Search failed: ' + (res.message || 'Unknown error'), 'error');
                        console.error('Search failed:', res);
                    }
                })
                .catch(err => {
                    console.error('Search error', err);
                    global.DI.utilities.notifications.showNotification('Network error performing search', 'error');
                });
        }

        searchBtn.onclick = performSearch;
        
        // Instant search with debounce on typing
        if (searchInput) {
            searchInput.addEventListener('keydown', e => { 
                if (e.key === 'Enter') { 
                    e.preventDefault(); 
                    performSearch(); 
                }
            });
            searchInput.addEventListener('input', () => {
                if (dsSearchTimer) clearTimeout(dsSearchTimer);
                dsSearchTimer = setTimeout(() => performSearch(), 300); // 300ms debounce
            });
        }
        
        // Add event listeners for all controls
        if (categorySel) categorySel.addEventListener('change', performSearch);
        if (limitInput) limitInput.addEventListener('change', performSearch);
        if (orderSelect) orderSelect.addEventListener('change', performSearch);
        
        if (clearBtn) clearBtn.onclick = () => {
            searchInput.value = '';
            if (categorySel) categorySel.value = '';
            if (limitInput) limitInput.value = 50;
            if (orderSelect) orderSelect.value = 'created_desc';
            const countEl = document.getElementById('dsResultCount');
            if (countEl) countEl.style.display = 'none';
            clearBtn.style.display = 'none';
            refreshDataSources();
        };
    }

// === MODULE: DATA SOURCES MANAGEMENT (target: data_sources.js) [BEGIN] ===
    // Data Sources Management
    // [MODULE data_sources] refreshDataSources -> target: data_sources.js
    function refreshDataSources() {
        // Show loading indicator
        const loadingIndicator = document.getElementById('dataSourcesList');
        if (loadingIndicator) {
            loadingIndicator.innerHTML = '<div class="text-center p-4"><i class="fas fa-spinner fa-spin"></i> Loading data sources...</div>';
        }
        
        return fetch('/data/api/sources')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(result => {
                if (result.success) {
                    // Store data sources globally for sync functionality
                    const sources = result.sources || [];
                    window.currentDataSources = sources;
                    
                    // Store user scope and try to get username for permission checks
                    window.currentUserScope = result.scope || 'database_sources';
                    
                    // Try to get current username from the page or make a simple request
                    if (!window.currentUsername) {
                        // We can extract username from any source that belongs to current user
                        if (result.scope === 'user_scoped' && sources.length > 0) {
                            const userSources = sources.filter(s => (s.metadata && s.metadata.created_by) || s.created_by);
                            if (userSources.length > 0) {
                                window.currentUsername = userSources[0].metadata?.created_by || userSources[0].created_by;
                            }
                        }
                    }
                    
                    displayDataSourcesList(sources);
                    // Populate category filter from visible sources to avoid confusing users
                    populateCategoryFilter(sources);
                    
                    // Check for notifications after loading data sources
                    checkForNotifications();
                    
                    // Show success message if there are sources
                    if (sources.length > 0) {
                        global.DI.utilities.notifications.showNotification(`There is ${sources.length} data source(s) available`, 'success');
                    }
                    
                    return sources; // Return the sources for chaining
                } else {
                    const errorMsg = result.error || 'Failed to load data sources';
                    global.DI.utilities.notifications.showNotification(errorMsg, 'error');
                    throw new Error(errorMsg);
                }
            })
            .catch(error => {
                console.error('Error loading data sources:', error);
                const errorMsg = error.message || 'Network error loading data sources';
                global.DI.utilities.notifications.showNotification(errorMsg, 'error');
                
                // Show error in the data sources list area
                if (loadingIndicator) {
                    loadingIndicator.innerHTML = `
                        <div class="alert alert-danger text-center">
                            <i class="fas fa-exclamation-triangle"></i> 
                            Failed to load data sources: ${errorMsg}
                            <br><button class="btn btn-sm btn-primary mt-2" onclick="refreshDataSources()">Try Again</button>
                        </div>
                    `;
                }
                
                throw error; // Re-throw to allow caller to handle
            });
    }

    // [MODULE data_sources] displayDataSourcesList -> target: data_sources.js
    function displayDataSourcesList(sources) {
            // Ensure ordering is applied before rendering when sources are passed directly
            try { applyOrderingToCurrentSources(sources); } catch (e) {}
            const container = document.getElementById('dataSourcesList');
            
            if (sources.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-4">
                        <i class="fas fa-database" style="font-size: 3rem; opacity: 0.3; margin-bottom: 1rem;"></i>
                        <h5>No Data Sources</h5>
                        <p class="text-muted">Upload files or connect to databases to get started.</p>
                    </div>
                `;
                return;
            }

            // Add bulk action controls if there are sources
            let bulkActionsHtml = `
                <div class="bulk-actions-bar" id="bulkActionsBar" style="display: none;">
                    <div class="bulk-selection-header">
                        <input type="checkbox" class="select-all-checkbox" id="selectAllCheckbox" onchange="toggleSelectAll()">
                        <label for="selectAllCheckbox" style="cursor: pointer; margin: 0;">
                            <i class="fas fa-check-square"></i> Select All
                        </label>
                        <small class="text-muted ms-auto" style="font-size: 0.75rem;">
                            <i class="fas fa-keyboard"></i> 
                            Shortcuts: Ctrl+A (select all), Del (delete), Esc (clear)
                        </small>
                    </div>
                    <div class="bulk-actions-controls">
                        <div class="bulk-selection-info">
                            <i class="fas fa-info-circle"></i>
                            <span id="selectedCount">0</span> selected
                        </div>
                        <div class="bulk-actions-buttons">
                            <button class="btn-bulk-action btn-bulk-delete" onclick="bulkDeleteSelected()" id="bulkDeleteBtn" disabled>
                                <i class="fas fa-trash"></i> Delete Selected
                            </button>
                            <button class="btn-bulk-action btn-bulk-clear" onclick="clearSelection()">
                                <i class="fas fa-times"></i> Clear Selection
                            </button>
                        </div>
                    </div>
                </div>
            `;

            container.innerHTML = bulkActionsHtml + sources.map(source => {
                // Map the raw type (for icons) and display type (for display)
                const rawType = source.type || source.source_type?.toLowerCase() || 'unknown';
                const displayType = source.source_type || source.type || 'Unknown';
                
                let typeIcon = {
                    'unknown': 'fas fa-question',
                    'text': 'fas fa-file-alt',
                    'txt': 'fas fa-file-alt',
                    'csv': 'fas fa-file-csv',
                    'xls': 'fas fa-file-xls',
                    'parquet': 'fas fa-file-parquet',
                    'json': 'fas fa-file-code', 
                    'xlsx': 'fas fa-file-excel',
                    'file': 'fas fa-file-alt',
                    'database': 'fas fa-database', 
                    'postgresql': 'fas fa-database',
                    'mysql': 'fas fa-database',
                    'sqlite': 'fas fa-database',
                    'api': 'fas fa-exchange-alt',
                    'cloud': 'fas fa-cloud',
                    's3': 'fas fa-cloud',
                    'gcs': 'fas fa-cloud',
                    'google_sheets': 'fas fa-cloud',
                    'scraped': 'fas fa-globe',
                    'snowflake': 'fas fa-snowflake',
                    'connector': 'fas fa-plug'
                }[rawType] || 'fas fa-question';

                // Special handling for cloud connectors that might not have proper source_type
                if (source.metadata && source.metadata.connector_type && 
                    ['s3', 'gcs', 'google_sheets'].includes(source.metadata.connector_type)) {
                    typeIcon = 'fas fa-cloud';
                }

                const category = (source.category || rawType || '').toLowerCase();
                const categoryBadge = category ? `<span class="category-badge ${category}">${category}</span>` : '';

                // Get source owner (prefer direct created_by field)
                const sourceOwner = source.created_by || source.metadata?.created_by || 'unknown';
                const safeSourceName = source.name ? source.name.replace(/'/g, "\\'") : '';
                const createdSourceTs = source.created_at || source.metadata?.upload_timestamp || source.config?.upload_timestamp || source.source_metadata?.upload?.timestamp;
                const createdStamp = formatTimestamp(createdSourceTs, { fallback: '' });
                const lastUsedStamp = formatTimestamp(source.last_accessed, { fallback: '' });
                
                // Check if current user can delete this source
                const currentUserScope = window.currentUserScope || 'user_scoped'; // Set by refreshDataSources
                const currentUser = window.currentUsername || 'unknown'; // We'll set this when loading data
                
                // For database_sources scope, allow deletion since there's no proper auth
                // For user_scoped or admin_full, check ownership
                const canDelete = currentUserScope === 'database_sources' || 
                                currentUserScope === 'admin_full' || 
                                sourceOwner === currentUser;

                const deleteActions = canDelete ? `
                    <div class="delete-action-row mt-2">
                        <button class="btn btn-gradient-danger-prominent" onclick="deleteDataSource('${source.id}')" title="Delete source">
                            <i class="fas fa-trash"></i> Delete Source
                        </button>
                    </div>` : '';

        const driftWarn = source.metadata && source.metadata.drift_warning ? `<div class="alert alert-warning p-1 mt-2 mb-0 small"><i class='fas fa-exclamation-triangle'></i> ${source.metadata.drift_warning}</div>` : '';
        
        // Check if this is a newly uploaded source
        const newlyUploadedId = sessionStorage.getItem('newly_uploaded_source');
        const isNewlyUploaded = newlyUploadedId === source.source_id; // Keep using source_id for highlighting since uploads set this
        const newlyUploadedClass = isNewlyUploaded ? ' newly-uploaded' : '';
        
        return `
                    <div class="data-source-card${newlyUploadedClass}" id="source-card-${source.id}">
                        <div class="data-source-header">
                            <div class="data-source-checkbox-container">
                                ${canDelete ? `<input type="checkbox" class="source-checkbox mt-1" data-source-id="${source.id}" onchange="updateBulkActions(); toggleCardSelection(this)">` : '<div style="width: 20px;"></div>'}
                                <div class="data-source-content">
                                    <span class="data-source-type ${rawType}">
                                        <i class="${typeIcon}"></i> ${displayType}
                                    </span>
                                    ${categoryBadge}
                                    <h5 class="mt-2 mb-1">${source.name}</h5>
                                    <small class="text-muted source-id-badge">Source ID: ${source.source_id}</small>
                                    ${createdStamp ? `<br><small class="text-muted">Created: ${createdStamp}</small>` : ''}
                                    ${sourceOwner && sourceOwner !== 'unknown' ? `<br><small class="text-muted">By: ${sourceOwner}</small>` : ''}
                                    ${lastUsedStamp ? `<br><small class="text-muted">Last used: ${lastUsedStamp}</small>` : ''}
                                </div>
                            </div>
                            <!-- Right-side actions column: small buttons above, EDA below -->
                            <div class="source-actions-column">
                                ${deleteActions}
                                <div class="eda-action-row mt-2">
                                    <button class="btn btn-gradient-eda-prominent" onclick="moveToEDAFromList('${source.source_id}', '${safeSourceName}')" title="Open ML Workflow Canvas">
                                        <i class="fas fa-project-diagram"></i> Open ML Workflow
                                    </button>
                                </div>
                            </div>
                        </div>
                        
            ${source.metadata ? displaySourceMetadata(source.metadata) : ''}
            ${driftWarn}
                    </div>
                `;
            }).join('');
        }

        function populateCategoryFilter(sources) {
            try {
                const categorySel = document.getElementById('dsCategoryFilter');
                if (!categorySel) return;
                // Collect unique categories from visible sources
                const cats = new Set();
                sources.forEach(s => {
                    const rawSourceType = s.type || (s.source_type ? s.source_type.toLowerCase() : '');
                    const c = (s.category || rawSourceType || '').toString().toLowerCase();
                    if (c) cats.add(c);
                });

                // Remember current selection
                const current = categorySel.value;

                // Rebuild options: keep 'All Categories' first
                categorySel.innerHTML = '';
                const allOpt = document.createElement('option'); allOpt.value = ''; allOpt.textContent = 'All Categories';
                categorySel.appendChild(allOpt);

                Array.from(cats).sort().forEach(cat => {
                    const opt = document.createElement('option'); opt.value = cat; opt.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
                    categorySel.appendChild(opt);
                });

                // Restore selection if still present
                if (current) {
                    const found = Array.from(categorySel.options).some(o => o.value === current);
                    if (found) categorySel.value = current;
                }
                // Hide the category control if there's 0 or 1 active category to avoid confusing users
                try {
                    const wrapper = categorySel.closest('.col-md-3') || categorySel.parentElement;
                    if (wrapper) {
                        if (cats.size <= 1) {
                            wrapper.style.display = 'none';
                        } else {
                            wrapper.style.display = '';
                        }
                    }
                } catch (e) {
                    // Non-critical
                }
            } catch (e) {
                console.warn('Failed to populate category filter', e);
            }
        }

        // [MODULE data_sources] previewDataSource -> target: data_sources.js (delegates to preview_eda module)
        function previewDataSource(sourceId) {
            // Show loading without timer
            global.DI.utilities.notifications.showLoading(true, 'Loading Preview...', 'Generating data preview and quality report...', false);

            // Fetch both preview and quality report
            Promise.all([
                fetch(`/ml-workflow/api/data-sources/${sourceId}/preview?sample_size=100`, getAuthFetchOptions()).then(r => r.json()),
                fetch(`/ml-workflow/api/data-sources/${sourceId}/quality-report?sample_size=500`, getAuthFetchOptions()).then(r => r.json())
            ])
                .then(([previewResult, qualityResult]) => {
                global.DI.utilities.notifications.showLoading(false);

                if (previewResult.success && qualityResult.success) {
                    displayEnhancedDataPreview(previewResult.preview, qualityResult.quality_report, sourceId);
                } else {
                    const error = previewResult.error || qualityResult.error || 'Failed to load preview';
                    global.DI.utilities.notifications.showNotification(error, 'error');
                }
            })
            .catch(error => {
                global.DI.utilities.notifications.showLoading(false);
                global.DI.utilities.notifications.showNotification('Network error: ' + error.message, 'error');
            });
        }

        // [MODULE data_sources] loadDataSource -> target: data_sources.js
        function loadDataSource(sourceId) {
            if (confirm('Load this dataset for ML workflow? This may take a moment for large datasets.')) {
                loadFullDatasetById(sourceId);
            }
        }

        // [MODULE data_sources] loadFullDatasetById -> target: data_sources.js
        function loadFullDatasetById(sourceId) {
            global.DI.utilities.notifications.showLoading(true, 'Loading Dataset...', 'Preparing data for ML workflow...');
            
            fetch(`/data/api/sources/${sourceId}/load`, getAuthFetchOptions('POST', JSON.stringify({ max_rows: 50000 }), {'Content-Type': 'application/json'}))
            .then(response => response.json())
            .then(result => {
                global.DI.utilities.notifications.showLoading(false);
                
                if (result.success) {
                    global.DI.utilities.notifications.showNotification(result.message, 'success');
                    console.log('Dataset loaded for ML:', result.data_summary);
                    // Here you would navigate to the ML workflow interface
                } else {
                    global.DI.utilities.notifications.showNotification(result.error || 'Failed to load dataset', 'error');
                }
            })
            .catch(error => {
                global.DI.utilities.notifications.showLoading(false);
                global.DI.utilities.notifications.showNotification('Network error: ' + error.message, 'error');
            });
        }

        // [MODULE data_sources] deleteDataSource -> target: data_sources.js
        function deleteDataSource(sourceId) {
            console.log('Deleting source:', sourceId);
            if (confirm('Are you sure you want to delete this data source? This action cannot be undone.')) {
                const startTime = Date.now();
                fetch(`/data/api/sources/${sourceId}`, getAuthFetchOptions('DELETE'))
                .then(response => {
                    const elapsed = Date.now() - startTime;
                    console.log(`Delete request completed in ${elapsed}ms, status: ${response.status}`);
                    return response.json();
                })
                .then(result => {
                    console.log('Delete result:', result);
                    if (result.success) {
                        global.DI.utilities.notifications.showNotification(result.message, 'success');
                        refreshDataSources();
                    } else {
                        global.DI.utilities.notifications.showNotification(result.error || 'Failed to delete data source', 'error');
                    }
                })
                .catch(error => {
                    console.error('Delete error:', error);
                    global.DI.utilities.notifications.showNotification('Network error: ' + error.message, 'error');
                });
            }
        }

        // === BULK ACTIONS FUNCTIONALITY ===

        // [MODULE data_sources] bulkDeleteSelected (bulk actions) -> target: data_sources.js
        function bulkDeleteSelected() {
            const checkboxes = document.querySelectorAll('.source-checkbox:checked');
            const sourceIds = Array.from(checkboxes).map(cb => cb.getAttribute('data-source-id'));
            
            if (sourceIds.length === 0) {
                global.DI.utilities.notifications.showNotification('No data sources selected', 'warning');
                return;
            }

            const confirmMessage = `Are you sure you want to delete ${sourceIds.length} data source(s)? This action cannot be undone.`;
            
            if (confirm(confirmMessage)) {
                // Show loading state
                const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
                const originalText = bulkDeleteBtn.innerHTML;
                bulkDeleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
                bulkDeleteBtn.disabled = true;

                fetch('/data/api/sources/bulk-delete', getAuthFetchOptions('POST', JSON.stringify({ source_ids: sourceIds }), {'Content-Type': 'application/json'}))
                .then(response => response.json())
                .then(result => {
                    bulkDeleteBtn.innerHTML = originalText;
                    bulkDeleteBtn.disabled = false;
                    
                    if (result.success) {
                        let message = result.message || `Successfully processed ${sourceIds.length} data sources`;
                        
                        // Show detailed results if there were failures
                        if (result.failed_count > 0) {
                            message += `\n\nFailed deletions:`;
                            result.failed_deletions.forEach(failure => {
                                message += `\n- ${failure.source_name || failure.source_id}: ${failure.error}`;
                            });
                        }
                        
                        if (result.notifications_sent && result.notifications_sent.length > 0) {
                            message += `\n\nNotified users: ${result.notifications_sent.join(', ')}`;
                        }
                        
                        global.DI.utilities.notifications.showNotification(message, result.failed_count > 0 ? 'warning' : 'success');
                        clearSelection();
                        refreshDataSources();
                    } else {
                        global.DI.utilities.notifications.showNotification(result.error || 'Bulk delete failed', 'error');
                    }
                })
                .catch(error => {
                    bulkDeleteBtn.innerHTML = originalText;
                    bulkDeleteBtn.disabled = false;
                    global.DI.utilities.notifications.showNotification('Network error during bulk delete: ' + error.message, 'error');
                });
            }
        }

        // === SELECT ALL FUNCTIONALITY ===

        // [MODULE data_sources] toggleSelectAll -> target: data_sources.js
        function toggleSelectAll() {
            const selectAllCheckbox = document.getElementById('selectAllCheckbox');
            const sourceCheckboxes = document.querySelectorAll('.source-checkbox');
            const shouldSelect = selectAllCheckbox.checked;
            
            sourceCheckboxes.forEach(checkbox => {
                checkbox.checked = shouldSelect;
                toggleCardSelection(checkbox);
            });
            
            updateBulkActions();
            updateSelectAllState();
        }

        // [MODULE data_sources] updateSelectAllState -> target: data_sources.js
        function updateSelectAllState() {
            const selectAllCheckbox = document.getElementById('selectAllCheckbox');
            const sourceCheckboxes = document.querySelectorAll('.source-checkbox');
            const checkedBoxes = document.querySelectorAll('.source-checkbox:checked');
            
            if (!selectAllCheckbox || sourceCheckboxes.length === 0) return;
            
            if (checkedBoxes.length === 0) {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = false;
            } else if (checkedBoxes.length === sourceCheckboxes.length) {
                selectAllCheckbox.checked = true;
                selectAllCheckbox.indeterminate = false;
            } else {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = true;
            }
        }

        // [MODULE data_sources] toggleCardSelection -> target: data_sources.js
        function toggleCardSelection(checkbox) {
            const sourceId = checkbox.getAttribute('data-source-id');
            const card = document.getElementById(`source-card-${sourceId}`);
            
            if (card) {
                if (checkbox.checked) {
                    card.classList.add('selected');
                } else {
                    card.classList.remove('selected');
                }
            }
        }

        // Override the existing updateBulkActions function to include select all state
        // [MODULE data_sources] updateBulkActions -> target: data_sources.js
        function updateBulkActions() {
            const checkboxes = document.querySelectorAll('.source-checkbox:checked');
            const bulkBar = document.getElementById('bulkActionsBar');
            const selectedCount = document.getElementById('selectedCount');
            const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
            
            if (!bulkBar) {
                console.warn('Bulk actions bar not found in DOM');
                return;
            }
            
            if (checkboxes.length > 0) {
                bulkBar.style.display = 'block';
                if (selectedCount) selectedCount.textContent = checkboxes.length;
                if (bulkDeleteBtn) bulkDeleteBtn.disabled = false;
            } else {
                bulkBar.style.display = 'none';
                if (bulkDeleteBtn) bulkDeleteBtn.disabled = true;
            }
            
            // Update select all state
            updateSelectAllState();
        }

        // [MODULE data_sources] clearSelection -> target: data_sources.js
        function clearSelection() {
            const checkboxes = document.querySelectorAll('.source-checkbox');
            const selectAllCheckbox = document.getElementById('selectAllCheckbox');
            
            checkboxes.forEach(cb => {
                cb.checked = false;
                toggleCardSelection(cb);
            });
            
            if (selectAllCheckbox) {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = false;
            }
            
            updateBulkActions();
        }

        // [MODULE data_sources] displaySourceMetadata -> target: data_sources.js (moved from utilities_formatting)
    function displaySourceMetadata(metadata) {
            let html = '<div class="mt-2"><small class="text-muted">';
            if (metadata.file_size) {
                html += `Size: ${(metadata.file_size / 1024 / 1024).toFixed(2)} MB | `;
            }
            if (metadata.preview_columns) {
                html += `Columns: ${metadata.preview_columns.length}`;
            }
            // Remove the "By:" display from metadata since it's already shown in the main card
            html += '</small></div>';
            return html;
        }

        // ---------- Ordering ----------
        // [MODULE data_sources] applyOrderingToCurrentSources -> target: data_sources.js
        function applyOrderingToCurrentSources(sources) {
            const orderSel = document.getElementById('dsOrderSelect');
            if (!orderSel || !Array.isArray(sources)) return;
            const val = orderSel.value;
            const getDate = (s, key) => {
                if (!s) return null;
                if (key === 'created') return new Date(s.created_at || (s.metadata && s.metadata.upload_timestamp) || 0);
                if (key === 'lastsync') return new Date((s.metadata && s.metadata.last_sync) || (s.connection_info && s.connection_info.last_sync) || 0);
                return null;
            };
            sources.sort((a,b) => {
                if (val.startsWith('created')) {
                    const da = getDate(a,'created');
                    const db = getDate(b,'created');
                    const ta = da ? da.getTime() : 0;
                    const tb = db ? db.getTime() : 0;
                    return val === 'created_desc' ? (tb - ta) : (ta - tb);
                }
                if (val.startsWith('lastsync')) {
                    const da = getDate(a,'lastsync');
                    const db = getDate(b,'lastsync');
                    const na = da ? da.getTime() : 0;
                    const nb = db ? db.getTime() : 0;
                    return val === 'lastsync_desc' ? (nb - na) : (na - nb);
                }
                if (val === 'name_asc') {
                    return (a.name || '').localeCompare(b.name || '');
                }
                return 0;
            });
        }

        // [MODULE data_sources] goToDataSources -> target: data_sources.js
        function goToDataSources(highlightSourceId) {
            switchTab('sources');
            if (highlightSourceId) sessionStorage.setItem('newly_uploaded_source', highlightSourceId);
            refreshDataSources();
        }
        // === KEYBOARD SHORTCUTS ===
        
        // Add keyboard shortcuts for bulk actions
    // [MODULE data_sources] keyboard shortcuts for bulk actions -> target: data_sources.js
    document.addEventListener('keydown', function(event) {
            // Only apply shortcuts when on data sources tab and not in input fields
            if (getCurrentActiveTab() !== 'sources' || 
                ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
                return;
            }

            // Ctrl+A or Cmd+A: Select/deselect all
            if ((event.ctrlKey || event.metaKey) && event.key === 'a') {
                event.preventDefault();
                const selectAllCheckbox = document.getElementById('selectAllCheckbox');
                if (selectAllCheckbox && !selectAllCheckbox.disabled) {
                    selectAllCheckbox.checked = !selectAllCheckbox.checked;
                    toggleSelectAll();
                }
            }

            // Delete key: Delete selected items (with confirmation)
            if (event.key === 'Delete') {
                const checkedBoxes = document.querySelectorAll('.source-checkbox:checked');
                if (checkedBoxes.length > 0) {
                    event.preventDefault();
                    bulkDeleteSelected();
                }
            }

            // Escape key: Clear selection
            if (event.key === 'Escape') {
                const checkedBoxes = document.querySelectorAll('.source-checkbox:checked');
                if (checkedBoxes.length > 0) {
                    event.preventDefault();
                    clearSelection();
                }
            }
        });

// Export functions to global scope for HTML onclick handlers
global.setupDataSourceSearch = setupDataSourceSearch;
global.refreshDataSources = refreshDataSources;
global.displayDataSourcesList = displayDataSourcesList;
global.previewDataSource = previewDataSource;
global.loadDataSource = loadDataSource;
global.deleteDataSource = deleteDataSource;
global.bulkDeleteSelected = bulkDeleteSelected;
global.toggleSelectAll = toggleSelectAll;
global.updateBulkActions = updateBulkActions;
global.clearSelection = clearSelection;
global.goToDataSources = goToDataSources;
global.applyOrderingToCurrentSources = applyOrderingToCurrentSources;
global.toggleCardSelection = toggleCardSelection;

})(window);