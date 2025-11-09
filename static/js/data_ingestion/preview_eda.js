/**
 * preview_eda.js
 * ==============
 * 
 * Data Preview and Exploratory Data Analysis Module
 * 
 * Purpose:
 *  - Data preview modal and visualization
 *  - Enhanced data quality reporting
 *  - EDA workflow management
 *  - Statistical analysis preview
 *  - Data loading and success handling
 *  - Pr                })
            })
            .catch(error => {
                console.error('EDA preparation error:', error);
                global.DI.utilities.notifications.showNotification('Failed to prepare dataset for EDA', 'error');
            })
            .finally(() => {
                global.DI.utilities.notifications.showLoading(false);
            });
        }switching and navigation
 * 
 * Features:
 *  - Interactive data preview tables
 *  - Comprehensive quality reports
 *  - Statistical summaries
 *  - EDA workflow integration
 *  - Model training pipeline access
 *  - Export functionality
 *  - Real-time data analysis
 * 
 * Dependencies:
 *  - showLoading (shared/notifications.js)
 *  - showNotification (shared/notifications.js)
 *  - generateQualityReportHTML (utilities_formatting.js)
 *  - generateDataPreviewHTML (utilities_formatting.js)
 *  - initializePreviewTables (utilities_formatting.js)
 *  - goToDataSources (data_sources.js)
 */

(function(global) {
    'use strict';
    
    // Initialize namespace
    global.DI = global.DI || {};
    global.DI.preview = global.DI.preview || {};
    
    // Note: Preview state is managed through global.DI.state from globals module

        function displayDataPreview(preview, sourceId) {
            global.DI.state.currentPreviewData = preview;
            global.DI.state.currentSourceId = sourceId;
            const sampleRows = preview.shape ? preview.shape[0] : (preview.sample_data ? preview.sample_data.length : 'N/A');
            const cols = preview.shape ? preview.shape[1] : (preview.columns ? preview.columns.length : 'N/A');
            const estimatedTotal = preview.estimated_total_rows && preview.estimated_total_rows > sampleRows ? preview.estimated_total_rows : null;

            let html = '<div class="data-stats">';
            html += '<div class="data-stat">';
            html += `<div class="data-stat-value">${sampleRows}</div>`;
            html += '<div class="data-stat-label">Sample Rows</div>';
            html += '</div>';
            if (estimatedTotal) {
                html += '<div class="data-stat">';
                html += `<div class="data-stat-value">${estimatedTotal.toLocaleString()}</div>`;
                html += '<div class="data-stat-label">Estimated Total Rows</div>';
                html += '</div>';
            }
            html += '<div class="data-stat">';
            html += `<div class="data-stat-value">${cols}</div>`;
            html += '<div class="data-stat-label">Columns</div>';
            html += '</div>';
            html += '</div>';
            
            html += displayTestPreview(preview);
            
            document.getElementById('dataPreviewContent').innerHTML = html;
            document.getElementById('dataPreviewModal').style.display = 'block';
            initializePreviewTables();
        }

        // [MODULE preview_eda] displayEnhancedDataPreview -> target: preview_eda.js
        function displayEnhancedDataPreview(preview, qualityReport, sourceId) {
            global.DI.state.currentPreviewData = preview;
            global.DI.state.currentSourceId = sourceId;
            
            // Create enhanced tabbed interface with text analysis
            let html = `
                <div class="enhanced-preview-container">
                    <!-- Tab Navigation -->
                    <div class="preview-tabs">
                        <button class="preview-tab-btn active" data-tab="quality" onclick="switchPreviewTab('quality')">
                            <i class="fas fa-chart-bar"></i> Data Quality Report
                        </button>
                        <button class="preview-tab-btn" data-tab="text" onclick="switchPreviewTab('text')">
                            <i class="fas fa-font"></i> Text Analysis
                        </button>
                        <button class="preview-tab-btn" data-tab="recommendations" onclick="switchPreviewTab('recommendations')">
                            <i class="fas fa-lightbulb"></i> Recommendations
                        </button>
                        <button class="preview-tab-btn" data-tab="preview" onclick="switchPreviewTab('preview')">
                            <i class="fas fa-table"></i> Data Preview
                        </button>
                    </div>

                    <!-- Quality Report Tab -->
                    <div class="preview-tab-content active" id="qualityTabContent">
                        ${generateQualityReportHTML(qualityReport)}
                    </div>

                    <!-- Text Analysis Tab -->
                    <div class="preview-tab-content" id="textTabContent">
                        ${generateTextAnalysisHTML(qualityReport)}
                    </div>

                    <!-- Recommendations Tab -->
                    <div class="preview-tab-content" id="recommendationsTabContent">
                        ${generateRecommendationsHTML(qualityReport.recommendations || [])}
                    </div>

                    <!-- Data Preview Tab -->
                    <div class="preview-tab-content" id="previewTabContent">
                        ${generateDataPreviewHTML(qualityReport.sample_preview)}
                    </div>
                </div>
            `;
            
            document.getElementById('dataPreviewContent').innerHTML = html;
            document.getElementById('dataPreviewModal').style.display = 'block';
            initializePreviewTables();
        }

        // [MODULE preview_eda] switchPreviewTab -> target: preview_eda.js
        function switchPreviewTab(tabName) {
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
            }
        }

        // ---------- Preview Helpers ----------
        // [MODULE preview_eda] displayConnectionTestResult -> target: preview_eda.js
        function displayConnectionTestResult(elementId, result) {
            const element = document.getElementById(elementId);
            if (!element) return;
            if (result.success) {
                element.innerHTML = `
                    <div class="connection-test-result success">
                        <h6><i class="fas fa-check-circle"></i> Connection Successful</h6>
                        <p>${result.message}</p>
                        ${result.preview ? displayTestPreview(result.preview) : ''}
                    </div>`;
            } else {
                element.innerHTML = `
                    <div class="connection-test-result error">
                        <h6><i class="fas fa-exclamation-circle"></i> Connection Failed</h6>
                        <p>${result.error}</p>
                    </div>`;
            }
            element.style.display = 'block';
            initializePreviewTables();
        }

        // [MODULE preview_eda] displayTestPreview -> target: preview_eda.js
        function displayTestPreview(preview) {
            if (!preview || !preview.columns) return '';
            const tableId = 'previewTable_' + Math.random().toString(36).substr(2,6);
            let html = `<div class="mt-2"><strong>Preview:</strong>
                <div class="table-responsive">
                <table id="${tableId}" data-preview-table="1" class="table table-sm table-striped table-bordered">
                <thead><tr>`;
            preview.columns.forEach(col => { html += `<th>${col}</th>`; });
            html += '</tr></thead><tbody>';
            (preview.sample_data || []).forEach(row => {
                html += '<tr>';
                preview.columns.forEach(col => {
                    const value = row[col];
                    html += `<td>${value !== null && value !== undefined ? value : '<em>null</em>'}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody></table></div></div>';
            return html;
        }

        // [MODULE preview_eda] generateDataPreviewHTML -> target: preview_eda.js
        function generateDataPreviewHTML(samplePreview) {
            return `
                <div class="data-preview-section">
                    <h4><i class="fas fa-table"></i> Sample Data</h4>
                    ${displayTestPreview(samplePreview)}
                </div>`;
        }

        // [MODULE preview_eda] generateMissingValuesChart -> target: preview_eda.js
        function generateMissingValuesChart(columnDetails) {
            const columnsWithMissing = (columnDetails || [])
                .filter(col => col.null_percentage > 0)
                .sort((a, b) => b.null_percentage - a.null_percentage)
                .slice(0, 10);
            if (!columnsWithMissing.length) {
                return '<div class="no-missing-values"><i class="fas fa-check-circle"></i> No missing values found in any column!</div>';
            }
            return `<div class="missing-values-bars">${columnsWithMissing.map(col => `
                <div class=\"missing-bar-row\">
                    <div class=\"missing-bar-label\">${col.name}</div>
                    <div class=\"missing-bar-container\"><div class=\"missing-bar\" style=\"width: ${col.null_percentage}%\"></div></div>
                    <div class=\"missing-bar-value\">${col.null_percentage.toFixed(1)}%</div>
                </div>`).join('')}</div>`;
        }

        // [MODULE preview_eda] generateQualityReportHTML -> target: preview_eda.js
        function generateQualityReportHTML(qualityReport) {
            const metadata = qualityReport.basic_metadata;
            const quality = qualityReport.quality_metrics;
            const issues = qualityReport.potential_issues || [];
            const textSummary = qualityReport.text_analysis_summary || {};
            const recommendations = qualityReport.recommendations || [];
            
            let html = `<div class=\"quality-report\">`;
            
            // Basic Metadata Section
            html += `<div class=\"quality-section\"><h4><i class=\"fas fa-info-circle\"></i> Basic Metadata</h4><div class=\"metadata-grid\">`;
            
            // Sample Rows with total rows info
            html += `<div class=\"metadata-card\">
                <div class=\"metadata-value\">${metadata.sample_rows.toLocaleString()}</div>
                <div class=\"metadata-label\">Sample Rows</div>
                ${metadata.estimated_total_rows && metadata.estimated_total_rows > 0 ? 
                    `<div class="total-rows-info">Total: ${metadata.estimated_total_rows.toLocaleString()} rows</div>` : 
                    ''}
            </div>`;
            
            if (metadata.estimated_total_rows && metadata.estimated_total_rows !== metadata.sample_rows) {
                html += `<div class=\"metadata-card\"><div class=\"metadata-value\">${metadata.estimated_total_rows.toLocaleString()}</div><div class=\"metadata-label\">Estimated Total</div></div>`;
            }
            html += `<div class=\"metadata-card\"><div class=\"metadata-value\">${metadata.total_columns}</div><div class=\"metadata-label\">Columns</div></div>`;
            if (metadata.file_size_bytes) {
                html += `<div class=\"metadata-card\"><div class=\"metadata-value\">${formatFileSize(metadata.file_size_bytes)}</div><div class=\"metadata-label\">File Size</div></div>`;
            }
            html += `<div class=\"metadata-card\"><div class=\"metadata-value\">${formatFileSize(metadata.memory_usage_bytes)}</div><div class=\"metadata-label\">Memory Usage</div></div>`;
            html += '</div>';
            
            // Data Types Section
            html += '<div class="data-types-section"><h5><i class="fas fa-tags"></i> Data Types</h5><div class="data-types-grid">';
            html += Object.entries(metadata.data_types).map(([dtype, columns]) => {
                // Clean dtype for CSS class (remove special characters, convert to lowercase)
                const dtypeClass = dtype.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
                return `<div class="dtype-card">
                    <div class="dtype-name">
                        <span class="dtype-badge dtype-${dtypeClass}">${dtype}</span>
                    </div>
                    <div class="dtype-count">${columns.length} column${columns.length !== 1 ? 's' : ''}</div>
                    <div class="dtype-columns">${columns.slice(0,5).join(', ')}${columns.length>5?' and '+(columns.length-5)+' more...':''}</div>
                </div>`;
            }).join('');
            html += '</div></div></div>';
            
            // Data Quality Metrics Section
            html += `<div class=\"quality-section\"><h4><i class=\"fas fa-chart-line\"></i> Data Quality Metrics</h4>`;
            html += `<div class=\"quality-summary\">`;
            
            // Overall Quality Score (NEW)
            if (qualityReport.overall_quality_score !== undefined) {
                const scoreClass = qualityReport.overall_quality_score >= 80 ? 'excellent' : 
                                 qualityReport.overall_quality_score >= 60 ? 'good' : 'needs-attention';
                html += `<div class=\"quality-stat ${scoreClass}\"><div class=\"quality-stat-value\">${qualityReport.overall_quality_score.toFixed(0)}%</div><div class=\"quality-stat-label\">Overall Quality Score</div></div>`;
            }
            
            html += `<div class=\"quality-stat ${quality.overall_completeness > 90 ? 'excellent' : quality.overall_completeness > 70 ? 'good' : 'needs-attention'}\"><div class=\"quality-stat-value\">${quality.overall_completeness.toFixed(1)}%</div><div class=\"quality-stat-label\">Data Completeness</div></div>`;
            html += `<div class=\"quality-stat\"><div class=\"quality-stat-value\">${quality.columns_with_missing}</div><div class=\"quality-stat-label\">Columns with Missing Values</div></div>`;
            html += `<div class=\"quality-stat ${quality.high_cardinality_columns > 0 ? 'warning' : 'good'}\"><div class=\"quality-stat-value\">${quality.high_cardinality_columns}</div><div class=\"quality-stat-label\">High Cardinality Columns</div></div></div>`;
            
            // Missing Values Chart
            html += `<div class=\"missing-values-chart\"><h5><i class=\"fas fa-chart-bar\"></i> Missing Values by Column</h5>${generateMissingValuesChart(quality.column_details)}</div>`;
            
            // Enhanced Column Details with Text Analysis
            html += generateEnhancedColumnDetailsHTML(quality.column_details);
            
            html += '</div>';
            
            // Potential Issues Section
            if (issues.length) {
                html += `<div class=\"quality-section\"><h4><i class=\"fas fa-exclamation-triangle\"></i> Potential Issues</h4><div class=\"issues-list\">`;
                html += issues.map(issue => `
                    <div class=\"issue-item issue-${issue.type}\"><div class=\"issue-icon\"><i class=\"fas fa-${issue.type === 'warning' ? 'exclamation-triangle' : issue.type === 'info' ? 'info-circle' : 'times-circle'}\"></i></div><div class=\"issue-content\"><div class=\"issue-message\">${issue.message}</div>${issue.column ? `<div class=\"issue-column\">Column: ${issue.column}</div>` : ''}</div></div>`).join('');
                html += '</div></div>';
            }
            html += '</div>';
            return html;
        }

        // [MODULE preview_eda] generateTextAnalysisHTML -> target: preview_eda.js
        function generateTextAnalysisHTML(qualityReport) {
            const textSummary = qualityReport.text_analysis_summary;
            const columnDetails = qualityReport.quality_metrics.column_details;
            const textColumns = columnDetails.filter(col => col.data_category === 'text');
            
            let html = `<div class="quality-section text-analysis-section">
                <h4><i class="fas fa-font"></i> Text Column Analysis</h4>
                <div class="text-analysis-summary">
                    <div class="text-stat-grid">
                        <div class="text-stat-card">
                            <div class="text-stat-value">${textSummary.total_text_columns}</div>
                            <div class="text-stat-label">Text Columns</div>
                        </div>
                        <div class="text-stat-card nlp-card">
                            <div class="text-stat-value">${textSummary.free_text_columns}</div>
                            <div class="text-stat-label">NLP Candidates</div>
                        </div>
                        <div class="text-stat-card categorical-card">
                            <div class="text-stat-value">${textSummary.categorical_text_columns}</div>
                            <div class="text-stat-label">Categorical Text</div>
                        </div>
                    </div>
                </div>`;
            
            if (textColumns.length > 0) {
                html += `<div class="text-columns-details">
                    <h5><i class="fas fa-list"></i> Text Column Details</h5>
                    <div class="text-columns-grid">`;
                
                textColumns.forEach(col => {
                    const isNLP = col.text_category === 'free_text' || col.text_category === 'descriptive_text';
                    const isCategorical = col.text_category === 'categorical';
                    const cardClass = isNLP ? 'text-column-card nlp-candidate' : 
                                     isCategorical ? 'text-column-card categorical-text' : 
                                     'text-column-card mixed-text';
                    const patterns = Array.isArray(col.text_patterns) ? col.text_patterns : [];
                    const flags = Array.isArray(col.text_quality_flags) ? col.text_quality_flags : [];
                    const patternList = patterns.map(pattern => {
                        const pct = pattern.percentage != null ? ` (${pattern.percentage.toFixed(1)}%)` : '';
                        const count = pattern.count != null ? ` - ${pattern.count}` : '';
                        return `<li><strong>${pattern.type}</strong>${pct}${count}</li>`;
                    }).join('');
                    const flagBadges = flags.map(flag => `<span class="quality-flag-badge">${flag}</span>`).join('');
                    
                    html += `<div class="${cardClass}">
                        <div class="text-column-header">
                            <h6><i class="fas fa-font"></i> ${col.name}</h6>
                            <span class="text-category-badge">${col.text_category || 'text'}</span>
                        </div>
                        <div class="text-column-metrics">
                            <div class="text-metric">
                                <span class="metric-label">Avg Length:</span>
                                <span class="metric-value">${col.avg_text_length || 0} chars</span>
                            </div>
                            <div class="text-metric">
                                <span class="metric-label">Range:</span>
                                <span class="metric-value">${col.min_text_length || 0}-${col.max_text_length || 0}</span>
                            </div>
                            <div class="text-metric">
                                <span class="metric-label">Unique Values:</span>
                                <span class="metric-value">${col.unique_count}</span>
                            </div>
                        </div>
                        ${isNLP ? '<div class="text-recommendation"><i class="fas fa-brain"></i> Suitable for NLP analysis, sentiment analysis, or topic modeling</div>' : ''}
                        ${isCategorical ? '<div class="text-recommendation"><i class="fas fa-tags"></i> Good candidate for label encoding or one-hot encoding</div>' : ''}
                        ${patternList ? `<div class="text-patterns"><i class="fas fa-search"></i> Detected Patterns<ul>${patternList}</ul></div>` : ''}
                        ${flagBadges ? `<div class="text-quality-flags"><i class="fas fa-flag"></i> ${flagBadges}</div>` : ''}
                    </div>`;
                });
                
                html += `</div></div>`;
            }

            const detectedPatterns = Array.isArray(textSummary.detected_patterns) ? textSummary.detected_patterns : [];
            if (detectedPatterns.length > 0) {
                html += `<div class="text-pattern-summary">
                    <h5><i class="fas fa-search"></i> Detected Text Patterns</h5>
                    <div class="pattern-grid">${detectedPatterns.map(pattern => `
                        <div class="pattern-card">
                            <div class="pattern-header">
                                <strong>${pattern.pattern_type}</strong>
                                <span class="pattern-count">${pattern.total_count || 0} matches</span>
                            </div>
                            ${Array.isArray(pattern.columns) && pattern.columns.length ? `<div class="pattern-columns"><strong>Columns:</strong> ${pattern.columns.join(', ')}</div>` : ''}
                        </div>
                    `).join('')}</div>
                </div>`;
            }

            const textQualityFlags = Array.isArray(textSummary.text_quality_flags) ? textSummary.text_quality_flags : [];
            if (textQualityFlags.length > 0) {
                html += `<div class="text-quality-alerts">
                    <h5><i class="fas fa-flag"></i> Text Quality Alerts</h5>
                    <div class="quality-flag-grid">${textQualityFlags.map(entry => `
                        <div class="quality-flag-card">
                            <div class="quality-flag-header"><strong>${entry.flag}</strong></div>
                            ${Array.isArray(entry.columns) && entry.columns.length ? `<div class="quality-flag-columns">Columns: ${entry.columns.join(', ')}</div>` : ''}
                        </div>
                    `).join('')}</div>
                </div>`;
            }

            const piiColumns = Array.isArray(textSummary.pii_columns) ? textSummary.pii_columns : [];
            if (piiColumns.length > 0) {
                html += `<div class="pii-warning">
                    <h5><i class="fas fa-user-shield"></i> Sensitive Data Detected</h5>
                    <p>The following columns may contain personally identifiable information. Apply masking or access controls before sharing:</p>
                    <div class="pii-columns">${piiColumns.join(', ')}</div>
                </div>`;
            }
            
            html += `</div>`;
            return html;
        }

        // [MODULE preview_eda] generateRecommendationsHTML -> target: preview_eda.js
        function generateRecommendationsHTML(recommendations) {
            let html = `<div class="quality-section recommendations-section">
                <h4><i class="fas fa-lightbulb"></i> Data Analysis Recommendations</h4>
                <div class="recommendations-grid">`;
            
            recommendations.forEach((rec, index) => {
                const title = rec.title || `Recommendation ${index + 1}`;
                const description = rec.description || rec;
                const iconClass = getRecommendationIcon(title);
                
                html += `<div class="recommendation-card">
                    <div class="recommendation-header">
                        <i class="${iconClass}"></i>
                        <h6>${title}</h6>
                    </div>
                    <div class="recommendation-content">
                        ${description}
                    </div>
                </div>`;
            });
            
            html += `</div></div>`;
            return html;
        }

        // [MODULE preview_eda] getRecommendationIcon -> target: preview_eda.js
        function getRecommendationIcon(title) {
            const iconMap = {
                'NLP': 'fas fa-brain',
                'Text': 'fas fa-font',
                'Missing': 'fas fa-exclamation-triangle',
                'Outlier': 'fas fa-chart-line',
                'Quality': 'fas fa-check-circle',
                'Feature': 'fas fa-cogs',
                'Multilingual': 'fas fa-globe',
                'Categorical': 'fas fa-tags',
                'Data': 'fas fa-database'
            };
            
            for (const [key, icon] of Object.entries(iconMap)) {
                if (title.toLowerCase().includes(key.toLowerCase())) {
                    return icon;
                }
            }
            return 'fas fa-lightbulb';
        }

        // [MODULE preview_eda] generateEnhancedColumnDetailsHTML -> target: preview_eda.js
        function generateEnhancedColumnDetailsHTML(columnDetails) {
            let html = `<div class="column-details-section">
                <h5><i class="fas fa-columns"></i> Column Details</h5>
                <div class="column-details-table-container">
                    <table class="column-details-table">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Data Type</th>
                                <th>Category</th>
                                <th>Completeness</th>
                                <th>Uniqueness</th>
                                <th>Text Info</th>
                                <th>Memory</th>
                            </tr>
                        </thead>
                        <tbody>`;
            
            columnDetails.forEach(col => {
                const rowClass = col.null_percentage > 30 ? 'high-missing' : 
                               col.null_percentage > 10 ? 'medium-missing' : '';
                
                const categoryBadge = col.data_category ? 
                    `<span class="category-badge category-${col.data_category}">${col.data_category}</span>` : 
                    '<span class="category-badge">unknown</span>';
                
                let textInfo = '-';
                if (col.data_category === 'text') {
                    const avgLen = col.avg_text_length || 0;
                    const category = col.text_category || 'mixed';
                    textInfo = `<div class="text-info">
                        <span class="text-category-mini">${category}</span>
                        <small>${avgLen.toFixed(0)} avg chars</small>
                    </div>`;
                }
                
                html += `<tr class="${rowClass}">
                    <td class="column-name">${col.name}</td>
                    <td class="dtype-badge dtype-${col.dtype.split('(')[0]}">${col.dtype}</td>
                    <td class="category-cell">${categoryBadge}</td>
                    <td class="completeness-cell">
                        <span class="completeness-percentage ${col.null_percentage > 30 ? 'poor' : col.null_percentage > 10 ? 'fair' : 'good'}">
                            ${(100 - col.null_percentage).toFixed(1)}%
                        </span>
                        ${col.null_percentage > 0 ? `<small>(${col.null_count} missing)</small>` : ''}
                    </td>
                    <td class="uniqueness-cell">
                        <span class="uniqueness-percentage ${col.unique_percentage > 95 ? 'very-high' : col.unique_percentage > 50 ? 'high' : 'normal'}">
                            ${col.unique_percentage.toFixed(1)}%
                        </span>
                        <small>(${col.unique_count} unique)</small>
                    </td>
                    <td class="text-info-cell">${textInfo}</td>
                    <td class="memory-cell">${formatFileSize(col.memory_usage)}</td>
                </tr>`;
            });
            
            html += `</tbody></table></div></div>`;
            return html;
        }

        

        // [MODULE preview_eda] closeDataPreviewModal -> target: preview_eda.js
        function closeDataPreviewModal() {
            const modal = document.getElementById('dataPreviewModal');
            if (modal) modal.style.display = 'none';
            global.DI.state.currentPreviewData = null;
            global.DI.state.currentSourceId = null;
        }

           // Launch ML workflow canvas from preview modal
       // [MODULE preview_eda] moveToEDA -> target: preview_eda.js
        function moveToEDA() {
            console.log('moveToEDA called');
            console.log('Current source ID:', global.DI?.state?.currentSourceId);
            
            let sourceId = global.DI?.state?.currentSourceId;
            
            // If no source ID in global state, try to prompt user or find an alternative
            if (!sourceId) {
                console.warn('moveToEDA: No currentSourceId available in global state');
                
                // Try to get it from URL parameters or other sources
                const urlParams = new URLSearchParams(window.location.search);
                sourceId = urlParams.get('source_id');
                
                if (!sourceId) {
                    // As a last resort, ask the user
                    sourceId = prompt('Please enter the dataset ID to proceed to EDA:');
                    
                    if (!sourceId) {
                        global.DI.utilities.notifications.showNotification('No dataset selected', 'error');
                        return;
                    }
                }
            }

            // Show confirmation modal with ML workflow information
            const confirmed = confirm(
                `Open "${global.DI.state?.currentPreviewData?.name || 'Selected Dataset'}" in ML Workflow?\n\n` +
                `This will:\n` +
                `• Launch the ML Workflow canvas\n` +
                `• Surface data quality insights\n` +
                `• Recommend feature engineering steps\n` +
                `• Prepare the dataset for downstream modeling\n\n` +
                `Continue to ML Workflow?`
            );

            if (!confirmed) {
                console.log('moveToEDA: User cancelled');
                return;
            }

            console.log('moveToEDA: Proceeding with navigation to source:', sourceId);
            global.DI.utilities.notifications.showLoading(true, 'Navigating to ML Workflow...', 'Opening ML Workflow canvas...');
            
            // Close the preview modal
            try {
                closeDataPreviewModal();
            } catch (e) {
                console.warn('Error closing preview modal:', e);
            }
            
            // Navigate to ML workflow canvas
            const workflowUrl = `/ml-workflow?source_id=${sourceId}`;
            console.log('Navigating to ML Workflow URL:', workflowUrl);
            window.location.href = workflowUrl;
        }

        // Make moveToEDA globally accessible
        window.moveToEDA = moveToEDA;

        // Move to ML Workflow from data source list (without opening modal)
        // [MODULE preview_eda] moveToEDAFromList -> target: preview_eda.js
        function moveToEDAFromList(sourceId, sourceName) {
            console.log('moveToEDAFromList called with:', sourceId, sourceName);
            
            if (!sourceId) {
                console.error('moveToEDAFromList: Invalid data source');
                global.DI.utilities.notifications.showNotification('Invalid data source', 'error');
                return;
            }

            // Show confirmation modal with ML workflow information
            const confirmed = confirm(
                `Open "${sourceName || 'Selected Dataset'}" in ML Workflow?\n\n` +
                `This will:\n` +
                `• Launch the ML Workflow canvas\n` +
                `• Surface data quality insights\n` +
                `• Recommend feature engineering steps\n` +
                `• Prepare the dataset for downstream modeling\n\n` +
                `Continue to ML Workflow?`
            );

            if (!confirmed) {
                console.log('moveToEDAFromList: User cancelled');
                return;
            }

            console.log('moveToEDAFromList: Proceeding with navigation to source:', sourceId);
            global.DI.utilities.notifications.showLoading(true, 'Navigating to ML Workflow...', 'Opening ML Workflow canvas...');
            
            // Navigate to ML workflow canvas
            const workflowUrl = `/ml-workflow?source_id=${sourceId}`;
            console.log('Navigating to ML Workflow URL:', workflowUrl);
            window.location.href = workflowUrl;
            
            // Refresh data sources to show updated status
            try {
                refreshDataSources();
            } catch (e) {
                // Ignore refresh errors as we're navigating away
                console.warn('Error refreshing data sources:', e);
            }
        }

        // Make moveToEDAFromList globally accessible
        window.moveToEDAFromList = moveToEDAFromList;

// ENHANCED: Load full dataset with better user feedback and next steps
// [MODULE preview_eda] loadFullDataset -> target: preview_eda.js
function loadFullDataset() {
    if (!global.DI.state.currentSourceId) {
        global.DI.utilities.notifications.showNotification('No dataset selected', 'error');
        return;
    }

    global.DI.utilities.notifications.showLoading(true, 'Loading Dataset...', 'Loading full dataset for ML workflow...');
    
    fetch(`/data/api/sources/${global.DI.state.currentSourceId}/load`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            max_rows: 50000  // Reasonable limit for initial loading
        })
    })
    .then(response => response.json())
    .then(result => {
        global.DI.utilities.notifications.showLoading(false);
        
        if (result.success) {
            global.DI.utilities.notifications.showNotification(result.message, 'success');
            closeDataPreviewModal();
            
            // ENHANCED: Show what happens next
            displayDataLoadedSuccess(result.data_summary, global.DI.state.currentSourceId);
            
        } else {
            global.DI.utilities.notifications.showNotification(result.error || 'Failed to load dataset', 'error');
        }
    })
    .catch(error => {
        global.DI.utilities.notifications.showLoading(false);
        global.DI.utilities.notifications.showNotification('Network error: ' + error.message, 'error');
    });
}

// NEW: Display success information and next steps after data loading
// [MODULE preview_eda] displayDataLoadedSuccess -> target: preview_eda.js
function displayDataLoadedSuccess(dataSummary, sourceId) {
    // Create a modal or section showing what was loaded and next steps
    const successModal = document.createElement('div');
    successModal.className = 'xml-modal';
    successModal.id = 'dataLoadedModal';
    successModal.innerHTML = `
        <div class="xml-modal-content">
            <div class="xml-modal-header">
                <h3><i class="fas fa-check-circle text-success"></i> Dataset Loaded Successfully</h3>
                <button class="xml-modal-close" onclick="closeDataLoadedModal()">&times;</button>
            </div>
            <div class="xml-modal-body">
                <div class="data-stats mb-3">
                    <div class="data-stat">
                        <div class="data-stat-value">${dataSummary.shape[0].toLocaleString()}</div>
                        <div class="data-stat-label">Rows Loaded</div>
                    </div>
                    <div class="data-stat">
                        <div class="data-stat-value">${dataSummary.shape[1]}</div>
                        <div class="data-stat-label">Columns</div>
                    </div>
                    <div class="data-stat">
                        <div class="data-stat-value">${dataSummary.memory_usage_mb} MB</div>
                        <div class="data-stat-label">Memory Usage</div>
                    </div>
                </div>
                
                <div class="alert alert-info">
                    <h5><i class="fas fa-info-circle"></i> What's Next?</h5>
                    <p>Your dataset is now loaded in memory and ready for ML workflows. Choose your next step:</p>
                </div>
                
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h6><i class="fas fa-chart-line"></i> Exploratory Data Analysis</h6>
                                <p class="text-muted">Analyze data distributions, correlations, and patterns</p>
                                <button class="btn btn-sm btn-primary" onclick="startEDA('${sourceId}')">
                                    Start EDA
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h6><i class="fas fa-cogs"></i> Feature Engineering</h6>
                                <p class="text-muted">Transform and prepare features for modeling</p>
                                <button class="btn btn-sm btn-primary" onclick="startFeatureEngineering('${sourceId}')">
                                    Prepare Features
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h6><i class="fas fa-robot"></i> Model Training</h6>
                                <p class="text-muted">Train machine learning models</p>
                                <button class="btn btn-sm btn-primary" onclick="startModelTraining('${sourceId}')">
                                    Train Models
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6 mb-3">
                        <div class="card h-100">
                            <div class="card-body">
                                <h6><i class="fas fa-download"></i> Export Data</h6>
                                <p class="text-muted">Export processed data for external tools</p>
                                <button class="btn btn-sm btn-secondary" onclick="exportProcessedData('${sourceId}')">
                                    Export
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(successModal);
    successModal.style.display = 'block';
}

// Helper functions for next steps (these would connect to your ML workflow components)
// [MODULE preview_eda] closeDataLoadedModal -> target: preview_eda.js
function closeDataLoadedModal() {
    const modal = document.getElementById('dataLoadedModal');
    if (modal) {
        modal.remove();
    }
}

// Initialize DataTables for any newly added preview tables
// (moved) initializePreviewTables now in Utilities section

// [MODULE preview_eda] startEDA -> target: preview_eda.js
function startEDA(sourceId) {
    console.log('startEDA called with sourceId:', sourceId);
    
    if (!sourceId) {
        console.error('startEDA: No sourceId provided');
        global.DI.utilities.notifications.showNotification('No dataset ID provided', 'error');
        return;
    }

    console.log('startEDA: Proceeding with ML Workflow navigation');
    global.DI.utilities.notifications.showLoading(true, 'Navigating to ML Workflow...', 'Opening ML Workflow canvas...');
    
    // Close any open modals
    try {
        closeDataLoadedModal();
    } catch (e) {
        console.warn('Error closing modal:', e);
    }
    
    // Navigate to ML workflow canvas
    const workflowUrl = `/ml-workflow?source_id=${sourceId}`;
    console.log('Navigating to ML Workflow URL:', workflowUrl);
    window.location.href = workflowUrl;
}

// [MODULE preview_eda] startFeatureEngineering -> target: preview_eda.js
function startFeatureEngineering(sourceId) {
    global.DI.utilities.notifications.showNotification('Feature engineering coming soon! Dataset ID: ' + sourceId, 'info');
    closeDataLoadedModal();
    // TODO: Navigate to feature engineering interface
}

// [MODULE preview_eda] startModelTraining -> target: preview_eda.js
function startModelTraining(sourceId) {
    global.DI.utilities.notifications.showNotification('Model training coming soon! Dataset ID: ' + sourceId, 'info');
    closeDataLoadedModal();
    // TODO: Navigate to model training interface
}

// [MODULE preview_eda] exportProcessedData -> target: preview_eda.js (calls export module)
function exportProcessedData(sourceId) {
    console.log('exportProcessedData called with sourceId:', sourceId);
    try {
        // Show export options modal
        showExportModal(sourceId, 'source');
    } catch (error) {
        console.error('Error in exportProcessedData:', error);
        global.DI.utilities.notifications.showNotification('Error opening export dialog: ' + error.message, 'error');
    }
}

// Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('dataPreviewModal');
                // [MODULE preview_eda] window click handler to close data preview modal -> target: preview_eda.js
                if (event.target == modal) {
                closeDataPreviewModal();
            }
        }
// === MODULE: DATA PREVIEW & EDA (target: preview_eda.js) [END] ===   

// Export functions to global scope for HTML onclick handlers
global.displayDataPreview = displayDataPreview;
global.displayEnhancedDataPreview = displayEnhancedDataPreview;
global.switchPreviewTab = switchPreviewTab;
global.displayConnectionTestResult = displayConnectionTestResult;
global.closeDataPreviewModal = closeDataPreviewModal;
global.moveToEDA = moveToEDA;
global.moveToEDAFromList = moveToEDAFromList;
global.loadFullDataset = loadFullDataset;
global.closeDataLoadedModal = closeDataLoadedModal;
global.startEDA = startEDA;
global.startFeatureEngineering = startFeatureEngineering;
global.startModelTraining = startModelTraining;
global.exportProcessedData = exportProcessedData;

// Add a simple test function for debugging
global.testEDANavigation = function() {
    console.log('Testing ML Workflow navigation...');
    const testSourceId = 'test123';
    const workflowUrl = `/ml-workflow?source_id=${testSourceId}`;
    console.log('Test ML Workflow URL:', workflowUrl);
    if (confirm('Navigate to ML Workflow test page?')) {
        window.location.href = workflowUrl;
    }
};

})(window);
