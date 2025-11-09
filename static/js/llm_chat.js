// Minimal LLM chat widget behavior
(function () {
    const DEFAULT_API_PATH = '/llm/query';
    const EDA_API_PATH = '/llm/eda/query';
    const DEFAULT_CHAT_KEY = 'global';
    const GLOBAL_HISTORY_MESSAGES = 12;
    const GLOBAL_HISTORY_CHAR_LIMIT = 12000;
    const PREVIEW_SAMPLE_ROW_LIMIT = 25;
    const PREVIEW_SAMPLE_COLUMN_LIMIT = 80;
    const PREVIEW_CELL_CHAR_LIMIT = 120;
    const PREVIEW_SAMPLE_CACHE_KEY_DEFAULT = '__default__';
    const CELL_HISTORY_MESSAGES = 8;
    const CELL_HISTORY_CHAR_LIMIT = 8000;
    const EDA_WORKSPACE_HISTORY_MESSAGES = 10;
    const EDA_WORKSPACE_HISTORY_CHAR_LIMIT = 9000;
    const conversationStore = new Map();
    if (!conversationStore.has(DEFAULT_CHAT_KEY)) {
        conversationStore.set(DEFAULT_CHAT_KEY, []);
    }
    const previewSampleCache = new Map();
    let previewSampleWarmupScheduled = false;
    let previewSampleRefreshScheduled = false;
    let lastPreviewCacheSourceId = null;
    let currentChatKey = DEFAULT_CHAT_KEY;
    // Only load model/provider info when user opens the chat (avoid background requests)
    let modelInfoLoaded = false;
    const MODEL_INFO_TIMEOUT = 7000; // ms - abort provider info fetch after this
    let floatingButtonEnabled = false;
    let floatingButtonElement = null;
    let floatingHintDisplayed = false;
    let currentEdaActiveTabId = null;

    function isEdaWorkspace() {
        try {
            const body = document.body;
            if (!body) return false;
            if (body.classList.contains('eda-container') || body.classList.contains('eda-shell')) return true;
            if (body.dataset && (body.dataset.pageType === 'eda' || body.dataset.workspace === 'eda')) return true;
        } catch (e) {
            // ignore detection errors
        }
        return false;
    }

    function isDataPreviewPage() {
        try {
            if (window.DATA_PREVIEW_CONFIG) return true;
            const body = document.body;
            if (body && (body.dataset?.pageType === 'data-preview' || body.classList.contains('data-preview-page'))) {
                return true;
            }
            if (document.querySelector('.preview-page-container')) {
                return true;
            }
        } catch (e) {
            // ignore detection errors
        }
        return false;
    }

    function isDataPreviewTabActive() {
        if (!isEdaWorkspace()) return false;
        if (currentEdaActiveTabId) {
            return currentEdaActiveTabId === 'data-preview-content';
        }

        try {
            const activeButton = document.querySelector('.eda-tab-button.active[data-bs-target]');
            if (activeButton) {
                const targetId = activeButton.getAttribute('data-bs-target') || '';
                if (targetId.replace('#', '') === 'data-preview-content') {
                    return true;
                }
            }

            const previewPane = document.getElementById('data-preview-content');
            if (previewPane && previewPane.classList.contains('show') && previewPane.classList.contains('active')) {
                return true;
            }

            const previewButton = document.getElementById('data-preview-tab');
            if (previewButton && previewButton.classList.contains('active')) {
                return true;
            }
        } catch (e) {
            // ignore detection errors
        }

        return false;
    }

    function shouldShowFloatingButton() {
        if (!isDataPreviewPage()) {
            return false;
        }

        if (!isEdaWorkspace()) {
            return true;
        }

        return isDataPreviewTabActive();
    }

    function removeFloatingButton() {
        try {
            if (floatingButtonElement) {
                floatingButtonElement.remove();
            }
            floatingButtonElement = null;

            const hint = document.querySelector('.llm-ai-hint');
            if (hint) {
                hint.remove();
            }
        } catch (e) {
            // ignore removal issues
        }
    }

    function syncFloatingButtonVisibility(options = {}) {
        const { showHint = false } = options;
        const shouldEnable = shouldShowFloatingButton();

        if (shouldEnable && !floatingButtonEnabled) {
            floatingButtonEnabled = true;
            const btn = createButton();
            if (btn) {
                btn.style.display = '';
            }
            schedulePreviewSampleWarmup();
            initPreviewSampleObserver();

            if (showHint && !floatingHintDisplayed) {
                setTimeout(() => {
                    if (floatingButtonEnabled && !floatingHintDisplayed) {
                        try {
                            showAiHint();
                        } catch (e) {
                            // ignore hint errors
                        }
                    }
                }, 800);
            }
        } else if (!shouldEnable && floatingButtonEnabled) {
            floatingButtonEnabled = false;
            removeFloatingButton();
        } else if (shouldEnable && floatingButtonEnabled && floatingButtonElement && floatingButtonElement.style.display === 'none') {
            floatingButtonElement.style.display = '';
        }
    }

    function safeTruncate(value, limit = 600) {
        if (!value || typeof value !== 'string') {
            return value;
        }
        return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
    }

    function buildEdaNotebookCondensedSummary(edaContext) {
        if (!edaContext || typeof edaContext !== 'object') {
            return null;
        }
        try {
            const lines = [];
            if (edaContext.summary) {
                lines.push(safeTruncate(String(edaContext.summary), 600));
            }
            if (Array.isArray(edaContext.analysisHighlights) && edaContext.analysisHighlights.length) {
                lines.push('Recent analyses:');
                edaContext.analysisHighlights.slice(0, 4).forEach(item => {
                    if (!item) return;
                    const name = item.analysisName || item.analysisType || 'Analysis';
                    const status = item.status ? ` (${item.status})` : '';
                    const summary = item.summary ? ` — ${safeTruncate(String(item.summary), 320)}` : '';
                    lines.push(`• ${name}${status}${summary}`);
                });
            }
            if (Array.isArray(edaContext.customHighlights) && edaContext.customHighlights.length) {
                lines.push('Recent custom code:');
                edaContext.customHighlights.slice(0, 3).forEach(item => {
                    if (!item) return;
                    const status = item.status ? ` (${item.status})` : '';
                    const summary = item.summary ? ` — ${safeTruncate(String(item.summary), 280)}` : '';
                    lines.push(`• Cell ${item.cellId || '?'}${status}${summary}`);
                });
            }
            return lines.length ? lines.join('\n') : null;
        } catch (summaryError) {
            console.warn('Unable to build condensed EDA summary:', summaryError);
            return null;
        }
    }

    function sanitizePreviewCell(value) {
        if (value === null || value === undefined) return '';
        if (typeof value === 'number' && Number.isFinite(value)) return value.toString();
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        let text;
        if (value instanceof Date) {
            text = value.toISOString();
        } else if (typeof value === 'object') {
            try {
                text = JSON.stringify(value);
            } catch (e) {
                text = String(value);
            }
        } else {
            text = String(value);
        }
        return text.length > PREVIEW_CELL_CHAR_LIMIT ? `${text.slice(0, PREVIEW_CELL_CHAR_LIMIT - 1)}…` : text;
    }

    function normalisePreviewRow(row, columns) {
        if (!Array.isArray(columns) || !columns.length) {
            return [];
        }
        return columns.map((col, index) => {
            if (Array.isArray(row)) {
                return sanitizePreviewCell(row[index]);
            }
            if (row && typeof row === 'object') {
                return sanitizePreviewCell(row[col]);
            }
            return index === 0 ? sanitizePreviewCell(row) : '';
        });
    }

    function categorizeDtype(dtype) {
        if (!dtype) return 'Other';
        const lower = String(dtype).toLowerCase();
        if (/(int|float|decimal|number|double|numeric)/.test(lower)) return 'Numeric';
        if (/(date|time|timestamp|datetime|year|month)/.test(lower)) return 'Datetime';
        if (/(bool)/.test(lower)) return 'Boolean';
        if (/(object|string|category|text|char)/.test(lower)) return 'Categorical';
        return 'Other';
    }

    function summariseTypeBuckets(dtypeMap) {
        if (!dtypeMap || typeof dtypeMap !== 'object') return [];
        const buckets = {
            Numeric: [],
            Categorical: [],
            Datetime: [],
            Boolean: [],
            Other: []
        };
        Object.entries(dtypeMap).forEach(([column, dtype]) => {
            const bucket = categorizeDtype(dtype);
            buckets[bucket].push(column);
        });
        return Object.entries(buckets)
            .filter(([, cols]) => cols.length)
            .map(([bucket, cols]) => {
                const preview = cols.slice(0, 6).join(', ');
                const suffix = cols.length > 6 ? ` (+${cols.length - 6})` : '';
                return `${bucket}: ${preview}${suffix}`;
            });
    }

    function resolvePreviewCacheKey(sourceId) {
        const explicit = sourceId || window?.DATA_PREVIEW_CONFIG?.sourceId;
        if (explicit) return explicit;
        if (typeof window?.sourceId !== 'undefined' && window.sourceId) {
            return window.sourceId;
        }
        return PREVIEW_SAMPLE_CACHE_KEY_DEFAULT;
    }

    function cachePreviewSample(sample, sourceId) {
        if (!sample || typeof sample !== 'object') return;
        const resolvedSource = sample.source_id || sourceId || null;
        const cacheKey = resolvePreviewCacheKey(resolvedSource);

        if (lastPreviewCacheSourceId && resolvedSource && resolvedSource !== lastPreviewCacheSourceId) {
            previewSampleCache.clear();
        }

        previewSampleCache.set(cacheKey, {
            sample,
            timestamp: Date.now()
        });

        if (resolvedSource) {
            lastPreviewCacheSourceId = resolvedSource;
        }
    }

    function getCachedPreviewSample(sourceId) {
        if (!previewSampleCache.size) {
            return null;
        }
        const key = resolvePreviewCacheKey(sourceId);
        const cached = previewSampleCache.get(key);
        if (cached) {
            return cached.sample;
        }
        const iterator = previewSampleCache.values().next();
        return iterator && iterator.value ? iterator.value.sample : null;
    }

    function warmupPreviewSample(force = false) {
        try {
            const sourceHint = window?.DATA_PREVIEW_CONFIG?.sourceId || window?.sourceId || null;
            const rawPreview = getPreviewDataSource();
            if (!rawPreview) {
                return null;
            }
            const sample = buildPreviewSample(rawPreview);
            if (sample) {
                cachePreviewSample(sample, sourceHint);
                return sample;
            }
        } catch (e) {
            console.warn('Failed to prepare preview sample:', e);
        }
        return null;
    }

    function schedulePreviewSampleWarmup() {
        if (!isDataPreviewPage()) return;
        if (previewSampleWarmupScheduled) return;
        previewSampleWarmupScheduled = true;

        const run = () => {
            previewSampleWarmupScheduled = false;
            warmupPreviewSample();
        };

        if (typeof window.requestIdleCallback === 'function') {
            window.requestIdleCallback(() => run(), { timeout: 2000 });
        } else {
            setTimeout(run, 250);
        }
    }

    function schedulePreviewSampleRefresh(timeout = 400) {
        if (!isDataPreviewPage()) return;
        if (previewSampleRefreshScheduled) return;
        previewSampleRefreshScheduled = true;

        const run = () => {
            previewSampleRefreshScheduled = false;
            warmupPreviewSample(true);
        };

        if (typeof window.requestIdleCallback === 'function') {
            window.requestIdleCallback(() => run(), { timeout });
        } else {
            setTimeout(run, timeout);
        }
    }

    function initPreviewSampleObserver() {
        if (!isDataPreviewPage()) return;
        try {
            if (window._llmPreviewSampleObserverInitialized) return;
            const target = document.querySelector('[data-tab="preview"].preview-tab-content')
                || document.getElementById('previewTabContent')
                || document.querySelector('.preview-tab-content');
            if (!target) return;

            const observer = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    if (mutation.type === 'childList' && mutation.addedNodes && mutation.addedNodes.length) {
                        schedulePreviewSampleRefresh(300);
                        break;
                    }
                }
            });

            observer.observe(target, { childList: true, subtree: true });
            window._llmPreviewSampleObserver = observer;
            window._llmPreviewSampleObserverInitialized = true;
        } catch (e) {
            console.warn('Unable to observe preview content for caching:', e);
        }
    }

    function buildPreviewSample(rawPreview) {
        if (!rawPreview || typeof rawPreview !== 'object') {
            return null;
        }

        const allColumns = Array.isArray(rawPreview.columns) ? rawPreview.columns.slice() : [];
        if (!allColumns.length) {
            return null;
        }

        const limitedColumns = allColumns.slice(0, PREVIEW_SAMPLE_COLUMN_LIMIT);

        let rows = rawPreview.data;
        if (!Array.isArray(rows)) rows = rawPreview.rows;
        if (!Array.isArray(rows)) rows = rawPreview.sample_data;
        if (!Array.isArray(rows)) rows = rawPreview.preview_rows;
        if (!Array.isArray(rows) && Array.isArray(rawPreview.first_rows) && Array.isArray(rawPreview.last_rows)) {
            rows = rawPreview.first_rows.concat(rawPreview.last_rows);
        }
        if (!Array.isArray(rows)) rows = rawPreview.records;
        if (!Array.isArray(rows)) rows = rawPreview.samples;
        if (!Array.isArray(rows)) {
            return null;
        }

        const limitedRows = rows.slice(0, PREVIEW_SAMPLE_ROW_LIMIT).map(row => normalisePreviewRow(row, limitedColumns));
        if (!limitedRows.length) {
            return null;
        }

        const dtypeSource = rawPreview.dtypes || rawPreview.data_types || rawPreview.column_types;
        const dtypeMap = {};
        if (dtypeSource && typeof dtypeSource === 'object') {
            limitedColumns.forEach(col => {
                if (Object.prototype.hasOwnProperty.call(dtypeSource, col)) {
                    dtypeMap[col] = dtypeSource[col];
                }
            });
        }

        const totalRows = typeof rawPreview.total_rows === 'number'
            ? rawPreview.total_rows
            : (typeof rawPreview.totalRows === 'number' ? rawPreview.totalRows : undefined);

        const sample = {
            columns: limitedColumns,
            rows: limitedRows,
            total_rows: totalRows,
            row_count: limitedRows.length,
            column_count: allColumns.length,
            truncated_columns: Math.max(allColumns.length - limitedColumns.length, 0),
            row_limit: PREVIEW_SAMPLE_ROW_LIMIT,
            column_limit: PREVIEW_SAMPLE_COLUMN_LIMIT
        };

        if (rawPreview.source_id || rawPreview.sourceId) {
            sample.source_id = rawPreview.source_id || rawPreview.sourceId;
        }
        if (rawPreview.name) {
            sample.dataset_name = String(rawPreview.name);
        }

        if (Object.keys(dtypeMap).length) {
            sample.dtypes = dtypeMap;
            sample.type_summary = summariseTypeBuckets(dtypeMap);
        }

        const numericCols = Array.isArray(rawPreview.numeric_columns) ? rawPreview.numeric_columns : [];
        if (numericCols.length) {
            sample.numeric_columns = numericCols.filter(col => limitedColumns.includes(col)).slice(0, PREVIEW_SAMPLE_COLUMN_LIMIT);
        }
        const categoricalCols = Array.isArray(rawPreview.categorical_columns) ? rawPreview.categorical_columns : [];
        if (categoricalCols.length) {
            sample.categorical_columns = categoricalCols.filter(col => limitedColumns.includes(col)).slice(0, PREVIEW_SAMPLE_COLUMN_LIMIT);
        }

        return sample;
    }

    function getPreviewDataSource() {
        try {
            if (window.DI?.previewPage && typeof window.DI.previewPage.getCurrentPreviewData === 'function') {
                return window.DI.previewPage.getCurrentPreviewData();
            }
        } catch (e) {
            console.warn('Could not access preview page module:', e);
        }
        return null;
    }

    // Helper: determine if the page/site is currently in dark mode
    function isPageDark() {
        try {
            // 1) Prefer an explicit saved site preference (localStorage) when available.
            //    The preview page stores the user's choice in localStorage.darkMode — honor that first
            //    so the site toggle always wins over OS/browser preferences.
            try {
                const stored = localStorage.getItem('darkMode');
                if (stored === 'true') return true;
                if (stored === 'false') return false;

                const storedTheme = localStorage.getItem('theme');
                if (storedTheme === 'dark') return true;
                if (storedTheme === 'light') return false;

                const edaTheme = localStorage.getItem('eda-theme');
                if (edaTheme === 'dark') return true;
                if (edaTheme === 'light') return false;
            } catch (e) {
                // ignore storage errors and continue to other checks
            }

            // 2) Check explicit page markers (body/html classes or data attributes)
            if (document.body) {
                const body = document.body;
                if (body.classList.contains('light') || body.classList.contains('light-mode') || body.classList.contains('theme-light')) return false;
                if (body.dataset && (body.dataset.theme === 'light' || body.dataset.mode === 'light' || body.dataset.bsTheme === 'light')) return false;

                if (body.classList.contains('dark') || body.classList.contains('dark-mode') || body.classList.contains('theme-dark')) return true;
                if (body.dataset && (body.dataset.theme === 'dark' || body.dataset.mode === 'dark' || body.dataset.bsTheme === 'dark')) return true;
            }
            if (document.documentElement) {
                const el = document.documentElement;
                if (el.classList.contains('light') || el.classList.contains('light-mode') || el.classList.contains('theme-light')) return false;
                if (el.dataset && (el.dataset.theme === 'light' || el.dataset.mode === 'light' || el.dataset.bsTheme === 'light')) return false;

                if (el.classList.contains('dark') || el.classList.contains('dark-mode') || el.classList.contains('theme-dark')) return true;
                if (el.dataset && (el.dataset.theme === 'dark' || el.dataset.mode === 'dark' || el.dataset.bsTheme === 'dark')) return true;
            }

            // 3) Check OS/browser preference (only used when no explicit site pref found)
            if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return true;

            // 4) Fallback: compute background color luminance
            const bg = window.getComputedStyle(document.body || document.documentElement).backgroundColor;
            if (bg) {
                // Parse rgb(a)
                const m = bg.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
                if (m) {
                    const r = Number(m[1]), g = Number(m[2]), b = Number(m[3]);
                    // Perceived luminance
                    const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    return lum < 128; // dark if luminance low
                }
            }
        } catch (e) {
            // ignore and default to light
        }
        return false;
    }

    function createButton() {
        if (!floatingButtonEnabled) {
            return floatingButtonElement;
        }

        if (floatingButtonElement && floatingButtonElement.isConnected) {
            return floatingButtonElement;
        }

        const btn = document.createElement('button');
        btn.className = 'llm-chat-button';
        btn.title = 'Chat with LLM';
        btn.innerHTML = '<i class="fas fa-robot"></i>';
        // Click opens the full chat modal directly
        btn.addEventListener('click', function() {
            showModal();
        });
        document.body.appendChild(btn);
        floatingButtonElement = btn;
        return btn;
    }

    // Show a simple "AI is here!" hint near the button for a few seconds
    function showAiHint() {
        if (!floatingButtonEnabled) {
            return;
        }
        if (floatingHintDisplayed) {
            return;
        }
        try {
            // Remove existing hint if present
            const prev = document.querySelector('.llm-ai-hint');
            if (prev) prev.remove();

            if (!floatingButtonElement || !floatingButtonElement.isConnected) {
                return;
            }

            const el = document.createElement('div');
            el.className = 'llm-ai-hint';
            el.textContent = 'AI is here!';
            el.style.position = 'fixed';
            el.style.right = '90px';
            el.style.bottom = '30px';
            el.style.zIndex = '1080';
            el.style.padding = '8px 12px';
            el.style.borderRadius = '16px';
            el.style.background = 'var(--llm-modal-bg, #fff)';
            el.style.color = 'var(--llm-btn-fg, #111)';
            el.style.boxShadow = '0 8px 20px rgba(0,0,0,0.16)';
            el.style.fontSize = '0.9rem';
            el.style.fontWeight = '500';
            el.style.cursor = 'pointer';

            // Click hint to open chat
            el.addEventListener('click', function() {
                showModal();
                el.remove();
            });

            document.body.appendChild(el);
            // Auto-remove after 3 seconds
            setTimeout(() => el.remove(), 3000);
            floatingHintDisplayed = true;
        } catch (e) {
            // ignore
        }
    }

    function createModal() {
        const backdrop = document.createElement('div');
        backdrop.className = 'llm-chat-modal-backdrop';
        backdrop.addEventListener('click', hideModal);
        document.body.appendChild(backdrop);

        const modal = document.createElement('div');
        modal.className = 'llm-chat-modal';

        modal.innerHTML = `
            <div class="llm-chat-header">
                <i class="fas fa-robot"></i>
                <span>Data Assistant</span>
                <div class="llm-status">
                    <small id="llm-active-model" class="text-muted">Loading...</small>
                </div>
                <button id="llm-close" class="btn-close ms-auto" aria-label="Close"></button>
            </div>
            <div class="llm-chat-body" id="llm-chat-body"></div>
            <div class="llm-chat-footer">
                <input id="llm-input" class="form-control llm-input" placeholder="Ask about this dataset..." />
                <button id="llm-send" class="btn btn-primary">Send</button>
            </div>
        `;

        document.body.appendChild(modal);

        document.getElementById('llm-close').addEventListener('click', hideModal);
        document.getElementById('llm-send').addEventListener('click', sendMessage);
        document.getElementById('llm-input').addEventListener('keydown', function (e) {
            if (e.key === 'Enter') sendMessage();
        });

        // Apply dark-mode class to the modal if the user or site prefers dark scheme
        try {
            if (isPageDark()) {
                modal.classList.add('dark');
            }

            // Listen for changes in OS/theme and toggle the class accordingly
            if (window.matchMedia) {
                const mq = window.matchMedia('(prefers-color-scheme: dark)');
                if (mq.addEventListener) {
                    mq.addEventListener('change', () => {
                        if (isPageDark()) modal.classList.add('dark'); else modal.classList.remove('dark');
                    });
                } else if (mq.addListener) {
                    mq.addListener(() => {
                        if (isPageDark()) modal.classList.add('dark'); else modal.classList.remove('dark');
                    });
                }
            }

            // Observe changes to the page (body and html attributes/classes) so the modal
            // follows site-level theme toggles (many apps toggle classes on body or html)
            try {
                if (!window._llmThemeObserverInitialized) {
                    const obsCb = function(mutations) {
                        const modalEl = document.querySelector('.llm-chat-modal');
                        if (!modalEl) return;
                        if (isPageDark()) modalEl.classList.add('dark'); else modalEl.classList.remove('dark');
                    };
                    const mo = new MutationObserver(obsCb);
                    const attrsToWatch = ['class', 'data-theme', 'data-bs-theme', 'data-mode'];
                    if (document.body) mo.observe(document.body, { attributes: true, attributeFilter: attrsToWatch });
                    if (document.documentElement) mo.observe(document.documentElement, { attributes: true, attributeFilter: attrsToWatch });
                    
                    // Also listen for storage changes (in case dark mode is synced across tabs)
                    window.addEventListener('storage', function(e) {
                        if (e.key === 'darkMode' || e.key === 'eda-theme' || e.key === 'theme') {
                            const modalEl = document.querySelector('.llm-chat-modal');
                            if (modalEl) {
                                if (isPageDark()) modalEl.classList.add('dark'); else modalEl.classList.remove('dark');
                            }
                        }
                    });
                    
                    window._llmThemeObserverInitialized = true;
                    window._llmThemeObserver = mo;
                }
            } catch (e) {
                // ignore
            }
        } catch (e) {
            // ignore
        }
    }

    // Ensure theme observer is initialized as early as possible so we don't miss
    // toggles made by other scripts that may be defined/overwritten after this
    // widget is created. This observer is lightweight: it only watches attribute
    // changes on <body> and <html> and will update the modal class when the
    // page theme changes.
    function initTopLevelThemeObserver() {
        try {
            if (window._llmTopLevelObserverInitialized) return;
            const obsCb = function(mutations) {
                const modalEl = document.querySelector('.llm-chat-modal');
                if (!modalEl) return;
                try {
                    if (isPageDark()) modalEl.classList.add('dark'); else modalEl.classList.remove('dark');
                } catch (e) {
                    // ignore
                }
            };
            const mo = new MutationObserver(obsCb);
            const attrsToWatch = ['class', 'data-theme', 'data-bs-theme', 'data-mode'];
            if (document.body) mo.observe(document.body, { attributes: true, attributeFilter: attrsToWatch });
            if (document.documentElement) mo.observe(document.documentElement, { attributes: true, attributeFilter: attrsToWatch });

            // Also listen for storage events (other tabs) and custom events
            window.addEventListener('storage', function(e) {
                if (e.key === 'darkMode' || e.key === 'eda-theme' || e.key === 'theme') {
                    const modalEl = document.querySelector('.llm-chat-modal');
                    if (modalEl) {
                        if (isPageDark()) modalEl.classList.add('dark'); else modalEl.classList.remove('dark');
                    }
                }
            });

            window._llmTopLevelObserverInitialized = true;
            window._llmTopLevelObserver = mo;
        } catch (e) {
            // ignore
        }
    }

    // Initialize immediately so theme changes are tracked even if createModal
    // runs later or other scripts overwrite global toggle functions.
    try { initTopLevelThemeObserver(); } catch (e) { /* ignore */ }

    function toggleModal() {
        const backdrop = document.querySelector('.llm-chat-modal-backdrop');
        const modal = document.querySelector('.llm-chat-modal');
        if (!backdrop || !modal) return;
        if (modal.style.display === 'flex') {
            hideModal();
        } else {
            showModal();
        }
    }

    function showModal() {
        const backdrop = document.querySelector('.llm-chat-modal-backdrop');
        const modal = document.querySelector('.llm-chat-modal');
        backdrop.style.display = 'block';
        modal.style.display = 'flex';
        modal.style.flexDirection = 'column';
    syncActiveChatKey();
        // focus input
        setTimeout(() => document.getElementById('llm-input').focus(), 50);
        // Re-evaluate theme each time the modal is shown so it follows the
        // current site/browser color scheme even if it changed since creation.
        try {
            if (isPageDark()) modal.classList.add('dark'); else modal.classList.remove('dark');
        } catch (e) {
            // ignore
        }

        // Load provider/model info when the modal is opened. Only mark as
        // loaded if the fetch succeeds — keep false on failure so subsequent
        // opens will retry.
        try {
            if (!modelInfoLoaded) {
                loadActiveModelInfo().then(success => {
                    if (success) modelInfoLoaded = true;
                }).catch(() => { /* no-op */ });
            }
        } catch (e) {
            console.warn('Error loading model info on open:', e);
        }
    }

    function hideModal() {
        const backdrop = document.querySelector('.llm-chat-modal-backdrop');
        const modal = document.querySelector('.llm-chat-modal');
        if (backdrop) backdrop.style.display = 'none';
        if (modal) modal.style.display = 'none';
    }

    function appendMessageToDOM(text, who = 'bot') {
        const body = document.getElementById('llm-chat-body');
        if (!body) return;
        const wrap = document.createElement('div');
        wrap.className = `llm-msg ${who}`;
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        if (who === 'bot') {
            // Enhanced formatting for bot responses
            bubble.innerHTML = formatBotResponse(text);
        } else {
            // Simple formatting for user messages
            if (typeof text === 'string' && text.includes('\n')) {
                bubble.innerHTML = text.replace(/\n/g, '<br>');
            } else {
                bubble.innerText = text;
            }
        }
        
        wrap.appendChild(bubble);
        body.appendChild(wrap);
        body.scrollTop = body.scrollHeight;
    }

    function getHistorySettings(chatKey) {
        const key = chatKey || DEFAULT_CHAT_KEY;
        if (key.startsWith('eda:')) {
            const parts = key.split(':');
            if (parts.length >= 4) {
                return {
                    messages: CELL_HISTORY_MESSAGES,
                    chars: CELL_HISTORY_CHAR_LIMIT
                };
            }
            return {
                messages: EDA_WORKSPACE_HISTORY_MESSAGES,
                chars: EDA_WORKSPACE_HISTORY_CHAR_LIMIT
            };
        }
        return {
            messages: GLOBAL_HISTORY_MESSAGES,
            chars: GLOBAL_HISTORY_CHAR_LIMIT
        };
    }

    function trimConversationHistory(conversation, chatKey) {
        if (!Array.isArray(conversation)) return;

        const settings = getHistorySettings(chatKey);
        const messageLimit = settings?.messages && settings.messages > 0
            ? settings.messages
            : GLOBAL_HISTORY_MESSAGES;
        const storageLimit = Math.max(2, messageLimit * 2);

        if (conversation.length > storageLimit) {
            conversation.splice(0, conversation.length - storageLimit);
        }

        const charLimit = settings?.chars && settings.chars > 0
            ? settings.chars
            : null;

        if (!charLimit) {
            return;
        }

        let result = [];
        let running = 0;

        for (let i = conversation.length - 1; i >= 0; i--) {
            const entry = conversation[i];
            const content = typeof entry.content === 'string'
                ? entry.content
                : String(entry.content ?? '');
            const available = charLimit - running;
            if (available <= 0) {
                break;
            }

            if (content.length <= available) {
                result.push({ role: entry.role, content });
                running += content.length;
            } else {
                const truncated = content.slice(Math.max(0, content.length - available));
                result.push({
                    role: entry.role,
                    content: `...(trimmed)\n${truncated}`
                });
                running += truncated.length;
                break;
            }
        }

        result = result.reverse();

        if (!result.length && conversation.length) {
            const lastEntry = conversation[conversation.length - 1];
            const lastContent = typeof lastEntry.content === 'string'
                ? lastEntry.content
                : String(lastEntry.content ?? '');
            const clipped = lastContent.slice(Math.max(0, lastContent.length - charLimit));
            const needsTrimNotice = clipped.length < lastContent.length;
            result = [{
                role: lastEntry.role,
                content: needsTrimNotice ? `...(trimmed)\n${clipped}` : (clipped || '(Earlier conversation trimmed)')
            }];
        }

        if (result.length) {
            conversation.splice(0, conversation.length, ...result);
        }
    }

    function appendMessage(text, who = 'bot', chatKey = currentChatKey) {
        const key = chatKey || DEFAULT_CHAT_KEY;
        const storedRole = who === 'bot' ? 'assistant' : 'user';
        const conversation = conversationStore.get(key) || [];
        conversation.push({ role: storedRole, content: text });
        trimConversationHistory(conversation, key);
        conversationStore.set(key, conversation);
        if (key === currentChatKey) {
            appendMessageToDOM(text, who);
        }
    }

    function renderConversation(chatKey) {
        const body = document.getElementById('llm-chat-body');
        if (!body) return;
        removeLoadingMessage();
        body.innerHTML = '';
        const conversation = conversationStore.get(chatKey) || [];
        conversation.forEach(entry => {
            appendMessageToDOM(entry.content, entry.role === 'user' ? 'user' : 'bot');
        });
        body.scrollTop = body.scrollHeight;
    }

    function ensureConversationKey(chatKey) {
        const key = chatKey || DEFAULT_CHAT_KEY;
        if (!conversationStore.has(key)) {
            conversationStore.set(key, []);
        }
        return key;
    }

    function extractSourceId(pageContext) {
        if (!pageContext || typeof pageContext !== 'object') {
            return window.DATA_PREVIEW_CONFIG?.sourceId
                || (typeof window.sourceId !== 'undefined' ? window.sourceId : null);
        }
        return pageContext?.dataStructures?.currentSourceId
            || pageContext?.dataset?.sourceId
            || pageContext?.edaNotebook?.dataset?.sourceId
            || window.DATA_PREVIEW_CONFIG?.sourceId
            || (typeof window.sourceId !== 'undefined' ? window.sourceId : null);
    }

    function getActiveCellFromContext(pageContext) {
        if (!pageContext || typeof pageContext !== 'object') {
            return null;
        }
        if (pageContext.targetCellId) {
            return {
                cellId: pageContext.targetCellId,
                scope: pageContext.targetCellScope || 'analysis'
            };
        }
        const notebook = pageContext.edaNotebook;
        const active = notebook && typeof notebook === 'object' ? notebook.activeContext : null;
        if (active && active.cellId) {
            return {
                cellId: active.cellId,
                scope: active.scope || 'analysis'
            };
        }
        return null;
    }

    function determineChatKey(pageContext) {
        const activeCell = getActiveCellFromContext(pageContext);
        const sourceId = extractSourceId(pageContext);
        if (activeCell && activeCell.cellId) {
            return `eda:${sourceId || 'unknown'}:${activeCell.scope || 'analysis'}:${activeCell.cellId}`;
        }
        if ((pageContext && pageContext.edaNotebook) || isEdaWorkspace()) {
            return `eda:${sourceId || 'unknown'}:workspace`;
        }
        if (pageContext?.dataStructures?.previewSample) {
            return `preview:${sourceId || 'unknown'}`;
        }
        return DEFAULT_CHAT_KEY;
    }

    function determineEndpoint(pageContext) {
        if ((pageContext && pageContext.edaNotebook) || isEdaWorkspace()) {
            return EDA_API_PATH;
        }
        return DEFAULT_API_PATH;
    }

    function isModalVisible() {
        const modal = document.querySelector('.llm-chat-modal');
        return Boolean(modal && modal.style.display === 'flex');
    }

    function syncActiveChatKey(latestContext) {
        const context = latestContext || gatherPageContext();
        const key = ensureConversationKey(determineChatKey(context));
        const changed = key !== currentChatKey;
        currentChatKey = key;
        if (isModalVisible()) {
            renderConversation(currentChatKey);
        }
        return { context, chatKey: key, changed };
    }

    function formatBotResponse(text) {
        if (!text || typeof text !== 'string') return text;
        
        let formatted = text;
        
        // Convert tables (simple markdown-like format)
        formatted = convertTables(formatted);
        
        // Convert headers
        formatted = convertHeaders(formatted);
        
        // Convert lists
        formatted = convertLists(formatted);
        
        // Enhanced emoji formatting
        formatted = enhanceEmojis(formatted);
        
        // Convert bold/italic text with enhanced metrics
        formatted = convertTextFormatting(formatted);
        
        // Convert code blocks with language support
        formatted = convertCodeBlocks(formatted);
        
        // Clean up excessive line breaks and convert to HTML
        formatted = formatted.replace(/\n{3,}/g, '\n\n'); // Reduce 3+ line breaks to 2
        formatted = formatted.replace(/\n\n/g, '</p><p>'); // Convert double breaks to paragraphs
        formatted = formatted.replace(/\n/g, '<br>'); // Single breaks to <br>
        
        // Wrap in paragraph tags if not already wrapped
        if (!formatted.startsWith('<') && formatted.length > 0) {
            formatted = '<p>' + formatted + '</p>';
        }
        
        // Clean up empty paragraphs and extra breaks
        formatted = formatted.replace(/<p><\/p>/g, '');
        formatted = formatted.replace(/<p><br><\/p>/g, '');
        formatted = formatted.replace(/<br><\/p>/g, '</p>');
        formatted = formatted.replace(/<p><br>/g, '<p>');
        
        return formatted;
    }

    function convertTables(text) {
        // Convert markdown-style tables to HTML tables
        const lines = text.split('\n');
        let inTable = false;
        let tableLines = [];
        let result = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Check if this line looks like a table row (contains |)
            if (line.includes('|') && line.split('|').length > 2) {
                if (!inTable) {
                    inTable = true;
                    tableLines = [];
                }
                tableLines.push(line);
            } else {
                // End of table
                if (inTable) {
                    result.push(formatTable(tableLines));
                    tableLines = [];
                    inTable = false;
                }
                result.push(line);
            }
        }
        
        // Handle table at end of text
        if (inTable && tableLines.length > 0) {
            result.push(formatTable(tableLines));
        }
        
        return result.join('\n');
    }

    function formatTable(tableLines) {
        if (tableLines.length < 2) return tableLines.join('\n');
        
        let html = '<div class="table-container"><table class="llm-table table table-striped table-sm">';
        
        // First line is header
        const headerCells = tableLines[0].split('|').map(cell => cell.trim()).filter(cell => cell);
        html += '<thead><tr>';
        headerCells.forEach(cell => {
            html += `<th>${cell}</th>`;
        });
        html += '</tr></thead>';
        
        // Skip separator line (if exists) and process data rows
        html += '<tbody>';
        for (let i = 1; i < tableLines.length; i++) {
            const line = tableLines[i].trim();
            if (line.includes('---') || line.includes('===')) continue; // Skip separator lines
            
            const cells = line.split('|').map(cell => cell.trim()).filter(cell => cell);
            if (cells.length > 0) {
                html += '<tr>';
                cells.forEach(cell => {
                    html += `<td>${cell}</td>`;
                });
                html += '</tr>';
            }
        }
        html += '</tbody></table></div>';
        
        return html;
    }

    function convertHeaders(text) {
        // Convert ## Header to HTML headers
        return text
            .replace(/^### (.*$)/gim, '<h5 class="llm-header">$1</h5>')
            .replace(/^## (.*$)/gim, '<h4 class="llm-header">$1</h4>')
            .replace(/^# (.*$)/gim, '<h3 class="llm-header">$1</h3>');
    }

    function convertLists(text) {
        const lines = text.split('\n');
        let result = [];
        let inList = false;
        let listItems = [];
        
        for (let line of lines) {
            const trimmed = line.trim();
            
            // Check for list items
            if (trimmed.match(/^[-*•]\s+/)) {
                if (!inList) {
                    inList = true;
                    listItems = [];
                }
                listItems.push(trimmed.replace(/^[-*•]\s+/, ''));
            } else if (trimmed.match(/^\d+\.\s+/)) {
                if (!inList) {
                    inList = true;
                    listItems = [];
                }
                listItems.push(trimmed.replace(/^\d+\.\s+/, ''));
            } else {
                // End of list
                if (inList) {
                    result.push('<ul class="llm-list">');
                    listItems.forEach(item => {
                        result.push(`<li>${item}</li>`);
                    });
                    result.push('</ul>');
                    listItems = [];
                    inList = false;
                }
                result.push(line);
            }
        }
        
        // Handle list at end
        if (inList && listItems.length > 0) {
            result.push('<ul class="llm-list">');
            listItems.forEach(item => {
                result.push(`<li>${item}</li>`);
            });
            result.push('</ul>');
        }
        
        return result.join('\n');
    }

    function convertTextFormatting(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **bold**
            .replace(/\*(.*?)\*/g, '<em>$1</em>')  // *italic*
            .replace(/`(.*?)`/g, '<code class="llm-code">$1</code>')  // `code`
            // Enhanced formatting for stats and metrics
            .replace(/(\d+\.?\d*%)/g, '<span class="highlight-stat">$1</span>')  // Percentages
            .replace(/(\d+\.?\d*) - (Good|Excellent)/gi, '<span class="metric-good">$1 - $2</span>')
            .replace(/(\d+\.?\d*) - (Warning|Moderate)/gi, '<span class="metric-warning">$1 - $2</span>')
            .replace(/(\d+\.?\d*) - (Poor|Critical|Risk)/gi, '<span class="metric-danger">$1 - $2</span>');
    }

    function convertCodeBlocks(text) {
        return text.replace(/```(python|sql|r)?\n?([\s\S]*?)```/g, function(match, lang, code) {
            return `<pre class="llm-code-block"><code class="language-${lang || 'python'}">${code.trim()}</code></pre>`;
        });
    }

    function enhanceEmojis(text) {
        // Add special formatting for emoji headers
        return text.replace(/(📊|📈|📋|🔧|⚠️|💡|❓|🔬|🎯|🚀)\s*\*\*(.*?)\*\*/g, 
            '<span class="emoji-header">$1</span><strong>$2</strong>');
    }

    function appendLoadingMessage() {
        const body = document.getElementById('llm-chat-body');
        if (!body) return;
        const wrap = document.createElement('div');
        wrap.className = 'llm-msg bot loading-message';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = `
            <div class="loading-indicator">
                <div class="loading-dots">
                    <span></span><span></span><span></span>
                </div>
                <span class="loading-text">Analyzing your data...</span>
            </div>
        `;
        wrap.appendChild(bubble);
        body.appendChild(wrap);
        body.scrollTop = body.scrollHeight;
        return wrap;
    }

    function removeLoadingMessage() {
        const loadingMessages = document.querySelectorAll('.loading-message');
        loadingMessages.forEach(msg => msg.remove());
    }

    async function loadActiveModelInfo() {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), MODEL_INFO_TIMEOUT);
        try {
            const response = await fetch('/llm/providers', { signal: controller.signal });
            clearTimeout(timeout);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            
            const statusElement = document.getElementById('llm-active-model');
            if (!statusElement) return true;
            
            if (data.default_provider && data.available_providers && data.available_providers.length > 0) {
                const providerInfo = data.provider_info?.[data.default_provider] || {};
                const modelName = providerInfo.default_model || 'default';
                const providerName = data.default_provider.charAt(0).toUpperCase() + data.default_provider.slice(1);
                
                statusElement.textContent = `${providerName} • ${modelName}`;
                statusElement.className = 'text-success';
                statusElement.title = `Active LLM: ${providerName} using ${modelName}`;
            } else {
                // Show configured provider from config even if health check fails
                statusElement.textContent = 'LLM Available — Ready for queries';
                statusElement.className = 'text-warning';
                statusElement.title = 'LLM service is configured and ready';
            }
            return true;
        } catch (error) {
            // AbortError is expected on timeout; handle gracefully
            if (error.name === 'AbortError') {
                console.warn('LLM providers request aborted due to timeout');
            } else {
                console.error('Failed to load model info:', error);
            }
            const statusElement = document.getElementById('llm-active-model');
            if (statusElement) {
                statusElement.textContent = 'LLM Available — Ready for queries';
                statusElement.className = 'text-warning';
                statusElement.title = 'LLM service is configured and ready';
            }
            return false;
        } finally {
            clearTimeout(timeout);
        }
    }

    async function sendMessage() {
        const input = document.getElementById('llm-input');
        const text = (input && input.value || '').trim();
        if (!text) return;
        const gatheredContext = gatherPageContext();
        const { context: hydratedContext, chatKey: conversationKey } = syncActiveChatKey(gatheredContext);
        const activeCell = getActiveCellFromContext(hydratedContext);
        const resolvedSourceId = extractSourceId(hydratedContext);

        const existingConversation = conversationStore.get(conversationKey) || [];
        const historyCopy = existingConversation.map(entry => ({ role: entry.role, content: entry.content }));
        trimConversationHistory(historyCopy, conversationKey);
        const baseMessages = historyCopy.map(entry => ({ role: entry.role, content: entry.content }));
        const payloadMessages = baseMessages.concat([{ role: 'user', content: text }]);

        appendMessage(text, 'user', conversationKey);
        input.value = '';

        appendLoadingMessage();
        try {
            const payload = {
                messages: payloadMessages,
                source_id: resolvedSourceId,
                include_context: true,
                max_tokens: 3000,
                page_context: hydratedContext
            };
            if (activeCell?.cellId) {
                payload.cell_id = activeCell.cellId;
            }
            if (activeCell?.scope) {
                payload.cell_scope = activeCell.scope;
            }
            
            // Use code model if selected from LLMModelManager
            const currentModel = window.LLMModelManager?.getCurrentModel();
            if (currentModel && currentModel.provider && currentModel.model) {
                payload.provider = currentModel.provider;
                payload.model = currentModel.model;
            }

            const endpoint = determineEndpoint(hydratedContext);
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            removeLoadingMessage();

            if (data.success && data.response) {
                appendMessage(data.response, 'bot', conversationKey);
                
                // Update the active model display if provider info is returned
                const statusElement = document.getElementById('llm-active-model');
                if (data.provider && statusElement) {
                    if (data.model) {
                        const providerName = data.provider.charAt(0).toUpperCase() + data.provider.slice(1);
                        statusElement.textContent = `${providerName} • ${data.model}`;
                        statusElement.className = 'text-success';
                    } else {
                        // If provider present but model missing, refresh provider info
                        loadActiveModelInfo();
                    }
                } else if (statusElement) {
                    // If server didn't give provider info, try to refresh providers endpoint
                    loadActiveModelInfo();
                }
            } else if (data.response) {
                // Handle older format or error responses that still have response
                appendMessage(data.response, 'bot', conversationKey);
            } else if (data.error) {
                appendMessage(`Error: ${data.error}`, 'bot', conversationKey);
                
                // Show available providers if none are configured
                if (data.available_providers) {
                    appendMessage(`Available providers: ${data.available_providers.join(', ')}`, 'bot', conversationKey);
                }
            } else {
                appendMessage('No response received from LLM service.', 'bot', conversationKey);
            }

        } catch (err) {
            removeLoadingMessage();
            appendMessage('Error contacting server: ' + err.message, 'bot', conversationKey);
        }
    }

    function gatherPageContext() {
        // Gather context from the current page state and data structures
        const context = {
            currentTab: getCurrentTab(),
            visibleData: {},
            dataStructures: {
                currentSourceId: window.DATA_PREVIEW_CONFIG?.sourceId || null,
                currentQualityReport: null
            },
            dataset: null,
            targetCellId: null,
            targetCellScope: null
        };

        const cachedSample = getCachedPreviewSample(context.dataStructures.currentSourceId);
        if (cachedSample) {
            context.dataStructures.previewSample = cachedSample;
        }

        // Try to access the global data structures used by preview modules
        try {
            if (window.DI && window.DI.previewPage) {
                const rawPreview = getPreviewDataSource();
                const previewSample = buildPreviewSample(rawPreview);
                if (previewSample) {
                    const sampleSourceId = previewSample.source_id
                        || context.dataStructures.currentSourceId
                        || window.DATA_PREVIEW_CONFIG?.sourceId
                        || null;
                    cachePreviewSample(previewSample, sampleSourceId);
                    context.dataStructures.previewSample = previewSample;
                    if (!context.dataStructures.currentSourceId && sampleSourceId) {
                        context.dataStructures.currentSourceId = sampleSourceId;
                    }
                } else if (!context.dataStructures.previewSample) {
                    schedulePreviewSampleRefresh(600);
                }
                if (window.currentQualityReport) {
                    context.dataStructures.currentQualityReport = window.currentQualityReport;
                }
            }
        } catch (e) {
            console.warn('Could not access DI data structures:', e);
        }

        // Try to merge context provided by the EDA notebook bridge
        let edaContext = null;
        try {
            if (window.EDAChatBridge && typeof window.EDAChatBridge.getContext === 'function') {
                edaContext = window.EDAChatBridge.getContext();
                if (edaContext) {
                    context.edaNotebook = edaContext;
                    const condensedSummary = buildEdaNotebookCondensedSummary(edaContext);
                    if (condensedSummary) {
                        context.edaNotebookSummary = condensedSummary;
                    }
                    if (Array.isArray(edaContext.analysisHighlights) || Array.isArray(edaContext.customHighlights)) {
                        context.edaHighlights = {
                            analyses: Array.isArray(edaContext.analysisHighlights) ? edaContext.analysisHighlights : [],
                            custom: Array.isArray(edaContext.customHighlights) ? edaContext.customHighlights : []
                        };
                    }
                    if (!context.dataStructures.currentSourceId && edaContext.dataset?.sourceId) {
                        context.dataStructures.currentSourceId = edaContext.dataset.sourceId;
                    }
                    if (!context.dataset && edaContext.dataset) {
                        context.dataset = edaContext.dataset;
                    }
                    if (edaContext.activeContext) {
                        context.targetCellId = edaContext.activeContext.cellId || null;
                        context.targetCellScope = edaContext.activeContext.scope || null;
                    }
                }
            }
        } catch (edaError) {
            console.warn('Could not collect EDA notebook context:', edaError);
        }

        if (!context.dataStructures.currentSourceId && typeof window.sourceId !== 'undefined' && window.sourceId) {
            context.dataStructures.currentSourceId = window.sourceId;
        }

        if (!context.dataset && edaContext?.dataset) {
            context.dataset = edaContext.dataset;
        }

        // Get data from currently visible tab
        const activeTab = document.querySelector('.preview-tab-content.active');
        if (activeTab) {
            const tabId = activeTab.id;
            const tabContent = extractTabContent(activeTab);
            context.visibleData[tabId] = tabContent;
        }

        // Try to get data from all tabs if available
        ['qualityTabContent', 'textTabContent', 'recommendationsTabContent', 'previewTabContent'].forEach(tabId => {
            const tab = document.getElementById(tabId);
            if (tab && !context.visibleData[tabId]) {
                context.visibleData[tabId] = extractTabContent(tab);
            }
        });

        // Extract specific data patterns from the DOM
        context.extractedData = extractStructuredData();

        return context;
    }

    function extractStructuredData() {
        const extracted = {};

        // Extract quality metrics from DOM
        try {
            const qualityStats = document.querySelectorAll('.quality-stat');
            if (qualityStats.length > 0) {
                extracted.qualityMetrics = [];
                qualityStats.forEach(stat => {
                    const value = stat.querySelector('.quality-stat-value')?.textContent;
                    const label = stat.querySelector('.quality-stat-label')?.textContent;
                    if (value && label) {
                        extracted.qualityMetrics.push({ label, value });
                    }
                });
            }
        } catch (e) {
            console.warn('Could not extract quality metrics:', e);
        }

        // Extract metadata from DOM
        try {
            const metadataCards = document.querySelectorAll('.metadata-card');
            if (metadataCards.length > 0) {
                extracted.metadata = [];
                metadataCards.forEach(card => {
                    const value = card.querySelector('.metadata-value')?.textContent;
                    const label = card.querySelector('.metadata-label')?.textContent;
                    if (value && label) {
                        extracted.metadata.push({ label, value });
                    }
                });
            }
        } catch (e) {
            console.warn('Could not extract metadata:', e);
        }

        // Extract column information from tables
        try {
            const dataTable = document.querySelector('#dataPreviewTable');
            if (dataTable) {
                const headers = Array.from(dataTable.querySelectorAll('thead th')).map(th => th.textContent.trim());
                const rows = Array.from(dataTable.querySelectorAll('tbody tr')).slice(0, 5).map(tr => 
                    Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim())
                );
                extracted.tableData = { headers, sampleRows: rows };
            }
        } catch (e) {
            console.warn('Could not extract table data:', e);
        }

        // Extract recommendation text
        try {
            const recommendations = document.querySelectorAll('.recommendation-card, .text-recommendation');
            if (recommendations.length > 0) {
                extracted.recommendations = Array.from(recommendations).map(rec => rec.textContent.trim()).slice(0, 10);
            }
        } catch (e) {
            console.warn('Could not extract recommendations:', e);
        }

        // Extract text analysis results
        try {
            const textStats = document.querySelectorAll('.text-stat-card, .text-column-card');
            if (textStats.length > 0) {
                extracted.textAnalysis = Array.from(textStats).map(stat => stat.textContent.trim()).slice(0, 10);
            }
        } catch (e) {
            console.warn('Could not extract text analysis:', e);
        }

        return extracted;
    }

    function getCurrentTab() {
        const activeBtn = document.querySelector('.preview-tab-btn.active');
        return activeBtn ? activeBtn.getAttribute('data-tab') : 'unknown';
    }

    window.addEventListener('eda-llm-context-updated', function(event) {
        try {
            syncActiveChatKey(event?.detail);
        } catch (e) {
            console.warn('Failed to sync LLM chat conversation:', e);
        }
    });

    function extractTabContent(tabElement) {
        if (!tabElement) return null;

        // Extract text content, but clean it up
        let content = tabElement.innerText || tabElement.textContent || '';
        
        // Remove loading messages
        content = content.replace(/Loading.*?\.\.\./g, '');
        content = content.replace(/Please wait.*?\./g, '');
        
        // Clean up extra whitespace
        content = content.replace(/\s+/g, ' ').trim();
        
        // If content is too short, it's probably not loaded yet
        if (content.length < 20) {
            return 'Content not yet loaded';
        }

        // Limit content length
        return content.substring(0, 1000);
    }

    // Init
    document.addEventListener('DOMContentLoaded', function () {
        if (isEdaWorkspace()) {
            try {
                const initialActiveButton = document.querySelector('.eda-tab-button.active[data-bs-target]');
                if (initialActiveButton) {
                    currentEdaActiveTabId = (initialActiveButton.getAttribute('data-bs-target') || '').replace('#', '') || null;
                } else {
                    const activePane = document.querySelector('.tab-pane.show.active');
                    if (activePane && activePane.id) {
                        currentEdaActiveTabId = activePane.id;
                    }
                }
            } catch (e) {
                currentEdaActiveTabId = currentEdaActiveTabId || null;
            }
        }

        syncFloatingButtonVisibility({ showHint: true });
        createModal();

        const handleTabActivatedEvent = (event) => {
            if (!isEdaWorkspace()) {
                return;
            }
            const tabId = event?.detail?.tabId;
            if (!tabId) {
                return;
            }
            currentEdaActiveTabId = tabId;
            syncFloatingButtonVisibility({ showHint: tabId === 'data-preview-content' });
        };

        const handleBootstrapTabShown = (event) => {
            if (!isEdaWorkspace()) {
                return;
            }
            const target = event?.target;
            if (!target) {
                return;
            }
            const tabTarget = target.getAttribute('data-bs-target');
            if (!tabTarget) {
                return;
            }
            const tabId = tabTarget.replace('#', '');
            currentEdaActiveTabId = tabId;
            syncFloatingButtonVisibility({ showHint: tabId === 'data-preview-content' });
        };

        document.addEventListener('tabActivated', handleTabActivatedEvent);
        document.addEventListener('shown.bs.tab', handleBootstrapTabShown);
        
        // Wait a bit more for other scripts to load, then try to hook into toggleDarkMode
        setTimeout(() => {
            try {
                const originalToggleDarkMode = window.toggleDarkMode;
                if (originalToggleDarkMode && !window._llmDarkModeHooked) {
                    window.toggleDarkMode = function() {
                        const result = originalToggleDarkMode.apply(this, arguments);
                        // Update modal theme after the site toggles
                        setTimeout(() => {
                            const modalEl = document.querySelector('.llm-chat-modal');
                            if (modalEl) {
                                if (isPageDark()) modalEl.classList.add('dark'); else modalEl.classList.remove('dark');
                            }
                        }, 10);
                        return result;
                    };
                    window._llmDarkModeHooked = true;
                }
            } catch (e) {
                console.warn('Could not hook into toggleDarkMode:', e);
            }
        }, 100);
    });

    // Failsafe poller: in case mutation observer or hooking doesn't catch the
    // theme change (race/loading differences), poll for a short time and sync
    // the modal class. Runs every 500ms for 30s.
    (function startThemePoller() {
        try {
            let attempts = 0;
            const maxAttempts = 60; // 60 * 500ms = 30s
            const id = setInterval(() => {
                const modal = document.querySelector('.llm-chat-modal');
                if (!modal) return;
                try {
                    if (isPageDark()) {
                        if (!modal.classList.contains('dark')) modal.classList.add('dark');
                    } else {
                        if (modal.classList.contains('dark')) modal.classList.remove('dark');
                    }
                } catch (e) {
                    // ignore
                }
                attempts++;
                if (attempts >= maxAttempts) clearInterval(id);
            }, 500);
        } catch (e) {
            // ignore
        }
    })();

    // === CHAT-SPECIFIC UTILITIES ===
    
    // Expose only chat-specific functions globally
    window.LLMChat = {
        showModal,
        hideModal,
        toggleModal
    };

})();
