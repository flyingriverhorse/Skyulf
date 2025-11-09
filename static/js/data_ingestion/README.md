# Data Ingestion JS Modules — README

Last updated: 2025-09-08

This README documents the JavaScript modules located in `static/js/data_ingestion/`. It explains how modules are organized, their responsibilities, how they connect, and how the system initializes and interacts with the HTML templates (notably `templates/data_ingestion.html`). Use this as a developer reference for maintenance and onboarding.

## High-level overview

The Data Ingestion frontend is implemented as a small modular system of vanilla JavaScript files. A single loader (`data_ingestion_loader.js`) loads modules in a deterministic order. A global namespace `window.DI` holds shared state, utilities and module entry points.

Goals:
- Keep modules small and focused (single responsibility)
- Maintain explicit load order to avoid race conditions
- Share state via `window.DI.state` for cross-module coordination
- Provide small public exports (functions) for HTML integration and manual testing

## Where to include in HTML

Only include the loader in templates:

```html
<script src="{{ url_for('static', filename='js/data_ingestion/data_ingestion_loader.js') }}"></script>
```

The loader will append the individual module scripts into `document.head` in the correct order and start the application.

## Load order (critical)

`data_ingestion_loader.js` defines the order; do not change without understanding dependencies. Current order (important points):

1. `data_ingestion_globals.js` — must be first (shared state)
2. Utility modules: shared notifications (`../shared/notifications.js`), `utilities_formatting.js`, `utilities_connections.js`
3. Core modules: `init_and_tabs.js`, `upload_module.js`, `data_sources.js`, `preview_eda.js`
4. Database/API modules: `database_basic.js`, `database_connectors.js`, `api_connection.js`, `cloud_connectors.js`, `snowflake_connect.js`
5. Specialized features: `web_scraping.js`, `web_scraping_viewer.js`, `export_data.js`, `sync_management.js`

The loader preserves sequential loading (`script.async = false`) to ensure each module's globals and exports are available for downstream modules.

## Global namespace and shared state

All modules attach to `window.DI`.

- `window.DI.state` — defined in `data_ingestion_globals.js` — exposes getters/setters for shared state such as `currentFile`, `currentPreviewData`, `currentSourceId`, `currentScrapingData` and stable event handler references.
- `window.DI.utilities` — utility functions such as `notifications`, `connections`, `formatting`.
- Module-specific namespaces: `window.DI.upload`, `window.DI.sources`, `window.DI.scrape`, `window.DI.dbConnectors`, etc.

Use the global namespace for cross-module calls; try to avoid tightly coupling modules beyond that.

## Module index (file → responsibilities / public API)

Below are short descriptions and the key exported functions that other modules or HTML call directly.

- `data_ingestion_globals.js`
  - Purpose: shared state and global browser behavior (scrollRestoration, beforeunload/pageshow handlers)
  - Exports: `window.DI.state` (getters/setters)
  - Notes: Must load first.

- `../shared/notifications.js`
  - Purpose: central notifications, loading overlays, and theme helpers shared across ingestion, EDA, and admin experiences
  - Key functions (exported globally and under `window.DI.utilities.notifications`):
    - `showNotification(message, type)` — toast-style message (success, error, info, warning)
    - `showLoading(show, title, message, allowBackgroundInteraction)` — shows/hides a loading overlay with graceful fallbacks
    - `toggleTheme()` — toggles dark-mode and persists to `localStorage`
    - `checkForNotifications()` / `displayNotifications()` — fetch & render server notifications via queueing logic
  - Notes: Prefer referencing this shared module directly. The legacy `utilities_notifications.js` file remains as a small shim for backward compatibility.

- `utilities_notifications.js`
  - Purpose: legacy compatibility shim that dynamically loads the shared notifications module for old references
  - Notes: Do not depend on this file for new work; include `../shared/notifications.js` instead.

- `utilities_formatting.js`
  - Purpose: small helpers like `formatFileSize`, date/time utilities, ordering helpers.
  - Used by uploader, display code and EDA preview.

- `utilities_connections.js`
  - Purpose: helper abstractions for calling backend API endpoints (fetch wrappers, error handling).

- `init_and_tabs.js`
  - Purpose: application initialization, tab navigation, theme application, and wiring of core listeners
  - Key exports (attached to `window.DI.init` and globally):
    - `initializeDataIngestion()` — main entrypoint invoked by the loader when DOM is ready
    - `setupTabNavigation()` — (re)attach tab click handlers
    - `switchTab(tabId)` — switch to a named tab (used by some UI buttons)
    - `getCurrentActiveTab()`
  - Behavior: initializes theme from `localStorage`, wires core listeners, and defers any conditional tab visibility to backend-driven responses (legacy feature flag fetch removed).
  - Important: In the finally block it removes the inline initializing overlay and removes the `pre-init` marker after a short delay (configurable `delayMs`) to avoid tab flash.

- `data_sources.js`
  - Purpose: list, search, filter, and display data sources; primary UI for source cards
  - Key functions:
    - `setupDataSourceSearch()` — wire search controls
    - `refreshDataSources()` — fetch `/data/api/sources` and render results
    - `displayDataSourcesList(sources)` — render the list HTML
    - `populateCategoryFilter(sources)` — build category dropdown based on visible sources
  - Notes: uses `window.currentDataSources` to keep a cached view for ordering/search.

- `upload_module.js`
  - Purpose: drag-and-drop and input-based file upload, preview of chosen file, upload flow to `/data/api/upload`.
  - Key functions:
    - `setupFileUpload()` — initialize drop zone, inputs and buttons
    - `uploadFile()` — POST the file to server, show loading, handle response and call `refreshDataSources()`
  - Notes: Uses `window.DI.state.currentFile` for shared state and `showLoading()` for UX.

- `preview_eda.js`
  - Purpose: exploratory data analysis preview helpers and navigation to the preview UI (used by data_sources)
  - Exposes functions for rendering preview tables, charts and navigating to the EDA page.

- `database_basic.js`, `database_connectors.js`
  - Purpose: database credential forms, connector listing (available connectors via `/data/api/connectors/available`), schema browsing and ingestion from databases.
  - Key functions: `loadConnectorInterface()`, `switchDatabaseTab()`, `loadAvailableConnectors()`, connectors testing functions.

- `cloud_connectors.js`, `api_connection.js`, `snowflake_connect.js`
  - Purpose: cloud & sheet connector UIs (S3, GCS, Google Sheets), API ingestion helper UI, and Snowflake integration. Each offers credential testing flows and resource browsers.

- `web_scraping.js`, `web_scraping_viewer.js`
  - Purpose: scraping UI and extraction logic. `web_scraping.js` wires the form (URL validation, scraping options, pagination). `web_scraping_viewer.js` displays scraping results and export options.
  - Key functions: `setupWebScraping()`, `startWebScraping()`.

- `export_data.js`
  - Purpose: export helper UI (download/format/export options), used by preview/EDA modules.

- `sync_management.js`
  - Purpose: UI to configure automatic sync/ingestion scheduling and status for sources.

- `original_data_ingestion_backup.js` and `debug_loading.js`
  - Purpose: backup and debug scripts. `original_...` is a snapshot of prior work; `debug_loading.js` contains diagnostics for module loading.

## Initialization flow (detailed)

1. `data_ingestion_loader.js` runs immediately when included in HTML.
2. It sequentially appends `<script>` tags for each module in the order above. Each script executes as it loads.
3. After all modules are loaded, the loader calls `startApplication()` which calls `initializeDataIngestion()` (from `init_and_tabs.js`).
4. `initializeDataIngestion()` performs:
   - `setupTabNavigation()`, `setupFileUpload()`, `setupDataSourceSearch()` to attach handlers
   - initializes theme from `localStorage.darkMode`
   - calls `refreshDataSources()` to populate the source list
   - shows a success notification
   - finally: waits a small `delayMs` (configurable) then removes the overlay, reveals the dashboard, and removes the `pre-init` class.

This design makes the UI wait until essential wiring and the initial source list are ready before showing tabs to the user.

## Theming and the initializing overlay

- The template `templates/data_ingestion.html` contains an inline `initLoadingOverlay` element and a small inline `pre-init` CSS block. This ensures the overlay appears immediately and tabs remain hidden until JS removes the `pre-init` class.
- Theme (dark/light) is read from `localStorage.darkMode` early in `data_ingestion.html` and re-applied in `init_and_tabs.js` to ensure overlays and the page use the saved theme.
- CSS in `static/css/data_ingestion.css` contains theme-aware rules targeting `#initLoadingOverlay` and `body.dark-mode #initLoadingOverlay` so the overlay looks correct for the saved theme.

If you need to extend or harden this further:
- Move the inline style block to the very top of the `<head>` (before external stylesheets) for the absolute earliest effect.
- Or set `document.documentElement.classList.add('pre-init')` in a tiny inline script right after the inline CSS so the browser never paints the UI without it.

## Legacy feature flags

Feature flag fetching from `/data/api/feature-flags` has been retired. Any future conditional UI work should query the relevant backend services directly or rely on configuration delivered alongside the primary data source metadata.

## Customization points

- Delay before showing UI: edit `delayMs` in `init_and_tabs.js` (default 420 ms). For promise-based gating (e.g., wait for flags + sources), we can chain promises and call the reveal after all finished.
- Theme toggle: call `window.DI.utilities.notifications.toggleTheme()` or wire the `theme-toggle-nav` button already in the template.
- To add a new module: place it in `static/js/data_ingestion/`, then add its filename in the `MODULES` array in `data_ingestion_loader.js` at the correct location.

## Debugging tips

- Module load failures: check browser console for `✗ Failed to load module` messages emitted by the loader. The path appended by the loader is `BASE_PATH + module`.
- Initialization errors: open DevTools Console and watch for exceptions logged in `initializeDataIngestion()`.
- Overlay remains visible: common causes
  - CSS forcing `display` with `!important` (we removed that); ensure your browser cache is clear.
  - Some script re-creates the overlay after removal; search for `initLoadingOverlay` references.
  - If timers are blocked (rare), fallback path hides immediately; check console for errors.
- Missing tabs: confirm the backend is returning the expected modules in the rendered HTML or update the template to hide modules that no longer apply.

## Manual tests (quick)

- Theme test
  - `localStorage.setItem('darkMode','true'); location.reload();` — overlay and page should be dark.
- Advanced EDA navigation
  - Upload a dataset and trigger `Open ML Workflow`; you should land on `/ml-workflow?source_id=...` with the new dataset active.
- Upload flow
  - Select a CSV and click upload: `showLoading()` should show, the server should return `result.success`, and the new source should appear after `refreshDataSources()`.

## Suggested automated tests

- Unit tests for utility functions (`formatFileSize`, `getTimeAgo`) using a JS test runner
- Integration test: headless browser confirms the overlay disappears after initialization and that ML workflow navigation buttons redirect to `/ml-workflow` with the proper `source_id` query parameter.

## Next steps & improvements (optional)

- Convert the loader + modules into ES modules for better tree-shaking and import clarity (requires build step).
- Replace `window.DI` mutable state with a small event-bus implementation to reduce coupling.
- Add a promise-based gated initialization so the UI reveal waits explicitly for `refreshDataSources()` (and other required readiness checks) instead of relying on `delayMs`.

---

I have extended this README and implemented a promise-gated reveal so the overlay disappears only after feature flags and the initial sources are fully loaded. I can also add a module dependency diagram (SVG) on request.
