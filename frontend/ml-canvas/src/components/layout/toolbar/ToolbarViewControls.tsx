import type { ToolbarState } from './_hooks/useToolbarState';
import { Loader2, History, Wand2, Download, ChevronDown, Sparkles, Gauge } from 'lucide-react';
import { autoLayoutGraph } from '../../../core/utils/autoLayout';
import { NotebookExportMenuItems } from './NotebookExportMenuItems';

export function ToolbarViewControls(
  { view, layout, menus, readOnly }: Pick<ToolbarState, 'view' | 'layout' | 'menus' | 'readOnly'>,
) {
  const { toggleDrawer, perfOverlayEnabled, setPerfOverlayEnabled } = view;
  const { isCompact } = layout;
  const { setShowTemplates } = menus;
  return (<>
    {!isCompact && (
      <button
        onClick={() => toggleDrawer()}
        title="Job runs history"
        aria-label="Job runs history"
        data-testid="toolbar-jobs"
        className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors"
      >
        <History className="w-4 h-4" />
        <span className="text-sm font-medium">Jobs</span>
      </button>
    )}
    {!isCompact && !readOnly && (
      <button
        onClick={() => setShowTemplates(true)}
        title="Start from a template"
        aria-label="Start from a template"
        data-testid="toolbar-templates"
        className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors"
      >
        <Sparkles className="w-4 h-4" />
        <span className="text-sm font-medium">Templates</span>
      </button>
    )}
    {!isCompact && (
      <button
        onClick={() => setPerfOverlayEnabled(!perfOverlayEnabled)}
        title={
          perfOverlayEnabled
            ? 'Hide per-node performance overlay'
            : 'Color-code nodes by last-run duration'
        }
        aria-label="Toggle performance overlay"
        aria-pressed={perfOverlayEnabled}
        data-testid="toolbar-perf-overlay"
        className={`flex items-center gap-2 px-3 py-2 border rounded-md shadow-sm transition-colors ${perfOverlayEnabled
          ? 'bg-primary/10 border-primary/40 text-primary hover:bg-primary/15'
          : 'bg-background hover:bg-accent'
          }`}
      >
        <Gauge className="w-4 h-4" />
        <span className="text-sm font-medium">Perf</span>
      </button>
    )}
  </>);
}

export function ToolbarTidyControl(
  { editing, layout, run, readOnly }: Pick<ToolbarState, 'editing' | 'layout' | 'run' | 'readOnly'>,
) {
  const { nodes, edges, setGraph } = editing;
  const { isCompact } = layout;
  const { isRunning } = run;
  return (<>
    {!isCompact && !readOnly && (
      <button
        onClick={() => {
          // Tidy: dagre topological layout for multi-branch canvases.
          const { nodes: laidOut, edges: keptEdges } = autoLayoutGraph(nodes, edges);
          setGraph(laidOut, keptEdges);
        }}
        disabled={isRunning || nodes.length === 0}
        title="Auto-arrange nodes left-to-right by data flow"
        aria-label="Tidy: auto-arrange nodes"
        className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors disabled:opacity-50"
      >
        <Wand2 className="w-4 h-4" />
        <span className="text-sm font-medium">Tidy</span>
      </button>
    )}
  </>);
}

export function ToolbarExportControl(
  { editing, layout, pipeline, menus }: Pick<ToolbarState, 'editing' | 'layout' | 'pipeline' | 'menus'>,
) {
  const { nodes } = editing;
  const { isCompact } = layout;
  const { currentDatasetId, exportNotebook } = pipeline;
  const { showExportMenu, setShowExportMenu, isExporting, exportMenuRef, handleExport } = menus;
  return (<>
    {!isCompact && (
      <div className="relative" ref={exportMenuRef}>
        <button
          onClick={() => setShowExportMenu((v) => !v)}
          disabled={isExporting || nodes.length === 0}
          title="Export canvas as image"
          aria-label="Export canvas as image"
          aria-haspopup="menu"
          aria-expanded={showExportMenu}
          className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors disabled:opacity-50"
        >
          {isExporting ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Download className="w-4 h-4" />
          )}
          <span className="text-sm font-medium">Export</span>
          <ChevronDown className="w-3 h-3" />
        </button>
        {showExportMenu && (
          <div
            role="menu"
            className="absolute top-full right-0 mt-1 w-40 bg-background border rounded-md shadow-lg overflow-hidden z-20"
          >
            <button
              role="menuitem"
              onClick={() => { void handleExport('png'); }}
              className="w-full text-left px-3 py-2 text-sm hover:bg-accent"
            >
              PNG (high-DPI)
            </button>
            <button
              role="menuitem"
              onClick={() => { void handleExport('svg'); }}
              className="w-full text-left px-3 py-2 text-sm hover:bg-accent"
            >
              SVG (vector)
            </button>
            <NotebookExportMenuItems
              currentDatasetId={currentDatasetId}
              onExportNotebook={exportNotebook}
              onBeforeSelect={() => setShowExportMenu(false)}
            />
          </div>
        )}
      </div>
    )}
  </>);
}
