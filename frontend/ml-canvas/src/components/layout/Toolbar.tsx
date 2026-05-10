import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Save,
  Loader2,
  FolderOpen,
  History,
  Rocket,
  Wand2,
  HelpCircle,
  Undo2,
  Redo2,
  Keyboard,
  Command,
  Download,
  ChevronDown,
  Clock,
  Sparkles,
  Gauge,
  MoreHorizontal,
  BookOpen,
  Trash2,
} from 'lucide-react';
import { useGraphStore, useTemporalStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { useViewStore } from '../../core/store/useViewStore';
import { getReadOnlyMode, useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { autoLayoutGraph } from '../../core/utils/autoLayout';
import {
  SHOW_SHORTCUTS_EVENT,
  SHOW_PALETTE_EVENT,
  SHOW_TEMPLATES_EVENT,
} from '../../core/hooks/useKeyboardShortcuts';
import { exportCanvasToPng, exportCanvasToSvg } from '../../core/utils/canvasExport';
import { toast } from '../../core/toast';
import { TemplatesGalleryModal } from '../canvas/TemplatesGalleryModal';
import { useConfirm } from '../shared';
import { useDismissable } from './toolbar/_hooks/useDismissable';
import { useRunControls } from './toolbar/_hooks/useRunControls';
import { usePipelineActions } from './toolbar/_hooks/usePipelineActions';
import { CanvasLegend } from './toolbar/CanvasLegend';
import { RecentPipelinesMenu } from './toolbar/RecentPipelinesMenu';
import { VersionLoadMenu } from './toolbar/VersionLoadMenu';

export const Toolbar: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setGraph = useGraphStore((state) => state.setGraph);

  const { toggleDrawer } = useJobStore();
  const isSidebarOpen = useViewStore((s) => s.isSidebarOpen);
  const perfOverlayEnabled = useViewStore((s) => s.perfOverlayEnabled);
  const setPerfOverlayEnabled = useViewStore((s) => s.setPerfOverlayEnabled);
  // Hide editor-only buttons on tablet or when read-only is toggled on.
  const readOnly = useReadOnlyMode();

  // Undo/redo from the temporal substore (zundo). Separate selectors so
  // the toolbar only re-renders when the counts flip across zero.
  const undo = useTemporalStore((s) => s.undo);
  const redo = useTemporalStore((s) => s.redo);
  const canUndo = useTemporalStore((s) => s.pastStates.length > 0);
  const canRedo = useTemporalStore((s) => s.futureStates.length > 0);

  // Clear Canvas: wipe every node + edge after explicit confirmation.
  // Lives next to Undo/Redo because it's the canonical "reset" action;
  // Ctrl+Z still restores the previous state via zundo so this is recoverable.
  const confirm = useConfirm();
  const canClear = !readOnly && (nodes.length > 0 || edges.length > 0);
  const handleClearCanvas = useCallback(async (): Promise<void> => {
    if (nodes.length === 0 && edges.length === 0) return;
    const ok = await confirm({
      title: 'Clear the canvas?',
      message: `Remove all ${nodes.length} node(s) and ${edges.length} edge(s)? You can undo with Ctrl+Z.`,
      confirmLabel: 'Clear canvas',
      variant: 'danger',
    });
    if (ok) setGraph([], []);
  }, [nodes.length, edges.length, confirm, setGraph]);

  // Global undo/redo hotkeys. Skip when focus is in a text input so we
  // don't fight native input undo.
  useEffect(() => {
    const handler = (e: KeyboardEvent): void => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      const isEditable =
        tag === 'INPUT' ||
        tag === 'TEXTAREA' ||
        tag === 'SELECT' ||
        target?.isContentEditable === true;
      if (isEditable) return;
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      // Skip in read-only mode — hotkey must not quietly mutate state behind a hidden button.
      if (getReadOnlyMode()) return;
      const key = e.key.toLowerCase();
      if (key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if ((key === 'z' && e.shiftKey) || key === 'y') {
        e.preventDefault();
        redo();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [undo, redo]);

  const {
    isRunning,
    isRunningAll,
    canRunPreview,
    hasMultipleBranches,
    handleRun,
    handleRunAll,
  } = useRunControls();

  const {
    isSaving,
    hasServerVersions,
    showLoadMenu,
    setShowLoadMenu,
    loadVersions,
    loadVersionsLoading,
    showAllVersions,
    setShowAllVersions,
    showRecentMenu,
    setShowRecentMenu,
    recentPipelines,
    renamingId,
    renameDraft,
    setRenameDraft,
    handleSave,
    openLoadMenu,
    handleLoadVersion,
    openRecentMenu,
    handleRestoreRecent,
    handleClearRecent,
    handleTogglePin,
    startRename,
    commitRename,
    cancelRename,
    handleDeleteRecent,
    formatRelativeTime,
    currentDatasetId,
    exportNotebook,
  } = usePipelineActions();

  const [showLegend, setShowLegend] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  // Compact-toolbar overflow menu (below xl).
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  // Templates gallery modal — controlled here so both the Toolbar button
  // and the canvas empty-state CTA (via event) can open it.
  const [showTemplates, setShowTemplates] = useState(false);

  const moreMenuRef = useRef<HTMLDivElement | null>(null);
  const legendRef = useRef<HTMLDivElement | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const recentMenuRef = useRef<HTMLDivElement | null>(null);
  const loadMenuRef = useRef<HTMLDivElement | null>(null);
  useDismissable(showMoreMenu, () => setShowMoreMenu(false), moreMenuRef);
  useDismissable(showLegend, () => setShowLegend(false), legendRef);
  useDismissable(showExportMenu, () => setShowExportMenu(false), exportMenuRef);
  useDismissable(showRecentMenu, () => setShowRecentMenu(false), recentMenuRef);
  useDismissable(showLoadMenu, () => setShowLoadMenu(false), loadMenuRef);

  const handleExport = async (kind: 'png' | 'svg'): Promise<void> => {
    setShowExportMenu(false);
    setIsExporting(true);
    try {
      const fn = kind === 'png' ? exportCanvasToPng : exportCanvasToSvg;
      const result = await fn(`skyulf-canvas.${kind}`);
      if (!result) {
        toast.error('Export failed', 'Canvas viewport not found');
      } else {
        toast.success(`Canvas exported as ${kind.toUpperCase()}`);
      }
    } catch (err) {
      console.error('Canvas export failed', err);
      toast.error('Export failed', String(err));
    } finally {
      setIsExporting(false);
    }
  };

  // Bridge: canvas empty-state CTA dispatches this event to open the
  // gallery without owning its own modal state.
  useEffect(() => {
    const open = (): void => setShowTemplates(true);
    window.addEventListener(SHOW_TEMPLATES_EVENT, open);
    return () => window.removeEventListener(SHOW_TEMPLATES_EVENT, open);
  }, []);

  return (
    <>
      {/* Left cluster: legend, keyboard help, command palette, undo/redo.
          Shifts right when the sidebar is collapsed so it doesn't overlap
          the floating "Expand" button (z-50, left-4/top-4). */}
      <div
        ref={legendRef}
        className={`absolute top-4 z-10 flex gap-2 transition-[left] duration-300 ${
          isSidebarOpen ? 'left-4' : 'left-16'
        }`}
      >
        <div className="relative">
          <button
            onClick={() => setShowLegend((v) => !v)}
            title="Show node badge legend"
            aria-label="Show node badge legend"
            aria-expanded={showLegend}
            className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
          >
            <HelpCircle className="w-4 h-4" />
          </button>
        </div>
        <button
          onClick={() => window.dispatchEvent(new CustomEvent(SHOW_SHORTCUTS_EVENT))}
          title="Keyboard shortcuts (?)"
          aria-label="Keyboard shortcuts"
          className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors focus-ring"
        >
          <Keyboard className="w-4 h-4" />
        </button>
        {!readOnly && (
          <button
            onClick={() => window.dispatchEvent(new CustomEvent(SHOW_PALETTE_EVENT))}
            title="Command palette (Ctrl/Cmd+K)"
            aria-label="Open command palette"
            className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors focus-ring"
          >
            <Command className="w-4 h-4" />
          </button>
        )}
        {!readOnly && (
          <button
            onClick={() => redo()}
            disabled={!canRedo}
            title="Redo (Ctrl+Shift+Z)"
            aria-label="Redo"
            data-testid="toolbar-redo"
            className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-ring"
          >
            <Redo2 className="w-4 h-4" />
          </button>
        )}
        {!readOnly && (
          <button
            onClick={() => undo()}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            aria-label="Undo"
            data-testid="toolbar-undo"
            className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-ring"
          >
            <Undo2 className="w-4 h-4" />
          </button>
        )}
        {!readOnly && (
          <button
            onClick={() => { void handleClearCanvas(); }}
            disabled={!canClear}
            title="Clear canvas (Ctrl+Z to undo)"
            aria-label="Clear canvas"
            data-testid="toolbar-clear-canvas"
            className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-red-50 hover:text-red-600 hover:border-red-300 dark:hover:bg-red-950/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed focus-ring"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
        {showLegend && <CanvasLegend onClose={() => setShowLegend(false)} />}
      </div>

      {/* Right cluster: history / load / save / tidy / export / run.
          max-width keeps the cluster from sliding under the left cluster. */}
      <div className="absolute top-4 right-4 z-10 flex flex-nowrap justify-end gap-2 max-w-[calc(100%-13rem)]">
        {/* Compact overflow menu (below xl). Collapses secondary actions so
            the cluster never wraps on 1024–1280 px screens. */}
        <div className="relative xl:hidden" ref={moreMenuRef}>
          <button
            onClick={() => setShowMoreMenu((v) => !v)}
            title="More canvas tools"
            aria-label="More canvas tools"
            aria-haspopup="menu"
            aria-expanded={showMoreMenu}
            data-testid="toolbar-more"
            className="flex items-center gap-1 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
          {showMoreMenu && (
            <div
              role="menu"
              aria-label="More canvas tools"
              className="absolute top-full right-0 mt-1 w-52 bg-background border rounded-md shadow-lg overflow-hidden z-20"
            >
              <button
                role="menuitem"
                onClick={() => { setShowMoreMenu(false); toggleDrawer(); }}
                className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
              >
                <History className="w-4 h-4" /> Jobs
              </button>
              {!readOnly && (
                <button
                  role="menuitem"
                  onClick={() => { setShowMoreMenu(false); setShowTemplates(true); }}
                  className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
                >
                  <Sparkles className="w-4 h-4" /> Templates
                </button>
              )}
              <button
                role="menuitemcheckbox"
                onClick={() => {
                  setShowMoreMenu(false);
                  setPerfOverlayEnabled(!perfOverlayEnabled);
                }}
                aria-checked={perfOverlayEnabled}
                className={`w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent ${
                  perfOverlayEnabled ? 'text-primary' : ''
                }`}
              >
                <Gauge className="w-4 h-4" />
                Perf overlay {perfOverlayEnabled ? '· on' : ''}
              </button>
              {!readOnly && (
                <button
                  role="menuitem"
                  onClick={() => {
                    setShowMoreMenu(false);
                    const { nodes: laidOut, edges: keptEdges } = autoLayoutGraph(nodes, edges);
                    setGraph(laidOut, keptEdges);
                  }}
                  disabled={isRunning || nodes.length === 0}
                  className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50"
                >
                  <Wand2 className="w-4 h-4" /> Tidy layout
                </button>
              )}
              <button
                role="menuitem"
                onClick={() => { setShowMoreMenu(false); void handleExport('png'); }}
                disabled={isExporting || nodes.length === 0}
                className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50"
              >
                <Download className="w-4 h-4" /> Export PNG
              </button>
              <button
                role="menuitem"
                onClick={() => { setShowMoreMenu(false); void handleExport('svg'); }}
                disabled={isExporting || nodes.length === 0}
                className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50"
              >
                <Download className="w-4 h-4" /> Export SVG
              </button>
              {currentDatasetId && (
                <>
                  <div className="border-t my-1" />
                  <button
                    role="menuitem"
                    onClick={() => { setShowMoreMenu(false); void exportNotebook('compact'); }}
                    className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
                  >
                    <BookOpen className="w-4 h-4" /> Notebook (compact)
                  </button>
                  <button
                    role="menuitem"
                    onClick={() => { setShowMoreMenu(false); void exportNotebook('full'); }}
                    className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
                  >
                    <BookOpen className="w-4 h-4" /> Notebook (full)
                  </button>
                </>
              )}
            </div>
          )}
        </div>
        <button
          onClick={() => toggleDrawer()}
          title="Job runs history"
          aria-label="Job runs history"
          data-testid="toolbar-jobs"
          className="hidden xl:flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
        >
          <History className="w-4 h-4" />
          <span className="text-sm font-medium hidden xl:inline">Jobs</span>
        </button>
        {!readOnly && (
          <button
            onClick={() => setShowTemplates(true)}
            title="Start from a template"
            aria-label="Start from a template"
            data-testid="toolbar-templates"
            className="hidden xl:flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            <span className="text-sm font-medium hidden xl:inline">Templates</span>
          </button>
        )}
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
          className={`hidden xl:flex items-center gap-2 px-3 py-2 border rounded-md shadow-sm transition-colors ${
            perfOverlayEnabled
              ? 'bg-primary/10 border-primary/40 text-primary hover:bg-primary/15'
              : 'bg-background hover:bg-accent'
          }`}
        >
          <Gauge className="w-4 h-4" />
          <span className="text-sm font-medium hidden xl:inline">Perf</span>
        </button>
        {/* Recent pipelines (localStorage fallback) — only shown when no
            server-side versions exist for the current dataset. */}
        {!readOnly && !hasServerVersions && recentPipelines.length > 0 && (
          <div className="relative" ref={recentMenuRef}>
            <button
              onClick={openRecentMenu}
              title="Per-browser fallback (localStorage). Server-side versions live in DataSources."
              aria-label="Recent pipelines (local fallback)"
              aria-haspopup="menu"
              aria-expanded={showRecentMenu}
              data-testid="toolbar-recent"
              className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
            >
              <Clock className="w-4 h-4" />
              <span className="text-sm font-medium hidden xl:inline">Recent</span>
              <ChevronDown className="w-3 h-3" />
            </button>
            {showRecentMenu && (
              <RecentPipelinesMenu
                recentPipelines={recentPipelines}
                renamingId={renamingId}
                renameDraft={renameDraft}
                onRenameDraftChange={setRenameDraft}
                onRestoreRecent={(e) => void handleRestoreRecent(e)}
                onTogglePin={handleTogglePin}
                onStartRename={startRename}
                onCommitRename={commitRename}
                onCancelRename={cancelRename}
                onDeleteRecent={(e) => void handleDeleteRecent(e)}
                onClearRecent={() => void handleClearRecent()}
                formatRelativeTime={formatRelativeTime}
              />
            )}
          </div>
        )}
        {!readOnly && (
          <div className="relative" ref={loadMenuRef}>
            <button
              onClick={() => { void openLoadMenu(); }}
              disabled={isRunning}
              title="Load a recent pipeline version (latest 5)"
              aria-label="Load pipeline"
              aria-haspopup="menu"
              aria-expanded={showLoadMenu}
              data-testid="toolbar-load"
              className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
            >
              <FolderOpen className="w-4 h-4" />
              <span className="text-sm font-medium hidden xl:inline">Load</span>
              <ChevronDown className="w-3 h-3" />
            </button>
            {showLoadMenu && (
              <VersionLoadMenu
                onClose={() => setShowLoadMenu(false)}
                loadVersions={loadVersions}
                loadVersionsLoading={loadVersionsLoading}
                showAllVersions={showAllVersions}
                onSetShowAllVersions={setShowAllVersions}
                onLoadVersion={(e) => void handleLoadVersion(e)}
              />
            )}
          </div>
        )}
        {!readOnly && (
          <button
            onClick={() => { void handleSave(); }}
            disabled={isSaving || isRunning}
            title="Save pipeline"
            aria-label="Save pipeline"
            data-testid="toolbar-save"
            className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            <span className="text-sm font-medium hidden xl:inline">
              {isSaving ? 'Saving...' : 'Save'}
            </span>
          </button>
        )}
        {!readOnly && (
          <button
            onClick={() => {
              // Tidy: dagre topological layout for multi-branch canvases.
              const { nodes: laidOut, edges: keptEdges } = autoLayoutGraph(nodes, edges);
              setGraph(laidOut, keptEdges);
            }}
            disabled={isRunning || nodes.length === 0}
            title="Auto-arrange nodes left-to-right by data flow"
            aria-label="Tidy: auto-arrange nodes"
            className="hidden xl:flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            <Wand2 className="w-4 h-4" />
            <span className="text-sm font-medium hidden xl:inline">Tidy</span>
          </button>
        )}
        <div className="relative hidden xl:block" ref={exportMenuRef}>
          <button
            onClick={() => setShowExportMenu((v) => !v)}
            disabled={isExporting || nodes.length === 0}
            title="Export canvas as image"
            aria-label="Export canvas as image"
            aria-haspopup="menu"
            aria-expanded={showExportMenu}
            className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            {isExporting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            <span className="text-sm font-medium hidden xl:inline">Export</span>
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
              {currentDatasetId && (
                <>
                  <div className="border-t my-1" />
                  <button
                    role="menuitem"
                    onClick={() => { setShowExportMenu(false); void exportNotebook('compact'); }}
                    className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
                  >
                    <BookOpen className="w-4 h-4" /> Notebook (compact)
                  </button>
                  <button
                    role="menuitem"
                    onClick={() => { setShowExportMenu(false); void exportNotebook('full'); }}
                    className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent"
                  >
                    <BookOpen className="w-4 h-4" /> Notebook (full)
                  </button>
                </>
              )}
            </div>
          )}
        </div>
        {!readOnly && hasMultipleBranches && (
          <button
            onClick={() => { void handleRunAll(); }}
            disabled={isRunningAll || isRunning}
            title="Run all branches as separate experiments"
            aria-label="Run all parallel branches as separate experiments"
            data-testid="toolbar-run-all"
            className="flex items-center gap-2 px-3 py-2 text-white bg-amber-600 rounded-md shadow-sm hover:bg-amber-700 transition-colors disabled:opacity-50"
          >
            {isRunningAll ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Rocket className="w-4 h-4" />
            )}
            <span className="text-sm font-medium hidden 2xl:inline">
              {isRunningAll ? 'Queuing...' : 'Run All Experiments'}
            </span>
          </button>
        )}
        {!readOnly && canRunPreview && (
          <button
            onClick={() => { void handleRun(); }}
            disabled={isRunning}
            title="Run Preview (Ctrl+Enter)"
            aria-label="Run Preview"
            data-testid="toolbar-run-preview"
            className="flex items-center gap-2 px-3 py-2 text-white rounded-md shadow-sm transition-all disabled:opacity-50"
            style={{ background: 'var(--main-gradient)' }}
          >
            {isRunning ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            <span className="text-sm font-medium hidden 2xl:inline">
              {isRunning ? 'Running...' : 'Run Preview'}
            </span>
          </button>
        )}
      </div>
      <TemplatesGalleryModal
        isOpen={showTemplates}
        onClose={() => setShowTemplates(false)}
      />
    </>
  );
};
