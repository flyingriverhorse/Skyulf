import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Save,
  Loader2,
  FolderOpen,
  History,
  Rocket,
  Wand2,
  Tag,
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
  Trash2,
} from 'lucide-react';
import { useGraphStore, useTemporalStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { useViewStore } from '../../core/store/useViewStore';
import { getReadOnlyMode, useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { useSidebarOpen } from '../../core/hooks/useSidebarOpen';
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
import { ToolbarIconButton } from './toolbar/ToolbarIconButton';
import { NotebookExportMenuItems } from './toolbar/NotebookExportMenuItems';

export const Toolbar: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setGraph = useGraphStore((state) => state.setGraph);

  const { toggleDrawer } = useJobStore();
  const isSidebarOpen = useSidebarOpen();
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

  // CAN-005: collapse secondary actions based on the *actual* Flow-pane
  // width (the toolbar's absolute-positioned containing block), not a
  // Tailwind viewport breakpoint. `xl:` media queries only track the
  // browser window, so at a wide viewport (e.g. 1440px) with the
  // Properties panel open — which narrows this same pane without
  // changing the window width — the old `hidden xl:flex` buttons stayed
  // visible and rendered on top of the left cluster (Undo/Clear).
  // Measuring the real container closes that gap at every panel
  // combination instead of only at the one viewport width devs tested.
  const toolbarRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const COMPACT_WIDTH = 1280;
  const isCompact = containerWidth < COMPACT_WIDTH;
  const isNarrow = containerWidth < 720;
  const hideUndoRedo = containerWidth < 380;
  useEffect(() => {
    const container = toolbarRef.current?.parentElement;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setContainerWidth(entry.contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const moreMenuRef = useRef<HTMLDivElement | null>(null);
  const legendRef = useRef<HTMLDivElement | null>(null);
  const legendPopoverRef = useRef<HTMLDivElement | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const recentMenuRef = useRef<HTMLDivElement | null>(null);
  const loadMenuRef = useRef<HTMLDivElement | null>(null);
  useDismissable(showMoreMenu, () => setShowMoreMenu(false), moreMenuRef);
  useDismissable(showLegend, () => setShowLegend(false), [legendRef, legendPopoverRef]);
  useDismissable(showExportMenu, () => setShowExportMenu(false), exportMenuRef);
  useDismissable(showRecentMenu, () => setShowRecentMenu(false), recentMenuRef);
  useDismissable(showLoadMenu, () => setShowLoadMenu(false), loadMenuRef);

  // Overflow items disappear when opening another menu. Keep keyboard focus
  // in the replacement content and return it to More when that content closes.
  useEffect(() => {
    if (!isNarrow) return;
    const popover = showLegend ? legendPopoverRef.current
      : showLoadMenu ? loadMenuRef.current
      : showRecentMenu ? recentMenuRef.current : null;
    if (!popover) return;
    const trigger = moreMenuRef.current?.querySelector('button');
    const target = popover.querySelector<HTMLElement>('[role="menu"], button:not(:disabled)');
    target?.focus({ preventScroll: true });
    return () => {
      if (document.activeElement === document.body || popover.contains(document.activeElement)) {
        trigger?.focus({ preventScroll: true });
      }
    };
  }, [isNarrow, showLegend, showLoadMenu, showRecentMenu]);

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
      {/* Shared layout prevents the action groups from occupying the same space.
          Reserve the floating sidebar toggle's slot when the library is closed. */}
      <div
        ref={toolbarRef}
        data-canvas-toolbar
        className={`absolute top-4 right-4 flex items-start gap-2 ${showMoreMenu || showLoadMenu || showRecentMenu || showExportMenu ? 'z-40' : 'z-10'} ${isSidebarOpen || readOnly ? 'left-4' : 'left-16'}`}
      >
      <div
        ref={legendRef}
        className="flex shrink-0 gap-2"
      >
        {!isNarrow && <>
        <div className="relative">
          <ToolbarIconButton
            icon={<Tag className="w-4 h-4" />}
            onClick={() => setShowLegend((v) => !v)}
            title="Show node badge legend"
            ariaLabel="Show node badge legend"
            ariaExpanded={showLegend}
          />
        </div>
        <ToolbarIconButton
          icon={<Keyboard className="w-4 h-4" />}
          onClick={() => window.dispatchEvent(new CustomEvent(SHOW_SHORTCUTS_EVENT))}
          title="Keyboard shortcuts (?)"
          ariaLabel="Keyboard shortcuts"
        />
        {!readOnly && (
          <ToolbarIconButton
            icon={<Command className="w-4 h-4" />}
            onClick={() => window.dispatchEvent(new CustomEvent(SHOW_PALETTE_EVENT))}
            title="Command palette (Ctrl/Cmd+K)"
            ariaLabel="Open command palette"
          />
        )}
        </>}
        {!readOnly && !hideUndoRedo && (
          <ToolbarIconButton
            icon={<Redo2 className="w-4 h-4" />}
            onClick={() => redo()}
            disabled={!canRedo}
            title="Redo (Ctrl+Shift+Z)"
            ariaLabel="Redo"
            testId="toolbar-redo"
          />
        )}
        {!readOnly && !hideUndoRedo && (
          <ToolbarIconButton
            icon={<Undo2 className="w-4 h-4" />}
            onClick={() => undo()}
            disabled={!canUndo}
            title="Undo (Ctrl+Z)"
            ariaLabel="Undo"
            testId="toolbar-undo"
          />
        )}
        {!readOnly && !isNarrow && (
          <ToolbarIconButton
            icon={<Trash2 className="w-4 h-4" />}
            onClick={() => { void handleClearCanvas(); }}
            disabled={!canClear}
            title="Clear canvas (Ctrl+Z to undo)"
            ariaLabel="Clear canvas"
            testId="toolbar-clear-canvas"
            variant="danger"
          />
        )}
      </div>
      <div className="flex flex-1 min-w-0 flex-wrap justify-end gap-2 [&>button]:shrink-0">
        {/* Compact overflow menu — collapses secondary actions so the
            cluster never overlaps the left cluster once the live Flow-pane
            width (not the viewport) drops below COMPACT_WIDTH. */}
        {isCompact && (
        <div className="relative" ref={moreMenuRef}>
          <button
            onClick={() => setShowMoreMenu((v) => !v)}
            title="More canvas tools"
            aria-label="More canvas tools"
            aria-haspopup="menu"
            aria-expanded={showMoreMenu}
            data-testid="toolbar-more"
            className="flex items-center gap-1 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>
          {showMoreMenu && (
            <div
              role="menu"
              aria-label="More canvas tools"
              className="absolute top-full right-0 mt-1 w-52 max-w-[calc(100vw-2rem)] max-h-[calc(100dvh-11rem)] overflow-y-auto overscroll-contain bg-background border rounded-md shadow-lg z-20"
            >
              {isNarrow && (
                <>
                  <button role="menuitem" onClick={() => { setShowMoreMenu(false); setShowLegend(true); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent">
                    <Tag className="w-4 h-4" /> Node badge legend
                  </button>
                  <button role="menuitem" onClick={() => { setShowMoreMenu(false); window.dispatchEvent(new CustomEvent(SHOW_SHORTCUTS_EVENT)); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent">
                    <Keyboard className="w-4 h-4" /> Keyboard shortcuts
                  </button>
                  {!readOnly && <>
                    <button role="menuitem" onClick={() => { setShowMoreMenu(false); window.dispatchEvent(new CustomEvent(SHOW_PALETTE_EVENT)); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent">
                      <Command className="w-4 h-4" /> Command palette
                    </button>
                    {hideUndoRedo && <>
                      <button role="menuitem" disabled={!canUndo} onClick={() => { setShowMoreMenu(false); undo(); }} className="w-full text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">Undo</button>
                      <button role="menuitem" disabled={!canRedo} onClick={() => { setShowMoreMenu(false); redo(); }} className="w-full text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">Redo</button>
                    </>}
                    <button role="menuitem" disabled={!canClear} onClick={() => { setShowMoreMenu(false); void handleClearCanvas(); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">
                      <Trash2 className="w-4 h-4" /> Clear canvas
                    </button>
                    {!hasServerVersions && recentPipelines.length > 0 && (
                      <button role="menuitem" onClick={() => { setShowMoreMenu(false); openRecentMenu(); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent">
                        <Clock className="w-4 h-4" /> Recent pipelines
                      </button>
                    )}
                    <button role="menuitem" disabled={isRunning} onClick={() => { setShowMoreMenu(false); void openLoadMenu(); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">
                      <FolderOpen className="w-4 h-4" /> Load pipeline
                    </button>
                    <button role="menuitem" disabled={isSaving || isRunning} onClick={() => { setShowMoreMenu(false); void handleSave(); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">
                      <Save className="w-4 h-4" /> {isSaving ? 'Saving...' : 'Save pipeline'}
                    </button>
                    {hasMultipleBranches && (
                      <button role="menuitem" disabled={isRunningAll || isRunning} onClick={() => { setShowMoreMenu(false); void handleRunAll(); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">
                        <Rocket className="w-4 h-4" /> {isRunningAll ? 'Queuing experiments...' : 'Run all experiments'}
                      </button>
                    )}
                  </>}
                </>
              )}
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
              <NotebookExportMenuItems
                currentDatasetId={currentDatasetId}
                onExportNotebook={exportNotebook}
                onBeforeSelect={() => setShowMoreMenu(false)}
              />
            </div>
          )}
        </div>
        )}
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
          className={`flex items-center gap-2 px-3 py-2 border rounded-md shadow-sm transition-colors ${
            perfOverlayEnabled
              ? 'bg-primary/10 border-primary/40 text-primary hover:bg-primary/15'
              : 'bg-background hover:bg-accent'
          }`}
        >
          <Gauge className="w-4 h-4" />
          <span className="text-sm font-medium">Perf</span>
        </button>
        )}
        {/* Recent pipelines (localStorage fallback) — only shown when no
            server-side versions exist for the current dataset. */}
        {!readOnly && !hasServerVersions && recentPipelines.length > 0 && (!isNarrow || showRecentMenu) && (
          <div className={isNarrow ? 'absolute top-full right-0' : 'relative'} ref={recentMenuRef}>
            {!isNarrow && (
            <button
              onClick={openRecentMenu}
              title="Per-browser fallback (localStorage). Server-side versions live in DataSources."
              aria-label="Recent pipelines (local fallback)"
              aria-haspopup="menu"
              aria-expanded={showRecentMenu}
              data-testid="toolbar-recent"
              className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors"
            >
              <Clock className="w-4 h-4" />
              {!isCompact && <span className="text-sm font-medium">Recent</span>}
              <ChevronDown className="w-3 h-3" />
            </button>
            )}
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
        {!readOnly && (!isNarrow || showLoadMenu) && (
          <div className={isNarrow ? 'absolute top-full right-0' : 'relative'} ref={loadMenuRef}>
            {!isNarrow && (
            <button
              onClick={() => { void openLoadMenu(); }}
              disabled={isRunning}
              title="Load a recent pipeline version (latest 5)"
              aria-label="Load pipeline"
              aria-haspopup="menu"
              aria-expanded={showLoadMenu}
              data-testid="toolbar-load"
              className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors disabled:opacity-50"
            >
              <FolderOpen className="w-4 h-4" />
              {!isCompact && <span className="text-sm font-medium">Load</span>}
              <ChevronDown className="w-3 h-3" />
            </button>
            )}
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
        {!readOnly && !isNarrow && (
          <button
            onClick={() => { void handleSave(); }}
            disabled={isSaving || isRunning}
            title="Save pipeline"
            aria-label="Save pipeline"
            data-testid="toolbar-save"
            className="flex items-center gap-2 px-3 py-2 action-secondary rounded-md shadow-sm transition-colors disabled:opacity-50"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            {!isCompact && (
              <span className="text-sm font-medium">
                {isSaving ? 'Saving...' : 'Save'}
              </span>
            )}
          </button>
        )}
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
        {!readOnly && !isNarrow && hasMultipleBranches && (
          <button
            onClick={() => { void handleRunAll(); }}
            disabled={isRunningAll || isRunning}
            title="Run all branches as separate experiments"
            aria-label="Run all parallel branches as separate experiments"
            data-testid="toolbar-run-all"
            className="flex items-center gap-2 px-3 py-2 action-primary rounded-md shadow-sm transition-colors disabled:opacity-50"
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
        {!readOnly && (
          <button
            onClick={() => { void handleRun(); }}
            disabled={isRunning}
            title="Preview data (Ctrl+Enter). Click to review any blocking issues."
            data-testid="toolbar-run-preview"
            className="flex shrink-0 items-center gap-2 px-3 py-2 action-primary rounded-md shadow-sm transition-all disabled:opacity-50 focus-ring"
          >
            {isRunning ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            <span className="text-sm font-medium whitespace-nowrap">
              {isRunning ? 'Previewing data...' : 'Preview data'}
            </span>
          </button>
        )}
      </div>
      </div>
      {showLegend && (
        <div ref={legendPopoverRef} className={`absolute top-4 right-4 z-40 ${isSidebarOpen || readOnly ? 'left-4' : 'left-16'}`}>
          <CanvasLegend onClose={() => setShowLegend(false)} />
        </div>
      )}
      <TemplatesGalleryModal
        isOpen={showTemplates}
        onClose={() => setShowTemplates(false)}
      />
    </>
  );
};
