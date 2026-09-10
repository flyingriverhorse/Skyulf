import type { ToolbarState } from './_hooks/useToolbarState';
import { Save, FolderOpen, History, Rocket, Wand2, Tag, Keyboard, Command, Download, Clock, Sparkles, Gauge, MoreHorizontal, Trash2 } from 'lucide-react';
import { autoLayoutGraph } from '../../../core/utils/autoLayout';
import { SHOW_SHORTCUTS_EVENT, SHOW_PALETTE_EVENT } from '../../../core/hooks/useKeyboardShortcuts';
import { NotebookExportMenuItems } from './NotebookExportMenuItems';

function ExperimentMenuItem(
  { run, menus }: Pick<ToolbarState, 'run' | 'menus'>,
) {
  const { isRunning, isRunningAll, hasMultipleBranches } = run;
  const { setReviewingExperiments, setShowMoreMenu } = menus;
  return (<>
    {hasMultipleBranches && (
      <button role="menuitem" disabled={isRunningAll || isRunning} onClick={() => { setShowMoreMenu(false); setReviewingExperiments(true); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent disabled:opacity-50">
        <Rocket className="w-4 h-4" /> {isRunningAll ? 'Queuing experiments...' : 'Run all experiments'}
      </button>
    )}
  </>);
}

function PipelineMenuItems(
  { run, pipeline, menus }: Pick<ToolbarState, 'run' | 'pipeline' | 'menus'>,
) {
  const { isRunning } = run;
  const {
    isSaving,
    hasServerVersions,
    recentPipelines,
    handleSave,
    openLoadMenu,
    openRecentMenu,
  } = pipeline;
  const { setShowMoreMenu } = menus;
  return (<>
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
    <ExperimentMenuItem run={run} menus={menus} />
  </>);
}

function NarrowEditingMenuItems(
  { editing, layout, run, pipeline, menus, readOnly }: Pick<ToolbarState, 'editing' | 'layout' | 'run' | 'pipeline' | 'menus' | 'readOnly'>,
) {
  const { undo, redo, canUndo, canRedo, canClear, handleClearCanvas } = editing;
  const { hideUndoRedo } = layout;


  const { setShowMoreMenu } = menus;
  return (<>
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
      <PipelineMenuItems run={run} pipeline={pipeline} menus={menus} />
    </>}
  </>);
}

function NarrowMenuItems(
  { editing, layout, run, pipeline, menus, readOnly }: Pick<ToolbarState, 'editing' | 'layout' | 'run' | 'pipeline' | 'menus' | 'readOnly'>,
) {

  const { isNarrow } = layout;


  const { setShowLegend, setShowMoreMenu } = menus;
  return (<>
    {isNarrow && (
      <>
        <button role="menuitem" onClick={() => { setShowMoreMenu(false); setShowLegend(true); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent">
          <Tag className="w-4 h-4" /> Node badge legend
        </button>
        <button role="menuitem" onClick={() => { setShowMoreMenu(false); window.dispatchEvent(new CustomEvent(SHOW_SHORTCUTS_EVENT)); }} className="w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent">
          <Keyboard className="w-4 h-4" /> Keyboard shortcuts
        </button>
        <NarrowEditingMenuItems editing={editing} layout={layout} run={run} pipeline={pipeline} menus={menus} readOnly={readOnly} />
      </>
    )}
  </>);
}

function ViewMenuItems(
  { view, menus, readOnly }: Pick<ToolbarState, 'view' | 'menus' | 'readOnly'>,
) {
  const { toggleDrawer, perfOverlayEnabled, setPerfOverlayEnabled } = view;
  const { setShowMoreMenu, setShowTemplates } = menus;
  return (<>
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
      className={`w-full flex items-center gap-2 text-left px-3 py-2 text-sm hover:bg-accent ${perfOverlayEnabled ? 'text-primary' : ''
        }`}
    >
      <Gauge className="w-4 h-4" />
      Perf overlay {perfOverlayEnabled ? '· on' : ''}
    </button>
  </>);
}

function CanvasExportMenuItems(
  { editing, pipeline, menus }: Pick<ToolbarState, 'editing' | 'pipeline' | 'menus'>,
) {
  const { nodes } = editing;
  const { currentDatasetId, exportNotebook } = pipeline;
  const { isExporting, setShowMoreMenu, handleExport } = menus;
  return (<>
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
  </>);
}

export function ToolbarOverflowMenu(
  { editing, view, layout, run, pipeline, menus, readOnly }: Pick<ToolbarState, 'editing' | 'view' | 'layout' | 'run' | 'pipeline' | 'menus' | 'readOnly'>,
) {
  const { nodes, edges, setGraph } = editing;

  const { isCompact } = layout;
  const { isRunning } = run;

  const { showMoreMenu, setShowMoreMenu, moreMenuRef } = menus;
  return (<>
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
            <NarrowMenuItems editing={editing} layout={layout} run={run} pipeline={pipeline} menus={menus} readOnly={readOnly} />
            <ViewMenuItems view={view} menus={menus} readOnly={readOnly} />
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
            <CanvasExportMenuItems editing={editing} pipeline={pipeline} menus={menus} />
          </div>
        )}
      </div>
    )}
  </>);
}
