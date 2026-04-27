import React, { useState, useMemo, useEffect, useRef } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { Play, Save, Loader2, FolderOpen, History, Rocket, Wand2, HelpCircle, Merge, GitFork, X, CheckCircle2, XCircle, Undo2, Redo2, Keyboard, AlertCircle, Command, Download, ChevronDown, Clock, Trash2, Pin, PinOff, Pencil } from 'lucide-react';
import { useGraphStore, useTemporalStore } from '../../core/store/useGraphStore';
import { useJobStore } from '../../core/store/useJobStore';
import { useViewStore } from '../../core/store/useViewStore';
import { getReadOnlyMode, useReadOnlyMode } from '../../core/hooks/useReadOnlyMode';
import { runPipelinePreview, savePipeline, fetchPipeline } from '../../core/api/client';
import { convertGraphToPipelineConfig } from '../../core/utils/pipelineConverter';
import { autoLayoutGraph } from '../../core/utils/autoLayout';
import { jobsApi } from '../../core/api/jobs';
import {
  RUN_PREVIEW_EVENT,
  SHOW_SHORTCUTS_EVENT,
  SHOW_PALETTE_EVENT,
} from '../../core/hooks/useKeyboardShortcuts';
import { exportCanvasToPng, exportCanvasToSvg } from '../../core/utils/canvasExport';
import {
  getRecentPipelines,
  pushRecentPipeline,
  clearRecentPipelines,
  togglePinRecentPipeline,
  renameRecentPipeline,
  deleteRecentPipeline,
  type RecentPipelineEntry,
} from '../../core/utils/recentPipelines';
import { toast } from '../../core/toast';
import { useConfirm } from '../shared';

const TRAINING_TYPES = new Set(['basic_training', 'advanced_tuning']);

export const Toolbar: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const setExecutionResult = useGraphStore((state) => state.setExecutionResult);
  const setGraph = useGraphStore((state) => state.setGraph);
  
  const { toggleDrawer, setActiveParallelRun, startPolling } = useJobStore();
  const isSidebarOpen = useViewStore((s) => s.isSidebarOpen);
  // Hide editor-only buttons (save/tidy/run/undo/redo/load/palette) on
  // tablet or when the user has toggled read-only on.
  const readOnly = useReadOnlyMode();
  const confirm = useConfirm();

  // Undo/redo state from the temporal substore (zundo). Keeping these
  // as separate selectors so the toolbar only re-renders when the
  // counts actually flip across zero.
  const undo = useTemporalStore((s) => s.undo);
  const redo = useTemporalStore((s) => s.redo);
  const canUndo = useTemporalStore((s) => s.pastStates.length > 0);
  const canRedo = useTemporalStore((s) => s.futureStates.length > 0);

  // Global undo/redo hotkeys: Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z (or Ctrl+Y).
  // We intentionally skip when focus is in a text input/textarea/contentEditable
  // so we don't fight native input undo.
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
      // Undo/redo are graph mutations: skip in read-only mode so the
      // hotkey doesn't quietly mutate state behind a hidden button.
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
  
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isRunningAll, setIsRunningAll] = useState(false);
  const [showLegend, setShowLegend] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  // Recent pipelines ring buffer (M2). Hydrated lazily on first open
  // so the toolbar mount cost stays zero, then refreshed on every save.
  const [showRecentMenu, setShowRecentMenu] = useState(false);
  const [recentPipelines, setRecentPipelines] = useState<RecentPipelineEntry[]>([]);
  // Inline rename state — `null` when no row is being edited.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');

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

  // Run Preview is only meaningful when the graph has a dataset node
  // wired into at least one downstream node. Hiding the button (and
  // gating the Ctrl+Enter hotkey) prevents the "No dataset node found!"
  // alert and avoids dispatching empty pipelines to the backend.
  const canRunPreview = useMemo(() => {
    const datasetNode = nodes.find(
      (n) => n.data.definitionType === 'dataset_node'
    );
    if (!datasetNode) return false;
    const datasetId = datasetNode.data.datasetId as string | undefined;
    if (!datasetId) return false;
    return edges.some((e) => e.source === datasetNode.id);
  }, [nodes, edges]);

  const hasMultipleBranches = useMemo(() => {
    const trainingNodes = nodes.filter(
      (n) => TRAINING_TYPES.has(n.data.definitionType as string) && edges.some(e => e.target === n.id)
    );
    if (trainingNodes.length < 2) return false;

    // Check that at least two training nodes are on separate branches
    // by comparing their immediate parent sets
    const parentSets = trainingNodes.map(tn =>
      new Set(edges.filter(e => e.target === tn.id).map(e => e.source))
    );
    // If any two training nodes have completely different parents, they're separate
    for (let i = 0; i < parentSets.length; i++) {
      for (let j = i + 1; j < parentSets.length; j++) {
        const overlap = [...parentSets[i]!].some(p => parentSets[j]!.has(p));
        if (!overlap) return true;
      }
    }
    // Even with shared parents, 2+ connected training nodes = separate branches
    return trainingNodes.length >= 2;
  }, [nodes, edges]);

  const getPipelinePayload = () => ({
    nodes: nodes.map(n => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: {
        ...n.data,
        catalogType: n.data.catalogType || n.data.definitionType
      }
    })),
    edges: edges.map(e => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
      type: e.type,
      data: e.data
    }))
  });



  const handleSave = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    const datasetName = datasetNode?.data.datasetName as string | undefined;

    if (!datasetId) {
      toast.error('No dataset node found', 'Cannot save pipeline without a dataset context.');
      return;
    }

    setIsSaving(true);
    try {
      await savePipeline(datasetId, {
        name: 'My Pipeline', // TODO: Add UI for naming
        description: 'Saved from Canvas',
        graph: getPipelinePayload()
      });
      // Mirror into the local recent-pipelines ring buffer so the user
      // can roll back to this exact graph from the Toolbar even if the
      // server copy gets overwritten by a later save. Best-effort —
      // never fail the Save toast on a localStorage error.
      pushRecentPipeline({
        name: 'My Pipeline',
        datasetId,
        ...(datasetName ? { datasetName } : {}),
        nodes,
        edges,
      });
      toast.success('Pipeline saved');
    } catch {
      toast.error('Failed to save pipeline');
    } finally {
      setIsSaving(false);
    }
  };

  const handleLoad = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      toast.error('No dataset node found', 'Cannot load pipeline without a dataset context.');
      return;
    }

    const ok = await confirm({
      title: 'Load saved pipeline?',
      message: 'Loading a pipeline will overwrite your current work. Continue?',
      confirmLabel: 'Load',
      variant: 'danger',
    });
    if (!ok) return;
    setIsLoading(true);
    try {
      const pipeline = await fetchPipeline(datasetId);
      if (pipeline) {
        const graphNodes: Node[] = Array.isArray(pipeline.graph.nodes) ? (pipeline.graph.nodes as Node[]) : [];
        const graphEdges: Edge[] = Array.isArray(pipeline.graph.edges) ? (pipeline.graph.edges as Edge[]) : [];
        setGraph(graphNodes, graphEdges);
        toast.success('Pipeline loaded');
      } else {
        toast.info('No saved pipeline found for this dataset');
      }
    } catch {
      toast.error('Failed to load pipeline');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunAll = async () => {
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      toast.error('No dataset node found');
      return;
    }

    setIsRunningAll(true);
    try {
      // Exclude Data Preview nodes from "Run All Experiments" -- preview is
      // an inspection-only sink and should never be queued as a training
      // experiment. Backend treats data_preview as a terminal in
      // partition_parallel_pipeline, so leaving it in would spawn a bogus
      // "preview" job alongside the real training runs.
      const previewIds = new Set(
        nodes.filter(n => n.data.definitionType === 'data_preview').map(n => n.id),
      );
      const filteredNodes = nodes.filter(n => !previewIds.has(n.id));
      const filteredEdges = edges.filter(
        e => !previewIds.has(e.source) && !previewIds.has(e.target),
      );
      const pipelineConfig = convertGraphToPipelineConfig(filteredNodes, filteredEdges);
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        job_type: 'basic_training',
      });
      const count = response.job_ids?.length || 1;
      if (response.job_ids?.length > 1) {
        setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
        startPolling();
      }
      toast.success(`${count} experiment${count > 1 ? 's' : ''} queued`);
      toggleDrawer();
      
      // Keep standard nodes populated with data while experiments run in the background.
      void handleRun();
    } catch {
      toast.error('Failed to run experiments');
    } finally {
      setIsRunningAll(false);
    }
  };

  const handleRun = async () => {
    // Find dataset node
    const datasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;

    if (!datasetId) {
      toast.error('No dataset node found');
      return;
    }

    setIsRunning(true);
    setExecutionResult(null); // Clear previous results

    try {
      // Exclude Data Preview nodes from the toolbar's "Run Preview". Data
      // Preview is an inspection sink with its own "Run Preview" button
      // inside the node settings panel; including it here would queue a
      // redundant preview job alongside the pipeline preview.
      const previewIds = new Set(
        nodes.filter(n => n.data.definitionType === 'data_preview').map(n => n.id),
      );
      const filteredNodes = nodes.filter(n => !previewIds.has(n.id));
      const filteredEdges = edges.filter(
        e => !previewIds.has(e.source) && !previewIds.has(e.target),
      );
      const pipelineConfig = convertGraphToPipelineConfig(filteredNodes, filteredEdges);
      const result = await runPipelinePreview(pipelineConfig);

      setExecutionResult(result);
    } catch (error) {
      console.error('Pipeline failed:', error);
      toast.error('Pipeline execution failed', 'Check console for details.');
    } finally {
      setIsRunning(false);
    }
  };

  // Bridge: the global keyboard hook (Ctrl/Cmd+Enter) dispatches a
  // CustomEvent so we don't have to lift handleRun into a store. We
  // route through a ref to avoid re-registering the listener on every
  // render and to always call the latest closure (which captures
  // current `nodes`/`edges`). The closure also gates on `canRunPreview`
  // so the hotkey is a no-op when the toolbar Run Preview button is hidden.
  const handleRunRef = useRef<() => void>(() => {});
  handleRunRef.current = () => {
    if (!isRunning && canRunPreview) void handleRun();
  };
  useEffect(() => {
    const fire = (): void => handleRunRef.current();
    window.addEventListener(RUN_PREVIEW_EVENT, fire);
    return () => window.removeEventListener(RUN_PREVIEW_EVENT, fire);
  }, []);

  // Recent-pipelines dropdown: hydrate lazily on open so the toolbar
  // doesn't touch localStorage on every mount. We re-read on each open
  // so a save in another tab is reflected next time the menu pops.
  const openRecentMenu = (): void => {
    setRecentPipelines(getRecentPipelines());
    setShowRecentMenu(v => !v);
  };

  const handleRestoreRecent = async (entry: RecentPipelineEntry): Promise<void> => {
    setShowRecentMenu(false);
    // Warn when the snapshot was saved against a different dataset than
    // the one currently loaded. Restoring is still allowed (some nodes
    // are dataset-agnostic) but the user should be aware schema-bound
    // nodes (column pickers, target column, …) may break.
    const currentDatasetNode = nodes.find(n => n.data.definitionType === 'dataset_node');
    const currentDatasetId = currentDatasetNode?.data.datasetId as string | undefined;
    const mismatch =
      entry.datasetId !== undefined &&
      currentDatasetId !== undefined &&
      entry.datasetId !== currentDatasetId;
    const message = mismatch
      ? `This snapshot was saved against "${entry.datasetName ?? entry.datasetId}", but the canvas is currently on a different dataset. Column-bound nodes may need to be reconfigured. Continue?`
      : 'Loading this snapshot will overwrite your current canvas. Continue?';
    const ok = await confirm({
      title: `Restore "${entry.name}"?`,
      message,
      confirmLabel: 'Restore',
      variant: 'danger',
    });
    if (!ok) return;
    setGraph(entry.nodes, entry.edges);
    toast.success(`Restored "${entry.name}"`);
  };

  const handleClearRecent = async (): Promise<void> => {
    const ok = await confirm({
      title: 'Clear pipeline history?',
      message: 'This removes all 5 recent snapshots from this browser. The server-side saved pipeline is unaffected.',
      confirmLabel: 'Clear',
      variant: 'danger',
    });
    if (!ok) return;
    clearRecentPipelines();
    setRecentPipelines([]);
    setShowRecentMenu(false);
    toast.success('Pipeline history cleared');
  };

  // Pin / unpin keeps a snapshot from being evicted by FIFO so the user
  // can lock in a known-good state alongside the rolling recent slots.
  const handleTogglePin = (entry: RecentPipelineEntry): void => {
    const updated = togglePinRecentPipeline(entry.id);
    setRecentPipelines(updated);
    toast.success(entry.pinned ? 'Unpinned' : 'Pinned');
  };

  const startRename = (entry: RecentPipelineEntry): void => {
    setRenamingId(entry.id);
    setRenameDraft(entry.name);
  };

  const commitRename = (): void => {
    if (renamingId === null) return;
    const updated = renameRecentPipeline(renamingId, renameDraft);
    // renameRecentPipeline silently rejects empty / clashing names; if
    // the list didn't change we surface a small toast so the user knows
    // why the row didn't update.
    const changed = updated.find((e) => e.id === renamingId)?.name === renameDraft.trim();
    if (!changed && renameDraft.trim()) {
      toast.error('Rename failed', 'Name is empty or already used by another snapshot.');
    }
    setRecentPipelines(updated);
    setRenamingId(null);
    setRenameDraft('');
  };

  const cancelRename = (): void => {
    setRenamingId(null);
    setRenameDraft('');
  };

  const handleDeleteRecent = async (entry: RecentPipelineEntry): Promise<void> => {
    const ok = await confirm({
      title: `Delete "${entry.name}"?`,
      message: 'This snapshot will be removed from your local history. The server-side saved pipeline is unaffected.',
      confirmLabel: 'Delete',
      variant: 'danger',
    });
    if (!ok) return;
    const updated = deleteRecentPipeline(entry.id);
    setRecentPipelines(updated);
    toast.success('Snapshot deleted');
  };

  // Compact human-readable "X minutes ago" — avoids pulling in a
  // formatting library for one tiny use site.
  const formatRelativeTime = (iso: string): string => {
    const diffMs = Date.now() - new Date(iso).getTime();
    if (Number.isNaN(diffMs) || diffMs < 0) return 'just now';
    const sec = Math.floor(diffMs / 1000);
    if (sec < 45) return 'just now';
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m ago`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h ago`;
    const day = Math.floor(hr / 24);
    if (day < 30) return `${day}d ago`;
    return new Date(iso).toLocaleDateString();
  };

  return (
    <>
      {/* Left-side cluster: legend, keyboard help, redo, undo (in
          reverse of the previous right-side order so destructive
          actions stay closest to the canvas action cluster on the
          right). The dropdown anchor below switches from right-0 to
          left-0 to follow the move.
          When the Sidebar is collapsed, its floating "Expand
          Components" button sits at left-4/top-4 with z-50 and would
          cover the legend button below; we shift the cluster right
          (left-16) so both stay reachable. */}
      <div
        className={`absolute top-4 z-10 flex gap-2 transition-[left] duration-300 ${
          isSidebarOpen ? 'left-4' : 'left-16'
        }`}
      >
        <button
          onClick={() => setShowLegend(v => !v)}
          title="Show node badge legend"
          aria-label="Show node badge legend"
          aria-expanded={showLegend}
          className="flex items-center justify-center w-10 h-10 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
        >
          <HelpCircle className="w-4 h-4" />
        </button>
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
        {showLegend && (
          <div className="absolute top-12 left-0 mt-2 w-96 p-4 bg-background border rounded-md shadow-lg text-sm max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold">Canvas Legend</h3>
              <button
                onClick={() => setShowLegend(false)}
                className="p-1 rounded hover:bg-accent"
                aria-label="Close legend"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Node Badges</div>
            <ul className="space-y-3 mb-4">
              <li className="flex items-start gap-3">
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 text-[10px] font-semibold shrink-0 mt-0.5">
                  <Merge size={10} />2
                </span>
                <div>
                  <div className="font-medium">Safe merge</div>
                  <div className="text-xs text-muted-foreground">Multiple inputs combined cleanly (no overlapping columns).</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-600 dark:text-amber-400 ring-1 ring-amber-500/40 text-[10px] font-semibold shrink-0 mt-0.5">
                  <Merge size={10} />2
                </span>
                <div>
                  <div className="font-medium">Risky merge</div>
                  <div className="text-xs text-muted-foreground">Inputs share columns &mdash; one branch wins (overwrite). Check Results banner; tweak strategy in properties.</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-500 text-[10px] font-semibold shrink-0 mt-0.5">
                  <GitFork size={10} />2
                </span>
                <div>
                  <div className="font-medium">Parallel experiments</div>
                  <div className="text-xs text-muted-foreground">Training/tuning node runs each upstream branch as a separate experiment (no merge).</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-red-500/15 text-red-500 ring-1 ring-red-500/40 shrink-0 mt-0.5">
                  <AlertCircle size={10} />
                </span>
                <div>
                  <div className="font-medium">Configuration issue</div>
                  <div className="text-xs text-muted-foreground">Node has missing or invalid settings. Hover the badge for the specific message; open the properties panel to fix it.</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-green-50 text-green-700 border border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-900 shrink-0 mt-0.5">
                  <CheckCircle2 size={10} />
                </span>
                <div>
                  <div className="font-medium">Success</div>
                  <div className="text-xs text-muted-foreground">Node ran successfully in the last preview / run.</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <span className="inline-flex items-center justify-center w-[22px] h-[22px] rounded-full bg-red-50 text-red-700 border border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-900 shrink-0 mt-0.5">
                  <XCircle size={10} />
                </span>
                <div>
                  <div className="font-medium">Failed</div>
                  <div className="text-xs text-muted-foreground">Node errored. Click it and open the Results panel for the traceback.</div>
                </div>
              </li>
            </ul>

            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Edges</div>
            <ul className="space-y-3">
              <li className="flex items-start gap-3">
                <svg width="44" height="12" className="shrink-0 mt-1">
                  <line x1="0" y1="6" x2="44" y2="6" stroke="#6366f1" strokeWidth="2" strokeDasharray="8 6" />
                </svg>
                <div>
                  <div className="font-medium">Standard edge</div>
                  <div className="text-xs text-muted-foreground">Animated dashed indigo line. Default flow from source to target.</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <svg width="44" height="12" className="shrink-0 mt-1">
                  <line x1="0" y1="6" x2="44" y2="6" stroke="hsl(0, 80%, 65%)" strokeWidth="2" strokeDasharray="8 6" />
                </svg>
                <div>
                  <div className="font-medium">Branch-colored edge</div>
                  <div className="text-xs text-muted-foreground">Dashed line in a per-branch HSL color (auto-generated, one hue per training/tuning terminal). Appears once 2+ training nodes form parallel branches.</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <svg width="44" height="12" className="shrink-0 mt-1">
                  <line x1="0" y1="6" x2="44" y2="6" stroke="hsl(0, 80%, 65%)" strokeWidth="2" strokeDasharray="6 4" opacity="0.7" />
                </svg>
                <div>
                  <div className="font-medium">Shared branch edge</div>
                  <div className="text-xs text-muted-foreground">Same per-branch color but tighter dashes and faded &mdash; this upstream edge feeds more than one parallel experiment.</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <svg width="44" height="12" className="shrink-0 mt-1">
                  <line x1="0" y1="6" x2="44" y2="6" stroke="#f59e0b" strokeWidth="4" strokeDasharray="8 6" />
                </svg>
                <div>
                  <div className="font-medium">Winning merge edge</div>
                  <div className="text-xs text-muted-foreground">After a preview run, the branch whose values survived an overlapping-column merge is rendered thicker in amber with a &quot;WINS MERGE&quot; label.</div>
                </div>
              </li>
            </ul>
          </div>
        )}
      </div>

      {/* Right-side action cluster: history / load / save / tidy /
          export / run. Wraps onto a second row when the viewport is
          too narrow to fit everything on one line, and the secondary
          buttons collapse to icon-only at xl and below so the cluster
          stays compact even when the Properties panel is open. The
          primary Run / Run All buttons keep their labels because
          they're the user's most-used affordance. The max-width keeps
          the cluster from sliding under the left-side cluster. */}
      <div className="absolute top-4 right-4 z-10 flex flex-wrap justify-end gap-2 max-w-[calc(100%-13rem)]">
        <button
          onClick={() => toggleDrawer()}
          title="Job runs history"
          aria-label="Job runs history"
          data-testid="toolbar-jobs"
          className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors"
        >
          <History className="w-4 h-4" />
          <span className="text-sm font-medium hidden xl:inline">Jobs</span>
        </button>
        {!readOnly && (
          <div className="relative">
            <button
              onClick={openRecentMenu}
              title="Recently saved pipelines (last 5)"
              aria-label="Recently saved pipelines"
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
              <div
                role="menu"
                aria-label="Recent pipelines"
                className="absolute top-full right-0 mt-1 w-72 bg-background border rounded-md shadow-lg overflow-hidden z-20"
              >
                {recentPipelines.length === 0 ? (
                  <div className="px-3 py-3 text-sm text-muted-foreground">
                    No recent pipelines yet. Save one to populate this list.
                  </div>
                ) : (
                  <>
                    {recentPipelines.map((entry) => {
                      const isRenaming = renamingId === entry.id;
                      return (
                        <div
                          key={entry.id}
                          role="menuitem"
                          className="group w-full px-3 py-2 text-sm hover:bg-accent border-b last:border-b-0"
                        >
                          {isRenaming ? (
                            <div className="flex items-center gap-1.5">
                              <input
                                type="text"
                                // Focus is moved to the input as a direct response
                                // to the user clicking the pencil — the inline-rename
                                // affordance is unusable without it.
                                // eslint-disable-next-line jsx-a11y/no-autofocus
                                autoFocus
                                value={renameDraft}
                                onChange={(e) => setRenameDraft(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') commitRename();
                                  else if (e.key === 'Escape') cancelRename();
                                }}
                                onBlur={commitRename}
                                aria-label="Rename pipeline"
                                className="flex-1 px-2 py-1 text-sm bg-background border rounded outline-none focus:ring-1 focus:ring-primary"
                              />
                            </div>
                          ) : (
                            <div className="flex items-start gap-1.5">
                              <button
                                onClick={() => { void handleRestoreRecent(entry); }}
                                className="flex-1 text-left min-w-0"
                                title="Restore this snapshot"
                              >
                                <div className="font-medium truncate flex items-center gap-1.5">
                                  {entry.pinned && <Pin className="w-3 h-3 text-amber-500 flex-shrink-0" aria-label="Pinned" />}
                                  <span className="truncate">{entry.name}</span>
                                </div>
                                {entry.datasetName && (
                                  <div className="text-xs text-muted-foreground truncate mt-0.5" title={entry.datasetName}>
                                    on <span className="font-medium text-foreground/80">{entry.datasetName}</span>
                                  </div>
                                )}
                                <div className="text-xs text-muted-foreground flex items-center justify-between mt-0.5">
                                  <span>{formatRelativeTime(entry.savedAt)}</span>
                                  <span className="tabular-nums">
                                    {entry.nodes.length} nodes · {entry.edges.length} edges
                                  </span>
                                </div>
                              </button>
                              {/* Per-row actions — visible on hover/focus to keep
                                  the resting row uncluttered. */}
                              <div className="flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                                <button
                                  onClick={() => handleTogglePin(entry)}
                                  title={entry.pinned ? 'Unpin' : 'Pin (exempt from FIFO)'}
                                  aria-label={entry.pinned ? 'Unpin pipeline' : 'Pin pipeline'}
                                  className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                                >
                                  {entry.pinned ? <PinOff className="w-3 h-3" /> : <Pin className="w-3 h-3" />}
                                </button>
                                <button
                                  onClick={() => startRename(entry)}
                                  title="Rename"
                                  aria-label="Rename pipeline"
                                  className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                                >
                                  <Pencil className="w-3 h-3" />
                                </button>
                                <button
                                  onClick={() => { void handleDeleteRecent(entry); }}
                                  title="Delete"
                                  aria-label="Delete pipeline"
                                  className="p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/20 text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                                >
                                  <Trash2 className="w-3 h-3" />
                                </button>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                    <button
                      role="menuitem"
                      onClick={() => { void handleClearRecent(); }}
                      className="w-full text-left px-3 py-2 text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-1.5"
                    >
                      <Trash2 className="w-3 h-3" />
                      Clear history
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        )}
        {!readOnly && (
          <button
            onClick={() => { void handleLoad(); }}
            disabled={isLoading || isRunning}
            title="Load pipeline"
            aria-label="Load pipeline"
            data-testid="toolbar-load"
            className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FolderOpen className="w-4 h-4" />}
            <span className="text-sm font-medium hidden xl:inline">{isLoading ? 'Loading...' : 'Load'}</span>
          </button>
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
            {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
            <span className="text-sm font-medium hidden xl:inline">{isSaving ? 'Saving...' : 'Save'}</span>
          </button>
        )}
        {!readOnly && (
          <button
            onClick={() => {
              // Tidy up multi-branch canvases via dagre topological layout.
              const { nodes: laidOut, edges: keptEdges } = autoLayoutGraph(nodes, edges);
              setGraph(laidOut, keptEdges);
            }}
            disabled={isRunning || nodes.length === 0}
            title="Auto-arrange nodes left-to-right by data flow"
            aria-label="Tidy: auto-arrange nodes"
            className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            <Wand2 className="w-4 h-4" />
            <span className="text-sm font-medium hidden xl:inline">Tidy</span>
          </button>
        )}
        <div className="relative">
          <button
            onClick={() => setShowExportMenu(v => !v)}
            disabled={isExporting || nodes.length === 0}
            title="Export canvas as image"
            aria-label="Export canvas as image"
            aria-haspopup="menu"
            aria-expanded={showExportMenu}
            className="flex items-center gap-2 px-3 py-2 bg-background border rounded-md shadow-sm hover:bg-accent transition-colors disabled:opacity-50"
          >
            {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
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
            </div>
          )}
        </div>
        {!readOnly && hasMultipleBranches && (
          <button
            onClick={() => { void handleRunAll(); }}
            disabled={isRunningAll || isRunning}
            title="Run all parallel branches as separate experiments"
            aria-label="Run all parallel branches as separate experiments"
            data-testid="toolbar-run-all"
            className="flex items-center gap-2 px-3 py-2 text-white bg-amber-600 rounded-md shadow-sm hover:bg-amber-700 transition-colors disabled:opacity-50"
          >
            {isRunningAll ? <Loader2 className="w-4 h-4 animate-spin" /> : <Rocket className="w-4 h-4" />}
            <span className="text-sm font-medium hidden md:inline">{isRunningAll ? 'Queuing...' : 'Run All Experiments'}</span>
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
            {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            <span className="text-sm font-medium hidden md:inline">{isRunning ? 'Running...' : 'Run Preview'}</span>
          </button>
        )}
      </div>
    </>
  );
};
