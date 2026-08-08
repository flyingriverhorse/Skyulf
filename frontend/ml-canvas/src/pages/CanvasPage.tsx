import React, { useEffect, useRef, useState } from 'react';
import { useSearchParams, useLocation, useNavigate, Link } from 'react-router-dom';
import { X } from 'lucide-react';
import { MainLayout } from '../components/layout/MainLayout';
import { ErrorState } from '../components/shared';
import { useGraphStore } from '../core/store/useGraphStore';
import { useViewStore } from '../core/store/useViewStore';
import type { PipelineVersionEntry } from '../core/api/pipelineVersions';
import { toast } from '../core/toast';
import { parseOperationalContext } from '../core/utils/operationalContext';
import { FOCUS_NODE_EVENT } from '../core/hooks/useKeyboardShortcuts';
import type { Node, Edge } from '@xyflow/react';

const SHELL_VIEWS = ['canvas', 'experiments', 'inference'] as const;
type ShellView = (typeof SHELL_VIEWS)[number];
const isShellView = (value: string | null): value is ShellView =>
  value !== null && (SHELL_VIEWS as readonly string[]).includes(value);

/** Friendly names for the Operations routes a node RecordLink's `origin` can carry. */
const ORIGIN_LABELS: Record<string, string> = {
  '/errors': 'Error Log',
  '/slow-nodes': 'Slow Nodes',
  '/jobs': 'Jobs',
  '/drift': 'Drift',
  '/audit': 'Audit Log',
};

/**
 * A `node` RecordLink target (OPS-007) the canvas couldn't select outright:
 * either it belongs to a pipeline other than the one currently open, or it
 * simply isn't present in this graph at all.
 */
interface NodeDeepLinkNotice {
  kind: 'different-pipeline' | 'not-found';
  nodeId: string;
  pipelineId?: string;
  origin?: string;
}

export const CanvasPage: React.FC = () => {
  const addNode = useGraphStore((state) => state.addNode);
  const setGraph = useGraphStore((state) => state.setGraph);
  const selectNode = useGraphStore((state) => state.selectNode);
  const nodes = useGraphStore((state) => state.nodes);
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const processedRef = useRef(false);
  const restoreProcessedRef = useRef(false);
  // Tracks the last `oc.*` query string this effect has already resolved,
  // so following a second node RecordLink while already on /canvas (a
  // client-side navigation, not a remount) re-triggers selection instead
  // of being skipped as "already handled".
  const nodeContextKeyRef = useRef<string | null>(null);
  const [nodeNotice, setNodeNotice] = useState<NodeDeepLinkNotice | null>(null);
  const activeView = useViewStore((state) => state.activeView);
  const setView = useViewStore((state) => state.setView);
  // Tracks whether we've already written the initial `view` param once,
  // so the very first sync (populating a fresh /canvas URL with the
  // default view) replaces rather than pushing an extra history entry
  // the user never asked for.
  const hasSyncedUrlRef = useRef(false);
  // Set right before the URL->store effect applies a param, so the
  // store->URL effect below can tell "this activeView change came from
  // the URL itself" and skip re-writing it. Without this, the two
  // effects each read a stale copy of the other's state and ping-pong
  // the URL between values forever (they run in the same commit, before
  // either has seen the other's update).
  const appliedFromUrlRef = useRef(false);

  // FND-006: make the selected shell view (Canvas/Experiments/Inference)
  // restorable via Back/Forward instead of living only in memory. The
  // `view` query param is the source of truth for navigation history.
  // This effect only reacts to the URL changing (initial load, deep
  // link, Back/Forward) — it intentionally does not depend on
  // `activeView` so that store-originated changes (Navbar clicks) don't
  // re-trigger it.
  useEffect(() => {
    const paramView = searchParams.get('view');
    if (isShellView(paramView) && paramView !== useViewStore.getState().activeView) {
      appliedFromUrlRef.current = true;
      setView(paramView);
    }
  }, [searchParams, setView]);

  // Mirror image of the effect above: only reacts to the store's
  // `activeView` changing, so it doesn't re-fire from its own URL
  // writes. Skips the one change that the effect above just applied
  // from the URL, since that's already reflected there. Reads the
  // store directly via getState() rather than the closed-over
  // `activeView` — React 18 StrictMode replays this effect body a
  // second time against the *same* stale render closure, and using
  // that stale value here was pushing the pre-update view back into
  // the URL right after the URL->store effect had just changed it.
  useEffect(() => {
    if (appliedFromUrlRef.current) {
      appliedFromUrlRef.current = false;
      return;
    }
    const currentView = useViewStore.getState().activeView;
    if (searchParams.get('view') === currentView) {
      hasSyncedUrlRef.current = true;
      return;
    }
    const newParams = new URLSearchParams(searchParams);
    newParams.set('view', currentView);
    // A view switch is a new place in the shell's history, not a detail
    // of the current one — push so Back actually restores the prior
    // view, except for the very first sync which just fills in the
    // default and shouldn't itself become a Back target.
    setSearchParams(newParams, { replace: !hasSyncedUrlRef.current });
    hasSyncedUrlRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeView]);

  // L7: when DataSources -> Versions -> Restore navigates here with a
  // version payload in router state, apply it to the canvas. Runs
  // before the source_id branch so we don't add an extra dataset node
  // on top of the snapshot.
  useEffect(() => {
    const state = location.state as { restoreVersion?: PipelineVersionEntry } | null;
    const entry = state?.restoreVersion;
    if (!entry || restoreProcessedRef.current) return;
    restoreProcessedRef.current = true;

    const graph = entry.graph as { nodes?: Node[]; edges?: Edge[] } | undefined;
    if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
      setGraph(graph.nodes, graph.edges);
      // Mark source_id as processed so the existing dataset-node branch
      // below doesn't fire on top of the restored graph.
      processedRef.current = true;
      const newParams = new URLSearchParams(searchParams);
      newParams.delete('source_id');
      setSearchParams(newParams, { replace: true });
      // Clear router state so a refresh doesn't re-apply.
      navigate(location.pathname + location.search, { replace: true, state: null });
      toast.success(`Restored "${entry.name}"`, `Version v${entry.versionInt}`);
    } else {
      toast.error(
        'Restore failed',
        'Snapshot graph is not in a recognised shape.',
      );
    }
  }, [location, navigate, setGraph, searchParams, setSearchParams]);

  useEffect(() => {
    const sourceId = searchParams.get('source_id');

    if (sourceId) {
      // Prevent double-firing in StrictMode
      if (processedRef.current) return;

      // Check if we already have this dataset on the canvas
      const alreadyExists = nodes.some(n =>
        n.type === 'dataset_node' &&
        (n.data as { datasetId?: string })?.datasetId === sourceId
      );

      if (alreadyExists) {
        // If it exists, just clean the URL
        const newParams = new URLSearchParams(searchParams);
        newParams.delete('source_id');
        setSearchParams(newParams, { replace: true });
        return;
      }

      // Mark as processed to prevent duplicates
      processedRef.current = true;

      // Add the node
      addNode('dataset_node', { x: 100, y: 100 }, { datasetId: sourceId });

      // Clean up URL
      const newParams = new URLSearchParams(searchParams);
      newParams.delete('source_id');
      setSearchParams(newParams, { replace: true });
    }
  }, [searchParams, addNode, nodes, setSearchParams]);

  // OPS-007: a `node` RecordLink (Error Log, Slow Nodes) lands here
  // carrying the target node's id (and, when known, the pipeline it
  // belongs to). Select + focus it if it's in the currently loaded
  // graph; otherwise tell the user plainly instead of leaving the
  // click a no-op.
  useEffect(() => {
    const context = parseOperationalContext(searchParams);
    if (!context || context.ref.kind !== 'node') return;

    const key = searchParams.toString();
    if (nodeContextKeyRef.current === key) return;
    nodeContextKeyRef.current = key;

    const { nodeId, pipelineId } = context.ref;
    const found = selectNode(nodeId);

    if (found) {
      setNodeNotice(null);
      // Reuse the existing CAN-001/CAN-003 focus mechanism (FlowCanvas)
      // rather than duplicating fitView logic here; opt into moving DOM
      // focus since, unlike a palette click, the user has no other
      // element on this page they were just interacting with.
      window.dispatchEvent(
        new CustomEvent(FOCUS_NODE_EVENT, { detail: { id: nodeId, focusWrapper: true } }),
      );
      return;
    }

    setNodeNotice(
      pipelineId !== undefined
        ? { kind: 'different-pipeline', nodeId, pipelineId, ...(context.origin ? { origin: context.origin } : {}) }
        : { kind: 'not-found', nodeId, ...(context.origin ? { origin: context.origin } : {}) },
    );
  }, [searchParams, selectNode]);

  return (
    <div className="relative h-full w-full">
      {nodeNotice && (
        <div className="absolute top-4 left-1/2 z-50 w-full max-w-lg -translate-x-1/2 px-4">
          <div className="relative rounded-lg border border-amber-300 bg-white shadow-lg dark:border-amber-700 dark:bg-slate-800">
            <button
              type="button"
              onClick={() => setNodeNotice(null)}
              aria-label="Dismiss"
              className="absolute right-2 top-2 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-700"
            >
              <X className="h-4 w-4" aria-hidden="true" />
            </button>
            <ErrorState
              error={
                nodeNotice.kind === 'different-pipeline'
                  ? `Node ${nodeNotice.nodeId} belongs to pipeline ${nodeNotice.pipelineId}, which isn't the pipeline currently open on this canvas. Load that pipeline's version from Data Sources to inspect it, or return to where you came from to re-run the investigation.`
                  : `Node ${nodeNotice.nodeId} could not be found on this canvas. It may have been removed or renamed.`
              }
            />
            {nodeNotice.origin && (
              <div className="border-t border-slate-100 px-4 pb-4 pt-2 text-center dark:border-slate-700">
                <Link
                  to={nodeNotice.origin}
                  className="text-sm font-medium text-blue-600 hover:underline dark:text-blue-400"
                >
                  Back to {ORIGIN_LABELS[nodeNotice.origin] ?? nodeNotice.origin}
                </Link>
              </div>
            )}
          </div>
        </div>
      )}
      <MainLayout />
    </div>
  );
};
