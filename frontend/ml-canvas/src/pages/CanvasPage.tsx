import React, { useEffect, useRef } from 'react';
import { useSearchParams, useLocation, useNavigate } from 'react-router-dom';
import { MainLayout } from '../components/layout/MainLayout';
import { useGraphStore } from '../core/store/useGraphStore';
import { useViewStore } from '../core/store/useViewStore';
import type { PipelineVersionEntry } from '../core/api/pipelineVersions';
import { toast } from '../core/toast';
import type { Node, Edge } from '@xyflow/react';

const SHELL_VIEWS = ['canvas', 'experiments', 'inference'] as const;
type ShellView = (typeof SHELL_VIEWS)[number];
const isShellView = (value: string | null): value is ShellView =>
  value !== null && (SHELL_VIEWS as readonly string[]).includes(value);

export const CanvasPage: React.FC = () => {
  const addNode = useGraphStore((state) => state.addNode);
  const setGraph = useGraphStore((state) => state.setGraph);
  const nodes = useGraphStore((state) => state.nodes);
  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const processedRef = useRef(false);
  const restoreProcessedRef = useRef(false);
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

  return <MainLayout />;
};
