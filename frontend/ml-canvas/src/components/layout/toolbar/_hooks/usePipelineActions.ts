import { useEffect, useMemo, useState } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { savePipeline } from '../../../../core/api/client';
import { pipelineVersionsApi, type PipelineVersionEntry } from '../../../../core/api/pipelineVersions';
import {
  getRecentPipelines,
  pushRecentPipeline,
  clearRecentPipelines,
  togglePinRecentPipeline,
  renameRecentPipeline,
  deleteRecentPipeline,
  type RecentPipelineEntry,
} from '../../../../core/utils/recentPipelines';
import { toast } from '../../../../core/toast';
import { useConfirm } from '../../../shared';

export interface PipelineActions {
  isSaving: boolean;
  currentDatasetId: string | undefined;
  hasServerVersions: boolean;
  showLoadMenu: boolean;
  setShowLoadMenu: (v: boolean) => void;
  loadVersions: PipelineVersionEntry[];
  loadVersionsLoading: boolean;
  showAllVersions: boolean;
  setShowAllVersions: (v: boolean) => void;
  showRecentMenu: boolean;
  setShowRecentMenu: (v: boolean) => void;
  recentPipelines: RecentPipelineEntry[];
  renamingId: string | null;
  renameDraft: string;
  setRenameDraft: (v: string) => void;
  handleSave: () => Promise<void>;
  openLoadMenu: () => Promise<void>;
  handleLoadVersion: (entry: PipelineVersionEntry) => Promise<void>;
  openRecentMenu: () => void;
  handleRestoreRecent: (entry: RecentPipelineEntry) => Promise<void>;
  handleClearRecent: () => Promise<void>;
  handleTogglePin: (entry: RecentPipelineEntry) => void;
  startRename: (entry: RecentPipelineEntry) => void;
  commitRename: () => void;
  cancelRename: () => void;
  handleDeleteRecent: (entry: RecentPipelineEntry) => Promise<void>;
  formatRelativeTime: (iso: string) => string;
}

export function usePipelineActions(): PipelineActions {
  const nodes = useGraphStore((s) => s.nodes);
  const edges = useGraphStore((s) => s.edges);
  const setGraph = useGraphStore((s) => s.setGraph);
  const confirm = useConfirm();

  const [isSaving, setIsSaving] = useState(false);
  const [showLoadMenu, setShowLoadMenu] = useState(false);
  const [loadVersions, setLoadVersions] = useState<PipelineVersionEntry[]>([]);
  const [loadVersionsLoading, setLoadVersionsLoading] = useState(false);
  const [hasServerVersions, setHasServerVersions] = useState(false);
  const [showAllVersions, setShowAllVersions] = useState(false);
  const [showRecentMenu, setShowRecentMenu] = useState(false);
  const [recentPipelines, setRecentPipelines] = useState<RecentPipelineEntry[]>([]);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');

  const currentDatasetId = useMemo<string | undefined>(() => {
    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    return datasetNode?.data.datasetId as string | undefined;
  }, [nodes]);

  // Hydrate recent pipelines on mount so the button count is correct.
  useEffect(() => {
    setRecentPipelines(getRecentPipelines());
  }, []);

  // Probe server versions whenever the active dataset changes — used to
  // gate the Recent (localStorage) fallback button.
  useEffect(() => {
    if (!currentDatasetId) {
      setHasServerVersions(false);
      return;
    }
    let cancelled = false;
    void pipelineVersionsApi
      .list(currentDatasetId)
      .then((list) => {
        if (!cancelled) setHasServerVersions(list.length > 0);
      })
      .catch(() => {
        if (!cancelled) setHasServerVersions(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentDatasetId]);

  const getPipelinePayload = () => ({
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: { ...n.data, catalogType: n.data.catalogType || n.data.definitionType },
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle,
      targetHandle: e.targetHandle,
      type: e.type,
      data: e.data,
    })),
  });

  const handleSave = async (): Promise<void> => {
    const datasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    const datasetId = datasetNode?.data.datasetId as string;
    const datasetName = datasetNode?.data.datasetName as string | undefined;
    if (!datasetId) {
      toast.error('No dataset node found', 'Cannot save pipeline without a dataset context.');
      return;
    }
    setIsSaving(true);
    try {
      await savePipeline(datasetId, {
        name: 'My Pipeline',
        description: 'Saved from Canvas',
        graph: getPipelinePayload(),
      });
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

  const openLoadMenu = async (): Promise<void> => {
    if (showLoadMenu) {
      setShowLoadMenu(false);
      return;
    }
    setShowLoadMenu(true);
    setShowAllVersions(false);
    if (!currentDatasetId) {
      setLoadVersions([]);
      return;
    }
    setLoadVersionsLoading(true);
    try {
      const list = await pipelineVersionsApi.list(currentDatasetId);
      setLoadVersions(list);
      setHasServerVersions(list.length > 0);
    } catch {
      setLoadVersions([]);
    } finally {
      setLoadVersionsLoading(false);
    }
  };

  const handleLoadVersion = async (entry: PipelineVersionEntry): Promise<void> => {
    setShowLoadMenu(false);
    const ok = await confirm({
      title: `Load v${entry.versionInt} "${entry.name}"?`,
      message: 'Loading this version will overwrite your current canvas. Continue?',
      confirmLabel: 'Load',
      variant: 'danger',
    });
    if (!ok) return;
    const graph = entry.graph as { nodes?: Node[]; edges?: Edge[] } | undefined;
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
      toast.error('Load failed', 'Snapshot graph is in an unrecognised shape.');
      return;
    }
    setGraph(graph.nodes, graph.edges);
    toast.success(`Loaded "${entry.name}"`, `Version v${entry.versionInt}`);
  };

  const openRecentMenu = (): void => {
    setRecentPipelines(getRecentPipelines());
    setShowRecentMenu((v) => !v);
  };

  const handleRestoreRecent = async (entry: RecentPipelineEntry): Promise<void> => {
    setShowRecentMenu(false);
    const currentDatasetNode = nodes.find((n) => n.data.definitionType === 'dataset_node');
    const curDatasetId = currentDatasetNode?.data.datasetId as string | undefined;
    // Warn when restoring across different datasets — column-bound nodes may break.
    const mismatch =
      entry.datasetId !== undefined &&
      curDatasetId !== undefined &&
      entry.datasetId !== curDatasetId;
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
      message:
        'This removes all 5 recent snapshots from this browser. The server-side saved pipeline is unaffected.',
      confirmLabel: 'Clear',
      variant: 'danger',
    });
    if (!ok) return;
    clearRecentPipelines();
    setRecentPipelines([]);
    setShowRecentMenu(false);
    toast.success('Pipeline history cleared');
  };

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
    // renameRecentPipeline silently rejects empty / clashing names.
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
      message:
        'This snapshot will be removed from your local history. The server-side saved pipeline is unaffected.',
      confirmLabel: 'Delete',
      variant: 'danger',
    });
    if (!ok) return;
    const updated = deleteRecentPipeline(entry.id);
    setRecentPipelines(updated);
    toast.success('Snapshot deleted');
  };

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

  return {
    isSaving,
    currentDatasetId,
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
  };
}
