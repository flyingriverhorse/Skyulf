/**
 * Pipeline Versions Modal (L7).
 *
 * Surfaces the server-side `pipeline_versions` history for a single
 * dataset. This is the durable, cross-device counterpart to the
 * Toolbar Recent dropdown (which stays per-browser for in-flight
 * shortcut history).
 *
 * Actions: pin/unpin, rename, edit note, delete, restore (loads the
 * snapshot into the canvas via /canvas?source_id=&version=).
 */
import React, { useCallback, useEffect, useState } from 'react';
import { Pin, PinOff, Pencil, Trash2, Loader2, Play, FileClock, MessageSquare, Save, X, Info, ChevronUp } from 'lucide-react';

import { ModalShell, useConfirm } from '../shared';
import { toast } from '../../core/toast';
import {
  pipelineVersionsApi,
  type PipelineVersionEntry,
} from '../../core/api/pipelineVersions';

interface PipelineVersionsModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasetId: string | null;
  datasetName?: string;
  /** Restore handler — caller decides whether to navigate or apply
   *  in-place. When omitted, the modal just closes after a no-op. */
  onRestore?: (entry: PipelineVersionEntry) => void;
}

function formatRelativeTime(iso: string): string {
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
}

/** Build a "node type → count" breakdown from a stored graph. Tolerates
 *  both React Flow snapshot shape ({nodes:[{type,data:{definitionType}}]})
 *  and engine config shape (list of {step_type}). Returns up to 8 entries
 *  sorted by count desc. */
function summariseGraph(graph: unknown): Array<[string, number]> {
  const counts = new Map<string, number>();
  const bump = (label: string): void => {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  };
  if (graph && typeof graph === 'object' && 'nodes' in (graph as Record<string, unknown>)) {
    const nodes = (graph as { nodes?: unknown }).nodes;
    if (Array.isArray(nodes)) {
      for (const n of nodes) {
        if (!n || typeof n !== 'object') continue;
        const data = (n as { data?: unknown }).data;
        const def =
          data && typeof data === 'object'
            ? (data as { definitionType?: unknown; catalogType?: unknown }).definitionType
              ?? (data as { catalogType?: unknown }).catalogType
            : undefined;
        const type = (n as { type?: unknown }).type;
        const label = String(def ?? type ?? 'unknown');
        bump(label);
      }
    }
  } else if (Array.isArray(graph)) {
    for (const n of graph) {
      if (n && typeof n === 'object') {
        const step = (n as { step_type?: unknown }).step_type;
        bump(String(step ?? 'unknown'));
      }
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
}

export const PipelineVersionsModal: React.FC<PipelineVersionsModalProps> = ({
  isOpen,
  onClose,
  datasetId,
  datasetName,
  onRestore,
}) => {
  const confirm = useConfirm();
  const [versions, setVersions] = useState<PipelineVersionEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Inline edit state — only one row can be in edit mode at a time.
  const [editingId, setEditingId] = useState<number | null>(null);
  const [nameDraft, setNameDraft] = useState('');
  const [noteDraft, setNoteDraft] = useState('');
  // Details expansion — separate from edit so user can inspect without
  // touching anything. Only one row expanded at a time to keep the
  // modal scroll height manageable.
  const [detailsId, setDetailsId] = useState<number | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    try {
      const list = await pipelineVersionsApi.list(datasetId);
      setVersions(list);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => {
    if (isOpen && datasetId) void refresh();
    if (!isOpen) {
      setEditingId(null);
      setNameDraft('');
      setNoteDraft('');
      setDetailsId(null);
    }
  }, [isOpen, datasetId, refresh]);

  const startEdit = (entry: PipelineVersionEntry): void => {
    setEditingId(entry.id);
    setNameDraft(entry.name);
    setNoteDraft(entry.note ?? '');
  };

  const cancelEdit = (): void => {
    setEditingId(null);
    setNameDraft('');
    setNoteDraft('');
  };

  const commitEdit = async (entry: PipelineVersionEntry): Promise<void> => {
    if (!datasetId) return;
    const trimmedName = nameDraft.trim();
    if (!trimmedName) {
      toast.error('Name cannot be empty');
      return;
    }
    try {
      const patch: { name?: string; note?: string | null } = {};
      if (trimmedName !== entry.name) patch.name = trimmedName;
      // Always send note (server treats empty string as cleared).
      patch.note = noteDraft.trim() || null;
      await pipelineVersionsApi.update(datasetId, entry.id, patch);
      cancelEdit();
      void refresh();
      toast.success('Version updated');
    } catch (err) {
      toast.error('Failed to update version', String(err));
    }
  };

  const handleTogglePin = async (entry: PipelineVersionEntry): Promise<void> => {
    if (!datasetId) return;
    try {
      await pipelineVersionsApi.togglePin(datasetId, entry.id, !entry.pinned);
      void refresh();
    } catch (err) {
      toast.error('Failed to toggle pin', String(err));
    }
  };

  const handleDelete = async (entry: PipelineVersionEntry): Promise<void> => {
    if (!datasetId) return;
    const ok = await confirm({
      title: `Delete v${entry.versionInt}?`,
      message: `"${entry.name}" will be permanently removed from the version history. This cannot be undone.`,
      confirmLabel: 'Delete',
      variant: 'danger',
    });
    if (!ok) return;
    try {
      await pipelineVersionsApi.remove(datasetId, entry.id);
      void refresh();
      toast.success('Version deleted');
    } catch (err) {
      toast.error('Failed to delete version', String(err));
    }
  };

  const handleRestore = async (entry: PipelineVersionEntry): Promise<void> => {
    const ok = await confirm({
      title: `Restore v${entry.versionInt} "${entry.name}"?`,
      message:
        'This will load the snapshot into the canvas and overwrite any unsaved work there. Continue?',
      confirmLabel: 'Restore',
      variant: 'danger',
    });
    if (!ok) return;
    onRestore?.(entry);
    onClose();
  };

  return (
    <ModalShell
      isOpen={isOpen}
      onClose={onClose}
      size="3xl"
      title={
        <div className="flex items-center gap-2">
          <FileClock className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <span>Pipeline Versions</span>
          {datasetName && (
            <span className="text-sm font-normal text-muted-foreground">
              · {datasetName}
            </span>
          )}
        </div>
      }
    >
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </div>
      )}
      {!loading && error && (
        <div className="p-4 text-sm text-red-600 dark:text-red-400">
          Failed to load versions: {error}
        </div>
      )}
      {!loading && !error && versions.length === 0 && (
        <div className="p-8 text-center text-sm text-muted-foreground">
          <FileClock className="w-10 h-10 mx-auto mb-2 opacity-40" />
          <p className="font-medium text-foreground/80 mb-1">No versions yet</p>
          <p>
            Saving a pipeline against this dataset stamps a version
            automatically. They show up here for cross-device restore.
          </p>
        </div>
      )}
      {!loading && !error && versions.length > 0 && (
        <div className="divide-y border rounded-md overflow-hidden">
          {versions.map((entry) => {
            const isEditing = editingId === entry.id;
            return (
              <div key={entry.id} className="p-3 hover:bg-accent/40 transition-colors">
                {isEditing ? (
                  <div className="space-y-2">
                    <input
                      type="text"
                      value={nameDraft}
                      onChange={(e) => setNameDraft(e.target.value)}
                      placeholder="Pipeline name"
                      aria-label="Pipeline name"
                      className="w-full px-2 py-1 text-sm bg-background border rounded outline-none focus:ring-1 focus:ring-primary"
                      // eslint-disable-next-line jsx-a11y/no-autofocus
                      autoFocus
                    />
                    <textarea
                      value={noteDraft}
                      onChange={(e) => setNoteDraft(e.target.value)}
                      placeholder="Note (optional) — describe what changed"
                      aria-label="Version note"
                      rows={2}
                      className="w-full px-2 py-1 text-sm bg-background border rounded outline-none focus:ring-1 focus:ring-primary resize-none"
                    />
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => { void commitEdit(entry); }}
                        className="flex items-center gap-1 px-2 py-1 text-xs bg-primary text-primary-foreground rounded hover:opacity-90"
                      >
                        <Save className="w-3 h-3" /> Save
                      </button>
                      <button
                        onClick={cancelEdit}
                        className="flex items-center gap-1 px-2 py-1 text-xs border rounded hover:bg-accent"
                      >
                        <X className="w-3 h-3" /> Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-start gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        {entry.pinned && (
                          <Pin
                            className="w-3 h-3 text-amber-500 flex-shrink-0"
                            aria-label="Pinned"
                          />
                        )}
                        <span className="font-medium truncate" title={entry.name}>
                          {entry.name}
                        </span>
                        <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-muted text-muted-foreground flex-shrink-0">
                          v{entry.versionInt}
                        </span>
                        {entry.kind === 'auto' && (
                          <span
                            className="text-xs px-1.5 py-0.5 rounded bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 flex-shrink-0"
                            title="Auto-snapshot from a successful save or run"
                          >
                            auto
                          </span>
                        )}
                      </div>
                      {entry.note && (
                        <div
                          className="text-xs text-foreground/80 mt-0.5 flex items-start gap-1"
                          title={entry.note}
                        >
                          <MessageSquare className="w-3 h-3 mt-0.5 flex-shrink-0 text-muted-foreground" />
                          <span className="break-words">{entry.note}</span>
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground flex items-center gap-3 mt-1 tabular-nums">
                        <span>{formatRelativeTime(entry.createdAt)}</span>
                        <span>
                          {entry.nodeCount} nodes · {entry.edgeCount} edges
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-0.5 flex-shrink-0">
                      <button
                        onClick={() => { void handleRestore(entry); }}
                        title="Restore this version"
                        aria-label="Restore version"
                        className="p-1.5 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 text-blue-600 dark:text-blue-400"
                      >
                        <Play className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() =>
                          setDetailsId((cur) => (cur === entry.id ? null : entry.id))
                        }
                        title={detailsId === entry.id ? 'Hide details' : 'Show details'}
                        aria-label="Toggle version details"
                        aria-expanded={detailsId === entry.id}
                        className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
                      >
                        {detailsId === entry.id ? (
                          <ChevronUp className="w-3.5 h-3.5" />
                        ) : (
                          <Info className="w-3.5 h-3.5" />
                        )}
                      </button>
                      <button
                        onClick={() => { void handleTogglePin(entry); }}
                        title={entry.pinned ? 'Unpin' : 'Pin (exempt from cleanup)'}
                        aria-label={entry.pinned ? 'Unpin version' : 'Pin version'}
                        className="p-1.5 rounded hover:bg-amber-50 dark:hover:bg-amber-900/20 text-muted-foreground hover:text-amber-600"
                      >
                        {entry.pinned ? (
                          <PinOff className="w-3.5 h-3.5" />
                        ) : (
                          <Pin className="w-3.5 h-3.5" />
                        )}
                      </button>
                      <button
                        onClick={() => startEdit(entry)}
                        title="Edit name & note"
                        aria-label="Edit version"
                        className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => { void handleDelete(entry); }}
                        title="Delete version"
                        aria-label="Delete version"
                        className="p-1.5 rounded hover:bg-red-50 dark:hover:bg-red-900/20 text-muted-foreground hover:text-red-600 dark:hover:text-red-400"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                )}
                {detailsId === entry.id && !isEditing && (
                  <div className="mt-2 pt-2 border-t text-xs space-y-1.5 text-muted-foreground">
                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                      <div>
                        <span className="text-foreground/70 font-medium">Created: </span>
                        {new Date(entry.createdAt).toLocaleString()}
                      </div>
                      <div>
                        <span className="text-foreground/70 font-medium">Kind: </span>
                        {entry.kind}
                      </div>
                      <div>
                        <span className="text-foreground/70 font-medium">Version: </span>
                        v{entry.versionInt} <span className="opacity-60">(#{entry.id})</span>
                      </div>
                      <div>
                        <span className="text-foreground/70 font-medium">Dataset: </span>
                        <span className="font-mono">{entry.datasetId}</span>
                      </div>
                    </div>
                    {(() => {
                      const breakdown = summariseGraph(entry.graph);
                      if (breakdown.length === 0) {
                        return (
                          <div className="italic opacity-70">
                            Snapshot graph is empty or in an unrecognised shape.
                          </div>
                        );
                      }
                      return (
                        <div>
                          <div className="text-foreground/70 font-medium mb-1">
                            Node breakdown
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {breakdown.map(([type, count]) => (
                              <span
                                key={type}
                                className="px-1.5 py-0.5 rounded bg-muted text-foreground/80 tabular-nums"
                                title={type}
                              >
                                {type}{' '}
                                <span className="opacity-60">×{count}</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </ModalShell>
  );
};
