import React from 'react';
import { Pin, PinOff, Pencil, Trash2 } from 'lucide-react';
import type { RecentPipelineEntry } from '../../../core/utils/recentPipelines';

interface RecentPipelinesMenuProps {
  recentPipelines: RecentPipelineEntry[];
  renamingId: string | null;
  renameDraft: string;
  onRenameDraftChange: (v: string) => void;
  onRestoreRecent: (entry: RecentPipelineEntry) => void;
  onTogglePin: (entry: RecentPipelineEntry) => void;
  onStartRename: (entry: RecentPipelineEntry) => void;
  onCommitRename: () => void;
  onCancelRename: () => void;
  onDeleteRecent: (entry: RecentPipelineEntry) => void;
  onClearRecent: () => void;
  formatRelativeTime: (iso: string) => string;
}

export const RecentPipelinesMenu: React.FC<RecentPipelinesMenuProps> = ({
  recentPipelines,
  renamingId,
  renameDraft,
  onRenameDraftChange,
  onRestoreRecent,
  onTogglePin,
  onStartRename,
  onCommitRename,
  onCancelRename,
  onDeleteRecent,
  onClearRecent,
  formatRelativeTime,
}) => (
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
                    // Focus is moved to the input as a direct response to the user clicking the
                    // pencil — the inline-rename affordance is unusable without it.
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                    value={renameDraft}
                    onChange={(e) => onRenameDraftChange(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') onCommitRename();
                      else if (e.key === 'Escape') onCancelRename();
                    }}
                    onBlur={onCommitRename}
                    aria-label="Rename pipeline"
                    className="flex-1 px-2 py-1 text-sm bg-background border rounded outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
              ) : (
                <div className="flex items-start gap-1.5">
                  <button
                    onClick={() => onRestoreRecent(entry)}
                    className="flex-1 text-left min-w-0"
                    title="Restore this snapshot"
                  >
                    <div className="font-medium truncate flex items-center gap-1.5">
                      {entry.pinned && (
                        <Pin className="w-3 h-3 text-amber-500 flex-shrink-0" aria-label="Pinned" />
                      )}
                      <span className="truncate">{entry.name}</span>
                    </div>
                    {entry.datasetName && (
                      <div
                        className="text-xs text-muted-foreground truncate mt-0.5"
                        title={entry.datasetName}
                      >
                        on{' '}
                        <span className="font-medium text-foreground/80">{entry.datasetName}</span>
                      </div>
                    )}
                    <div className="text-xs text-muted-foreground flex items-center justify-between mt-0.5">
                      <span>{formatRelativeTime(entry.savedAt)}</span>
                      <span className="tabular-nums">
                        {entry.nodes.length} nodes · {entry.edges.length} edges
                      </span>
                    </div>
                  </button>
                  {/* Per-row actions — visible on hover/focus to keep the resting row uncluttered. */}
                  <div className="flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                    <button
                      onClick={() => onTogglePin(entry)}
                      title={entry.pinned ? 'Unpin' : 'Pin (exempt from FIFO)'}
                      aria-label={entry.pinned ? 'Unpin pipeline' : 'Pin pipeline'}
                      className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                    >
                      {entry.pinned ? (
                        <PinOff className="w-3 h-3" />
                      ) : (
                        <Pin className="w-3 h-3" />
                      )}
                    </button>
                    <button
                      onClick={() => onStartRename(entry)}
                      title="Rename"
                      aria-label="Rename pipeline"
                      className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => onDeleteRecent(entry)}
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
          onClick={onClearRecent}
          className="w-full text-left px-3 py-2 text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-1.5"
        >
          <Trash2 className="w-3 h-3" />
          Clear history
        </button>
      </>
    )}
  </div>
);
