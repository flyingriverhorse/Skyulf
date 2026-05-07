import React from 'react';
import { Loader2, Pin, ChevronDown } from 'lucide-react';
import type { PipelineVersionEntry } from '../../../core/api/pipelineVersions';

interface VersionLoadMenuProps {
  onClose: () => void;
  loadVersions: PipelineVersionEntry[];
  loadVersionsLoading: boolean;
  showAllVersions: boolean;
  onSetShowAllVersions: (v: boolean) => void;
  onLoadVersion: (entry: PipelineVersionEntry) => void;
}

export const VersionLoadMenu: React.FC<VersionLoadMenuProps> = ({
  onClose,
  loadVersions,
  loadVersionsLoading,
  showAllVersions,
  onSetShowAllVersions,
  onLoadVersion,
}) => (
  <>
    {/*
      Bulletproof outside-click catcher. Sits behind the dropdown
      but in front of every other toolbar / canvas element, so any click
      outside the menu is captured here and dismisses cleanly. This avoids
      the fragile document-level mousedown listener in `useDismissable`
      misbehaving under React Flow's pointer handling.
    */}
    <div
      className="fixed inset-0 z-10"
      onClick={onClose}
      aria-hidden="true"
    />
    <div
      role="menu"
      aria-label="Load pipeline version"
      className="absolute top-full right-0 mt-1 w-72 bg-background border rounded-md shadow-lg overflow-hidden z-20"
    >
      {loadVersionsLoading ? (
        <div className="px-3 py-4 flex items-center justify-center text-muted-foreground">
          <Loader2 className="w-4 h-4 animate-spin" />
        </div>
      ) : loadVersions.length === 0 ? (
        <div className="px-3 py-3 text-sm text-muted-foreground">
          No versions yet for this dataset. Save your pipeline to stamp the first one.
        </div>
      ) : (
        <>
          <div className="px-3 py-1.5 text-xs uppercase tracking-wide text-muted-foreground border-b flex items-center justify-between">
            <span>
              {showAllVersions
                ? `All versions (${loadVersions.length})`
                : 'Latest versions'}
            </span>
          </div>
          <div className={showAllVersions ? 'max-h-80 overflow-y-auto' : ''}>
            {(showAllVersions ? loadVersions : loadVersions.slice(0, 5)).map((entry) => (
              <button
                key={entry.id}
                role="menuitem"
                onClick={() => onLoadVersion(entry)}
                className="w-full text-left px-3 py-2 text-sm hover:bg-accent border-b last:border-b-0"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  {entry.pinned && (
                    <Pin className="w-3 h-3 text-amber-500 flex-shrink-0" aria-label="Pinned" />
                  )}
                  <span className="font-medium truncate flex-1">{entry.name}</span>
                  <span className="text-xs font-mono text-muted-foreground flex-shrink-0">
                    v{entry.versionInt}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground tabular-nums mt-0.5">
                  {entry.nodeCount} nodes · {entry.edgeCount} edges
                </div>
              </button>
            ))}
          </div>
        </>
      )}
      {loadVersions.length > 5 && !showAllVersions && (
        <button
          role="menuitem"
          onClick={() => onSetShowAllVersions(true)}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 border-t"
        >
          <ChevronDown className="w-3 h-3" />
          Show more versions ({loadVersions.length - 5} more)
        </button>
      )}
    </div>
  </>
);
