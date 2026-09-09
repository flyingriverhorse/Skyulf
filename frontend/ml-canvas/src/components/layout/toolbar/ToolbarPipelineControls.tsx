import type { ToolbarState } from './_hooks/useToolbarState';
import { Save, Loader2, FolderOpen, ChevronDown, Clock } from 'lucide-react';
import { RecentPipelinesMenu } from './RecentPipelinesMenu';
import { VersionLoadMenu } from './VersionLoadMenu';

export function ToolbarRecentControl(
  { layout, pipeline, menus }: Pick<ToolbarState, 'layout' | 'pipeline' | 'menus'>,
) {
  const { isCompact, isNarrow } = layout;
  const {
    showRecentMenu,
    recentPipelines,
    renamingId,
    renameDraft,
    setRenameDraft,
    openRecentMenu,
    handleRestoreRecent,
    handleClearRecent,
    handleTogglePin,
    startRename,
    commitRename,
    cancelRename,
    handleDeleteRecent,
    formatRelativeTime,
    hasRecentPipelines,
  } = pipeline;
  const { recentMenuRef } = menus;
  return (<>
    {hasRecentPipelines && (!isNarrow || showRecentMenu) && (
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
  </>);
}

export function ToolbarLoadControl(
  { layout, run, pipeline, menus, readOnly }: Pick<ToolbarState, 'layout' | 'run' | 'pipeline' | 'menus' | 'readOnly'>,
) {
  const { isCompact, isNarrow } = layout;
  const { isRunning } = run;
  const {
    showLoadMenu,
    setShowLoadMenu,
    loadVersions,
    loadVersionsLoading,
    showAllVersions,
    setShowAllVersions,
    openLoadMenu,
    handleLoadVersion,
  } = pipeline;
  const { loadMenuRef } = menus;
  return (<>
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
  </>);
}

export function ToolbarSaveControl(
  { layout, run, pipeline, readOnly }: Pick<ToolbarState, 'layout' | 'run' | 'pipeline' | 'readOnly'>,
) {
  const { isCompact, isNarrow } = layout;
  const { isRunning } = run;
  const { isSaving, handleSave } = pipeline;
  return (<>
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
  </>);
}
