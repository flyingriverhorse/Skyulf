import React from 'react';
import { useToolbarState } from './toolbar/_hooks/useToolbarState';
import { ToolbarEditingControls } from './toolbar/ToolbarEditingControls';
import { ToolbarOverflowMenu } from './toolbar/ToolbarOverflowMenu';
import { ToolbarViewControls, ToolbarTidyControl, ToolbarExportControl } from './toolbar/ToolbarViewControls';
import { ToolbarRecentControl, ToolbarLoadControl, ToolbarSaveControl } from './toolbar/ToolbarPipelineControls';
import { ToolbarExperimentControl, ToolbarPreviewControl } from './toolbar/ToolbarRunControls';
import { ToolbarOverlays } from './toolbar/ToolbarOverlays';

export const Toolbar: React.FC = () => {
  const { editing, view, layout, run, pipeline, menus, readOnly } = useToolbarState();
  const { toolbarRef, isSidebarOpen } = layout;
  const { showMoreMenu, showExportMenu } = menus;
  const { showLoadMenu, showRecentMenu } = pipeline;
  return (
    <>
      {/* Shared layout prevents the action groups from occupying the same space.
          Reserve the floating sidebar toggle's slot when the library is closed. */}
      <div
        ref={toolbarRef}
        data-canvas-toolbar
        className={`absolute top-4 right-4 flex flex-col gap-2 ${showMoreMenu || showLoadMenu || showRecentMenu || showExportMenu ? 'z-40' : 'z-10'} ${isSidebarOpen || readOnly ? 'left-4' : 'left-16'}`}
      >
        <div className="flex w-full items-start gap-2">
          <ToolbarEditingControls editing={editing} layout={layout} menus={menus} readOnly={readOnly} />
          <div className="flex flex-1 min-w-0 flex-wrap justify-end gap-2 [&>button]:shrink-0">
            {/* Compact overflow menu — collapses secondary actions so the
            cluster never overlaps the left cluster once the live Flow-pane
            width (not the viewport) drops below COMPACT_WIDTH. */}
            <ToolbarOverflowMenu editing={editing} view={view} layout={layout} run={run} pipeline={pipeline} menus={menus} readOnly={readOnly} />
            <ToolbarViewControls view={view} layout={layout} menus={menus} readOnly={readOnly} />
            {/* Recent pipelines (localStorage fallback) — only shown when no
            server-side versions exist for the current dataset. */}
            <ToolbarRecentControl layout={layout} pipeline={pipeline} menus={menus} />
            <ToolbarLoadControl layout={layout} run={run} pipeline={pipeline} menus={menus} readOnly={readOnly} />
            <ToolbarSaveControl layout={layout} run={run} pipeline={pipeline} readOnly={readOnly} />
            <ToolbarTidyControl editing={editing} layout={layout} run={run} readOnly={readOnly} />
            <ToolbarExportControl editing={editing} layout={layout} pipeline={pipeline} menus={menus} />
            <ToolbarExperimentControl layout={layout} run={run} menus={menus} readOnly={readOnly} />
            <ToolbarPreviewControl run={run} readOnly={readOnly} />
          </div>
        </div>
      </div>
      <ToolbarOverlays layout={layout} run={run} menus={menus} readOnly={readOnly} />
    </>
  );
};
