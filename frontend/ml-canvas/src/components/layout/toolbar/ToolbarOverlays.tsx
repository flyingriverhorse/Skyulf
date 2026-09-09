import type { ToolbarState } from './_hooks/useToolbarState';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { TemplatesGalleryModal } from '../../canvas/TemplatesGalleryModal';
import { CanvasLegend } from './CanvasLegend';
import { ExperimentRunDialog } from './ExperimentRunDialog';

export function ToolbarOverlays(
  { layout, run, menus, readOnly }: Pick<ToolbarState, 'layout' | 'run' | 'menus' | 'readOnly'>,
) {
  const { isSidebarOpen, toolbarRef } = layout;
  const { isRunning, handleRunAll, experimentModels, experimentBlockReason } = run;
  const {
    reviewingExperiments,
    setReviewingExperiments,
    reviewPreviewResults,
    showLegend,
    setShowLegend,
    showTemplates,
    setShowTemplates,
    legendPopoverRef,
  } = menus;
  return (<>
    <ExperimentRunDialog isOpen={reviewingExperiments && !readOnly} models={experimentModels}
      blockReason={experimentBlockReason || (isRunning ? 'Wait for the data preview to finish.' : '')}
      onClose={() => {
        setReviewingExperiments(false);
        requestAnimationFrame(() => {
          const opener = toolbarRef.current?.querySelector<HTMLButtonElement>('[data-testid="toolbar-run-all"]')
            ?? toolbarRef.current?.querySelector<HTMLButtonElement>('[data-testid="toolbar-more"]');
          opener?.focus();
        });
      }}
      onSubmit={() => { setReviewingExperiments(false); void handleRunAll(); }}
      onReviewIssues={() => { setReviewingExperiments(false); useGraphStore.getState().validateGraph(); reviewPreviewResults(); }} />
    {showLegend && (
      <div ref={legendPopoverRef} className={`absolute top-4 right-4 z-40 ${isSidebarOpen || readOnly ? 'left-4' : 'left-16'}`}>
        <CanvasLegend onClose={() => setShowLegend(false)} />
      </div>
    )}
    <TemplatesGalleryModal
      isOpen={showTemplates}
      onClose={() => setShowTemplates(false)}
    />
  </>);
}
