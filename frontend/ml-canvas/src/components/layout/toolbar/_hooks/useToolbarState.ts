import { useState, useEffect, useRef } from 'react';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useViewStore } from '../../../../core/store/useViewStore';
import { useReadOnlyMode } from '../../../../core/hooks/useReadOnlyMode';
import { SHOW_TEMPLATES_EVENT } from '../../../../core/hooks/useKeyboardShortcuts';
import { exportCanvasToPng, exportCanvasToSvg } from '../../../../core/utils/canvasExport';
import { toast } from '../../../../core/toast';
import { useDismissable } from './useDismissable';
import { useRunControls } from './useRunControls';
import { usePipelineActions } from './usePipelineActions';
import { useToolbarEditing } from './useToolbarEditing';
import { useToolbarLayout } from './useToolbarLayout';

/** Own toolbar interaction state and preserve refs across responsive control changes. */
export function useToolbarState() {
  const { toggleDrawer } = useJobStore();
  const perfOverlayEnabled = useViewStore((s) => s.perfOverlayEnabled);
  const setPerfOverlayEnabled = useViewStore((s) => s.setPerfOverlayEnabled);
  // Hide editor-only buttons on tablet or when read-only is toggled on.
  const readOnly = useReadOnlyMode();

  const editing = useToolbarEditing(readOnly);

  const run = useRunControls();
  const [reviewingExperiments, setReviewingExperiments] = useState(false);
  useEffect(() => { if (readOnly) setReviewingExperiments(false); }, [readOnly]);
  const reviewPreviewResults = () => useViewStore.getState().setResultsPanelExpanded(true);

  const pipeline = usePipelineActions();
  const {
    showLoadMenu,
    setShowLoadMenu,
    showRecentMenu,
    setShowRecentMenu,
    hasServerVersions,
    recentPipelines,
  } = pipeline;

  const [showLegend, setShowLegend] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  // Compact-toolbar overflow menu (below xl).
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  // Templates gallery modal — controlled here so both the Toolbar button
  // and the canvas empty-state CTA (via event) can open it.
  const [showTemplates, setShowTemplates] = useState(false);

  const layout = useToolbarLayout();
  const { isNarrow } = layout;

  const moreMenuRef = useRef<HTMLDivElement | null>(null);
  const legendRef = useRef<HTMLDivElement | null>(null);
  const legendPopoverRef = useRef<HTMLDivElement | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const recentMenuRef = useRef<HTMLDivElement | null>(null);
  const loadMenuRef = useRef<HTMLDivElement | null>(null);
  useDismissable(showMoreMenu, () => setShowMoreMenu(false), moreMenuRef);
  useDismissable(showLegend, () => setShowLegend(false), [legendRef, legendPopoverRef]);
  useDismissable(showExportMenu, () => setShowExportMenu(false), exportMenuRef);
  useDismissable(showRecentMenu, () => setShowRecentMenu(false), recentMenuRef);
  useDismissable(showLoadMenu, () => setShowLoadMenu(false), loadMenuRef);

  // Overflow items disappear when opening another menu. Keep keyboard focus
  // in the replacement content and return it to More when that content closes.
  useEffect(() => {
    if (!isNarrow) return;
    const popover = showLegend ? legendPopoverRef.current
      : showLoadMenu ? loadMenuRef.current
        : showRecentMenu ? recentMenuRef.current : null;
    if (!popover) return;
    const trigger = moreMenuRef.current?.querySelector('button');
    const target = popover.querySelector<HTMLElement>('[role="menu"], button:not(:disabled)');
    target?.focus({ preventScroll: true });
    return () => {
      if (document.activeElement === document.body || popover.contains(document.activeElement)) {
        trigger?.focus({ preventScroll: true });
      }
    };
  }, [isNarrow, showLegend, showLoadMenu, showRecentMenu]);

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

  // Bridge: canvas empty-state CTA dispatches this event to open the
  // gallery without owning its own modal state.
  useEffect(() => {
    const open = (): void => setShowTemplates(true);
    window.addEventListener(SHOW_TEMPLATES_EVENT, open);
    return () => window.removeEventListener(SHOW_TEMPLATES_EVENT, open);
  }, []);
  const hasRecentPipelines = !readOnly && !hasServerVersions && recentPipelines.length > 0;
  return {
    editing,
    view: { toggleDrawer, perfOverlayEnabled, setPerfOverlayEnabled },
    layout,
    run,
    pipeline: { ...pipeline, hasRecentPipelines },
    menus: { reviewingExperiments, setReviewingExperiments, reviewPreviewResults, showLegend, setShowLegend, showExportMenu, setShowExportMenu, isExporting, showMoreMenu, setShowMoreMenu, showTemplates, setShowTemplates, moreMenuRef, legendRef, legendPopoverRef, exportMenuRef, recentMenuRef, loadMenuRef, handleExport },
    readOnly,
  };
}

export type ToolbarState = ReturnType<typeof useToolbarState>;
