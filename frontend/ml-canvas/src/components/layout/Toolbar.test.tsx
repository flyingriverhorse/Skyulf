import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Toolbar } from './Toolbar';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { useRunControls, type RunControls } from './toolbar/_hooks/useRunControls';
import { usePipelineActions } from './toolbar/_hooks/usePipelineActions';
import { exportCanvasToPng, exportCanvasToSvg } from '../../core/utils/canvasExport';
import { toast } from '../../core/toast';

const { confirm } = vi.hoisted(() => ({ confirm: vi.fn() }));
vi.mock('../shared', () => ({ useConfirm: () => confirm }));
vi.mock('./toolbar/_hooks/useRunControls', () => ({ useRunControls: vi.fn() }));
vi.mock('./toolbar/_hooks/usePipelineActions', () => ({ usePipelineActions: vi.fn() }));
vi.mock('../../core/utils/canvasExport', () => ({ exportCanvasToPng: vi.fn(), exportCanvasToSvg: vi.fn() }));
vi.mock('../../core/toast', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock('../canvas/TemplatesGalleryModal', () => ({ TemplatesGalleryModal: () => null }));

let runControls: RunControls;
let containerWidth = 1400;

/** Render the real toolbar in a pane whose measured width is deterministic. */
function renderToolbar() {
  return render(<div><Toolbar /></div>);
}

/** Reach experiment review using the visible control at the current pane width. */
function openExperimentReview() {
  if (containerWidth < 720) {
    fireEvent.click(screen.getByRole('button', { name: 'More canvas tools' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Run all experiments' }));
  } else {
    fireEvent.click(screen.getByRole('button', { name: 'Run all parallel branches as separate experiments' }));
  }
}

describe('Toolbar experiment controls', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    containerWidth = 1400;
    vi.stubGlobal('ResizeObserver', class {
      constructor(private callback: ResizeObserverCallback) {}
      observe() {
        this.callback([{ contentRect: { width: containerWidth } } as ResizeObserverEntry], this as unknown as ResizeObserver);
      }
      disconnect() {}
      unobserve() {}
    });
    useViewStore.setState({ readOnlyOverride: 'off', isResultsPanelExpanded: false });
    useGraphStore.setState({
      nodes: [{ id: 'dataset', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'ds-1' } }],
      edges: [],
    });
    useGraphStore.temporal.getState().clear();
    runControls = {
      isRunning: false,
      isRunningAll: false,
      canRunPreview: false,
      hasMultipleBranches: true,
      experimentModels: [
        { id: 'first', name: 'First classifier', model: 'random forest classifier', action: 'Train' },
        { id: 'second', name: 'Second classifier', model: 'logistic regression', action: 'Tune' },
      ],
      experimentBlockReason: '',
      handleRun: vi.fn().mockResolvedValue(undefined),
      handleRunAll: vi.fn().mockResolvedValue(undefined),
    };
    vi.mocked(useRunControls).mockImplementation(() => runControls);
    vi.mocked(usePipelineActions).mockReturnValue({
      isSaving: false, hasServerVersions: false, showLoadMenu: false,
      setShowLoadMenu: vi.fn(), loadVersions: [], loadVersionsLoading: false,
      showAllVersions: false, setShowAllVersions: vi.fn(), showRecentMenu: false,
      setShowRecentMenu: vi.fn(), recentPipelines: [], renamingId: null,
      renameDraft: '', setRenameDraft: vi.fn(), handleSave: vi.fn(),
      openLoadMenu: vi.fn(), handleLoadVersion: vi.fn(), openRecentMenu: vi.fn(),
      handleRestoreRecent: vi.fn(), handleClearRecent: vi.fn(), handleTogglePin: vi.fn(),
      startRename: vi.fn(), commitRename: vi.fn(), cancelRename: vi.fn(),
      handleDeleteRecent: vi.fn(), formatRelativeTime: vi.fn(), currentDatasetId: 'ds-1',
      exportNotebook: vi.fn(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it('queues the reviewed models only after confirmation', () => {
    // Opening the review must not immediately submit background jobs.
    renderToolbar();
    openExperimentReview();
    expect(screen.getByRole('dialog', { name: 'Run all experiments?' })).toBeInTheDocument();
    expect(screen.getByText('First classifier')).toBeInTheDocument();
    expect(screen.getByText('Second classifier')).toBeInTheDocument();
    expect(runControls.handleRunAll).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Queue experiments' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(runControls.handleRunAll).toHaveBeenCalledOnce();
  });

  it.each([1400, 600, 350])('cancels review and returns focus at pane width %i', async width => {
    // Narrow layouts remove the menu item, so focus must return to the surviving More button.
    containerWidth = width;
    renderToolbar();
    openExperimentReview();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(runControls.handleRunAll).not.toHaveBeenCalled();
    const opener = screen.getByRole('button', {
      name: width < 720 ? 'More canvas tools' : 'Run all parallel branches as separate experiments',
    });
    await waitFor(() => expect(opener).toHaveFocus());
  });

  it('opens validation results and closes the blocked review', () => {
    // Reviewing errors must route users to actionable canvas feedback without submitting.
    runControls.experimentBlockReason = 'Fix the missing dataset connection first.';
    const validate = vi.spyOn(useGraphStore.getState(), 'validateGraph');
    renderToolbar();
    openExperimentReview();
    expect(screen.getByRole('button', { name: 'Queue experiments' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Review validation issues' }));
    expect(validate).toHaveBeenCalledOnce();
    expect(useViewStore.getState().isResultsPanelExpanded).toBe(true);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(runControls.handleRunAll).not.toHaveBeenCalled();
  });

  it('blocks an open review while preview is running and keeps validation errors first', () => {
    // A preview started after review opens must prevent an overlapping experiment submission.
    const { rerender } = renderToolbar();
    openExperimentReview();
    runControls.isRunning = true;
    rerender(<div><Toolbar /></div>);
    expect(screen.getByRole('button', { name: 'Queue experiments' })).toBeDisabled();
    expect(screen.getByText('Wait for the data preview to finish.')).toBeInTheDocument();
    runControls.experimentBlockReason = 'Select a model before submitting.';
    rerender(<div><Toolbar /></div>);
    expect(screen.getByRole('button', { name: 'Queue experiments' }))
      .toHaveAccessibleDescription('Select a model before submitting.');
    expect(screen.queryByText('Wait for the data preview to finish.')).not.toBeInTheDocument();
  });

  it('closes experiment review when read-only mode is enabled', () => {
    // A previously opened dialog must not leave a mutation route available in read-only mode.
    renderToolbar();
    openExperimentReview();
    act(() => useViewStore.getState().setReadOnlyOverride('on'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Preview data' })).not.toBeInTheDocument();
    act(() => useViewStore.getState().setReadOnlyOverride('off'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(runControls.handleRunAll).not.toHaveBeenCalled();
  });

  it.each([false, true])('disables experiment review during pending submission in narrow mode %s', narrow => {
    // Duplicate submissions must be blocked through both toolbar layouts.
    containerWidth = narrow ? 600 : 1400;
    runControls.isRunningAll = true;
    renderToolbar();
    if (narrow) fireEvent.click(screen.getByRole('button', { name: 'More canvas tools' }));
    const button = screen.getByRole(narrow ? 'menuitem' : 'button', {
      name: narrow ? 'Queuing experiments...' : 'Run all parallel branches as separate experiments',
    });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('lets an invalid graph request preview feedback while hiding single-branch experiment review', () => {
    // Preview stays actionable so its handler can explain validation failures.
    runControls.hasMultipleBranches = false;
    renderToolbar();
    expect(screen.queryByRole('button', { name: 'Run all parallel branches as separate experiments' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Preview data' }));
    expect(runControls.handleRun).toHaveBeenCalledOnce();
  });

  it('clears only after confirmation and preserves undo recovery', async () => {
    // Cancelling must preserve the graph, and confirming must remain recoverable through Undo.
    confirm.mockResolvedValueOnce(false).mockResolvedValueOnce(true);
    renderToolbar();
    fireEvent.click(screen.getByRole('button', { name: 'Clear canvas' }));
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce());
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: 'Clear canvas' }));
    await waitFor(() => expect(useGraphStore.getState().nodes).toHaveLength(0));
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(useGraphStore.getState().nodes[0]?.id).toBe('dataset');
  });

  it.each([{ key: 'z', shiftKey: true }, { key: 'y', shiftKey: false }])('supports undo and redo hotkeys with $key', redoKey => {
    // Canvas history must remain reachable through both supported redo shortcuts.
    renderToolbar();
    act(() => useGraphStore.getState().setGraph([], []));
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(1);
    fireEvent.keyDown(window, { ...redoKey, metaKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(0);
  });

  it('preserves input undo and blocks canvas history shortcuts in read-only mode', () => {
    // Native editing and read-only mode must never accidentally mutate the graph.
    render(<div><input aria-label="Pipeline name" /><Toolbar /></div>);
    act(() => useGraphStore.getState().setGraph([], []));
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'z', ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    act(() => useViewStore.getState().setReadOnlyOverride('on'));
    fireEvent.keyDown(window, { key: 'z', ctrlKey: true });
    expect(useGraphStore.getState().nodes).toHaveLength(0);
    expect(screen.queryByRole('button', { name: 'Undo' })).not.toBeInTheDocument();
  });

  it('moves focus from the narrow overflow into the legend and returns it on Escape', async () => {
    // The overflow item unmounts, so focus must move to surviving accessible controls.
    containerWidth = 600;
    renderToolbar();
    const more = screen.getByRole('button', { name: 'More canvas tools' });
    fireEvent.click(more);
    fireEvent.click(screen.getByRole('menuitem', { name: 'Node badge legend' }));
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close legend' })).toHaveFocus();
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByText('Canvas Legend')).not.toBeInTheDocument();
    await waitFor(() => expect(more).toHaveFocus());
  });

  it.each([1400, 600])('preserves save/load and pending-preview restrictions at width %i', width => {
    // The narrow menu and full toolbar expose the same pipeline actions and busy state.
    containerWidth = width;
    const { rerender } = renderToolbar();
    if (width < 720) fireEvent.click(screen.getByRole('button', { name: 'More canvas tools' }));
    fireEvent.click(screen.getByRole(width < 720 ? 'menuitem' : 'button', { name: 'Save pipeline' }));
    expect(vi.mocked(usePipelineActions).mock.results[0]!.value.handleSave).toHaveBeenCalledOnce();
    if (width < 720) fireEvent.click(screen.getByRole('button', { name: 'More canvas tools' }));
    fireEvent.click(screen.getByRole(width < 720 ? 'menuitem' : 'button', { name: 'Load pipeline' }));
    expect(vi.mocked(usePipelineActions).mock.results[0]!.value.openLoadMenu).toHaveBeenCalledOnce();
    runControls.isRunning = true;
    rerender(<div><Toolbar /></div>);
    if (width < 720) fireEvent.click(screen.getByRole('button', { name: 'More canvas tools' }));
    expect(screen.getByRole(width < 720 ? 'menuitem' : 'button', { name: 'Save pipeline' })).toBeDisabled();
    expect(screen.getByRole(width < 720 ? 'menuitem' : 'button', { name: 'Load pipeline' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Previewing data...' })).toBeDisabled();
  });

  it.each([1400, 600])('exports images and notebooks from the visible menu at width %i', async width => {
    // Both layouts must dispatch the chosen format and close their menu before export.
    containerWidth = width;
    vi.mocked(exportCanvasToPng).mockResolvedValue('data:image/png;base64,canvas');
    vi.mocked(exportCanvasToSvg).mockResolvedValue('data:image/svg+xml,canvas');
    renderToolbar();
    const trigger = screen.getByRole('button', { name: width < 720 ? 'More canvas tools' : 'Export canvas as image' });
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('menuitem', { name: width < 720 ? 'Export PNG' : 'PNG (high-DPI)' }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Canvas exported as PNG'));
    expect(exportCanvasToPng).toHaveBeenCalledWith('skyulf-canvas.png');
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('menuitem', { name: width < 720 ? 'Export SVG' : 'SVG (vector)' }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Canvas exported as SVG'));
    expect(exportCanvasToSvg).toHaveBeenCalledWith('skyulf-canvas.svg');
    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole('menuitem', { name: 'Notebook (full)' }));
    expect(vi.mocked(usePipelineActions).mock.results[0]!.value.exportNotebook).toHaveBeenCalledWith('full');
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();
  });
});
