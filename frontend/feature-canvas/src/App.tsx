// @ts-nocheck
import React, { useCallback, useMemo, useRef, useState } from 'react';
import { ReactFlowProvider } from 'react-flow-renderer';
import 'react-flow-renderer/dist/style.css';
import { useQueryClient } from '@tanstack/react-query';
import CanvasShell from './canvas/components/CanvasShell/CanvasShell';
import { CanvasSidepanel } from './canvas/components/CanvasSidepanel/CanvasSidepanel';
import { useDatasetSelectionHandler } from './canvas/hooks';
import { useCanvasSnapshotState } from './canvas/hooks';
import { useClearCanvasHandler } from './canvas/hooks';
import { usePipelineHistory } from './canvas/hooks';
import { usePipelineHydration } from './canvas/hooks';
import { usePipelineSave } from './canvas/hooks';
import { useSaveFeedbackState } from './canvas/hooks';
import { useSidepanelViewModel } from './canvas/hooks';
import { useSidepanelToggle } from './canvas/hooks';
import type { CanvasShellHandle } from './canvas/types/pipeline';
import { FeaturePipelinePayload } from './api';
import { buildPipelineSavePayload } from './canvas/utils/buildPipelineSavePayload';
import './styles.css';

const App: React.FC<AppProps> = ({ sourceId }) => {
  const queryClient = useQueryClient();
  const canvasShellRef = useRef<CanvasShellHandle | null>(null);
  const isHydratingRef = useRef(true);
  const { isSidepanelExpanded, handleToggleSidepanel } = useSidepanelToggle(true);
  const [isDirty, setIsDirty] = useState(false);
  const { saveFeedback, setSaveFeedback, feedbackIcon, feedbackClass } = useSaveFeedbackState();
  const {
    datasets,
    ownedDatasets,
    selectedDataset,
    setSelectedDataset,
    activeSourceId,
    setActiveSourceId,
    datasetOptions,
    canSelectDatasets,
    hasDatasets,
    ownedDatasetsCount,
    isDatasetLoading,
    datasetErrorMessage,
    handleDatasetSelection,
  } = useDatasetSelectionHandler({
    sourceId,
    setIsDirty,
    setSaveFeedback,
    isHydratingRef,
  });
  const [activePipelineId, setActivePipelineId] = useState<number | null>(null);
  const [activePipelineUpdatedAt, setActivePipelineUpdatedAt] = useState<string | null>(null);
  const [activePipelineName, setActivePipelineName] = useState<string | null>(null);

  const pipelineQueryKey = useMemo(
    () => ['feature-canvas', 'pipeline', activeSourceId],
    [activeSourceId]
  );
  const pendingHistoryIdRef = useRef<number | null>(null);
  const { graphSnapshotRef, canClearCanvas, handleGraphChange, applyHydratedSnapshot } =
    useCanvasSnapshotState({
      setIsDirty,
      setSaveFeedback,
      isHydratingRef,
    });

  const handleClearCanvas = useClearCanvasHandler({
    canvasShellRef,
    canClearCanvas,
    setSaveFeedback,
  });

  const {
    historyItems,
    isHistoryLoading,
    historyErrorMessage,
    refetchHistory,
    handleHistorySelection,
  } = usePipelineHistory({
    activeSourceId,
    isDirty,
    setSaveFeedback,
    pipelineQueryKey,
    pendingHistoryIdRef,
    isHydratingRef,
  });

  const { triggerSave, isSaving } = usePipelineSave({
    queryClient,
    setSaveFeedback,
    setIsDirty,
    setActivePipelineId,
    setActivePipelineName,
    setActivePipelineUpdatedAt,
    pendingHistoryIdRef,
  });

  const { handlePipelineHydrated, handlePipelineError } = usePipelineHydration({
    applyHydratedSnapshot,
    selectedDataset,
    activeSourceId,
    setActivePipelineId,
    setActivePipelineName,
    setActivePipelineUpdatedAt,
    setIsDirty,
    setSaveFeedback,
    pendingHistoryIdRef,
    isHydratingRef,
  });

  const handleSaveClick = useCallback(() => {
    if (!activeSourceId) {
      setSaveFeedback({ message: 'Select a dataset before saving.', tone: 'error' });
      return;
    }

    const snapshot = graphSnapshotRef.current;
    if (!snapshot.nodes || !snapshot.nodes.length) {
      setSaveFeedback({ message: 'Add nodes to the canvas before saving.', tone: 'error' });
      return;
    }

    const payload: FeaturePipelinePayload = buildPipelineSavePayload({
      snapshot,
      selectedDataset,
      activeSourceId,
    });

    setSaveFeedback({ message: 'Saving draft…', tone: 'info' });
    triggerSave({ sourceId: activeSourceId, payload });
  }, [activeSourceId, selectedDataset?.name, triggerSave]);

  const { sidepanelProps } = useSidepanelViewModel({
    activePipelineUpdatedAt,
    datasetErrorMessage,
    isDatasetLoading,
    hasDatasets,
    ownedDatasetsCount,
    canSelectDatasets,
    datasetOptions,
    selectedDataset,
    activeSourceId,
    canClearCanvas,
    onDatasetSelection: handleDatasetSelection,
    onClearCanvas: handleClearCanvas,
    onRefreshHistory: () => {
      void refetchHistory();
    },
    onSaveClick: handleSaveClick,
    isHistoryLoading,
    historyErrorMessage,
    historyItems,
    activePipelineId,
    isDirty,
    isSaving,
    saveFeedback,
    feedbackIcon,
    feedbackClass,
    onSelectHistory: handleHistorySelection,
  });
  return (
    <div
      className="feature-canvas-app"
      data-sidepanel-expanded={isSidepanelExpanded ? 'true' : 'false'}
    >
      <button
        type="button"
        className="canvas-sidepanel__toggle"
        onClick={handleToggleSidepanel}
        aria-label={isSidepanelExpanded ? 'Collapse details panel' : 'Expand details panel'}
        aria-expanded={isSidepanelExpanded}
        aria-controls="feature-canvas-sidepanel"
      >
        {isSidepanelExpanded ? '⟨' : '⟩'}
      </button>

      {!isSidepanelExpanded && saveFeedback && (
        <div
          className={`canvas-sidepanel__feedback canvas-sidepanel__feedback--floating ${feedbackClass}`}
        >
          <span className="canvas-sidepanel__feedback-icon" aria-hidden="true">{feedbackIcon}</span>
          <span className="canvas-sidepanel__feedback-text">{saveFeedback.message}</span>
        </div>
      )}

      {isSidepanelExpanded && <CanvasSidepanel {...sidepanelProps} />}

      <div className="feature-canvas-app__viewport">
        <ReactFlowProvider>
          <CanvasShell
            ref={canvasShellRef}
            sourceId={activeSourceId}
            datasetName={selectedDataset?.name ?? selectedDataset?.source_id ?? null}
            onGraphChange={handleGraphChange}
            onPipelineHydrated={handlePipelineHydrated}
            onPipelineError={handlePipelineError}
          />
        </ReactFlowProvider>
      </div>
    </div>
  );
};

export default App;
