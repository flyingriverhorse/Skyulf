import type { DatasetSourceSummary, FeaturePipelineResponse } from '../../api';
import type { SaveFeedback } from '../types/feedback';
import type { DatasetOption } from './useDatasetSelection';

type UseSidepanelViewModelOptions = {
  activePipelineUpdatedAt: string | null;
  datasetErrorMessage: string | null;
  isDatasetLoading: boolean;
  hasDatasets: boolean;
  ownedDatasetsCount: number;
  canSelectDatasets: boolean;
  datasetOptions: DatasetOption[];
  selectedDataset: DatasetSourceSummary | null;
  activeSourceId: string | null;
  canClearCanvas: boolean;
  onDatasetSelection: (value: string) => void;
  onClearCanvas: () => void;
  onRefreshHistory: () => void;
  onSaveClick: () => void;
  isHistoryLoading: boolean;
  historyErrorMessage: string | null;
  historyItems: FeaturePipelineResponse[];
  activePipelineId: number | null;
  isDirty: boolean;
  isSaving: boolean;
  saveFeedback: SaveFeedback | null;
  feedbackIcon: string;
  feedbackClass: string;
  onSelectHistory: (pipeline: FeaturePipelineResponse) => void;
};

type SidepanelProps = {
  activePipelineUpdatedAt: string | null;
  datasetErrorMessage: string | null;
  isDatasetLoading: boolean;
  hasDatasets: boolean;
  ownedDatasetsCount: number;
  canSelectDatasets: boolean;
  datasetOptions: DatasetOption[];
  selectedDataset: DatasetSourceSummary | null;
  activeSourceId: string | null;
  canClearCanvas: boolean;
  onDatasetSelection: (value: string) => void;
  onClearCanvas: () => void;
  onRefreshHistory: () => void;
  onSaveClick: () => void;
  isHistoryLoading: boolean;
  historyErrorMessage: string | null;
  historyItems: FeaturePipelineResponse[];
  activePipelineId: number | null;
  isDirty: boolean;
  isSaveDisabled: boolean;
  saveButtonLabel: string;
  saveFeedback: SaveFeedback | null;
  feedbackIcon: string;
  feedbackClass: string;
  onSelectHistory: (pipeline: FeaturePipelineResponse) => void;
};

type UseSidepanelViewModelResult = {
  sidepanelProps: SidepanelProps;
};

export const useSidepanelViewModel = ({
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
  onDatasetSelection,
  onClearCanvas,
  onRefreshHistory,
  onSaveClick,
  isHistoryLoading,
  historyErrorMessage,
  historyItems,
  activePipelineId,
  isDirty,
  isSaving,
  saveFeedback,
  feedbackIcon,
  feedbackClass,
  onSelectHistory,
}: UseSidepanelViewModelOptions): UseSidepanelViewModelResult => {
  const saveButtonLabel = isSaving ? 'Saving…' : isDirty ? 'Save draft*' : 'Save draft';
  const isSaveDisabled = !activeSourceId || isSaving;

  return {
    sidepanelProps: {
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
      onDatasetSelection,
      onClearCanvas,
      onRefreshHistory,
      onSaveClick,
      isHistoryLoading,
      historyErrorMessage,
      historyItems,
      activePipelineId,
      isDirty,
      isSaveDisabled,
      saveButtonLabel,
      saveFeedback,
      feedbackIcon,
      feedbackClass,
      onSelectHistory,
    },
  };
};
