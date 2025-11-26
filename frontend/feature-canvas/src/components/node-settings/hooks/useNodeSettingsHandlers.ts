import { useColumnSelectionHandlers } from './preview/useColumnSelectionHandlers';
import { useNodeSaveHandlers } from './core/useNodeSaveHandlers';
import { useImputationStrategyHandlers } from './imputation/useImputationStrategyHandlers';
import { useAliasStrategyHandlers } from './cleaning/useAliasStrategyHandlers';
import { useDateStrategyHandlers } from './cleaning/useDateStrategyHandlers';
import { useFeatureMathHandlers } from './feature_math/useFeatureMathHandlers';
import type { CatalogFlagMap } from './core/useCatalogFlags';

type UseNodeSettingsHandlersArgs = {
  catalogFlags: CatalogFlagMap;
  setConfigState: any;
  binningExcludedColumns: any;
  scalingExcludedColumns: any;
  availableColumns: string[];
  recommendations: any;
  nodeId: string;
  onUpdateConfig: any;
  onClose: any;
  sourceId: string | null;
  graphSnapshot: any;
  onUpdateNodeData: any;
  canResetNode: boolean;
  defaultConfigTemplate: any;
  onResetConfig: any;
  imputationMethodValues: any;
  imputationMethodOptions: any;
  imputerStrategyCount: number;
  setCollapsedStrategies: any;
  setImputerMissingFilter: any;
  node: any;
  aliasColumnSummary: any;
  aliasStrategyCount: number;
  dateStrategies: any;
  standardizeDatesColumnSummary: any;
  standardizeDatesMode: any;
  setCollapsedFeatureMath: any;
  configState: any;
};

export const useNodeSettingsHandlers = ({
  catalogFlags,
  setConfigState,
  binningExcludedColumns,
  scalingExcludedColumns,
  availableColumns,
  recommendations,
  nodeId,
  onUpdateConfig,
  onClose,
  sourceId,
  graphSnapshot,
  onUpdateNodeData,
  canResetNode,
  defaultConfigTemplate,
  onResetConfig,
  imputationMethodValues,
  imputationMethodOptions,
  imputerStrategyCount,
  setCollapsedStrategies,
  setImputerMissingFilter,
  node,
  aliasColumnSummary,
  aliasStrategyCount,
  dateStrategies,
  standardizeDatesColumnSummary,
  standardizeDatesMode,
  setCollapsedFeatureMath,
  configState,
}: UseNodeSettingsHandlersArgs) => {
  const columnSelectionHandlers = useColumnSelectionHandlers({
    catalogFlags,
    setConfigState,
    binningExcludedColumns,
    scalingExcludedColumns,
    availableColumns,
    recommendations,
  });

  const nodeSaveHandlers = useNodeSaveHandlers({
    configState,
    catalogFlags,
    nodeId,
    onUpdateConfig,
    onClose,
    sourceId,
    graphSnapshot,
    onUpdateNodeData,
    setConfigState,
    canResetNode,
    defaultConfigTemplate,
    onResetConfig,
  });

  const imputationStrategyHandlers = useImputationStrategyHandlers({
    setConfigState,
    imputationMethodValues,
    imputationMethodOptions,
    imputerStrategyCount,
    setCollapsedStrategies,
    setImputerMissingFilter,
  });

  const aliasStrategyHandlers = useAliasStrategyHandlers({
    catalogFlags,
    node,
    setConfigState,
    setCollapsedStrategies,
    aliasColumnSummary,
    aliasStrategyCount,
  });

  const dateStrategyHandlers = useDateStrategyHandlers({
    catalogFlags,
    node,
    setConfigState,
    setCollapsedStrategies,
    dateStrategies,
    standardizeDatesColumnSummary,
    standardizeDatesMode,
  });

  const featureMathHandlers = useFeatureMathHandlers({
    catalogFlags,
    setConfigState,
    setCollapsedFeatureMath,
  });

  return {
    columnSelectionHandlers,
    nodeSaveHandlers,
    imputationStrategyHandlers,
    aliasStrategyHandlers,
    dateStrategyHandlers,
    featureMathHandlers,
  };
};
