import { useMemo } from 'react';
import { collectGraphValidationIssues, useGraphStore } from '../store/useGraphStore';
import { useNodeInspectionStore } from '../store/useNodeInspectionStore';
import { buildPreviewConfiguration, previewConfigurationKey } from '../utils/previewConfiguration';
import { useReadOnlyMode } from './useReadOnlyMode';
import { groupNodeInspections } from '../utils/nodeInspectionPaths';

/** Present only the selected node's captured execution, tied to its submitted graph. */
export function useNodeInspection(nodeId: string) {
  const nodes = useGraphStore(state => state.nodes);
  const edges = useGraphStore(state => state.edges);
  const predictedSchema = useGraphStore(state => state.predictedSchemas[nodeId] ?? null);
  const { receipt, isLoading, error } = useNodeInspectionStore();
  const readOnly = useReadOnlyMode();
  const config = useMemo(() => buildPreviewConfiguration(nodes, edges), [nodes, edges]);
  const configurationKey = useMemo(() => previewConfigurationKey(config), [config]);
  const issues = useMemo(() => collectGraphValidationIssues(nodes, edges), [nodes, edges]);
  const blockReason = readOnly ? 'Turn off read-only mode to run Preview data.'
    : !nodes.some(node => node.data.definitionType === 'dataset_node' && node.data.datasetId)
      ? 'Select a dataset before running a preview.'
      : issues.length ? `Fix ${issues.length} validation issue${issues.length === 1 ? '' : 's'} first. ${issues[0]!.message}`
        : !config.nodes.some(node => node.node_id === nodeId) ? 'This node does not run in data previews. Use its settings to inspect its own job.' : null;
  const branches = useMemo(() => groupNodeInspections(
    receipt?.response.node_inspections?.filter(branch => branch.node_id === nodeId) ?? [],
  ), [receipt, nodeId]);
  const selectedReceipt = branches.length ? receipt : null;

  return {
    branches,
    runId: selectedReceipt?.response.run_id ?? null,
    isStale: selectedReceipt !== null && selectedReceipt.configurationKey !== configurationKey,
    isLoading,
    error,
    predictedSchema,
    blockReason,
  };
}
