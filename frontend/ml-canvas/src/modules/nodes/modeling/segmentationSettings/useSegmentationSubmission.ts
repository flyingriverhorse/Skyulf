import type { Edge, Node } from '@xyflow/react';
import { jobsApi, type RunPipelineResponse } from '../../../../core/api/jobs';
import { useJobStore } from '../../../../core/store/useJobStore';
import { useViewStore } from '../../../../core/store/useViewStore';
import { toast } from '../../../../core/toast';
import type { NodeSubmission } from '../../../../core/types/runFeedback';
import { convertGraphToPipelineConfig } from '../../../../core/utils/pipelineConverter';
import { warnAndBlockOnLeakage } from '../../../../core/utils/pipelineLeakageValidation';
import { getLeakageErrorMessage, graphSemanticSignature } from '../../../../core/utils/leakageFeedback';
import type { SegmentationConfig } from '../SegmentationSettings';

/** Coordinate a node-scoped submission and preserve its captured graph and store actions. */
export function useSegmentationSubmission(
  config: SegmentationConfig, nodeId: string | undefined, datasetId: string | undefined, nodes: Node[], edges: Edge[],
) {
  const feedback = useJobStore(state => nodeId ? state.nodeSubmissions[nodeId] : undefined);
  const isSubmitting = feedback?.pending ?? false;
  const submissionMessage = feedback?.message ?? '';
  const runFeedback = feedback?.run;
  const { toggleDrawer: toggleJobDrawer, setTab, setActiveParallelRun, startPolling } = useJobStore();

  const handleSubmitted = (response: RunPipelineResponse, label: string, update: (value: NodeSubmission) => void) => {
    const jobCount = response.job_ids?.length || 1;
    const run = { label, jobIds: response.job_ids?.length ? response.job_ids : [response.job_id] };
    update({ pending: false, run, message: '' });
    startPolling();
    if (jobCount > 1) {
      setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
      toast.success('Parallel execution started', `${jobCount} branches submitted.`);
    } else {
      toast.success('Segmentation job submitted');
    }
    setTab('segmentation');
    useJobStore.getState().setInspectedRun(null);
    toggleJobDrawer(true);
  };

  const handleTrain = async () => {
    if (!nodeId || useJobStore.getState().nodeSubmissions[nodeId]?.pending) return;
    if (!datasetId) {
      toast.error('No dataset connected', 'Connect a dataset node upstream before starting training.');
      return;
    }
    const update = (value: NodeSubmission) => useJobStore.getState().setNodeSubmission(nodeId, value);
    const label = `Segmentation — ${config.model_type.replace(/_/g, ' ')}`;
    useViewStore.getState().setLeakageNotice(null);
    const graphSignature = graphSemanticSignature(nodes, edges);
    update({ pending: true, run: null, message: `${label}: Submitting...` });
    try {
      const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
      if (warnAndBlockOnLeakage(pipelineConfig)) {
        update({ pending: false, run: null, message: `${label} blocked. Move data-learning preprocessing after the train/test split.` });
        return;
      }
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        target_node_id: nodeId,
        job_type: 'training'
      });
      handleSubmitted(response, label, update);
    } catch (error) {
      console.error('Failed to submit segmentation job:', error);
      const leakageMessage = getLeakageErrorMessage(error);
      if (leakageMessage) useViewStore.getState().setLeakageNotice({ message: leakageMessage, graphSignature });
      update({ pending: false, run: null, message: `${label}: Submission failed. ${leakageMessage ?? 'Check your connection and settings, then try again.'}` });
      toast.error('Failed to submit segmentation job', leakageMessage ?? 'Check console for details.');
    }
  };

  return { isSubmitting, submissionMessage, runFeedback, handleTrain };
}
