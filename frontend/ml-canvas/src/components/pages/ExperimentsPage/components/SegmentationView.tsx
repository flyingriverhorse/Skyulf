import React, { useMemo } from 'react';
import { Boxes } from 'lucide-react';
import { SegmentationSummary } from './segmentationView/SegmentationSummary';
import { SegmentationStatus, getClusteringSplit } from './segmentationView/SegmentationStatus';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import { getMetricDirection } from '../../../../core/utils/metricMeta';
import { useChartTheme } from '../../../../core/hooks/useChartTheme';
import type { EvaluationData } from '../types';
import { ArtifactCoverageList, type ArtifactCoverageEntry } from './ArtifactCoverageList';

const METRIC_LABELS: Record<string, string> = {
  silhouette_score: 'Silhouette Score',
  calinski_harabasz_score: 'Calinski-Harabasz Score',
  davies_bouldin_score: 'Davies-Bouldin Score',
};

interface Props {
  selectedJobIds: string[];
  /** Per-selected-run availability of the clustering summary, so a run that
   * doesn't support segmentation is distinguished from one that's still
   * pending or failed (UX finding EXP-003). */
  coverageEntries: ArtifactCoverageEntry[];
  evalJobId: string | null;
  fetchEvaluationData: (jobId: string) => void | Promise<void>;
  isEvalLoading: boolean;
  evalError: string | null;
  evaluationData: EvaluationData | null;
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const SegmentationView: React.FC<Props> = ({
  selectedJobIds,
  coverageEntries,
  evalJobId,
  fetchEvaluationData,
  isEvalLoading,
  evalError,
  evaluationData,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartTheme = useChartTheme();

  const splitTabs = useMemo(() => {
    if (!evaluationData || evaluationData.problem_type !== 'clustering') return [];
    return ['train', 'test', 'validation'].filter((s) => evaluationData.splits[s] != null);
  }, [evaluationData]);

  const [activeSplit, setActiveSplit] = React.useState<string | null>(null);
  const currentSplitName = activeSplit && splitTabs.includes(activeSplit) ? activeSplit : splitTabs[0] ?? null;

  const currentSplit = getClusteringSplit(evaluationData, currentSplitName);

  const clusterSizeChartData = useMemo(() => {
    if (!currentSplit?.clustering) return [];
    return Object.entries(currentSplit.clustering.cluster_sizes)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([clusterId, size]) => ({ cluster: `Cluster ${clusterId}`, size, clusterId: Number(clusterId) }));
  }, [currentSplit]);

  const clusterSizeTableRows = useMemo(
    () => clusterSizeChartData.map((row) => ({ cluster: row.cluster, size: row.size })),
    [clusterSizeChartData],
  );

  const metricTableRows = useMemo(() => {
    if (!currentSplit) return [];
    return ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'].map((key) => {
      const value = currentSplit.metrics?.[key];
      const direction = getMetricDirection(key);
      return {
        metric: METRIC_LABELS[key] ?? key,
        value: typeof value === 'number' ? Number(value.toFixed(4)) : 'not reported',
        direction: direction === 'higher' ? 'Higher is better' : direction === 'lower' ? 'Lower is better' : 'Direction unknown',
      };
    });
  }, [currentSplit]);

  const retryJobId = evalJobId ?? selectedJobIds[0] ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100 flex items-center gap-2">
          <Boxes className="w-5 h-5 text-purple-500" />
          Segmentation
        </h3>
        <InfoTooltip
          text="Cluster quality metrics, sizes, per-cluster centroids (mean feature values), and an auto-generated characteristic profile for this run. There is no ground truth here — these metrics describe how well-separated the discovered groups are, not prediction accuracy. If you set a Reference Column, its breakdown per cluster is shown below to help you interpret what each cluster represents."
          align="center"
        />
      </div>

      <ArtifactCoverageList entries={coverageEntries} />

      {/* Job selector if multiple */}
      {selectedJobIds.length > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-2" role="tablist" aria-label="Select run for segmentation">
          {selectedJobIds.map((id) => {
            const isActive = evalJobId === id;
            return (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={isActive}
                onClick={() => { void fetchEvaluationData(id); }}
                className={`px-3 py-1 text-xs font-mono rounded border whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                  isActive
                    ? 'bg-blue-100 border-blue-300 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300'
                    : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700'
                }`}
              >
                {id.slice(0, 8)}
              </button>
            );
          })}
        </div>
      )}

      <SegmentationStatus evalError={evalError} evaluationData={evaluationData} isEvalLoading={isEvalLoading} retryJobId={retryJobId} fetchEvaluationData={fetchEvaluationData} hasSummary={!!currentSplit?.clustering}>
        {currentSplit?.clustering && <SegmentationSummary isEvalLoading={isEvalLoading} currentSplit={currentSplit} clustering={currentSplit.clustering} currentSplitName={currentSplitName} splitTabs={splitTabs} setActiveSplit={setActiveSplit} clusterSizeChartData={clusterSizeChartData} clusterSizeTableRows={clusterSizeTableRows} metricTableRows={metricTableRows} chartTheme={chartTheme} handleDownload={handleDownload} downloadingChart={downloadingChart} doneChart={doneChart} />}
      </SegmentationStatus>
    </div>
  );
};
