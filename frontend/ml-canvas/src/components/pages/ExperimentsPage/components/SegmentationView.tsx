import React, { useMemo } from 'react';
import { Boxes, Download, Loader2, Check } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { LoadingState, ErrorState } from '../../../shared';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import { useChartTheme } from '../../../../core/hooks/useChartTheme';
import type { EvaluationData, ClusteringSplit } from '../types';

const CLUSTER_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#ff6b6b', '#4ecdc4'];

const METRIC_LABELS: Record<string, string> = {
  silhouette_score: 'Silhouette Score',
  calinski_harabasz_score: 'Calinski-Harabasz Score',
  davies_bouldin_score: 'Davies-Bouldin Score',
};

interface Props {
  selectedJobIds: string[];
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

  const currentSplit: ClusteringSplit | null =
    evaluationData && evaluationData.problem_type === 'clustering' && currentSplitName
      ? (evaluationData.splits[currentSplitName] ?? null)
      : null;

  const clusterSizeChartData = useMemo(() => {
    if (!currentSplit?.clustering) return [];
    return Object.entries(currentSplit.clustering.cluster_sizes)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([clusterId, size]) => ({ cluster: `Cluster ${clusterId}`, size, clusterId: Number(clusterId) }));
  }, [currentSplit]);

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

      {evalError ? (
        <div className="h-64 flex items-center justify-center">
          <ErrorState error={evalError} />
        </div>
      ) : !evaluationData ? (
        isEvalLoading ? (
          <div className="h-64 flex items-center justify-center">
            <LoadingState message="Loading segmentation data..." />
          </div>
        ) : (
          <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
            <p>Select a completed Segmentation run to view cluster details.</p>
          </div>
        )
      ) : evaluationData.problem_type !== 'clustering' ? (
        <div className="h-64 flex flex-col items-center justify-center text-gray-400 italic text-center">
          <p>The selected run is not a Segmentation (clustering) job.</p>
        </div>
      ) : !currentSplit?.clustering ? (
        <div className="h-64 flex items-center justify-center text-gray-400 italic">
          No cluster summary available for this run.
        </div>
      ) : (
        <div className={`space-y-6 transition-opacity ${isEvalLoading ? 'opacity-60' : ''}`}>
          {/* Split tabs */}
          {splitTabs.length > 1 && (
            <div className="flex items-center gap-0.5">
              <span className="text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide mr-1">Split:</span>
              {splitTabs.map((tab) => (
                <button
                  key={tab}
                  onClick={() => { setActiveSplit(tab); }}
                  className={`px-3 py-1 text-xs font-medium rounded ${
                    currentSplitName === tab
                      ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                      : 'text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700'
                  }`}
                >
                  {tab === 'validation' ? 'Validation' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
          )}

          {/* Score cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 bg-purple-50 dark:bg-purple-900/10 rounded-lg border border-purple-100 dark:border-purple-800">
              <h4 className="text-xs font-semibold text-purple-900 dark:text-purple-100 mb-1">Clusters Found</h4>
              <p className="text-2xl font-bold text-purple-700 dark:text-purple-300">{currentSplit.clustering.n_clusters}</p>
            </div>
            {['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'].map((key) => {
              const value = currentSplit.metrics?.[key];
              return (
                <div key={key} className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                  <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">{METRIC_LABELS[key]}</h4>
                  <p className="text-2xl font-bold text-gray-800 dark:text-gray-100">
                    {typeof value === 'number' ? value.toFixed(3) : '—'}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Cluster size bar chart */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id="segmentation-cluster-sizes-chart">
            <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
              <button
                onClick={() => void handleDownload('segmentation-cluster-sizes-chart', 'segmentation_cluster_sizes')}
                disabled={downloadingChart === 'segmentation-cluster-sizes-chart'}
                className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                title="Download Graph"
              >
                {downloadingChart === 'segmentation-cluster-sizes-chart' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === 'segmentation-cluster-sizes-chart' ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
              </button>
            </div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Cluster Sizes</h4>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={clusterSizeChartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke={chartTheme.gridColor} />
                  <XAxis dataKey="cluster" tick={{ fontSize: 12, fill: chartTheme.axisColor }} />
                  <YAxis tick={{ fontSize: 12, fill: chartTheme.axisColor }} />
                  <Tooltip
                    contentStyle={chartTheme.tooltipContentStyle}
                    itemStyle={chartTheme.tooltipItemStyle}
                    labelStyle={chartTheme.tooltipLabelStyle}
                  />
                  <Bar dataKey="size" radius={[4, 4, 0, 0]}>
                    {clusterSizeChartData.map((entry) => (
                      <Cell key={entry.clusterId} fill={CLUSTER_COLORS[entry.clusterId % CLUSTER_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-cluster centroid cards */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Cluster Profiles (Centroids)</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {currentSplit.clustering.centroids.map((cluster) => (
                <div key={cluster.cluster_id} className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-gray-900 dark:text-white flex items-center">
                      <span
                        className="w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: CLUSTER_COLORS[cluster.cluster_id % CLUSTER_COLORS.length] }}
                      />
                      Cluster {cluster.cluster_id}
                    </span>
                    <span className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-gray-600 dark:text-gray-300">
                      {cluster.percentage.toFixed(1)}% ({cluster.size})
                    </span>
                  </div>
                  <div className="space-y-1">
                    {Object.entries(cluster.center)
                      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                      .slice(0, 5)
                      .map(([col, val]) => (
                        <div key={col} className="flex justify-between text-xs">
                          <span className="text-gray-500 dark:text-gray-400 truncate w-24" title={col}>{col}</span>
                          <span className="font-mono text-gray-700 dark:text-gray-200">{val.toFixed(2)}</span>
                        </div>
                      ))}
                    {Object.keys(cluster.center).length > 5 && (
                      <div className="text-xs text-center text-gray-400 italic pt-1">
                        + {Object.keys(cluster.center).length - 5} more features
                      </div>
                    )}
                  </div>
                  {cluster.profile && (
                    <div className="mt-2 pt-2 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 italic">
                      {cluster.profile}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Reference column breakdown, if the user set one aside for interpretation */}
          {currentSplit.clustering.reference_crosstab && currentSplit.clustering.reference_column && (
            <div>
              <div className="flex items-center gap-1.5 mb-3">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Reference Column Breakdown ({currentSplit.clustering.reference_column})
                </h4>
                <InfoTooltip
                  text={`How rows are distributed across "${currentSplit.clustering.reference_column}" within each cluster. This column was excluded from training — it's shown here only to help you interpret which cluster corresponds to which real-world group.`}
                  align="center"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(currentSplit.clustering.reference_crosstab)
                  .sort(([a], [b]) => Number(a) - Number(b))
                  .map(([clusterId, counts]) => {
                    const total = Object.values(counts).reduce((sum, n) => sum + n, 0);
                    const sortedCounts = Object.entries(counts).sort(([, a], [, b]) => b - a);
                    return (
                      <div key={clusterId} className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center mb-2">
                          <span
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: CLUSTER_COLORS[Number(clusterId) % CLUSTER_COLORS.length] }}
                          />
                          <span className="font-bold text-gray-900 dark:text-white">Cluster {clusterId}</span>
                        </div>
                        <div className="space-y-1">
                          {sortedCounts.map(([label, count]) => (
                            <div key={label} className="flex justify-between text-xs">
                              <span className="text-gray-500 dark:text-gray-400 truncate w-24" title={label}>{label}</span>
                              <span className="font-mono text-gray-700 dark:text-gray-200">
                                {count} ({total ? ((count / total) * 100).toFixed(0) : 0}%)
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
