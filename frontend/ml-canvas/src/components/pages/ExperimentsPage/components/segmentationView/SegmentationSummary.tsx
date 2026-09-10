import { Download, Loader2, Check } from 'lucide-react';
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
import { InfoTooltip } from '../../../../ui/InfoTooltip';
import { ChartDataTable } from '../../../../eda/ChartDataTable';
import { MetricDirectionBadge } from '../../../../ui/MetricDirectionBadge';
import { getMetricDirection } from '../../../../../core/utils/metricMeta';
import { useChartTheme } from '../../../../../core/hooks/useChartTheme';
import type { ClusteringSplit } from '../../types';

const CLUSTER_COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#ff6b6b', '#4ecdc4'];

const METRIC_LABELS: Record<string, string> = {
  silhouette_score: 'Silhouette Score',
  calinski_harabasz_score: 'Calinski-Harabasz Score',
  davies_bouldin_score: 'Davies-Bouldin Score',
};

interface Props {
  isEvalLoading: boolean;
  currentSplit: ClusteringSplit;
  clustering: NonNullable<ClusteringSplit['clustering']>;
  currentSplitName: string | null;
  splitTabs: string[];
  setActiveSplit: (split: string) => void;
  clusterSizeChartData: { cluster: string; size: number; clusterId: number }[];
  clusterSizeTableRows: { cluster: string; size: number }[];
  metricTableRows: { metric: string; value: number | string; direction: string }[];
  chartTheme: ReturnType<typeof useChartTheme>;
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

/** Ready clustering summary and its downloadable plot and tables. */
export function SegmentationSummary({ isEvalLoading, currentSplit, clustering, currentSplitName, splitTabs, setActiveSplit, clusterSizeChartData, clusterSizeTableRows, metricTableRows, chartTheme, handleDownload, downloadingChart, doneChart }: Props) {
  return (
    <div className={`space-y-6 transition-opacity ${isEvalLoading ? 'opacity-60' : ''}`}>
      {/* Split tabs */}
      {splitTabs.length > 1 && (
        <div className="flex items-center gap-0.5">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mr-1">Split:</span>
          {splitTabs.map((tab) => (
            <button
              key={tab}
              onClick={() => { setActiveSplit(tab); }}
              className={`px-3 py-1 text-xs font-medium rounded ${currentSplitName === tab
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
          <p className="text-2xl font-bold text-purple-700 dark:text-purple-300">{clustering.n_clusters}</p>
        </div>
        {['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'].map((key) => {
          const value = currentSplit.metrics?.[key];
          return (
            <div key={key} className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between gap-1 mb-1">
                <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400">{METRIC_LABELS[key]}</h4>
                <MetricDirectionBadge direction={getMetricDirection(key)} />
              </div>
              <p className="text-2xl font-bold text-gray-800 dark:text-gray-100">
                {typeof value === 'number' ? value.toFixed(3) : 'not reported'}
              </p>
            </div>
          );
        })}
      </div>

      <ChartDataTable
        columns={[
          { key: 'metric', label: 'Metric' },
          { key: 'value', label: 'Value' },
          { key: 'direction', label: 'Direction' },
        ]}
        rows={metricTableRows}
        filename={`segmentation_metrics_${currentSplitName ?? 'split'}`}
        caption="Cluster quality metrics with direction (higher/lower is better) for this run and split"
      />

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
        <ChartDataTable
          columns={[
            { key: 'cluster', label: 'Cluster' },
            { key: 'size', label: 'Size (rows)' },
          ]}
          rows={clusterSizeTableRows}
          filename={`segmentation_cluster_sizes_${currentSplitName ?? 'split'}`}
          caption="Cluster size data, as an alternative to the bar chart above"
        />
      </div>

      {/* Per-cluster centroid cards */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Cluster Profiles (Centroids)</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {clustering.centroids.map((cluster) => (
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
      {clustering.reference_crosstab && clustering.reference_column && (
        <div>
          <div className="flex items-center gap-1.5 mb-3">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Reference Column Breakdown ({clustering.reference_column})
            </h4>
            <InfoTooltip
              text={`How rows are distributed across "${clustering.reference_column}" within each cluster. This column was excluded from training — it's shown here only to help you interpret which cluster corresponds to which real-world group.`}
              align="center"
            />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(clustering.reference_crosstab)
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
    </div>);
}
