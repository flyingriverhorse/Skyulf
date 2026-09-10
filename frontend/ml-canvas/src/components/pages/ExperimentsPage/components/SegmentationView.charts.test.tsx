import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { ReactNode } from 'react';
import type { ClusteringSplit, EvaluationData } from '../types';
import { SegmentationView } from './SegmentationView';

vi.mock('recharts', () => {
  const chart = ({ children, data }: { children?: ReactNode; data?: unknown }) => <div data-series={JSON.stringify(data)}>{children}</div>;
  const empty = () => null;
  return { ResponsiveContainer: chart, BarChart: chart, Bar: chart, Cell: empty, XAxis: empty, YAxis: empty, CartesianGrid: empty, Tooltip: empty };
});

const train: ClusteringSplit = { labels: [-1, 2, 10], metrics: { silhouette_score: 0, calinski_harabasz_score: NaN }, clustering: { n_clusters: 2, cluster_sizes: { '10': 1, '2': 1, '-1': 1 }, centroids: [{ cluster_id: 2, size: 1, percentage: 33.33, center: { low: -3, high: 8, a: 1, b: 2, c: 5, d: 4 }, profile: 'High values' }], reference_column: 'species', reference_crosstab: { '2': { cat: 0, dog: 0 } } } };
const props = { selectedJobIds: ['a', 'b'], coverageEntries: [], evalJobId: 'a', fetchEvaluationData: vi.fn(), isEvalLoading: false, evalError: null, handleDownload: vi.fn(), downloadingChart: null, doneChart: null };

describe('segmentation chart and split contracts', () => {
  it('orders train/test/validation and numeric clusters while preserving noise, zero and missing metrics', () => {
    // Cluster -1 and absent quality scores retain their existing presentation.
    const evaluationData: EvaluationData = { problem_type: 'clustering', splits: { validation: train, test: { ...train, clustering: { n_clusters: 0, cluster_sizes: {}, centroids: [] } }, train } };
    const { container, rerender } = render(<SegmentationView {...props} evaluationData={evaluationData} />);
    expect(Array.from(container.querySelectorAll('[data-series]'), node => JSON.parse(node.getAttribute('data-series')!))).toContainEqual([{ cluster: 'Cluster -1', size: 1, clusterId: -1 }, { cluster: 'Cluster 2', size: 1, clusterId: 2 }, { cluster: 'Cluster 10', size: 1, clusterId: 10 }]);
    expect(screen.getByText('0.000')).toBeInTheDocument();
    expect(screen.getByText('NaN')).toBeInTheDocument();
    expect(screen.getByText('not reported')).toBeInTheDocument();
    expect(screen.getByText('Reference Column Breakdown (species)')).toBeInTheDocument();
    expect(screen.getAllByText('0 (0%)')).toHaveLength(2);
    expect(screen.getByText('+ 1 more features')).toBeInTheDocument();
    fireEvent.click(screen.getByTitle('Download Graph'));
    expect(props.handleDownload).toHaveBeenCalledWith('segmentation-cluster-sizes-chart', 'segmentation_cluster_sizes');
    fireEvent.click(screen.getByRole('button', { name: 'Test' }));
    expect(screen.queryByText('High values')).not.toBeInTheDocument();
    rerender(<SegmentationView {...props} evaluationData={{ problem_type: 'clustering', splits: { train } }} />);
    expect(screen.getByText('High values')).toBeInTheDocument();
  });
});
