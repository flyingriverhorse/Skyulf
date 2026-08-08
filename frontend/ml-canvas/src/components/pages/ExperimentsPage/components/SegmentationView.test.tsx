import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SegmentationView } from './SegmentationView';
import type { ArtifactCoverageEntry } from './ArtifactCoverageList';
import type { EvaluationData } from '../types';

afterEach(() => {
  document.documentElement.classList.remove('dark');
  Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 1024 });
});

const clusteringEvaluationData: EvaluationData = {
  problem_type: 'clustering',
  splits: {
    train: {
      labels: [0, 0, 1],
      metrics: {
        silhouette_score: 0.62,
        calinski_harabasz_score: 145.3,
        davies_bouldin_score: 0.71,
      },
      clustering: {
        n_clusters: 2,
        cluster_sizes: { '0': 2, '1': 1 },
        centroids: [
          { cluster_id: 0, size: 2, percentage: 66.7, center: { feature_a: 1.2, feature_b: -0.3 } },
          { cluster_id: 1, size: 1, percentage: 33.3, center: { feature_a: -1.1, feature_b: 0.8 } },
        ],
      },
    },
  },
};

describe('SegmentationView error retry', () => {
  it('renders a retry button when evaluation loading fails', () => {
    const fetchEvaluationData = vi.fn();
    render(
      <SegmentationView
        selectedJobIds={['job-1']}
        coverageEntries={[]}
        evalJobId="job-1"
        fetchEvaluationData={fetchEvaluationData}
        isEvalLoading={false}
        evalError="Failed to fetch evaluation data"
        evaluationData={null}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(fetchEvaluationData).toHaveBeenCalledTimes(1);
    expect(fetchEvaluationData).toHaveBeenCalledWith('job-1');
  });
});

describe('SegmentationView artifact coverage and metric direction', () => {
  it('renders the availability list distinguishing supported, unsupported, not-yet-computed, and failed runs', () => {
    const coverageEntries: ArtifactCoverageEntry[] = [
      { jobId: 'a', label: 'kmeans (aaaaaaaa)', status: 'available', reason: 'Clustering summary is available for this run.' },
      { jobId: 'b', label: 'random_forest (bbbbbbbb)', status: 'unsupported', reason: 'Not a Segmentation (clustering) run.' },
      { jobId: 'c', label: 'kmeans (cccccccc)', status: 'not_computed', reason: 'Run has not finished yet.' },
      { jobId: 'd', label: 'kmeans (dddddddd)', status: 'failed', reason: 'Run failed before this artifact could be produced.' },
    ];
    render(
      <SegmentationView
        selectedJobIds={['a', 'b', 'c', 'd']}
        coverageEntries={coverageEntries}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={clusteringEvaluationData}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    expect(screen.getByText('Available')).toBeInTheDocument();
    expect(screen.getByText('Unsupported')).toBeInTheDocument();
    expect(screen.getByText('Not computed')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('marks silhouette and calinski-harabasz as higher-is-better, and davies-bouldin as lower-is-better', () => {
    render(
      <SegmentationView
        selectedJobIds={['a']}
        coverageEntries={[]}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={clusteringEvaluationData}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    expect(screen.getAllByText(/higher is better/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/lower is better/i).length).toBeGreaterThanOrEqual(1);
  });

  it('provides a cluster-metrics data table with an explicit direction column', () => {
    render(
      <SegmentationView
        selectedJobIds={['a']}
        coverageEntries={[]}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={clusteringEvaluationData}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    const metricsTableButtons = screen.getAllByRole('button', { name: /view data table/i });
    fireEvent.click(metricsTableButtons[0]!);
    expect(screen.getByText('Direction')).toBeInTheDocument();
    expect(screen.getAllByText(/higher is better/i).length).toBeGreaterThan(0);
  });

  it('provides a cluster-size data table as an alternative to the bar chart', () => {
    render(
      <SegmentationView
        selectedJobIds={['a']}
        coverageEntries={[]}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={clusteringEvaluationData}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );

    const tableButtons = screen.getAllByRole('button', { name: /view data table/i });
    fireEvent.click(tableButtons[tableButtons.length - 1]!);
    expect(screen.getByText('Size (rows)')).toBeInTheDocument();
  });

  it('renders correctly in dark mode', () => {
    document.documentElement.classList.add('dark');
    render(
      <SegmentationView
        selectedJobIds={['a']}
        coverageEntries={[]}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={clusteringEvaluationData}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    expect(screen.getByText('Segmentation')).toBeInTheDocument();
  });

  it('renders correctly at a 390px viewport', () => {
    Object.defineProperty(window, 'innerWidth', { writable: true, configurable: true, value: 390 });
    window.dispatchEvent(new Event('resize'));
    render(
      <SegmentationView
        selectedJobIds={['a']}
        coverageEntries={[]}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={clusteringEvaluationData}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    expect(screen.getByText('Segmentation')).toBeInTheDocument();
  });

  it('references the availability list above when the active run is not a clustering job', () => {
    render(
      <SegmentationView
        selectedJobIds={['a']}
        coverageEntries={[
          { jobId: 'a', label: 'random_forest (aaaaaaaa)', status: 'unsupported', reason: 'Not a Segmentation (clustering) run.' },
        ]}
        evalJobId="a"
        fetchEvaluationData={vi.fn()}
        isEvalLoading={false}
        evalError={null}
        evaluationData={{ problem_type: 'classification', splits: {} }}
        handleDownload={vi.fn()}
        downloadingChart={null}
        doneChart={null}
      />,
    );
    expect(screen.getByText('The selected run is not a Segmentation (clustering) job.')).toBeInTheDocument();
    expect(screen.getByText(/see the availability list above/i)).toBeInTheDocument();
  });
});
