import React from 'react';
import type { ComponentProps } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { shortRunId } from '../utils/jobMeta';
import { EvaluationView } from './EvaluationView';
import { FeatureImportanceView } from './FeatureImportanceView';
import { ShapSummaryView } from './ShapSummaryView';
import { PipelineDiffView } from '../../experiments/PipelineDiffView';

vi.mock('../../../../core/api/jobs', () => ({
  jobsApi: {
    getJob: vi.fn(),
  },
}));

vi.mock('@xyflow/react', () => ({
  Background: () => null,
  Controls: () => null,
  ReactFlow: () => <div data-testid="reactflow" />,
  ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    BarChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Bar: ({ dataKey }: { dataKey: string }) => <div>{dataKey}</div>,
    CartesianGrid: () => null,
    ComposedChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Area: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Legend: () => null,
    Line: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    ReferenceDot: () => null,
    ReferenceLine: () => null,
    Tooltip: () => null,
    XAxis: () => null,
    YAxis: () => null,
    ScatterChart: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    Scatter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
    ZAxis: () => null,
    Cell: () => null,
  };
});

const evaluationData = {
  problem_type: 'classification' as const,
  splits: {
    train: {
      y_true: ['a', 'b'],
      y_pred: ['a', 'b'],
      y_proba: {
        classes: ['a', 'b'],
        values: [
          [0.9, 0.1],
          [0.2, 0.8],
        ],
      },
    },
  },
};

const renderWithClient = (ui: React.ReactElement) => {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
    },
  });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
};

const noop = async () => {};

const jobA = {
  jobId: 'job-1111aaaa',
  job_id: 'job-1111aaaa',
  pipeline_id: 'preview_alpha_run__branch_7',
  parent_pipeline_id: 'preview_parent_alpha_run__branch_2',
  dataset_name: 'Dataset Alpha',
  model_type: 'logistic_regression',
  created_at: '2026-01-01T08:30:00Z',
};

const jobB = {
  jobId: 'job-2222bbbb',
  job_id: 'job-2222bbbb',
  pipeline_id: 'preview_beta_run__branch_5',
  parent_pipeline_id: 'preview_parent_beta_run__branch_4',
  dataset_name: 'Dataset Beta',
  model_type: 'xgboost',
  created_at: '2026-01-02T09:45:00Z',
};

describe('run identifier labels', () => {
  it('EvaluationView shows short pipeline ids instead of job ids', () => {
    const evaluationViewProps: ComponentProps<typeof EvaluationView> = {
      eligibleJobIds: [jobA.job_id, jobB.job_id],
      eligibleJobs: [jobA, jobB],
      evalJobId: jobA.job_id,
      fetchEvaluationData: noop,
      isEvalLoading: false,
      evalError: null,
      evaluationData: evaluationData as never,
      selectedRegressionSplit: null,
      setSelectedRegressionSplit: vi.fn(),
      showTrainMetrics: true,
      setShowTrainMetrics: vi.fn(),
      showTestMetrics: true,
      setShowTestMetrics: vi.fn(),
      showValMetrics: true,
      setShowValMetrics: vi.fn(),
      threshold: 0.5,
      setThreshold: vi.fn(),
      selectedRocClass: 'a',
      setSelectedRocClass: vi.fn(),
      cmView: 'overall' as const,
      setCmView: vi.fn(),
      activeTab: 'slider' as const,
      setActiveTab: vi.fn(),
      selectedMetric: 'f1_weighted' as const,
      setSelectedMetric: vi.fn(),
      bestMetricInfos: [],
      handleDownload: noop,
      downloadingChart: null,
      doneChart: null,
      selectedTuningMetric: 'f1',
      onSelectedTuningMetricChange: vi.fn(),
      tuningPreview: null,
      tuningError: null,
      useTunedThresholds: false,
      onPreviewThresholds: noop,
      onSaveThresholds: noop,
      onToggleThresholds: noop,
      onClearThresholds: noop,
    };
    renderWithClient(
      <EvaluationView {...evaluationViewProps} />,
    );

    expect(screen.getByRole('tab', { name: shortRunId(jobA) })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: shortRunId(jobB) })).toBeInTheDocument();
    expect(screen.queryByText(jobA.job_id.slice(0, 8))).not.toBeInTheDocument();
  });

  it('FeatureImportanceView shows short pipeline ids in the legend', () => {
    const featureImportanceProps: ComponentProps<typeof FeatureImportanceView> = {
      featureImportancesByJob: [
        {
          jobId: jobA.job_id,
          pipeline_id: jobA.pipeline_id,
          parent_pipeline_id: jobA.parent_pipeline_id,
          modelType: 'unknown',
          importances: { feature_one: 0.8, feature_two: 0.2 },
        },
      ],
      coverageInputs: [
        {
          jobId: jobA.job_id,
          label: shortRunId(jobA),
          task: 'classification',
          status: 'completed',
          hasArtifact: true,
        },
      ],
      handleDownload: noop,
      downloadingChart: null,
      doneChart: null,
    };
    renderWithClient(
      <FeatureImportanceView {...featureImportanceProps} />,
    );

    expect(screen.getAllByText(shortRunId(jobA)).length).toBeGreaterThan(0);
    expect(screen.queryByText(jobA.job_id.slice(0, 8))).not.toBeInTheDocument();
  });

  it('ShapSummaryView shows short pipeline ids in the legend', () => {
    const shapSummaryProps: ComponentProps<typeof ShapSummaryView> = {
      shapSummaryByJob: [
        {
          jobId: jobA.job_id,
          pipeline_id: jobA.pipeline_id,
          parent_pipeline_id: jobA.parent_pipeline_id,
          modelType: 'unknown',
          shapSummary: { feature_one: 0.8, feature_two: 0.2 },
        },
      ],
      coverageInputs: [
        {
          jobId: jobA.job_id,
          label: shortRunId(jobA),
          task: 'classification',
          status: 'completed',
          hasArtifact: true,
        },
      ],
      handleDownload: noop,
      downloadingChart: null,
      doneChart: null,
    };
    renderWithClient(
      <ShapSummaryView {...shapSummaryProps} />,
    );

    expect(screen.getAllByText(shortRunId(jobA)).length).toBeGreaterThan(0);
    expect(screen.queryByText(jobA.job_id.slice(0, 8))).not.toBeInTheDocument();
  });

  it('PipelineDiffView shows metadata, supports swapping, and reverses diff direction', async () => {
    const { jobsApi } = await import('../../../../core/api/jobs');
    vi.mocked(jobsApi.getJob).mockResolvedValueOnce({
      job_id: jobA.job_id,
      pipeline_id: jobA.pipeline_id,
      parent_pipeline_id: jobA.parent_pipeline_id,
      dataset_name: jobA.dataset_name,
      model_type: jobA.model_type,
      created_at: jobA.created_at,
      node_id: 'node-a',
      job_type: 'training',
      status: 'succeeded',
      start_time: null,
      end_time: null,
      error: null,
      result: null,
      graph: {
        nodes: [{ id: 'step-1', position: { x: 0, y: 0 }, data: { label: 'prep', method: 'mean' } }],
        edges: [],
      },
    } as never);
    vi.mocked(jobsApi.getJob).mockResolvedValueOnce({
      job_id: jobB.job_id,
      pipeline_id: jobB.pipeline_id,
      parent_pipeline_id: jobB.parent_pipeline_id,
      dataset_name: jobB.dataset_name,
      model_type: jobB.model_type,
      created_at: jobB.created_at,
      node_id: 'node-b',
      job_type: 'training',
      status: 'succeeded',
      start_time: null,
      end_time: null,
      error: null,
      result: null,
      graph: {
        nodes: [{ id: 'step-1', position: { x: 0, y: 0 }, data: { label: 'prep', method: 'median' } }],
        edges: [],
      },
    } as never);

    renderWithClient(<PipelineDiffView jobs={[jobA, jobB] as never} />);

    await waitFor(() => {
      expect(screen.getByText('Baseline')).toBeInTheDocument();
      expect(screen.getByText('Candidate')).toBeInTheDocument();
      expect(screen.getByText(/Dataset Alpha/)).toBeInTheDocument();
      expect(screen.getByText(/Dataset Beta/)).toBeInTheDocument();
      expect(screen.getByText(/logistic_regression/)).toBeInTheDocument();
      expect(screen.getByText(/xgboost/)).toBeInTheDocument();
      expect(screen.getByText(new RegExp(new Date(jobA.created_at).toLocaleString()))).toBeInTheDocument();
      expect(screen.getByText(new RegExp(new Date(jobB.created_at).toLocaleString()))).toBeInTheDocument();
    });

    expect(screen.getByText(/method: "mean" → "median"/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /swap/i }));
    expect(screen.getByText(/method: "median" → "mean"/)).toBeInTheDocument();
    expect(screen.getAllByText('Baseline')[0]).toBeInTheDocument();
    expect(screen.getAllByText('Candidate')[0]).toBeInTheDocument();
    expect(screen.getByText(/Dataset Alpha/)).toBeInTheDocument();
    expect(screen.getByText(/Dataset Beta/)).toBeInTheDocument();
    expect(screen.getByText(shortRunId(jobA))).toBeInTheDocument();
    expect(screen.getByText(shortRunId(jobB))).toBeInTheDocument();
  });

  it('PipelineDiffView names the run when a snapshot is missing', async () => {
    const { jobsApi } = await import('../../../../core/api/jobs');
    vi.mocked(jobsApi.getJob).mockResolvedValueOnce({
      job_id: jobA.job_id,
      pipeline_id: jobA.pipeline_id,
      parent_pipeline_id: jobA.parent_pipeline_id,
      dataset_name: jobA.dataset_name,
      model_type: jobA.model_type,
      created_at: jobA.created_at,
      node_id: 'node-a',
      job_type: 'training',
      status: 'succeeded',
      start_time: null,
      end_time: null,
      error: null,
      result: null,
      graph: null,
    } as never);
    vi.mocked(jobsApi.getJob).mockResolvedValueOnce({
      job_id: jobB.job_id,
      pipeline_id: jobB.pipeline_id,
      parent_pipeline_id: jobB.parent_pipeline_id,
      dataset_name: jobB.dataset_name,
      model_type: jobB.model_type,
      created_at: jobB.created_at,
      node_id: 'node-b',
      job_type: 'training',
      status: 'succeeded',
      start_time: null,
      end_time: null,
      error: null,
      result: null,
      graph: {
        nodes: [{ id: 'step-1', position: { x: 0, y: 0 }, data: { label: 'prep', method: 'median' } }],
        edges: [],
      },
    } as never);

    renderWithClient(<PipelineDiffView jobs={[jobA, jobB] as never} />);

    await waitFor(() => {
      expect(screen.getByText(/has no saved pipeline snapshot/i)).toBeInTheDocument();
      expect(screen.getByText(new RegExp(shortRunId(jobA)))).toBeInTheDocument();
    });
  });
});
