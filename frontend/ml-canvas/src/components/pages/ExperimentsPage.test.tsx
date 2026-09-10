import type { ComponentProps } from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ConfirmProvider } from '../shared/ConfirmDialog';
import { jobsApi, type JobInfo } from '../../core/api/jobs';
import { useJobStore } from '../../core/store/useJobStore';
import { ExperimentsPage } from './ExperimentsPage';
import type { MetricsComparisonChart } from './ExperimentsPage/components/MetricsComparisonChart';

const captured = vi.hoisted(() => ({ chart: vi.fn() }));
vi.mock('./ExperimentsPage/components/MermaidDiagram', () => ({
  MermaidDiagram: ({ chart }: { chart: string }) => <pre>{chart}</pre>,
}));
vi.mock('../../core/api/jobs', async (importOriginal) => {
  const original = await importOriginal<typeof import('../../core/api/jobs')>();
  return { ...original, jobsApi: { ...original.jobsApi, getJobs: vi.fn() } };
});
vi.mock('../../core/api/client', () => ({ apiClient: { get: vi.fn().mockResolvedValue({ data: [{ id: 'a', name: 'Dataset A' }, { id: 'b', name: 'Dataset B' }] }) } }));
vi.mock('../../core/api/registry', () => ({ registryApi: { getAllNodes: vi.fn().mockResolvedValue([
  { id: 'forest', tags: ['classification'] }, { id: 'linear', tags: ['regression'] },
]) } }));
vi.mock('./ExperimentsPage/components/MetricsComparisonChart', async (importOriginal) => {
  const original = await importOriginal<typeof import('./ExperimentsPage/components/MetricsComparisonChart')>();
  return { MetricsComparisonChart: (props: ComponentProps<typeof MetricsComparisonChart>) => {
    captured.chart(props);
    return <original.MetricsComparisonChart {...props} />;
  } };
});

function run(id: string, dataset: string, metrics: Record<string, number>): JobInfo {
  /** Two independent runs reveal ordering and split fallback at the page boundary. */
  return { job_id: id, dataset_id: dataset, pipeline_id: id, node_id: 'model',
    status: 'completed', job_type: 'training', model_type: 'forest', error: null, result: null,
    created_at: '2026-01-01T00:00:00Z', start_time: null, end_time: null, metrics };
}

function renderExperiments() {
  /** Keep router, store, sidebar and tabs real while replacing only HTTP. */
  return render(<MemoryRouter><ConfirmProvider><ExperimentsPage /></ConfirmProvider></MemoryRouter>);
}

function chartProps(): ComponentProps<typeof MetricsComparisonChart> {
  /** Observe data at the chart boundary rather than inspecting extracted helpers. */
  return captured.chart.mock.lastCall![0] as ComponentProps<typeof MetricsComparisonChart>;
}

describe('ExperimentsPage comparison state', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useJobStore.setState({ jobs: [], isLoading: false, hasMore: false, skip: 0 });
    vi.mocked(jobsApi.getJobs).mockResolvedValue([
      run('first-run', 'a', { train_accuracy: 0, test_accuracy: 0.8, cv_accuracy: 0.7 }),
      run('second-run', 'b', { train_accuracy: 0.9, test_accuracy: 1 }),
    ]);
  });

  it('keeps store order, zero scores and split choices across comparison views', async () => {
    /** Reversing click order must not reorder bars or reset the lifted split state. */
    renderExperiments();
    fireEvent.click(await screen.findByText('second-run'));
    fireEvent.click(screen.getAllByText('first-run')[0]!);
    await waitFor(() => { expect(chartProps().metricsData).toHaveLength(2); });
    expect(chartProps().metricsData).toEqual([
      { name: 'first-run', train_accuracy: 0, test_accuracy: 0.8, cv_accuracy: 0.7 },
      { name: 'second-run', train_accuracy: 0.9, test_accuracy: 1 },
    ]);
    expect(chartProps().metricGroups.get('accuracy')).toEqual(['test_accuracy', 'train_accuracy']);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Train' }));
    fireEvent.click(screen.getByRole('button', { name: 'Detailed Metrics & Params' }));
    fireEvent.click(screen.getByRole('button', { name: 'Visual Comparison' }));
    expect(screen.getByRole('checkbox', { name: 'Train' })).not.toBeChecked();
    expect(chartProps().metricGroups.get('accuracy')).toEqual(['test_accuracy']);
    expect(screen.getByRole('checkbox', { name: 'Cross-Validation' })).not.toBeChecked();
  });

  it('retains hidden selections until the user clears them', async () => {
    /** Dataset filters must never silently drop runs from the comparison. */
    renderExperiments();
    fireEvent.click(await screen.findByText('first-run'));
    fireEvent.click(screen.getAllByText('second-run')[0]!);
    fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'a' } });
    expect(screen.getByRole('status')).toHaveTextContent('1 of 2 selected runs visible');
    expect(chartProps().metricsData).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Clear hidden' }));
    expect(screen.queryByText(/hidden by the current filters/)).not.toBeInTheDocument();
    expect(chartProps().metricsData).toEqual([{ name: 'first-run', train_accuracy: 0, test_accuracy: 0.8, cv_accuracy: 0.7 }]);
  });

  it('opens selected-run importance, SHAP and pipeline artifacts through real tabs', async () => {
    /** Artifact gates must deliver the selected run data to each existing view. */
    const job = run('artifact', 'a', {});
    delete job.metrics;
    job.result = { metrics: {
      test_accuracy: 0.8, pipeline_diagram: 'graph TD; A-->B;',
      feature_importances: { age: 0, income: 0.75 },
      shap_explanation: { feature_names: ['age'], mean_abs_importance: { age: 0.2 }, samples: [] },
    } };
    vi.mocked(jobsApi.getJobs).mockResolvedValue([job]);
    renderExperiments();
    fireEvent.click(await screen.findByText('artifact'));
    fireEvent.click(screen.getByRole('button', { name: 'Feature Importance' }));
    expect(screen.getByText('Feature Importance Comparison')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /view data table/i }));
    const region = screen.getByRole('region', { name: /feature importance comparison data/i });
    expect(within(region).getByText('age')).toBeInTheDocument();
    expect(within(region).getByText('income')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'SHAP Explainability' }));
    expect(screen.getByRole('button', { name: 'Beeswarm' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Pipeline Diagram' }));
    expect(screen.getByTestId('copy-mermaid-button')).toBeInTheDocument();
    expect(screen.queryByText('Feature Importance Comparison')).not.toBeInTheDocument();
  });
});
