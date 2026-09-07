import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { JobsDrawer } from './JobsDrawer';
import { useJobStore } from '../../core/store/useJobStore';
import { jobsApi, type JobInfo } from '../../core/api/jobs';
import { registryApi, type RegistryItem } from '../../core/api/registry';

vi.mock('../../core/realtime/jobEventsSocket', () => ({
  jobEventsSocket: { subscribe: vi.fn(() => () => {}), onStatus: vi.fn(() => () => {}) },
}));

// Details own polling, result charts, and inspectors; their separate suite covers those boundaries.
vi.mock('./jobs/JobDetailsView', () => ({
  JobDetailsView: ({ job, onBack, onClose }: {
    job: JobInfo; onBack: () => void; onClose: () => void;
  }) => <div>
    <h3>Details for {job.job_id}</h3>
    <button onClick={onBack}>Back to history</button>
    <button onClick={onClose}>Close details</button>
  </div>,
}));

/** Supply realistic records while allowing each case to select its lifecycle state. */
const makeJob = (id: string, overrides: Partial<JobInfo> = {}): JobInfo => ({
  job_id: id, pipeline_id: 'pipeline', node_id: 'trainer', job_type: 'training',
  status: 'completed', start_time: null, end_time: null, error: null, result: null,
  created_at: '2026-09-07T12:00:00Z', model_type: 'logistic_regression', ...overrides,
});

const registry: RegistryItem[] = [
  { id: 'random_forest', name: 'Random Forest', category: 'model', description: '', params: {}, tags: ['classification'] },
  { id: 'linear_regression', name: 'Linear Regression', category: 'model', description: '', params: {}, tags: ['regression'] },
];

/** Settle the one-time registry request before interacting with task filters. */
async function renderDrawer(): Promise<void> {
  await act(async () => { render(<JobsDrawer />); });
}

beforeEach(() => {
  vi.restoreAllMocks();
  useJobStore.getState().stopPolling();
  useJobStore.setState({
    jobs: [], runJobs: {}, nodeSubmissions: {}, inspectedRun: null, activeParallelRun: null,
    pendingJobActions: {}, isDrawerOpen: true, activeTab: 'classification', isLoading: false,
    hasMore: false, skip: 0,
  });
  vi.spyOn(registryApi, 'getAllNodes').mockResolvedValue(registry);
  vi.spyOn(jobsApi, 'getJobs').mockResolvedValue([]);
});

afterEach(() => {
  useJobStore.getState().stopPolling();
  vi.restoreAllMocks();
});

describe('JobsDrawer submitted runs', () => {
  it('resolves scoped jobs from snapshots and history even when existing filters exclude them', async () => {
    // View jobs must show the receipt's exact jobs across task types, regardless of old search filters.
    useJobStore.setState({ jobs: [makeJob('cached'), makeJob('history'), makeJob('unrelated')] });
    await renderDrawer();
    fireEvent.change(screen.getByPlaceholderText(/Search by job ID/), { target: { value: 'no match' } });
    expect(screen.getByText('No jobs match the current filters.')).toBeInTheDocument();

    await act(async () => {
      const cached = makeJob('cached', { dataset_name: 'Fresh snapshot', status: 'running' });
      delete cached.model_type;
      useJobStore.setState({
        hasMore: true,
        runJobs: { cached },
      });
      useJobStore.getState().setInspectedRun({ label: 'Selected training', jobIds: ['cached', 'history', 'missing'] });
    });

    expect(screen.getByText(/Selected training.*3 submitted jobs/)).toBeInTheDocument();
    expect(screen.getByText('Fresh snapshot')).toBeInTheDocument();
    expect(screen.getByText('Unknown Model')).toBeInTheDocument();
    expect(screen.getAllByText('cached')).toHaveLength(1);
    expect(screen.getByText('history')).toBeInTheDocument();
    expect(screen.queryByText('unrelated')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/Search by job ID/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load More History' })).not.toBeInTheDocument();
    expect(jobsApi.getJobs).not.toHaveBeenCalled();
  });

  it('returns to the previous history filters when the receipt scope is cleared', async () => {
    // Show all jobs must restore normal history instead of retaining the receipt-only list.
    useJobStore.setState({ jobs: [makeJob('chosen'), makeJob('other')] });
    await renderDrawer();
    fireEvent.change(screen.getByPlaceholderText(/Search by job ID/), { target: { value: 'chosen' } });
    await act(async () => {
      useJobStore.getState().setInspectedRun({ label: 'Run', jobIds: ['other'] });
    });
    expect(screen.getByText('other')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Show all jobs' }));

    expect(screen.getByPlaceholderText(/Search by job ID/)).toHaveValue('chosen');
    expect(screen.getByText('chosen')).toBeInTheDocument();
    expect(screen.queryByText('other')).not.toBeInTheDocument();
    expect(useJobStore.getState().inspectedRun).toBeNull();
  });

  it('shows a waiting receipt until refresh retrieves the submitted job', async () => {
    // A newly submitted job may not exist on the first page yet and must be recoverable with Refresh.
    useJobStore.setState({ inspectedRun: { label: 'New run', jobIds: ['new-job'] }, hasMore: true });
    vi.spyOn(jobsApi, 'getJob').mockResolvedValue(makeJob('new-job', { status: 'pending' }));
    await renderDrawer();
    expect(screen.getByText(/Waiting for the submitted jobs to appear/)).toBeInTheDocument();
    expect(jobsApi.getJobs).not.toHaveBeenCalled();

    await act(async () => { fireEvent.click(screen.getByTitle('Refresh')); });

    expect(screen.getByText('new-job')).toBeInTheDocument();
    expect(screen.queryByText(/Waiting for the submitted jobs/)).not.toBeInTheDocument();
    expect(jobsApi.getJob).toHaveBeenCalledWith('new-job');
  });

  it('counts terminal cached and history branches while missing branches remain unfinished', async () => {
    // All terminal statuses count, but absent records cannot report an entire run as complete.
    useJobStore.setState({
      jobs: [makeJob('success', { status: 'succeeded' }), makeJob('cancelled', { status: 'cancelled' })],
      runJobs: { cached: makeJob('cached'), failed: makeJob('failed', { status: 'failed' }) },
      activeParallelRun: { jobIds: ['cached', 'success', 'failed', 'cancelled', 'missing'], startedAt: '2026-09-07T12:00:00Z' },
    });
    await renderDrawer();
    expect(screen.getByText('Parallel Run: 4/5 branches complete')).toBeInTheDocument();
    expect(screen.getByText('80%')).toBeInTheDocument();

    await act(async () => {
      useJobStore.setState(state => ({ runJobs: { ...state.runJobs, missing: makeJob('missing') } }));
    });

    expect(screen.getByText('All branches complete!')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('shows progress only when the inspected receipt overlaps the active parallel run', async () => {
    // Inspecting an older notification must not display another run's progress as its own.
    useJobStore.setState({
      jobs: [makeJob('active', { status: 'running' }), makeJob('older')],
      activeParallelRun: { jobIds: ['active'], startedAt: '2026-09-07T12:00:00Z' },
      inspectedRun: { label: 'Older run', jobIds: ['older'] },
    });
    await renderDrawer();
    expect(screen.queryByText(/Parallel Run:/)).not.toBeInTheDocument();

    await act(async () => {
      useJobStore.getState().setInspectedRun({ label: 'Active run', jobIds: ['active'] });
    });

    expect(screen.getByText('Parallel Run: 0/1 branches complete')).toBeInTheDocument();
    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  it('resets details when the inspected receipt changes and closes its scope with the drawer', async () => {
    // A second View jobs action must not leave the user inside the previous job's details.
    useJobStore.setState({ jobs: [makeJob('first'), makeJob('second')] });
    await renderDrawer();
    fireEvent.click(screen.getByText('first'));
    expect(screen.getByRole('heading', { name: 'Details for first' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Back to history' }));
    fireEvent.click(screen.getByText('first'));

    await act(async () => {
      useJobStore.getState().setInspectedRun({ label: 'Second run', jobIds: ['second'] });
    });
    expect(screen.queryByRole('heading', { name: 'Details for first' })).not.toBeInTheDocument();
    expect(screen.getByText('second')).toBeInTheDocument();
    fireEvent.click(screen.getByText('second'));
    fireEvent.click(screen.getByRole('button', { name: 'Close details' }));

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(useJobStore.getState().inspectedRun).toBeNull();
  });
});

describe('JobsDrawer normal history', () => {
  it.each(['pending', 'queued', 'running', 'completed', 'succeeded', 'failed', 'cancelled'] as const)(
    'filters %s jobs independently of completed history', async status => {
      // Every backend lifecycle status must remain selectable, including pending and succeeded aliases.
      useJobStore.setState({ jobs: [
        makeJob('selected-status', { status }),
        makeJob('different-status', { status: status === 'failed' ? 'running' : 'failed' }),
      ] });
      await renderDrawer();
      fireEvent.click(screen.getByRole('button', { name: 'Filters' }));
      fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: status } });

      expect(screen.getByText('selected-status')).toBeInTheDocument();
      expect(screen.queryByText('different-status')).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Clear all' })).toBeInTheDocument();
    },
  );

  it('combines model filters with case-insensitive dataset, model, and job ID searches', async () => {
    // Search must support legacy dataset IDs and clear stale filters without losing the history list.
    useJobStore.setState({ jobs: [
      makeJob('named-job', { dataset_name: 'Sales Data' }),
      makeJob('legacy-job', { dataset_id: 'legacy-dataset', model_type: 'random_forest' }),
      makeJob('ID-ONLY'),
    ] });
    await renderDrawer();
    const search = screen.getByPlaceholderText(/Search by job ID/);
    fireEvent.change(search, { target: { value: 'SALES' } });
    expect(screen.getByText('named-job')).toBeInTheDocument();
    expect(screen.queryByText('legacy-job')).not.toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'LEGACY-DATASET' } });
    expect(screen.getByText('legacy-job')).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'id-only' } });
    expect(screen.getByText('ID-ONLY')).toBeInTheDocument();
    fireEvent.change(search, { target: { value: 'random_forest' } });
    expect(screen.getByText('legacy-job')).toBeInTheDocument();
    fireEvent.change(search, { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Filters' }));
    fireEvent.change(screen.getAllByRole('combobox')[1]!, { target: { value: 'random_forest' } });
    expect(screen.queryByText('named-job')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));

    expect(screen.getByText('named-job')).toBeInTheDocument();
    expect(screen.getByText('legacy-job')).toBeInTheDocument();
    expect(screen.getByText('ID-ONLY')).toBeInTheDocument();
  });

  it.each([
    ['Classification', 'No classification jobs found.'],
    ['Regression', 'No regression jobs found.'],
    ['Text Classification', 'No text classification jobs found.'],
    ['Segmentation', 'No segmentation jobs found.'],
    ['Ensemble', 'No ensemble jobs found.'],
  ])('labels an empty %s tab by task', async (tab, message) => {
    // Empty history must tell users which task is selected instead of implying all jobs disappeared.
    await renderDrawer();
    fireEvent.click(screen.getByRole('button', { name: tab }));

    expect(screen.getByText(message)).toBeInTheDocument();
  });

  it('keeps normal history pagination available and disables it during the request', async () => {
    // Users need manual pagination after enough jobs are visible, without duplicate page requests.
    useJobStore.setState({ jobs: Array.from({ length: 5 }, (_, index) => makeJob(`row-${index}`)), hasMore: true });
    let resolvePage: ((jobs: JobInfo[]) => void) | undefined;
    vi.mocked(jobsApi.getJobs).mockImplementation(() => new Promise(resolve => { resolvePage = resolve; }));
    await renderDrawer();
    fireEvent.click(screen.getByRole('button', { name: 'Load More History' }));
    expect(screen.getByRole('button', { name: 'Load More History' })).toBeDisabled();
    await act(async () => { resolvePage?.([makeJob('older-page')]); });

    expect(jobsApi.getJobs).toHaveBeenCalledWith(50, 50);
    expect(screen.getByText('older-page')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load More History' })).not.toBeInTheDocument();
  });

  it('dismisses normal history with the accessible close control', async () => {
    // The dialog close action must update the shared store so navbar state stays in sync.
    await renderDrawer();
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Close job history' }));

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(useJobStore.getState().isDrawerOpen).toBe(false);
  });
});
