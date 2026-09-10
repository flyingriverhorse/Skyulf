import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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
  { id: 'kmeans', name: 'K-Means', category: 'model', description: '', params: {}, tags: ['clustering'] },
  { id: 'multinomial_nb', name: 'Naive Bayes', category: 'model', description: '', params: {}, tags: ['text'] },
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
  it('keeps facet options in first-seen order and Clear all preserves the search', async () => {
    // Sorting facets or resetting search with select filters would change the existing controls.
    useJobStore.setState({ jobs: [
      makeJob('first', { model_type: 'random_forest', status: 'running' }),
      makeJob('second', { status: 'failed' }),
      makeJob('third', { model_type: 'random_forest', status: 'completed' }),
    ] });
    await renderDrawer();
    fireEvent.click(screen.getByRole('button', { name: 'Filters' }));
    const [status, model] = screen.getAllByRole('combobox');
    expect(within(status!).getAllByRole('option').map(option => option.textContent))
      .toEqual(['All', 'Running', 'Failed', 'Completed']);
    expect(within(model!).getAllByRole('option').map(option => option.textContent))
      .toEqual(['All', 'random forest', 'logistic regression']);
    fireEvent.change(screen.getByPlaceholderText(/Search by job ID/), { target: { value: 'first' } });
    fireEvent.change(status!, { target: { value: 'failed' } });
    fireEvent.change(model!, { target: { value: 'logistic_regression' } });
    expect(screen.getByRole('button', { name: 'Filters 2' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }));
    expect(screen.getByPlaceholderText(/Search by job ID/)).toHaveValue('first');
    expect(screen.getByText('first')).toBeInTheDocument();
    expect(screen.queryByText('second')).not.toBeInTheDocument();
  });

  it('selects task tabs and resets ensemble subfilters only when leaving Ensemble', async () => {
    // Ensemble scope survives a close but returning from another task starts at All.
    useJobStore.setState({ jobs: [
      makeJob('classification-row'), makeJob('regression-row', { model_type: 'linear_regression' }),
      makeJob('text-row', { model_type: 'multinomial_nb' }), makeJob('segment-row', { model_type: 'kmeans' }),
      makeJob('ensemble-class', { model_type: 'voting_classifier' }),
      makeJob('ensemble-reg', { model_type: 'stacking_regressor' }),
    ] });
    await renderDrawer();
    for (const [tab, row] of [
      ['Regression', 'regression-row'], ['Text Classification', 'text-row'],
      ['Segmentation', 'segment-row'], ['Classification', 'classification-row'],
    ] as const) {
      fireEvent.click(screen.getByRole('button', { name: tab }));
      expect(screen.getByText(row)).toBeInTheDocument();
      expect(screen.queryByText('ensemble-class')).not.toBeInTheDocument();
    }
    fireEvent.click(screen.getByRole('button', { name: 'Ensemble' }));
    fireEvent.click(screen.getAllByRole('button', { name: 'Regression' })[1]!);
    expect(screen.getByText('ensemble-reg')).toBeInTheDocument();
    expect(screen.queryByText('ensemble-class')).not.toBeInTheDocument();
    await act(async () => { useJobStore.setState({ isDrawerOpen: false }); });
    await act(async () => { useJobStore.setState({ isDrawerOpen: true }); });
    expect(screen.queryByText('ensemble-class')).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('button', { name: 'Classification' })[1]!);
    expect(screen.getByText('ensemble-class')).toBeInTheDocument();
    expect(screen.queryByText('ensemble-reg')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Segmentation' }));
    fireEvent.click(screen.getByRole('button', { name: 'Ensemble' }));
    expect(screen.getByText('ensemble-reg')).toBeInTheDocument();
    expect(screen.getByText('ensemble-class')).toBeInTheDocument();
  });

  it('preserves filters through details and reopen while restoring focus on Escape', async () => {
    // Extracted subviews must not take ownership of state that currently outlives them.
    useJobStore.setState({ jobs: [makeJob('chosen'), makeJob('other')] });
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    await renderDrawer();
    await waitFor(() => { expect(screen.getByRole('dialog')).toHaveFocus(); });
    fireEvent.click(screen.getByRole('button', { name: 'Filters' }));
    fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'completed' } });
    fireEvent.change(screen.getAllByRole('combobox')[1]!, { target: { value: 'logistic_regression' } });
    fireEvent.change(screen.getByPlaceholderText(/Search by job ID/), { target: { value: 'chosen' } });
    fireEvent.click(screen.getByText('chosen'));
    fireEvent.click(screen.getByRole('button', { name: 'Back to history' }));
    expect(screen.getByPlaceholderText(/Search by job ID/)).toHaveValue('chosen');
    expect(screen.getAllByRole('combobox')[0]).toHaveValue('completed');
    expect(screen.getAllByRole('combobox')[1]).toHaveValue('logistic_regression');
    fireEvent.click(screen.getByText('chosen'));
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
    await act(async () => { useJobStore.setState({ isDrawerOpen: true }); });
    expect(screen.queryByRole('heading', { name: 'Details for chosen' })).not.toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Search by job ID/)).toHaveValue('chosen');
    expect(screen.getAllByRole('combobox')[0]).toHaveValue('completed');
    expect(screen.getAllByRole('combobox')[1]).toHaveValue('logistic_regression');
    trigger.remove();
    expect(registryApi.getAllNodes).toHaveBeenCalledTimes(1);
  });

  it('retains inspected receipt order and cached progress precedence over history', async () => {
    // Snapshot order follows the submission receipt, and stale history must not override it.
    useJobStore.setState({
      jobs: [makeJob('first'), makeJob('second')],
      runJobs: { first: makeJob('first', { status: 'running' }) },
      inspectedRun: { label: 'Receipt', jobIds: ['second', 'missing', 'first'] },
      activeParallelRun: { jobIds: ['first', 'second', 'missing'], startedAt: '2026-09-07T12:00:00Z' },
    });
    await renderDrawer();
    const first = screen.getByText('first');
    const second = screen.getByText('second');
    expect(second.compareDocumentPosition(first) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByText('Parallel Run: 1/3 branches complete')).toBeInTheDocument();
    expect(screen.getByText('33%')).toBeInTheDocument();
  });

  it('caps automatic pages at five, resetting on tab change and reopen', async () => {
    // Rare tasks must not fetch history indefinitely, but a fresh view gets a new allowance.
    useJobStore.setState({ hasMore: true });
    vi.mocked(jobsApi.getJobs).mockImplementation(async (_limit, skip) =>
      Array.from({ length: 50 }, (_, index) => makeJob(`page-${skip}-${index}`, { model_type: 'linear_regression' })));
    await renderDrawer();
    await waitFor(() => { expect(jobsApi.getJobs).toHaveBeenCalledTimes(5); });
    await act(async () => { useJobStore.setState(state => ({ jobs: [...state.jobs] })); });
    expect(jobsApi.getJobs).toHaveBeenCalledTimes(5);
    fireEvent.click(screen.getByRole('button', { name: 'Segmentation' }));
    await waitFor(() => { expect(jobsApi.getJobs).toHaveBeenCalledTimes(10); });
    await act(async () => { useJobStore.setState({ isDrawerOpen: false }); });
    await act(async () => { useJobStore.setState({ isDrawerOpen: true }); });
    await waitFor(() => { expect(jobsApi.getJobs).toHaveBeenCalledTimes(15); });
    expect(jobsApi.getJobs).toHaveBeenLastCalledWith(50, 750);
  });

  it('does not auto-page while closed, loading, exhausted, or inspecting a run', async () => {
    // Hook extraction must preserve every request guard even while the component returns null.
    useJobStore.setState({ isDrawerOpen: false, hasMore: true });
    await renderDrawer();
    expect(registryApi.getAllNodes).toHaveBeenCalledTimes(1);
    expect(jobsApi.getJobs).not.toHaveBeenCalled();
    await act(async () => { useJobStore.setState({ isDrawerOpen: true, isLoading: true }); });
    expect(jobsApi.getJobs).not.toHaveBeenCalled();
    await act(async () => { useJobStore.setState({ isLoading: false, hasMore: false }); });
    expect(jobsApi.getJobs).not.toHaveBeenCalled();
    await act(async () => { useJobStore.setState({ hasMore: true, inspectedRun: { label: 'Run', jobIds: ['missing'] } }); });
    expect(jobsApi.getJobs).not.toHaveBeenCalled();
  });

  it('stops automatic pagination once five tab jobs exist, regardless of search filtering', async () => {
    // The pagination threshold measures task jobs, not the currently visible search result.
    useJobStore.setState({ hasMore: true, jobs: Array.from({ length: 5 }, (_, index) => makeJob(`row-${index}`)) });
    await renderDrawer();
    fireEvent.change(screen.getByPlaceholderText(/Search by job ID/), { target: { value: 'unmatched' } });
    await act(async () => { useJobStore.setState(state => ({ jobs: [...state.jobs] })); });
    expect(screen.getByText('No jobs match the current filters.')).toBeInTheDocument();
    expect(jobsApi.getJobs).not.toHaveBeenCalled();
  });

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
