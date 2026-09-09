import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ComponentProps } from 'react';

import type { Dataset } from '../core/types/api';
import type { EDAReport } from '../core/api/eda';
import { EDAService } from '../core/api/eda';
import { DatasetService } from '../core/api/datasets';
import { useEDAStore } from '../core/store/useEDAStore';
import { EDAPage } from './EDAPage';
import { edaKeys } from '../core/hooks/useEdaJobs';

const observed = vi.hoisted(() => ({
  dashboard: vi.fn(), decomposition: vi.fn(), history: vi.fn(),
}));

beforeEach(() => vi.clearAllMocks());

const duplicateNamedDatasets: Dataset[] = [
  {
    id: '101',
    source_id: 'source-a',
    name: 'Shared dataset',
    type: 'file',
    created_at: '2026-08-07T08:00:00.000Z',
    rows: 1200,
    columns: 12,
    format: 'csv',
  },
  {
    id: '202',
    source_id: 'source-b',
    name: 'Shared dataset',
    type: 'file',
    created_at: '2026-08-07T09:00:00.000Z',
    rows: 900,
    columns: 9,
    format: 'csv',
  },
];

vi.mock('../core/api/datasets', async () => {
  const actual = await vi.importActual<typeof import('../core/api/datasets')>('../core/api/datasets');
  return {
    ...actual,
    DatasetService: {
      ...actual.DatasetService,
      getUsable: vi.fn(),
    },
  };
});

describe('EDAPage PII review', () => {
  const profileReport: EDAReport = {
    id: 10,
    status: 'COMPLETED',
    profile_data: {
      row_count: 2,
      column_count: 2,
      columns: {
        contact: {
          name: 'contact', dtype: 'Categorical', missing_count: 0, missing_percentage: 0,
          categorical_stats: {
            unique_count: 2, rare_labels_count: 0,
            top_k: [{ value: 'private@example.com', count: 1 }],
          },
        },
        age: { name: 'age', dtype: 'Numeric', missing_count: 1, missing_percentage: 50 },
      },
      sample_data: [{ contact: 'private@example.com', phone: '+1-202-555-0123' }],
      alerts: [
        { type: 'PII', column: 'contact', severity: 'error', message: 'Found private@example.com' },
        { type: 'PII', column: null, severity: '+1-202-555-0123', message: 'Untrusted private@example.com' },
        { type: 'High Null', column: 'age', severity: 'warning', message: 'Missing values need review' },
        { column: 'legacy', severity: 'info', message: 'PII Email/Phone mentioned in legacy message' },
      ],
    },
  };

  function renderProfilePage() {
    vi.mocked(DatasetService.getUsable).mockResolvedValue(duplicateNamedDatasets);
    vi.mocked(EDAService.getHistory).mockResolvedValue([]);
    useEDAStore.getState().resetForDataset();
    useEDAStore.setState({ selectedDataset: 101, taskType: '' });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/eda?dataset_id=101']}>
          <EDAPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  }

  it('reviews structured PII alerts without exposing messages, samples, or filter values', async () => {
    // Accidental rendering of alert text, raw statistics, or sidebar filters can disclose personal data.
    vi.mocked(EDAService.getLatestReport).mockResolvedValue(profileReport);
    renderProfilePage();
    const reviewButton = await screen.findByRole('button', { name: 'PII Review' });
    act(() => {
      useEDAStore.getState().addFilterDraft({ column: 'contact', operator: '==', value: 'filter@example.com' });
      useEDAStore.getState().toggleExclude('contact', true);
    });
    fireEvent.click(reviewButton);

    const panel = await screen.findByRole('region', { name: 'PII review' });
    const rows = within(panel).getAllByRole('row');
    expect(rows).toHaveLength(3);
    expect(within(panel).getByRole('rowheader', { name: 'contact' })).toBeInTheDocument();
    expect(within(panel).getAllByText('Email / phone')).toHaveLength(2);
    expect(within(panel).getByText('Error')).toBeInTheDocument();
    expect(within(panel).getByText('Unspecified')).toBeInTheDocument();
    expect(within(panel).queryByText('age')).not.toBeInTheDocument();
    expect(within(panel).queryByText('legacy')).not.toBeInTheDocument();
    expect(document.body.innerHTML).not.toMatch(/private@example.com|filter@example.com|\+1-202-555-0123/);
    expect(profileReport.profile_data?.alerts).toHaveLength(4);
  });

  it('loads the selected dataset findings and clears them when switching to a clean profile', async () => {
    // A dataset switch must not leave the previous dataset's PII findings visible.
    vi.mocked(EDAService.getLatestReport).mockImplementation(async (datasetId) => (
      datasetId === 101 ? profileReport : {
        id: 20, status: 'COMPLETED',
        profile_data: { row_count: 1, column_count: 0, columns: {}, alerts: [] },
      }
    ));
    renderProfilePage();
    fireEvent.click(await screen.findByRole('button', { name: 'PII Review' }));
    expect(await screen.findByRole('rowheader', { name: 'contact' })).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Dataset' }), { target: { value: '202' } });
    fireEvent.click(await screen.findByRole('button', { name: 'PII Review' }));
    expect(await screen.findByText('No PII findings recorded')).toBeInTheDocument();
    expect(screen.queryByRole('rowheader', { name: 'contact' })).not.toBeInTheDocument();
  });
});

vi.mock('../core/api/eda', async () => {
  const actual = await vi.importActual<typeof import('../core/api/eda')>('../core/api/eda');
  return {
    ...actual,
    EDAService: {
      ...actual.EDAService,
      analyze: vi.fn(),
      getLatestReport: vi.fn(),
      getHistory: vi.fn(),
      getReport: vi.fn(),
    },
  };
});

vi.mock('../components/eda/JobsHistoryModal', () => ({
  JobsHistoryModal: (props: ComponentProps<typeof import('../components/eda/JobsHistoryModal').JobsHistoryModal>) => {
    observed.history(props);
    return null;
  },
}));

vi.mock('../components/shared', () => ({
  LoadingState: ({ message }: { message?: string }) => <div>{message}</div>,
  ErrorState: ({ error, onRetry }: { error: string; onRetry: () => void }) => <div>{error}<button onClick={onRetry}>Retry loading</button></div>,
}));

vi.mock('../core/utils/chartUtils', () => ({
  downloadChart: vi.fn(),
  getTooltipContentStyle: vi.fn(() => ({})),
}));

vi.mock('../components/eda/tabs/DashboardTab', () => ({ DashboardTab: (props: unknown) => {
  observed.dashboard(props);
  return <input aria-label="Dashboard local state" defaultValue="" />;
} }));
vi.mock('../components/eda/tabs/InsightsTab', () => ({ InsightsTab: () => null }));
vi.mock('../components/eda/tabs/PCATab', () => ({ PCATab: () => null }));
vi.mock('../components/eda/tabs/GeospatialTab', () => ({ GeospatialTab: () => null }));
vi.mock('../components/eda/tabs/TargetAnalysisTab', () => ({ TargetAnalysisTab: () => null }));
vi.mock('../components/eda/tabs/TimeSeriesTab', () => ({ TimeSeriesTab: () => null }));
vi.mock('../components/eda/tabs/VariablesTab', () => ({ VariablesTab: () => null }));
vi.mock('../components/eda/tabs/BivariateTab', () => ({ BivariateTab: () => null }));
vi.mock('../components/eda/tabs/OutliersTab', () => ({ OutliersTab: () => null }));
vi.mock('../components/eda/tabs/CorrelationsTab', () => ({ CorrelationsTab: () => null }));
vi.mock('../components/eda/tabs/SampleDataTab', () => ({ SampleDataTab: () => null }));
vi.mock('../components/eda/tabs/CausalTab', () => ({ CausalTab: () => null }));
vi.mock('../components/eda/tabs/RuleDiscoveryTab', () => ({ RuleDiscoveryTab: () => null }));
vi.mock('../components/eda/tabs/DecompositionTab', () => ({ DecompositionTab: (props: unknown) => {
  observed.decomposition(props);
  return null;
} }));

describe('EDAPage analysis lifecycle', () => {
  const completedReport: EDAReport = {
    id: 30, status: 'COMPLETED', profile_data: {
      row_count: 2, column_count: 2,
      columns: {
        age: { name: 'age', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
        income: { name: 'income', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
      },
    },
  };

  beforeEach(() => {
    useEDAStore.getState().resetForDataset();
    useEDAStore.setState({ selectedDataset: 101, taskType: '' });
    vi.mocked(DatasetService.getUsable).mockResolvedValue(duplicateNamedDatasets);
    vi.mocked(EDAService.getLatestReport).mockResolvedValue(completedReport);
    vi.mocked(EDAService.getHistory).mockResolvedValue([]);
    vi.mocked(EDAService.analyze).mockResolvedValue({});
    observed.dashboard.mockClear();
    observed.decomposition.mockClear();
    observed.history.mockClear();
  });

  function NavigationProbe() {
    const location = useLocation();
    const navigate = useNavigate();
    return <><output aria-label="Current URL">{location.search}</output><button onClick={() => navigate(-1)}>Back dataset</button></>;
  }

  function renderPage(initialEntry = '/eda?dataset_id=101&keep=yes') {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
    const view = render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[initialEntry]}>
      <EDAPage /><NavigationProbe />
    </MemoryRouter></QueryClientProvider>);
    return { ...view, client };
  }

  it('treats a missing report as setup and submits optional target and task without losing input focus', async () => {
    // A 404 is an empty analysis slot, and editing setup must not remount its input.
    vi.mocked(EDAService.getLatestReport).mockRejectedValue({ response: { status: 404 } });
    renderPage();
    const target = await screen.findByRole('textbox', { name: 'Target Column (Optional)' });
    target.focus();
    fireEvent.change(target, { target: { value: 'age' } });
    fireEvent.change(screen.getAllByRole('combobox', { name: 'Task Type' })[1]!, { target: { value: 'Classification' } });
    expect(screen.getByRole('textbox', { name: 'Target Column (Optional)' })).toBe(target);
    expect(target).toHaveFocus();
    fireEvent.click(screen.getByRole('button', { name: 'Run Analysis' }));
    await waitFor(() => expect(EDAService.analyze).toHaveBeenCalledWith(101, 'age', [], [], 'Classification'));
  });

  it('keeps non-404 loading failures distinct and retries the report request', async () => {
    // A server failure must not invite a new analysis as though no report exists.
    vi.mocked(EDAService.getLatestReport).mockRejectedValueOnce({ response: { status: 500 } });
    renderPage();
    expect(await screen.findByText('Failed to load report')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry loading' }));
    expect(await screen.findByRole('button', { name: 'PII Review' })).toBeInTheDocument();
  });

  it.each([
    [{ status: 'PENDING' }, 'Analysis in progress...'],
    [{ status: 'FAILED', error_message: 'worker stopped' }, 'Analysis Failed'],
    [{ status: 'COMPLETED' }, 'No profile data'],
  ])('keeps report status rendering for %j', async (report, message) => {
    // Pending, failed and empty completed reports have different recovery paths.
    vi.mocked(EDAService.getLatestReport).mockResolvedValue(report);
    renderPage();
    expect(await screen.findByText(message)).toBeInTheDocument();
  });

  it('retries rejected submissions with the applied payload and retains pending draft edits', async () => {
    // Retry must reuse the failed payload even if unsubmitted drafts changed afterward.
    vi.mocked(EDAService.analyze).mockRejectedValueOnce(new Error('queue unavailable'));
    renderPage();
    await screen.findByRole('button', { name: 'PII Review' });
    const applied = [{ column: 'age', operator: '>=' as const, value: 0 }];
    act(() => {
      useEDAStore.getState().setFiltersApplied(applied);
      useEDAStore.getState().setFiltersDraft([{ column: 'income', operator: '>' as const, value: 10 }]);
      useEDAStore.getState().setExcludedApplied(['income']);
    });
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('queue unavailable');
    act(() => useEDAStore.getState().setExcludedApplied([]));
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(EDAService.analyze).toHaveBeenLastCalledWith(101, undefined, ['income'], applied, undefined));
    expect(useEDAStore.getState().filtersDraft).toEqual([{ column: 'income', operator: '>', value: 10 }]);
  });

  it('filters draft exclusions for display without changing the cache or decomposition applied inputs', async () => {
    // Chart filtering is immediate, while decomposition receives all columns and only applied filters.
    const { client } = renderPage();
    const local = await screen.findByRole('textbox', { name: 'Dashboard local state' });
    fireEvent.change(local, { target: { value: 'retained' } });
    const applied = [{ column: 'age', operator: '==' as const, value: 0 }];
    act(() => {
      useEDAStore.getState().toggleExclude('income', true);
      useEDAStore.getState().setFiltersApplied(applied);
    });
    expect(observed.dashboard).toHaveBeenLastCalledWith(expect.objectContaining({ profile: expect.objectContaining({ columns: { age: completedReport.profile_data!.columns.age } }) }));
    expect(client.getQueryData<EDAReport>(edaKeys.report(101))?.profile_data?.columns).toHaveProperty('income');
    expect(screen.getByRole('textbox', { name: 'Dashboard local state' })).toBe(local);
    expect(local).toHaveValue('retained');
    fireEvent.click(screen.getByRole('button', { name: 'Decomposition' }));
    expect(observed.decomposition).toHaveBeenLastCalledWith({ datasetId: 101, columns: ['age', 'income'], initialFilters: applied });
  });

  it('resets dataset-specific state while URL navigation preserves unrelated parameters', async () => {
    // Back/forward and explicit deep links must drive the store even after previous page visits.
    useEDAStore.setState({ selectedDataset: 202 });
    renderPage();
    await screen.findByRole('button', { name: 'PII Review' });
    expect(useEDAStore.getState().selectedDataset).toBe(101);
    act(() => {
      useEDAStore.getState().setTargetCol('age');
      useEDAStore.getState().setTaskType('Regression');
      useEDAStore.getState().setScatter({ x: 'age', is3D: true });
    });
    fireEvent.change(screen.getByRole('combobox', { name: 'Dataset' }), { target: { value: '202' } });
    await waitFor(() => expect(useEDAStore.getState().selectedDataset).toBe(202));
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('dataset_id=202&keep=yes');
    expect(useEDAStore.getState()).toMatchObject({ targetCol: '', taskType: 'Regression', scatter: { x: '', is3D: false } });
    fireEvent.click(screen.getByRole('button', { name: 'Back dataset' }));
    await waitFor(() => expect(useEDAStore.getState().selectedDataset).toBe(101));
  });

  it('keeps a newer dataset displayed when an older report request resolves late', async () => {
    // React Query dataset keys must prevent an old response from replacing the selected report.
    let resolveFirst!: (report: EDAReport) => void;
    vi.mocked(EDAService.getLatestReport).mockImplementation(datasetId => datasetId === 101
      ? new Promise(resolve => { resolveFirst = resolve; })
      : Promise.resolve({ ...completedReport, id: 202, profile_data: { ...completedReport.profile_data!, row_count: 202 } }));
    const { client } = renderPage();
    await waitFor(() => expect(EDAService.getLatestReport).toHaveBeenCalledWith(101));
    await screen.findByRole('option', { name: 'Shared dataset (202)' });
    fireEvent.change(screen.getByRole('combobox', { name: 'Dataset' }), { target: { value: '202' } });
    await screen.findByRole('button', { name: 'PII Review' });
    await act(async () => resolveFirst(completedReport));
    expect(observed.dashboard).toHaveBeenLastCalledWith(expect.objectContaining({ profile: expect.objectContaining({ row_count: 202 }) }));
    expect(client.getQueryData<EDAReport>(edaKeys.report(101))?.id).toBe(30);
    expect(useEDAStore.getState().selectedDataset).toBe(202);
  });

  it('discloses an unavailable deep-linked dataset without substituting another selection', async () => {
    // A requested but unavailable dataset must remain the query target and show a blank selector.
    vi.mocked(EDAService.getLatestReport).mockRejectedValue({ response: { status: 404 } });
    renderPage('/eda?dataset_id=999');
    expect(await screen.findByRole('alert')).toHaveTextContent("Dataset #999 isn't available for analysis");
    expect(screen.getByRole('combobox', { name: 'Dataset' })).toHaveValue('');
    expect(useEDAStore.getState().selectedDataset).toBe(999);
    expect(EDAService.getLatestReport).toHaveBeenCalledWith(999);
  });

  it.each(['constructor', 'toString', 'unknown-module'])('renders no tab content for an unrecognized stored tab %s', async activeTab => {
    // Tab dispatch must ignore inherited object keys as well as ordinary unknown names.
    renderPage();
    await screen.findByRole('textbox', { name: 'Dashboard local state' });
    act(() => useEDAStore.getState().setActiveTab(activeTab));
    expect(screen.queryByRole('textbox', { name: 'Dashboard local state' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'PII Review' })).toBeInTheDocument();
  });

  it('hydrates target and exclusions by report ID while retaining drafts on same-report refresh', async () => {
    // Polling the same report must not erase local exclusions on every response.
    const { client } = renderPage();
    await screen.findByRole('button', { name: 'PII Review' });
    act(() => useEDAStore.getState().toggleExclude('income', true));
    act(() => client.setQueryData(edaKeys.report(101), { ...completedReport, profile_data: { ...completedReport.profile_data, target_col: 'income', excluded_columns: ['age'] } }));
    await waitFor(() => expect(observed.dashboard).toHaveBeenLastCalledWith(expect.objectContaining({ profile: expect.objectContaining({ target_col: 'income' }) })));
    expect(useEDAStore.getState().excludedColsDraft).toEqual(['income']);
    act(() => client.setQueryData(edaKeys.report(101), { ...completedReport, id: 31, profile_data: { ...completedReport.profile_data, target_col: 'income', excluded_columns: ['age'] } }));
    await waitFor(() => expect(useEDAStore.getState()).toMatchObject({ targetCol: 'income', excludedColsDraft: ['age'], excludedColsApplied: ['age'] }));
  });

  it('loads saved reports into the current cache and preserves history modal fetch/select contracts', async () => {
    // Recent targets and the modal share the displayed cache slot without changing report identity.
    vi.mocked(EDAService.getHistory).mockResolvedValue([{ id: 41, status: 'COMPLETED', created_at: '2026-09-09T10:00:00Z', target_col: 'age' }]);
    const saved = { ...completedReport, id: 41, profile_data: { ...completedReport.profile_data!, target_col: 'age', excluded_columns: ['income'] } };
    vi.mocked(EDAService.getReport).mockResolvedValue(saved);
    const { client } = renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'age' }));
    await waitFor(() => expect(client.getQueryData<EDAReport>(edaKeys.report(101))?.id).toBe(41));
    expect(useEDAStore.getState().targetCol).toBe('age');
    fireEvent.click(screen.getByRole('button', { name: 'History' }));
    type HistoryProps = ComponentProps<typeof import('../components/eda/JobsHistoryModal').JobsHistoryModal>;
    const props = observed.history.mock.lastCall![0] as HistoryProps;
    expect(props).toMatchObject({ isOpen: true, datasetId: 101 });
    vi.mocked(EDAService.getReport).mockResolvedValue({ status: 'COMPLETED' });
    await expect(props.onFetchReport(99)).resolves.toEqual({ id: 99, status: 'COMPLETED' });
    act(() => props.onSelect({ ...saved, id: 42 }));
    expect(client.getQueryData<EDAReport>(edaKeys.report(101))?.id).toBe(42);
    expect(useEDAStore.getState().excludedColsApplied).toEqual(['income']);
  });
});

describe('EDAPage dataset selector', () => {
  it('shows distinguishable dataset labels when names collide and preserves the selected id', async () => {
    vi.mocked(DatasetService.getUsable).mockResolvedValue(duplicateNamedDatasets);
    vi.mocked(EDAService.getLatestReport).mockResolvedValue(null as never);
    vi.mocked(EDAService.getHistory).mockResolvedValue([] as never);
    vi.mocked(EDAService.analyze).mockResolvedValue({} as never);
    vi.mocked(EDAService.getReport).mockResolvedValue({} as never);
    useEDAStore.setState({
      activeTab: 'dashboard',
      selectedDataset: 101,
      targetCol: '',
      taskType: '',
      excludedColsDraft: [],
      excludedColsApplied: [],
      filtersDraft: [],
      filtersApplied: [],
      scatter: {
        x: '',
        y: '',
        z: '',
        color: '',
        is3D: false,
        isPCA3D: false,
      },
    });

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <EDAPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    const datasetSelect = (await screen.findByRole('combobox', { name: 'Dataset' })) as HTMLSelectElement;
    await waitFor(() => {
      expect(
        Array.from(datasetSelect.options).filter((option) => option.value !== ''),
      ).toHaveLength(2);
    });
    const datasetOptions = Array.from(datasetSelect.options).filter((option) => option.value !== '');

    expect(datasetOptions.map((option) => option.textContent)).toHaveLength(2);
    expect(new Set(datasetOptions.map((option) => option.textContent)).size).toBe(2);

    fireEvent.change(datasetSelect, { target: { value: '202' } });
    expect(datasetSelect).toHaveValue('202');
  });
});

describe('EDAPage filter workflow', () => {
  it('keeps filter edits draft-only until Apply is pressed and blocks duplicate submits', async () => {
    vi.mocked(DatasetService.getUsable).mockResolvedValue([
      {
        id: '101',
        source_id: 'source-a',
        name: 'EDA dataset',
        type: 'file',
        created_at: '2026-08-07T08:00:00.000Z',
        rows: 1200,
        columns: 12,
        format: 'csv',
      },
    ] as Dataset[]);
    vi.mocked(EDAService.getLatestReport).mockResolvedValue({
      id: 11,
      status: 'COMPLETED',
      profile_data: {
        columns: {
          age: {},
          income: {},
        },
      },
    } as never);
    vi.mocked(EDAService.getHistory).mockResolvedValue([] as never);
    let resolveAnalyze: (() => void) | undefined;
    vi.mocked(EDAService.analyze).mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveAnalyze = () => resolve({} as never);
      }),
    );
    useEDAStore.setState({
    activeTab: 'dashboard',
    selectedDataset: null,
      targetCol: '',
      taskType: '',
      excludedColsDraft: [],
      excludedColsApplied: [],
      filtersDraft: [],
      filtersApplied: [],
      scatter: {
        x: '',
        y: '',
        z: '',
        color: '',
        is3D: false,
        isPCA3D: false,
      },
    });

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/eda?dataset_id=101']}>
          <EDAPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    const addFilterButton = await screen.findByRole('button', { name: /Add Filter/i });
    fireEvent.click(addFilterButton);

    fireEvent.change(screen.getByRole('combobox', { name: 'Filter column' }), {
      target: { value: 'age' },
    });
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter operator' }), {
      target: { value: '>' },
    });
    fireEvent.change(screen.getByRole('textbox', { name: 'Filter value' }), {
      target: { value: '18' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Save draft/i }));

    expect(EDAService.analyze).not.toHaveBeenCalled();
    expect(useEDAStore.getState().filtersDraft).toHaveLength(1);
    expect(useEDAStore.getState().filtersApplied).toHaveLength(0);
    expect(screen.getByText(/Draft Filters \(1\)/)).toBeInTheDocument();
    const applyButton = screen.getByRole('button', { name: /Apply filters/i });
    await waitFor(() => expect(applyButton).toBeEnabled());
    fireEvent.click(applyButton);
    fireEvent.click(applyButton);

    await waitFor(() => expect(EDAService.analyze).toHaveBeenCalledTimes(1));
    expect(resolveAnalyze).toBeDefined();
    resolveAnalyze?.();

    // Once the draft has been applied there is nothing left to apply, so the
    // control goes back to disabled rather than inviting a duplicate re-run.
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Apply filters/i })).toBeDisabled();
    });
    expect(useEDAStore.getState().filtersApplied).toHaveLength(1);
  });
});
