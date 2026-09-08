import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { Dataset } from '../core/types/api';
import type { EDAReport } from '../core/api/eda';
import { EDAService } from '../core/api/eda';
import { DatasetService } from '../core/api/datasets';
import { useEDAStore } from '../core/store/useEDAStore';
import { EDAPage } from './EDAPage';

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
    useEDAStore.setState({ selectedDataset: 101 });
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
  JobsHistoryModal: () => null,
}));

vi.mock('../components/shared', () => ({
  LoadingState: ({ message }: { message?: string }) => <div>{message}</div>,
  ErrorState: ({ error }: { error: string }) => <div>{error}</div>,
}));

vi.mock('../core/utils/chartUtils', () => ({
  downloadChart: vi.fn(),
  getTooltipContentStyle: vi.fn(() => ({})),
}));

vi.mock('../components/eda/tabs/DashboardTab', () => ({ DashboardTab: () => null }));
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
vi.mock('../components/eda/tabs/DecompositionTab', () => ({ DecompositionTab: () => null }));

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
