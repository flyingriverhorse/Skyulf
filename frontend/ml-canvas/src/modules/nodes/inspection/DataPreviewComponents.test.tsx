import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { jobsApi } from '../../../core/api/jobs';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useViewStore } from '../../../core/store/useViewStore';
import { DataPreviewSettings } from './DataPreviewComponents';

const polling = vi.hoisted(() => ({ jobs: {} as Record<string, unknown> }));

vi.mock('../../../core/api/jobs', () => ({ jobsApi: { runPipeline: vi.fn() } }));
vi.mock('../../../core/hooks/useJobPolling', () => ({ useJobPolling: () => polling }));
vi.mock('../../../core/utils/pipelineConverter', () => ({ convertGraphToPipelineConfig: () => ({ nodes: [] }) }));
vi.mock('../../../core/utils/pipelineLeakageValidation', () => ({ warnAndBlockOnLeakage: () => false }));
vi.mock('../../../core/toast', () => ({ toast: { error: vi.fn() } }));

beforeEach(() => {
  vi.clearAllMocks();
  polling.jobs = {};
  useViewStore.setState({ leakageNotice: null, isResultsPanelExpanded: false });
  useGraphStore.setState({
    nodes: [{ id: 'preview', position: { x: 0, y: 0 }, data: { definitionType: 'data_preview' } }], edges: [],
  });
});

it('renders ordered sample columns, null cells and capped rows while switching splits', () => {
  // Preview tables must show backend column order and retain zero dimensions and empty values.
  polling.jobs = { job: { status: 'completed', result: { metrics: {
    operation_mode: 'split', data_summary: {
      train: { name: 'Training', shape: [0, 0], sample: Array.from({ length: 22 }, (_, index) => ({ z: index, a: null, flag: false })) },
      test: { sample: [{ name: 'Test row' }] },
      validation: { sample: [] },
    }, applied_transformations: [{ transformer_name: 'Scale', transformer_type: 'numeric' }] } } } };
  render(<DataPreviewSettings config={{ lastRunJobId: 'job' }} onChange={vi.fn()} nodeId="preview" />);
  expect(screen.getAllByRole('columnheader').map(cell => cell.textContent)).toEqual(['z', 'a', 'flag']);
  expect(screen.getAllByRole('row')).toHaveLength(21);
  expect(screen.getByText(/0 rows x 0 cols/)).toHaveTextContent('(showing 20)');
  expect(screen.getAllByRole('cell').slice(0, 3).map(cell => cell.textContent)).toEqual(['0', '', 'false']);
  expect(screen.getByText('Scale')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Test' }));
  expect(screen.getByText('Test row')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Validation' }));
  expect(screen.getByText('No data available')).toBeInTheDocument();
});

it('submits exact preview fields and preserves config while storing returned branch IDs', async () => {
  // Parallel preview responses must retain job order for labels and polling.
  const onChange = vi.fn();
  vi.mocked(jobsApi.runPipeline).mockResolvedValue({ job_id: 'first', job_ids: ['first', 'second'] } as never);
  render(<DataPreviewSettings config={{ lastRunJobId: 'old' }} onChange={onChange} nodeId="preview" />);
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Run Preview' })));
  expect(jobsApi.runPipeline).toHaveBeenCalledWith({ nodes: [], target_node_id: 'preview', job_type: 'preview' });
  expect(onChange).toHaveBeenCalledWith({ lastRunJobId: 'first', lastRunJobIds: ['first', 'second'] });
});

/** Preview-node submission failures must explain safety errors without opening the large panel. */
it('routes backend leakage errors into a canvas notice and remains retryable', async () => {
  const detail = 'Data leakage risk: preprocessing fits on unsplit data.';
  vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce({ response: { data: { detail } } });
  render(<DataPreviewSettings config={{}} onChange={vi.fn()} nodeId="preview" />);
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Run Preview' })));
  expect(useViewStore.getState().leakageNotice?.message).toBe(detail);
  expect(useViewStore.getState().isResultsPanelExpanded).toBe(false);
  expect(screen.getByRole('button', { name: 'Run Preview' })).toBeEnabled();
});
