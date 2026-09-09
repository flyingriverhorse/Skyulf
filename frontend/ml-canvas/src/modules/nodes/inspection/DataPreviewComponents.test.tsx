import { act, fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { jobsApi } from '../../../core/api/jobs';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useViewStore } from '../../../core/store/useViewStore';
import { DataPreviewSettings } from './DataPreviewComponents';

vi.mock('../../../core/api/jobs', () => ({ jobsApi: { runPipeline: vi.fn() } }));
vi.mock('../../../core/hooks/useJobPolling', () => ({ useJobPolling: () => ({ jobs: {} }) }));
vi.mock('../../../core/utils/pipelineConverter', () => ({ convertGraphToPipelineConfig: () => ({ nodes: [] }) }));
vi.mock('../../../core/utils/pipelineLeakageValidation', () => ({ warnAndBlockOnLeakage: () => false }));
vi.mock('../../../core/toast', () => ({ toast: { error: vi.fn() } }));

beforeEach(() => {
  vi.clearAllMocks();
  useViewStore.setState({ leakageNotice: null, isResultsPanelExpanded: false });
  useGraphStore.setState({
    nodes: [{ id: 'preview', position: { x: 0, y: 0 }, data: { definitionType: 'data_preview' } }], edges: [],
  });
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
