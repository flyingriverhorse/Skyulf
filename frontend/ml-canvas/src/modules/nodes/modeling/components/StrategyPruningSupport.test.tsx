import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { apiClient } from '../../../../core/api/client';
import { useGraphStore } from '../../../../core/store/useGraphStore';
import { StrategySettingsModal } from './StrategySettingsModal';

const post = vi.spyOn(apiClient, 'post');

beforeEach(() => {
  post.mockReset();
  useGraphStore.setState({ nodes: [
    { id: 'source', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetId: 'data' } },
    { id: 'train', type: 'custom', position: { x: 400, y: 0 }, data: { definitionType: 'classification', run_mode: 'advanced', model_type: 'sgd_classifier', target_column: 'y' } },
  ], edges: [{ id: 'source-train', source: 'source', target: 'train' }] });
  useGraphStore.temporal.getState().clear();
});

const props = { isOpen: true, strategy: 'optuna', nodeId: 'train', modelKey: 'sgd_classifier',
  searchSpace: { alpha: [0.001] }, onSave: vi.fn(), onClose: vi.fn() };

it('disables unavailable pruning, explains why, and applies None without discarding other options', async () => {
  // Unsupported saved choices must be honest on screen and in the applied configuration.
  post.mockResolvedValue({ data: { supported: false, reason: 'Random Forest does not support incremental training.' } });
  const onSave = vi.fn();
  render(<StrategySettingsModal {...props} modelKey="random_forest_classifier" onSave={onSave}
    initialConfig={{ pruner: 'hyperband', sampler: 'random', timeout: 120 }} />);
  const pruner = screen.getByRole('combobox', { name: 'Pruner' });
  await waitFor(() => expect(pruner).toHaveAccessibleDescription(/Random Forest does not support/));
  expect(pruner).toBeDisabled();
  expect(pruner).toHaveValue('none');
  expect(useGraphStore.temporal.getState().pastStates).toHaveLength(0);
  expect(onSave).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Apply Settings' }));
  expect(onSave).toHaveBeenCalledWith({ pruner: 'none', sampler: 'random', timeout: 120 });
});

it('checks current model, search options and executable graph while retaining supported choices', async () => {
  // The frontend must ask the same configuration that the backend will train.
  post.mockResolvedValue({ data: { supported: true, reason: null } });
  render(<StrategySettingsModal {...props} initialConfig={{ pruner: 'hyperband' }} />);
  const pruner = screen.getByRole('combobox', { name: 'Pruner' });
  await waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/pipeline/pruning-support');
  expect(post.mock.calls[0]?.[1]).toMatchObject({ node_id: 'train', model_type: 'sgd_classifier',
    search_space: { alpha: [0.001] }, pipeline: { nodes: [
      { node_id: 'source', step_type: 'data_loader' }, { node_id: 'train', step_type: 'training', inputs: ['source'] },
    ] } });
  await waitFor(() => expect(pruner).toBeEnabled());
  expect(pruner).toHaveValue('hyperband');
});

it('rechecks graph changes but ignores moves and selection without losing the saved pruner', async () => {
  // Capability follows pipeline semantics, and opening/checking must not create hidden edits.
  post.mockResolvedValueOnce({ data: { supported: true, reason: null } });
  const { rerender } = render(<StrategySettingsModal {...props} initialConfig={{ pruner: 'hyperband' }} />);
  const pruner = screen.getByRole('combobox', { name: 'Pruner' });
  await waitFor(() => expect(pruner).toBeEnabled());
  const calls = post.mock.calls.length;
  act(() => useGraphStore.setState({ nodes: useGraphStore.getState().nodes.map(node => ({ ...node, selected: true, position: { x: 30, y: 40 } })) }));
  expect(post).toHaveBeenCalledTimes(calls);
  post.mockResolvedValueOnce({ data: { supported: false, reason: 'Preprocessing is fitted inside each validation fold.' } });
  rerender(<StrategySettingsModal {...props} searchSpace={{ max_iter: [5, 10] }} initialConfig={{ pruner: 'hyperband' }} />);
  await waitFor(() => expect(pruner).toHaveValue('none'));
  expect(pruner).toBeDisabled();
  post.mockResolvedValueOnce({ data: { supported: true, reason: null } });
  rerender(<StrategySettingsModal {...props} initialConfig={{ pruner: 'hyperband' }} />);
  await waitFor(() => expect(pruner).toBeEnabled());
  expect(pruner).toHaveValue('hyperband');
});

it('ignores a stale success after model switch and aborts its request', async () => {
  // Old supported models cannot enable pruning for the newly selected unsupported model.
  let resolveOld!: (value: unknown) => void;
  post.mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve; }));
  const { rerender } = render(<StrategySettingsModal {...props} />);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  const signal = post.mock.calls[0]?.[2]?.signal;
  post.mockResolvedValueOnce({ data: { supported: false, reason: 'This model cannot train incrementally.' } });
  rerender(<StrategySettingsModal {...props} modelKey="random_forest_classifier" />);
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Pruner' })).toHaveAccessibleDescription(/cannot train incrementally/));
  await act(async () => resolveOld({ data: { supported: true, reason: null } }));
  expect(signal?.aborted).toBe(true);
  expect(screen.getByRole('combobox', { name: 'Pruner' })).toBeDisabled();
});

it('preserves saved pruning on a check failure and aborts on close', async () => {
  // An unavailable metadata service must not rewrite a saved user choice to None.
  post.mockRejectedValue(new Error('offline'));
  const onSave = vi.fn();
  const { rerender } = render(<StrategySettingsModal {...props} onSave={onSave} initialConfig={{ pruner: 'hyperband' }} />);
  const pruner = screen.getByRole('combobox', { name: 'Pruner' });
  await waitFor(() => expect(pruner).toHaveAccessibleDescription(/Could not check pruning support/));
  expect(pruner).toBeDisabled();
  expect(pruner).toHaveValue('hyperband');
  fireEvent.click(screen.getByRole('button', { name: 'Apply Settings' }));
  expect(onSave).toHaveBeenCalledWith({ pruner: 'hyperband' });
  const signal = post.mock.calls[0]?.[2]?.signal;
  rerender(<StrategySettingsModal {...props} isOpen={false} />);
  expect(signal?.aborted).toBe(true);
});

it('checks again on reopen without exposing the previous supported result while loading', async () => {
  // Reopening must wait for fresh capability instead of briefly enabling a stale choice.
  post.mockResolvedValueOnce({ data: { supported: true, reason: null } });
  const initialConfig = { pruner: 'hyperband' as const };
  const { rerender } = render(<StrategySettingsModal {...props} initialConfig={initialConfig} />);
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Pruner' })).toBeEnabled());
  rerender(<StrategySettingsModal {...props} initialConfig={initialConfig} isOpen={false} />);
  post.mockImplementationOnce(() => new Promise(() => {}));
  rerender(<StrategySettingsModal {...props} initialConfig={initialConfig} />);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  const pruner = screen.getByRole('combobox', { name: 'Pruner' });
  expect(pruner).toBeDisabled();
  expect(pruner).toHaveValue('hyperband');
  expect(pruner).toHaveAccessibleDescription(/Checking pruning support/);
});
