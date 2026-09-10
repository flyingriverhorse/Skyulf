import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { TrainTestSplitNode } from './TrainTestSplitNode';

const boundary = vi.hoisted(() => ({ upstream: [] as Record<string, unknown>[] }));
vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => boundary.upstream }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: { columns: { target: { name: 'target', dtype: 'int' } } } }) }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));
beforeEach(() => { boundary.upstream = []; });

/** Ratios allow NaN while seed clearing retains its prior value; configuration patches remain complete. */
it('preserves ratio/seed parsing, toggles and split warning text', () => {
  const Settings = TrainTestSplitNode.settings!;
  const config = { ...TrainTestSplitNode.getDefaultConfig(), test_size: 0.8, validation_size: 0.2 };
  const onChange = vi.fn();
  render(<Settings config={config} onChange={onChange} />);
  expect(screen.getByText('Error: Total split size exceeds 100%')).toBeVisible();
  expect(screen.getByLabelText('Target Column')).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Test Size (0.0 - 1.0)'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, test_size: NaN });
  fireEvent.change(screen.getByLabelText('Random State'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, random_state: 42 });
  fireEvent.change(screen.getByLabelText('Random State'), { target: { value: '0' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, random_state: 0 });
  fireEvent.click(screen.getByLabelText('Stratify by Target'));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, stratify: true });
});

/** Upstream propagation uses first matches and removes datasetId with an explicit undefined property. */
it('preserves upstream update ordering and disconnected dataset cleanup', () => {
  boundary.upstream = [{ datasetId: 'first', target_column: 'target' }, { datasetId: 'second' }];
  const Settings = TrainTestSplitNode.settings!;
  const config = TrainTestSplitNode.getDefaultConfig();
  const onChange = vi.fn();
  const { rerender } = render(<Settings config={config} onChange={onChange} />);
  expect(onChange).toHaveBeenCalledExactlyOnceWith({ ...config, datasetId: 'first', target_column: 'target' });
  boundary.upstream = [];
  rerender(<Settings config={{ ...config, datasetId: 'first', target_column: 'target' }} onChange={onChange} />);
  expect(onChange).toHaveBeenLastCalledWith({ ...config, datasetId: undefined, target_column: 'target' });
});
