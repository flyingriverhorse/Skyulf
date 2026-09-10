import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { TransformationNode } from './TransformationNode';

vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: { columns: { value: { name: 'value', dtype: 'float' } } } }) }));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set() }));
vi.mock('../../../core/hooks/useRecommendations', () => ({ useRecommendations: () => [] }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));

/** Expanding rules retains zero fallback display, NaN callbacks and unrelated transformation parameters. */
it('preserves rule-specific numeric and method controls with real column selection', () => {
  const Settings = TransformationNode.settings!;
  const rule = { columns: [] as string[], method: 'exponential' as const, params: { clip_threshold: 0, standardize: false } };
  const config = { transformations: [rule] };
  const onChange = vi.fn();
  const { rerender } = render(<Settings config={config} onChange={onChange} />);
  fireEvent.click(screen.getByRole('button', { name: /Expand.*transformation rule 1/ }));
  expect(screen.getByLabelText('Clip Threshold for rule 1')).toHaveValue(700);
  fireEvent.change(screen.getByLabelText('Clip Threshold for rule 1'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ transformations: [{ ...rule, params: { ...rule.params, clip_threshold: NaN } }] });
  fireEvent.click(screen.getByRole('checkbox', { name: 'value' }));
  expect(onChange).toHaveBeenLastCalledWith({ transformations: [{ ...rule, columns: ['value'] }] });
  fireEvent.change(screen.getByLabelText('Method type for rule 1'), { target: { value: 'power' } });
  expect(onChange).toHaveBeenLastCalledWith({ transformations: [{ ...rule, method: 'yeo-johnson' }] });
  rerender(<Settings config={{ transformations: [{ ...rule, method: 'yeo-johnson' }] }} onChange={onChange} />);
  expect(screen.getByLabelText('Standardize result for rule 1')).not.toBeChecked();
  fireEvent.click(screen.getByLabelText('Standardize result for rule 1'));
  expect(onChange).toHaveBeenLastCalledWith({ transformations: [{ ...rule, method: 'yeo-johnson', params: { ...rule.params, standardize: true } }] });
});
