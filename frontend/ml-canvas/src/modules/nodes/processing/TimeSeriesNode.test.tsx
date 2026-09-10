import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { TimeSeriesNode } from './TimeSeriesNode';

vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: { columns: { value: { name: 'value', dtype: 'float' } } } }) }));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set() }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));

/** Missing columns take precedence over method-specific errors, which keep their exact field and message. */
it.each([
  { method: 'lag' as const, field: 'lags', message: 'Provide at least one lag value' },
  { method: 'rolling' as const, field: 'aggregations', message: 'Select at least one aggregation' },
  { method: 'date' as const, field: 'features', message: 'Select at least one calendar feature' },
])('preserves $method validation precedence', ({ method, field, message }) => {
  const config = { ...TimeSeriesNode.getDefaultConfig(), method, lags: [], aggregations: [], features: [] };
  expect(TimeSeriesNode.validate(config)).toEqual({ isValid: false, field: 'columns', message: 'Select at least one column' });
  expect(TimeSeriesNode.validate({ ...config, columns: ['value'] })).toEqual({ isValid: false, field, message });
  expect(TimeSeriesNode.validate({ ...TimeSeriesNode.getDefaultConfig(), method, columns: ['value'] })).toEqual({ isValid: true });
});

/** Method changes retain unrelated values and existing lag/rolling parsing rather than normalizing them. */
it('preserves method patches, lag filtering and empty numeric callbacks', () => {
  const Settings = TimeSeriesNode.settings!;
  const config = { ...TimeSeriesNode.getDefaultConfig(), columns: ['value'] };
  const onChange = vi.fn();
  const { rerender } = render(<Settings config={config} onChange={onChange} />);
  fireEvent.change(screen.getByLabelText('Lags (comma-separated)'), { target: { value: '0, -1, 2.8, invalid' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, lags: [0, -1, 2] });
  fireEvent.change(screen.getByLabelText('Sort By (optional)'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, sort_by: undefined });
  fireEvent.change(screen.getByLabelText('Method'), { target: { value: 'rolling' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, method: 'rolling' });
  const rolling = { ...config, method: 'rolling' as const };
  rerender(<Settings config={rolling} onChange={onChange} />);
  fireEvent.change(screen.getByLabelText('Window'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...rolling, window: NaN });
  fireEvent.click(screen.getByRole('button', { name: 'sum' }));
  expect(onChange).toHaveBeenLastCalledWith({ ...rolling, aggregations: ['mean', 'sum'] });
  rerender(<Settings config={{ ...config, method: 'date' }} onChange={onChange} />);
  expect(screen.queryByLabelText('Sort By (optional)')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'year', pressed: true })).toBeVisible();
});

/** Omitted method fields retain their validation errors instead of borrowing the settings display defaults. */
it.each([
  { method: 'lag' as const, field: 'lags' },
  { method: 'rolling' as const, field: 'aggregations' },
  { method: 'date' as const, field: 'features' },
])('rejects omitted $field values', ({ method, field }) => {
  expect(TimeSeriesNode.validate({ method, columns: ['value'] })).toMatchObject({ isValid: false, field });
});
