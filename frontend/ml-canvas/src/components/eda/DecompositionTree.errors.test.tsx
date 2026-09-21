import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { EDAService } from '../../core/api/eda';
import { DecompositionTree } from './DecompositionTree';

const props = { measureCol: 'count', measureAgg: 'count', columns: ['region'], initialFilters: [] };
const rows = (name: string) => [{ name, value: 2, ratio: 1 }];
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, 'scrollTo', { configurable: true, value: vi.fn() });
  vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => vi.restoreAllMocks());

it('shows root failures and retries the same population', async () => {
  // A failed root must not leave an unexplained empty tree.
  const fetch = vi.spyOn(EDAService, 'getDecomposition').mockRejectedValueOnce(new Error('root failed')).mockResolvedValueOnce(rows('Recovered'));
  render(<DecompositionTree {...props} datasetId={10301} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Could not load decomposition');
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(await screen.findByText('Recovered')).toBeVisible();
  expect(fetch.mock.calls[1]).toEqual(fetch.mock.calls[0]);
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('keeps successful levels and retries a failed split without reopening its menu', async () => {
  // Split retries need the original split context after its popup has closed.
  const fetch = vi.spyOn(EDAService, 'getDecomposition').mockResolvedValueOnce(rows('Total'))
    .mockRejectedValueOnce(new Error('split failed')).mockResolvedValueOnce(rows('EU'));
  render(<DecompositionTree {...props} datasetId={10302} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Total 2 (100%)' }));
  fireEvent.click(screen.getByTitle('Split further'));
  fireEvent.click(screen.getByRole('button', { name: 'region' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('region');
  expect(screen.getByText('Total', { selector: '[title="Total"]' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(await screen.findByText('EU')).toBeVisible();
  expect(fetch.mock.calls[2]).toEqual(fetch.mock.calls[1]);
});

it('ignores late errors after filters change', async () => {
  // Old requests must not place stale retry actions over the current population.
  let reject!: (error: Error) => void;
  vi.spyOn(EDAService, 'getDecomposition').mockImplementationOnce(() => new Promise((_resolve, rejectRequest) => { reject = rejectRequest; }))
    .mockResolvedValueOnce(rows('New population'));
  const view = render(<DecompositionTree {...props} datasetId={10303} />);
  view.rerender(<DecompositionTree {...props} datasetId={10303} initialFilters={[{ column: 'region', operator: '==', value: 'EU' }]} />);
  await screen.findByText('New population');
  await act(async () => reject(new Error('old error')));
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  expect(screen.getByText('New population')).toBeVisible();
});

it('keeps the old selection and descendants until a failed refresh is retried successfully', async () => {
  // A parent selection and its dependent children must update atomically.
  const fetch = vi.spyOn(EDAService, 'getDecomposition').mockResolvedValueOnce(rows('Total'))
    .mockResolvedValueOnce([...rows('EU'), ...rows('US')]).mockResolvedValueOnce(rows('EU detail'))
    .mockRejectedValueOnce(new Error('refresh failed')).mockResolvedValueOnce(rows('US detail'));
  render(<DecompositionTree {...props} columns={['region', 'detail']} datasetId={10304} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Total 2 (100%)' }));
  fireEvent.click(screen.getByTitle('Split further'));
  fireEvent.click(screen.getByRole('button', { name: 'region' }));
  fireEvent.click(await screen.findByRole('button', { name: 'EU 2 (100%)' }));
  fireEvent.click(screen.getByTitle('Split further'));
  fireEvent.click(screen.getByRole('button', { name: 'detail' }));
  await screen.findByText('EU detail');
  fireEvent.click(screen.getByRole('button', { name: 'US 2 (100%)' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Could not refresh detail');
  expect(screen.getByRole('button', { name: 'EU 2 (100%)' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByText('EU detail')).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
  expect(await screen.findByText('US detail')).toBeVisible();
  expect(screen.queryByText('EU detail')).not.toBeInTheDocument();
  expect(fetch.mock.calls[4]).toEqual(fetch.mock.calls[3]);
  expect(screen.getByRole('button', { name: 'US 2 (100%)' })).toHaveAttribute('aria-pressed', 'true');
});
