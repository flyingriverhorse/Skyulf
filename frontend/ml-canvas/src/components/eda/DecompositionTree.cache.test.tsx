import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { EDAService, type Filter } from '../../core/api/eda';
import { DecompositionTree } from './DecompositionTree';
import { DecompositionTab } from './tabs/DecompositionTab';

const eu: Filter = { column: 'region', operator: '==', value: 'EU' };
const us: Filter = { ...eu, value: 'US' };
const positive: Filter = { column: 'revenue', operator: '>', value: 0 };
const props = { measureCol: 'count', measureAgg: 'count', columns: ['region'] };

/** Distinct labels expose which population supplied the rendered result. */
function rows(name: string) {
  return [{ name, value: 2, ratio: 1 }];
}

describe('decomposition cache population', () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, 'scrollTo', { configurable: true, value: vi.fn() });
  });
  afterEach(() => vi.restoreAllMocks());

  it('loads changed filters and restores only their own cached tree on remount', async () => {
    // A report filter change must never reuse another population's totals.
    const fetch = vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce(rows('EU total')).mockResolvedValueOnce(rows('US total'));
    const view = render(<DecompositionTree {...props} datasetId={7501} initialFilters={[eu]} />);
    await screen.findByText('EU total');
    view.rerender(<DecompositionTree {...props} datasetId={7501} initialFilters={[us]} />);
    await screen.findByText('US total');
    expect(fetch).toHaveBeenLastCalledWith(7501, null, 'count', '', [us]);
    view.unmount();
    render(<DecompositionTree {...props} datasetId={7501} initialFilters={[eu]} />);
    expect(await screen.findByText('EU total')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('does not copy old levels into a different dataset cache', async () => {
    // Direct prop updates must be as safe as the tab's keyed measure changes.
    const fetch = vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce(rows('dataset A')).mockResolvedValueOnce(rows('dataset B'));
    const view = render(<DecompositionTree {...props} datasetId={7502} initialFilters={[]} />);
    await screen.findByText('dataset A');
    view.rerender(<DecompositionTree {...props} datasetId={7503} initialFilters={[]} />);
    expect(await screen.findByText('dataset B')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('reuses semantically equivalent conjunctions and membership sets', async () => {
    // Filter order and the order of an IN set do not change the population.
    const members: Filter = { column: 'region', operator: 'in', value: ['EU', 'US'] };
    const fetch = vi.spyOn(EDAService, 'getDecomposition').mockResolvedValue(rows('same population'));
    const view = render(<DecompositionTree {...props} datasetId={7504} initialFilters={[members, positive]} />);
    await screen.findByText('same population');
    view.unmount();
    render(<DecompositionTree {...props} datasetId={7504} initialFilters={[positive, { ...members, value: ['US', 'EU', 'EU'] }]} />);
    expect(await screen.findByText('same population')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('ignores a pending root when switching to an already cached filter', async () => {
    // A cache hit must cancel the previous request even though it starts no new request.
    let resolveUS!: (value: ReturnType<typeof rows>) => void;
    const fetch = vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce(rows('cached EU'))
      .mockImplementationOnce(() => new Promise(resolve => { resolveUS = resolve; }));
    const view = render(<DecompositionTree {...props} datasetId={7505} initialFilters={[eu]} />);
    await screen.findByText('cached EU');
    view.rerender(<DecompositionTree {...props} datasetId={7505} initialFilters={[us]} />);
    expect(fetch).toHaveBeenCalledTimes(2);
    view.rerender(<DecompositionTree {...props} datasetId={7505} initialFilters={[eu]} />);
    await act(async () => { resolveUS(rows('late US')); });
    expect(screen.getByText('cached EU')).toBeInTheDocument();
    expect(screen.queryByText('late US')).not.toBeInTheDocument();
  });

  it('invalidates an old split response after changing filters', async () => {
    // Drill-down responses belong to the filters that initiated them.
    let resolveSplit!: (value: ReturnType<typeof rows>) => void;
    vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce(rows('EU root'))
      .mockImplementationOnce(() => new Promise(resolve => { resolveSplit = resolve; }))
      .mockResolvedValueOnce(rows('US root'));
    const view = render(<DecompositionTree {...props} datasetId={7506} initialFilters={[eu]} />);
    fireEvent.click(await screen.findByRole('button', { name: 'EU root 2 (100%)' }));
    fireEvent.click(screen.getByTitle('Split further'));
    fireEvent.click(screen.getByRole('button', { name: 'region' }));
    view.rerender(<DecompositionTree {...props} datasetId={7506} initialFilters={[us]} />);
    await screen.findByText('US root');
    await act(async () => { resolveSplit(rows('late EU split')); });
    expect(screen.getByText('US root')).toBeInTheDocument();
    expect(screen.queryByText('late EU split')).not.toBeInTheDocument();
  });

  it('resets by fetching fresh rows and persists the fresh tree across tab remount', async () => {
    // Reset must evict the cached split path rather than remounting it unchanged.
    const fetch = vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce(rows('old root')).mockResolvedValueOnce(rows('old split'))
      .mockResolvedValueOnce(rows('fresh root'));
    const view = render(<DecompositionTab datasetId={7507} columns={['region']} initialFilters={[eu]} />);
    fireEvent.click(await screen.findByRole('button', { name: 'old root 2 (100%)' }));
    fireEvent.click(screen.getByTitle('Split further'));
    fireEvent.click(screen.getByRole('button', { name: 'region' }));
    await screen.findByText('old split');
    fireEvent.click(screen.getByRole('button', { name: 'Reset Tree' }));
    await screen.findByText('fresh root');
    expect(screen.queryByText('old split')).not.toBeInTheDocument();
    view.unmount();
    render(<DecompositionTab datasetId={7507} columns={['region']} initialFilters={[eu]} />);
    expect(await screen.findByText('fresh root')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(3);
  });
});
