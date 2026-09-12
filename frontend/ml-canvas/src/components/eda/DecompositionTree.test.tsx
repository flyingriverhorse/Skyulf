import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { EDAService } from '../../core/api/eda';
import { DecompositionTree } from './DecompositionTree';

const baseProps = {
  datasetId: 192,
  measureCol: 'count',
  measureAgg: 'count',
  columns: ['group', 'detail'],
  initialFilters: [],
};

/** Open the selected bucket's next split through the user-facing menu. */
function splitBy(column: string) {
  fireEvent.click(screen.getByTitle('Split further'));
  fireEvent.click(screen.getByRole('button', { name: column }));
}

describe('decomposition missing bucket identity', () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, 'scrollTo', { configurable: true, value: vi.fn() });
  });

  afterEach(() => vi.restoreAllMocks());

  it('keeps missing and literal Unknown selections separate through split, refresh, and cache restore', async () => {
    // Using labels for filters or selection would select both buckets and fetch the wrong rows.
    const getDecomposition = vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce([{ name: 'Total', value: 4, ratio: 1 }])
      .mockResolvedValueOnce([
        { name: 'Unknown', filter_value: null, value: 2, ratio: 0.5 },
        { name: 'Unknown', filter_value: 'Unknown', value: 1, ratio: 0.25 },
        { name: 'a', filter_value: 'a', value: 1, ratio: 0.25 },
      ])
      .mockResolvedValueOnce([{ name: 'missing detail', filter_value: 'missing detail', value: 2, ratio: 1 }])
      .mockResolvedValueOnce([{ name: 'literal detail', filter_value: 'literal detail', value: 1, ratio: 1 }])
      .mockResolvedValueOnce([{ name: 'missing detail', filter_value: 'missing detail', value: 2, ratio: 1 }]);
    const { unmount } = render(<DecompositionTree {...baseProps} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Total 4 (100%)' }));
    splitBy('group');
    const missing = await screen.findByRole('button', { name: 'Unknown (missing) 2 (50%)' });
    const literal = screen.getByRole('button', { name: 'Unknown 1 (25%)' });
    expect(missing.id).not.toBe(literal.id);
    fireEvent.click(missing);
    expect(missing).toHaveAttribute('aria-pressed', 'true');
    expect(literal).toHaveAttribute('aria-pressed', 'false');
    splitBy('detail');
    await screen.findByText('missing detail');
    expect(getDecomposition).toHaveBeenLastCalledWith(192, null, 'count', 'detail', [
      { column: 'group', operator: '==', value: null },
    ]);

    fireEvent.click(literal);
    await screen.findByText('literal detail');
    expect(getDecomposition).toHaveBeenLastCalledWith(192, null, 'count', 'detail', [
      { column: 'group', operator: '==', value: 'Unknown' },
    ]);
    expect(missing).toHaveAttribute('aria-pressed', 'false');
    expect(literal).toHaveAttribute('aria-pressed', 'true');

    unmount();
    render(<DecompositionTree {...baseProps} />);
    fireEvent.click(screen.getByRole('button', { name: 'Unknown (missing) 2 (50%)' }));
    await screen.findByText('missing detail');
    expect(getDecomposition).toHaveBeenLastCalledWith(192, null, 'count', 'detail', [
      { column: 'group', operator: '==', value: null },
    ]);
  });

  it('uses the existing label filter when an older response omits filter_value', async () => {
    // Cached or older API rows remain drillable without guessing a null identity from their label.
    const getDecomposition = vi.spyOn(EDAService, 'getDecomposition')
      .mockResolvedValueOnce([{ name: 'Total', value: 1, ratio: 1 }])
      .mockResolvedValueOnce([{ name: 'Unknown', value: 1, ratio: 1 }])
      .mockResolvedValueOnce([{ name: 'legacy detail', value: 1, ratio: 1 }]);
    render(<DecompositionTree {...baseProps} datasetId={193} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Total 1 (100%)' }));
    splitBy('group');
    fireEvent.click(await screen.findByRole('button', { name: 'Unknown 1 (100%)' }));
    splitBy('detail');
    await waitFor(() => expect(screen.getByText('legacy detail')).toBeInTheDocument());
    expect(getDecomposition).toHaveBeenLastCalledWith(193, null, 'count', 'detail', [
      { column: 'group', operator: '==', value: 'Unknown' },
    ]);
  });
});
