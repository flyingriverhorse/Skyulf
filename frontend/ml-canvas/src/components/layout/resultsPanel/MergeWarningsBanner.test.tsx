import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { MergeWarning } from '../../../core/api/client';
import { MergeWarningsBanner } from './MergeWarningsBanner';

describe('MergeWarningsBanner', () => {
  it('preserves warning duplicates and order and confirms exact ordered chain inputs', async () => {
    // Advisory order and repeated nodes must survive rendering and the rewire action.
    const warning = { kind: 'sibling_fan_in', node_id: 'merge', inputs: ['b', 'a'], overlap_columns: ['x'] } as MergeWarning;
    const confirm = vi.fn().mockResolvedValueOnce(false).mockResolvedValueOnce(true);
    const chainSiblings = vi.fn().mockReturnValue(true);
    const { container } = render(<MergeWarningsBanner mergeWarnings={[warning, warning]}
      mergeWarningsOpen setMergeWarningsOpen={vi.fn()} nodeLabelMap={{ merge: 'Merge', a: 'A', b: 'B' }}
      confirm={confirm} chainSiblings={chainSiblings} />);
    expect(container.querySelectorAll('.pl-5')).toHaveLength(2);
    expect(screen.getByRole('button', { name: /2 merge advisories/ })).toHaveTextContent('Merge, Merge');
    fireEvent.click(screen.getAllByRole('button', { name: 'Chain instead' })[0]!);
    await waitFor(() => expect(confirm).toHaveBeenCalledOnce());
    expect(chainSiblings).not.toHaveBeenCalled();
    fireEvent.click(screen.getAllByRole('button', { name: 'Chain instead' })[1]!);
    await waitFor(() => expect(chainSiblings).toHaveBeenCalledWith('merge', ['b', 'a']));
    expect(confirm).toHaveBeenLastCalledWith(expect.objectContaining({ title: 'Rewire as a linear chain?', confirmLabel: 'Rewire' }));
  });
  it('renders upstream drop reapplied warnings without fan-in misinformation', () => {
    const warning: MergeWarning = {
      node_id: 'missing-indicator',
      kind: 'upstream_drop_reapplied',
      inputs: ['source-a'],
      dropped_columns: ['Id'],
      message: 'Id was removed again because an upstream Drop Columns step removed it.',
    };

    render(
      <MergeWarningsBanner
        mergeWarnings={[warning]}
        mergeWarningsOpen
        setMergeWarningsOpen={vi.fn()}
        nodeLabelMap={{ 'missing-indicator': 'MissingIndicator' }}
        confirm={vi.fn()}
        chainSiblings={vi.fn()}
      />,
    );

    expect(
      screen.getByText(/removed by an upstream Drop Columns step/i),
    ).toBeTruthy();
    expect(screen.getByText('Id')).toBeTruthy();
    expect(screen.queryByText(/merges 0 parallel branches/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/all columns from all branches are kept/i)).not.toBeInTheDocument();
  });

  it('renders without a backend message (runs cached before the field existed)', () => {
    const warning = {
      node_id: 'missing-indicator',
      kind: 'upstream_drop_reapplied',
      inputs: ['source-a', 'source-b'],
      dropped_columns: ['Id'],
    } as unknown as MergeWarning;

    render(
      <MergeWarningsBanner
        mergeWarnings={[warning]}
        mergeWarningsOpen
        setMergeWarningsOpen={vi.fn()}
        nodeLabelMap={{ 'missing-indicator': 'MissingIndicator' }}
        confirm={vi.fn()}
        chainSiblings={vi.fn()}
      />,
    );

    expect(screen.getByText('Id')).toBeTruthy();
    expect(screen.queryByText(/all columns from all branches are kept/i)).not.toBeInTheDocument();
  });

  it('names the node exactly once', () => {
    const warning: MergeWarning = {
      node_id: 'missing-indicator',
      kind: 'upstream_drop_reapplied',
      inputs: ['source-a', 'source-b'],
      dropped_columns: ['Id'],
      message: "Node 'missing-indicator': a sibling branch reintroduced column(s) ['Id'].",
    };

    const { container } = render(
      <MergeWarningsBanner
        mergeWarnings={[warning]}
        mergeWarningsOpen
        setMergeWarningsOpen={vi.fn()}
        nodeLabelMap={{ 'missing-indicator': 'MissingIndicator' }}
        confirm={vi.fn()}
        chainSiblings={vi.fn()}
      />,
    );

    const detail = container.querySelectorAll('.pl-5')[0]?.textContent ?? '';
    expect(detail.match(/MissingIndicator/g) ?? []).toHaveLength(1);
    expect(detail).not.toContain('missing-indicator');
  });

  it('renders row-count mismatch without fan-in misinformation', () => {
    const warning: MergeWarning = {
      node_id: 'merge-1',
      kind: 'row_count_mismatch',
      part: 'train',
      row_counts: [5, 4],
      message:
        "Node 'merge-1': inputs have different row counts (5 vs 4), so they were " +
        'stacked row-wise into 9 rows instead of joined column-wise.',
    };

    const { container } = render(
      <MergeWarningsBanner
        mergeWarnings={[warning]}
        mergeWarningsOpen
        setMergeWarningsOpen={vi.fn()}
        nodeLabelMap={{ 'merge-1': 'Merge' }}
        confirm={vi.fn()}
        chainSiblings={vi.fn()}
      />,
    );

    const detail = container.querySelectorAll('.pl-5')[0]?.textContent ?? '';
    expect(detail).toContain('different row counts (train)');
    expect(detail).toContain('5 vs 4');
    expect(detail).toContain('9');
    expect(detail).toContain('move that step after the merge');
    // The fan-in default is actively wrong here: nothing was kept column-wise
    // and rewiring as a chain would not fix a row-count mismatch.
    expect(screen.queryByText(/all columns from all branches are kept/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Chain instead/i)).not.toBeInTheDocument();
    expect(detail.match(/Merge/g) ?? []).toHaveLength(1);
    expect(detail).not.toContain('merge-1');
  });
});
