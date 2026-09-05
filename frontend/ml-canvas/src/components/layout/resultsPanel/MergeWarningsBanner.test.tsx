import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { MergeWarning } from '../../../core/api/client';
import { MergeWarningsBanner } from './MergeWarningsBanner';

describe('MergeWarningsBanner', () => {
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
