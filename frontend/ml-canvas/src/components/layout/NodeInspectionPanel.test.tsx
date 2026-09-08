import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NodeInspectionPanel } from './NodeInspectionPanel';
import { useNodeInspection } from '../../core/hooks/useNodeInspection';
import type { InspectionTable, NodeInspection } from '../../core/types/nodeInspection';

vi.mock('../../core/hooks/useNodeInspection');

const sample: InspectionTable = {
  port: 'X', split: null, row_count: 800, column_count: 2,
  columns: [{ name: 'score', dtype: 'float64' }, { name: 'enabled', dtype: 'bool' }],
  rows: [{ score: 42, enabled: false }, { score: null, enabled: true }], truncated: true,
};

/** Separate before/after samples catch accidental substitution of the other side. */
function branch(branchId = 'path-a', label = 'Scaler'): NodeInspection {
  return {
    node_id: 'selected', branch_id: branchId, branch_label: label,
    input: { status: 'available', reason: null, tables: [{ ...sample, rows: [{ score: 420, enabled: true }] }] },
    output: { status: 'available', reason: null, tables: [sample] },
  };
}

let inspection: ReturnType<typeof useNodeInspection>;

describe('NodeInspectionPanel', () => {
  beforeEach(() => {
    inspection = {
      branches: [branch()], runId: 'preview-run-1', isStale: false, isLoading: false,
      error: null, predictedSchema: null, blockReason: null,
    };
    vi.mocked(useNodeInspection).mockImplementation(() => inspection);
  });

  it('shows measured shape, typed columns and the selected side within preview scope', () => {
    // Counts belong to the preview data, and output must never be replaced with input.
    inspection.predictedSchema = { columns: ['predicted_only'], dtypes: { predicted_only: 'string' } };
    const { rerender } = render(<NodeInspectionPanel nodeId="selected" side="output" />);
    const table = screen.getByRole('table', { name: 'Measured output sample' });
    expect(screen.getByText(/^800 rows/)).toHaveTextContent(/2 columns/);
    expect(screen.getByText(/up to 1,000 source rows/i)).toBeVisible();
    expect(within(table).getByRole('columnheader', { name: /score float64/ })).toBeVisible();
    expect(within(table).getByRole('cell', { name: '42' })).toBeVisible();
    expect(within(table).getByRole('cell', { name: 'false' })).toBeVisible();
    expect(within(table).getByRole('cell', { name: 'null' })).toBeVisible();
    expect(screen.queryByText('predicted_only')).not.toBeInTheDocument();
    rerender(<NodeInspectionPanel nodeId="selected" side="input" />);
    expect(within(screen.getByRole('table', { name: 'Measured input sample' })).getByRole('cell', { name: '420' })).toBeVisible();
  });

  it('keeps measured schema when a transformation produces zero rows', () => {
    // Empty frames still communicate which columns survived the transformation.
    inspection.branches[0]!.output.tables = [{ ...sample, row_count: 0, rows: [], truncated: false }];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByText(/^0 rows/)).toHaveTextContent(/2 columns/);
    expect(screen.getByRole('columnheader', { name: /score float64/ })).toBeVisible();
    expect(screen.getByText(/No rows in this preview table/)).toBeVisible();
  });

  it('distinguishes repeated branch labels and selects each resolved port and split', () => {
    // Branch identity and X/y splits must select the corresponding captured sample.
    const second = branch('path-b', 'Shared');
    second.output.tables = [
      { ...sample, port: 'X', split: 'train', rows: [{ score: 12 }] },
      { ...sample, port: 'y', split: 'test', rows: [{ score: 99 }] },
    ];
    inspection.branches = [branch('path-a', 'Shared'), second];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    const branches = screen.getByRole('combobox', { name: 'Data path' });
    expect(within(branches).getAllByRole('option').map(option => option.textContent)).toEqual(['Shared (1)', 'Shared (2)']);
    fireEvent.change(branches, { target: { value: 'path-b' } });
    const tables = screen.getByRole('combobox', { name: 'Port and split' });
    expect(tables).toHaveDisplayValue('X · train');
    expect(screen.getByRole('cell', { name: '12' })).toBeVisible();
    fireEvent.change(tables, { target: { value: '1' } });
    expect(tables).toHaveDisplayValue('y · test');
    expect(screen.getByRole('cell', { name: '99' })).toBeVisible();
    fireEvent.change(branches, { target: { value: 'path-a' } });
    expect(screen.getByRole('cell', { name: '42' })).toBeVisible();
  });

  it('bounds displayed rows, columns and long cell values while retaining actual counts', () => {
    // A large response must not expand the side panel into an unbounded grid.
    const columns = Array.from({ length: 120 }, (_, index) => ({ name: `column_${index}`, dtype: 'string' }));
    const row = Object.fromEntries(columns.map(column => [column.name, 'x'.repeat(500)]));
    inspection.branches[0]!.output.tables = [{ ...sample, row_count: 900, column_count: 120, columns,
      rows: Array.from({ length: 70 }, () => row) }];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    const table = screen.getByRole('table', { name: 'Measured output sample' });
    expect(within(table).getAllByRole('row')).toHaveLength(51);
    expect(within(table).getAllByRole('columnheader')).toHaveLength(100);
    expect(within(table).getAllByRole('cell')[0]!.textContent!.length).toBeLessThanOrEqual(201);
    expect(screen.getByText(/^900 rows/)).toHaveTextContent(/120 columns/);
    expect(screen.getByText(/Showing 50 of 900 rows/)).toHaveTextContent(/100 of 120 columns/);
    expect(screen.getByText(/Truncated/)).toBeVisible();
  });

  it('keeps branch and split identity when switching between output and input', () => {
    // Comparing one branch must not silently select another branch or target table.
    const second = branch('path-b', 'Second path');
    second.output.tables = [
      { ...sample, port: 'X', split: 'train', rows: [{ score: 12 }] },
      { ...sample, port: 'y', split: 'test', rows: [{ score: 99 }] },
    ];
    second.input.tables = [
      { ...sample, port: 'y', split: 'test', rows: [{ score: 990 }] },
      { ...sample, port: 'X', split: 'train', rows: [{ score: 120 }] },
    ];
    inspection.branches = [branch(), second];
    const { rerender } = render(<NodeInspectionPanel nodeId="selected" side="output" />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Data path' }), { target: { value: 'path-b' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Port and split' }), { target: { value: '1' } });
    rerender(<NodeInspectionPanel nodeId="selected" side="input" />);
    expect(screen.getByRole('combobox', { name: 'Data path' })).toHaveValue('path-b');
    expect(screen.getByRole('combobox', { name: 'Port and split' })).toHaveDisplayValue('y · test');
    expect(screen.getByRole('cell', { name: '990' })).toBeVisible();
  });

  it('summarizes measured row and column changes for a single plain table', () => {
    // Changes must use full measured shape/schema rather than the bounded sample rows.
    inspection.branches[0]!.input.tables = [{ ...sample, port: 'data_in', row_count: 100,
      columns: [{ name: 'score', dtype: 'float64' }, { name: 'old_column', dtype: 'string' }] }];
    inspection.branches[0]!.output.tables = [{ ...sample, port: 'data_out', row_count: 75,
      columns: [{ name: 'score', dtype: 'float64' }, { name: 'new_column', dtype: 'bool' }] }];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    const changes = screen.getByRole('region', { name: 'Measured changes' });
    expect(changes).toHaveTextContent('Rows: 100 → 75 (25 removed)');
    expect(changes).toHaveTextContent('Columns: 2 → 2');
    expect(changes).toHaveTextContent('Added: new_column');
    expect(changes).toHaveTextContent('Removed: old_column');
  });

  it('compares matching X and y splits individually without adding their row counts', () => {
    // Features and targets describe the same observations and must never be summed.
    inspection.branches[0]!.input.tables = [
      { ...sample, port: 'X', split: 'train', row_count: 100 },
      { ...sample, port: 'y', split: 'train', row_count: 100 },
    ];
    inspection.branches[0]!.output.tables = [
      { ...sample, port: 'X', split: 'train', row_count: 90 },
      { ...sample, port: 'y', split: 'train', row_count: 90 },
    ];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent('Rows: 100 → 90 (10 removed)');
    fireEvent.change(screen.getByRole('combobox', { name: 'Port and split' }), { target: { value: '1' } });
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent('Rows: 100 → 90 (10 removed)');
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent('No columns added or removed.');
  });

  it('matches plain split frames across input and output port names', () => {
    // Wire ports change from input to output even when train/test identities are preserved.
    inspection.branches[0]!.input.tables = [
      { ...sample, port: 'input', split: 'train', row_count: 100, rows: [{ score: 111 }] },
      { ...sample, port: 'input', split: 'test', row_count: 40, rows: [{ score: 444 }] },
    ];
    inspection.branches[0]!.output.tables = [
      { ...sample, port: 'output', split: 'train', row_count: 90, rows: [{ score: 11 }] },
      { ...sample, port: 'output', split: 'test', row_count: 35, rows: [{ score: 44 }] },
    ];
    const { rerender } = render(<NodeInspectionPanel nodeId="selected" side="output" />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Port and split' }), { target: { value: '1' } });
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent('Rows: 40 → 35 (5 removed)');
    rerender(<NodeInspectionPanel nodeId="selected" side="input" />);
    expect(screen.getByRole('combobox', { name: 'Port and split' })).toHaveDisplayValue('input · test');
    expect(screen.getByRole('cell', { name: '444' })).toBeVisible();
    rerender(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByRole('combobox', { name: 'Port and split' })).toHaveDisplayValue('output · test');
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent('Rows: 40 → 35 (5 removed)');
  });

  it('withholds comparisons when plain split matches are ambiguous or split topology changes', () => {
    // One matching split does not establish that two different artifact layouts correspond.
    inspection.branches[0]!.input.tables = [
      { ...sample, port: 'input', split: 'train' },
      { ...sample, port: 'input', split: 'test' },
    ];
    inspection.branches[0]!.output.tables = [{ ...sample, port: 'output', split: 'train' }];
    const { rerender } = render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent(/different table shapes/);
    inspection.branches[0]!.input.tables = [
      { ...sample, port: 'first', split: 'train' },
      { ...sample, port: 'second', split: 'train' },
    ];
    inspection.branches[0]!.output.tables = [
      { ...sample, port: 'first_out', split: 'train' },
      { ...sample, port: 'second_out', split: 'train' },
    ];
    rerender(<NodeInspectionPanel nodeId="selected" side="output" />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Port and split' }), { target: { value: '1' } });
    expect(screen.getByRole('combobox', { name: 'Port and split' })).toHaveDisplayValue('second_out · train');
    expect(screen.getByRole('region', { name: 'Measured changes' })).not.toHaveTextContent(/Rows:/);
    expect(screen.getByRole('region', { name: 'Measured changes' })).toHaveTextContent(/different table shapes/);
  });

  it('explains incompatible split shapes instead of inventing a comparison', () => {
    // Splitting a dataset does not mean the rows in the other splits were removed.
    inspection.branches[0]!.input.tables = [{ ...sample, port: 'data_in', split: null }];
    inspection.branches[0]!.output.tables = [{ ...sample, port: 'X', split: 'train', row_count: 600 }];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    const changes = screen.getByRole('region', { name: 'Measured changes' });
    expect(changes).toHaveTextContent('Input and output use different table shapes; compare their samples separately.');
    expect(changes).not.toHaveTextContent(/Rows:/);
  });

  it('keeps row changes but withholds column changes when captured schemas are truncated', () => {
    // Missing captured column names cannot establish which columns were removed.
    inspection.branches[0]!.input.tables = [{ ...sample, row_count: 80, column_count: 105,
      columns: [{ name: 'old_column', dtype: 'string' }] }];
    inspection.branches[0]!.output.tables = [{ ...sample, row_count: 90, column_count: 110,
      columns: [{ name: 'new_column', dtype: 'string' }] }];
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    const changes = screen.getByRole('region', { name: 'Measured changes' });
    expect(changes).toHaveTextContent('Rows: 80 → 90 (10 added)');
    expect(changes).toHaveTextContent('Columns: 105 → 110');
    expect(changes).toHaveTextContent(/schema is truncated/);
    expect(changes).not.toHaveTextContent(/Added:|Removed:/);
  });

  it('keeps captured rows visible while a shared refresh is pending', () => {
    // The inspector observes toolbar progress without providing another execution button.
    inspection.isLoading = true;
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.queryByRole('button', { name: /preview/i })).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent(/Running preview/);
    expect(screen.getByRole('cell', { name: '42' })).toBeVisible();
  });

  it('announces stale data and a failed refresh without discarding its receipt', () => {
    // Failed refreshes must not make an old measurement look current or disappear.
    inspection.isStale = true;
    inspection.error = 'Preview request timed out';
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByRole('status')).toHaveTextContent(/Stale preview/);
    expect(screen.getByRole('alert')).toHaveTextContent('Preview request timed out');
    expect(screen.getByText(/preview-run-1/)).toBeVisible();
    expect(screen.getByRole('cell', { name: '42' })).toBeVisible();
  });

  it('labels predicted output separately and never invents predicted input', () => {
    // Schema predictions have no measured rows and must not be presented as input.
    inspection.branches = [];
    inspection.runId = null;
    inspection.predictedSchema = { columns: ['forecast'], dtypes: { forecast: 'float64' } };
    const { rerender } = render(<NodeInspectionPanel nodeId="selected" side="input" />);
    expect(screen.getByText(/No measured input yet/)).toBeVisible();
    expect(screen.queryByText('forecast')).not.toBeInTheDocument();
    rerender(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByText(/No measured output yet/)).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Predicted output schema' })).toBeVisible();
    expect(screen.getByRole('table', { name: 'Predicted output schema' })).toHaveTextContent('forecast');
    expect(screen.queryByRole('table', { name: 'Measured output sample' })).not.toBeInTheDocument();
  });

  it.each(['unavailable', 'error'] as const)('explains an %s side without showing unrelated samples', (status) => {
    // Skipped or failed nodes must expose their capture reason instead of a blank table.
    inspection.branches[0]!.output = { status, reason: 'Model execution is skipped in data preview.', tables: [] };
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.getByText('Model execution is skipped in data preview.')).toBeVisible();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.getByText(status === 'error' ? 'Output inspection failed' : 'Output unavailable')).toBeVisible();
  });

  it('explains blocked toolbar previews without a disabled inspector action', () => {
    // Read-only viewers can inspect captured data without a redundant refresh control.
    inspection.blockReason = 'This canvas is read-only.';
    render(<NodeInspectionPanel nodeId="selected" side="input" />);
    expect(screen.getByText('This canvas is read-only.')).toBeVisible();
    expect(screen.queryByRole('button', { name: /preview/i })).not.toBeInTheDocument();
  });

  it('directs empty and stale states to the toolbar without another run button', () => {
    // Preview data is the single execution action for every node inspection.
    inspection.branches = [];
    inspection.isStale = true;
    render(<NodeInspectionPanel nodeId="selected" side="input" />);
    expect(screen.getByText(/No measured input yet/)).toHaveTextContent(/Preview data.*toolbar/i);
    expect(screen.getByRole('status')).toHaveTextContent(/Preview data.*toolbar/i);
    expect(screen.queryByRole('button', { name: /preview/i })).not.toBeInTheDocument();
  });

  it('hides the path selector when this node has a single result', () => {
    // A source or shared preprocessing step must not force a meaningless branch choice.
    render(<NodeInspectionPanel nodeId="selected" side="output" />);
    expect(screen.queryByRole('combobox', { name: /Branch|Data path/ })).not.toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '42' })).toBeVisible();
  });
});
