import { describe, expect, it } from 'vitest';
import type { NodeInspection } from '../types/nodeInspection';
import { groupNodeInspections } from './nodeInspectionPaths';

/** Equal samples may belong to separate data paths or repeated model-bound executions. */
function capture(branchId: string, pathId?: string): NodeInspection {
  return { node_id: 'scale', branch_id: branchId, branch_label: `Model ${branchId}`,
    path_id: pathId ?? null, path_label: 'Dataset → Scaler',
    input: { status: 'available', reason: null, tables: [{ port: 'input', split: null,
      row_count: 100, column_count: 1, columns: [{ name: 'x', dtype: 'float64' }],
      rows: [{ x: 1 }], truncated: true }] },
    output: { status: 'available', reason: null, tables: [] },
  };
}

describe('node-local inspection paths', () => {
  it('collapses repeated captures of the same data path while retaining raw receipts', () => {
    // Downstream model choices must not create identical options at an upstream node.
    const entries = [capture('a', 'shared'), capture('b', 'shared')];
    expect(groupNodeInspections(entries)).toEqual([{ ...entries[0], branch_label: 'Dataset → Scaler' }]);
    expect(entries).toHaveLength(2);
    expect(entries[0]!.branch_label).toBe('Model a');
  });

  it('keeps different upstream paths even when bounded samples happen to match', () => {
    // Equal first rows do not establish that differently processed datasets are equal.
    expect(groupNodeInspections([capture('a', 'left'), capture('b', 'right')])).toHaveLength(2);
  });

  it('keeps legacy captures without path provenance separate', () => {
    // Missing provenance must never cause unrelated branches to be silently combined.
    expect(groupNodeInspections([capture('a'), capture('b')])).toHaveLength(2);
  });

  it.each(['input', 'output'] as const)('retains differing %s measurements on repeated paths', side => {
    // Repeated execution may differ, so one sample must not hide another execution's error.
    const first = capture('a', 'shared');
    const second = capture('b', 'shared');
    second[side] = { status: 'error', reason: 'Conversion failed', tables: [] };
    expect(groupNodeInspections([first, second])).toHaveLength(2);
  });
});
