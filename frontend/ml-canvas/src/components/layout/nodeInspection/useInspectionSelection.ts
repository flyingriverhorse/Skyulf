import { useState } from 'react';
import type { InspectionTable, NodeInspection } from '../../../core/types/nodeInspection';

export type InspectionDirection = 'input' | 'output';

interface InspectionSelection {
  nodeId: string;
  runId: string | null;
  branchId: string;
  port?: string;
  split?: string | null;
}

/** Plain artifacts rename their port across a node; X and y remain distinct. */
export function tableIdentity(table: Pick<InspectionTable, 'port' | 'split'>): string {
  return JSON.stringify([table.split, table.port === 'X' || table.port === 'y' ? table.port : 'data']);
}

/** Only the selected branch's complete port/split identity can carry across sides. */
function selectedIdentity(current: InspectionSelection | null, branch: NodeInspection | undefined): string | null {
  if (!current || current.branchId !== branch?.branch_id || current.port === undefined || current.split === undefined) return null;
  return tableIdentity({ port: current.port, split: current.split });
}

/** Prefer the exact port, then a unique plain-table match, then the first table. */
function selectedTableIndex(tables: InspectionTable[], current: InspectionSelection | null, branch: NodeInspection | undefined): number {
  const identity = selectedIdentity(current, branch);
  const exactIndex = tables.findIndex(item => identity !== null && item.port === current?.port && item.split === current?.split);
  const matchingTables = tables.filter(item => tableIdentity(item) === identity);
  return exactIndex >= 0 ? exactIndex : matchingTables.length === 1 ? tables.indexOf(matchingTables[0]!) : 0;
}

/** Resolve each receipt synchronously so resetting selection keeps focused controls mounted. */
export function useInspectionSelection(nodeId: string, runId: string | null, branches: NodeInspection[], side: InspectionDirection) {
  const [selection, setSelection] = useState<InspectionSelection | null>(null);
  const current = selection?.nodeId === nodeId && selection.runId === runId ? selection : null;
  const branch = branches.find(item => item.branch_id === current?.branchId) ?? branches[0];
  const capturedSide = branch?.[side];
  const tables = capturedSide?.status === 'available' ? capturedSide.tables.slice(0, 6) : [];
  const tableIndex = selectedTableIndex(tables, current, branch);
  return { branch, capturedSide, tables, tableIndex, table: tables[tableIndex], setSelection };
}
