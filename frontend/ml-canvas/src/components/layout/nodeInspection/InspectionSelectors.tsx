import { useId } from 'react';
import type { InspectionSide, InspectionTable, NodeInspection } from '../../../core/types/nodeInspection';
import type { InspectionDirection, useInspectionSelection } from './useInspectionSelection';

/** Explain an unavailable side without confusing it with an available empty table list. */
function CapturedAvailability({ capturedSide, side, table }: {
  capturedSide: InspectionSide | undefined; side: InspectionDirection; table: InspectionTable | undefined;
}) {
  if (!capturedSide) return null;
  if (capturedSide.status === 'available') {
    return !table && <p className="text-xs text-muted-foreground">No tabular {side} was captured for this node.</p>;
  }
  const sideLabel = side === 'input' ? 'Input' : 'Output';
  const presentation = capturedSide.status === 'error'
    ? { role: 'alert' as const, style: 'border-destructive/40 text-destructive', label: 'inspection failed' }
    : { role: undefined, style: 'bg-muted/30 text-muted-foreground', label: 'unavailable' };
  return <div role={presentation.role} className={`rounded-md border p-3 text-xs ${presentation.style}`}>
    <p className="font-medium">{sideLabel} {presentation.label}</p>
    <p className="mt-1">{capturedSide.reason ?? `No tabular ${side} was captured for this node.`}</p>
  </div>;
}

/** Browse paths and bounded port/split choices without resetting the receipt selection. */
export function InspectionSelectors({ nodeId, runId, branches, side, selection }: {
  nodeId: string; runId: string | null; branches: NodeInspection[]; side: InspectionDirection;
  selection: ReturnType<typeof useInspectionSelection>;
}) {
  const id = useId();
  const { branch, capturedSide, tables, tableIndex, table, setSelection } = selection;
  if (!branch) return null;
  return (
    <div className="space-y-3">
      {runId && <p className="break-all text-[10px] text-muted-foreground">Preview run: {runId}</p>}
      {branches.length > 1 && <div className="space-y-1">
        <label htmlFor={`${id}-branch`} className="text-xs font-medium">Data path</label>
        <select id={`${id}-branch`} value={branch.branch_id}
          onChange={(event) => setSelection({ nodeId, runId, branchId: event.target.value })}
          className="w-full min-w-0 rounded-md border bg-background px-2 py-1.5 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
          {branches.map((item, index) => <option key={item.branch_id} value={item.branch_id}>
            {item.branch_label}{branches.some(other => other.branch_id !== item.branch_id && other.branch_label === item.branch_label) ? ` (${index + 1})` : ''}
          </option>)}
        </select>
      </div>}
      {tables.length > 0 && <div className="space-y-1">
        <label htmlFor={`${id}-table`} className="text-xs font-medium">Port and split</label>
        <select id={`${id}-table`} value={tableIndex}
          onChange={(event) => {
            const selected = tables[Number(event.target.value)];
            if (selected) setSelection({ nodeId, runId, branchId: branch.branch_id, port: selected.port, split: selected.split });
          }}
          className="w-full min-w-0 rounded-md border bg-background px-2 py-1.5 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
          {tables.map((item, index) => <option key={index} value={index}>{item.port} · {item.split ?? 'unsplit'}</option>)}
        </select>
      </div>}
      <CapturedAvailability capturedSide={capturedSide} side={side} table={table} />
    </div>
  );
}
