import type { InspectionTable, NodeInspection } from '../../../core/types/nodeInspection';
import { tableIdentity } from './useInspectionSelection';

/** Match measured input only when the complete port/split layout agrees. */
function matchingInput(branch: NodeInspection, output: InspectionTable) {
  const inputShape = branch.input.tables.map(tableIdentity).sort((a, b) => a.localeCompare(b));
  const outputShape = branch.output.tables.map(tableIdentity).sort((a, b) => a.localeCompare(b));
  const sameShape = inputShape.length === outputShape.length && inputShape.every((key, index) => key === outputShape[index]);
  const matches = branch.input.tables.filter(input => tableIdentity(input) === tableIdentity(output));
  return sameShape && matches.length === 1 ? matches[0] : undefined;
}

/** Captured names can establish column changes only when the schema is complete. */
function hasCompleteSchema(table: InspectionTable): boolean {
  return table.columns.length === table.column_count && table.column_count <= 100;
}

/** Report name changes separately from row counts, which do not require a complete schema. */
function ColumnChanges({ input, output }: { input: InspectionTable; output: InspectionTable }) {
  if (!hasCompleteSchema(input) || !hasCompleteSchema(output)) {
    return <p className="text-muted-foreground">Column changes unavailable because the captured schema is truncated.</p>;
  }
  const inputColumns = new Set(input.columns.map(column => column.name));
  const outputColumns = new Set(output.columns.map(column => column.name));
  const added = [...outputColumns].filter(column => !inputColumns.has(column));
  const removed = [...inputColumns].filter(column => !outputColumns.has(column));

  return <div className="max-h-32 overflow-y-auto break-words text-muted-foreground">
    {added.length > 0 && <p>Added: {added.join(', ')}</p>}
    {removed.length > 0 && <p>Removed: {removed.join(', ')}</p>}
    {added.length === 0 && removed.length === 0 && <p>No columns added or removed.</p>}
  </div>;
}

/** Compare only matching measured tables; bounded samples cannot establish cell changes. */
export function MeasuredChanges({ branch, output }: { branch: NodeInspection; output: InspectionTable }) {
  if (branch.input.status !== 'available') return null;
  const input = matchingInput(branch, output);
  if (!input) return <section aria-label="Measured changes" className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
    Input and output use different table shapes; compare their samples separately.
  </section>;

  const rowChange = output.row_count - input.row_count;

  return <section aria-label="Measured changes" className="space-y-1 rounded-md border bg-muted/30 p-3 text-xs">
    <h3 className="font-semibold">Measured changes</h3>
    <p className="tabular-nums">Rows: {input.row_count.toLocaleString()} → {output.row_count.toLocaleString()} ({rowChange === 0
      ? 'unchanged' : `${Math.abs(rowChange).toLocaleString()} ${rowChange > 0 ? 'added' : 'removed'}`})</p>
    <p className="tabular-nums">Columns: {input.column_count.toLocaleString()} → {output.column_count.toLocaleString()}</p>
    <ColumnChanges input={input} output={output} />
  </section>;
}
