import { useId, useState } from 'react';
import { useNodeInspection } from '../../core/hooks/useNodeInspection';
import type { InspectionTable, NodeInspection } from '../../core/types/nodeInspection';

/** Preserve null/boolean values and bound the text rendered in every sample cell. */
function formatCell(value: unknown): string {
  const text = value === null || value === undefined ? 'null'
    : typeof value === 'object' ? JSON.stringify(value) : String(value);
  return text.length > 200 ? `${text.slice(0, 200)}…` : text;
}

/** Keep empty-frame schema visible and put wide samples in their own scroll area. */
function MeasuredTable({ table, side }: { table: InspectionTable; side: 'input' | 'output' }) {
  const columns = table.columns.slice(0, 100);
  const rows = table.rows.slice(0, 50);
  const truncated = table.truncated || rows.length < table.row_count || columns.length < table.column_count;
  return <div className="min-w-0 space-y-2">
    <div>
      <h3 className="text-sm font-semibold">Measured {side}</h3>
      <p className="text-xs tabular-nums">{table.row_count.toLocaleString()} rows × {table.column_count.toLocaleString()} columns</p>
    </div>
    <p className="text-xs text-muted-foreground">
      Showing {rows.length} of {table.row_count.toLocaleString()} rows and {columns.length} of {table.column_count.toLocaleString()} columns.
      {truncated && ' Truncated sample; some rows, columns or cell values are omitted.'}
    </p>
    {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- Keyboard users need focus to scroll the bounded sample. */}
    <div role="region" aria-label={`Measured ${side} sample scroll area`} tabIndex={0}
      className="max-h-80 max-w-full overflow-auto rounded-md border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
      <table aria-label={`Measured ${side} sample`} className="w-full border-collapse text-xs">
        <thead className="sticky top-0 z-10 bg-muted">
          <tr>{columns.map((column, index) => <th key={index} scope="col"
            className="max-w-[180px] border-b border-r px-2 py-2 text-left align-top last:border-r-0">
            <span className="block truncate font-medium" title={column.name}>{column.name}</span>{' '}
            <span className="block truncate text-[10px] font-normal text-muted-foreground" title={column.dtype}>{column.dtype}</span>
          </th>)}</tr>
        </thead>
        <tbody>{rows.map((row, index) => <tr key={index} className="border-b last:border-b-0 hover:bg-muted/30">
          {columns.map((column, columnIndex) => {
            const value = formatCell(row[column.name]);
            return <td key={columnIndex} title={value}
              className="max-w-[180px] truncate border-r px-2 py-1.5 font-mono text-[11px] last:border-r-0">{value}</td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>
    {table.row_count === 0 && <p className="text-xs text-muted-foreground">No rows in this preview table. The measured schema is shown above.</p>}
    {table.column_count === 0 && <p className="text-xs text-muted-foreground">No columns in this preview table.</p>}
  </div>;
}

/** Plain artifacts rename their port across a node; X and y remain distinct. */
function tableIdentity(table: Pick<InspectionTable, 'port' | 'split'>): string {
  return JSON.stringify([table.split, table.port === 'X' || table.port === 'y' ? table.port : 'data']);
}

/** Compare only matching measured tables; bounded samples cannot establish cell changes. */
function MeasuredChanges({ branch, output }: { branch: NodeInspection; output: InspectionTable }) {
  if (branch.input.status !== 'available') return null;
  const inputShape = branch.input.tables.map(tableIdentity).sort();
  const outputShape = branch.output.tables.map(tableIdentity).sort();
  const sameShape = inputShape.length === outputShape.length && inputShape.every((key, index) => key === outputShape[index]);
  const matches = branch.input.tables.filter(input => tableIdentity(input) === tableIdentity(output));
  const input = sameShape && matches.length === 1 ? matches[0] : undefined;
  if (!input) return <section aria-label="Measured changes" className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
    Input and output use different table shapes; compare their samples separately.
  </section>;

  const rowChange = output.row_count - input.row_count;
  const completeSchema = input.columns.length === input.column_count && output.columns.length === output.column_count
    && input.column_count <= 100 && output.column_count <= 100;
  const inputColumns = new Set(input.columns.map(column => column.name));
  const outputColumns = new Set(output.columns.map(column => column.name));
  const added = completeSchema ? [...outputColumns].filter(column => !inputColumns.has(column)) : [];
  const removed = completeSchema ? [...inputColumns].filter(column => !outputColumns.has(column)) : [];

  return <section aria-label="Measured changes" className="space-y-1 rounded-md border bg-muted/30 p-3 text-xs">
    <h3 className="font-semibold">Measured changes</h3>
    <p className="tabular-nums">Rows: {input.row_count.toLocaleString()} → {output.row_count.toLocaleString()} ({rowChange === 0
      ? 'unchanged' : `${Math.abs(rowChange).toLocaleString()} ${rowChange > 0 ? 'added' : 'removed'}`})</p>
    <p className="tabular-nums">Columns: {input.column_count.toLocaleString()} → {output.column_count.toLocaleString()}</p>
    {completeSchema ? <div className="max-h-32 overflow-y-auto break-words text-muted-foreground">
      {added.length > 0 && <p>Added: {added.join(', ')}</p>}
      {removed.length > 0 && <p>Removed: {removed.join(', ')}</p>}
      {added.length === 0 && removed.length === 0 && <p>No columns added or removed.</p>}
    </div> : <p className="text-muted-foreground">Column changes unavailable because the captured schema is truncated.</p>}
  </section>;
}

/** Inspect a selected node's captured input or output from data preview. */
export function NodeInspectionPanel({ nodeId, side }: { nodeId: string; side: 'input' | 'output' }) {
  const { branches, runId, isStale, isLoading, error, predictedSchema, blockReason } = useNodeInspection(nodeId);
  const id = useId();
  const [selection, setSelection] = useState<{
    nodeId: string; runId: string | null; branchId: string; port?: string; split?: string | null;
  } | null>(null);
  // Resolve a new receipt synchronously without remounting a focused control.
  const current = selection?.nodeId === nodeId && selection.runId === runId ? selection : null;
  const branch = branches.find(item => item.branch_id === current?.branchId) ?? branches[0];
  const capturedSide = branch?.[side];
  const tables = capturedSide?.status === 'available' ? capturedSide.tables.slice(0, 6) : [];
  const selectedIdentity = current?.branchId === branch?.branch_id && current?.port !== undefined && current.split !== undefined
    ? tableIdentity({ port: current.port, split: current.split }) : null;
  const exactIndex = tables.findIndex(item => selectedIdentity !== null && item.port === current?.port && item.split === current?.split);
  const matchingTables = tables.filter(item => tableIdentity(item) === selectedIdentity);
  const tableIndex = exactIndex >= 0 ? exactIndex : matchingTables.length === 1 ? tables.indexOf(matchingTables[0]!) : 0;
  const table = tables[tableIndex];
  const sideLabel = side === 'input' ? 'Input' : 'Output';
  const predicted = side === 'output' && !table ? predictedSchema : null;

  return <div className="min-w-0 space-y-4">
    <div className="space-y-2">
      {blockReason && <p className="text-xs text-muted-foreground">{blockReason}</p>}
      <p className="text-xs leading-relaxed text-muted-foreground">
        Preview uses up to 1,000 source rows. Counts describe data at this node within that preview.
      </p>
    </div>

    <div role="status" aria-live="polite" className="space-y-2 text-xs">
      {isLoading && <p className="text-muted-foreground">Running preview…</p>}
      {isStale && <p className="rounded-md border border-amber-300 bg-amber-50 p-2 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200">
        <strong>Stale preview.</strong> The graph changed after this capture. Run Preview data in the toolbar to measure the current settings.
      </p>}
      {!capturedSide && !isLoading && <p className="rounded-md border bg-muted/30 p-3 text-muted-foreground">
        No measured {side} yet. Run Preview data in the toolbar to inspect this node.
      </p>}
    </div>
    {error && <p role="alert" className="rounded-md border border-destructive/40 p-2 text-xs text-destructive">{error}</p>}

    {branch && <div className="space-y-3">
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
      {capturedSide && capturedSide.status !== 'available' && <div role={capturedSide.status === 'error' ? 'alert' : undefined}
        className={`rounded-md border p-3 text-xs ${capturedSide.status === 'error' ? 'border-destructive/40 text-destructive' : 'bg-muted/30 text-muted-foreground'}`}>
        <p className="font-medium">{sideLabel} {capturedSide.status === 'error' ? 'inspection failed' : 'unavailable'}</p>
        <p className="mt-1">{capturedSide.reason ?? `No tabular ${side} was captured for this node.`}</p>
      </div>}
      {capturedSide?.status === 'available' && !table && <p className="text-xs text-muted-foreground">No tabular {side} was captured for this node.</p>}
    </div>}

    {table && <MeasuredTable table={table} side={side} />}
    {table && branch && side === 'output' && <MeasuredChanges branch={branch} output={table} />}

    {predicted && <section className="min-w-0 space-y-2 border-t pt-4">
      <h3 className="text-sm font-semibold">Predicted output schema</h3>
      <p className="text-xs text-muted-foreground">Based on current settings. This prediction contains no measured rows.</p>
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- Keyboard users need focus to scroll the bounded schema. */}
      <div role="region" aria-label="Predicted output schema scroll area" tabIndex={0}
        className="max-h-64 overflow-auto rounded-md border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary">
        <table aria-label="Predicted output schema" className="w-full text-xs">
          <thead className="sticky top-0 bg-muted"><tr>
            <th scope="col" className="px-2 py-1.5 text-left font-medium">Column</th>
            <th scope="col" className="px-2 py-1.5 text-left font-medium">Dtype</th>
          </tr></thead>
          <tbody>{predicted.columns.slice(0, 100).map((column, index) => <tr key={index} className="border-t">
            <td className="max-w-[180px] truncate px-2 py-1.5" title={column}>{column}</td>
            <td className="max-w-[100px] truncate px-2 py-1.5 text-muted-foreground">{predicted.dtypes[column] ?? 'unknown'}</td>
          </tr>)}</tbody>
        </table>
      </div>
      {predicted.columns.length === 0 && <p className="text-xs text-muted-foreground">No output columns predicted.</p>}
      {predicted.columns.length > 100 && <p className="text-xs text-muted-foreground">Showing 100 of {predicted.columns.length} predicted columns.</p>}
    </section>}
  </div>;
}
