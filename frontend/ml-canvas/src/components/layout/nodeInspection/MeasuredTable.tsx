import type { InspectionTable } from '../../../core/types/nodeInspection';

/** Preserve null/boolean values and bound the text rendered in every sample cell. */
function formatCell(value: unknown): string {
  const text = value === null || value === undefined ? 'null'
    : typeof value === 'object' ? JSON.stringify(value) : String(value);
  return text.length > 200 ? `${text.slice(0, 200)}…` : text;
}

/** Keep empty-frame schema visible and put wide samples in their own scroll area. */
export function MeasuredTable({ table, side }: { table: InspectionTable; side: 'input' | 'output' }) {
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
