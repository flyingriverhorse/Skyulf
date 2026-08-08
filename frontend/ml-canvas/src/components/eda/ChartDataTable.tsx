import React, { useState } from 'react';
import { Table2, Download } from 'lucide-react';

export interface ChartDataTableColumn {
  /** Key used to look up the cell value on each row object. */
  key: string;
  /** Header text shown for this column. */
  label: string;
}

interface ChartDataTableProps {
  /** Column definitions, in display order. */
  columns: ChartDataTableColumn[];
  /** Row objects; each must have a value for every column `key`. */
  rows: Array<Record<string, string | number | null | undefined>>;
  /** Base filename (without extension) used for the CSV download. */
  filename: string;
  /** Short label identifying what this table represents, e.g. "Bivariate scatter data". */
  caption: string;
  /** Collapsed by default; the toggle button always renders so the table
   * alternative is discoverable without reading the chart. */
  defaultOpen?: boolean;
}

const escapeCsvCell = (value: string | number | null | undefined): string => {
  const text = value === null || value === undefined ? '' : String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
};

/**
 * Renders a collapsible, sortable-free HTML table that carries the same
 * information as a chart, plus a CSV download. This is the "text/table
 * alternative" required whenever a chart encodes meaning in color, position,
 * or truncated axes (see UX finding DAT-007) — it must never depend on
 * hovering the chart itself.
 */
export const ChartDataTable: React.FC<ChartDataTableProps> = ({
  columns,
  rows,
  filename,
  caption,
  defaultOpen = false,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  const downloadCsv = () => {
    const header = columns.map((c) => escapeCsvCell(c.label)).join(',');
    const lines = rows.map((row) => columns.map((c) => escapeCsvCell(row[c.key])).join(','));
    const csv = [header, ...lines].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen((prev) => !prev)}
          aria-expanded={open}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border border-gray-300 dark:border-gray-600 rounded-md px-2.5 py-1.5"
        >
          <Table2 className="w-3.5 h-3.5" aria-hidden="true" />
          {open ? 'Hide data table' : 'View data table'}
        </button>
        {rows.length > 0 && (
          <button
            type="button"
            onClick={downloadCsv}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white border border-gray-300 dark:border-gray-600 rounded-md px-2.5 py-1.5"
          >
            <Download className="w-3.5 h-3.5" aria-hidden="true" />
            Download CSV
          </button>
        )}
      </div>
      {open && (
        <div
          role="region"
          aria-label={caption}
          className="mt-2 max-h-80 overflow-auto border border-gray-200 dark:border-gray-700 rounded-md"
        >
          <table className="min-w-full text-xs">
            <caption className="sr-only">{caption}</caption>
            <thead className="bg-gray-50 dark:bg-gray-900 sticky top-0">
              <tr>
                {columns.map((col) => (
                  <th
                    key={col.key}
                    scope="col"
                    className="px-3 py-2 text-left font-semibold text-gray-700 dark:text-gray-300 whitespace-nowrap"
                  >
                    {col.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={columns.length} className="px-3 py-3 text-gray-500 dark:text-gray-400 italic">
                    No rows to display.
                  </td>
                </tr>
              ) : (
                rows.map((row, i) => (
                  <tr key={i} className="text-gray-700 dark:text-gray-300">
                    {columns.map((col) => (
                      <td key={col.key} className="px-3 py-1.5 whitespace-nowrap">
                        {row[col.key] ?? '—'}
                      </td>
                    ))}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
