/** Show the bounded predicted schema only when no output measurement is selected. */
export function PredictedSchemaTable({ predicted }: {
  predicted: { columns: string[]; dtypes: Record<string, string> };
}) {
  return (
    <section className="min-w-0 space-y-2 border-t pt-4">
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
    </section>
  );
}
