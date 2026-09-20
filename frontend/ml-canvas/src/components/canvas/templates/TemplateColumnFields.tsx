import type { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import type { PipelineTemplate } from '../../../core/templates/pipelineTemplates';

interface Props {
  id: string; category: PipelineTemplate['category']; schema: ReturnType<typeof useDatasetSchema>;
  target: string; text: string; disabled: boolean;
  onTarget: (value: string) => void; onText: (value: string) => void;
}

/** Use current schema choices and keep text features separate from labels. */
export function TemplateColumnFields({ id, category, schema, target, text, disabled, onTarget, onText }: Props) {
  if (schema.isError) return <p role="alert" className="text-sm text-destructive">Could not load columns. <button type="button" className="underline" onClick={() => { void schema.refetch(); }}>Retry</button></p>;
  if (schema.isLoading) return <p role="status" className="text-sm text-muted-foreground">Loading columns…</p>;
  if (category === 'clustering') return <p className="rounded-md bg-muted p-3 text-sm">No target needed. Select useful numeric features and review the number of groups in Canvas.</p>;
  const columns = Object.values(schema.data?.columns ?? {});
  const textColumns = columns.filter(column => /object|string|category|text|utf8/i.test(String(column.dtype)) && column.name !== target);
  return <>
    <label htmlFor={`${id}-target`} className="block text-sm font-medium">Target column</label>
    <select id={`${id}-target`} value={target} disabled={disabled} onChange={event => onTarget(event.target.value)} className="w-full rounded-md border bg-background p-2.5 text-sm">
      <option value="">Choose what to predict</option>
      {columns.map(column => <option key={column.name} value={column.name}>{column.name} ({column.dtype})</option>)}
    </select>
    {category === 'regression' && <p className="text-xs text-muted-foreground">Choose a numeric target for regression.</p>}
    {category === 'text' && <>
      <label htmlFor={`${id}-text`} className="block text-sm font-medium">Text column</label>
      <select id={`${id}-text`} value={text} disabled={disabled} onChange={event => onText(event.target.value)} className="w-full rounded-md border bg-background p-2.5 text-sm">
        <option value="">Choose the text to classify</option>
        {textColumns.map(column => <option key={column.name} value={column.name}>{column.name}</option>)}
      </select>
      <p className="text-xs text-muted-foreground">Used by both text cleaning and TF-IDF. The target is excluded.</p>
    </>}
  </>;
}
