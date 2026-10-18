import type { EncodingConfig, EncodingSettingsProps } from './types';
import { ValidationField } from '../../../../components/shared/ValidationField';

/** Preserve exact level text and order, including commas and surrounding spaces. */
export function OnehotCategories({ config, onChange }: EncodingSettingsProps) {
  const fixed = config.categories !== undefined;
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-xs">
        <input type="checkbox" checked={fixed} onChange={event => onChange(event.target.checked
          ? { ...config, categories: Object.fromEntries(config.columns.map(column => [column, []])), max_categories: null, include_missing: false }
          : { ...config, categories: undefined, max_categories: config.max_categories ?? 20 })} />
        Use fixed categories
      </label>
      {fixed && <p className="text-[10px] text-muted-foreground">Select columns below, then enter one category per line. Order and spaces are preserved. Unknown or missing values follow Handle Unknown.</p>}
      {fixed && <ValidationField field="categories">{config.columns.map(column => (
        <label key={column} className="block space-y-1 text-xs">
          <span>Categories for {column}</span>
          <textarea className="w-full p-2 border rounded bg-background font-mono text-xs" rows={4}
            value={(config.categories?.[column] ?? []).join('\n')}
            onChange={event => onChange({ ...config, categories: { ...config.categories,
              [column]: event.target.value === '' ? [] : event.target.value.split('\n'),
            } })} />
        </label>
      ))}</ValidationField>}
      <label className="block space-y-1 text-xs">
        <span>Column names</span>
        <select className="w-full p-1 border rounded bg-background" value={config.column_names ?? 'sklearn'}
          onChange={event => onChange({ ...config, column_names: event.target.value as EncodingConfig['column_names'] })}>
          <option value="sklearn">Original category names</option>
          <option value="slug">Lowercase ASCII names</option>
        </select>
      </label>
    </div>
  );
}

/** Prevent incomplete fixed vocabularies from reaching the training request. */
export function fixedCategoriesValid(config: EncodingConfig): boolean {
  if (config.method !== 'onehot' || config.categories === undefined) return true;
  const categories = config.categories;
  return Object.keys(categories).length === config.columns.length && config.columns.every(column => {
    const levels = categories[column];
    return Array.isArray(levels) && levels.length > 0 && levels.every(level => typeof level === 'string') && new Set(levels).size === levels.length;
  });
}
