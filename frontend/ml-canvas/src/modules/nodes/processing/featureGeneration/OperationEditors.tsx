import { ValidationField } from '../../../../components/shared/ValidationField';
import { ColumnMultiSelect } from '../../shared/ColumnMultiSelect';
import { DATE_METHODS, DATE_METHOD_META } from './options';
import type { OperationEditorProps } from './types';

export function ArithmeticEditor({ op, idx, updateOperation, numericColumns }: OperationEditorProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
        {op.method === 'divide'
          ? 'Performs row-by-row division (Col A / Col B) for each record.'
          : 'Performs row-by-row arithmetic between two columns.'}
      </div>
      <ValidationField field={`operations.${idx}.input_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Column A (Left Operand)"
          aria-label={`Column A (Left Operand) for operation ${idx + 1}`}
          columns={numericColumns}
          selected={op.input_columns.slice(0, 1)}
          onChange={(cols) => { updateOperation(idx, { input_columns: cols }); }}
          single
        />
      </ValidationField>

      <div className="flex items-center gap-2">
         <div className="h-px bg-border flex-1"></div>
         <span className="text-lg font-bold text-muted-foreground bg-muted/20 w-8 h-8 rounded flex items-center justify-center">
          {op.method === 'add' ? '+' : op.method === 'subtract' ? '-' : op.method === 'multiply' ? '×' : '÷'}
        </span>
         <div className="h-px bg-border flex-1"></div>
      </div>

      <ValidationField field={`operations.${idx}.secondary_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Column B (Right Operand)"
          aria-label={`Column B (Right Operand) for operation ${idx + 1}`}
          columns={numericColumns}
          selected={op.secondary_columns?.slice(0, 1) || []}
          onChange={(cols) => { updateOperation(idx, { secondary_columns: cols }); }}
          single
        />
      </ValidationField>
    </div>
  );
}

export function RatioEditor({ op, idx, updateOperation, numericColumns }: OperationEditorProps) {
  return (
    <div className="space-y-3">
      <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
        Calculates aggregate ratio: <strong>Sum(Numerator) / Sum(Denominator)</strong>.
      </div>
      <ValidationField field={`operations.${idx}.input_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Numerator (Sum)"
          aria-label={`Numerator (Sum) for operation ${idx + 1}`}
          columns={numericColumns}
          selected={op.input_columns}
          onChange={(cols) => { updateOperation(idx, { input_columns: cols }); }}
        />
      </ValidationField>

      <div className="relative flex items-center justify-center">
        <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-dashed"></div></div>
        <span className="relative bg-card px-2 text-xs text-muted-foreground font-medium">Divided By</span>
      </div>

      <ValidationField field={`operations.${idx}.secondary_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Denominator (Sum)"
          aria-label={`Denominator (Sum) for operation ${idx + 1}`}
          columns={numericColumns}
          selected={op.secondary_columns || []}
          onChange={(cols) => { updateOperation(idx, { secondary_columns: cols }); }}
        />
      </ValidationField>
    </div>
  );
}

export function SimilarityEditor({ op, idx, updateOperation, stringColumns }: OperationEditorProps) {
  return (
    <div className="flex flex-col gap-3">
      <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20 space-y-1">
        <p>Calculates string similarity score (0-100) between two text columns.</p>
        <ul className="list-disc pl-3 space-y-0.5 opacity-80">
          <li><strong>Ratio:</strong> Strict character matching (Levenshtein). &quot;apple banana&quot; != &quot;banana apple&quot;.</li>
          <li><strong>Token Sort:</strong> Sorts words alphabetically then compares. &quot;apple banana&quot; == &quot;banana apple&quot;.</li>
          <li><strong>Token Set:</strong> Compares intersection of words. Handles duplicates/subsets well.</li>
        </ul>
      </div>
      <ValidationField field={`operations.${idx}.input_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="String A"
          aria-label={`String A for operation ${idx + 1}`}
          columns={stringColumns}
          selected={op.input_columns.slice(0, 1)}
          onChange={(cols) => updateOperation(idx, { input_columns: cols })}
          single
        />
      </ValidationField>

      <div className="flex items-center gap-2">
         <div className="h-px bg-border flex-1"></div>
         <span className="text-xs font-bold text-muted-foreground bg-muted/20 px-2 py-1 rounded">vs</span>
         <div className="h-px bg-border flex-1"></div>
      </div>

      <ValidationField field={`operations.${idx}.secondary_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="String B"
          aria-label={`String B for operation ${idx + 1}`}
          columns={stringColumns}
          selected={op.secondary_columns?.slice(0, 1) || []}
          onChange={(cols) => updateOperation(idx, { secondary_columns: cols })}
          single
        />
      </ValidationField>
    </div>
  );
}

export function GroupAggregationEditor({ op, idx, updateOperation, stringColumns, allColumns, numericColumns }: OperationEditorProps) {
  return (
    <div className="space-y-3">
      <div className="text-[10px] text-muted-foreground bg-muted/20 p-1.5 rounded border border-muted/20">
        Calculates aggregate statistics (e.g., Mean Salary) grouped by a categorical column (e.g., Department) and assigns it back to each row.
      </div>
      <ValidationField field={`operations.${idx}.input_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Group By (Categorical)"
          aria-label={`Group By (Categorical) for operation ${idx + 1}`}
          columns={stringColumns.length > 0 ? stringColumns : allColumns}
          selected={op.input_columns.slice(0, 1)}
          onChange={(cols) => updateOperation(idx, { input_columns: cols })}
          single
        />
      </ValidationField>

      <div className="relative flex items-center justify-center">
        <div className="absolute inset-0 flex items-center"><div className="w-full border-t border-dashed"></div></div>
        <span className="relative bg-card px-2 text-xs text-muted-foreground font-medium">Target Column</span>
      </div>

      <ValidationField field={`operations.${idx}.secondary_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Target (Numeric)"
          aria-label={`Target (Numeric) for operation ${idx + 1}`}
          columns={numericColumns}
          selected={op.secondary_columns?.slice(0, 1) || []}
          onChange={(cols) => updateOperation(idx, { secondary_columns: cols })}
          single
        />
      </ValidationField>
    </div>
  );
}

export function DateExtractionEditor({ op, idx, updateOperation, dateColumns, allColumns }: OperationEditorProps) {
  return (
    <div className="space-y-3">
      <ValidationField field={`operations.${idx}.input_columns`}>
        <ColumnMultiSelect
          variant="compact"
          label="Date Column"
          aria-label={`Date Column for operation ${idx + 1}`}
          columns={dateColumns.length > 0 ? dateColumns : allColumns}
          selected={op.input_columns.slice(0, 1)}
          onChange={(cols) => updateOperation(idx, { input_columns: cols })}
          single
        />
      </ValidationField>

      <div className="space-y-1.5">
        <span className="text-xs font-medium text-muted-foreground">Features to Extract</span>
        <div className="grid grid-cols-2 gap-2">
          {DATE_METHODS.map(method => {
            const meta = DATE_METHOD_META[method];
            const tooltip = meta ? `${meta.desc}\n→ output type: ${meta.type}` : undefined;
            return (
              <label
                key={method}
                title={tooltip}
                className="flex items-center gap-2 text-xs p-1.5 border rounded hover:bg-accent cursor-pointer transition-colors"
              >
                <input
                  type="checkbox"
                  className="rounded border-muted-foreground/40 text-primary focus:ring-1 focus:ring-primary"
                  checked={(op.datetime_features || []).includes(method)}
                  onChange={(e) => {
                    const current = op.datetime_features || [];
                    const newFeatures = e.target.checked
                      ? [...current, method]
                      : current.filter(f => f !== method);
                    updateOperation(idx, { datetime_features: newFeatures });
                  }}
                />
                <span className="capitalize">{method.replace(/_/g, ' ')}</span>
                {meta && (
                  <span
                    className={`ml-auto shrink-0 text-[9px] font-mono px-1 rounded ${
                      meta.type === 'string'
                        ? 'bg-purple-500/20 text-purple-400'
                        : 'bg-sky-500/20 text-sky-400'
                    }`}
                  >
                    {meta.type}
                  </span>
                )}
              </label>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/** Keep operation-specific controls mounted only for the selected operation type. */
export function OperationEditor(props: OperationEditorProps) {
  switch (props.op.operation_type) {
    case 'arithmetic': return <ArithmeticEditor {...props} />;
    case 'ratio': return <RatioEditor {...props} />;
    case 'similarity': return <SimilarityEditor {...props} />;
    case 'group_agg': return <GroupAggregationEditor {...props} />;
    case 'datetime_extract': return <DateExtractionEditor {...props} />;
    default: return null;
  }
}
