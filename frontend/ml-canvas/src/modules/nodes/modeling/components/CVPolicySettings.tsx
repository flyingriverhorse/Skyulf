import { requiresExplicitCVTimeColumn } from '../../../../core/utils/cvPolicy';
import { ValidationField } from '../../../../components/shared/ValidationField';
import { numericDraft, numericInputValue } from '../../../../core/utils/numericValidation';

export interface CVPolicyConfig {
  cv_type: string;
  cv_nested_type?: string;
  cv_group_column?: string;
  cv_time_column?: string;
  cv_gap?: number;
  cv_test_size?: number | null;
  cv_max_train_size?: number | null;
  cv_shuffle: boolean;
}

type Props = { config: CVPolicyConfig; update: (patch: Partial<CVPolicyConfig>) => void; fieldId: string; columns: { name: string; dtype?: unknown }[] };
const controlClass = 'w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100';

/** Temporal policies cannot shuffle chronology. */
export function temporalPolicy(config: CVPolicyConfig): boolean {
  return (config.cv_type === 'nested_cv' ? config.cv_nested_type : config.cv_type) === 'time_series_split';
}

/** Resolve the next policy before clearing settings whose controls become inapplicable. */
export function policyChange(field: 'cv_type' | 'cv_nested_type', value: string, config: CVPolicyConfig): Partial<CVPolicyConfig> {
  const next = { ...config, [field]: value };
  if (temporalPolicy(next)) return { [field]: value, cv_shuffle: false };
  return { [field]: value,
    ...(config.cv_gap !== undefined ? { cv_gap: 0 } : {}),
    ...(config.cv_test_size !== undefined ? { cv_test_size: null } : {}),
    ...(config.cv_max_train_size !== undefined ? { cv_max_train_size: null } : {}),
  };
}

/** Shared controls keep model and ensemble split policies identical. */
export function CVPolicySettings({ config, update, fieldId, columns }: Props) {
  const policy = config.cv_type === 'nested_cv' ? config.cv_nested_type : config.cv_type;
  return <div className="space-y-3">
    {config.cv_type === 'nested_cv' && <div>
      <label htmlFor={`${fieldId}-nested-policy`} className="block text-xs text-gray-500 mb-1">Nested split policy</label>
      <select id={`${fieldId}-nested-policy`} className={controlClass} value={config.cv_nested_type ?? 'auto'} onChange={e => update(policyChange('cv_nested_type', e.target.value, config))}>
        <option value="auto">Automatic by task</option><option value="k_fold">K-Fold</option><option value="stratified_k_fold">Stratified</option>
        <option value="time_series_split">Time Series</option><option value="group_k_fold">Group K-Fold</option><option value="stratified_group_k_fold">Stratified Group K-Fold</option>
      </select>
      <p className="text-xs text-gray-500">Applies to outer evaluation, inner searches and the final training search.</p>
    </div>}
    {['group_k_fold', 'stratified_group_k_fold'].includes(policy ?? '') && <MetadataColumn {...{ config, update, fieldId, columns }} field="cv_group_column" label="Group column" />}
    {temporalPolicy(config) && <TemporalSettings {...{ config, update, fieldId, columns }} />}
  </div>;
}

function MetadataColumn({ config, update, fieldId, columns, field, label }: Props & { field: 'cv_group_column' | 'cv_time_column'; label: string }) {
  return <ValidationField field={field}><label htmlFor={`${fieldId}-${field}`} className="block text-xs text-gray-500 mb-1">{label}</label>
    <select id={`${fieldId}-${field}`} className={controlClass} value={config[field] ?? ''} onChange={e => update({ [field]: e.target.value })}>
      <option value="">{field === 'cv_time_column' && !requiresExplicitCVTimeColumn(config) ? 'Auto-detect' : 'Select a column'}</option>
      {orderedColumns(columns, field).map(column => <option key={column.name} value={column.name}>{column.name}</option>)}
    </select><p className="text-xs text-gray-500">Used only for splitting; excluded from model features.</p>
  </ValidationField>;
}

function TemporalSettings(props: Props) {
  const { config, update, fieldId } = props;
  return <div className="space-y-2">
    <p className="text-xs text-amber-700 dark:text-amber-400">Training must precede validation and the final holdout. Missing times and tied timestamps across fold boundaries are rejected.</p>
    <MetadataColumn {...props} field="cv_time_column" label={requiresExplicitCVTimeColumn(config) ? 'Time column' : 'Time Column (optional)'} />
    {([['cv_gap', 'Gap (rows)'], ['cv_test_size', 'Test size (rows)'], ['cv_max_train_size', 'Maximum training size (rows)']] as const).map(([field, label]) => <ValidationField field={field} key={field}>
      <label htmlFor={`${fieldId}-${field}`} className="block text-xs text-gray-500 mb-1">{label}</label>
      <input id={`${fieldId}-${field}`} className={controlClass} type="number" min={field === 'cv_gap' ? 0 : 1}
        value={config[field] == null ? (field === 'cv_gap' ? 0 : '') : numericInputValue(config[field], 0)}
        onChange={e => update({ [field]: e.target.value === '' && field !== 'cv_gap' ? null : numericDraft(e.target.value) })} />
    </ValidationField>)}
    <p className="text-xs text-gray-500">Leave test size blank for automatic sizing. Leave maximum training size blank for expanding windows; set it for rolling windows.</p>
  </div>;
}

/** Keep existing source order within the date and nondate groups. */
function orderedColumns(columns: Props['columns'], field: string): Props['columns'] {
  if (field !== 'cv_time_column') return columns;
  const isTime = (column: Props['columns'][number]) => /date|time/.test(String(column.dtype).toLowerCase());
  return [...columns.filter(isTime), ...columns.filter(column => !isTime(column))];
}
