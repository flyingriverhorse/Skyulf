import { ValidationField } from '../../../../components/shared/ValidationField';
import { numericDraft, numericInputValue } from '../../../../core/utils/numericValidation';
import type { FeatureSelectionConfig, SelectionConfigProps } from './types';

// Helper to determine available score functions based on problem type
function getScoreFunctions(config: FeatureSelectionConfig) {
  const type = config.problem_type || 'auto';
  if (type === 'classification') {
    return [
      { value: 'f_classif', label: 'ANOVA F-value' },
      { value: 'mutual_info_classif', label: 'Mutual Information' },
      { value: 'chi2', label: 'Chi-squared' },
    ];
  } else if (type === 'regression') {
    return [
      { value: 'f_regression', label: 'F-value' },
      { value: 'mutual_info_regression', label: 'Mutual Information' },
      { value: 'r_regression', label: 'Pearson Correlation' },
    ];
  }
  return [
    { value: 'f_classif', label: 'ANOVA F-value (Classif)' },
    { value: 'f_regression', label: 'F-value (Reg)' },
    { value: 'mutual_info_classif', label: 'Mutual Info (Classif)' },
    { value: 'mutual_info_regression', label: 'Mutual Info (Reg)' },
  ];
}

function SimpleParameters({ config, onChange }: SelectionConfigProps) {

  return <>
    {/* Variance Threshold */}
    {config.method === 'variance_threshold' && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Threshold</span>
        <ValidationField field="threshold"><input
          aria-label="Threshold"
          type="number"
          step="0.01"
          className="w-full p-2 border rounded bg-background text-sm"
          value={numericInputValue(config.threshold, 0)}
          onChange={(e) => onChange({ ...config, threshold: numericDraft(e.target.value) })}
        /></ValidationField>
        <p className="text-xs text-muted-foreground">Features with variance lower than this will be removed.</p>
      </div>
    )}

    {/* Correlation Threshold */}
    {config.method === 'correlation_threshold' && (
      <>
        <div className="space-y-2">
          <span className="text-sm font-medium">Threshold</span>
          <ValidationField field="threshold"><input
            aria-label="Threshold"
            type="number"
            step="0.01"
            max="1"
            min="0"
            className="w-full p-2 border rounded bg-background text-sm"
            value={numericInputValue(config.threshold, 0.95)}
            onChange={(e) => onChange({ ...config, threshold: numericDraft(e.target.value) })}
          /></ValidationField>
        </div>
        <div className="space-y-2">
          <span className="text-sm font-medium">Method</span>
          <select
            aria-label="Method"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.correlation_method ?? 'pearson'}
            onChange={(e) => onChange({ ...config, correlation_method: e.target.value as FeatureSelectionConfig['correlation_method'] })}
          >
            <option value="pearson">Pearson</option>
            <option value="spearman">Spearman</option>
            <option value="kendall">Kendall</option>
          </select>
        </div>
      </>
    )}


  </>;
}

function ScoringFunction({ config, onChange }: SelectionConfigProps) {
  const isUnivariate = [
    'select_k_best', 'select_percentile', 'select_fpr', 'select_fdr', 'select_fwe', 'generic_univariate_select'
  ].includes(config.method);


  return <>
    {/* Univariate Common: Score Function */}
    {isUnivariate && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Scoring Function</span>
        <select
          aria-label="Scoring Function"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.score_func ?? ''}
          onChange={(e) => onChange({ ...config, score_func: e.target.value })}
        >
          <option value="">Auto (Default)</option>
          {getScoreFunctions(config).map(f => (
            <option key={f.value} value={f.value}>{f.label}</option>
          ))}
        </select>
      </div>
    )}


  </>;
}

function FeatureCount({ config, onChange }: SelectionConfigProps) {

  return <>
    {/* K Best / RFE */}
    {(config.method === 'select_k_best' || config.method === 'rfe') && (
      <div className="space-y-2">
        <span className="text-sm font-medium">K (Number of Features)</span>
        <ValidationField field="k"><input
          aria-label="K (Number of Features)"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          value={numericInputValue(config.k, 10)}
          onChange={(e) => onChange({ ...config, k: numericDraft(e.target.value) })}
        /></ValidationField>
      </div>
    )}


  </>;
}

function UnivariateParameters({ config, onChange }: SelectionConfigProps) {

  return <>
    {/* Percentile */}
    {config.method === 'select_percentile' && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Percentile</span>
        <ValidationField field="percentile"><input
          aria-label="Percentile"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          value={numericInputValue(config.percentile, 10)}
          onChange={(e) => onChange({ ...config, percentile: numericDraft(e.target.value) })}
        /></ValidationField>
      </div>
    )}

    {/* Alpha (FPR, FDR, FWE) */}
    {['select_fpr', 'select_fdr', 'select_fwe'].includes(config.method) && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Alpha (Significance)</span>
        <ValidationField field="alpha"><input
          aria-label="Alpha (Significance)"
          type="number"
          step="0.001"
          className="w-full p-2 border rounded bg-background text-sm"
          value={numericInputValue(config.alpha, 0.05)}
          onChange={(e) => onChange({ ...config, alpha: numericDraft(e.target.value) })}
        /></ValidationField>
      </div>
    )}

    {/* Generic Univariate */}
    {config.method === 'generic_univariate_select' && (
      <>
        <div className="space-y-2">
          <span className="text-sm font-medium">Mode</span>
          <select
            aria-label="Mode"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.mode ?? 'k_best'}
            onChange={(e) => onChange({ ...config, mode: e.target.value as FeatureSelectionConfig['mode'] })}
          >
            <option value="k_best">K Best</option>
            <option value="percentile">Percentile</option>
            <option value="fpr">FPR</option>
            <option value="fdr">FDR</option>
            <option value="fwe">FWE</option>
          </select>
        </div>
        <div className="space-y-2">
          <span className="text-sm font-medium">Parameter</span>
          <ValidationField field="param"><input
            aria-label="Parameter"
            type="number"
            step="0.001"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.param ?? 1e-5}
            onChange={(e) => onChange({ ...config, param: numericDraft(e.target.value) })}
          /></ValidationField>
          <p className="text-xs text-muted-foreground">Value for the selected mode (e.g., k, percentile, or alpha).</p>
        </div>
      </>
    )}


  </>;
}

function ModelParameters({ config, onChange }: SelectionConfigProps) {
  const isModelBased = ['select_from_model', 'rfe', 'rfecv', 'sequential_feature_selection'].includes(config.method);
  return <>
    {/* Model Based Common */}
    {isModelBased && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Estimator</span>
        <select
          aria-label="Estimator"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.estimator ?? 'auto'}
          onChange={(e) => onChange({ ...config, estimator: e.target.value as FeatureSelectionConfig['estimator'] })}
        >
          <option value="auto">Auto</option>
          <option value="RandomForest">Random Forest</option>
          <option value="LogisticRegression">Logistic Regression</option>
          <option value="LinearRegression">Linear Regression</option>
        </select>
      </div>
    )}

    {/* Select From Model Specific */}
    {config.method === 'select_from_model' && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Threshold</span>
        <ValidationField field="threshold"><input
          aria-label="Threshold"
          type="text"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="e.g., median, mean, 1.25*mean"
          value={config.threshold ?? 'median'}
          onChange={(e) => onChange({ ...config, threshold: e.target.value.trim() !== '' && Number.isFinite(Number(e.target.value)) ? Number(e.target.value) : e.target.value })}
        /></ValidationField>
        <p className="text-xs text-muted-foreground">String (e.g. &quot;median&quot;) or float.</p>
        <span className="text-sm font-medium">Max Features</span>
        <ValidationField field="max_features"><input
          aria-label="Max Features"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Optional"
          value={config.max_features ?? ''}
          onChange={(e) => onChange({ ...config, max_features: e.target.value.trim() === '' ? undefined : numericDraft(e.target.value) })}
        /></ValidationField>
        <p className="text-xs text-muted-foreground">Optional cap on the number of features to select. Leave empty for no cap.</p>
      </div>
    )}

    {/* RFE Specific */}
    {(config.method === 'rfe' || config.method === 'rfecv') && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Step</span>
        <ValidationField field="step"><input
          aria-label="Step"
          type="number"
          min="1"
          className="w-full p-2 border rounded bg-background text-sm"
          value={numericInputValue(config.step, 1)}
          onChange={(e) => onChange({ ...config, step: numericDraft(e.target.value) })}
        /></ValidationField>
        <p className="text-xs text-muted-foreground">Features to remove at each iteration.</p>
      </div>
    )}


  </>;
}

function CrossValidatedParameters({ config, onChange }: SelectionConfigProps) {
  if (config.method !== 'rfecv' && config.method !== 'sequential_feature_selection') return null;
  return <div className="space-y-2">
    <label className="block text-xs">CV folds
      <ValidationField field="cv"><input aria-label="CV folds" type="number" min="2" step="1" className="w-full p-2 border rounded" value={numericInputValue(config.cv, 5)}
        onChange={event => onChange({ ...config, cv: numericDraft(event.target.value) })} /></ValidationField>
    </label>
    <label className="block text-xs">Scorer name (optional)
      <input aria-label="Scorer name" type="text" className="w-full p-2 border rounded" placeholder="Estimator default" value={config.scoring ?? ''}
        onChange={event => {
          const next = { ...config };
          if (event.target.value) next.scoring = event.target.value;
          else delete next.scoring;
          onChange(next);
        }} />
    </label>
    {config.method === 'rfecv' ? <label className="block text-xs">Minimum features
      <ValidationField field="min_features_to_select"><input aria-label="Minimum features" type="number" min="1" step="1" className="w-full p-2 border rounded" value={numericInputValue(config.min_features_to_select, 1)}
        onChange={event => onChange({ ...config, min_features_to_select: numericDraft(event.target.value) })} /></ValidationField>
    </label> : <SequentialParameters config={config} onChange={onChange} />}
    <p className="text-xs text-muted-foreground">Cross-validation repeats estimator fitting and can be costly. Folds are not shuffled or automatically reduced for small data.</p>
  </div>;
}

function SequentialParameters({ config, onChange }: SelectionConfigProps) {
  return <>
    <label className="block text-xs">Direction
      <select aria-label="Direction" className="w-full p-2 border rounded" value={config.direction ?? 'forward'} onChange={event => onChange({ ...config, direction: event.target.value as 'forward' | 'backward' })}>
        <option value="forward">Forward</option><option value="backward">Backward</option>
      </select>
    </label>
    <label className="block text-xs">Features to select
      <ValidationField field="n_features_to_select"><input aria-label="Features to select" type="text" className="w-full p-2 border rounded" value={config.n_features_to_select ?? 'auto'}
        onChange={event => onChange({ ...config, n_features_to_select: event.target.value === 'auto' ? 'auto' : numericDraft(event.target.value) })} /></ValidationField>
    </label>
    <p className="text-xs text-muted-foreground">Auto selects half the features. Enter an integer count or a fraction between zero and one.</p>
  </>;
}

export function SelectionParameters({ id, ...props }: SelectionConfigProps & { id: string }) {
  const { config, onChange } = props;
  return <>
    <SimpleParameters {...props} />
    <ScoringFunction {...props} />
    <FeatureCount {...props} />
    <UnivariateParameters {...props} />
    <ModelParameters {...props} />
    <CrossValidatedParameters {...props} />
    {/* Drop Columns Checkbox */}
    <div className="flex items-center space-x-2 pt-2 border-t">
      <input
        type="checkbox"
        id={`${id}-drop-columns`}
        className="rounded border-gray-300"
        checked={config.drop_columns !== false}
        onChange={(e) => onChange({ ...config, drop_columns: e.target.checked })}
      />
      <label htmlFor={`${id}-drop-columns`} className="text-sm font-medium">
        Drop Columns
      </label>
    </div>
    <p className="text-xs text-muted-foreground">
      If unchecked, columns will be identified but not removed from the dataset.
    </p>


  </>;
}
