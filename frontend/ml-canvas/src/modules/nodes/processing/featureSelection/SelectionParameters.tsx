import { parseIntSafe } from '../../../../core/utils/numberInput';
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
        <input
          aria-label="Threshold"
          type="number"
          step="0.01"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.threshold ?? 0}
          onChange={(e) => onChange({ ...config, threshold: Number.parseFloat(e.target.value) })}
        />
        <p className="text-xs text-muted-foreground">Features with variance lower than this will be removed.</p>
      </div>
    )}

    {/* Correlation Threshold */}
    {config.method === 'correlation_threshold' && (
      <>
        <div className="space-y-2">
          <span className="text-sm font-medium">Threshold</span>
          <input
            aria-label="Threshold"
            type="number"
            step="0.01"
            max="1"
            min="0"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.threshold ?? 0.95}
            onChange={(e) => onChange({ ...config, threshold: Number.parseFloat(e.target.value) })}
          />
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
        <input
          aria-label="K (Number of Features)"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.k ?? 10}
          onChange={(e) => onChange({ ...config, k: parseIntSafe(e.target.value, config.k) })}
        />
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
        <input
          aria-label="Percentile"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.percentile ?? 10}
          onChange={(e) => onChange({ ...config, percentile: parseIntSafe(e.target.value, config.percentile) })}
        />
      </div>
    )}

    {/* Alpha (FPR, FDR, FWE) */}
    {['select_fpr', 'select_fdr', 'select_fwe'].includes(config.method) && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Alpha (Significance)</span>
        <input
          aria-label="Alpha (Significance)"
          type="number"
          step="0.001"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.alpha ?? 0.05}
          onChange={(e) => onChange({ ...config, alpha: Number.parseFloat(e.target.value) })}
        />
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
          <input
            aria-label="Parameter"
            type="number"
            step="0.001"
            className="w-full p-2 border rounded bg-background text-sm"
            value={config.param ?? 1e-5}
            onChange={(e) => onChange({ ...config, param: Number.parseFloat(e.target.value) })}
          />
          <p className="text-xs text-muted-foreground">Value for the selected mode (e.g., k, percentile, or alpha).</p>
        </div>
      </>
    )}


  </>;
}

function ModelParameters({ config, onChange }: SelectionConfigProps) {
  const isModelBased = ['select_from_model', 'rfe'].includes(config.method);
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
        <input
          aria-label="Threshold"
          type="text"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="e.g., median, mean, 1.25*mean"
          value={config.threshold ?? 'median'}
          onChange={(e) => onChange({ ...config, threshold: e.target.value })}
        />
        <p className="text-xs text-muted-foreground">String (e.g. &quot;median&quot;) or float.</p>
        <span className="text-sm font-medium">Max Features</span>
        <input
          aria-label="Max Features"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          placeholder="Optional"
          value={config.max_features ?? ''}
          onChange={(e) => onChange({ ...config, max_features: parseIntSafe(e.target.value, config.max_features) })}
        />
        <p className="text-xs text-muted-foreground">Optional cap on the number of features to select. Leave empty for no cap.</p>
      </div>
    )}

    {/* RFE Specific */}
    {config.method === 'rfe' && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Step</span>
        <input
          aria-label="Step"
          type="number"
          min="1"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.step ?? 1}
          onChange={(e) => onChange({ ...config, step: parseIntSafe(e.target.value, config.step) })}
        />
        <p className="text-xs text-muted-foreground">Features to remove at each iteration.</p>
      </div>
    )}


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
