import { ValidationField } from '../../../../components/shared/ValidationField';
import { parseIntSafe } from '../../../../core/utils/numberInput';
import type { ImputationConfig, ImputationSettingsProps } from './types';

/** Keep the shared method selector mounted while conditional options change. */
export function ImputationMethodOptions({ config, onChange }: ImputationSettingsProps) {
  return <>
    <div>
      <span className="block text-sm font-medium mb-1">Imputation Method</span>
      <select
        aria-label="Imputation Method"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.method || 'simple'}
        onChange={(e) => onChange({ ...config, method: e.target.value as ImputationConfig['method'] })}
      >
        <option value="simple">Simple Imputer (Univariate)</option>
        <option value="knn">KNN Imputer (Multivariate)</option>
        <option value="iterative">Iterative Imputer (MICE)</option>
      </select>
    </div>

    {(config.method === 'simple' || !config.method) && <SimpleOptions config={config} onChange={onChange} />}
    {config.method === 'knn' && <KnnOptions config={config} onChange={onChange} />}
    {config.method === 'iterative' && <IterativeOptions config={config} onChange={onChange} />}
  </>;
}

/** Univariate strategies retain raw text for constant fill values. */
function SimpleOptions({ config, onChange }: ImputationSettingsProps) {
  return (
    <>
      <div>
        <span className="block text-sm font-medium mb-1">Strategy</span>
        <select
          aria-label="Strategy"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.strategy}
          onChange={(e) => onChange({ ...config, strategy: e.target.value as ImputationConfig['strategy'] })}
        >
          <option value="mean">Mean (Average)</option>
          <option value="median">Median (Middle Value)</option>
          <option value="most_frequent">Most Frequent (Mode)</option>
          <option value="constant">Constant Value</option>
        </select>
        <p className="text-[10px] text-muted-foreground mt-1">
          {config.strategy === 'mean' && 'Replaces missing values with the mean of the column. (Numeric only)'}
          {config.strategy === 'median' && 'Replaces missing values with the median. (Robust to outliers)'}
          {config.strategy === 'most_frequent' && 'Replaces missing with the most common value. (Categorical/Numeric)'}
          {config.strategy === 'constant' && 'Replaces missing values with a specific value.'}
        </p>
      </div>

      {config.strategy === 'constant' && (
        <div>
          <span className="block text-sm font-medium mb-1">Fill Value</span>
          <ValidationField field="fill_value">
            <input
              aria-label="Fill Value"
              type="text"
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.fill_value || ''}
              onChange={(e) => onChange({ ...config, fill_value: e.target.value })}
              placeholder="Enter value..."
            />
          </ValidationField>
        </div>
      )}
    </>

  );
}

/** Nearest-neighbor imputation retains its weight and numeric fallback behavior. */
function KnnOptions({ config, onChange }: ImputationSettingsProps) {
  return (
    <>
      <div>
        <span className="block text-sm font-medium mb-1">Number of Neighbors</span>
        <input
          aria-label="Number of Neighbors"
          type="number"
          min="1"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.n_neighbors || 5}
          onChange={(e) => onChange({ ...config, n_neighbors: parseIntSafe(e.target.value, config.n_neighbors) })}
        />
      </div>
      <div>
        <span className="block text-sm font-medium mb-1">Weights</span>
        <select
          aria-label="Weights"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.weights || 'uniform'}
          onChange={(e) => onChange({ ...config, weights: e.target.value as ImputationConfig['weights'] })}
        >
          <option value="uniform">Uniform</option>
          <option value="distance">Distance</option>
        </select>
        <p className="text-[10px] text-muted-foreground mt-1">
          Uniform: All points in each neighborhood are weighted equally.
          Distance: Weight points by the inverse of their distance.
        </p>
      </div>
    </>

  );
}

/** Iterative imputation shares estimator, iteration and reproducibility settings. */
function IterativeOptions({ config, onChange }: ImputationSettingsProps) {
  return (
    <>
      <div>
        <span className="block text-sm font-medium mb-1">Max Iterations</span>
        <input
          aria-label="Max Iterations"
          type="number"
          min="1"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.max_iter || 10}
          onChange={(e) => onChange({ ...config, max_iter: parseIntSafe(e.target.value, config.max_iter) })}
        />
      </div>
      <div>
        <span className="block text-sm font-medium mb-1">Estimator</span>
        <select
          aria-label="Estimator"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.estimator || 'bayesian_ridge'}
          onChange={(e) => onChange({ ...config, estimator: e.target.value as ImputationConfig['estimator'] })}
        >
          <option value="bayesian_ridge">Bayesian Ridge</option>
          <option value="decision_tree">Decision Tree</option>
          <option value="extra_trees">Extra Trees</option>
          <option value="knn">KNN</option>
        </select>
        <p className="text-[10px] text-muted-foreground mt-1">
          The estimator to use for the imputation steps.
        </p>
      </div>
      <div>
        <span className="block text-sm font-medium mb-1">Random State</span>
        <input
          aria-label="Random State"
          type="number"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.random_state ?? 0}
          onChange={(e) => onChange({ ...config, random_state: parseIntSafe(e.target.value, config.random_state) })}
        />
      </div>
    </>

  );
}
