import { ChevronDown, ChevronUp } from 'lucide-react';
import { ValidationField } from '../../../../components/shared/ValidationField';
import { RecommendationsPanel } from '../../../../components/panels/RecommendationsPanel';
import type { ColumnProfile, Recommendation } from '../../../../core/api/client';
import { parseIntSafe } from '../../../../core/utils/numberInput';
import type { ResamplingNode } from '../ResamplingNode';
import type { useResamplingData } from './useResamplingData';
import { TargetColumnField } from './TargetColumnField';

type Config = ReturnType<typeof ResamplingNode.getDefaultConfig>;
type FieldProps = { config: Config; handleChange: (key: keyof Config, value: unknown) => void };

/** Shared method selection, target, sampling strategy and random seed controls. */
export function GeneralSettings({ config, handleChange, id, data }: FieldProps & {
  id: string; data: ReturnType<typeof useResamplingData>;
}) {
  const { schema, droppedUpstream, upstreamTarget } = data;
  return <>
    <div className="space-y-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Resampling Type</span>
      <select
        aria-label="Resampling Type"
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        value={config.type}
        onChange={(e) => { handleChange('type', e.target.value as Config['type']); }}
      >
        <option value="oversampling">Oversampling (Minority)</option>
        <option value="undersampling">Undersampling (Majority)</option>
      </select>
    </div>

    <div className="space-y-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Method</span>
      <select
        aria-label="Method"
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        value={config.method}
        onChange={(e) => { handleChange('method', e.target.value); }}
      >
        {config.type === 'oversampling' ? (
          <>
            <option value="random_over">Random Over Sampler</option>
            <option value="smote">SMOTE</option>
            <option value="adasyn">ADASYN</option>
            <option value="borderline_smote">Borderline SMOTE</option>
            <option value="svm_smote">SVM SMOTE</option>
            <option value="kmeans_smote">KMeans SMOTE</option>
            <option value="smote_tomek">SMOTE + Tomek</option>
          </>
        ) : (
          <>
            <option value="random_under_sampling">Random Under Sampling</option>
            <option value="nearmiss">NearMiss</option>
            <option value="tomek_links">Tomek Links</option>
            <option value="edited_nearest_neighbours">Edited Nearest Neighbours</option>
          </>
        )}
      </select>
      <p className="text-xs text-gray-500 dark:text-gray-400">
        {config.type === 'oversampling'
          ? 'Balance minority classes with duplicated or synthetic samples.'
          : 'Remove samples from the majority class.'}
      </p>
    </div>

    <div className="space-y-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Target Column</span>
      <TargetColumnField id={id} value={config.target_column}
        columns={Object.values(schema?.columns ?? {}).filter((col: ColumnProfile) => !droppedUpstream.has(col.name)).map(col => col.name)}
        onChange={value => handleChange('target_column', value)} />
      <p className="text-xs text-gray-500 dark:text-gray-400">
        {upstreamTarget ? 'Auto-detected from upstream node.' : 'The column containing the class labels.'}
      </p>
    </div>

    <div className="space-y-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Sampling Strategy</span>
      <select
        aria-label="Sampling Strategy"
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        value={config.sampling_strategy}
        onChange={(e) => { handleChange('sampling_strategy', e.target.value); }}
      >
        <option value="auto">Auto (Resample all classes but majority)</option>
        <option value="minority">Minority (Resample only minority class)</option>
        <option value="not minority">Not Minority (Resample all but minority)</option>
        <option value="not majority">Not Majority (Resample all but majority)</option>
        <option value="all">All (Resample all classes)</option>
      </select>
      <p className="text-xs text-gray-500 dark:text-gray-400">Defines which classes to resample.</p>
    </div>

    <div className="space-y-2">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Random State</span>
      <input
        aria-label="Random State"
        type="number"
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
        value={config.random_state}
        onChange={(e) => { handleChange('random_state', parseIntSafe(e.target.value, config.random_state)); }}
      />
    </div>
  </>;
}

/** Dispatch without changing the type/method combinations that expose extra fields. */
export function MethodSettings(props: FieldProps) {
  if (props.config.type === 'oversampling') return <OversamplingSettings {...props} />;
  if (props.config.type === 'undersampling') return <UndersamplingSettings {...props} />;
  return null;
}

/** Shared neighborhood controls for synthetic minority-sampling methods. */
function OversamplingSettings({ config, handleChange }: FieldProps) {
  return <>
    {['smote', 'adasyn', 'borderline_smote', 'svm_smote', 'kmeans_smote', 'smote_tomek'].includes(config.method) && (
      <div className="space-y-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">k Neighbors</span>
        <ValidationField field="k_neighbors">
          <input
            aria-label="k Neighbors"
            type="number"
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={config.k_neighbors ?? 5}
            onChange={(e) => { handleChange('k_neighbors', parseIntSafe(e.target.value, config.k_neighbors)); }}
          />
        </ValidationField>
      </div>
    )}

    {['borderline_smote', 'svm_smote'].includes(config.method) && (
      <div className="space-y-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">m Neighbors</span>
        <input
          aria-label="m Neighbors"
          type="number"
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          value={config.m_neighbors ?? 10}
          onChange={(e) => { handleChange('m_neighbors', parseIntSafe(e.target.value, config.m_neighbors)); }}
        />
      </div>
    )}

    <SmoteVariantSettings config={config} handleChange={handleChange} />
  </>;
}

/** Borderline, SVM and KMeans variants each retain their additional algorithm parameters. */
function SmoteVariantSettings({ config, handleChange }: FieldProps) {
  return <>
    {config.method === 'borderline_smote' && (
      <div className="space-y-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Kind</span>
        <select
          aria-label="Kind"
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          value={config.kind ?? 'borderline-1'}
          onChange={(e) => { handleChange('kind', e.target.value); }}
        >
          <option value="borderline-1">Borderline-1</option>
          <option value="borderline-2">Borderline-2</option>
        </select>
      </div>
    )}

    {config.method === 'svm_smote' && (
      <div className="space-y-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Out Step</span>
        <input
          aria-label="Out Step"
          type="number"
          step="0.1"
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          value={config.out_step ?? 0.5}
          onChange={(e) => { handleChange('out_step', Number.parseFloat(e.target.value)); }}
        />
      </div>
    )}

    {config.method === 'kmeans_smote' && (
      <>
        <div className="space-y-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Cluster Balance Threshold</span>
          <input
            aria-label="Cluster Balance Threshold"
            type="number"
            step="0.1"
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={config.cluster_balance_threshold ?? 0.1}
            onChange={(e) => { handleChange('cluster_balance_threshold', Number.parseFloat(e.target.value)); }}
          />
        </div>
        <div className="space-y-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Density Exponent</span>
          <input
            aria-label="Density Exponent"
            type="text"
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={config.density_exponent ?? 'auto'}
            onChange={(e) => { handleChange('density_exponent', e.target.value); }}
          />
        </div>
      </>
    )}
  </>;
}

/** Majority-sampling controls vary between random selection, NearMiss and ENN. */
function UndersamplingSettings({ config, handleChange }: FieldProps) {
  return <>
    {config.method === 'random_under_sampling' && (
      <div className="flex items-center justify-between p-3 border rounded-md bg-gray-50 dark:bg-gray-800 dark:border-gray-700">
        <div>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300 block">Replacement</span>
          <span className="text-xs text-gray-500 dark:text-gray-400">Sample with replacement</span>
        </div>
        <input
          aria-label="Replacement"
          type="checkbox"
          className="h-4 w-4 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500"
          checked={config.replacement ?? false}
          onChange={(e) => { handleChange('replacement', e.target.checked); }}
        />
      </div>
    )}

    {config.method === 'nearmiss' && (
      <div className="space-y-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Version</span>
        <select
          aria-label="Version"
          className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          value={config.version ?? 1}
          onChange={(e) => { handleChange('version', parseIntSafe(e.target.value, config.version)); }}
        >
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="3">3</option>
        </select>
      </div>
    )}

    {config.method === 'edited_nearest_neighbours' && (
      <>
        <div className="space-y-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">n Neighbors</span>
          <input
            aria-label="n Neighbors"
            type="number"
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={config.n_neighbors ?? 3}
            onChange={(e) => { handleChange('n_neighbors', parseIntSafe(e.target.value, config.n_neighbors)); }}
          />
        </div>
        <div className="space-y-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Selection Kind</span>
          <select
            aria-label="Selection Kind"
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 p-2 text-sm text-gray-900 dark:text-gray-100 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            value={config.kind_sel ?? 'all'}
            onChange={(e) => { handleChange('kind_sel', e.target.value); }}
          >
            <option value="all">All</option>
            <option value="mode">Mode</option>
          </select>
        </div>
      </>
    )}
  </>;
}

/** Recommendation expansion remains owned by the settings form across mode changes. */
export function ResamplingRecommendations({ recommendations, showRecommendations, setShowRecommendations, handleApplyRecommendation }: {
  recommendations: Recommendation[];
  showRecommendations: boolean;
  setShowRecommendations: (show: boolean) => void;
  handleApplyRecommendation: (recommendation: Recommendation) => void;
}) {
  return <>
    {recommendations.length > 0 && (
      <div className="mt-0 border rounded-md overflow-hidden border-gray-200 dark:border-gray-700">
        <button
          className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          onClick={() => { setShowRecommendations(!showRecommendations); }}
        >
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            Recommendations ({recommendations.length})
          </span>
          {showRecommendations ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>

        {showRecommendations && (
          <div className="p-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
            <RecommendationsPanel
              recommendations={recommendations}
              onApply={handleApplyRecommendation}
            />
          </div>
        )}
      </div>
    )}
  </>;
}
