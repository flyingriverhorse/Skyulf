import { ValidationField } from '../../../../components/shared/ValidationField';
import type { FeatureSelectionConfig, SelectionConfigProps } from './types';

interface SelectionControlsProps extends SelectionConfigProps {
  columns: string[];
  upstreamTargetColumn: string | undefined;
  showInvalidTarget: boolean;
}

export function SelectionStatus({ upstreamDatasetId, isLoading }: { upstreamDatasetId: string | undefined; isLoading: boolean }) {
  return (

    <div className="shrink-0 p-4 pb-0 space-y-2">
      {!upstreamDatasetId && (
        <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400 text-xs rounded border border-yellow-200 dark:border-yellow-800">
          Connect a dataset node to configure.
        </div>
      )}
      {isLoading && !!upstreamDatasetId && (
        <div className="text-xs text-muted-foreground animate-pulse">Loading schema...</div>
      )}
    </div>


  );
}

function TargetSelection({ config, onChange, columns, upstreamTargetColumn, showInvalidTarget }: SelectionControlsProps) {
  return <>
    {((config.method !== 'variance_threshold' && config.method !== 'correlation_threshold') || showInvalidTarget) && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Target Column</span>
        {upstreamTargetColumn ? (
          <div className="p-2 bg-muted rounded text-sm text-muted-foreground border">
            {upstreamTargetColumn} <span className="text-xs italic">(Auto-detected)</span>
          </div>
        ) : (
          <ValidationField field="target_column">
            <select
              aria-label="Target Column"
              className="w-full p-2 border rounded bg-background text-sm"
              value={config.target_column ?? ''}
              onChange={(e) => onChange({ ...config, target_column: e.target.value })}
            >
              <option value="">Select Target...</option>
              {columns.map(col => (
                <option key={col} value={col}>{col}</option>
              ))}
            </select>
          </ValidationField>
        )}
        <p className="text-xs text-muted-foreground">Required for supervised selection methods.</p>
      </div>
    )}


  </>;
}

export function SelectionControls(props: SelectionControlsProps) {
  const { config, onChange } = props;
  const isUnivariate = [
    'select_k_best', 'select_percentile', 'select_fpr', 'select_fdr', 'select_fwe', 'generic_univariate_select'
  ].includes(config.method);

  const isModelBased = ['select_from_model', 'rfe'].includes(config.method);


  return <>
    <div className="space-y-2">
      <span className="text-sm font-medium">Selection Method</span>
      <select
        aria-label="Selection Method"
        className="w-full p-2 border rounded bg-background text-sm"
        value={config.method}
        onChange={(e) => onChange({ ...config, method: e.target.value as FeatureSelectionConfig['method'] })}
      >
        <optgroup label="Simple">
          <option value="variance_threshold">Variance Threshold</option>
          <option value="correlation_threshold">Correlation Threshold</option>
        </optgroup>
        <optgroup label="Univariate">
          <option value="select_k_best">Select K Best</option>
          <option value="select_percentile">Select Percentile</option>
          <option value="select_fpr">False Positive Rate (FPR)</option>
          <option value="select_fdr">False Discovery Rate (FDR)</option>
          <option value="select_fwe">Family-wise Error (FWE)</option>
          <option value="generic_univariate_select">Generic Univariate</option>
        </optgroup>
        <optgroup label="Model Based">
          <option value="select_from_model">Select From Model</option>
          <option value="rfe">Recursive Feature Elimination (RFE)</option>
        </optgroup>
      </select>
    </div>


    <TargetSelection {...props} />
    {(isUnivariate || isModelBased) && (
      <div className="space-y-2">
        <span className="text-sm font-medium">Problem Type</span>
        <select
          aria-label="Problem Type"
          className="w-full p-2 border rounded bg-background text-sm"
          value={config.problem_type ?? 'auto'}
          onChange={(e) => onChange({ ...config, problem_type: e.target.value as FeatureSelectionConfig['problem_type'] })}
        >
          <option value="auto">Auto-detect</option>
          <option value="classification">Classification</option>
          <option value="regression">Regression</option>
        </select>
      </div>
    )}


  </>;
}
