import React, { useMemo, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import type { FeatureNodeParameter } from '../../../../api';
import type { PreviewState } from '../dataset/DataSnapshotSection';
import {
  extractPendingConfigurationDetails,
  type PendingConfigurationDetail,
} from '../../utils/pendingConfiguration';

export type TrainTestSplitConfig = {
  test_size: number;
  validation_size: number;
  random_state?: number;
  shuffle: boolean;
  stratify: boolean;
  target_column?: string;
};

type TrainTestSplitNodeSignal = {
  node_id?: string;
  train_size?: number;
  validation_size?: number;
  test_size?: number;
  total_size?: number;
  test_ratio?: number;
  validation_ratio?: number;
  stratified: boolean;
  target_column?: string;
  random_state?: number;
  shuffle: boolean;
  splits_created: string[];
};

type TrainTestSplitSectionProps = {
  nodeId: string;
  sourceId?: string | null;
  hasReachableSource: boolean;
  previewState: PreviewState;
  onRefreshPreview: () => void;
  config: TrainTestSplitConfig | null;
  parameters: FeatureNodeParameter[];
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  formatMetricValue: (value?: number | null, precision?: number) => string;
  onPendingConfigurationWarning?: (
    details: PendingConfigurationDetail[]
  ) => void;
  onPendingConfigurationCleared?: () => void;
};

const resolveSignal = (
  signals: TrainTestSplitNodeSignal[],
  nodeId: string,
): TrainTestSplitNodeSignal | null => {
  if (!signals.length) {
    return null;
  }
  const matching = signals.filter((signal) => !signal?.node_id || signal.node_id === nodeId);
  if (matching.length) {
    return matching[matching.length - 1];
  }
  return signals[signals.length - 1];
};

const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

export const TrainTestSplitSection: React.FC<TrainTestSplitSectionProps> = ({
  nodeId,
  sourceId,
  hasReachableSource,
  previewState,
  onRefreshPreview,
  config,
  parameters,
  renderParameterField,
  formatMetricValue,
  onPendingConfigurationWarning,
  onPendingConfigurationCleared,
}) => {
  useEffect(() => {
    if (!previewState.data?.signals?.full_execution) {
      onPendingConfigurationCleared?.();
      return;
    }

    const details = extractPendingConfigurationDetails(
      previewState.data.signals.full_execution,
    );

    if (details.length > 0) {
      onPendingConfigurationWarning?.(details);
    } else {
      onPendingConfigurationCleared?.();
    }
  }, [
    previewState.data?.signals?.full_execution,
    onPendingConfigurationWarning,
    onPendingConfigurationCleared,
  ]);
  const preview = previewState.data;
  const previewStatus = previewState.status;

  const rawSignals = Array.isArray(preview?.signals?.train_test_split)
    ? (preview?.signals?.train_test_split as TrainTestSplitNodeSignal[])
    : [];
  const activeSignal = resolveSignal(rawSignals, nodeId);

  const testSize = config?.test_size ?? 0.2;
  const validationSize = config?.validation_size ?? 0.0;
  const randomState = config?.random_state;
  const shuffle = config?.shuffle ?? true;
  const stratify = config?.stratify ?? false;
  const targetColumn = config?.target_column ?? '';

  const showValidation = validationSize > 0;

  const splitInfo = useMemo(() => {
    if (activeSignal) {
      return {
        trainSize: activeSignal.train_size ?? 0,
        validationSize: activeSignal.validation_size ?? 0,
        testSize: activeSignal.test_size ?? 0,
        totalSize: activeSignal.total_size ?? 0,
        trainRatio: activeSignal.train_size && activeSignal.total_size 
          ? activeSignal.train_size / activeSignal.total_size 
          : 0,
        validationRatio: activeSignal.validation_ratio ?? 0,
        testRatio: activeSignal.test_ratio ?? 0,
      };
    }

    // Calculate expected splits from config
    const trainRatio = 1 - testSize - validationSize;
    return {
      trainSize: 0,
      validationSize: 0,
      testSize: 0,
      totalSize: 0,
      trainRatio,
      validationRatio: validationSize,
      testRatio: testSize,
    };
  }, [activeSignal, testSize, validationSize]);

  const testSizeParam = parameters.find(p => p.name === 'test_size');
  const validationSizeParam = parameters.find(p => p.name === 'validation_size');
  const randomStateParam = parameters.find(p => p.name === 'random_state');
  const shuffleParam = parameters.find(p => p.name === 'shuffle');
  const stratifyParam = parameters.find(p => p.name === 'stratify');
  const targetColumnParam = parameters.find(p => p.name === 'target_column');

  return (
    <div className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Train/Test Split Configuration</h3>
      </div>

      <div className="node-settings__field-group">
        {testSizeParam && renderParameterField(testSizeParam)}
        {validationSizeParam && renderParameterField(validationSizeParam)}
        {randomStateParam && renderParameterField(randomStateParam)}
        {shuffleParam && renderParameterField(shuffleParam)}
        {stratifyParam && renderParameterField(stratifyParam)}
        {stratify && targetColumnParam && renderParameterField(targetColumnParam)}
      </div>

      {/* Split Preview */}
      {previewStatus === 'success' && activeSignal && (
        <div className="node-settings__info-panel">
          <h4 className="node-settings__info-title">Split Summary</h4>
          <div className="node-settings__split-visualization">
            <div className="split-bar">
              <div 
                className="split-bar__segment split-bar__segment--train"
                style={{ width: formatPercentage(splitInfo.trainRatio) }}
                title={`Training: ${splitInfo.trainSize} rows (${formatPercentage(splitInfo.trainRatio)})`}
              >
                <span className="split-bar__label">Train</span>
              </div>
              {showValidation && (
                <div 
                  className="split-bar__segment split-bar__segment--validation"
                  style={{ width: formatPercentage(splitInfo.validationRatio) }}
                  title={`Validation: ${splitInfo.validationSize} rows (${formatPercentage(splitInfo.validationRatio)})`}
                >
                  <span className="split-bar__label">Val</span>
                </div>
              )}
              <div 
                className="split-bar__segment split-bar__segment--test"
                style={{ width: formatPercentage(splitInfo.testRatio) }}
                title={`Test: ${splitInfo.testSize} rows (${formatPercentage(splitInfo.testRatio)})`}
              >
                <span className="split-bar__label">Test</span>
              </div>
            </div>
          </div>

          <div className="node-settings__stats-grid">
            <div className="node-settings__stat-item">
              <span className="node-settings__stat-label">Total rows:</span>
              <span className="node-settings__stat-value">{formatMetricValue(splitInfo.totalSize, 0)}</span>
            </div>
            <div className="node-settings__stat-item">
              <span className="node-settings__stat-label">Training:</span>
              <span className="node-settings__stat-value">
                {formatMetricValue(splitInfo.trainSize, 0)} ({formatPercentage(splitInfo.trainRatio)})
              </span>
            </div>
            {showValidation && (
              <div className="node-settings__stat-item">
                <span className="node-settings__stat-label">Validation:</span>
                <span className="node-settings__stat-value">
                  {formatMetricValue(splitInfo.validationSize, 0)} ({formatPercentage(splitInfo.validationRatio)})
                </span>
              </div>
            )}
            <div className="node-settings__stat-item">
              <span className="node-settings__stat-label">Test:</span>
              <span className="node-settings__stat-value">
                {formatMetricValue(splitInfo.testSize, 0)} ({formatPercentage(splitInfo.testRatio)})
              </span>
            </div>
            {stratify && targetColumn && (
              <div className="node-settings__stat-item">
                <span className="node-settings__stat-label">Stratified by:</span>
                <span className="node-settings__stat-value">{targetColumn}</span>
              </div>
            )}
            {randomState !== undefined && (
              <div className="node-settings__stat-item">
                <span className="node-settings__stat-label">Random state:</span>
                <span className="node-settings__stat-value">{randomState}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Connection status */}
      {!hasReachableSource && (
        <div className="node-settings__warning" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="node-settings__warning-icon" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <AlertTriangle size={16} />
          </span>
          Connect this node to a data source to configure the split.
        </div>
      )}

      {previewStatus === 'loading' && (
        <div className="node-settings__loading">Loading preview...</div>
      )}

      {previewStatus === 'error' && (
        <div className="node-settings__error">
          <span className="node-settings__error-icon">✕</span>
          Failed to load preview. Try refreshing.
        </div>
      )}
    </div>
  );
};
