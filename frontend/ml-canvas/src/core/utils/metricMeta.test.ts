import { describe, expect, it } from 'vitest';

import { getMetricDirection, getMetricSplitLabel, pickBestIndex } from './metricMeta';

describe('getMetricDirection', () => {
  it.each(['accuracy', 'balanced_accuracy', 'f1_weighted', 'roc_auc_ovr', 'pr_auc', 'r2', 'g_score', 'matthews_corrcoef', 'explained_variance', 'precision', 'recall'])(
    'reports %s as higher-is-better',
    (key) => {
      expect(getMetricDirection(key)).toBe('higher');
    },
  );

  it.each(['rmse', 'mae', 'mse', 'log_loss', 'mape'])('reports %s as lower-is-better', (key) => {
    expect(getMetricDirection(key)).toBe('lower');
  });

  it('resolves direction through a split prefix', () => {
    expect(getMetricDirection('test_f1_weighted')).toBe('higher');
    expect(getMetricDirection('train_rmse')).toBe('lower');
    expect(getMetricDirection('val_accuracy')).toBe('higher');
    expect(getMetricDirection('cv_accuracy_mean')).toBe('higher');
  });

  it('treats any cross-validation standard deviation as lower-is-better', () => {
    expect(getMetricDirection('cv_accuracy_std')).toBe('lower');
    expect(getMetricDirection('cv_rmse_std')).toBe('lower');
  });

  it('treats the tuning best_score as higher-is-better', () => {
    expect(getMetricDirection('best_score')).toBe('higher');
  });

  it('knows cluster-quality directions', () => {    expect(getMetricDirection('silhouette')).toBe('higher');
    expect(getMetricDirection('calinski_harabasz')).toBe('higher');
    expect(getMetricDirection('davies_bouldin')).toBe('lower');
  });

  it('treats resource cost metrics emitted by the runner as lower-is-better', () => {
    expect(getMetricDirection('fit_time')).toBe('lower');
    expect(getMetricDirection('score_time')).toBe('lower');
    expect(getMetricDirection('predict_time')).toBe('lower');
    expect(getMetricDirection('peak_memory_bytes')).toBe('lower');
  });

  it('leaves row counts unranked because neither direction is better', () => {
    expect(getMetricDirection('rows_in')).toBe('unknown');
    expect(getMetricDirection('rows_out')).toBe('unknown');
  });

  it('admits ignorance rather than guessing', () => {
    expect(getMetricDirection('custom_business_kpi')).toBe('unknown');
    expect(getMetricDirection('')).toBe('unknown');
  });
});

describe('getMetricSplitLabel', () => {
  it('names each split a metric was evaluated on', () => {
    expect(getMetricSplitLabel('train_f1')).toBe('Train');
    expect(getMetricSplitLabel('test_f1')).toBe('Test (held-out)');
    expect(getMetricSplitLabel('val_f1')).toBe('Validation');
    expect(getMetricSplitLabel('cv_f1_mean')).toBe('CV mean');
    expect(getMetricSplitLabel('cv_f1_std')).toBe('CV std');
  });

  it('labels the tuning score by the population that produced it', () => {
    expect(getMetricSplitLabel('best_score')).toBe('CV mean');
  });

  it('returns null when a metric carries no split context', () => {
    expect(getMetricSplitLabel('accuracy')).toBeNull();
    expect(getMetricSplitLabel('silhouette')).toBeNull();
  });
});

describe('pickBestIndex', () => {
  it('picks the largest value for a higher-is-better metric', () => {
    expect(pickBestIndex([0.7, 0.9, 0.8], 'higher')).toBe(1);
  });

  it('picks the smallest value for a lower-is-better metric', () => {
    expect(pickBestIndex([0.7, 0.9, 0.2], 'lower')).toBe(2);
  });

  it('never picks a winner when the direction is unknown', () => {
    expect(pickBestIndex([0.7, 0.9], 'unknown')).toBeNull();
  });

  it('ignores missing values when choosing', () => {
    expect(pickBestIndex([undefined, 0.4, 0.9, undefined], 'higher')).toBe(2);
  });

  it('declines to crown a winner when only one run reported the metric', () => {
    expect(pickBestIndex([undefined, 0.4, undefined], 'higher')).toBeNull();
  });

  it('returns null when nothing is comparable', () => {
    expect(pickBestIndex([undefined, undefined], 'higher')).toBeNull();
    expect(pickBestIndex([], 'higher')).toBeNull();
  });

  it('declines to crown a winner when the best value is tied', () => {
    expect(pickBestIndex([0.9, 0.9], 'higher')).toBeNull();
  });

  it('does not treat a single run as a comparison winner', () => {
    expect(pickBestIndex([0.9], 'higher')).toBeNull();
  });
});
