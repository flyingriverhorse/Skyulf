import { describe, it, expect } from 'vitest';
import { formatMetricName, formatBytes } from './format';

describe('formatMetricName', () => {
  it('returns empty string for null/undefined/empty input', () => {
    expect(formatMetricName(undefined)).toBe('');
    expect(formatMetricName(null)).toBe('');
    expect(formatMetricName('')).toBe('');
  });

  it('returns the canonical label for known metrics', () => {
    expect(formatMetricName('accuracy')).toBe('Accuracy');
    expect(formatMetricName('roc_auc')).toBe('ROC AUC');
    expect(formatMetricName('r2')).toBe('R²');
    expect(formatMetricName('rmse')).toBe('RMSE');
    expect(formatMetricName('f1_weighted')).toBe('F1 Weighted');
  });

  it('collapses sklearn neg_* aliases to their positive name', () => {
    expect(formatMetricName('neg_mean_squared_error')).toBe('MSE');
    expect(formatMetricName('neg_mean_absolute_error')).toBe('MAE');
    expect(formatMetricName('neg_root_mean_squared_error')).toBe('RMSE');
    expect(formatMetricName('neg_log_loss')).toBe('Log Loss');
  });

  it('falls back to title-cased snake_case for unknown metrics', () => {
    expect(formatMetricName('balanced_accuracy_score')).toBe('Balanced Accuracy Score');
    expect(formatMetricName('custom_metric')).toBe('Custom Metric');
  });
});

describe('formatBytes', () => {
  it('returns "0 Bytes" for zero / falsy / NaN input', () => {
    expect(formatBytes(0)).toBe('0 Bytes');
    expect(formatBytes(NaN)).toBe('0 Bytes');
  });

  it('formats common sizes with the right unit', () => {
    expect(formatBytes(500)).toBe('500 Bytes');
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1024 * 1024)).toBe('1 MB');
    expect(formatBytes(1024 * 1024 * 1024)).toBe('1 GB');
  });

  it('respects the decimals argument', () => {
    expect(formatBytes(1536, 0)).toBe('2 KB'); // 1.5 → rounded
    expect(formatBytes(1536, 2)).toBe('1.5 KB');
    expect(formatBytes(1536, 4)).toBe('1.5 KB');
  });

  it('clamps negative decimals to 0', () => {
    expect(formatBytes(1536, -1)).toBe('2 KB');
  });
});
