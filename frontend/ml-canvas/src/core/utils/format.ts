/** Pretty-print a scoring metric name, e.g. "f1_weighted" → "F1 Weighted". */
export const formatMetricName = (metric?: string | null): string => {
  if (!metric) return '';
  const map: Record<string, string> = {
    accuracy: 'Accuracy', f1: 'F1', precision: 'Precision', recall: 'Recall',
    roc_auc: 'ROC AUC', r2: 'R²', mse: 'MSE', mae: 'MAE', rmse: 'RMSE',
    f1_weighted: 'F1 Weighted', precision_weighted: 'Precision Weighted',
    recall_weighted: 'Recall Weighted', roc_auc_weighted: 'ROC AUC Weighted',
    f1_macro: 'F1 Macro', f1_micro: 'F1 Micro',
    precision_macro: 'Precision Macro', recall_macro: 'Recall Macro',
    roc_auc_ovr: 'ROC AUC OVR', roc_auc_ovo: 'ROC AUC OVO',
    roc_auc_ovr_weighted: 'ROC AUC OVR Weighted', roc_auc_ovo_weighted: 'ROC AUC OVO Weighted',
    neg_mean_squared_error: 'MSE', neg_mean_absolute_error: 'MAE',
    neg_root_mean_squared_error: 'RMSE', neg_log_loss: 'Log Loss',
  };
  return map[metric] || metric.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
};

export const formatBytes = (bytes: number, decimals = 2) => {
  if (!+bytes) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};
