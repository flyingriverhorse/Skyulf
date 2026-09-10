import type { YProba } from '../../types';

/** Resolve the selected probability label to its matrix row. */
export function getLiveMetrics(proba: YProba | undefined, selectedClass: string | null, classes: (string | number)[], matrix: number[][]) {
  if(!selectedClass || !proba) return null;
  const labels = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
  const classIndex = (labels ?? proba.classes).findIndex(c => String(c) === selectedClass);
  if(classIndex === -1) return null;
  const posVal = proba.classes[classIndex];
  const matrixIndex = classes.findIndex(c => String(c) === String(posVal));
  return matrixIndex === -1 ? null : calculateLiveMetrics(matrix, matrixIndex);
}

/** Derive live one-versus-rest rates from the selected matrix row. */
function calculateLiveMetrics(matrix: number[][], posMatrixIdx: number) {
  const tp = matrix[posMatrixIdx]?.[posMatrixIdx] ?? 0;
  const fp = matrix.reduce((s, row, ri) => ri !== posMatrixIdx ? s + (row[posMatrixIdx] ?? 0) : s, 0);
  const fn = (matrix[posMatrixIdx] ?? []).reduce((s, v, ci) => ci !== posMatrixIdx ? s + v : s, 0);
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const tn = total - tp - fp - fn;
  const accuracy = total > 0 ? (tp + tn) / total : 0;
  const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
  const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return { accuracy, precision, recall, f1 };
}
