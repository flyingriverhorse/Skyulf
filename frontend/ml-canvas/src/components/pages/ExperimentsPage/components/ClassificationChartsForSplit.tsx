/**
 * All classification evaluation charts for one split (train/test/val).
 * Extracted from ExperimentsPage. Renders:
 *   1. Confusion Matrix card (with binary per-class panel + live OvR metrics)
 *   2. ROC Curve (with AUC + operating point)
 *   3. PR Curve + Score Distribution (when y_proba and a class are selected)
 *   4. Calibration / Cumulative Gains / MCC vs Threshold (3-up grid)
 */

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Line, ReferenceLine, ReferenceDot, ComposedChart, Area,
} from 'recharts';
import { Loader2, Check, Download } from 'lucide-react';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import type { EvaluationSplit } from '../types';
import {
  calculateConfusionMatrix,
  calculateROC,
  calculatePR,
  getScoreDistribution,
  getCalibrationData,
  getCumulativeGainsData,
  getMCCByThreshold,
} from '../utils/classificationCharts';

interface Props {
  splitName: string;
  splitData: EvaluationSplit;
  selectedRocClass: string | null;
  threshold: number;
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const ClassificationChartsForSplit: React.FC<Props> = ({
  splitName,
  splitData,
  selectedRocClass,
  threshold,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  return (
    <>
                                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                                {/* Confusion Matrix */}
                                                <div className="flex flex-col items-center justify-center relative group" id={`${splitName}-confusion-matrix`}>
                                                    <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                       <button 
                                                         onClick={() => void handleDownload(`${splitName}-confusion-matrix`, `${splitName}_confusion_matrix`)}
                                                         disabled={downloadingChart === `${splitName}-confusion-matrix`}
                                                         className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                         title="Download Graph"
                                                       >
                                                          {downloadingChart === `${splitName}-confusion-matrix` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-confusion-matrix` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                       </button>
                                                    </div>
                                                    {(() => {
                                                        const proba = splitData.y_proba;
                                                        const classOrder = proba?.classes;
                                                        let yTrueForCm: (string | number)[] = splitData.y_true;
                                                        let yPredForCm: (string | number)[] = splitData.y_pred;
                                                        if (proba?.labels && proba.labels.length === proba.classes.length) {
                                                            const labelToClass = new Map<string, string | number>();
                                                            proba.labels.forEach((label, idx) => {
                                                                const cls = proba.classes[idx];
                                                                if (cls !== undefined) labelToClass.set(String(label), cls);
                                                            });
                                                            yTrueForCm = splitData.y_true.map(y => labelToClass.get(String(y)) ?? y);
                                                            yPredForCm = splitData.y_pred.map(y => labelToClass.get(String(y)) ?? y);
                                                        }

                                                        // Apply OvR threshold for the selected class (works for binary and multiclass)
                                                        if (proba && selectedRocClass) {
                                                            const labelList = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
                                                            const posIdx = (labelList ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                                            if (posIdx !== -1) {
                                                                const posVal = proba.classes[posIdx];
                                                                const origPred = [...yPredForCm];
                                                                if (posVal !== undefined) {
                                                                    yPredForCm = proba.values.map((v, i) => {
                                                                        if ((v[posIdx] ?? 0) >= threshold) return posVal;
                                                                        // Argmax of all other classes
                                                                        let bestIdx = -1, bestProb = -Infinity;
                                                                        v.forEach((p, idx) => {
                                                                            if (idx !== posIdx && p > bestProb) { bestProb = p; bestIdx = idx; }
                                                                        });
                                                                        return bestIdx >= 0 ? (proba.classes[bestIdx] ?? origPred[i]!) : (origPred[i]!);
                                                                    });
                                                                }
                                                            }
                                                        }

                                                        const { classes, matrix } = calculateConfusionMatrix(yTrueForCm, yPredForCm, classOrder);

                                                        // Compute live OvR metrics for the selected class
                                                        let liveMetrics: { accuracy: number; precision: number; recall: number; f1: number } | null = null;
                                                        if (selectedRocClass && proba) {
                                                            const labelList = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
                                                            const posClassIdx = (labelList ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                                            if (posClassIdx !== -1) {
                                                                const posVal = proba.classes[posClassIdx];
                                                                const posMatrixIdx = classes.findIndex(c => String(c) === String(posVal));
                                                                if (posMatrixIdx !== -1) {
                                                                    const tp = matrix[posMatrixIdx]?.[posMatrixIdx] ?? 0;
                                                                    const fp = matrix.reduce((s, row, ri) => ri !== posMatrixIdx ? s + (row[posMatrixIdx] ?? 0) : s, 0);
                                                                    const fn = (matrix[posMatrixIdx] ?? []).reduce((s, v, ci) => ci !== posMatrixIdx ? s + v : s, 0);
                                                                    const total = matrix.flat().reduce((a, b) => a + b, 0);
                                                                    const tn = total - tp - fp - fn;
                                                                    const accuracy = total > 0 ? (tp + tn) / total : 0;
                                                                    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
                                                                    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
                                                                    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
                                                                    liveMetrics = { accuracy, precision, recall, f1 };
                                                                }
                                                            }
                                                        }

                                                        const cellSize = classes.length <= 3 ? 'w-20 h-16' : classes.length <= 5 ? 'w-14 h-12' : 'w-10 h-9';
                                                        const cellText = classes.length <= 5 ? 'text-xs' : 'text-[10px]';
                                                        const cellW = cellSize.split(' ')[0]!;
                                                        const cellH = cellSize.split(' ')[1]!;

                                                        return (
                                                            <div className="flex flex-col items-center w-full">
                                                                {/* ── OVERALL MATRIX — Actual label lives alongside ONLY the data rows ── */}
                                                                <div className="flex flex-col">
                                                                    {/* Predicted header: spacer = row-label (76px) + Actual-label (20px) + gap (4px) = 100px */}
                                                                    <div className="flex items-center mb-1">
                                                                        <div className="w-[100px] shrink-0" />
                                                                        <div className="flex-1 flex items-center justify-center gap-1">
                                                                            <span className="text-[11px] text-gray-400 dark:text-gray-500">Predicted</span>
                                                                            <InfoTooltip text="Columns = what the model predicted. Each column is one class. Read a column down ↓ to see all samples predicted as that class." size="sm" />
                                                                        </div>
                                                                    </div>
                                                                    {/* Col-name headers — same 100px spacer so they sit directly above cells */}
                                                                    <div className="flex mb-0.5">
                                                                        <div className="w-[100px] shrink-0" />
                                                                        {classes.map(c => (
                                                                            <div key={String(c)} className={`${cellW} text-center text-[11px] font-medium text-gray-500 dark:text-gray-400 pb-1 truncate`} title={String(c)}>
                                                                                {String(c)}
                                                                            </div>
                                                                        ))}
                                                                    </div>
                                                                    {/* Body: Actual label sits inside items-stretch so its height = matrix height → perfectly centered */}
                                                                    <div className="flex items-stretch">
                                                                        <div className="flex flex-col items-center justify-center mr-1" style={{ width: '20px' }}>
                                                                            <InfoTooltip text="Rows = actual / true labels. Read a row across → to see where each true class ended up. Green diagonal = correct; red off-diagonal = misclassification." size="sm" />
                                                                            <span className="text-[11px] text-gray-400 dark:text-gray-500" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual</span>
                                                                        </div>
                                                                        <div className="border border-gray-200 dark:border-gray-700 rounded overflow-hidden">
                                                                            {matrix.map((row, i) => {
                                                                                const rowTotal = row.reduce((a, b) => a + b, 0);
                                                                                return (
                                                                                    <div key={i} className="flex">
                                                                                        <div className={`w-[76px] ${cellH} flex items-center justify-end pr-2 text-[11px] font-medium text-gray-500 dark:text-gray-400 truncate border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 shrink-0`} title={String(classes[i])}>
                                                                                            {String(classes[i])}
                                                                                        </div>
                                                                                        {row.map((count, j) => {
                                                                                            const isDiag = i === j;
                                                                                            const intensity = rowTotal > 0 ? count / rowTotal : 0;
                                                                                            const bgColor = isDiag
                                                                                                ? `rgba(34, 197, 94, ${intensity * 0.75 + 0.08})`
                                                                                                : `rgba(239, 68, 68, ${intensity * 0.65 + 0.04})`;
                                                                                            const textColor = intensity > 0.45 ? 'white' : undefined;
                                                                                            const pct = rowTotal > 0 ? ((count / rowTotal) * 100).toFixed(0) : '0';
                                                                                            return (
                                                                                                <div
                                                                                                    key={j}
                                                                                                    className={`${cellSize} flex flex-col items-center justify-center border border-gray-100 dark:border-gray-800 cursor-default`}
                                                                                                    style={{ backgroundColor: bgColor, color: textColor }}
                                                                                                    title={`True: ${classes[i]}, Pred: ${classes[j]}\nCount: ${count}  |  ${pct}% of actual "${classes[i]}"\n${isDiag ? '✓ Correct prediction' : '✗ Misclassification'}`}
                                                                                                >
                                                                                                    <span className={`${cellText} font-mono font-bold leading-none`}>{count}</span>
                                                                                                    <span className="text-[9px] leading-none opacity-75 mt-0.5">{pct}%</span>
                                                                                                </div>
                                                                                            );
                                                                                        })}
                                                                                    </div>
                                                                                );
                                                                            })}
                                                                        </div>
                                                                    </div>
                                                                </div>

                                                                {/* Footer: label + threshold tooltip */}
                                                                <div className="mt-3 flex items-center gap-1.5 text-xs text-gray-400">
                                                                    <span>Confusion Matrix</span>
                                                                    {selectedRocClass && (
                                                                        <>
                                                                            <span className="font-mono text-blue-500 dark:text-blue-400">@ t={threshold.toFixed(2)}</span>
                                                                            <InfoTooltip
                                                                                text={`Threshold rule (≥): a sample is predicted as "${selectedRocClass}" when P("${selectedRocClass}") ≥ ${threshold.toFixed(2)} (equal or above). Otherwise the class with the highest remaining probability wins.\n↑ Raise threshold → fewer positives predicted, lower recall, higher precision.\n↓ Lower threshold → more positives predicted, higher recall, lower precision.\nGreen cells = correct; red cells = errors. Percentages show % of each actual class row.`}
                                                                                align="center"
                                                                            />
                                                                        </>
                                                                    )}
                                                                </div>
                                                                {/* Live metric tiles */}
                                                                {liveMetrics && (
                                                                    <div className="mt-2 grid grid-cols-4 gap-1.5 text-xs w-full">
                                                                        {([
                                                                            { label: 'Accuracy', value: liveMetrics.accuracy, tip: 'Overall fraction of correct predictions (OvR: treats selected class as positive).' },
                                                                            { label: 'Precision', value: liveMetrics.precision, tip: 'Of all samples predicted as this class, how many actually are? High = few false alarms.' },
                                                                            { label: 'Recall', value: liveMetrics.recall, tip: 'Of all actual samples of this class, how many did the model catch? High = few misses.' },
                                                                            { label: 'F1', value: liveMetrics.f1, tip: 'Harmonic mean of Precision and Recall. Balances both — best single metric for imbalanced classes.' },
                                                                        ] as { label: string; value: number; tip: string }[]).map(({ label, value, tip }) => {
                                                                            const color = value >= 0.8 ? 'text-green-600 dark:text-green-400' : value >= 0.6 ? 'text-yellow-500 dark:text-yellow-400' : 'text-red-500 dark:text-red-400';
                                                                            return (
                                                                                <div key={label} className="flex flex-col items-center bg-gray-50 dark:bg-gray-900 rounded px-1.5 py-1.5 gap-0.5">
                                                                                    <div className="flex items-center gap-0.5">
                                                                                        <span className="text-gray-500 dark:text-gray-400">{label}</span>
                                                                                        <InfoTooltip text={tip} size="sm" align="center" />
                                                                                    </div>
                                                                                    <span className={`font-mono font-semibold ${color}`}>{value.toFixed(3)}</span>
                                                                                </div>
                                                                            );
                                                                        })}
                                                                    </div>
                                                                )}
                                                                {/* Binary: show both classes' Prec/Rec/F1 inline — no need to switch to Per Class tab */}
                                                                {classes.length === 2 && (
                                                                    <div className="mt-3 border-t border-gray-100 dark:border-gray-700 pt-3">
                                                                        <div className="flex items-center gap-1 mb-2">
                                                                            <span className="text-[11px] font-medium text-gray-400 dark:text-gray-500">Per Class</span>
                                                                            <InfoTooltip text="Precision, Recall and F1 for each class individually. For binary problems both classes are always shown here." size="sm" />
                                                                        </div>
                                                                        <div className="grid grid-cols-2 gap-2">
                                                                            {classes.map((cls, clsIdx) => {
                                                                                const btp = matrix[clsIdx]?.[clsIdx] ?? 0;
                                                                                const bfp = matrix.reduce((s, row, ri) => ri !== clsIdx ? s + (row[clsIdx] ?? 0) : s, 0);
                                                                                const bfn = (matrix[clsIdx] ?? []).reduce((s, v, ci) => ci !== clsIdx ? s + v : s, 0);
                                                                                const bprec = (btp + bfp) > 0 ? btp / (btp + bfp) : 0;
                                                                                const brec  = (btp + bfn) > 0 ? btp / (btp + bfn) : 0;
                                                                                const bf1   = bprec + brec > 0 ? (2 * bprec * brec) / (bprec + brec) : 0;
                                                                                return (
                                                                                    <div key={String(cls)} className="flex flex-col items-center p-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                                                                                        <span className="text-[10px] font-semibold text-gray-600 dark:text-gray-300 mb-1.5">{String(cls)}</span>
                                                                                        <div className="grid grid-cols-3 gap-1 text-[10px] w-full">
                                                                                            {([{ l: 'Prec', v: bprec }, { l: 'Rec', v: brec }, { l: 'F1', v: bf1 }] as { l: string; v: number }[]).map(({ l, v }) => (
                                                                                                <div key={l} className="flex flex-col items-center bg-white dark:bg-gray-800 rounded py-1">
                                                                                                    <span className="text-gray-400">{l}</span>
                                                                                                    <span className={`font-mono font-semibold ${v >= 0.8 ? 'text-green-500' : v >= 0.6 ? 'text-yellow-500' : 'text-red-500'}`}>{v.toFixed(2)}</span>
                                                                                                </div>
                                                                                            ))}
                                                                                        </div>
                                                                                    </div>
                                                                                );
                                                                            })}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        );
                                                    })()}
                                                </div>

                                                {/* ROC Curve (if available) */}
                                                {splitData.y_proba && (
                                                    <div className="h-[340px] w-full relative group" id={`${splitName}-roc`}>
                                                        <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                           <button 
                                                             onClick={() => void handleDownload(`${splitName}-roc`, `${splitName}_roc_curve`)}
                                                             disabled={downloadingChart === `${splitName}-roc`}
                                                             className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                             title="Download Graph"
                                                           >
                                                              {downloadingChart === `${splitName}-roc` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-roc` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                           </button>
                                                        </div>
                                                        {(() => {
                                                            if (!selectedRocClass) return <div className="text-center text-xs text-gray-400">Select a class</div>;
                                                            const rocData = calculateROC(splitData.y_true, splitData.y_proba!, selectedRocClass);
                                                            if (!rocData) return <div className="text-center text-xs text-gray-400">ROC not available (multiclass or missing proba)</div>;
                                                            // Embed random-classifier diagonal value so both lines share one data array
                                                            const rocDataMerged = rocData.map(pt => ({ ...pt, random: pt.fpr }));

                                                            // AUC via trapezoid rule
                                                            const auc = rocData.reduce((sum, pt, i) => {
                                                                if (i === 0) return 0;
                                                                const prev = rocData[i - 1]!;
                                                                return sum + Math.abs(pt.fpr - prev.fpr) * (pt.tpr + prev.tpr) / 2;
                                                            }, 0);

                                                            // Operating point at current threshold
                                                            const proba = splitData.y_proba!;
                                                            let operatingPoint: { fpr: number; tpr: number } | null = null;
                                                            const labelList = proba.labels && proba.labels.length === proba.classes.length ? proba.labels : undefined;
                                                            const classIndex = (labelList ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                                            if (classIndex !== -1) {
                                                                const scores = proba.values.map(v => v[classIndex] ?? 0);
                                                                const actual = splitData.y_true.map(t => String(t) === selectedRocClass ? 1 : 0);
                                                                const totalPos = actual.filter(a => a === 1).length;
                                                                const totalNeg = actual.length - totalPos;
                                                                if (totalPos > 0 && totalNeg > 0) {
                                                                    let tp = 0, fp = 0;
                                                                    scores.forEach((s, i) => {
                                                                        if (s >= threshold) {
                                                                            if (actual[i] === 1) tp++;
                                                                            else fp++;
                                                                        }
                                                                    });
                                                                    operatingPoint = { fpr: fp / totalNeg, tpr: tp / totalPos };
                                                                }
                                                            }

                                                            return (
                                                                <>
                                                                    <div className="flex items-center justify-center gap-1.5 mb-1">
                                                                        <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">
                                                                            ROC Curve — {selectedRocClass}
                                                                            <span className="ml-2 font-mono text-purple-600 dark:text-purple-400">AUC={auc.toFixed(3)}</span>
                                                                        </h5>
                                                                        <InfoTooltip
                                                                            text={`ROC (Receiver Operating Characteristic) curve for class "${selectedRocClass}" vs all others. AUC=${auc.toFixed(3)}: closer to 1.0 is better; 0.5 = random. The red dot marks where the model operates at threshold t=${threshold.toFixed(2)} — drag the slider to move it along the curve and see the precision/recall trade-off in real time.`}
                                                                            align="center"
                                                                        />
                                                                    </div>
                                                                    {/* TPR / FPR definitions */}
                                                                    <div className="flex items-center justify-center gap-4 text-[10px] text-gray-400 dark:text-gray-500 mb-1">
                                                                        <div className="flex items-center gap-0.5">
                                                                            <span className="font-semibold">TPR</span>
                                                                            <InfoTooltip text="True Positive Rate (Recall / Sensitivity): TP ÷ (TP + FN). Of all actual positives, how many did the model correctly detect? Higher = fewer misses. This is the Y-axis." size="sm" />
                                                                            <span className="ml-0.5 font-mono">= TP / (TP+FN)</span>
                                                                        </div>
                                                                        <span className="text-gray-300 dark:text-gray-600">·</span>
                                                                        <div className="flex items-center gap-0.5">
                                                                            <span className="font-semibold">FPR</span>
                                                                            <InfoTooltip text="False Positive Rate (Fall-out): FP ÷ (FP + TN). Of all actual negatives, how many were incorrectly flagged as positive? Lower = fewer false alarms. This is the X-axis." size="sm" />
                                                                            <span className="ml-0.5 font-mono">= FP / (FP+TN)</span>
                                                                        </div>
                                                                    </div>
                                                                    <ResponsiveContainer width="100%" height="92%">
                                                                        <ComposedChart data={rocDataMerged} margin={{ top: 5, right: 20, bottom: 42, left: 40 }}>
                                                                            <defs>
                                                                                <linearGradient id="aucFill" x1="0" y1="0" x2="0" y2="1">
                                                                                    <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.25} />
                                                                                    <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.03} />
                                                                                </linearGradient>
                                                                            </defs>
                                                                            <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                            <XAxis
                                                                                type="number"
                                                                                dataKey="fpr"
                                                                                domain={[0, 1]}
                                                                                tickFormatter={(v: number) => v.toFixed(1)}
                                                                                tick={{ fontSize: 10 }}
                                                                                label={{ value: 'FPR (Fall-out)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }}
                                                                            />
                                                                            <YAxis
                                                                                type="number"
                                                                                dataKey="tpr"
                                                                                domain={[0, 1]}
                                                                                tickFormatter={(v: number) => v.toFixed(1)}
                                                                                tick={{ fontSize: 10 }}
                                                                                label={{ value: 'TPR (Recall)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: '#9ca3af' }}
                                                                            />
                                                                            <Tooltip
                                                                                content={({ active, payload }) => {
                                                                                    if (!active || !payload?.length) return null;
                                                                                    const d = payload[0]?.payload as { fpr: number; tpr: number };
                                                                                    return (
                                                                                        <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-sm p-2 text-xs space-y-0.5">
                                                                                            <p className="font-semibold text-gray-600 dark:text-gray-300 mb-1">ROC point</p>
                                                                                            <p className="text-gray-700 dark:text-gray-200">TPR (Recall) <span className="font-mono text-purple-600 dark:text-purple-400">{d.tpr.toFixed(3)}</span></p>
                                                                                            <p className="text-gray-500 dark:text-gray-400">FPR (Fall-out) <span className="font-mono">{d.fpr.toFixed(3)}</span></p>
                                                                                            <p className="text-gray-400 dark:text-gray-500 text-[10px] pt-0.5 border-t border-gray-100 dark:border-gray-700">Precision = TP / (TP+FP) &nbsp;·&nbsp; Recall = TP / (TP+FN)</p>
                                                                                        </div>
                                                                                    );
                                                                                }}
                                                                            />
                                                                            {/* Gradient fill — AUC area */}
                                                                            <Area type="monotone" dataKey="tpr" stroke="none" fill="url(#aucFill)" isAnimationActive={false} legendType="none" />
                                                                            {/* ROC curve */}
                                                                            <Line type="monotone" dataKey="tpr" stroke="#8b5cf6" dot={false} activeDot={{ r: 5 }} strokeWidth={2.5} name="ROC" isAnimationActive={false} />
                                                                            {/* Random classifier diagonal — same data array, no separate data prop */}
                                                                            <Line type="linear" dataKey="random" stroke="#d1d5db" strokeDasharray="4 3" dot={false} tooltipType="none" legendType="none" strokeWidth={1} isAnimationActive={false} />
                                                                            {/* Operating point at current threshold — ReferenceDot doesn't affect tooltip data domain */}
                                                                            {operatingPoint && (
                                                                                <ReferenceDot
                                                                                    x={operatingPoint.fpr}
                                                                                    y={operatingPoint.tpr}
                                                                                    r={7}
                                                                                    fill="#ef4444"
                                                                                    stroke="#fff"
                                                                                    strokeWidth={2}
                                                                                    label={{ value: `t=${threshold.toFixed(2)}`, position: 'top', fontSize: 10, fill: '#ef4444' }}
                                                                                />
                                                                            )}
                                                                        </ComposedChart>
                                                                    </ResponsiveContainer>
                                                                </>
                                                            );
                                                        })()}
                                                    </div>
                                                )}
                                            </div>
                                        {/* PR Curve + Score Distribution — classification only, shown when a class is selected */}
                                        {splitData.y_proba && selectedRocClass && (() => {
                                            const prData = calculatePR(splitData.y_true, splitData.y_proba, selectedRocClass);
                                            const scoreDist = getScoreDistribution(splitData.y_true, splitData.y_proba, selectedRocClass);
                                            if (!prData || !scoreDist) return null;
                                            // AUC-PR via trapezoidal rule
                                            let aucPR = 0;
                                            for (let i = 1; i < prData.length; i++) {
                                                const dr = (prData[i]!.recall - prData[i - 1]!.recall);
                                                aucPR += dr * ((prData[i]!.precision + prData[i - 1]!.precision) / 2);
                                            }
                                            // Embed no-skill baseline so all lines share one data array
                                            const prTotal = splitData.y_true.length;
                                            const prPos = splitData.y_true.filter(y => String(y) === selectedRocClass).length;
                                            const noSkill = prTotal > 0 ? prPos / prTotal : 0;
                                            const prDataMerged = prData.map(pt => ({ ...pt, noSkill }));
                                            return (
                                                <>
                                                <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-6">
                                                    {/* PR Curve */}
                                                    <div className="h-[280px] relative group" id={`${splitName}-pr-curve`}>
                                                        <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                           <button
                                                             onClick={() => void handleDownload(`${splitName}-pr-curve`, `${splitName}_pr_curve`)}
                                                             disabled={downloadingChart === `${splitName}-pr-curve`}
                                                             className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                             title="Download Graph"
                                                           >
                                                             {downloadingChart === `${splitName}-pr-curve` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-pr-curve` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                           </button>
                                                        </div>
                                                        <div className="flex items-center justify-center gap-1.5 mb-1">
                                                            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">
                                                                PR Curve — {selectedRocClass}
                                                                <span className="ml-2 font-mono text-blue-600 dark:text-blue-400">AUC-PR={aucPR.toFixed(3)}</span>
                                                            </h5>
                                                            <InfoTooltip
                                                                text={`Precision-Recall curve for class "${selectedRocClass}". AUC-PR=${aucPR.toFixed(3)}: closer to 1.0 is better. Unlike ROC, PR is not inflated by true negatives — prefer PR for imbalanced classes. High precision = few false alarms; high recall = few misses.`}
                                                                align="center"
                                                            />
                                                        </div>
                                                        <ResponsiveContainer width="100%" height="90%">
                                                            <ComposedChart data={prDataMerged} margin={{ top: 5, right: 20, bottom: 35, left: 40 }}>
                                                                <defs>
                                                                    <linearGradient id="prFill" x1="0" y1="0" x2="0" y2="1">
                                                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                                                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                                                                    </linearGradient>
                                                                </defs>
                                                                <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                <XAxis type="number" dataKey="recall" domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} tick={{ fontSize: 10 }} label={{ value: 'Recall', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
                                                                <YAxis type="number" dataKey="precision" domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} tick={{ fontSize: 10 }} label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: '#9ca3af' }} />
                                                                <Tooltip
                                                                    content={({ active, payload }) => {
                                                                        if (active && payload?.length) {
                                                                            const entry = payload.find(p => (p.payload as Record<string, unknown>).score != null);
                                                                            if (!entry) return null;
                                                                            const d = entry.payload as { recall: number; precision: number };
                                                                            return (
                                                                                <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                                                                                    <p className="font-semibold text-gray-600 dark:text-gray-300 mb-1">PR point</p>
                                                                                    <p>Recall <span className="font-mono text-blue-600 dark:text-blue-400">{d.recall.toFixed(3)}</span></p>
                                                                                    <p>Precision <span className="font-mono">{d.precision.toFixed(3)}</span></p>
                                                                                </div>
                                                                            );
                                                                        }
                                                                        return null;
                                                                    }}
                                                                />
                                                                <Area type="monotone" dataKey="precision" stroke="none" fill="url(#prFill)" isAnimationActive={false} legendType="none" />
                                                                <Line type="monotone" dataKey="precision" stroke="#3b82f6" dot={false} activeDot={{ r: 5 }} strokeWidth={2.5} name="PR" isAnimationActive={false} />
                                                                {/* No-skill baseline — same data array, no separate data prop */}
                                                                <Line type="linear" dataKey="noSkill" stroke="#d1d5db" strokeDasharray="4 3" dot={false} tooltipType="none" legendType="none" strokeWidth={1} isAnimationActive={false} />
                                                                {/* Operating point: last PR point where score >= threshold */}
                                                                {(() => {
                                                                    const opPoint = [...prData].reverse().find(p => p.score >= threshold) ?? prData[prData.length - 1];
                                                                    if (!opPoint) return null;
                                                                    return (
                                                                        <ReferenceDot
                                                                            x={opPoint.recall}
                                                                            y={opPoint.precision}
                                                                            r={6}
                                                                            fill="#ef4444"
                                                                            stroke="#fff"
                                                                            strokeWidth={2}
                                                                            label={{ value: `t=${threshold.toFixed(2)}`, position: 'top', fontSize: 10, fill: '#ef4444' }}
                                                                        />
                                                                    );
                                                                })()}
                                                            </ComposedChart>
                                                        </ResponsiveContainer>
                                                    </div>

                                                    {/* Score Distribution Histogram */}
                                                    <div className="h-[280px] relative group" id={`${splitName}-score-dist`}>
                                                        <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                           <button
                                                             onClick={() => void handleDownload(`${splitName}-score-dist`, `${splitName}_score_distribution`)}
                                                             disabled={downloadingChart === `${splitName}-score-dist`}
                                                             className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
                                                             title="Download Graph"
                                                           >
                                                             {downloadingChart === `${splitName}-score-dist` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-score-dist` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                           </button>
                                                        </div>
                                                        <div className="flex items-center justify-center gap-1.5 mb-1">
                                                            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">
                                                                Score Distribution — {selectedRocClass}
                                                            </h5>
                                                            <InfoTooltip
                                                                text={`How well does the model separate the two classes? Green bars = samples that truly belong to "${selectedRocClass}"; red = samples that don't. Good separation: green stacks up near score 1, red stacks up near 0, with little overlap in the middle. Heavy overlap around the current threshold (red line) means many borderline predictions — consider raising or lowering t.`}
                                                                align="center"
                                                            />
                                                        </div>
                                                        <ResponsiveContainer width="100%" height="90%">
                                                            <BarChart data={scoreDist} barCategoryGap="1%" margin={{ top: 5, right: 20, bottom: 35, left: 40 }}>
                                                                <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                <XAxis dataKey="range" tick={{ fontSize: 9 }} interval={3} label={{ value: `P("${selectedRocClass}")`, position: 'insideBottom', offset: -8, fontSize: 11, fill: '#9ca3af' }} />
                                                                <YAxis tick={{ fontSize: 10 }} label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle' }, fontSize: 11, fill: '#9ca3af' }} />
                                                                <Tooltip
                                                                    content={({ active, payload }) => {
                                                                        if (active && payload?.length) {
                                                                            const d = payload[0]!.payload as { range: string; pos: number; neg: number };
                                                                            return (
                                                                                <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 shadow-sm rounded text-xs">
                                                                                    <p className="font-semibold mb-1">Score bin ≥ {d.range}</p>
                                                                                    <p className="text-green-600 dark:text-green-400">Actual positive: {d.pos}</p>
                                                                                    <p className="text-red-500 dark:text-red-400">Actual negative: {d.neg}</p>
                                                                                </div>
                                                                            );
                                                                        }
                                                                        return null;
                                                                    }}
                                                                />
                                                                {/* Threshold reference line at closest bin */}
                                                                <ReferenceLine x={scoreDist[Math.min(Math.floor(threshold * 20), 19)]?.range ?? ''} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: `t=${threshold.toFixed(2)}`, position: 'top', fontSize: 10, fill: '#ef4444' }} />
                                                                <Bar dataKey="pos" name="Actual +" stackId="a" fill="#22c55e" fillOpacity={0.7} isAnimationActive={false} />
                                                                <Bar dataKey="neg" name="Actual −" stackId="a" fill="#ef4444" fillOpacity={0.5} isAnimationActive={false} />
                                                                <Legend verticalAlign="top" height={18} iconSize={10} formatter={(value) => <span className="text-xs text-gray-600 dark:text-gray-400">{value}</span>} />
                                                            </BarChart>
                                                        </ResponsiveContainer>
                                                    </div>
                                                </div>

                                                {/* Calibration Curve + Cumulative Gains + MCC vs Threshold */}
                                                <div className="mt-5 grid grid-cols-1 lg:grid-cols-3 gap-6">
                                                    {/* Calibration Curve */}
                                                    {(() => {
                                                        const calRaw = getCalibrationData(splitData.y_true, splitData.y_proba!, selectedRocClass);
                                                        if (!calRaw || calRaw.length < 2) return null;
                                                        // Embed perfect-calibration value into every data point so both lines
                                                        // share one data array — prevents Recharts snapping to the 2-point
                                                        // diagonal and missing the calibration bins in tooltip payload.
                                                        const calData = calRaw.map(pt => ({ ...pt, perfect: pt.midpoint }));
                                                        return (
                                                            <div className="h-[260px] relative group" id={`${splitName}-calibration`}>
                                                                <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                                    <button onClick={() => void handleDownload(`${splitName}-calibration`, `${splitName}_calibration`)} disabled={downloadingChart === `${splitName}-calibration`} className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50" title="Download Graph">
                                                                        {downloadingChart === `${splitName}-calibration` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-calibration` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                                    </button>
                                                                </div>
                                                                <div className="flex items-center justify-center gap-1.5 mb-1">
                                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Calibration — {selectedRocClass}</h5>
                                                                    <InfoTooltip text={`Is the model's confidence score trustworthy? X = what score the model gave; Y = how often that class actually appeared at that score. If the model says 80% confidence but only 40% of those samples are truly positive, it is over-confident (point sits below the diagonal). A well-calibrated model's dots hug the diagonal closely.`} align="center" size="sm" />
                                                                </div>
                                                                <ResponsiveContainer width="100%" height="90%">
                                                                    <ComposedChart data={calData} margin={{ top: 5, right: 15, bottom: 32, left: 40 }}>
                                                                        <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                        <XAxis type="number" dataKey="midpoint" domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} tick={{ fontSize: 10 }} label={{ value: 'Mean predicted prob.', position: 'insideBottom', offset: -8, fontSize: 10, fill: '#9ca3af' }} />
                                                                        <YAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} tick={{ fontSize: 10 }} label={{ value: 'Fraction positives', angle: -90, position: 'insideLeft', offset: 8, fontSize: 10, fill: '#9ca3af' }} />
                                                                        <Tooltip content={({ active, payload }) => {
                                                                            if (!active || !payload?.length) return null;
                                                                            const d = payload[0]!.payload as { midpoint: number; fracPos: number; count: number; perfect: number };
                                                                            return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 rounded text-xs shadow-sm"><p className="font-semibold mb-1">Calibration bin</p><p>Predicted confidence <span className="font-mono text-purple-600 dark:text-purple-400">{d.midpoint.toFixed(2)}</span></p><p>Actual positive rate <span className="font-mono">{d.fracPos.toFixed(3)}</span></p><p className="text-gray-400 text-[10px]">{d.count} samples in this bin</p></div>;
                                                                        }} />
                                                                        {/* Perfect calibration diagonal — uses same data array so tooltip fires correctly */}
                                                                        <Line type="linear" dataKey="perfect" stroke="#d1d5db" strokeDasharray="4 3" dot={false} legendType="none" tooltipType="none" strokeWidth={1} isAnimationActive={false} />
                                                                        <Line type="linear" dataKey="fracPos" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 5, fill: '#8b5cf6', stroke: '#fff', strokeWidth: 1.5 }} activeDot={{ r: 7 }} isAnimationActive={false} name="Calibration" />
                                                                    </ComposedChart>
                                                                </ResponsiveContainer>
                                                            </div>
                                                        );
                                                    })()}

                                                    {/* Cumulative Gains */}
                                                    {(() => {
                                                        const gainsData = getCumulativeGainsData(splitData.y_true, splitData.y_proba!, selectedRocClass);
                                                        if (!gainsData) return null;
                                                        // Embed random baseline so all lines share one data array
                                                        const gainsDataMerged = gainsData.map(pt => ({ ...pt, random: pt.pct }));
                                                        return (
                                                            <div className="h-[260px] relative group" id={`${splitName}-cum-gains`}>
                                                                <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                                    <button onClick={() => void handleDownload(`${splitName}-cum-gains`, `${splitName}_cumulative_gains`)} disabled={downloadingChart === `${splitName}-cum-gains`} className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50" title="Download Graph">
                                                                        {downloadingChart === `${splitName}-cum-gains` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-cum-gains` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                                    </button>
                                                                </div>
                                                                <div className="flex items-center justify-center gap-1.5 mb-1">
                                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">Cumulative Gains — {selectedRocClass}</h5>
                                                                    <InfoTooltip text={`Samples sorted by score (highest first). X = % of population contacted; Y = % of total positives captured. The diagonal = random. If the blue curve reaches 80% of positives after contacting only 40% — the model has 2× lift there. Useful for targeting campaigns.`} align="center" size="sm" />
                                                                </div>
                                                                <ResponsiveContainer width="100%" height="90%">
                                                                    <ComposedChart data={gainsDataMerged} margin={{ top: 5, right: 15, bottom: 32, left: 40 }}>
                                                                        <defs>
                                                                            <linearGradient id="gainsFill" x1="0" y1="0" x2="0" y2="1">
                                                                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                                                                                <stop offset="95%" stopColor="#10b981" stopOpacity={0.02} />
                                                                            </linearGradient>
                                                                        </defs>
                                                                        <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                        <XAxis type="number" dataKey="pct" domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 10 }} label={{ value: '% Population', position: 'insideBottom', offset: -8, fontSize: 10, fill: '#9ca3af' }} />
                                                                        <YAxis type="number" dataKey="gain" domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 10 }} label={{ value: '% Positives', angle: -90, position: 'insideLeft', offset: 8, fontSize: 10, fill: '#9ca3af' }} />
                                                                        <Tooltip content={({ active, payload }) => {
                                                                            if (!active || !payload?.length) return null;
                                                                            const entry = payload.find(p => (p.payload as Record<string, unknown>).lift != null);
                                                                            if (!entry) return null;
                                                                            const d = entry.payload as { pct: number; gain: number; lift: number };
                                                                            return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 rounded text-xs shadow-sm"><p className="font-semibold mb-1">Gains point</p><p>Population <span className="font-mono text-green-600 dark:text-green-400">{(d.pct * 100).toFixed(1)}%</span></p><p>Positives caught <span className="font-mono">{(d.gain * 100).toFixed(1)}%</span></p><p className="text-gray-500">Lift <span className="font-mono">{d.lift.toFixed(2)}×</span></p></div>;
                                                                        }} />
                                                                        {/* Random baseline — same data array, no separate data prop */}
                                                                        <Line type="linear" dataKey="random" stroke="#d1d5db" strokeDasharray="4 3" dot={false} tooltipType="none" legendType="none" strokeWidth={1} isAnimationActive={false} />
                                                                        <Area type="monotone" dataKey="gain" stroke="none" fill="url(#gainsFill)" isAnimationActive={false} legendType="none" />
                                                                        <Line type="monotone" dataKey="gain" stroke="#10b981" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} isAnimationActive={false} name="Gains" />
                                                                    </ComposedChart>
                                                                </ResponsiveContainer>
                                                            </div>
                                                        );
                                                    })()}

                                                    {/* MCC vs Threshold */}
                                                    {(() => {
                                                        const mccData = getMCCByThreshold(splitData.y_true, splitData.y_proba!, selectedRocClass);
                                                        if (!mccData) return null;
                                                        const bestMCC = mccData.reduce((best, pt) => pt.mcc > best.mcc ? pt : best, mccData[0]!);
                                                        return (
                                                            <div className="h-[260px] relative group" id={`${splitName}-mcc`}>
                                                                <div className="absolute top-0 right-0 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
                                                                    <button onClick={() => void handleDownload(`${splitName}-mcc`, `${splitName}_mcc`)} disabled={downloadingChart === `${splitName}-mcc`} className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50" title="Download Graph">
                                                                        {downloadingChart === `${splitName}-mcc` ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === `${splitName}-mcc` ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
                                                                    </button>
                                                                </div>
                                                                <div className="flex items-center justify-center gap-1.5 mb-1">
                                                                    <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 text-center">
                                                                        MCC vs Threshold — {selectedRocClass}
                                                                        <span className="ml-2 font-mono text-orange-500 dark:text-orange-400">best t={bestMCC.threshold.toFixed(2)} (MCC={bestMCC.mcc.toFixed(2)})</span>
                                                                    </h5>
                                                                    <InfoTooltip text={`Which threshold gives the best overall balance between precision and recall — even for imbalanced classes? MCC (Matthews Correlation Coefficient) scores from −1 (completely wrong) to +1 (perfect); 0 = random guessing. The orange peak is the threshold where MCC is highest. The red line is your current threshold. If the red line is far from the orange peak, consider moving your threshold there for better overall performance.`} align="center" size="sm" />
                                                                </div>
                                                                <ResponsiveContainer width="100%" height="90%">
                                                                    <ComposedChart data={mccData} margin={{ top: 5, right: 15, bottom: 32, left: 40 }}>
                                                                        <CartesianGrid strokeDasharray="3 3" opacity={0.08} />
                                                                        <XAxis type="number" dataKey="threshold" domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(1)} tick={{ fontSize: 10 }} label={{ value: 'Threshold', position: 'insideBottom', offset: -8, fontSize: 10, fill: '#9ca3af' }} />
                                                                        <YAxis type="number" dataKey="mcc" domain={[-1, 1]} tickFormatter={(v: number) => v.toFixed(1)} tick={{ fontSize: 10 }} label={{ value: 'MCC', angle: -90, position: 'insideLeft', offset: 8, fontSize: 10, fill: '#9ca3af' }} />
                                                                        <Tooltip content={({ active, payload }) => {
                                                                            if (!active || !payload?.length) return null;
                                                                            const d = payload[0]!.payload as { threshold: number; mcc: number };
                                                                            return <div className="bg-white dark:bg-gray-800 p-2 border border-gray-200 dark:border-gray-700 rounded text-xs shadow-sm"><p className="font-semibold mb-1">MCC point</p><p>t = <span className="font-mono">{d.threshold.toFixed(2)}</span></p><p>MCC = <span className="font-mono text-orange-500">{d.mcc.toFixed(4)}</span></p></div>;
                                                                        }} />
                                                                        <ReferenceLine y={0} stroke="#d1d5db" strokeDasharray="3 3" strokeWidth={1} />
                                                                        <ReferenceLine x={threshold} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1.5} label={{ value: `t=${threshold.toFixed(2)}`, position: 'top', fontSize: 10, fill: '#ef4444' }} />
                                                                        <ReferenceLine x={bestMCC.threshold} stroke="#f97316" strokeDasharray="4 2" strokeWidth={1.5} label={{ value: `best`, position: 'insideTopLeft', fontSize: 9, fill: '#f97316' }} />
                                                                        <Line type="monotone" dataKey="mcc" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} name="MCC" />
                                                                    </ComposedChart>
                                                                </ResponsiveContainer>
                                                            </div>
                                                        );
                                                    })()}
                                                </div>
                                                </>
                                            );
                                        })()}
    </>
  );
};