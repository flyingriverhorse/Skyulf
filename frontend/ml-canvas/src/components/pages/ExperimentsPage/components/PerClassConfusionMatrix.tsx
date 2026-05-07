/**
 * Per-class confusion matrix view for multiclass classification.
 * Renders one OvR (one-vs-rest) 2x2 matrix per class for each enabled split,
 * with live recall/precision/F1 chips and a plain-language verdict.
 *
 * Caller is responsible for the outer gate; this component only enforces the
 * "multiclass with y_proba" precondition and renders nothing otherwise.
 */

import React from 'react';
import { Loader2, Check, Download } from 'lucide-react';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import type { EvaluationSplit, EvaluationData } from '../types';
import { calculateConfusionMatrix } from '../utils/classificationCharts';

interface Props {
  evaluationData: EvaluationData;
  selectedRocClass: string | null;
  threshold: number;
  showTrainMetrics: boolean;
  showTestMetrics: boolean;
  showValMetrics: boolean;
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const PerClassConfusionMatrix: React.FC<Props> = ({
  evaluationData,
  selectedRocClass,
  threshold,
  showTrainMetrics,
  showTestMetrics,
  showValMetrics,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  if ((evaluationData.splits.train?.y_proba?.classes.length ?? 0) <= 2) return null;


                            // Compute confusion matrix for a split with OvR threshold applied
                            const getMatrix = (splitData: EvaluationSplit) => {
                                const proba = splitData.y_proba;
                                let yTrue: (string | number)[] = splitData.y_true;
                                let yPred: (string | number)[] = splitData.y_pred;
                                if (proba?.labels && proba.labels.length === proba.classes.length) {
                                    const lm = new Map<string, string | number>();
                                    proba.labels.forEach((l, i) => { const c = proba.classes[i]; if (c !== undefined) lm.set(String(l), c); });
                                    yTrue = yTrue.map(y => lm.get(String(y)) ?? y);
                                    yPred = yPred.map(y => lm.get(String(y)) ?? y);
                                }
                                if (proba && selectedRocClass) {
                                    const ll = proba.labels?.length === proba.classes.length ? proba.labels : undefined;
                                    const posIdx = (ll ?? proba.classes).findIndex(c => String(c) === selectedRocClass);
                                    if (posIdx !== -1) {
                                        const posVal = proba.classes[posIdx];
                                        const orig = [...yPred];
                                        if (posVal !== undefined) {
                                            yPred = proba.values.map((v, i) => {
                                                if ((v[posIdx] ?? 0) >= threshold) return posVal;
                                                let bi = -1, bp = -Infinity;
                                                v.forEach((p, idx) => { if (idx !== posIdx && p > bp) { bp = p; bi = idx; } });
                                                return bi >= 0 ? (proba.classes[bi] ?? orig[i]!) : (orig[i]!);
                                            });
                                        }
                                    }
                                }
                                return calculateConfusionMatrix(yTrue, yPred, proba?.classes);
                            };

                            const renderSplitPerClass = (splitName: string, splitData: EvaluationSplit) => {
                                const { classes, matrix } = getMatrix(splitData);
                                const splitId = `per-class-${splitName}`;
                                return (
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-1.5 mb-1">
                                            <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-300 capitalize">{splitName} Set</h4>
                                            <button
                                                id={`${splitId}-dl`}
                                                onClick={() => void handleDownload(splitId, `${splitName}_per_class`)}
                                                disabled={downloadingChart === splitId}
                                                className="p-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-400 hover:text-blue-600 disabled:opacity-50"
                                                title="Download Per-Class View"
                                            >
                                                {downloadingChart === splitId ? <Loader2 className="w-3 h-3 animate-spin" /> : doneChart === splitId ? <Check className="w-3 h-3 text-green-500" /> : <Download className="w-3 h-3" />}
                                            </button>
                                        </div>
                                        <div id={splitId} className={`grid ${classes.length <= 4 ? 'grid-cols-2' : 'grid-cols-3'} gap-2`}>
                                            {classes.map((cls, clsIdx) => {
                                                const tp = matrix[clsIdx]?.[clsIdx] ?? 0;
                                                const fp = matrix.reduce((s, row, ri) => ri !== clsIdx ? s + (row[clsIdx] ?? 0) : s, 0);
                                                const fn = (matrix[clsIdx] ?? []).reduce((s, v, ci) => ci !== clsIdx ? s + v : s, 0);
                                                const total = matrix.flat().reduce((a, b) => a + b, 0);
                                                const tn = total - tp - fp - fn;
                                                const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
                                                const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
                                                const f1c = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
                                                const isHighlighted = String(cls) === selectedRocClass;
                                                const otherClasses = classes.filter((_, i) => i !== clsIdx);
                                                // Binary: use actual other class name; multiclass: 'Others'
                                                const otherLabel = otherClasses.length === 1 ? String(otherClasses[0]) : 'Others';
                                                const rowLabels = [String(cls), otherLabel];
                                                const cellLbls = [['TP', 'FN'], ['FP', 'TN']];
                                                const recPct = Math.round(rec * 100);
                                                const fnCount = tp + fn > 0 ? fn : 0;
                                                const insight =
                                                    f1c >= 0.8 ? `t=${threshold.toFixed(2)}: catches ${recPct}% of ${String(cls)} — strong.`
                                                    : f1c >= 0.6 ? `t=${threshold.toFixed(2)}: catches ${recPct}% of ${String(cls)} — room to improve.`
                                                    : `t=${threshold.toFixed(2)}: only ${recPct}% caught — model struggles here.`;
                                                const insightTip = [
                                                    `Out of ${tp + fnCount} actual "${String(cls)}" samples, the model correctly caught ${tp} of them = ${recPct}%.`,
                                                    ``,
                                                    `How: Recall = TP ÷ (TP + FN) = ${tp} ÷ ${tp + fnCount} = ${recPct}%`,
                                                    `  TP ${tp}: labelled "${String(cls)}" and actually "${String(cls)}" ✓`,
                                                    `  FN ${fnCount}: actually "${String(cls)}" but predicted as something else ✗`,
                                                    ``,
                                                    `Threshold (≥ ${threshold.toFixed(2)}): the model predicts "${String(cls)}" only when it is ${Math.round(threshold * 100)}%+ confident.`,
                                                    `↑ Raise threshold → harder to trigger, fewer catches (recall ↓), more precise (precision ↑).`,
                                                    `↓ Lower threshold → easier to trigger, more catches (recall ↑), more false alarms (precision ↓).`,
                                                ].join('\n');
                                                const insightColor = f1c >= 0.8 ? 'text-green-600 dark:text-green-400' : f1c >= 0.6 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-500 dark:text-red-400';
                                                return (
                                                    <div key={String(cls)} className={`flex flex-col items-center p-2 rounded-lg border ${isHighlighted ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700'}`}>
                                                        <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1.5 w-full text-center truncate" title={`${String(cls)} vs Rest`}>{String(cls)} vs Rest</span>
                                                        <div className="flex flex-col">
                                                            <div className="flex mb-0.5 gap-0.5" style={{ marginLeft: '60px' }}>
                                                                <span className="w-14 text-center text-[9px] text-gray-500 dark:text-gray-400 truncate font-medium" title={String(cls)}>{String(cls)}</span>
                                                                <span className="w-14 text-center text-[9px] text-gray-500 dark:text-gray-400 truncate font-medium" title={otherClasses.map(String).join(', ')}>{otherLabel}</span>
                                                            </div>
                                                            {[[tp, fn], [fp, tn]].map((row2, ri) => (
                                                                <div key={ri} className="flex items-center gap-0.5 mb-0.5">
                                                                    <span className="text-right text-[9px] text-gray-500 dark:text-gray-400 pr-1 truncate font-medium" style={{ width: '60px' }} title={rowLabels[ri]}>{rowLabels[ri]}</span>
                                                                    {row2.map((count, ci) => {
                                                                        const isCorrect = ri === ci;
                                                                        const rowMax = Math.max(...row2, 1);
                                                                        const bg = isCorrect
                                                                            ? `rgba(34,197,94,${Math.min((count / rowMax) * 0.75 + 0.1, 0.85)})`
                                                                            : `rgba(239,68,68,${Math.min((count / rowMax) * 0.65 + 0.05, 0.75)})`;
                                                                        return (
                                                                            <div key={ci} className="w-14 h-11 flex flex-col items-center justify-center rounded border border-gray-100 dark:border-gray-700 cursor-default" style={{ backgroundColor: bg }} title={`${cellLbls[ri]?.[ci] ?? ''}=${count}`}>
                                                                                <span className="text-[11px] font-mono font-bold leading-none">{count}</span>
                                                                                <span className="text-[9px] font-semibold opacity-80 mt-0.5">{cellLbls[ri]?.[ci]}</span>
                                                                            </div>
                                                                        );
                                                                    })}
                                                                </div>
                                                            ))}
                                                        </div>
                                                        <div className="mt-1.5 grid grid-cols-3 gap-1 text-[10px] w-full">
                                                            {([{ l: 'Prec', v: prec }, { l: 'Rec', v: rec }, { l: 'F1', v: f1c }] as { l: string; v: number }[]).map(({ l, v }) => (
                                                                <div key={l} className="flex flex-col items-center bg-gray-50 dark:bg-gray-900 rounded py-1">
                                                                    <span className="text-gray-400">{l}</span>
                                                                    <span className={`font-mono font-semibold ${v >= 0.8 ? 'text-green-500' : v >= 0.6 ? 'text-yellow-500' : 'text-red-500'}`}>{v.toFixed(2)}</span>
                                                                </div>
                                                            ))}
                                                        </div>
                                                        {/* 1-sentence plain-language verdict with hover explanation of the calculation */}
                                                        <div className={`mt-1.5 flex items-center justify-center gap-0.5 ${insightColor}`}>
                                                            <p className="text-[9px] leading-snug text-center">{insight}</p>
                                                            <InfoTooltip text={insightTip} size="sm" />
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                );
                            };

                            const allSplitEntries = Object.entries(evaluationData.splits) as [string, EvaluationSplit][];
                            const trainEntry = showTrainMetrics ? allSplitEntries.find(([n]) => n === 'train') : undefined;
                            const testEntry  = showTestMetrics  ? allSplitEntries.find(([n]) => n === 'test')  : undefined;
                            const valEntry   = showValMetrics   ? allSplitEntries.find(([n]) => n === 'validation') : undefined;

                            return (
                                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        {trainEntry && renderSplitPerClass(trainEntry[0], trainEntry[1])}
                                        {testEntry  && renderSplitPerClass(testEntry[0],  testEntry[1])}
                                        {!trainEntry && !testEntry && (
                                            <p className="col-span-2 text-xs text-gray-400 text-center py-8">Enable Train or Test splits above to compare.</p>
                                        )}
                                    </div>
                                    {valEntry && (
                                        <div className="mt-6 pt-4 border-t border-gray-100 dark:border-gray-700">
                                            {renderSplitPerClass(valEntry[0], valEntry[1])}
                                        </div>
                                    )}
                                </div>
                            );
                        
};