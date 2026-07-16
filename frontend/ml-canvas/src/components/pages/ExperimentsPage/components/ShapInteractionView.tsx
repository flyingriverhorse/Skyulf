/**
 * SHAP feature-interaction heatmap: a global summary of how much each pair
 * of features jointly influences predictions (`shap.TreeExplainer.
 * shap_interaction_values`, tree models only). Unlike Summary/Beeswarm/
 * Dependence/Waterfall/Force — which all reuse per-sample `shap_values` —
 * this reads the separate `interactions` matrix computed once per job.
 */

import React, { useMemo } from 'react';
import { Loader2, Check, Download } from 'lucide-react';
import { InfoTooltip } from '../../../ui/InfoTooltip';
import { useChartTheme } from '../../../../core/hooks/useChartTheme';
import type { ShapExplanationData } from '../types';

interface Props {
  jobId: string;
  modelType: string;
  shapExplanation: ShapExplanationData;
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

export const ShapInteractionView: React.FC<Props> = ({
  jobId,
  modelType,
  shapExplanation,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const chartId = `shap-interaction-chart-${jobId}`;
  const chartTheme = useChartTheme();
  const interactions = shapExplanation.interactions;

  const maxValue = useMemo(() => {
    if (!interactions) return 0;
    return interactions.matrix.flat().reduce((m, v) => Math.max(m, v), 0);
  }, [interactions]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 flex-wrap">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">
          SHAP Interaction Plot — {modelType !== 'unknown' ? modelType : jobId.slice(0, 8)}
        </h3>
        <InfoTooltip
          text="How much each pair of features jointly drives predictions, beyond their individual effects (mean |SHAP interaction value|, averaged across sampled rows). Darker cells = stronger interaction. Only available for tree-based models (Random Forest, Gradient Boosting, XGBoost, etc.)."
          align="center"
        />
      </div>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 relative group" id={chartId}>
        {interactions && interactions.feature_names.length > 0 && (
          <div className="absolute top-4 right-4 z-10 opacity-0 group-hover:opacity-100 transition-opacity" data-export-ignore="true">
            <button
              onClick={() => void handleDownload(chartId, `shap_interaction_${jobId.slice(0, 8)}`)}
              disabled={downloadingChart === chartId}
              className="p-1.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-500 hover:text-blue-600 disabled:opacity-50"
              title="Download Graph"
            >
              {downloadingChart === chartId ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : doneChart === chartId ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Download className="w-3.5 h-3.5" />}
            </button>
          </div>
        )}
        {!interactions || interactions.feature_names.length === 0 ? (
          <p className="text-sm text-gray-400 dark:text-gray-500 text-center py-12">
            Interaction values are not available for this run — they only support tree-based
            models (Random Forest, Gradient Boosting, XGBoost, etc.).
          </p>
        ) : (
          <>
            <div className="overflow-x-auto">
              <div
                className="grid gap-1 mx-auto"
                style={{
                  gridTemplateColumns: `100px repeat(${interactions.feature_names.length}, minmax(56px, 1fr))`,
                  maxWidth: `${100 + interactions.feature_names.length * 72}px`,
                }}
              >
                <div />
                {interactions.feature_names.map(name => (
                  <div
                    key={`col-${name}`}
                    className="text-[10px] font-medium text-gray-500 dark:text-gray-400 text-center truncate px-0.5"
                    title={name}
                  >
                    {name}
                  </div>
                ))}
                {interactions.matrix.map((row, i) => (
                  <React.Fragment key={`row-${interactions.feature_names[i]}`}>
                    <div
                      className="text-[10px] font-medium text-gray-500 dark:text-gray-400 truncate pr-1 flex items-center justify-end"
                      title={interactions.feature_names[i]}
                    >
                      {interactions.feature_names[i]}
                    </div>
                    {row.map((value, j) => {
                      const intensity = maxValue > 0 ? value / maxValue : 0;
                      const isDiagonal = i === j;
                      return (
                        <div
                          key={`cell-${i}-${j}`}
                          className="aspect-square rounded flex items-center justify-center border border-gray-100 dark:border-gray-700"
                          style={{
                            backgroundColor: isDiagonal
                              ? `rgba(99,102,241,${Math.min(intensity * 0.7 + 0.15, 0.85)})`
                              : `rgba(139,92,246,${Math.min(intensity * 0.75 + 0.05, 0.8)})`,
                          }}
                          title={`${interactions.feature_names[i]} × ${interactions.feature_names[j]}: ${value.toFixed(4)}`}
                        >
                          <span
                            className="text-[9px] font-mono font-semibold"
                            style={{ color: chartTheme.textColor }}
                          >
                            {value.toFixed(2)}
                          </span>
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
            <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-3 text-center">
              Diagonal cells (indigo) show each feature&apos;s main effect strength · off-diagonal cells
              (purple) show pairwise interaction strength · darker = stronger
            </p>
          </>
        )}
      </div>
    </div>
  );
};
