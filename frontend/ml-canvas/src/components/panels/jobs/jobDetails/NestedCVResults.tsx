type ThresholdSelection = { decision_thresholds: Record<string, number>; decision_threshold_metric?: string; selection?: string; oof_rows?: number };

type NestedReport = {
  status: 'nested_cv'; outer_folds: number; inner_folds: number;
  scoring_metric: string; mean_score: number; std_score: number; total_trials: number;
  split_policy?: Record<string, unknown>;
  threshold_selection?: ThresholdSelection;
  folds: { split?: Record<string, unknown>; inner_splits?: Record<string, unknown>[]; threshold_selection?: ThresholdSelection; fold: number; inner_best_score: number; outer_score: number; best_params: Record<string, unknown> }[];
};

/** Read current evidence while leaving legacy diagnostic job results unchanged. */
export function nestedReport(result: Record<string, unknown>): NestedReport | null {
  const metrics = result.metrics as Record<string, unknown> | undefined;
  const raw = (metrics?.nested_cv ?? result.nested_cv) as Partial<NestedReport> | undefined;
  if (!raw || raw.status !== 'nested_cv' || !Array.isArray(raw.folds)
    || !Number.isFinite(raw.mean_score) || !Number.isFinite(raw.std_score)) return null;
  return raw as NestedReport;
}

/** Outer evaluation remains separate from the final model's inner-search optimum. */
export function NestedCVResults({ result }: { result: Record<string, unknown> }) {
  const report = nestedReport(result);
  if (!report) return null;
  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-200">Nested CV evaluation</h4>
      <p className="text-xs text-gray-500 dark:text-gray-400">
        {report.outer_folds} outer folds · {report.inner_folds} inner folds · {report.scoring_metric}
        {' · '}Outer mean: {report.mean_score.toFixed(4)} ± {report.std_score.toFixed(4)}
      </p>
      <p className="text-xs text-gray-500 dark:text-gray-400">Each outer score evaluates an independent search on untouched rows. Higher scores are better; negative loss scores remain negative.</p>
      <PolicyEvidence report={report} />
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead><tr><th scope="col">Fold</th><th scope="col">Inner best score</th><th scope="col">Outer score</th><th scope="col">Selected parameters</th><th scope="col">Threshold</th><th scope="col">Split evidence</th></tr></thead>
          <tbody>{report.folds.map((fold) => (
            <tr key={fold.fold} className="border-t border-gray-200 dark:border-gray-700">
              <td className="py-2">{fold.fold}</td><td>{fold.inner_best_score.toFixed(4)}</td><td>{fold.outer_score.toFixed(4)}</td>
              <td className="font-mono">{JSON.stringify(fold.best_params)}</td>
              <td><ThresholdEvidence selection={fold.threshold_selection} /></td>
              <td><SplitEvidence split={fold.split} inner={fold.inner_splits} /></td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </section>
  );
}

/** Surface the independent final selection separately from outer-fold selections. */
function PolicyEvidence({ report }: { report: NestedReport }) {
  return <div className="space-y-1 text-xs text-gray-500 dark:text-gray-400">
    {report.split_policy && <p>Split policy: {Object.entries(report.split_policy).filter(([, value]) => value != null).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${String(value)}`).join(' ? ')}</p>}
    {report.threshold_selection && <p>Final training threshold: <ThresholdEvidence selection={report.threshold_selection} />. Selected independently from training out-of-fold predictions.</p>}
  </div>;
}

function ThresholdEvidence({ selection }: { selection: ThresholdSelection | undefined }) {
  if (!selection) return <span>?</span>;
  return <span>{JSON.stringify(selection.decision_thresholds)} {selection.decision_threshold_metric} {selection.oof_rows != null && `(${selection.oof_rows} OOF rows)`}</span>;
}

/** Keep bounded membership evidence inspectable without expanding every inner split. */
function SplitEvidence({ split, inner }: { split: Record<string, unknown> | undefined; inner: Record<string, unknown>[] | undefined }) {
  if (!split) return <span>?</span>;
  return <details><summary>Boundaries and counts</summary><pre className="whitespace-pre-wrap">{JSON.stringify({ outer: split, inner }, null, 2)}</pre></details>;
}
