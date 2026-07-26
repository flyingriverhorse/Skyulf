# Evaluation View Threshold Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the Evaluation page's classification threshold controls into two tabs — "Threshold Slider" (today's manual, client-side threshold slider, unchanged) and "Threshold Tuning" (the server-side optimizer, now visualized for both binary and multiclass jobs) — sharing one set of Train/Test/Validation split checkboxes.

**Architecture:** Frontend-only change in `frontend/ml-canvas/`. A new `activeTab: 'slider' | 'tuning'` piece of state (owned by the existing parent `ExperimentsPage.tsx`, mirroring how `cmView` is already lifted there) gates which control panel and which confusion-matrix call renders inside `EvaluationView.tsx`. `PerClassConfusionMatrix.tsx` gains a binary (2-class) rendering path so Tab 2 can show tuned-threshold effects for binary jobs too — something it cannot do today.

**Tech Stack:** React + TypeScript, Vitest + React Testing Library, Tailwind CSS classes (no new dependencies).

## Global Constraints

- No backend/API changes — this plan only touches `frontend/ml-canvas/src/**`.
- No change to `optimize_thresholds()` / `apply_thresholds()` in `skyulf-core` (already fixed, out of scope).
- No new classification-node parameters (confirmed with user — not exposing optimizer internals).
- No "positive class" selector added for binary Tab 2 (confirmed with user — binary confusion matrices are symmetric).
- Train/Test/Validation split checkboxes are **shared** across both tabs — one instance of state, not duplicated (confirmed with user).
- After every task: run `npx eslint <changed files>` and `npx tsc --project tsconfig.json --noEmit` from `frontend/ml-canvas/`; run the full gate (`npm run lint`, `npx tsc --noEmit`, `npm run build`, targeted `vitest run`) before considering the plan done (Task 3).
- Design reference: `docs/superpowers/specs/2026-07-26-evaluation-view-threshold-tabs-design.md` — read it before starting if anything below is unclear.

---

## File Map

- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx` — add `activeTab`/`setActiveTab` state, thread into `<EvaluationView>` props.
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx` — add `activeTab`/`setActiveTab` to `Props`, add tab-switch buttons, gate existing slider controls and Tuning panel behind the active tab, add hint tooltips, add Tab 2 placeholder-when-no-preview state, decouple Tab 1's `PerClassConfusionMatrix` call from tuned thresholds.
- Create: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.test.tsx` — new test file (none exists today).
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx` — relax the `classes.length <= 2` early-return guard, add a binary (2×2) rendering path.
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx` — add binary-path test cases.

---

### Task 1: Add tab state and restructure `EvaluationView` into two tabs (with hints)

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx:79` (state) and `:513-549` (props passed to `<EvaluationView>`)
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx:17-51` (Props interface + destructure), `:209-467` (JSX)
- Test: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.test.tsx` (new)

**Interfaces:**
- Consumes: existing `PerClassConfusionMatrix` props (`evaluationData`, `selectedRocClass`, `threshold`, `showTrainMetrics`, `showTestMetrics`, `showValMetrics`, `handleDownload`, `downloadingChart`, `doneChart`, `tunedThresholds?: Record<string, number> | null`, `useTunedThresholds?: boolean`) — unchanged signature, just called from two sites now.
- Produces: `EvaluationView`'s `Props` interface gains `activeTab: 'slider' | 'tuning'` and `setActiveTab: (v: 'slider' | 'tuning') => void`. Task 2 (`PerClassConfusionMatrix`) relies on the two call sites this task creates (Tab 1: `useTunedThresholds={false}`, `tunedThresholds={null}`; Tab 2: `useTunedThresholds` truthy iff `tuningPreview` exists, `tunedThresholds={tuningPreview.thresholds}`).

- [ ] **Step 1: Add `activeTab` state to `ExperimentsPage.tsx`**

In `frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx`, find:

```tsx
  const [cmView, setCmView] = useState<'overall' | 'per-class'>('overall');
```

Add right after it:

```tsx
  const [cmView, setCmView] = useState<'overall' | 'per-class'>('overall');
  // Evaluation page tab switch: "Threshold Slider" (today's manual,
  // client-side slider) vs "Threshold Tuning" (server-side optimizer
  // preview/save flow). Lifted here, not reset on job switch, mirroring
  // how `cmView` above already behaves — the threshold-tuning-specific
  // state (`tuningPreview` etc.) is separately reset per job already.
  const [activeTab, setActiveTab] = useState<'slider' | 'tuning'>('slider');
```

- [ ] **Step 2: Pass `activeTab`/`setActiveTab` into `<EvaluationView>`**

In the same file, find the `<EvaluationView` call (around line 513) and its prop:

```tsx
                  cmView={cmView}
                  setCmView={setCmView}
```

Add right after:

```tsx
                  cmView={cmView}
                  setCmView={setCmView}
                  activeTab={activeTab}
                  setActiveTab={setActiveTab}
```

- [ ] **Step 3: Add `activeTab`/`setActiveTab` to `EvaluationView`'s `Props` interface and destructure**

In `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx`, find:

```tsx
  cmView: 'overall' | 'per-class';
  setCmView: (v: 'overall' | 'per-class') => void;
```

Replace with:

```tsx
  cmView: 'overall' | 'per-class';
  setCmView: (v: 'overall' | 'per-class') => void;
  activeTab: 'slider' | 'tuning';
  setActiveTab: (v: 'slider' | 'tuning') => void;
```

Then find the destructure:

```tsx
  cmView,
  setCmView,
```

Replace with:

```tsx
  cmView,
  setCmView,
  activeTab,
  setActiveTab,
```

- [ ] **Step 4: Replace the classification JSX block with the two-tab layout**

In the same file, find the whole block starting at `{/* Controls bar` and ending right before the final `</div>\n      )}\n    </div>\n  );\n};` (originally lines 209–467). Replace it in full with:

```tsx
          {/* Controls bar — sticky so it stays visible while scrolling splits */}
          <div className="sticky top-0 z-10 flex flex-wrap items-center gap-x-6 gap-y-2 bg-white dark:bg-gray-800 px-4 py-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
            {/* Regression: split tabs inline in the control bar */}
            {evaluationData.problem_type === 'regression' && (
              <div className="flex items-center gap-0.5">
                <span className="text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide mr-1">Split:</span>
                {regressionSplitTabs.map(tab => (
                  <button
                    key={tab}
                    onClick={() => setSelectedRegressionSplit(tab)}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      activeRegressionSplit === tab
                        ? 'bg-blue-500 text-white'
                        : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >{regressionSplitLabels[tab] ?? tab}</button>
                ))}
              </div>
            )}
            {/* Classification: Threshold Slider / Threshold Tuning tab switch —
                decides which control panel + confusion-matrix view below is
                shown. Independent of the "Splits:" checkboxes just below,
                which stay shared across both tabs. */}
            {evaluationData.problem_type === 'classification' && (
              <div className="flex items-center rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 text-xs font-medium">
                <button
                  onClick={() => setActiveTab('slider')}
                  className={`px-3 py-1.5 transition-colors ${activeTab === 'slider' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}
                >
                  Threshold Slider
                </button>
                <button
                  onClick={() => setActiveTab('tuning')}
                  className={`px-3 py-1.5 transition-colors border-l border-gray-200 dark:border-gray-700 ${activeTab === 'tuning' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}
                >
                  Threshold Tuning
                </button>
              </div>
            )}
            {/* Split visibility toggles — classification only, shared by both tabs */}
            {evaluationData.problem_type !== 'regression' && (<>
              <div className="flex items-center gap-1 text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide">Splits:</div>
              {hasTrainSplit && (
              <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                <input type="checkbox" checked={showTrainMetrics} onChange={e => { setShowTrainMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700" />
                <span className="text-gray-700 dark:text-gray-300">Train</span>
              </label>
              )}
              {hasTestSplit && (
              <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                <input type="checkbox" checked={showTestMetrics} onChange={e => { setShowTestMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700" />
                <span className="text-gray-700 dark:text-gray-300">Test</span>
              </label>
              )}
              {hasValidationSplit && (
              <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                <input type="checkbox" checked={showValMetrics} onChange={e => { setShowValMetrics(e.target.checked); }} className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700" />
                <span className="text-gray-700 dark:text-gray-300">Validation</span>
              </label>
              )}
            </>)}

            {/* Classification controls — Tab 1 (Threshold Slider) only */}
            {activeTab === 'slider' && evaluationData.problem_type === 'classification' && evaluationData.splits.train?.y_proba && (() => {
              const proba = evaluationData.splits.train.y_proba!;
              const isBinary = proba.classes.length === 2;
              return (
                <>
                  <div className="w-px h-5 bg-gray-200 dark:bg-gray-600" />
                  {/* Class selector — hidden for binary: both classes always shown inline */}
                  {!isBinary && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Class:</span>
                      <select
                        className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
                        value={selectedRocClass || ''}
                        onChange={(e) => { setSelectedRocClass(e.target.value); }}
                      >
                        {proba.classes.map((c: string | number, idx: number) => {
                          const label = proba.labels?.[idx] ?? c;
                          return <option key={String(c)} value={String(label)}>{String(label)}</option>;
                        })}
                      </select>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Metric:</span>
                    <select
                      className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
                      value={normalizeThresholdMetric(selectedMetric, isBinary)}
                      onChange={(e) => { setSelectedMetric(normalizeThresholdMetric(e.target.value as ThresholdMetric, isBinary)); }}
                    >
                      {thresholdMetricOptions(isBinary).map(m => (
                        <option key={m} value={m}>{metricLabel(m, isBinary)}</option>
                      ))}
                    </select>
                    <InfoTooltip
                      text={`Which metric the best-threshold badges below and ROC/PR-based scan optimize for. Precision/Recall/F1 use the selected class as positive for binary jobs, and a support-weighted average across all classes for multiclass jobs (accuracy is unaffected either way).`}
                      align="center"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Threshold:</span>
                    <InfoTooltip
                      text={`Threshold (t): a sample is predicted as the selected class when P(class) ≥ t.\n\n↑ Raise t → fewer positives predicted → lower recall, higher precision (fewer false alarms, more misses).\n↓ Lower t → more positives predicted → higher recall, lower precision (fewer misses, more false alarms).\n\nDefault 0.5 works well for balanced classes. Adjust for imbalanced data or when the cost of false positives ≠ false negatives.`}
                      align="center"
                    />
                    <input
                      type="range" min={0.01} max={0.99} step={0.01}
                      value={threshold}
                      onChange={(e) => { setThreshold(parseFloat(e.target.value)); }}
                      className="w-28 accent-blue-500"
                    />
                    <span className="text-sm font-mono font-semibold text-blue-600 dark:text-blue-400 w-9">{threshold.toFixed(2)}</span>
                    {bestMetricInfos.map(info => {
                      const colors: Record<string, string> = {
                        train: 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-700 hover:bg-blue-100 dark:hover:bg-blue-900/50',
                        test: 'bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 border-emerald-200 dark:border-emerald-700 hover:bg-emerald-100 dark:hover:bg-emerald-900/50',
                        validation: 'bg-orange-50 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-700 hover:bg-orange-100 dark:hover:bg-orange-900/50',
                      };
                      const isActive = Math.abs(threshold - info.threshold) < 0.001;
                      const badgeMetricLabel = metricLabel(info.metricName as ThresholdMetric, isBinary);
                      return (
                        <button
                          key={info.splitLabel}
                          onClick={() => { setThreshold(info.threshold); }}
                          className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border transition-colors whitespace-nowrap ${colors[info.splitLabel] ?? colors.test} ${isActive ? 'ring-2 ring-offset-1 ring-current' : ''}`}
                          title={`Best ${badgeMetricLabel}=${info.value.toFixed(3)} on ${info.splitLabel} split — click to apply`}
                        >
                          ★ {info.splitLabel} {badgeMetricLabel}: {info.threshold.toFixed(2)}
                        </button>
                      );
                    })}
                    {bestMetricInfos.length > 0 && (
                      <InfoTooltip
                        text={`Each badge shows the threshold that maximises ${metricLabel(bestMetricInfos[0]!.metricName as ThresholdMetric, isBinary)} for the selected class on that split (found by scanning every unique prediction score, same method sklearn uses internally) — one per split currently checked in "Splits:" above. Click a badge to snap the slider to that split's optimal value.`}
                        align="center"
                      />
                    )}
                  </div>
                  {proba.labels && proba.labels.length === proba.classes.length && (
                    <div className="text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                      ({proba.classes.map((c, idx) => `${String(c)}→${String(proba.labels?.[idx] ?? c)}`).join(', ')})
                    </div>
                  )}
                  {/* Overall / Per Class toggle — hidden for binary */}
                  {!isBinary && (
                    <>
                      <div className="w-px h-5 bg-gray-200 dark:bg-gray-600" />
                      <div className="flex items-center rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 text-xs font-medium">
                        <button onClick={() => setCmView('overall')} className={`px-3 py-1.5 transition-colors ${cmView === 'overall' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}>Overall</button>
                        <button onClick={() => setCmView('per-class')} className={`px-3 py-1.5 transition-colors border-l border-gray-200 dark:border-gray-700 ${cmView === 'per-class' ? 'bg-blue-500 text-white' : 'bg-white dark:bg-gray-900 text-gray-500 hover:bg-gray-50 dark:hover:bg-gray-800'}`}>Per Class</button>
                      </div>
                    </>
                  )}
                </>
              );
            })()}
          </div>

          {/* Tab-level one-line description of what's driving the charts below */}
          {evaluationData.problem_type === 'classification' && (
            <p className="text-xs text-gray-400 dark:text-gray-500 italic px-1">
              {activeTab === 'slider'
                ? 'Manually explore how a single threshold changes predictions — nothing here is saved or used for real predictions.'
                : "Let the optimizer find the best per-class threshold(s) for a metric you choose, preview its effect, then save it to actually change how this model predicts."}
            </p>
          )}

          {activeTab === 'tuning' && evaluationData.problem_type === 'classification' && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 bg-white dark:bg-gray-800 px-4 py-3 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">Threshold Tuning</h4>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">Metric:</span>
                <select
                  className="bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 p-1.5"
                  value={selectedTuningMetric}
                  onChange={(e) => { onSelectedTuningMetricChange(e.target.value); }}
                >
                  <option value="accuracy">Accuracy</option>
                  <option value="f1">F1</option>
                  <option value="precision">Precision</option>
                  <option value="recall">Recall</option>
                  <option value="balanced_accuracy">Balanced Accuracy</option>
                  <option value="roc_auc">ROC AUC</option>
                </select>
                <InfoTooltip
                  text="Which metric the optimizer maximizes when you click Preview. Uses your validation split if available, otherwise test."
                  align="center"
                />
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => { void onPreviewThresholds(); }}
                  className="px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-500 text-white hover:bg-blue-600 transition-colors"
                >
                  Preview
                </button>
                <InfoTooltip
                  text="Runs the optimizer now and shows the result below — does not save or affect real predictions yet."
                  align="center"
                />
              </div>
              {tuningPreview && (
                <>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    Computed from {tuningPreview.split_used} split
                    {tuningPreview.split_used === 'test' && (
                      <em className="ml-1">(no validation split available — using test split)</em>
                    )}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => { void onSaveThresholds(); }}
                      className="px-3 py-1.5 rounded-lg text-sm font-medium bg-emerald-500 text-white hover:bg-emerald-600 transition-colors"
                    >
                      Save
                    </button>
                    <InfoTooltip
                      text="Persists the previewed thresholds to this model version. Still inactive until you also enable 'Use tuned thresholds.'"
                      align="center"
                    />
                  </div>
                </>
              )}
              <div className="flex items-center gap-1">
                <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                  <input
                    type="checkbox"
                    checked={useTunedThresholds}
                    onChange={(e) => { void onToggleThresholds(e.target.checked); }}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700"
                  />
                  <span className="text-gray-700 dark:text-gray-300">Use tuned thresholds at prediction time</span>
                </label>
                <InfoTooltip
                  text="When ON, every real /predict call for this model uses these saved thresholds instead of the default 0.5/argmax rule."
                  align="center"
                />
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => { void onClearThresholds(); }}
                  className="px-3 py-1.5 rounded-lg text-sm font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  Clear
                </button>
                <InfoTooltip
                  text="Deletes saved thresholds entirely and reverts predictions to the default rule."
                  align="center"
                />
              </div>
              {tuningError && (
                <span className="text-xs text-red-600 dark:text-red-400">{tuningError}</span>
              )}
            </div>
          )}

          {(evaluationData.problem_type === 'regression' || (activeTab === 'slider' && (cmView === 'overall' || evaluationData.splits.train?.y_proba?.classes.length === 2))) && (
            <div className="flex flex-col gap-6">
              {/* Charts per split */}
              {Object.entries(evaluationData.splits)
                .filter(([splitName]) => {
                  if (evaluationData.problem_type === 'regression') {
                    // Only show the active tab split
                    return splitName === activeRegressionSplit;
                  }
                  if (splitName === 'train' && !showTrainMetrics) return false;
                  if (splitName === 'test' && !showTestMetrics) return false;
                  if (splitName === 'validation' && !showValMetrics) return false;
                  return true;
                })
                .map(([splitName, splitData]: [string, EvaluationSplit]) => (
                  <div key={splitName} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                    {evaluationData.problem_type !== 'regression' && (
                      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4 capitalize">{splitName} Set</h4>
                    )}

                    {evaluationData.problem_type === 'regression' ? (
                      <RegressionChartsForSplit
                        splitName={splitName}
                        splitData={splitData}
                        handleDownload={handleDownload}
                        downloadingChart={downloadingChart}
                        doneChart={doneChart}
                      />
                    ) : (
                      <ClassificationChartsForSplit
                        splitName={splitName}
                        splitData={splitData}
                        selectedRocClass={selectedRocClass}
                        threshold={threshold}
                        handleDownload={handleDownload}
                        downloadingChart={downloadingChart}
                        doneChart={doneChart}
                      />
                    )}
                  </div>
                ))}
            </div>
          )}
          {activeTab === 'slider' && evaluationData.problem_type === 'classification' && cmView === 'per-class' && (
            <PerClassConfusionMatrix
              evaluationData={evaluationData}
              selectedRocClass={selectedRocClass}
              threshold={threshold}
              showTrainMetrics={showTrainMetrics}
              showTestMetrics={showTestMetrics}
              showValMetrics={showValMetrics}
              handleDownload={handleDownload}
              downloadingChart={downloadingChart}
              doneChart={doneChart}
              tunedThresholds={null}
              useTunedThresholds={false}
            />
          )}
          {activeTab === 'tuning' && evaluationData.problem_type === 'classification' && (
            tuningPreview ? (
              <PerClassConfusionMatrix
                evaluationData={evaluationData}
                selectedRocClass={selectedRocClass}
                threshold={threshold}
                showTrainMetrics={showTrainMetrics}
                showTestMetrics={showTestMetrics}
                showValMetrics={showValMetrics}
                handleDownload={handleDownload}
                downloadingChart={downloadingChart}
                doneChart={doneChart}
                tunedThresholds={tuningPreview.thresholds}
                useTunedThresholds
              />
            ) : (
              <div className="text-xs text-gray-400 dark:text-gray-500 italic text-center py-8 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                Click Preview above to see tuned thresholds applied to your confusion matrix.
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
};
```

Key behavior changes from today, called out explicitly (both intentional, confirmed with the user during design):
- Tab 1's `PerClassConfusionMatrix` call now hardcodes `tunedThresholds={null}` and `useTunedThresholds={false}` (was previously tied to the real `tuningPreview`/`useTunedThresholds` state) — Tab 1 must never be affected by tuning.
- Tab 2's `PerClassConfusionMatrix` call passes `useTunedThresholds` as a bare truthy prop only when `tuningPreview` exists (via the `tuningPreview ? (...) : (...)` branch) — Tab 2's matrix reflects the *previewed* thresholds immediately, independent of whether "Use tuned thresholds at prediction time" is checked.

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd frontend/ml-canvas && npx tsc --project tsconfig.json --noEmit`
Expected: no errors referencing `EvaluationView.tsx` or `ExperimentsPage.tsx`.

- [ ] **Step 6: Write `EvaluationView.test.tsx`**

Create `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.test.tsx`:

```tsx
// Tests for the Threshold Slider / Threshold Tuning tab split in
// EvaluationView: verifies the two tabs show/hide the right controls, the
// Train/Test/Validation checkboxes are shared (rendered regardless of the
// active tab), and Tab 2 shows a placeholder until a preview exists.

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EvaluationView } from './EvaluationView';
import type { EvaluationData } from '../types';
import type { ThresholdPreviewResult } from '../../../../core/api/thresholdTuning';

const evaluationData: Extract<EvaluationData, { problem_type: 'classification' | 'regression' }> = {
  problem_type: 'classification',
  splits: {
    train: {
      y_true: ['a', 'b', 'c', 'a'],
      y_pred: ['a', 'b', 'c', 'a'],
      y_proba: {
        classes: ['a', 'b', 'c'],
        values: [
          [0.7, 0.2, 0.1],
          [0.2, 0.6, 0.2],
          [0.1, 0.2, 0.7],
          [0.6, 0.3, 0.1],
        ],
      },
    },
  },
};

const noop = async () => {};

function baseProps(overrides: Partial<React.ComponentProps<typeof EvaluationView>> = {}) {
  return {
    selectedJobIds: ['job-1'],
    evalJobId: 'job-1',
    fetchEvaluationData: noop,
    isEvalLoading: false,
    evalError: null,
    evaluationData,
    selectedRegressionSplit: null,
    setSelectedRegressionSplit: vi.fn(),
    showTrainMetrics: true,
    setShowTrainMetrics: vi.fn(),
    showTestMetrics: true,
    setShowTestMetrics: vi.fn(),
    showValMetrics: true,
    setShowValMetrics: vi.fn(),
    threshold: 0.5,
    setThreshold: vi.fn(),
    selectedRocClass: 'a',
    setSelectedRocClass: vi.fn(),
    cmView: 'overall' as const,
    setCmView: vi.fn(),
    activeTab: 'slider' as const,
    setActiveTab: vi.fn(),
    selectedMetric: 'f1_weighted' as const,
    setSelectedMetric: vi.fn(),
    bestMetricInfos: [],
    handleDownload: noop,
    downloadingChart: null,
    doneChart: null,
    selectedTuningMetric: 'f1',
    onSelectedTuningMetricChange: vi.fn(),
    tuningPreview: null as ThresholdPreviewResult | null,
    tuningError: null,
    useTunedThresholds: false,
    onPreviewThresholds: noop,
    onSaveThresholds: noop,
    onToggleThresholds: noop,
    onClearThresholds: noop,
    ...overrides,
  };
}

describe('EvaluationView — Threshold Slider / Threshold Tuning tabs', () => {
  it('renders both tab buttons', () => {
    render(<EvaluationView {...baseProps()} />);
    expect(screen.getByText('Threshold Slider')).toBeInTheDocument();
    expect(screen.getByText('Threshold Tuning')).toBeInTheDocument();
  });

  it('shows the manual slider controls and hides the Tuning panel when activeTab is "slider"', () => {
    render(<EvaluationView {...baseProps({ activeTab: 'slider' })} />);
    expect(screen.getByText('Class:')).toBeInTheDocument();
    expect(screen.queryByText('Preview')).not.toBeInTheDocument();
  });

  it('shows the Tuning panel and hides the manual slider controls when activeTab is "tuning"', () => {
    render(<EvaluationView {...baseProps({ activeTab: 'tuning' })} />);
    expect(screen.getByText('Preview')).toBeInTheDocument();
    expect(screen.queryByText('Class:')).not.toBeInTheDocument();
  });

  it('clicking the Threshold Tuning tab button calls setActiveTab("tuning")', () => {
    const setActiveTab = vi.fn();
    render(<EvaluationView {...baseProps({ activeTab: 'slider', setActiveTab })} />);
    fireEvent.click(screen.getByText('Threshold Tuning'));
    expect(setActiveTab).toHaveBeenCalledWith('tuning');
  });

  it('renders the shared Splits: checkboxes regardless of the active tab', () => {
    const { rerender } = render(<EvaluationView {...baseProps({ activeTab: 'slider' })} />);
    expect(screen.getByText('Train')).toBeInTheDocument();
    rerender(<EvaluationView {...baseProps({ activeTab: 'tuning' })} />);
    expect(screen.getByText('Train')).toBeInTheDocument();
  });

  it('shows a placeholder in Tab 2 until a tuning preview exists', () => {
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', tuningPreview: null })} />);
    expect(screen.getByText(/Click Preview above/)).toBeInTheDocument();
  });

  it('renders the confusion matrix in Tab 2 once a tuning preview exists', () => {
    const tuningPreview: ThresholdPreviewResult = {
      thresholds: { a: 1, b: 1, c: 1 },
      classes: [0, 1, 2],
      metric: 'f1',
      split_used: 'train',
    };
    render(<EvaluationView {...baseProps({ activeTab: 'tuning', tuningPreview })} />);
    expect(screen.queryByText(/Click Preview above/)).not.toBeInTheDocument();
    expect(screen.getByText('a vs Rest')).toBeInTheDocument();
  });
});
```

- [ ] **Step 7: Run the new test file and verify it passes**

Run: `cd frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/components/EvaluationView.test.tsx`
Expected: all 7 tests PASS.

- [ ] **Step 8: Lint the touched files**

Run: `cd frontend/ml-canvas && npx eslint src/components/pages/ExperimentsPage.tsx src/components/pages/ExperimentsPage/components/EvaluationView.tsx src/components/pages/ExperimentsPage/components/EvaluationView.test.tsx`
Expected: no errors (0 warnings, since the project runs `--max-warnings 0`).

- [ ] **Step 9: Commit**

```bash
git add frontend/ml-canvas/src/components/pages/ExperimentsPage.tsx \
        frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.tsx \
        frontend/ml-canvas/src/components/pages/ExperimentsPage/components/EvaluationView.test.tsx
git commit -m "feat: split Evaluation view threshold controls into Slider/Tuning tabs

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Support binary classification in `PerClassConfusionMatrix`

**Files:**
- Modify: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx`
- Test: `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx`

**Interfaces:**
- Consumes: `applyThreshold`, `applyMulticlassThresholds` from `../utils/classificationCharts` (both already imported, unchanged signatures) — `applyMulticlassThresholds` already generalizes correctly over any class count, including 2.
- Produces: `PerClassConfusionMatrix` now renders for 2-class (binary) jobs too, showing one plain N×N matrix instead of returning `null`. No prop changes — `Props` interface is unchanged.

- [ ] **Step 1: Write the failing tests for binary rendering**

In `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx`, add (after the existing `describe` block, same file):

```tsx
const binaryEvaluationData: Extract<EvaluationData, { problem_type: 'classification' | 'regression' }> = {
  problem_type: 'classification' as const,
  splits: {
    train: {
      y_true: ['yes', 'no', 'yes', 'no'],
      y_pred: ['yes', 'no', 'yes', 'no'],
      y_proba: {
        classes: ['yes', 'no'],
        values: [
          [0.8, 0.2],
          [0.3, 0.7],
          [0.9, 0.1],
          [0.4, 0.6],
        ],
      },
    },
  },
};

describe('PerClassConfusionMatrix — binary classification support', () => {
  it('no longer returns null for a 2-class job', () => {
    const { container } = render(
      <PerClassConfusionMatrix
        evaluationData={binaryEvaluationData}
        selectedRocClass={null}
        threshold={0.5}
        showTrainMetrics
        showTestMetrics={false}
        showValMetrics={false}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
        tunedThresholds={{ yes: 1, no: 1 }}
        useTunedThresholds
      />,
    );
    expect(container.firstChild).not.toBeNull();
  });

  it('renders one plain matrix (not "vs Rest" mirror panels) for a binary job', () => {
    render(
      <PerClassConfusionMatrix
        evaluationData={binaryEvaluationData}
        selectedRocClass={null}
        threshold={0.5}
        showTrainMetrics
        showTestMetrics={false}
        showValMetrics={false}
        handleDownload={noop}
        downloadingChart={null}
        doneChart={null}
        tunedThresholds={{ yes: 1, no: 1 }}
        useTunedThresholds
      />,
    );
    // Equal thresholds (all 1) reduce applyMulticlassThresholds to plain
    // argmax, matching y_true exactly here — both classes get 2/2 correct.
    expect(screen.queryByText('yes vs Rest')).not.toBeInTheDocument();
    expect(screen.queryByText('no vs Rest')).not.toBeInTheDocument();
    expect(screen.getByTitle('true=yes, pred=yes: 2')).toBeInTheDocument();
    expect(screen.getByTitle('true=no, pred=no: 2')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx`
Expected: the two new tests FAIL (component still returns `null` for `classes.length <= 2`).

- [ ] **Step 3: Relax the early-return guard**

In `frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx`, find:

```tsx
  if ((evaluationData.splits.train?.y_proba?.classes.length ?? 0) <= 2) return null;
```

Replace with:

```tsx
  // Only bail out for genuinely invalid data (0 or 1 classes) — 2-class
  // (binary) jobs now render via `renderSplitBinary` below instead of the
  // "N vs Rest" multiclass grid.
  if ((evaluationData.splits.train?.y_proba?.classes.length ?? 0) < 2) return null;
```

- [ ] **Step 4: Add the binary render function**

In the same file, find:

```tsx
                            const renderSplitPerClass = (splitName: string, splitData: EvaluationSplit) => {
```

Add immediately before it:

```tsx
                            // Binary jobs: one plain N×N (2×2) matrix with real class
                            // names on both axes, instead of two redundant "vs Rest"
                            // mirror panels (which carry no extra information for only
                            // 2 classes). Precision/Recall/F1 chips are still shown
                            // per class below the grid, using the same tp/fp/fn
                            // formulas as the multiclass per-class panels.
                            const renderSplitBinary = (splitName: string, splitData: EvaluationSplit) => {
                                const { classes, matrix } = matrixBySplit[splitName] ?? applyThreshold(splitData, selectedRocClass, threshold);
                                const splitId = `binary-cm-${splitName}`;
                                const perClass = classes.map((cls, idx) => {
                                    const tp = matrix[idx]?.[idx] ?? 0;
                                    const fp = matrix.reduce((s, row, ri) => (ri !== idx ? s + (row[idx] ?? 0) : s), 0);
                                    const fn = (matrix[idx] ?? []).reduce((s, v, ci) => (ci !== idx ? s + v : s), 0);
                                    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
                                    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
                                    const f1c = prec + rec > 0 ? (2 * prec * rec) / (prec + rec) : 0;
                                    return { cls, prec, rec, f1: f1c };
                                });
                                const scoreColor = (v: number) => (v >= 0.8 ? 'text-green-500' : v >= 0.6 ? 'text-yellow-500' : 'text-red-500');
                                return (
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-1.5 mb-1">
                                            <h4 className="text-sm font-semibold text-gray-600 dark:text-gray-300 capitalize">{splitName} Set</h4>
                                            <button
                                                id={`${splitId}-dl`}
                                                onClick={() => void handleDownload(splitId, `${splitName}_binary_tuned`)}
                                                disabled={downloadingChart === splitId}
                                                className="p-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded shadow-sm text-gray-400 hover:text-blue-600 disabled:opacity-50"
                                                title="Download Confusion Matrix"
                                            >
                                                {downloadingChart === splitId ? <Loader2 className="w-3 h-3 animate-spin" /> : doneChart === splitId ? <Check className="w-3 h-3 text-green-500" /> : <Download className="w-3 h-3" />}
                                            </button>
                                        </div>
                                        <div id={splitId} className="flex flex-col items-center">
                                            <div className="flex flex-col">
                                                <div className="flex mb-0.5 gap-0.5" style={{ marginLeft: '70px' }}>
                                                    {classes.map(c => (
                                                        <span key={String(c)} className="w-16 text-center text-[10px] text-gray-500 dark:text-gray-400 truncate font-medium" title={String(c)}>{String(c)}</span>
                                                    ))}
                                                </div>
                                                {matrix.map((row, ri) => (
                                                    <div key={ri} className="flex items-center gap-0.5 mb-0.5">
                                                        <span className="text-right text-[10px] text-gray-500 dark:text-gray-400 pr-1 truncate font-medium" style={{ width: '70px' }} title={String(classes[ri])}>{String(classes[ri])}</span>
                                                        {row.map((count, ci) => {
                                                            const isCorrect = ri === ci;
                                                            const rowMax = Math.max(...row, 1);
                                                            const bg = isCorrect
                                                                ? `rgba(34,197,94,${Math.min((count / rowMax) * 0.75 + 0.1, 0.85)})`
                                                                : `rgba(239,68,68,${Math.min((count / rowMax) * 0.65 + 0.05, 0.75)})`;
                                                            return (
                                                                <div key={ci} className="w-16 h-14 flex flex-col items-center justify-center rounded border border-gray-100 dark:border-gray-700 cursor-default" style={{ backgroundColor: bg }} title={`true=${String(classes[ri])}, pred=${String(classes[ci])}: ${count}`}>
                                                                    <span className="text-sm font-mono font-bold leading-none">{count}</span>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                ))}
                                            </div>
                                            <div className="mt-2 grid grid-cols-2 gap-2 text-[10px] w-full max-w-xs">
                                                {perClass.map(({ cls, prec, rec, f1 }) => (
                                                    <div key={String(cls)} className="flex flex-col items-center bg-gray-50 dark:bg-gray-900 rounded py-1.5 px-1">
                                                        <span className="text-gray-500 dark:text-gray-400 font-medium truncate w-full text-center" title={String(cls)}>{String(cls)}</span>
                                                        <div className="flex gap-2 mt-0.5">
                                                            <span title="Precision">P:<span className={`font-mono font-semibold ml-0.5 ${scoreColor(prec)}`}>{prec.toFixed(2)}</span></span>
                                                            <span title="Recall">R:<span className={`font-mono font-semibold ml-0.5 ${scoreColor(rec)}`}>{rec.toFixed(2)}</span></span>
                                                            <span title="F1">F1:<span className={`font-mono font-semibold ml-0.5 ${scoreColor(f1)}`}>{f1.toFixed(2)}</span></span>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                );
                            };

                            const renderSplitPerClass = (splitName: string, splitData: EvaluationSplit) => {
```

- [ ] **Step 5: Wire the binary branch into the final render**

In the same file, find the final `return` block:

```tsx
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
```

Replace with:

```tsx
                            const allSplitEntries = Object.entries(evaluationData.splits) as [string, EvaluationSplit][];
                            const trainEntry = showTrainMetrics ? allSplitEntries.find(([n]) => n === 'train') : undefined;
                            const testEntry  = showTestMetrics  ? allSplitEntries.find(([n]) => n === 'test')  : undefined;
                            const valEntry   = showValMetrics   ? allSplitEntries.find(([n]) => n === 'validation') : undefined;
                            const isBinary = (evaluationData.splits.train?.y_proba?.classes.length ?? 0) === 2;
                            const renderSplit = isBinary ? renderSplitBinary : renderSplitPerClass;

                            return (
                                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        {trainEntry && renderSplit(trainEntry[0], trainEntry[1])}
                                        {testEntry  && renderSplit(testEntry[0],  testEntry[1])}
                                        {!trainEntry && !testEntry && (
                                            <p className="col-span-2 text-xs text-gray-400 text-center py-8">Enable Train or Test splits above to compare.</p>
                                        )}
                                    </div>
                                    {valEntry && (
                                        <div className="mt-6 pt-4 border-t border-gray-100 dark:border-gray-700">
                                            {renderSplit(valEntry[0], valEntry[1])}
                                        </div>
                                    )}
                                </div>
                            );

};
```

- [ ] **Step 6: Run the tests to verify they now pass**

Run: `cd frontend/ml-canvas && npx vitest run src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx`
Expected: all tests (existing + new) PASS.

- [ ] **Step 7: Lint and type-check**

Run:
```bash
cd frontend/ml-canvas
npx eslint src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx
npx tsc --project tsconfig.json --noEmit
```
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.tsx \
        frontend/ml-canvas/src/components/pages/ExperimentsPage/components/PerClassConfusionMatrix.test.tsx
git commit -m "feat: render a single confusion matrix for binary jobs in PerClassConfusionMatrix

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Full verification pass

**Files:** none (verification only)

**Interfaces:** N/A

- [ ] **Step 1: Full lint**

Run: `cd frontend/ml-canvas && npm run lint`
Expected: exits 0 (no errors, no warnings — `--max-warnings 0`).

- [ ] **Step 2: Full type-check**

Run: `cd frontend/ml-canvas && npx tsc --project tsconfig.json --noEmit`
Expected: exits 0.

- [ ] **Step 3: Production build**

Run: `cd frontend/ml-canvas && npm run build`
Expected: build succeeds with no errors.

- [ ] **Step 4: Full test suite**

Run: `cd frontend/ml-canvas && npx vitest run`
Expected: all tests pass, including the new `EvaluationView.test.tsx` and the extended `PerClassConfusionMatrix.test.tsx`.

- [ ] **Step 5: Manual smoke check (optional but recommended)**

If a local dev server is available: `npm run dev`, open the Experiments page, select a completed classification job, and confirm:
- Both "Threshold Slider" and "Threshold Tuning" tab buttons appear.
- Tab 1 behaves exactly as before (slider, badges, Overall/Per-Class toggle for multiclass).
- Tab 2 shows the placeholder text until "Preview" is clicked, then shows a confusion matrix — for both a binary job and a multiclass job.
- Train/Test/Validation checkboxes affect both tabs identically.

- [ ] **Step 6: Final commit (if any cleanup was needed)**

```bash
git status
# If anything is dirty from the smoke check, commit or discard as appropriate.
```
