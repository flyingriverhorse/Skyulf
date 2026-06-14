import { useEffect, useMemo, useState } from 'react';
import { ChevronRight, SlidersHorizontal } from 'lucide-react';
import { jobsApi } from '../../../../core/api/jobs';
import type { HyperparameterDef } from './types';

/**
 * Maps each selectable base-learner key to its registry key so the matching
 * hyperparameter definitions can be fetched. Mirrors the core maps in
 * `skyulf.modeling.hyperparameters._registry`.
 */
const REGISTRY_KEY: Record<'classification' | 'regression', Record<string, string>> = {
  classification: {
    logistic_regression: 'logistic_regression',
    random_forest: 'random_forest_classifier',
    extra_trees: 'extra_trees_classifier',
    gradient_boosting: 'gradient_boosting_classifier',
    hist_gradient_boosting: 'hist_gradient_boosting_classifier',
    adaboost: 'adaboost_classifier',
    decision_tree: 'decision_tree_classifier',
    gaussian_nb: 'gaussian_nb',
    sgd_classifier: 'sgd_classifier',
    svc: 'svc',
    knn: 'k_neighbors_classifier',
    xgboost: 'xgboost_classifier',
    lightgbm: 'lgbm_classifier',
  },
  regression: {
    linear_regression: 'linear_regression',
    ridge: 'ridge_regression',
    lasso: 'lasso_regression',
    elasticnet: 'elasticnet_regression',
    random_forest: 'random_forest_regressor',
    extra_trees: 'extra_trees_regressor',
    gradient_boosting: 'gradient_boosting_regressor',
    hist_gradient_boosting: 'hist_gradient_boosting_regressor',
    adaboost: 'adaboost_regressor',
    decision_tree: 'decision_tree_regressor',
    svr: 'svr',
    knn: 'k_neighbors_regressor',
    xgboost: 'xgboost_regressor',
    lightgbm: 'lgbm_regressor',
  },
};

type ParamMap = Record<string, unknown>;

function coerce(def: HyperparameterDef, raw: string): unknown {
  if (def.type === 'number') return raw === '' ? undefined : Number(raw);
  if (def.type === 'boolean') return raw === 'true';
  return raw === '' ? undefined : raw;
}

/** Renders a single typed input (number / select / boolean) for one param. */
function ParamInput({ def, value, onChange }: {
  def: HyperparameterDef;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const current = value ?? '';
  const cls =
    'w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-xs ' +
    'bg-white dark:bg-gray-800 dark:text-gray-100';
  if (def.type === 'select' && def.options) {
    return (
      <select className={cls} value={String(current)} onChange={(e) => { onChange(coerce(def, e.target.value)); }}>
        <option value="">default</option>
        {def.options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>{o.label}</option>
        ))}
      </select>
    );
  }
  if (def.type === 'boolean') {
    return (
      <select className={cls} value={current === '' ? '' : String(current)} onChange={(e) => { onChange(e.target.value === '' ? undefined : e.target.value === 'true'); }}>
        <option value="">default</option>
        <option value="true">True</option>
        <option value="false">False</option>
      </select>
    );
  }
  return (
    <input
      type="number"
      className={cls}
      min={def.min}
      max={def.max}
      step={def.step}
      value={String(current)}
      placeholder="default"
      onChange={(e) => { onChange(coerce(def, e.target.value)); }}
    />
  );
}

/** Collapsible form of one base model's tunable params. */
function ModelSection({ label, defs, values, onParam }: {
  label: string;
  defs: HyperparameterDef[];
  values: ParamMap;
  onParam: (param: string, value: unknown) => void;
}) {
  const [open, setOpen] = useState(false);
  const setCount = Object.keys(values).length;
  const renderable = defs.filter((d) => d.type !== 'multiselect');
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={() => { setOpen(!open); }}
        className="w-full flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <span className="text-xs font-medium text-gray-700 dark:text-gray-200">{label}</span>
        <span className="flex items-center gap-1.5">
          {setCount > 0 && (
            <span className="text-[10px] text-purple-600 dark:text-purple-300">{setCount} set</span>
          )}
          <ChevronRight className={`w-3.5 h-3.5 text-gray-400 transition-transform ${open ? 'rotate-90' : ''}`} />
        </span>
      </button>
      {open && (
        <div className="p-2 grid grid-cols-2 gap-2 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          {renderable.length === 0 && (
            <p className="col-span-2 text-[11px] text-gray-400">No tunable parameters.</p>
          )}
          {renderable.map((def) => (
            <div key={def.name}>
              <span className="block text-[10px] text-gray-500 mb-0.5 truncate" title={def.description}>{def.label}</span>
              <ParamInput
                def={def}
                value={values[def.name]}
                onChange={(v) => { onParam(def.name, v); }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Lets the user set fixed hyperparameters for each selected base model (and the
 * stacking final estimator). Writes a `{ name: { param: value } }` map that the
 * backend applies via `set_params` on each base learner.
 */
export function BaseModelParamsEditor({
  task, baseEstimators, finalEstimator, optionLabels,
  baseParams, finalParams, onChange,
}: {
  task: 'classification' | 'regression';
  baseEstimators: string[];
  finalEstimator?: string | undefined;
  optionLabels: Record<string, string>;
  baseParams: Record<string, ParamMap>;
  finalParams: ParamMap;
  onChange: (baseParams: Record<string, ParamMap>, finalParams: ParamMap) => void;
}) {
  const [defsByKey, setDefsByKey] = useState<Record<string, HyperparameterDef[]>>({});

  // Every distinct model whose defs we need (base learners + final estimator).
  const wanted = useMemo(() => {
    const keys = new Set(baseEstimators);
    if (finalEstimator) keys.add(finalEstimator);
    return Array.from(keys);
  }, [baseEstimators, finalEstimator]);

  useEffect(() => {
    let cancelled = false;
    const missing = wanted.filter((k) => !(k in defsByKey));
    if (missing.length === 0) return;
    void Promise.all(
      missing.map(async (key) => {
        const registryKey = REGISTRY_KEY[task][key];
        if (!registryKey) return [key, [] as HyperparameterDef[]] as const;
        try {
          const defs = (await jobsApi.getHyperparameters(registryKey)) as HyperparameterDef[];
          return [key, defs] as const;
        } catch {
          return [key, [] as HyperparameterDef[]] as const;
        }
      }),
    ).then((entries) => {
      if (cancelled) return;
      setDefsByKey((prev) => ({ ...prev, ...Object.fromEntries(entries) }));
    });
    return () => { cancelled = true; };
  }, [wanted, task, defsByKey]);

  const setBaseParam = (name: string, param: string, value: unknown) => {
    const bucket: ParamMap = { ...(baseParams[name] ?? {}) };
    if (value === undefined || value === '') delete bucket[param];
    else bucket[param] = value;
    const next = { ...baseParams };
    if (Object.keys(bucket).length === 0) delete next[name];
    else next[name] = bucket;
    onChange(next, finalParams);
  };

  const setFinalParam = (param: string, value: unknown) => {
    const next = { ...finalParams };
    if (value === undefined || value === '') delete next[param];
    else next[param] = value;
    onChange(baseParams, next);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5">
        <SlidersHorizontal className="w-3.5 h-3.5 text-purple-500" />
        <span className="text-[11px] text-gray-500 dark:text-gray-400">
          Optional — leave blank to use each model&apos;s defaults.
        </span>
      </div>
      {baseEstimators.map((name) => (
        <ModelSection
          key={name}
          label={optionLabels[name] ?? name}
          defs={defsByKey[name] ?? []}
          values={baseParams[name] ?? {}}
          onParam={(param, value) => { setBaseParam(name, param, value); }}
        />
      ))}
      {finalEstimator && (
        <ModelSection
          key={`final-${finalEstimator}`}
          label={`Final · ${optionLabels[finalEstimator] ?? finalEstimator}`}
          defs={defsByKey[finalEstimator] ?? []}
          values={finalParams}
          onParam={setFinalParam}
        />
      )}
    </div>
  );
}
