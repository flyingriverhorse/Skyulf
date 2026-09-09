import type { ComponentProps } from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { JobInfo } from '../../../../core/api/jobs';
import { getMetricDescription } from '../../../../core/utils/format';
import { ComparisonTableView } from './ComparisonTableView';

vi.mock('../../../ui/InfoTooltip', () => ({
  InfoTooltip: ({ text }: { text: string }) => <span data-testid="info-tooltip" title={text} />,
}));

function job(id: string, overrides: Partial<JobInfo> = {}): JobInfo {
  return {
    job_id: id, pipeline_id: `preview_${id}__branch_0`, node_id: id,
    job_type: 'training', status: 'completed', start_time: null, end_time: null,
    error: null, result: null, created_at: '2026-09-09', model_type: 'random_forest',
    ...overrides,
  };
}

function props(selectedJobs: JobInfo[], overrides: Partial<ComponentProps<typeof ComparisonTableView>> = {}) {
  return {
    selectedJobs, metricKeys: [], isPipelineExpanded: true, setIsPipelineExpanded: vi.fn(),
    isMetricsExpanded: true, setIsMetricsExpanded: vi.fn(), isParamsExpanded: true,
    setIsParamsExpanded: vi.fn(), isTuningExpanded: true, setIsTuningExpanded: vi.fn(),
    ...overrides,
  };
}

function row(label: string): HTMLElement {
  return screen.getByText(label).closest('tr')!;
}

function values(label: string): string[] {
  return within(row(label)).getAllByRole('cell').slice(1).map(cell => cell.textContent!);
}

describe('ComparisonTableView characterization', () => {
  it('renders the full default row order, run labels, badges and empty section fallbacks', () => {
    // Default rows and identity badges remain available without stored artifacts.
    render(<ComparisonTableView {...props([job('abcdefghijk', { branch_index: 0, promoted_at: '2026-09-09' })])} />);
    expect(screen.getAllByRole('columnheader').map(cell => cell.textContent)).toEqual(['Parameter / Metric', 'abcdefghij']);
    expect(screen.getAllByRole('row').map(r => r.firstElementChild?.textContent)).toEqual([
      'Parameter / Metric', 'Model Type', 'Pipeline Steps', 'No upstream pipeline steps captured for these runs.',
      'Key Metrics', 'Hyperparameters', 'Default parameters (none customized)', 'Training Configuration',
      'Target Column', 'CV Enabled', 'CV Method', 'CV Folds', 'CV Shuffle', 'CV Random State',
    ]);
    expect(values('Model Type')).toEqual(['random_forest Path A Winner']);
    expect(values('CV Enabled')).toEqual(['-']);
    expect(screen.queryByText('Base Models')).not.toBeInTheDocument();
  });

  it('calls all four expansion setters with the inverse controlled value and hides collapsed content', () => {
    // Parent-controlled expansion must retain callback arguments and section visibility.
    const input = props([job('a')]);
    const view = render(<ComparisonTableView {...input} />);
    const sections = ['Pipeline Steps', 'Key Metrics', 'Hyperparameters', 'Training Configuration'];
    sections.forEach(label => fireEvent.click(screen.getByText(label)));
    [input.setIsPipelineExpanded, input.setIsMetricsExpanded, input.setIsParamsExpanded, input.setIsTuningExpanded]
      .forEach(setter => expect(setter).toHaveBeenLastCalledWith(false));
    view.rerender(<ComparisonTableView {...input} isPipelineExpanded={false} isMetricsExpanded={false} isParamsExpanded={false} isTuningExpanded={false} />);
    sections.forEach(label => fireEvent.click(screen.getByText(label)));
    [input.setIsPipelineExpanded, input.setIsMetricsExpanded, input.setIsParamsExpanded, input.setIsTuningExpanded]
      .forEach(setter => expect(setter).toHaveBeenLastCalledWith(true));
    expect(screen.getAllByRole('row')).toHaveLength(6);
  });

  it('preserves metric source precedence, zero, numeric formatting, direction, ties and missing tooltips', () => {
    // Empty top-level metrics still shadow result metrics and only finite comparable values earn a star.
    render(<ComparisonTableView {...props([
      job('a', { metrics: { test_accuracy: 0, rmse: 2, cv_accuracy_std: 0.01234567, r2: 1, custom: 5, mae: Infinity }, result: { metrics: { test_accuracy: 0.9 } } }),
      job('b', { result: { metrics: { test_accuracy: 0.7, rmse: 1, cv_accuracy_std: 0, r2: 1, custom: 6, mae: NaN } } }),
      job('c', { metrics: {}, result: { metrics: { test_accuracy: 1 } } }),
      job('d', { result: { metrics: { test_accuracy: '0.8', mae: 4 } } }),
    ], { metricKeys: ['test_accuracy', 'rmse', 'cv_accuracy_std', 'r2', 'custom', 'mae'] })} />);
    expect(values('Test Accuracy')).toEqual(['0.0000', '0.7000 ★', '—', '—']);
    expect(values('Rmse')).toEqual(['2.0000', '1.0000 ★', '—', '—']);
    expect(values('Cv Accuracy Std')).toEqual(['0.012346', '0.000000 ★', '—', '—']);
    expect(values('R2')).toEqual(['1.0000', '1.0000', '—', '—']);
    expect(values('Custom')).toEqual(['5.0000', '6.0000', '—', '—']);
    expect(values('Mae')).toEqual(['Infinity', 'NaN', '—', '4.0000']);
    expect(within(row('Test Accuracy')).getAllByTitle('test_accuracy was not reported by this run')).toHaveLength(2);
    expect(within(row('Test Accuracy')).getByText('Test (held-out)')).toBeInTheDocument();
    expect(within(row('Test Accuracy')).getByTitle(getMetricDescription('test_accuracy')!)).toBeInTheDocument();
    expect(within(row('Test Accuracy')).getAllByRole('cell')[2]).toHaveClass('text-green-600', 'font-semibold');
  });

  it('groups best scores in scoring-metric appearance order and omits groups without numeric values', () => {
    // Stars are confined to scoring groups; best_score keeps its existing higher-is-better direction.
    render(<ComparisonTableView {...props([
      job('a', { metrics: { best_score: 0.4 }, result: { scoring_metric: 'accuracy' } }),
      job('b', { metrics: { best_score: 8 }, config: { tuning_config: { metric: 'rmse' } } }),
      job('c', { result: { metrics: { best_score: 0.8, scoring_metric: 'accuracy' } } }),
      job('d', { metrics: { best_score: 2 }, result: { scoring_metric: 'rmse' } }),
      job('e', { metrics: { best_score: 0 } }),
      job('f', { result: { scoring_metric: 'f1' } }),
    ], { metricKeys: ['best_score'] })} />);
    expect(values('Best Score (Accuracy)')).toEqual(['0.4000', '—', '0.8000 ★', '—', '—', '—']);
    expect(values('Best Score (RMSE)')).toEqual(['—', '8.0000 ★', '—', '2.0000', '—', '—']);
    expect(values('Best Score (CV)')).toEqual(['—', '—', '—', '—', '0.0000', '—']);
    expect(screen.queryByText('Best Score (F1 Score)')).not.toBeInTheDocument();
    expect(screen.getAllByText(/^Best Score \(/).map(el => el.firstChild?.textContent)).toEqual(['Best Score (Accuracy)', 'Best Score (RMSE)', 'Best Score (CV)']);
  });

  it('unions nested basic and direct tuned model parameters in first-appearance order', () => {
    // Config wrapper keys and invalid nested arrays must never become model parameter rows.
    render(<ComparisonTableView {...props([
      job('a', { hyperparameters: { target_column: 'label', hyperparameters: { n_estimators: 0, max_depth: null, custom: { a: 1 } } } }),
      job('b', { search_strategy: 'random', hyperparameters: { max_depth: 4, enabled: false, list: [1, 2] } }),
      job('c', { hyperparameters: { hyperparameters: ['bad'] } }),
    ])} />);
    expect(values('n_estimators')).toEqual(['0', '-', '-']);
    expect(values('max_depth')).toEqual(['null', '4', '-']);
    expect(values('custom')).toEqual(['{"a":1}', '-', '-']);
    expect(values('enabled')).toEqual(['-', 'false', '-']);
    expect(values('list')).toEqual(['-', '[1,2]', '-']);
    const labels = screen.getAllByRole('row').map(r => r.firstElementChild?.textContent);
    expect(labels.slice(labels.indexOf('Hyperparameters') + 1, labels.indexOf('Training Configuration'))).toEqual(['n_estimators', 'max_depth', 'custom', 'enabled', 'list']);
  });

  it('resolves training and tuning configs with graph fallbacks and retains zero and empty scalar semantics', () => {
    // Nested tuning CV settings and explicit empty config objects retain their original precedence.
    render(<ComparisonTableView {...props([
      job('a', { hyperparameters: { target_column: '', cv_enabled: true, cv_type: '', cv_folds: 0, cv_shuffle: false, cv_random_state: 0 }, target_column: 'fallback' }),
      job('b', { search_strategy: 'optuna', config: { target_column: 'y', tuning_config: { cv_enabled: true, cv_type: 'stratified', cv_folds: 5, cv_shuffle: true, cv_random_state: 7, strategy: 'optuna', metric: 'f1', n_trials: 0 } } }),
      job('c', { job_type: 'tuning', graph: { nodes: [{ node_id: 'c', params: { target_column: 'graph_y', tuning_config: { cv_enabled: false, search_strategy: 'halving_grid' } } }] } }),
      job('d', { hyperparameters: {}, graph: { nodes: [{ node_id: 'd', params: { target_column: 'ignored' } }] }, target_column: 'fallback' }),
      job('e', { graph: { nodes: [{ node_id: 'e', params: { target_column: 'basic_graph', cv_enabled: true, cv_type: null } }] } }),
    ])} />);
    expect(values('Target Column')).toEqual(['-', 'y', 'graph_y', 'fallback', 'basic_graph']);
    expect(values('CV Enabled')).toEqual(['Yes', 'Yes', 'No', 'No', 'Yes']);
    expect(values('CV Method')).toEqual(['Unknown', 'stratified', '-', '-', 'Unknown']);
    expect(values('CV Folds')).toEqual(['0', '5', '-', '-', '-']);
    expect(values('CV Shuffle')).toEqual(['No', 'Yes', '-', '-', 'No']);
    expect(values('CV Random State')).toEqual(['0', '7', '-', '-', '-']);
    expect(values('Strategy')).toEqual(['-', 'optuna', 'halving_grid', '-', '-']);
    expect(values('Strategy Params')).toEqual(['-', 'sampler: tpe · pruner: median (defaults)', 'factor: 3 · min: exhaust (defaults)', '-', '-']);
    expect(values('Trials')).toEqual(['-', '0', '-', '-', '-']);
  });

  it('renders explicit strategy parameters and all remaining strategy defaults', () => {
    // Explicit strategy params win over defaults, while unknown strategies stay blank.
    render(<ComparisonTableView {...props([
      job('a', { job_type: 'tuning', config: { tuning_config: { strategy: 'optuna', strategy_params: { seed: 0 } } } }),
      job('b', { job_type: 'tuning', config: { tuning_config: { strategy: 'halving_random', strategy_params: {} } } }),
      job('c', { job_type: 'tuning', config: { tuning_config: { strategy: 'random' } } }),
      job('d', { job_type: 'tuning', config: {}, graph: { nodes: [{ node_id: 'd', params: { target_column: 'ignored' } }] } }),
      job('e', { job_type: 'tuning' }),
    ])} />);
    expect(values('Strategy Params')).toEqual(['{"seed":0}', 'factor: 3 · min: exhaust (defaults)', '-', '-', '-']);
    expect(values('Target Column')).toEqual(['-', '-', '-', '-', '-']);
    expect(within(row('CV Enabled')).getAllByRole('cell')[5]).toHaveClass('text-gray-400');
  });

  it('renders ensemble summaries from basic nested/flat params and tuned config/graph selection', () => {
    // Structural estimator choices are separate from tuned best parameters and absent for ordinary models.
    render(<ComparisonTableView {...props([
      job('a', { model_type: 'stacking_classifier', hyperparameters: { hyperparameters: { base_estimators: ['random_forest', 'svc'], final_estimator: 'logistic_regression' } } }),
      job('b', { model_type: 'voting_classifier', hyperparameters: { base_estimators: ['custom_tree'] } }),
      job('c', { model_type: 'stacking_regressor', job_type: 'tuning', hyperparameters: { base_estimators: ['ignored'] }, config: { tuning_config: { base_estimators: ['ridge'], final_estimator: 'linear_regression' } } }),
      job('d', { model_type: 'voting_regressor', search_strategy: 'random', graph: { nodes: [{ node_id: 'd', params: { tuning_config: { base_estimators: ['lasso'] } } }] } }),
      job('e'), job('f', { model_type: 'stacking_classifier' }),
    ])} />);
    expect(values('Base Models')).toEqual(['Random Forest, SVC', 'Custom Tree', 'Ridge', 'Lasso', '—', '—']);
    expect(values('Final Estimator')).toEqual(['Logistic Regression', '—', 'Linear Regression', '—', '—', '—']);
  });

  it('keeps missing ensemble configurations and empty stacking summaries distinct', () => {
    // An empty stacking bucket still creates its final-estimator row; absent buckets do not.
    const view = render(<ComparisonTableView {...props([job('a', { model_type: 'stacking_classifier' })])} />);
    expect(values('Base Models')).toEqual(['—']);
    expect(screen.queryByText('Final Estimator')).not.toBeInTheDocument();
    view.rerender(<ComparisonTableView {...props([job('a', { model_type: 'stacking_classifier', hyperparameters: {} })])} />);
    expect(values('Base Models')).toEqual(['—']);
    expect(values('Final Estimator')).toEqual(['—']);
  });

  it('aligns shared, divergent and cyclic ancestor chains by id in the original merge order', () => {
    // Shared nodes occur once, cycles terminate, and branch-only rows preserve their gaps and tones.
    const common = { node_id: 'root', step_type: 'data_source', params: { columns: ['a', 'b'] } };
    render(<ComparisonTableView {...props([
      job('a', { graph: { nodes: [common, { node_id: 'left', step_type: 'StandardScaler', inputs: ['root', 'missing'] }, { node_id: 'a', inputs: ['left'] }] } }),
      job('b', { graph: { nodes: [common, { node_id: 'right', step_type: 'simple_imputer', inputs: ['root', 'cycle'], params: { strategy: 'mean' } }, { node_id: 'cycle', inputs: ['right'], params: { _display_name: 'Cycle' } }, { node_id: 'b', inputs: ['right'] }] } }),
    ])} />);
    expect(values('Step 1')).toEqual(['Data Source (columns=[a, b])', 'Data Source (columns=[a, b])']);
    expect(values('Step 2')).toEqual(['Standard Scaler', '—']);
    expect(values('Step 3')).toEqual(['—', 'Cycle']);
    expect(values('Step 4')).toEqual(['—', 'Simple Imputer (strategy=mean)']);
    expect(row('Step 1')).toHaveClass('bg-white');
    expect(row('Step 2')).toHaveClass('bg-amber-50/40');
    expect(within(row('Step 2')).getAllByRole('cell')[2]).toHaveClass('text-gray-300');
    expect(screen.queryByText('Step 5')).not.toBeInTheDocument();
  });

  it('preserves operation summaries, key ordering, truncation and skipped empty details', () => {
    // Pipeline cells retain useful operation details without changing truncation or scalar formatting.
    const operations = [
      { method: 'multiply', input_columns: ['a', 'b'], secondary_columns: ['c'] },
      { operation_type: 'month', input_columns: ['date'] }, null, { method: 'unused' },
    ];
    render(<ComparisonTableView {...props([job('a', { graph: { nodes: [
      { node_id: 'features', step_type: 'feature_generation', params: { _display_name: 'Derived', method: { kind: 'x' }, strategy: '', columns: ['a', 'b', 'c', 'd', 'e'], target_column: null, test_size: 0, val_size: false, random_state: 0, n_neighbors: [], operations } },
      { node_id: 'a', inputs: ['features'] },
    ] } })])} />);
    expect(values('Step 1')).toEqual(['Derived (method={"kind":"x"}, columns=[a, b, c, d, +1 more], test_size=0, val_size=false, random_state=0, ops=[multiply(a, b, +1), month(date), op, +1 more])']);
  });
});
