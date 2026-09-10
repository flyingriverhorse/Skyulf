import { test, expect, type Page, type Locator } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';
import type { EvaluationData, ClusteringSplit } from '../src/components/pages/ExperimentsPage/types';

const classificationSplit = {
  y_true: [0, 0, 1, 1], y_pred: [0, 0, 1, 1],
  // The page selects the first probability class; deliberately explore class 1.
  y_proba: { classes: [1, 0], values: [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]] },
};

/** Keep cluster counts, profiles and labels consistent across the two split fixtures. */
function clusterSplit(sizes: [number, number], silhouette: number): ClusteringSplit {
  const total = sizes[0] + sizes[1];
  return {
    labels: [...Array<number>(sizes[0]).fill(0), ...Array<number>(sizes[1]).fill(1)],
    metrics: { silhouette_score: silhouette, calinski_harabasz_score: 145.3, davies_bouldin_score: 0.71 },
    clustering: {
      n_clusters: 2,
      cluster_sizes: { '0': sizes[0], '1': sizes[1] },
      centroids: sizes.map((size, cluster_id) => ({
        cluster_id, size, percentage: 100 * size / total,
        center: { feature_a: cluster_id === 0 ? 1.2 : -1.1 },
        profile: cluster_id === 0 ? 'Higher feature_a' : 'Lower feature_a',
      })),
    },
  };
}

const evaluations: Record<string, EvaluationData> = {
  'ccn-classifier': { problem_type: 'classification', splits: { train: classificationSplit, test: classificationSplit } },
  'ccn-regressor': {
    problem_type: 'regression',
    splits: {
      train: { y_true: [10, 20, 30, 40], y_pred: [10, 20, 30, 40] },
      test: { y_true: [10, 20, 30, 40], y_pred: [9, 19, 29, 39] },
    },
  },
  'ccn-clusters': {
    problem_type: 'clustering',
    splits: { train: clusterSplit([2, 1], 0.62), test: clusterSplit([1, 3], 0.48) },
  },
};

/** Override only real experiment HTTP endpoints; chart libraries remain unmocked. */
async function mockExperimentResults(page: Page) {
  await mockBackend(page);
  const definitions = [
    { id: 'random_forest_classifier', tags: ['classification'] },
    { id: 'random_forest_regressor', tags: ['regression'] },
    { id: 'kmeans', tags: ['clustering'] },
  ];
  const runs = [
    { job_id: 'ccn-classifier', pipeline_id: 'class-run', model_type: 'random_forest_classifier', dataset_name: 'CCN classification dataset', metrics: { train_accuracy: 1, test_accuracy: 1 } },
    { job_id: 'ccn-regressor', pipeline_id: 'reg-run', model_type: 'random_forest_regressor', dataset_name: 'CCN regression dataset', metrics: { train_r2: 1, test_r2: 0.992 } },
    { job_id: 'ccn-clusters', pipeline_id: 'cluster-run', model_type: 'kmeans', dataset_name: 'CCN cluster dataset', metrics: { train_silhouette_score: 0.62, test_silhouette_score: 0.48 } },
  ];
  const jobs = runs.map(run => ({
    ...run, node_id: 'model', job_type: 'training', status: 'completed',
    start_time: '2026-09-10T10:00:00Z', end_time: '2026-09-10T10:01:00Z',
    created_at: '2026-09-10T10:00:00Z', error: null, result: {}, config: {},
  }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: jobs }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({
    json: definitions.map(definition => ({ ...definition, name: definition.id, category: 'modeling', description: '', params: {} })),
  }));
  await page.route('**/api/pipeline/jobs/*/evaluation', route => {
    const jobId = new URL(route.request().url()).pathname.split('/').at(-2)!;
    expect(evaluations[jobId]).toBeDefined();
    return route.fulfill({ json: evaluations[jobId] });
  });
  await page.route('**/api/pipeline/jobs/*/thresholds', route => route.fulfill({
    json: { thresholds: null, classes: null, metric: null, split_used: null, computed_at: null, source: null, enabled: false },
  }));
}

test.beforeEach(async ({ page }) => {
  await mockExperimentResults(page);
  await page.goto('/canvas');
  await page.getByRole('tab', { name: 'Experiments', exact: true }).click();
});

/** Wait for drawn SVG shapes to have stable, visible geometry after Recharts animation. */
async function expectRenderedShapes(chart: Locator, selector: string, count: number) {
  await chart.scrollIntoViewIfNeeded();
  const paths = chart.locator(selector);
  await expect(paths).toHaveCount(count);
  let previous = '';
  await expect.poll(async () => {
    const geometry = await paths.evaluateAll(elements => elements.map(element => {
      const bounds = element.getBoundingClientRect();
      return {
        path: element.getAttribute('d'), transform: element.getAttribute('transform'),
        width: bounds.width, height: bounds.height, x: bounds.x, y: bounds.y,
      };
    }));
    const snapshot = JSON.stringify(geometry);
    const stable = snapshot === previous;
    previous = snapshot;
    return stable && geometry.length === count && geometry.every(shape =>
      shape.width > 1 && shape.height > 1 && /^M/.test(shape.path ?? '')
      && !/NaN|Infinity/.test(`${shape.path} ${shape.transform}`));
  }).toBe(true);
}

test('classification charts respond to threshold and split changes, then switch to regression through selected runs', async ({ page }) => {
  // Real plot geometry and numerical summaries must follow the active run and controls.
  await page.getByText('CCN classification dataset', { exact: false }).click();
  await page.getByRole('button', { name: 'Model Evaluation', exact: true }).click();
  const testRoc = page.locator('#test-roc');
  await expect(testRoc).toContainText('ROC Curve — 1');
  await expect(testRoc).toContainText('AUC=1.000');
  await expect(testRoc.locator('.recharts-line-curve').first()).toHaveAttribute('d', /^M.+/);
  const truePositive = page.locator('#test-confusion-matrix [title^="True: 1, Pred: 1"]');
  await expect(truePositive).toHaveAttribute('title', /Count: 2/);
  await page.getByRole('slider').press('End');
  await expect(page.getByRole('slider')).toHaveValue('0.99');
  await expect(truePositive).toHaveAttribute('title', /Count: 0/);
  await page.getByRole('checkbox', { name: 'Train', exact: true }).uncheck();
  await expect(page.locator('#train-roc')).toHaveCount(0);
  await expect(testRoc).toBeVisible();

  await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Expand Sidebar', exact: true }).click();
  await expect(page.getByText('Select Runs (1)', { exact: true })).toBeVisible();
  await page.getByText('CCN regression dataset', { exact: false }).click();
  const runs = page.getByRole('tablist', { name: 'Select run for evaluation' });
  await runs.getByRole('tab', { name: 'reg-run', exact: true }).click();
  await expect(runs.getByRole('tab', { name: 'reg-run', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expectRenderedShapes(page.locator('#test-actual-pred'), '.recharts-scatter-symbol path', 4);
  await expect(page.getByText('P50 (median): 1.0000', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  const trainPlot = page.locator('#train-actual-pred');
  await expectRenderedShapes(trainPlot, '.recharts-scatter-symbol path', 4);
  await expect(page.locator('#test-actual-pred')).toHaveCount(0);
  await expect(page.getByText('P50 (median): 0.0000', { exact: true })).toBeVisible();
  await trainPlot.screenshot({ path: 'test-results/ccn9-regression-results.png' });
  await runs.getByRole('tab', { name: 'class-run', exact: true }).click();
  await expect(testRoc).toContainText('AUC=1.000');
  await expect(page.locator('#train-actual-pred')).toHaveCount(0);
  await page.screenshot({ path: 'test-results/ccn9-classification-results.png', fullPage: true });
});

test('segmentation renders real cluster bars and updates profiles, metrics and table rows for the selected split', async ({ page }) => {
  // Changing the split must replace chart data and its accessible table together.
  await page.getByText('CCN cluster dataset', { exact: false }).click();
  await page.getByRole('button', { name: 'Segmentation', exact: true }).click();
  const chart = page.locator('#segmentation-cluster-sizes-chart');
  await expectRenderedShapes(chart, '.recharts-bar-rectangle path', 2);
  await expect(page.getByText('0.620', { exact: true })).toBeVisible();
  await expect(page.getByText('66.7% (2)', { exact: true })).toBeVisible();
  await expect(page.getByText('Higher feature_a', { exact: true })).toBeVisible();
  await chart.getByRole('button', { name: 'View data table', exact: true }).click();
  const table = chart.getByRole('table');
  await expect(table.getByRole('row', { name: 'Cluster 0 2', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Test', exact: true }).click();
  await expect(page.getByText('0.480', { exact: true })).toBeVisible();
  await expect(page.getByText('75.0% (3)', { exact: true })).toBeVisible();
  await expect(table.getByRole('row', { name: 'Cluster 0 1', exact: true })).toBeVisible();
  await expect(table.getByRole('row', { name: 'Cluster 1 3', exact: true })).toBeVisible();
  await expectRenderedShapes(chart, '.recharts-bar-rectangle path', 2);
  await chart.screenshot({ path: 'test-results/ccn9-segmentation-results.png' });
});
