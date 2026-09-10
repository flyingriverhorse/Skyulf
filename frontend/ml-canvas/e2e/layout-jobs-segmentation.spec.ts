import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

const registry = [
  { id: 'random_forest_classifier', name: 'Random Forest Classifier', category: 'Modeling', description: '', params: {}, tags: ['classification'] },
  { id: 'random_forest_regressor', name: 'Random Forest Regressor', category: 'Modeling', description: '', params: {}, tags: ['regression'] },
  { id: 'kmeans', name: 'K-Means', category: 'Modeling', description: '', params: {}, tags: ['clustering', 'requires_scaling'] },
  { id: 'birch', name: 'Birch', category: 'Modeling', description: '', params: {}, tags: ['clustering'] },
];
const parameters = [
  { name: 'n_clusters', label: 'Number of Clusters', type: 'number', default: 3, min: 2, max: 20, step: 1 },
  { name: 'random_state', label: 'Random State', type: 'number', default: 42, min: 0, step: 1 },
];
const baseJob = {
  pipeline_id: 'history', node_id: 'model', job_type: 'training', status: 'completed',
  dataset_id: 'iris-demo', dataset_name: 'Iris', model_type: 'random_forest_classifier',
  created_at: '2026-09-10T10:00:00Z', start_time: '2026-09-10T10:00:00Z',
  end_time: '2026-09-10T10:00:05Z', error: null, result: { metrics: { test_accuracy: 0.91234 } },
};
const jobs = [
  { ...baseJob, job_id: 'classification-done' },
  { ...baseJob, job_id: 'classification-failed', status: 'failed', error: 'Example failure', result: null, model_type: 'logistic_regression' },
  { ...baseJob, job_id: 'regression-done', model_type: 'random_forest_regressor', dataset_name: 'Housing', result: { metrics: { test_r2: 0.81234 } } },
  { ...baseJob, job_id: 'ensemble-done', model_type: 'voting_classifier' },
  { ...baseJob, job_id: 'segmentation-done', model_type: 'kmeans', result: { metrics: { train_silhouette_score: 0.72345 } } },
];

/** Keep real page and store behavior while isolating this browser check from services. */
async function prepare(page: Page, width: number) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: registry }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: parameters }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: jobs }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    sepal_length: { name: 'sepal_length', dtype: 'float64' },
    species: { name: 'species', dtype: 'object' },
  } } }));
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
  await page.setViewportSize({ width, height: 900 });
}

/** Use the same history action exposed by the responsive toolbar. */
async function openJobs(page: Page) {
  const direct = page.getByRole('button', { name: 'Job runs history', exact: true });
  if (await direct.isVisible()) await direct.click();
  else {
    await page.getByRole('button', { name: 'More canvas tools', exact: true }).click();
    await page.getByRole('menuitem', { name: 'Jobs', exact: true }).click();
  }
  await expect(page.getByRole('dialog', { name: 'Job History', exact: true })).toBeVisible();
}

for (const width of [1440, 390]) {
  test(`app navigation keeps route, theme and keyboard access at ${width}px`, async ({ page }) => {
    // Moving between real pages must retain active links, a single bell and menu focus.
    await prepare(page, width);
    await page.goto('/data');
    const sidebar = page.locator('#app-sidebar');
    const opener = page.getByRole('button', { name: 'Open navigation menu', exact: true });
    if (width < 768) {
      await opener.click();
      await expect(sidebar.getByRole('link', { name: 'Dashboard', exact: true })).toBeFocused();
      await page.keyboard.press('Escape');
      await expect(opener).toBeFocused();
      await expect(opener).toHaveAttribute('aria-expanded', 'false');
      await opener.click();
    }
    await expect(sidebar.getByRole('link', { name: 'Data Sources', exact: true })).toHaveAttribute('aria-current', 'page');
    await sidebar.getByRole('button', { name: 'Switch to dark mode', exact: true }).click();
    await expect(page.locator('html')).toHaveClass(/dark/);
    await sidebar.getByRole('link', { name: 'ML Canvas', exact: true }).click();
    await expect(page).toHaveURL(/\/canvas(?:\?|$)/);
    await expect(page.locator('.react-flow')).toBeVisible();
    await expect(page.getByRole('button', { name: /^Notifications/ })).toHaveCount(1);
    if (width < 768) {
      await expect(opener).toHaveAttribute('aria-expanded', 'false');
      await opener.click();
    } else await expect(sidebar).toHaveCSS('width', '64px');
    await expect(sidebar.getByRole('link', { name: 'ML Canvas', exact: true })).toHaveAttribute('aria-current', 'page');
    await sidebar.getByRole('link', { name: 'Data Sources', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Upload File', exact: true })).toBeVisible();
    await expect(page.locator('html')).toHaveClass(/dark/);
    await expect(page.getByRole('button', { name: /^Notifications/ })).toHaveCount(1);
    if (width >= 768) await expect(sidebar).toHaveCSS('width', '256px');
  });
}

for (const width of [1440, 1100]) {
  test(`job history filters and keyboard detail navigation survive at ${width}px`, async ({ page }) => {
    // The real drawer/card pair must preserve filtering and selection through detail views.
    await prepare(page, width);
    await page.goto('/canvas');
    await openJobs(page);
    const drawer = page.getByRole('dialog', { name: 'Job History', exact: true });
    await expect(drawer).toBeFocused();
    await expect(drawer.getByText('classification-done', { exact: true })).toBeVisible();
    await expect(drawer.getByText('classification-failed', { exact: true })).toBeVisible();
    await expect(drawer.getByTitle('test split')).toContainText('0.912');
    await drawer.getByRole('button', { name: 'Filters', exact: true }).click();
    await drawer.getByRole('combobox').first().selectOption('failed');
    await expect(drawer.getByText('classification-done', { exact: true })).toHaveCount(0);
    await expect(drawer.getByText('classification-failed', { exact: true })).toBeVisible();
    await drawer.getByRole('button', { name: 'Clear all', exact: true }).click();
    const search = drawer.getByPlaceholder('Search by job ID, dataset, or model...');
    await search.fill('logistic');
    await expect(drawer.getByText('classification-done', { exact: true })).toHaveCount(0);
    await search.fill('');
    await drawer.getByRole('button', { name: 'Regression', exact: true }).click();
    const card = drawer.getByRole('button').filter({ hasText: 'regression-done' });
    await expect(card).toContainText('0.812');
    await card.press('Enter');
    const details = page.getByRole('dialog').filter({ has: page.getByRole('heading', { name: /^Job Details/ }) });
    await expect(details).toBeVisible();
    await details.getByRole('button').first().click();
    await expect(card).toBeVisible();
    await drawer.getByRole('button', { name: 'Ensemble', exact: true }).click();
    await expect(drawer.getByText('ensemble-done', { exact: true })).toBeVisible();
    await drawer.getByRole('button', { name: 'Regression', exact: true }).last().click();
    await expect(drawer.getByText('ensemble-done', { exact: true })).toHaveCount(0);
    await drawer.getByRole('button', { name: 'Classification', exact: true }).last().click();
    await expect(drawer.getByText('ensemble-done', { exact: true })).toBeVisible();
    await drawer.getByRole('button', { name: 'Segmentation', exact: true }).click();
    await expect(drawer.getByText('segmentation-done', { exact: true })).toBeVisible();
    await page.screenshot({ path: `test-results/ccn7-jobs-${width}.png`, animations: 'disabled' });
    await page.keyboard.press('Escape');
    await expect(drawer).toHaveCount(0);
    await openJobs(page);
    await expect(drawer.getByText('segmentation-done', { exact: true })).toBeVisible();
  });

  test(`segmentation edits persist through layout changes and reach submission at ${width}px`, async ({ page }) => {
    // Public controls must update the graph and submit the unsupervised pipeline contract.
    await prepare(page, width);
    let submission: Record<string, unknown> | undefined;
    await page.route('**/api/pipeline/run', async route => {
      submission = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({ json: { job_id: 'segmentation-done', job_ids: ['segmentation-done'], pipeline_id: 'history', message: 'Submitted' } });
    });
    await page.goto('/canvas');
    await page.waitForFunction(() => '__skyulfTest' in window);
    const modelId = await page.evaluate(() => {
      const state = window.__skyulfTest!.graphStore.getState();
      state.setGraph([], []);
      const dataset = state.addNode('dataset_node', { x: 0, y: 0 });
      const model = state.addNode('SegmentationNode', { x: 240, y: 0 });
      state.updateNodeData(dataset, { datasetId: 'iris-demo', datasetName: 'Iris' });
      state.onConnect({ source: dataset, sourceHandle: 'data', target: model, targetHandle: 'in' });
      state.onNodesChange(window.__skyulfTest!.graphStore.getState().nodes.map(node => ({ id: node.id, type: 'select', selected: node.id === model })));
      return model;
    });
    const algorithm = page.getByRole('combobox', { name: 'Clustering Algorithm', exact: true });
    const reference = page.getByRole('combobox', { name: 'Reference Column (optional)', exact: true });
    await expect(algorithm).toHaveValue('kmeans');
    await expect(reference).toBeEnabled();
    await reference.selectOption('species');
    await page.getByRole('button', { name: 'Hyperparameters', exact: true }).click();
    const clusters = page.getByRole('textbox', { name: 'Number of Clusters', exact: true });
    await clusters.fill('5');
    await clusters.press('Tab');
    await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
    await expect(algorithm).toBeVisible();
    await expect(clusters).toHaveValue('5');
    await expect(reference).toHaveValue('species');
    await algorithm.selectOption('birch');
    await expect(clusters).toHaveValue('3');
    await clusters.fill('4');
    await clusters.press('Tab');
    await page.screenshot({ path: `test-results/ccn7-segmentation-${width}.png`, animations: 'disabled' });
    await page.getByRole('button', { name: 'Train segmentation', exact: true }).click();
    await expect.poll(() => submission?.target_node_id).toBe(modelId);
    expect(submission?.job_type).toBe('training');
    const submittedNodes = submission?.nodes as { params: Record<string, unknown> }[];
    const submittedModel = submittedNodes.find(node => node.params.model_type === 'birch');
    expect(submittedModel?.params).toMatchObject({
      target_column: '', cv_enabled: false, reference_column: 'species', hyperparameters: { n_clusters: 4 },
    });
    const drawer = page.getByRole('dialog', { name: 'Job History', exact: true });
    await expect(drawer.getByRole('button', { name: 'Segmentation', exact: true })).toHaveClass(/border-blue-500/);
    await expect(drawer.getByText('segmentation-done', { exact: true })).toBeVisible();
  });
}
