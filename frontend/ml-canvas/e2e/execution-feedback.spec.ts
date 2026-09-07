import { test, expect, type Page, type Route } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

/** Seed configured training branches while retaining the real submission and job stores. */
async function seed(page: Page, width = 1440, modelType = 'classification') {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {} } }));
  await page.setViewportSize({ width, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(modelType => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    const add = state.addNode as (type: string, position: unknown) => string;
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([], []);
    const ids = ['dataset_node', 'TrainTestSplitter', modelType, modelType].map((type, index) => add(type, { x: index * 220, y: index === 3 ? 160 : 0 }));
    const update = state.updateNodeData as (id: string, data: unknown) => void;
    update(ids[0]!, { datasetId: 'iris-demo', datasetName: 'Iris' });
    update(ids[2]!, { target_column: 'species', label: 'Forest A' });
    update(ids[3]!, { target_column: 'species', label: 'Forest B', run_mode: 'advanced' });
    (state.onConnect as (edge: unknown) => void)({ source: ids[0], sourceHandle: 'data', target: ids[1], targetHandle: 'in' });
    for (const target of ids.slice(2)) (state.onConnect as (edge: unknown) => void)({ source: ids[1], sourceHandle: 'train', target, targetHandle: 'in' });
    (state.onNodesChange as (changes: unknown[]) => void)(ids.map(id => ({ id, type: 'select', selected: false })));
  }, modelType);
}

/** Select through React Flow's normal change handler so keyed settings really remount. */
async function selectModel(page: Page, index: number) {
  await page.evaluate(index => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.onNodesChange(state.nodes.map((node, i) => ({ id: node.id, type: 'select', selected: i === index })));
  }, index);
}

/** Reach the same experiment review from either responsive toolbar location. */
async function review(page: Page) {
  const direct = page.getByTestId('toolbar-run-all');
  if (await direct.isVisible()) await direct.click();
  else {
    await page.getByRole('button', { name: 'More canvas tools', exact: true }).click();
    await page.getByRole('menuitem', { name: 'Run all experiments', exact: true }).click();
  }
}

test('app errors remain accessible in the bell outside canvas without toast popups', async ({ page }) => {
  // Removing the global toast surface must not hide errors on pages without the canvas navbar.
  await mockBackend(page);
  await page.route('**/data/api/sources', route => route.fulfill({ json: {
    sources: sampleDatasets.map(dataset => ({ ...dataset, type: 'file', format: 'csv', created_at: '2026-09-07T12:00:00Z' })),
  } }));
  await page.route('**/data/api/sources/*/export?*', route => route.fulfill({ status: 500, json: { detail: 'Unavailable' } }));
  await page.goto('/data');
  await page.getByRole('button', { name: 'CSV', exact: true }).click();
  const bell = page.getByRole('button', { name: 'Notifications (1)', exact: true });
  await expect(bell).toBeVisible();
  await expect(page.locator('[data-sonner-toaster]')).toHaveCount(0);
  await bell.click();
  await expect(page.getByText('Export failed Could not export the dataset. Please try again.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Close', exact: true }).click();
  await page.setViewportSize({ width: 390, height: 800 });
  await expect(bell).toHaveCount(1);
  await expect(bell).toBeInViewport();
  await bell.click();
  await expect(page.getByText('Export failed Could not export the dataset. Please try again.', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Clear all', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Notifications', exact: true })).toBeVisible();
});

for (const theme of ['light', 'dark']) {
  test(`experiment review names the models and follows submitted jobs on a ${theme} canvas`, async ({ page }) => {
    // Review/cancel must make no request; the confirmed run must keep mixed outcomes visible.
    await page.addInitScript(theme => localStorage.setItem('skyulf-theme', theme), theme);
    await seed(page, theme === 'light' ? 1440 : 1100);
    let requests = 0;
    let pendingRoute: Route | undefined;
    await page.route('**/api/pipeline/run', route => { requests++; pendingRoute = route; });
    await review(page);
    const dialog = page.getByRole('dialog', { name: 'Run all experiments?', exact: true });
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('Forest A');
    await expect(dialog).toContainText('Forest B');
    await expect(dialog).toContainText('Train');
    await expect(dialog).toContainText('Tune');
    await expect(dialog).toContainText('random forest classifier');
    await page.screenshot({ path: `test-results/experiment-review-${theme}.png`, animations: 'disabled' });
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
    const opener = await page.getByTestId('toolbar-run-all').isVisible() ? page.getByTestId('toolbar-run-all') : page.getByTestId('toolbar-more');
    await expect(opener).toBeFocused();
    expect(requests).toBe(0);
    await review(page);
    await dialog.getByRole('button', { name: 'Queue experiments', exact: true }).click();
    await expect.poll(() => requests).toBe(1);
    await expect(page.locator('[data-canvas-toolbar] [role="status"]')).toHaveCount(0);
    const base = { pipeline_id: 'run', node_id: 'training', job_type: 'training', model_type: 'random_forest_classifier',
      created_at: '2026-09-07T12:00:00Z', start_time: null, end_time: null, error: null, result: null };
    await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [
      { ...base, job_id: 'first', status: 'completed' }, { ...base, job_id: 'second', status: 'failed' },
    ] }));
    await pendingRoute!.fulfill({ json: { job_id: 'first', job_ids: ['first', 'second'], pipeline_id: 'run', message: 'Submitted' } });
    await page.getByRole('button', { name: 'Show all jobs', exact: true }).click();
    await page.getByRole('button', { name: 'Regression', exact: true }).click();
    await page.getByPlaceholder('Search by job ID, dataset, or model...').fill('unrelated');
    await page.getByRole('button', { name: 'Close job history', exact: true }).click();
    await page.getByRole('tab', { name: 'Experiments', exact: true }).click();
    await page.getByRole('button', { name: /^Notifications/ }).click();
    const status = page.getByRole('status').filter({ hasText: 'Experiments:' });
    await expect(status).toContainText('1 completed');
    await expect(status).toContainText('1 failed');
    await page.getByRole('button', { name: 'View jobs', exact: true }).click();
    await expect(page.getByRole('tab', { name: 'Canvas', exact: true })).toHaveAttribute('aria-selected', 'true');
    const jobs = page.getByRole('dialog', { name: 'Job History', exact: true });
    await expect(jobs.getByText('first', { exact: true })).toBeVisible();
    await expect(jobs.getByText('second', { exact: true })).toBeVisible();
  });
}

test('keyboard preview explains blocked input and reports completion for a valid graph', async ({ page }) => {
  // Keyboard and mouse preview share visible feedback and retain the existing preview API.
  await seed(page);
  await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => { setGraph: (n: unknown[], e: unknown[]) => void } } } }).__skyulfTest.graphStore.getState();
    state.setGraph([], []);
  });
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+Enter');
  await expect(page.locator('[data-sonner-toaster]')).toHaveCount(0);
  await expect(page.locator('[data-canvas-toolbar]')).not.toContainText('Preview blocked');
  await page.getByRole('tab', { name: 'Inference', exact: true }).click();
  await page.getByRole('button', { name: /^Notifications/ }).click();
  await expect(page.getByText('Preview blocked. Add a dataset node and select a dataset.', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Notifications (1)', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Review preview results', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Review preview results', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Canvas', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: 'Review preview results', exact: true })).toHaveCount(0);
  await seed(page);
  let pendingRoute: Route | undefined;
  await page.route('**/api/pipeline/preview', route => { pendingRoute = route; });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Previewing data...', exact: true })).toBeDisabled();
  await expect.poll(() => Boolean(pendingRoute)).toBe(true);
  await pendingRoute!.fulfill({ json: { pipeline_id: 'preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  await expect(page.getByRole('button', { name: 'Preview data', exact: true })).toBeEnabled();
  await expect(page.getByRole('status').filter({ hasText: 'Preview completed' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Review preview results', exact: true })).toHaveCount(0);
});

for (const scenario of [
  { type: 'classification', index: 2, action: 'Train model', label: 'Training — random forest classifier' },
  { type: 'classification', index: 3, action: 'Tune model', label: 'Tuning — random forest classifier' },
  { type: 'EnsembleNode', index: 2, action: 'Train ensemble', label: 'Training — voting classifier' },
  { type: 'SegmentationNode', index: 2, action: 'Train segmentation', label: 'Segmentation — kmeans' },
]) {
  test(`${scenario.action} retains submission feedback across settings remounts`, async ({ page }) => {
    // Every training entry point must identify its action and keep its guard when users switch nodes.
    if (scenario.index === 3) {
      await page.emulateMedia({ reducedMotion: 'reduce' });
      await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'dark'));
    }
    await seed(page, scenario.index === 3 ? 1100 : 1440, scenario.type);
    let requests = 0;
    let pendingRoute: Route | undefined;
    await page.route('**/api/pipeline/run', route => { requests++; pendingRoute = route; });
    await selectModel(page, scenario.index);
    await page.getByRole('button', { name: scenario.action, exact: true }).click();
    await expect.poll(() => requests).toBe(1);
    await expect(page.getByRole('button', { name: 'Submitting job...', exact: true })).toBeDisabled();
    await selectModel(page, scenario.index === 2 ? 3 : 2);
    await expect(page.getByRole('button', { name: 'Submitting job...', exact: true })).toHaveCount(0);
    await selectModel(page, scenario.index);
    await expect(page.getByRole('button', { name: 'Submitting job...', exact: true })).toBeDisabled();
    await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [{
      job_id: 'selected-run', pipeline_id: 'run', node_id: 'selected', job_type: 'training', status: 'queued',
      created_at: new Date().toISOString(), start_time: null, end_time: null, result: null, error: null,
    }] }));
    await pendingRoute!.fulfill({ json: { job_id: 'selected-run', job_ids: ['selected-run'], pipeline_id: 'run', message: 'Submitted' } });
    const history = page.getByRole('dialog', { name: 'Job History', exact: true });
    await expect(history).toBeVisible();
    const tab = scenario.type === 'EnsembleNode' ? 'Ensemble' : scenario.type === 'SegmentationNode' ? 'Segmentation' : 'Classification';
    await expect(history.getByRole('button', { name: tab, exact: true })).toHaveClass(/border-blue-500/);
    await expect(history.getByRole('button', { name: 'Show all jobs', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Close job history', exact: true }).click();
    await expect(page.getByRole('status').filter({ hasText: scenario.label })).toContainText('1 queued');
    const feedback = page.getByRole('status').filter({ hasText: scenario.label });
    await expect.poll(() => feedback.evaluate(element => {
      const box = element.getBoundingClientRect();
      return box.left >= 0 && box.right <= innerWidth && box.top >= 0 && box.bottom <= innerHeight;
    })).toBe(true);
    await selectModel(page, scenario.index === 2 ? 3 : 2);
    await expect(page.getByRole('button', { name: 'View jobs', exact: true })).toHaveCount(0);
    expect(requests).toBe(1);
  });
}

test('blocked experiment review explains the fix and closes when canvas becomes read-only', async ({ page }) => {
  // Disabled submission must expose its reason and never reopen after a read-only round trip.
  await seed(page, 1100);
  await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.updateNodeData(state.nodes[2]!.id, { target_column: '' });
  });
  await review(page);
  const dialog = page.getByRole('dialog', { name: 'Run all experiments?', exact: true });
  await expect(dialog.getByRole('button', { name: 'Queue experiments' })).toBeDisabled();
  await expect(dialog).toContainText('Target column');
  await dialog.getByRole('button', { name: 'Review validation issues' }).click();
  await expect(dialog).toHaveCount(0);
  await review(page);
  await page.setViewportSize({ width: 900, height: 800 });
  await expect(dialog).toHaveCount(0);
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(dialog).toHaveCount(0);
});

for (const theme of ['light', 'dark']) {
  test(`training actions stay compact while scrolling and expand at the end in ${theme} mode`, async ({ page }) => {
    // Long tuning settings must keep the action reachable without covering fields with help text.
    await page.addInitScript(theme => localStorage.setItem('skyulf-theme', theme), theme);
    await page.emulateMedia({ reducedMotion: theme === 'dark' ? 'reduce' : 'no-preference' });
    await seed(page, theme === 'dark' ? 1100 : 1440);
    await selectModel(page, 3);
    const footer = page.getByTestId('training-action-footer');
    await expect(footer).toHaveAttribute('data-expanded', 'false');
    const compactHeight = await footer.evaluate(element => element.getBoundingClientRect().height);
    expect(compactHeight).toBeLessThan(80);
    await expect(footer.getByRole('button', { name: 'Tune model', exact: true })).toBeInViewport();
    await expect(footer.getByRole('button', { name: 'View Best Parameters History' })).toHaveCount(0);
    await page.screenshot({ path: `test-results/training-compact-${theme}.png`, animations: 'disabled' });
    await footer.evaluate(element => {
      let parent = element.parentElement;
      while (parent && !/auto|scroll/.test(getComputedStyle(parent).overflowY)) parent = parent.parentElement;
      if (!parent) throw new Error('Settings scroll container missing');
      parent.scrollTop = parent.scrollHeight;
    });
    await expect(footer).toHaveAttribute('data-expanded', 'true');
    await expect(footer.getByRole('button', { name: 'View Best Parameters History' })).toBeInViewport();
    await expect.poll(() => footer.evaluate(element => element.getBoundingClientRect().height)).toBeGreaterThan(compactHeight + 30);
    await page.screenshot({ path: `test-results/training-expanded-${theme}.png`, animations: 'disabled' });
    await footer.getByRole('button', { name: 'Tune model', exact: true }).focus();
    await footer.evaluate(element => {
      let parent = element.parentElement;
      while (parent && !/auto|scroll/.test(getComputedStyle(parent).overflowY)) parent = parent.parentElement;
      parent!.scrollTop = 0;
    });
    await expect(footer).toHaveAttribute('data-expanded', 'false');
    await expect(footer.getByRole('button', { name: 'Tune model', exact: true })).toBeFocused();
    await expect(footer.getByRole('button', { name: 'Tune model', exact: true })).toBeInViewport();
  });
}
