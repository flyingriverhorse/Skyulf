import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

const contextJob = {
  job_id: 'context-job', pipeline_id: 'context-pipeline', node_id: 'model',
  job_type: 'training', status: 'completed', model_type: 'random_forest_classifier',
  dataset_id: 'iris-demo', dataset_name: 'A&B Iris',
  created_at: '2026-09-10T10:00:00Z', start_time: '2026-09-10T10:00:00Z',
  end_time: '2026-09-10T10:00:05Z', error: null, result: {},
};

/** Keep real stores and routing while supplying deterministic dataset and job responses. */
async function prepare(page: Page, width = 1440) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [{
    id: 'random_forest_classifier', name: 'Random Forest', category: 'Modeling',
    description: '', params: {}, tags: ['classification'],
  }] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [contextJob] }));
  await page.route('**/api/pipeline/jobs/context-job', route => route.fulfill({ json: contextJob }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    age: { name: 'age', dtype: 'float64' },
    height: { name: 'height', dtype: 'int64' },
    species: { name: 'species', dtype: 'object' },
  } } }));
  await page.setViewportSize({ width, height: 900 });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
}

/** Seed real node definitions and select settings using the public graph actions. */
async function seedSettings(page: Page) {
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  return page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 0 });
    const scaling = state.addNode('scale_numeric_features', { x: 240, y: 0 });
    const outlier = state.addNode('outlier', { x: 480, y: 0 });
    state.updateNodeData(dataset, { datasetId: 'iris-demo', datasetName: 'Iris' });
    state.onConnect({ source: dataset, sourceHandle: 'data', target: scaling, targetHandle: 'in' });
    state.onConnect({ source: scaling, sourceHandle: 'out', target: outlier, targetHandle: 'in' });
    state.onNodesChange(window.__skyulfTest!.graphStore.getState().nodes.map(node => ({
      id: node.id, type: 'select', selected: node.id === scaling,
    })));
    return { scaling, outlier };
  });
}

/** Selection updates must not discard either node's controlled settings. */
async function selectNode(page: Page, id: string) {
  await page.evaluate(id => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.onNodesChange(state.nodes.map(node => ({ id: node.id, type: 'select', selected: node.id === id })));
  }, id);
}

/** Wait for both the width transition and ResizeObserver before inspecting layout. */
async function expectSettingsLayout(page: Page, wide: boolean) {
  const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
  await expect.poll(() => panel.evaluate(element => {
    element.getBoundingClientRect();
    return element.getAnimations().every(animation => animation.playState === 'finished');
  })).toBe(true);
  const content = page.getByRole('tabpanel', { name: 'Settings', exact: true }).locator('.gap-4').first();
  await expect(content).toHaveCSS('display', wide ? 'grid' : 'flex');
  await expect(content).toHaveClass(wide ? /\bgrid-cols-2\b/ : /\bflex-col\b/);
  await expect.poll(() => content.evaluate(element => element.getBoundingClientRect().width > 450)).toBe(wide);
}

test('processing fan-in cancel is atomic and accepted wiring supports undo and redo', async ({ page }) => {
  // Canceling the real confirmation must leave the first input and both source nodes intact.
  await prepare(page);
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    state.addNode('imputation_node', { x: 0, y: 0 });
    state.addNode('imputation_node', { x: 0, y: 240 });
    state.addNode('encoding', { x: 400, y: 120 });
    state.setGraph(window.__skyulfTest!.graphStore.getState().nodes.map((node, index) => ({
      ...node, id: `fan-${index}`, selected: false,
    })), [{ id: 'existing', source: 'fan-0', sourceHandle: 'out', target: 'fan-2', targetHandle: 'in', type: 'custom' }]);
  });
  const collapse = page.getByRole('button', { name: 'Collapse results panel', exact: true });
  if (await collapse.isVisible()) await collapse.click();
  await page.locator('.react-flow__controls-fitview').click();
  const trigger = page.locator('[data-id="fan-1"]').getByRole('button', { name: 'Next step from Cleaned Data', exact: true });
  const picker = page.getByRole('dialog', { name: 'Connect next step', exact: true });
  const messages: string[] = [];
  for (const accept of [false, true]) {
    await trigger.click();
    await picker.getByRole('button', { name: 'Existing node', exact: true }).click();
    page.once('dialog', dialog => {
      messages.push(dialog.message());
      void (accept ? dialog.accept() : dialog.dismiss());
    });
    await picker.getByRole('button', { name: 'Connect to Encoding using Data', exact: true }).click();
    await expect(page.locator('.react-flow__edge')).toHaveCount(accept ? 2 : 1);
    await expect(page.locator('.react-flow__node')).toHaveCount(3);
    if (await picker.isVisible()) await page.keyboard.press('Escape');
  }
  expect(messages).toHaveLength(2);
  expect(messages[0]).toBe(messages[1]);
  expect(messages[0]).toContain('merge');
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+z');
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
  await expect(page.locator('.react-flow__node')).toHaveCount(3);
  await page.keyboard.press('Control+Shift+z');
  await expect(page.locator('.react-flow__edge')).toHaveCount(2);
});

for (const width of [1440, 1100]) {
  test(`scaling and outlier controls preserve values across node and panel changes at ${width}px`, async ({ page }) => {
    // Exercise all method sections and verify the values held by the real graph, including numeric zero.
    await prepare(page, width);
    let submitted: { nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[] } | undefined;
    await page.route('**/api/pipeline/preview?*', route => {
      submitted = route.request().postDataJSON() as typeof submitted;
      return route.fulfill({ json: { pipeline_id: 'ccn8-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
    });
    const ids = await seedSettings(page);
    const settings = page.getByRole('tabpanel', { name: 'Settings', exact: true });
    const method = settings.getByRole('combobox', { name: /^(Scaling )?Method$/ });
    await expect(settings.getByRole('checkbox', { name: 'age', exact: true })).toBeVisible();
    await expectSettingsLayout(page, false);
    await expect(settings.getByRole('checkbox', { name: 'species', exact: true })).toHaveCount(0);
    await settings.getByRole('checkbox', { name: 'age', exact: true }).check();
    await method.selectOption('minmax');
    await settings.getByRole('spinbutton', { name: 'Feature Range Minimum', exact: true }).fill('1');
    await settings.getByRole('spinbutton', { name: 'Feature Range Minimum', exact: true }).fill('0');
    await settings.getByRole('spinbutton', { name: 'Feature Range Maximum', exact: true }).fill('2');
    await method.selectOption('maxabs');
    await expect(settings.getByRole('spinbutton')).toHaveCount(0);
    await method.selectOption('robust');
    await settings.getByRole('spinbutton', { name: 'Quantile Range Minimum', exact: true }).fill('10');
    await settings.getByRole('spinbutton', { name: 'Quantile Range Maximum', exact: true }).fill('90');
    await settings.getByRole('checkbox', { name: 'Center Data (Median)', exact: true }).uncheck();
    await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
    await expectSettingsLayout(page, true);
    await expect(settings.getByRole('spinbutton', { name: 'Quantile Range Minimum', exact: true })).toHaveValue('10');
    await expect(settings.getByRole('checkbox', { name: 'age', exact: true })).toBeChecked();
    await selectNode(page, ids.outlier);
    await expectSettingsLayout(page, true);
    await settings.getByRole('checkbox', { name: 'height', exact: true }).check();
    await settings.getByRole('spinbutton', { name: 'Multiplier', exact: true }).fill('2');
    await method.selectOption('zscore');
    await settings.getByRole('spinbutton', { name: 'Threshold (Sigma)', exact: true }).fill('4');
    await method.selectOption('winsorize');
    await settings.getByRole('spinbutton', { name: 'Lower Percentile', exact: true }).fill('0');
    await settings.getByRole('spinbutton', { name: 'Upper Percentile', exact: true }).fill('90');
    await method.selectOption('elliptic_envelope');
    await settings.getByRole('spinbutton', { name: 'Contamination', exact: true }).fill('0.05');
    await page.screenshot({ path: `test-results/ccn8-outlier-expanded-${width}.png`, animations: 'disabled' });
    await page.getByRole('button', { name: 'Collapse settings panel', exact: true }).click();
    await expectSettingsLayout(page, false);
    await method.selectOption('winsorize');
    await expect(settings.getByRole('spinbutton', { name: 'Lower Percentile', exact: true })).toHaveValue('0');
    await selectNode(page, ids.scaling);
    await expect(method).toHaveValue('robust');
    await expect(settings.getByRole('checkbox', { name: 'Center Data (Median)', exact: true })).not.toBeChecked();
    await method.selectOption('minmax');
    await expect(settings.getByRole('spinbutton', { name: 'Feature Range Maximum', exact: true })).toHaveValue('2');
    await expectSettingsLayout(page, false);
    await page.screenshot({ path: `test-results/ccn8-settings-${width}.png`, animations: 'disabled' });
    const configs = await page.evaluate(ids => {
      const nodes = window.__skyulfTest!.graphStore.getState().nodes;
      return [ids.scaling, ids.outlier].map(id => nodes.find(node => node.id === id)!.data);
    }, ids);
    expect(configs[0]).toMatchObject({ columns: ['age'], method: 'minmax', feature_range_min: 0, feature_range_max: 2, quantile_range_min: 10, quantile_range_max: 90, with_centering: false });
    expect(configs[1]).toMatchObject({ columns: ['height'], method: 'winsorize', multiplier: 2, threshold: 4, lower_percentile: 0, upper_percentile: 90, contamination: 0.05 });
    await page.getByRole('button', { name: 'Preview data', exact: true }).click();
    await expect.poll(() => submitted?.nodes.length).toBe(3);
    expect(submitted?.nodes.find(node => node.node_id === ids.scaling)).toMatchObject({ step_type: 'MinMaxScaler', params: { columns: ['age'], feature_range: [0, 2] } });
    expect(submitted?.nodes.find(node => node.node_id === ids.outlier)).toMatchObject({ step_type: 'Winsorize', params: { columns: ['height'], lower_percentile: 0, upper_percentile: 90 } });
  });
}

for (const width of [1440, 900]) {
  test(`job record links restore encoded search and status after reload at ${width}px`, async ({ page }) => {
    // Navigate through a real RecordLink and decode its context in the owning Jobs page.
    await prepare(page, width);
    await page.goto('/jobs');
    await page.getByPlaceholder('Search jobs...').fill('A&B');
    await page.getByRole('button', { name: 'Filters', exact: true }).click();
    await page.getByRole('combobox').selectOption('completed');
    const link = page.getByRole('link', { name: 'Job context-job', exact: true });
    await expect(link).toHaveAttribute('href', '/jobs?oc.kind=job&oc.jobId=context-job&oc.origin=%2Fjobs&oc.f.tab=classification&oc.f.q=A%26B&oc.f.status=completed');
    await link.click();
    const heading = page.getByRole('heading', { name: /^Job Details/ });
    await expect(heading).toBeVisible();
    await expect(page).toHaveURL(/oc\.f\.q=A%26B/);
    await page.reload();
    await expect(heading).toBeVisible();
    // The existing Back icon lacks an accessible name (OC-225), so scope by its header.
    await heading.locator('../..').getByRole('button').first().click();
    await expect(page.getByPlaceholder('Search jobs...')).toHaveValue('A&B');
    await page.getByRole('button', { name: /^Filters/ }).click();
    await expect(page.getByRole('combobox')).toHaveValue('completed');
    await expect(link).toHaveAttribute('href', /oc\.f\.status=completed/);
  });
}
