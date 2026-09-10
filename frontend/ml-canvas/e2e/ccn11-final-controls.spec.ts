import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Use real registry definitions, schema filtering and graph conversion for all vectorizers. */
async function prepareVectorizers(page: Page, width: number) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    review: { name: 'review', dtype: 'object' },
    rating: { name: 'rating', dtype: 'int64' },
  } } }));
  await page.setViewportSize({ width, height: 900 });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  return page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 100 });
    state.updateNodeData(dataset, { datasetId: 'reviews-demo', datasetName: 'Reviews' });
    const ids = ['count_vectorizer', 'tfidf_vectorizer', 'hashing_vectorizer'].map((type, index) => {
      const id = state.addNode(type, { x: 240, y: index * 160 });
      state.onConnect({ source: dataset, sourceHandle: 'data', target: id, targetHandle: 'in' });
      return id;
    });
    state.onNodesChange(window.__skyulfTest!.graphStore.getState().nodes.map(node => ({
      id: node.id, type: 'select', selected: node.id === ids[0],
    })));
    return { count: ids[0]!, tfidf: ids[1]!, hashing: ids[2]! };
  });
}

/** Revisit settings through the same graph selection action as a canvas click. */
async function selectNode(page: Page, id: string) {
  await page.evaluate(id => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.onNodesChange(state.nodes.map(node => ({ id: node.id, type: 'select', selected: node.id === id })));
  }, id);
}

for (const width of [1440, 1100]) {
  test(`vectorizer variants preserve values and Preview payloads at ${width}px`, async ({ page }) => {
    // Conditional controls must remain editable and retain independent per-node values.
    const ids = await prepareVectorizers(page, width);
    let submitted: { nodes: { node_id: string; params: Record<string, unknown> }[] } | undefined;
    await page.route('**/api/pipeline/preview?*', route => {
      submitted = route.request().postDataJSON() as typeof submitted;
      return route.fulfill({ json: { pipeline_id: 'ccn11-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
    });
    const settings = page.getByRole('tabpanel', { name: 'Settings', exact: true });
    await settings.getByRole('checkbox', { name: 'review', exact: true }).check();
    await expect(settings.getByRole('checkbox', { name: 'rating', exact: true })).toHaveCount(0);
    await settings.getByRole('spinbutton', { name: 'Max features', exact: true }).fill('25');
    await settings.getByRole('spinbutton', { name: 'N-gram max', exact: true }).fill('2');
    await settings.getByRole('checkbox', { name: /^Binary counts/ }).check();
    await settings.getByRole('checkbox', { name: /^Remove English stop words/ }).check();
    await selectNode(page, ids.tfidf);
    await settings.getByRole('checkbox', { name: 'review', exact: true }).check();
    await settings.getByRole('spinbutton', { name: 'Max features', exact: true }).fill('50');
    await settings.getByRole('spinbutton', { name: 'Max features', exact: true }).fill('');
    await settings.getByRole('checkbox', { name: /^Sublinear TF scaling/ }).check();
    await expect(settings.getByRole('checkbox', { name: /^Binary counts/ })).toHaveCount(0);
    await selectNode(page, ids.hashing);
    await settings.getByRole('checkbox', { name: 'review', exact: true }).check();
    await settings.getByRole('spinbutton', { name: 'Number of features (hash buckets)', exact: true }).fill('2048');
    await settings.getByRole('combobox', { name: 'Normalization', exact: true }).selectOption('none');
    await settings.getByRole('checkbox', { name: /^Alternate sign/ }).uncheck();
    await expect(settings.getByRole('spinbutton', { name: 'Max features', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
    const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
    await expect.poll(() => panel.evaluate(element => {
      element.getBoundingClientRect();
      return element.getAnimations().every(animation => animation.playState === 'finished');
    })).toBe(true);
    await expect(settings.locator('.grid.grid-cols-1.gap-4')).toHaveClass(/md:grid-cols-2/);
    await expect(settings.getByRole('combobox', { name: 'Normalization', exact: true })).toHaveValue('none');
    await page.screenshot({ path: `test-results/ccn11-hashing-expanded-${width}.png`, animations: 'disabled' });
    await page.getByRole('button', { name: 'Collapse settings panel', exact: true }).click();
    await selectNode(page, ids.count);
    await expect(settings.getByRole('spinbutton', { name: 'Max features', exact: true })).toHaveValue('25');
    await expect(settings.getByRole('checkbox', { name: /^Binary counts/ })).toBeChecked();
    await selectNode(page, ids.tfidf);
    await expect(settings.getByRole('spinbutton', { name: 'Max features', exact: true })).toHaveValue('');
    await expect(settings.getByRole('checkbox', { name: /^Sublinear TF scaling/ })).toBeChecked();
    await page.getByRole('button', { name: 'Preview data', exact: true }).click();
    await expect.poll(() => submitted?.nodes.length).toBe(4);
    const params = (id: string) => submitted!.nodes.find(node => node.node_id === id)!.params;
    expect(params(ids.count)).toMatchObject({ columns: ['review'], max_features: 25, ngram_range: [1, 2], binary: true, stop_words: 'english' });
    expect(params(ids.tfidf)).toMatchObject({ columns: ['review'], max_features: null, sublinear_tf: true });
    expect(params(ids.hashing)).toMatchObject({ columns: ['review'], n_features: 2048, norm: 'none', alternate_sign: false });
  });
}
