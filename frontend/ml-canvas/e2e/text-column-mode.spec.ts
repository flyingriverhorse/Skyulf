import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('per-column vocabularies retain keyboard selection and actual preview configuration on mobile return', async ({ page }) => {
  // The existing Count panel exports mode and keeps it when the mobile view becomes read-only.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: { title: { name: 'title', dtype: 'object' }, body: { name: 'body', dtype: 'object' } } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const graph = window.__skyulfTest!.graphStore.getState();
    graph.setGraph([], []);
    const dataset = graph.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'documents', datasetName: 'Documents' });
    const vectorizer = graph.addNode('count_vectorizer', { x: 250, y: 100 }, { columns: ['title', 'body'] });
    graph.onConnect({ source: dataset, sourceHandle: 'data', target: vectorizer, targetHandle: 'in' });
    return vectorizer;
  });
  const mode = page.getByRole('combobox', { name: 'Vocabulary columns' });
  await mode.focus();
  await mode.press('End');
  await mode.press('Tab');
  await expect(mode).toHaveValue('per_column');
  let submitted: { nodes: { node_id: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'text-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.length).toBe(2);
  expect(submitted!.nodes.find(node => node.node_id === id)!.params).toMatchObject({ columns: ['title', 'body'], column_mode: 'per_column' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(mode).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(mode).toHaveValue('per_column');
});
