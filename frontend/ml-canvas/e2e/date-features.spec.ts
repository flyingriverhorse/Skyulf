import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('local calendar settings support keyboard validation, preview and mobile persistence', async ({ page }) => {
  // The real form must submit its timezone/cycles and preserve them across the read-only breakpoint.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    date: { name: 'date', dtype: 'object' },
  } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const source = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'dates', datasetName: 'Dates' });
    const id = state.addNode('TimeSeriesNode', { x: 250, y: 100 }, { method: 'date', columns: ['date'], features: ['second'] });
    state.onConnect({ source, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  const timezone = page.getByRole('textbox', { name: 'Timezone', exact: true });
  await expect(timezone).toHaveValue('UTC');
  await timezone.fill('No/Such_Zone');
  const issue = page.getByRole('button', { name: /Configuration Time Series.*IANA timezone/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(timezone).toBeFocused();
  await expect(timezone).toHaveAccessibleDescription(/IANA timezone/);
  await timezone.fill('Europe/Istanbul');
  await expect(issue).toHaveCount(0);
  const sine = page.getByRole('button', { name: 'hour_sin', exact: true });
  await sine.focus();
  await sine.press('Space');
  await page.getByRole('button', { name: 'hour_cos', exact: true }).click();
  await expect(sine).toHaveAttribute('aria-pressed', 'true');
  let submitted: { nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'date-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.length).toBe(2);
  expect(submitted!.nodes.find(node => node.node_id === id)).toMatchObject({ step_type: 'DateFeatures', params: { timezone: 'Europe/Istanbul', features: ['second', 'hour_sin', 'hour_cos'] } });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(timezone).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(timezone).toHaveValue('Europe/Istanbul');
  await expect(sine).toHaveAttribute('aria-pressed', 'true');
});
