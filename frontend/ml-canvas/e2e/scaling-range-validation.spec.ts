import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('a real scaling setting changes once in history while opening and closing does not', async ({ page }) => {
  // Undo/redo must retain real configuration edits without counting panel visibility.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const store = window.__skyulfTest!.graphStore;
    const id = store.getState().addNode('scale_numeric_features', { x: 0, y: 0 }, { method: 'standard', columns: ['x'] });
    store.temporal.getState().clear();
    return id;
  });
  await page.getByRole('checkbox', { name: 'Center Data (with_mean)' }).uncheck();
  await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  expect(await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().pastStates.length)).toBe(1);
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+z');
  expect(await page.evaluate(id => window.__skyulfTest!.graphStore.getState().nodes.find(node => node.id === id)!.data.with_mean, id)).toBeUndefined();
  await page.keyboard.press('Control+Shift+z');
  await page.locator(`.react-flow__node[data-id="${id}"]`).click();
  await expect(page.getByRole('checkbox', { name: 'Center Data (with_mean)' })).not.toBeChecked();
  expect(await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().pastStates.length)).toBe(1);
});

for (const method of ['minmax', 'robust'] as const) {
  test(`cleared ${method} bounds appear in Issues and recover after editing`, async ({ page }) => {
    // Real input edits must block validation and navigate back to the failing control.
    await mockBackend(page);
    await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto('/canvas');
    await page.waitForFunction(() => '__skyulfTest' in window);
    const id = await page.evaluate(method => window.__skyulfTest!.graphStore.getState().addNode(
      'scale_numeric_features', { x: 0, y: 0 }, { method, columns: ['x'] },
    ), method);
    const label = method === 'minmax' ? 'Feature Range Minimum' : 'Quantile Range Minimum';
    const input = page.getByRole('spinbutton', { name: label });
    await input.fill('');
    const issue = page.getByRole('button', { name: /Configuration Scaling.*requires two finite numbers/i });
    await expect(issue).toBeVisible();
    await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
    await issue.click();
    await expect(input).toBeFocused();
    await expect(input).toHaveValue('');
    await expect(input).toHaveAccessibleDescription(/requires two finite numbers/);
    expect(await page.evaluate(id => {
      const node = window.__skyulfTest!.graphStore.getState().nodes.find(node => node.id === id)!;
      return JSON.parse(JSON.stringify(node.data));
    }, id)).toMatchObject({ [method === 'minmax' ? 'feature_range_min' : 'quantile_range_min']: null });
    await input.fill(method === 'minmax' ? '-1' : '10');
    await expect(issue).toHaveCount(0);
    await expect(input).not.toHaveAttribute('aria-invalid');
  });
}
