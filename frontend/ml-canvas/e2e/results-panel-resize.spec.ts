import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';
import AxeBuilder from '@axe-core/playwright';

/** Supply measured rows through the same store action used by preview requests. */
async function showResults(page: Page) {
  await page.evaluate(() => {
    const hook = (window as unknown as {
      __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } };
    }).__skyulfTest;
    const setResult = hook.graphStore.getState().setExecutionResult as (result: unknown) => void;
    setResult({
      pipeline_id: 'resize-preview', status: 'success', node_results: {},
      preview_data: { Result: [{ outcome: 'sample-result' }] }, recommendations: [],
    });
  });
}

test('results resizing preserves settings and rows across panel and viewport changes', async ({ page }) => {
  // Reclaiming canvas space must not discard edits, preview data, or the user's preferred height.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  const target = page.getByRole('textbox', { name: 'Target Column', exact: true });
  await target.fill('outcome');
  await showResults(page);
  await page.getByRole('tab', { name: 'Data', exact: true }).click();

  const handle = page.getByRole('separator', { name: 'Resize results panel' });
  const panel = page.getByRole('region', { name: 'Preview results', exact: true });
  const controls = page.locator('.react-flow__controls');
  await expect(handle).toHaveAttribute('aria-valuenow', '384');
  const start = await handle.boundingBox();
  if (!start) throw new Error('Resize handle has no bounds');
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2);
  await page.mouse.down();
  await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2 - 60, { steps: 5 });
  await page.mouse.up();
  await expect(handle).toHaveAttribute('aria-valuenow', '444');
  await expect(handle).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(handle).toHaveAttribute('aria-valuenow', '424');
  await page.keyboard.press('ArrowUp');
  await expect(handle).toHaveAttribute('aria-valuenow', '444');
  await expect(target).toHaveValue('outcome');
  await expect(panel.getByText('sample-result')).toBeVisible();

  await page.getByRole('button', { name: 'Maximize results panel', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(handle).toHaveCount(0);
  await expect(controls).toBeHidden();
  await page.getByRole('button', { name: 'Collapse results panel', exact: true }).press('Enter');
  await expect(controls).toBeVisible();
  await page.getByRole('button', { name: 'Expand results panel', exact: true }).press('Space');
  await expect(controls).toBeHidden();
  await page.getByRole('button', { name: 'Restore results panel', exact: true }).press('Space');
  await expect(handle).toHaveAttribute('aria-valuenow', '444');
  await page.getByRole('button', { name: 'Collapse results panel', exact: true }).press('Enter');
  await expect(handle).toHaveCount(0);
  await page.getByRole('button', { name: 'Expand results panel', exact: true }).press('Space');
  await expect(handle).toHaveAttribute('aria-valuenow', '444');

  await page.setViewportSize({ width: 1100, height: 700 });
  await expect.poll(async () => Number(await handle.getAttribute('aria-valuenow'))).toBeLessThan(444);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(handle).toHaveAttribute('aria-valuenow', '444');
  await page.setViewportSize({ width: 1100, height: 600 });
  await expect.poll(async () => Number(await handle.getAttribute('aria-valuemax'))).toBeLessThan(384);
  await handle.press('End');
  await expect.poll(async () => {
    const canvas = await page.locator('.react-flow').boundingBox();
    const results = await panel.boundingBox();
    return canvas && results ? results.y - canvas.y : 0;
  }).toBeGreaterThanOrEqual(239);
  await expect.poll(async () => {
    const buttons = await controls.boundingBox();
    const results = await panel.boundingBox();
    return buttons && results ? results.y - buttons.y - buttons.height : -1;
  }).toBeGreaterThanOrEqual(0);
  await handle.press('Home');
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(handle).toHaveAttribute('aria-valuenow', '384');
  await handle.press('ArrowDown');
  await page.getByRole('button', { name: 'Close preview results', exact: true }).click();
  await expect(panel).toHaveCount(0);
  await showResults(page);
  await expect(handle).toHaveAttribute('aria-valuenow', '364');
  await expect(target).toHaveValue('outcome');
  await page.screenshot({ path: 'test-results/results-resize-light-editing.png' });
  await expect(panel.getByText('sample-result')).toBeVisible();
});

test('results remain resizable in dark read-only mode with reduced motion and error content', async ({ page }) => {
  // Inspection layout remains adjustable when editing is disabled, including a failed preview.
  await mockBackend(page);
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'dark'));
  await page.setViewportSize({ width: 900, height: 800 });
  await page.goto('/canvas');
  await page.waitForFunction(() => Boolean((window as unknown as { __skyulfTest?: unknown }).__skyulfTest));
  await page.evaluate(() => {
    const hook = (window as unknown as {
      __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } };
    }).__skyulfTest;
    const setError = hook.graphStore.getState().setLastRunError as (error: string) => void;
    setError('Preview could not load the selected dataset. Select another dataset and try again.');
  });
  const handle = page.getByRole('separator', { name: 'Resize results panel' });
  const panel = page.getByRole('region', { name: 'Preview results', exact: true });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('html')).toHaveClass(/dark/);
  await handle.focus();
  await expect(handle).toBeFocused();
  await expect(handle).not.toHaveCSS('box-shadow', 'none');
  await handle.press('ArrowDown');
  await expect(handle).toHaveAttribute('aria-valuenow', '364');
  await expect(panel.getByRole('alert')).toContainText('Select another dataset');
  const accessibility = await new AxeBuilder({ page }).include('[aria-label="Preview results"]').analyze();
  expect(accessibility.violations).toEqual([]);
  await page.screenshot({ path: 'test-results/results-resize-dark-readonly.png' });
  await handle.press('ArrowDown');
  await expect(handle).toHaveAttribute('aria-valuenow', '344');
  for (let step = 0; step < 10; step++) await handle.press('ArrowDown');
  await expect(handle).toHaveAttribute('aria-valuenow', '200');
});
