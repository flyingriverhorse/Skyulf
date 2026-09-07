import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Check the node against usable canvas bounds, including floating toolbar and results. */
async function expectNodeUnobscured(page: Page) {
  await expect.poll(async () => {
    const node = await page.locator('.react-flow__node.selected').boundingBox();
    const canvas = await page.locator('.react-flow').boundingBox();
    const results = await page.getByRole('region', { name: 'Preview results', exact: true }).boundingBox();
    if (!node || !canvas) return false;
    return node.x >= canvas.x + 60 && node.x + node.width <= canvas.x + canvas.width - 15
      && node.y >= canvas.y + 60 && node.y + node.height <= (results?.y ?? canvas.y + canvas.height) - 10;
  }).toBe(true);
}

test('show node restores panels and reveals the selection without discarding settings or preview rows', async ({ page }) => {
  // Finding a node must use the unobscured canvas after maximized panels restore their docked size.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  const target = page.getByRole('textbox', { name: 'Target Column', exact: true });
  await target.fill('outcome');
  await page.evaluate(() => {
    const hook = (window as unknown as {
      __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } };
    }).__skyulfTest;
    const state = hook.graphStore.getState();
    (state.setExecutionResult as (result: unknown) => void)({
      pipeline_id: 'reveal-preview', status: 'success', node_results: {},
      preview_data: { Result: [{ outcome: 'retained-preview-row' }] }, recommendations: [],
    });
    const nodes = state.nodes as { id: string }[];
    (state.onNodesChange as (changes: unknown[]) => void)([
      { type: 'position', id: nodes[0].id, position: { x: 3000, y: 2000 } },
    ]);
  });
  await page.getByRole('tab', { name: 'Data', exact: true }).click();
  await page.getByRole('button', { name: 'Maximize results panel', exact: true }).click();
  await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
  const reveal = page.getByRole('button', { name: 'Show node on canvas', exact: true });
  await reveal.press('Enter');
  await expect(page.getByRole('button', { name: 'Expand settings panel', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Maximize results panel', exact: true })).toBeVisible();
  await expectNodeUnobscured(page);
  await expect(reveal).toBeFocused();
  await expect(target).toHaveValue('outcome');
  await expect(page.getByText('retained-preview-row')).toBeVisible();

  await page.setViewportSize({ width: 1100, height: 800 });
  await page.getByRole('separator', { name: 'Resize results panel' }).press('End');
  await reveal.press('Space');
  await expectNodeUnobscured(page);
  await expect(target).toHaveValue('outcome');
  await page.screenshot({ path: 'test-results/reveal-node-laptop.png' });
  await expect(page.locator('.react-flow__node.selected')).toHaveCount(1);
});

test('sidebar additions use the visible canvas above results in dark reduced-motion mode', async ({ page }) => {
  // Existing reveal requests must account for the results overlay and preserve focus at their origin.
  await mockBackend(page);
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'dark'));
  await page.setViewportSize({ width: 1440, height: 800 });
  await page.goto('/canvas');
  const add = page.getByRole('button', { name: 'Add Missing Indicator node', exact: true });
  await add.press('Enter');
  const results = page.getByRole('separator', { name: 'Resize results panel' });
  await expect(results).toBeVisible();
  await results.press('End');
  await add.press('Enter');
  await expect(page.locator('.react-flow__node')).toHaveCount(2);
  await expectNodeUnobscured(page);
  await expect(add).toBeFocused();
});

test('read-only reveal moves canvas focus and later resizing preserves a manual pan', async ({ page }) => {
  // Deep-link reveal must remain usable in read-only mode without locking the viewport to a node afterward.
  await mockBackend(page);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Missing Indicator node', exact: true }).click();
  await page.setViewportSize({ width: 900, height: 800 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.evaluate(() => {
    const hook = (window as unknown as {
      __skyulfTest: { graphStore: { getState: () => { nodes: { id: string }[] } } };
    }).__skyulfTest;
    window.dispatchEvent(new CustomEvent('skyulf:focus-node', {
      detail: { id: hook.graphStore.getState().nodes[0].id, focusWrapper: true },
    }));
  });
  await expectNodeUnobscured(page);
  await expect(page.locator('.react-flow').locator('..')).toBeFocused();
  const viewport = page.locator('.react-flow__viewport');
  const beforePan = await viewport.getAttribute('style');
  const canvas = await page.locator('.react-flow').boundingBox();
  if (!canvas) throw new Error('Canvas has no bounds');
  await page.mouse.move(canvas.x + canvas.width - 50, canvas.y + 100);
  await page.mouse.down();
  await page.mouse.move(canvas.x + canvas.width - 200, canvas.y + 120, { steps: 5 });
  await page.mouse.up();
  await expect(viewport).not.toHaveAttribute('style', beforePan ?? '');
  const afterPan = await viewport.getAttribute('style');
  await page.setViewportSize({ width: 1000, height: 800 });
  await expect.poll(async () => (await page.locator('.react-flow').boundingBox())?.width).toBe(936);
  await expect(viewport).toHaveAttribute('style', afterPan ?? '');
});
