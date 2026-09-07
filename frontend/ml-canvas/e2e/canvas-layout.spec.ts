import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Report intersecting visible controls without counting their decorative children. */
async function overlappingControls(page: Page, selector: string) {
  return page.locator(selector).evaluateAll(elements => {
    const controls = elements.filter(element => element.getClientRects().length > 0)
      .map(element => ({ name: element.getAttribute('aria-label') || element.textContent?.trim(), rect: element.getBoundingClientRect() }));
    return controls.flatMap((a, index) => controls.slice(index + 1).flatMap(b =>
      Math.min(a.rect.right, b.rect.right) - Math.max(a.rect.left, b.rect.left) > 1
      && Math.min(a.rect.bottom, b.rect.bottom) - Math.max(a.rect.top, b.rect.top) > 1
        ? [`${a.name} / ${b.name}`] : []));
  });
}

/** Seed a populated canvas with long dataset context and parallel model actions. */
async function openPopulatedCanvas(page: Page) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.goto('/canvas');
  await page.waitForFunction(() => Boolean((window as unknown as { __skyulfTest?: unknown }).__skyulfTest));
  await page.evaluate(() => {
    const hook = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest;
    const setGraph = hook.graphStore.getState().setGraph as (nodes: unknown[], edges: unknown[]) => void;
    setGraph([
      { id: 'source', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'dataset_node', datasetName: 'Quarterly customer retention training dataset with a very long name' } },
      ...['first', 'second'].map((id, index) => ({ id, type: 'custom', selected: index === 0, position: { x: 300, y: index * 200 }, data: { definitionType: 'classification', label: 'Classification', model_type: 'random_forest_classifier', target_column: 'outcome', hyperparameters: {}, search_space: {}, run_mode: 'basic', cv_enabled: false } })),
    ], ['first', 'second'].map(id => ({ id: `edge-${id}`, source: 'source', target: id, sourceHandle: 'data', targetHandle: 'in' })));
  });
}

test('canvas actions do not overlap at laptop widths with both side panels open', async ({ page }) => {
  // Pinning both panels must never make preview and editing controls cover each other.
  await page.setViewportSize({ width: 1440, height: 900 });
  await openPopulatedCanvas(page);
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
  const names = ['Show node badge legend', 'Keyboard shortcuts', 'Open command palette', 'Undo', 'Redo', 'Clear canvas', 'More canvas tools', 'Load pipeline', 'Save pipeline', 'Run all parallel branches as separate experiments', 'Job runs history', 'Start from a template', 'Toggle performance overlay', 'Tidy: auto-arrange nodes', 'Export canvas as image', 'Recent pipelines (local fallback)'];
  const selector = names.map(name => `button[aria-label="${name}"]`).concat('[data-testid="toolbar-run-preview"]').join(',');
  for (const width of [1920, 1600, 1440, 1280, 1100, 1024]) {
    await page.setViewportSize({ width, height: 800 });
    await expect.poll(() => overlappingControls(page, selector)).toEqual([]);
    await expect(page.getByRole('button', { name: 'Preview data', exact: true })).toBeInViewport();
    await expect.poll(() => overlappingControls(page, '[aria-label="Show node on canvas"], [aria-label="Expand settings panel"], [aria-label="Close settings panel"]')).toEqual([]);
  }
  await page.screenshot({ path: 'test-results/canvas-layout-laptop.png' });
  await expect(page.getByRole('textbox', { name: 'Target Column', exact: true })).toHaveValue('outcome');
});

test('navigation and results controls fit tablet and phone widths with long dataset names', async ({ page }) => {
  // Dataset context, view tabs, and utility actions must share the header without collisions.
  await page.setViewportSize({ width: 900, height: 800 });
  await openPopulatedCanvas(page);
  for (const width of [900, 768, 640, 390, 320]) {
    await page.setViewportSize({ width, height: 800 });
    await expect.poll(() => overlappingControls(page, '[data-testid="navbar-views"] [role="tab"], [aria-label="Breadcrumb"], [data-testid="navbar-help"], button[title^="Read-only canvas"], [aria-label^="Notifications"]')).toEqual([]);
    await expect.poll(() => overlappingControls(page, '[aria-label="Toggle preview results"], [aria-label="Maximize results panel"], [aria-label="Collapse results panel"], [aria-label="Close preview results"]')).toEqual([]);
    await expect(page.getByRole('button', { name: 'Close preview results', exact: true })).toBeInViewport();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await expect.poll(async () => {
    const sidebar = await page.locator('#app-sidebar').boundingBox();
    return sidebar ? sidebar.x + sidebar.width : 0;
  }).toBeLessThanOrEqual(0);
  await page.screenshot({ path: 'test-results/canvas-layout-phone.png' });
  await expect(page.getByRole('button', { name: 'Pipeline guide', exact: true })).toBeInViewport();
});

test('compact overflow actions and menus remain reachable above results', async ({ page }) => {
  // Moving actions into overflow must retain their handlers and keep the entire menu above results.
  await page.setViewportSize({ width: 1440, height: 800 });
  await openPopulatedCanvas(page);
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
  await page.setViewportSize({ width: 1100, height: 800 });
  const more = page.getByRole('button', { name: 'More canvas tools', exact: true });
  await expect(page.getByRole('button', { name: 'Save pipeline', exact: true })).toBeHidden();
  await more.press('Enter');
  const menu = page.getByRole('menu', { name: 'More canvas tools', exact: true });
  await page.keyboard.press('Tab');
  await expect(menu.getByRole('menuitem', { name: 'Node badge legend', exact: true })).toBeFocused();
  await expect(menu.getByRole('menuitem', { name: 'Save pipeline', exact: true })).toBeVisible();
  await expect(menu.getByRole('menuitem', { name: 'Run all experiments', exact: true })).toBeVisible();
  await menu.getByRole('menuitem', { name: 'Export SVG', exact: true }).click({ trial: true });
  await menu.getByRole('menuitem', { name: 'Load pipeline', exact: true }).press('Enter');
  const load = page.getByRole('menu', { name: 'Load pipeline version', exact: true });
  await expect(load).toContainText('No versions yet');
  await expect(load).toBeFocused();
  const box = await load.boundingBox();
  expect(box && box.x >= 0 && box.x + box.width <= 1100).toBeTruthy();
  await page.keyboard.press('Escape');
  await expect(more).toBeFocused();
});

test('legend and notification menus stay within a dark phone viewport', async ({ page }) => {
  // Small screens must expose complete popovers with scrolling instead of clipping controls beyond the screen.
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'dark'));
  await page.setViewportSize({ width: 390, height: 700 });
  await openPopulatedCanvas(page);
  await page.getByRole('button', { name: 'More canvas tools', exact: true }).press('Enter');
  await page.getByRole('menuitem', { name: 'Node badge legend', exact: true }).press('Enter');
  await expect(page.getByRole('button', { name: 'Close legend', exact: true })).toBeFocused();
  const legend = page.getByRole('heading', { name: 'Canvas Legend', exact: true }).locator('../..');
  const legendBox = await legend.boundingBox();
  expect(legendBox && legendBox.x >= 0 && legendBox.x + legendBox.width <= 390 && legendBox.y + legendBox.height <= 700).toBeTruthy();
  await page.getByRole('button', { name: 'Close legend', exact: true }).click();
  await expect(page.getByRole('button', { name: 'More canvas tools', exact: true })).toBeFocused();
  const notifications = page.getByRole('button', { name: 'Notifications', exact: true });
  await notifications.click();
  const notificationPanel = notifications.locator('..').locator(':scope > div');
  await expect(notificationPanel).toContainText('No notifications');
  const notificationBox = await notificationPanel.boundingBox();
  expect(notificationBox && notificationBox.x >= 0 && notificationBox.x + notificationBox.width <= 390).toBeTruthy();
  await page.screenshot({ path: 'test-results/canvas-layout-phone-dark.png' });
  await expect(page.locator('html')).toHaveClass(/dark/);
});
