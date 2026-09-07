import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('automatic sidebar collapse preserves browsing state and keyboard focus', async ({ page }) => {
  // A narrower window must reclaim canvas space without losing the current search or keyboard position.
  await mockBackend(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  const search = page.getByRole('textbox', { name: 'Search nodes' });
  const preprocessing = page.getByRole('button', { name: 'Preprocessing', exact: true });
  const reopen = page.getByRole('button', { name: 'Expand components sidebar', exact: true });
  await preprocessing.click();
  await search.fill('missing values');
  await expect(search).toBeFocused();
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(search).toBeHidden();
  await expect(reopen).toBeFocused();
  await expect(reopen).not.toHaveCSS('box-shadow', 'none');
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(search).toBeFocused();
  await expect(search).toHaveValue('missing values');
  await search.fill('');
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'false');
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(reopen).toBeVisible();
  await reopen.press('Enter');
  await expect(search).toBeFocused();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(search).toBeVisible();
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'false');
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).press('Space');
  await expect(reopen).toBeFocused();
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(search).toBeHidden();
  await expect(reopen).toBeVisible();
});

test('a laptop starts with more canvas space and keeps settings aligned when the sidebar reopens', async ({ page }) => {
  // Every consumer must use the effective sidebar width so settings and toolbar never overlap the reopen control.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1100, height: 800 });
  await page.goto('/canvas');
  const reopen = page.getByRole('button', { name: 'Expand components sidebar', exact: true });
  await expect(reopen).toBeVisible();
  await expect(page.getByRole('textbox', { name: 'Search nodes' })).toBeHidden();
  const reopenBounds = await reopen.boundingBox();
  const firstTool = await page.getByRole('button', { name: 'Show node badge legend', exact: true }).boundingBox();
  expect(reopenBounds && firstTool && firstTool.x >= reopenBounds.x + reopenBounds.width).toBeTruthy();
  await reopen.click();
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  const target = page.getByRole('textbox', { name: 'Target Column', exact: true });
  await target.fill('outcome');
  const resize = page.getByRole('separator', { name: 'Resize settings panel' });
  await resize.press('End');
  await expect(resize).toHaveAttribute('aria-valuenow', '380');
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).click();
  await resize.press('End');
  await expect(resize).toHaveAttribute('aria-valuenow', '636');
  await reopen.click();
  await expect(resize).toHaveAttribute('aria-valuenow', '380');
  await expect(target).toHaveValue('outcome');
});

test('automatic collapse leaves an edited setting focused and read-only transitions preserve the layout choice', async ({ page }) => {
  // A responsive layout change must not interrupt a field edit or override an explicit library choice.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'dark'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  const target = page.getByRole('textbox', { name: 'Target Column', exact: true });
  const reopen = page.getByRole('button', { name: 'Expand components sidebar', exact: true });
  await target.fill('outcome');
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(reopen).toBeVisible();
  await expect(target).toBeFocused();
  await expect(target).toHaveValue('outcome');
  await page.screenshot({ path: 'test-results/sidebar-auto-dark.png' });
  await reopen.click();
  await page.setViewportSize({ width: 900, height: 800 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('textbox', { name: 'Search nodes' })).toBeHidden();
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(page.getByRole('textbox', { name: 'Search nodes' })).toBeVisible();
  await expect(target).toHaveValue('outcome');
});
