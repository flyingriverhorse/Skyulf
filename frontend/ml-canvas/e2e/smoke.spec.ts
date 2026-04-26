import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/**
 * Smoke: canvas boots, sidebar mounts, clicking a sidebar component
 * adds a node to the React Flow viewport.
 *
 * This proves: routing, lazy-loaded canvas mount, registry init,
 * sidebar render, zustand mutation, React Flow render. Catches the
 * "render-blanks-the-canvas" class of bug Vitest cannot.
 */
test.describe('Canvas smoke', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  test('boots and renders the React Flow viewport', async ({ page }) => {
    await page.goto('/canvas');

    // React Flow renders this wrapper div on mount.
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });

    // Sidebar with categories.
    await expect(page.getByRole('heading', { name: 'Components' })).toBeVisible();
  });

  test('clicking sidebar items adds nodes to the canvas', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });

    // Sidebar items are draggable divs but ALSO have an onClick that
    // calls addNode — we exploit that to avoid HTML5 drag synthesis,
    // which is unreliable in Playwright + React Flow.
    const datasetItem = page.locator('aside [draggable="true"]').filter({ hasText: 'Dataset' }).first();
    await expect(datasetItem).toBeVisible();
    await datasetItem.click();

    const dropColsItem = page.locator('aside [draggable="true"]').filter({ hasText: 'Drop Columns' }).first();
    await expect(dropColsItem).toBeVisible();
    await dropColsItem.click();

    // Two React Flow nodes should be on the canvas.
    await expect(page.locator('.react-flow__node')).toHaveCount(2);
  });
});
