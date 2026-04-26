import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/**
 * Routes smoke: every top-level route mounts without triggering an
 * <ErrorBoundary>. Cheapest possible regression net for "did we break
 * the lazy import / route element of page X".
 */
const routes: Array<{ path: string; expect: RegExp | string }> = [
  { path: '/', expect: /dashboard/i },
  { path: '/canvas', expect: 'Components' },
  { path: '/jobs', expect: /jobs/i },
  { path: '/data', expect: /data/i },
  { path: '/eda', expect: /eda|exploratory/i },
];

test.describe('Routes mount cleanly', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  for (const route of routes) {
    test(`route ${route.path}`, async ({ page }) => {
      await page.goto(route.path);
      // The shared ErrorBoundary renders a "Something went wrong" UI.
      await expect(page.getByText(/something went wrong/i)).toHaveCount(0);
      // Some content from the page itself is visible.
      await expect(page.locator('body')).toContainText(route.expect, { timeout: 10_000 });
    });
  }
});
