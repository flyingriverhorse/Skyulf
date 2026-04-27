import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mockBackend } from './fixtures/mockApi';

// M4 — A11y CI gate. Walks the four primary routes with axe-core and
// fails the build on any `critical` violation. `serious` findings are
// logged but do not block: the dashboard currently surfaces a slate
// color-contrast issue and React Flow's `scrollable-region-focusable`
// is a known false positive (the viewport supports keyboard nav via
// its own handlers). Both are tracked for a dedicated follow-up; this
// spec exists to prevent NEW regressions sneaking in.

const routes: Array<{ path: string; name: string }> = [
  { path: '/', name: 'dashboard' },
  { path: '/canvas', name: 'canvas' },
  { path: '/data', name: 'data' },
  { path: '/eda', name: 'eda' },
];

test.describe('A11y — no critical axe violations', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  for (const route of routes) {
    test(`${route.name} (${route.path})`, async ({ page }) => {
      await page.goto(route.path);
      // Wait for the SPA to mount so axe sees the rendered tree, not the
      // empty <div id="root"/>.
      await expect(page.locator('body')).not.toBeEmpty({ timeout: 10_000 });

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze();

      const critical = results.violations.filter((v) => v.impact === 'critical');
      const serious = results.violations.filter((v) => v.impact === 'serious');
      if (serious.length > 0) {
        // eslint-disable-next-line no-console
        console.warn(
          `[a11y] ${route.name}: ${serious.length} serious violation(s) (non-blocking):\n` +
            serious.map((v) => `  - ${v.id}: ${v.help}`).join('\n')
        );
      }
      expect(
        critical,
        `Found ${critical.length} critical a11y violation(s):\n` +
          critical.map((v) => `- ${v.id}: ${v.help}`).join('\n')
      ).toEqual([]);
    });
  }
});
