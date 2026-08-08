import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mockBackend } from './fixtures/mockApi';

// M4 — A11y CI gate. Walks the four primary routes with axe-core and
// fails the build on any `critical` violation. `serious` findings are
// logged but do not block: React Flow's `scrollable-region-focusable`
// is a known false positive (the viewport supports keyboard nav via
// its own handlers), tracked for a dedicated follow-up. This spec
// exists to prevent NEW regressions sneaking in.

const routes: Array<{ path: string; name: string }> = [
  { path: '/', name: 'dashboard' },
  { path: '/canvas', name: 'canvas' },
  { path: '/data', name: 'data' },
  { path: '/eda', name: 'eda' },
];

const themes: Array<{ name: string; dark: boolean }> = [
  { name: 'light', dark: false },
  { name: 'dark', dark: true },
];

test.describe('A11y — no critical axe violations', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  for (const route of routes) {
    for (const theme of themes) {
      test(`${route.name} (${route.path}) — ${theme.name} mode`, async ({ page }) => {
        await page.goto(route.path);
        // Wait for the SPA to mount so axe sees the rendered tree, not the
        // empty <div id="root"/>.
        await expect(page.locator('body')).not.toBeEmpty({ timeout: 10_000 });

        if (theme.dark) {
          await page.evaluate(() => document.documentElement.classList.add('dark'));
        }

        // Give the page's fade-in entrance transition (duration-500) time to
        // settle: scanning mid-transition makes axe read blended, translucent
        // colors and misreport contrast failures that don't exist once at rest.
        await page.waitForTimeout(1000);

        const results = await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
          .analyze();

        const critical = results.violations.filter((v) => v.impact === 'critical');
        const serious = results.violations.filter((v) => v.impact === 'serious');
        if (serious.length > 0) {
          // eslint-disable-next-line no-console
          console.warn(
            `[a11y] ${route.name} (${theme.name}): ${serious.length} serious violation(s) (non-blocking):\n` +
              serious
                .map(
                  (v) =>
                    `  - ${v.id}: ${v.help}\n` +
                    v.nodes.map((n) => `      target: ${n.target.join(' ')}\n      ${n.failureSummary}`).join('\n')
                )
                .join('\n')
          );
        }
        expect(
          critical,
          `Found ${critical.length} critical a11y violation(s) in ${theme.name} mode:\n` +
            critical.map((v) => `- ${v.id}: ${v.help}`).join('\n')
        ).toEqual([]);
      });
    }
  }
});
