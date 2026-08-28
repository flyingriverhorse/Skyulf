import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/**
 * Visual verification of the per-node random-state controls:
 * Tuning Strategy "Random State", CV "Fold Split Seed" (training +
 * ensemble nodes), and the basic-mode hyperparameter seed field.
 *
 * The hyperparameters endpoint is stubbed with a tunable param plus a
 * `tunable: false` random_state, mirroring the core definitions.
 */

const HYPERPARAMETER_DEFS = [
  {
    name: 'n_estimators',
    label: 'Number of Trees',
    type: 'number',
    default: 100,
    min: 10,
    max: 1000,
    tunable: true,
  },
  {
    name: 'random_state',
    label: 'Random State',
    type: 'number',
    default: 42,
    min: 0,
    max: 10000,
    tunable: false,
  },
];

const addNode = async (page: import('@playwright/test').Page, label: string) => {
  // Sidebar click adds the node; it arrives pre-selected, which opens the
  // properties panel — no need to click the node (the bottom drawer would
  // intercept pointer events anyway).
  await page.locator('aside [draggable="true"]').filter({ hasText: label }).first().click();
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
};

test.describe('Seed controls', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
    // Registered after the catch-all so these take precedence. Playwright
    // matches routes in reverse registration order, so the more specific
    // defaults route goes last to win over the generic one.
    await page.route('**/api/pipeline/hyperparameters/*', (route) =>
      route.fulfill({ json: HYPERPARAMETER_DEFS })
    );
    await page.route('**/api/pipeline/hyperparameters/*/defaults*', (route) =>
      route.fulfill({ json: { n_estimators: [50, 100, 200] } })
    );
  });

  test('classification: advanced mode shows tuner seed, fold seed, and keeps seed out of the search space', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Classification');

    // Switch to advanced mode (segmented button control).
    await page.getByRole('button', { name: 'Advanced (Tuning)' }).click();

    // Tuner seed lives in the Tuning Strategy grid.
    await expect(page.getByText('Tuning Strategy')).toBeVisible();
    await expect(page.getByText('Random State', { exact: true }).first()).toBeVisible();

    // Search space renders the tunable param only — the seed is filtered out.
    // (Advanced mode renames the Hyperparameters accordion to Search Space.
    // The accordion is single-open: expanding it collapses Configuration and
    // its tuner seed, so any remaining Random State text would have to come
    // from the search space itself.)
    await page.getByRole('button', { name: 'Search Space' }).click();
    await expect(page.getByText('Number of Trees').first()).toBeVisible();
    await expect(page.getByText('Random State', { exact: true })).toHaveCount(0);

    // CV fold seed is visible with the default shuffle-on config. (The CV
    // section nests inside Configuration, which Search Space just closed.)
    await page.getByRole('button', { name: 'Configuration', exact: true }).click();
    await page.getByRole('button', { name: 'Cross Validation' }).click();
    await expect(page.getByText('Fold Split Seed')).toBeVisible();

    await page.screenshot({ path: 'test-results/seed-classification-advanced.png', fullPage: true });
  });

  test('classification: basic mode exposes the seed via Customize hyperparameters', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Classification');

    await page.getByRole('button', { name: 'Hyperparameters' }).click();
    await page.getByText('Customize', { exact: true }).first().click();

    await expect(page.getByText('Random State', { exact: true }).first()).toBeVisible();

    await page.screenshot({ path: 'test-results/seed-classification-basic.png', fullPage: true });
  });

  test('ensemble: tuning seed and fold seed controls render', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Ensemble');

    // Tuning controls only render in advanced run mode (segmented toggle).
    await page.getByRole('button', { name: 'Advanced (Tuning)' }).click();

    await expect(page.getByText('Hyperparameter Tuning')).toBeVisible();
    // The label wraps a HelpTooltip child, so exact matching fails here.
    await expect(page.getByText('Random State').first()).toBeVisible();

    await page.getByRole('button', { name: 'Cross Validation' }).click();
    await expect(page.getByText('Fold Split Seed')).toBeVisible();

    await page.screenshot({ path: 'test-results/seed-ensemble.png', fullPage: true });
  });
});
