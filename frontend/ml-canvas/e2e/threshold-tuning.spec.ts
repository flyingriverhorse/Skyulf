import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/**
 * Visual verification of the F-13 "Tune decision threshold" checkbox:
 * shown for classification nodes in Advanced (tuning) mode, hidden for
 * regression, off by default, and toggleable.
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
];

const addNode = async (page: import('@playwright/test').Page, label: string) => {
  await page.locator('aside [draggable="true"]').filter({ hasText: label }).first().click();
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
};

test.describe('Decision threshold tuning control', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
    // Registered after the catch-all so these take precedence (Playwright
    // matches routes in reverse registration order).
    await page.route('**/api/pipeline/hyperparameters/*', (route) =>
      route.fulfill({ json: HYPERPARAMETER_DEFS })
    );
    await page.route('**/api/pipeline/hyperparameters/*/defaults*', (route) =>
      route.fulfill({ json: { n_estimators: [50, 100, 200] } })
    );
  });

  test('classification: advanced mode shows the checkbox, off by default, toggleable', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Classification');

    await page.getByRole('button', { name: 'Advanced (Tuning)' }).click();
    await expect(page.getByText('Tuning Strategy')).toBeVisible();

    const checkbox = page.locator('#tune_threshold');
    await expect(checkbox).toBeVisible();
    await expect(checkbox).not.toBeChecked();

    await page.getByText('Tune decision threshold').click();
    await expect(checkbox).toBeChecked();

    await page.screenshot({ path: 'test-results/threshold-tuning-classification.png', fullPage: true });
  });

  test('regression: advanced mode does not show the checkbox', async ({ page }) => {
    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });
    await addNode(page, 'Regression');

    await page.getByRole('button', { name: 'Advanced (Tuning)' }).click();
    await expect(page.getByText('Tuning Strategy')).toBeVisible();

    await expect(page.locator('#tune_threshold')).toHaveCount(0);
  });
});
