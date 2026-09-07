import { test, expect, type Locator, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Confirm all description lines fit instead of merely checking text hidden by CSS. */
async function expectReadable(element: Locator) {
  await expect(element).toBeVisible();
  expect(await element.evaluate(node => node.scrollHeight <= node.clientHeight + 1 && node.scrollWidth <= node.clientWidth + 1)).toBe(true);
}

/** Open an editable canvas with the component library available at either desktop size. */
async function openCanvas(page: Page, width: number) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width, height: 900 });
  await page.goto('/canvas');
  if (width < 1280) await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
}

for (const theme of ['light', 'dark'] as const) {
  test(`task search keeps descriptions readable and keyboard insertion working on a ${theme} canvas`, async ({ page }) => {
    // Both entry points must expose the same relevant nodes at laptop and desktop widths.
    await page.addInitScript(value => localStorage.setItem('skyulf-theme', value), theme);
    await openCanvas(page, theme === 'light' ? 1440 : 1100);
    const sidebar = page.getByRole('complementary', { name: 'Components', exact: true });
    const search = sidebar.getByRole('textbox', { name: 'Search nodes', exact: true });
    await search.fill('  NORMALIZE  ');
    await expect(sidebar.getByRole('button', { name: /^Add .* node$/ })).toHaveCount(1);
    const scaling = sidebar.getByRole('button', { name: 'Add Scaling node', exact: true });
    await expectReadable(scaling.getByText('Scale numeric features to a standard range.', { exact: true }));
    await expect(scaling).toHaveAccessibleDescription('Scale numeric features to a standard range.');
    await scaling.press('Enter');
    await expect(page.locator('.react-flow__node')).toHaveCount(1);
    await search.fill('predict');
    const classification = sidebar.getByRole('button', { name: 'Add Classification node', exact: true });
    await expectReadable(classification.getByText('Train a classifier to predict a categorical target — fixed parameters or automatic tuning.', { exact: true }));
    await page.screenshot({ path: `test-results/node-discovery-${theme}.png` });

    await page.locator('.react-flow').locator('..').focus();
    await page.keyboard.press('Control+k');
    const palette = page.getByRole('dialog', { name: 'Command palette', exact: true });
    const paletteSearch = palette.getByRole('textbox', { name: 'Search nodes', exact: true });
    await paletteSearch.fill('predict');
    await expect(palette.getByRole('option')).toHaveCount(await sidebar.getByRole('button', { name: /^Add .* node$/ }).count());
    await expectReadable(palette.getByText('Train a classifier to predict a categorical target — fixed parameters or automatic tuning.', { exact: true }));
    await paletteSearch.fill('normalize');
    await expect(palette.getByRole('option')).toHaveCount(1);
    await paletteSearch.press('Enter');
    await expect(palette).toHaveCount(0);
    await expect(page.locator('.react-flow__node')).toHaveCount(2);
    await expect(page.getByTestId('canvas-node-scale_numeric_features')).toHaveCount(2);
  });
}

test('task search retains click, drag, empty results, and restored categories', async ({ page }) => {
  // Search must preserve both mouse insertion routes and the user's category choices.
  await openCanvas(page, 1440);
  const sidebar = page.getByRole('complementary', { name: 'Components', exact: true });
  const search = sidebar.getByRole('textbox', { name: 'Search nodes', exact: true });
  const preprocessing = sidebar.getByRole('button', { name: 'Preprocessing', exact: true });
  await preprocessing.click();
  await search.fill('fill blanks');
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'true');
  await sidebar.getByRole('button', { name: 'Add Imputation node', exact: true }).click();
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
  await search.fill('normalize');
  await sidebar.getByRole('button', { name: 'Add Scaling node', exact: true }).dragTo(page.locator('.react-flow__pane'), { targetPosition: { x: 180, y: 150 } });
  await expect(page.locator('.react-flow__node')).toHaveCount(2);
  await expect(page.getByTestId('canvas-node-scale_numeric_features')).toHaveCount(1);
  await search.fill('nothingmatchesxyz');
  await expect(sidebar.getByText('No components found', { exact: true })).toBeVisible();
  await search.fill('');
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'false');
  await expect(sidebar.getByRole('button', { name: 'Add Scaling node', exact: true })).toBeHidden();
});
