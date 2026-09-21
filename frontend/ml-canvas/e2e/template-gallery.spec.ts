import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mockBackend } from './fixtures/mockApi';

for (const width of [1440, 390]) {
  test(`template setup works at ${width}px`, async ({ page }, testInfo) => {
    // Verify real modal layout and chosen-data transfer without requiring a live backend.
    await page.setViewportSize({ width, height: 900 });
    await mockBackend(page);
    await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{ id: '1', name: 'Customer outcomes' }] } }));
    await page.route('**/api/pipeline/datasets/1/schema', route => route.fulfill({ json: {
      columns: { outcome: { name: 'outcome', dtype: 'string' }, spend: { name: 'spend', dtype: 'float64' } },
    } }));
    await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
    await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
    await page.goto('/canvas');
    if (width < 768) {
      await page.getByRole('button', { name: 'Read-only', exact: true }).click();
    }
    await page.getByTestId('empty-state-templates').click();
    const dialog = page.getByRole('dialog', { name: 'Start from a template' });
    await expect(dialog.getByRole('heading', { name: 'What would you like to discover?' })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('gallery.png'), animations: 'disabled' });
    await page.getByTestId('template-card-tabular_classification').click();
    await page.getByLabel('Dataset', { exact: true }).selectOption('1');
    await page.getByLabel('Target column', { exact: true }).selectOption('outcome');
    await expect(page.getByRole('button', { name: 'Open in Canvas' })).toBeEnabled();
    expect(await dialog.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
    const accessibility = await new AxeBuilder({ page }).include('[role="dialog"]').withTags(['wcag2a', 'wcag2aa']).analyze();
    expect(accessibility.violations).toEqual([]);
    await page.screenshot({ path: testInfo.outputPath('setup.png'), animations: 'disabled' });
    await page.getByRole('button', { name: 'Open in Canvas' }).click();
    await expect(dialog).toBeHidden();
    await expect(page.locator('.react-flow__node')).toHaveCount(7);
  });
}
