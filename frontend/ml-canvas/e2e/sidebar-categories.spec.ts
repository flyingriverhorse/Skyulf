import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('category collapse keeps search matches reachable and restores the browsing layout', async ({ page }) => {
  // Searching must reveal nodes from collapsed groups without discarding the user's browsing choices.
  await mockBackend(page);
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
  const preprocessing = page.getByRole('button', { name: 'Preprocessing', exact: true });
  const indicator = page.getByRole('button', { name: 'Add Missing Indicator node', exact: true });
  const search = page.getByRole('textbox', { name: 'Search nodes' });
  await page.getByRole('button', { name: 'Data cleaning', exact: true }).click();
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'true');
  await preprocessing.focus();
  await page.keyboard.press('Enter');
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'false');
  await expect(indicator).toBeHidden();
  await expect(page.getByRole('button', { name: 'Modeling', exact: true })).toBeInViewport();

  await search.fill('missing values');
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'true');
  await expect(preprocessing).toBeDisabled();
  await expect(indicator).toBeVisible();
  await indicator.focus();
  await page.keyboard.press('Space');
  await expect(page.locator('.react-flow__node')).toHaveCount(1);

  await search.fill('');
  await expect(preprocessing).toBeEnabled();
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'false');
  await expect(indicator).toBeHidden();
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'false');
  await preprocessing.focus();
  await page.keyboard.press('Space');
  await expect(preprocessing).toHaveAttribute('aria-expanded', 'true');
  await expect(indicator).toBeVisible();
});
