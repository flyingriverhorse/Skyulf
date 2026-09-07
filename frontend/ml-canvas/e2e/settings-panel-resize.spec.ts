import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('resizing settings preserves edits, remembers width, and leaves room for the canvas', async ({ page }) => {
  // Resizing must preserve unfinished configuration and a usable canvas across panel changes.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  // Pin the library open so this test exercises settings limits with both panels visible.
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  const target = page.getByRole('textbox', { name: 'Target Column', exact: true });
  await target.fill('outcome');

  const handle = page.getByRole('separator', { name: 'Resize settings panel' });
  await expect(handle).toBeVisible();
  await handle.hover();
  const start = await handle.boundingBox();
  if (!start) throw new Error('Resize handle has no bounds');
  await page.mouse.move(start.x + start.width / 2, start.y + 100);
  await page.mouse.down();
  await page.mouse.move(start.x + start.width / 2 - 100, start.y + 100, { steps: 5 });
  await page.mouse.up();
  await expect(handle).toHaveAttribute('aria-valuenow', '420');
  await expect(target).toHaveValue('outcome');

  await handle.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(handle).toHaveAttribute('aria-valuenow', '440');
  await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
  await expect(handle).toHaveCount(0);
  await expect(target).toHaveValue('outcome');
  await page.getByRole('button', { name: 'Collapse settings panel', exact: true }).click();
  await expect(handle).toHaveAttribute('aria-valuenow', '440');

  await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  await expect(handle).toHaveCount(0);
  await page.locator('.react-flow__node').first().click();
  await expect(handle).toHaveAttribute('aria-valuenow', '440');
  await expect(target).toHaveValue('outcome');

  await page.setViewportSize({ width: 1100, height: 900 });
  await expect(handle).toHaveAttribute('aria-valuenow', '380');
  await page.getByRole('button', { name: 'Collapse sidebar', exact: true }).click();
  await expect(handle).toHaveAttribute('aria-valuenow', '440');
  await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
  await expect(handle).toHaveAttribute('aria-valuenow', '380');
  await handle.focus();
  await page.keyboard.press('End');
  await expect.poll(async () => (await page.locator('.react-flow').boundingBox())?.width).toBeGreaterThanOrEqual(399);
  await page.keyboard.press('Home');
  await expect(handle).toHaveAttribute('aria-valuenow', '320');
  await expect(target).toHaveValue('outcome');
});
