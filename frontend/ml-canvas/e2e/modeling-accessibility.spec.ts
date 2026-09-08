import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test.beforeEach(async ({ page }) => {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*/defaults*', route => route.fulfill({ json: {} }));
  await page.route('**/api/pipeline/jobs/tuning/history/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await expect(page.locator('.react-flow')).toBeVisible();
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  await page.getByRole('button', { name: 'Advanced (Tuning)', exact: true }).click();
});

test('modeling help dismisses before its dialog and returns after refocus or hover', async ({ page }) => {
  // Escape must remove obscuring help without losing edits, focus, or access to the same explanation.
  await page.getByRole('combobox', { name: 'Search Method', exact: true }).selectOption('optuna');
  const opener = page.getByRole('button', { name: 'Search strategy settings', exact: true });
  await opener.click();
  const dialog = page.getByRole('dialog', { name: 'Optuna Settings', exact: true });
  const help = dialog.getByRole('button', { name: 'Help', exact: true }).first();
  const tooltip = page.getByRole('tooltip');
  await expect.poll(() => dialog.evaluate(element => element.contains(document.activeElement))).toBe(true);
  await dialog.evaluate(async element => { await Promise.all(element.getAnimations().map(animation => animation.finished)); });
  await help.focus();
  await expect(tooltip).toContainText('TPE provides smart Bayesian learning');
  await expect(help).toHaveAccessibleDescription(/TPE provides smart Bayesian learning/);
  const helpCard = page.locator('[data-radix-popper-content-wrapper]');
  await expect(helpCard).toBeVisible();
  const bounds = await helpCard.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(1440);
  await help.press('Escape');
  await expect(tooltip).toHaveCount(0);
  await expect(dialog).toBeVisible();
  await expect(help).toBeFocused();

  await dialog.getByRole('combobox', { name: 'Sampler', exact: true }).focus();
  await help.focus();
  await expect(tooltip).toBeVisible();
  await help.press('Escape');
  await expect(tooltip).toHaveCount(0);
  await page.mouse.move(0, 0);
  await help.hover();
  await expect(tooltip).toBeVisible();
  await help.press('Escape');
  await expect(tooltip).toHaveCount(0);
  await expect(dialog).toBeVisible();
  await expect(help).toBeFocused();
  await help.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
});

test('best parameter history traps focus and restores its opener', async ({ page }) => {
  // Keyboard navigation must stay inside history until dismissal returns to the live settings action.
  const footer = page.getByTestId('training-action-footer');
  await footer.evaluate(element => {
    let parent = element.parentElement;
    while (parent && !/auto|scroll/.test(getComputedStyle(parent).overflowY)) parent = parent.parentElement;
    if (!parent) throw new Error('Settings scroll container missing');
    parent.scrollTop = parent.scrollHeight;
  });
  await expect(footer).toHaveAttribute('data-expanded', 'true');
  const opener = page.getByRole('button', { name: 'View Best Parameters History', exact: true });
  await opener.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Best Parameters History', exact: true });
  const model = dialog.getByRole('combobox', { name: 'View parameters for:', exact: true });
  const close = dialog.getByRole('button', { name: 'Close history', exact: true });
  await expect.poll(() => dialog.evaluate(element => element.contains(document.activeElement))).toBe(true);
  await dialog.evaluate(async element => { await Promise.all(element.getAnimations().map(animation => animation.finished)); });
  await model.focus();
  await model.press('Shift+Tab');
  await expect(close).toBeFocused();
  await close.press('Tab');
  await expect(model).toBeFocused();
  await model.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(opener).toBeFocused();
});
