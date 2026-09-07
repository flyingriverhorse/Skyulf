import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';
import AxeBuilder from '@axe-core/playwright';

test('theme switching updates every surface together and preserves normal motion', async ({ page }) => {
  // Mixed color-transition durations must not leave panels in the previous theme.
  await mockBackend(page);
  await page.addInitScript(() => {
    if (!localStorage.getItem('skyulf-theme')) localStorage.setItem('skyulf-theme', 'light');
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await expect(page.getByRole('button', { name: 'Switch to dark mode' })).toBeVisible();
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  const transitions = await page.evaluate(async () => {
    document.querySelector<HTMLButtonElement>('[aria-label="Switch to dark mode"]')!.click();
    await new Promise(requestAnimationFrame);
    return document.getAnimations().filter(animation => animation instanceof CSSTransition)
      .map(animation => (animation as CSSTransition).transitionProperty)
      .filter(property => /color|background|border|shadow|fill|stroke/.test(property));
  });
  expect(transitions).toEqual([]);
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(22, 24, 27)');
  await expect(page.locator('#app-sidebar')).toHaveCSS('background-color', 'rgb(28, 31, 35)');
  await expect(page.locator('html')).not.toHaveClass(/theme-switching/);
  expect((await new AxeBuilder({ page }).withRules(['color-contrast']).analyze()).violations).toEqual([]);
  await page.screenshot({ path: 'test-results/theme-dark.png' });
  await page.getByRole('button', { name: 'Switch to light mode' }).click();
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(245, 246, 252)');
  await expect(page.locator('#app-sidebar')).toHaveCSS('background-color', 'rgb(255, 255, 255)');
  await expect(page.locator('html')).not.toHaveClass(/theme-switching/);
  await expect(page.locator('#app-sidebar')).toHaveCSS('transition-duration', '0.2s');
  expect((await new AxeBuilder({ page }).withRules(['color-contrast']).analyze()).violations).toEqual([]);
  await page.screenshot({ path: 'test-results/theme-light.png' });
  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  expect(await page.evaluate(() => localStorage.getItem('skyulf-theme'))).toBe('dark');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Switch to light mode' })).toBeVisible();
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(22, 24, 27)');
});

test('the real logo loads on desktop and mobile with reduced motion', async ({ page }) => {
  // Branding must resolve in Vite and stay available when navigation becomes a drawer.
  await mockBackend(page);
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.goto('/canvas');
  const logo = page.locator('#app-sidebar').getByRole('img', { name: 'Skyulf logo' });
  await expect(logo).toBeVisible();
  expect(await logo.evaluate(img => (img as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
  await page.setViewportSize({ width: 390, height: 700 });
  await page.getByRole('button', { name: 'Open navigation menu' }).click();
  await expect(logo).toBeInViewport();
  await page.getByRole('button', { name: 'Switch to light mode' }).click();
  await expect(page.locator('body')).toHaveCSS('background-color', 'rgb(245, 246, 252)');
  await page.getByRole('button', { name: 'Close navigation menu' }).click();
  await expect(page.locator('html')).not.toHaveClass(/theme-switching/);
  await expect(page.getByRole('button', { name: 'Open navigation menu' })).toBeFocused();
});

test('primary actions share colors across pages in both themes', async ({ page }) => {
  // Shared brand colors must also keep labels readable on existing upload actions.
  await mockBackend(page);
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
  await page.goto('/data');
  const upload = page.getByRole('button', { name: 'Upload File', exact: true });
  await expect(upload).toBeVisible();
  await expect(upload).toHaveCSS('color', 'rgb(23, 16, 43)');
  await expect(upload).toHaveCSS('background-color', 'rgb(242, 192, 37)');
  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  await expect(upload).toHaveCSS('color', 'rgb(17, 24, 32)');
  await expect(upload).toHaveCSS('background-color', 'rgb(117, 184, 245)');
  expect((await new AxeBuilder({ page }).include('.action-primary').withRules(['color-contrast']).analyze()).violations).toEqual([]);
  await page.getByRole('link', { name: 'ML Canvas', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Preview data', exact: true })).toHaveCSS('background-color', 'rgb(117, 184, 245)');
  await expect(page.getByRole('button', { name: 'Browse templates', exact: true })).toHaveCSS('background-color', 'rgb(117, 184, 245)');
});

test('the logo header divider aligns with the canvas navbar', async ({ page }) => {
  // The navigation rail and canvas header must form one continuous horizontal line.
  await mockBackend(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  for (const width of [1440, 1100, 768]) {
    await page.setViewportSize({ width, height: 900 });
    await expect.poll(async () => {
      const logoHeader = await page.locator('#app-sidebar > div').first().boundingBox();
      const navbar = await page.getByTestId('navbar-views').locator('..').boundingBox();
      return logoHeader && navbar ? Math.abs(logoHeader.y + logoHeader.height - navbar.y - navbar.height) : Infinity;
    }).toBeLessThan(1);
  }
});
