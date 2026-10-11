import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const width of [1440, 390]) {
  test(`application branding and loading motion at ${width}px`, async ({ page }, testInfo) => {
    // The supplied scene stays transparent, while reduced motion stops animated frame work.
    await mockBackend(page);
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
    const mediaRequests: string[] = [];
    page.on('request', request => { if (/\.(mp4|webm)(\?|$)/.test(request.url())) mediaRequests.push(request.url()); });
    let finishLoading!: () => void;
    const loading = new Promise<void>(resolve => { finishLoading = resolve; });
    await page.route('**/data/api/sources', async route => {
      await loading;
      await route.fulfill({ json: { sources: [] } });
    });
    await page.goto('/data');
    await expect(page).toHaveTitle('Kayra');
    const status = page.getByRole('status').filter({ hasText: 'Loading datasets...' });
    const scene = status.locator('svg');
    await expect(scene).toBeVisible();
    await expect(scene).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)');
    await expect.poll(() => scene.locator('line').count()).toBeGreaterThan(0);
    const camera = scene.locator(':scope > g');
    const initialTransform = await camera.getAttribute('transform');
    await expect.poll(() => camera.getAttribute('transform')).not.toBe(initialTransform);
    await page.screenshot({ path: testInfo.outputPath('loading-light.png') });

    const favicon = page.locator('link[rel="icon"]');
    await expect(favicon).toHaveAttribute('type', 'image/png');
    const faviconUrl = await favicon.evaluate(element => (element as HTMLLinkElement).href);
    const response = await page.request.get(faviconUrl);
    expect(response.ok()).toBe(true);
    expect((await response.body()).subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');

    if (width < 768) await page.getByRole('button', { name: 'Open navigation menu' }).click();
    const sidebar = page.locator('#app-sidebar');
    const logo = sidebar.getByRole('img', { name: 'Kayra logo', exact: true });
    await expect(sidebar.getByRole('heading', { name: 'Kayra', exact: true })).toBeVisible();
    await expect(sidebar.getByText('by Skyulf', { exact: true })).toBeVisible();
    await expect(logo).toBeVisible();
    await expect.poll(() => logo.evaluate(element => (element as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
    await sidebar.getByRole('button', { name: 'Switch to dark mode' }).click();
    await expect(page.locator('html')).toHaveClass(/dark/);
    await page.screenshot({ path: testInfo.outputPath('navigation-dark.png') });
    if (width < 768) await page.getByRole('button', { name: 'Close navigation menu' }).click();

    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect(status.locator('svg')).toHaveCount(0);
    await expect(status.locator('img')).toBeVisible();
    expect(mediaRequests).toEqual([]);
    finishLoading();
    await expect(status).toHaveCount(0);
  });
}
