import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const width of [1440, 390]) {
  test(`application branding and loading motion at ${width}px`, async ({ page }, testInfo) => {
    // The real assets must load, while a reduced-motion preference stops video playback.
    await mockBackend(page);
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: 'no-preference' });
    await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
    let finishLoading!: () => void;
    const loading = new Promise<void>(resolve => { finishLoading = resolve; });
    await page.route('**/data/api/sources', async route => {
      await loading;
      await route.fulfill({ json: { sources: [] } });
    });
    await page.goto('/data');
    const status = page.getByRole('status').filter({ hasText: 'Loading datasets...' });
    const video = status.locator('video');
    await expect(video).toBeVisible();
    await expect.poll(() => video.evaluate(element => element.currentTime)).toBeGreaterThan(0);
    expect(await video.evaluate(element => element.error)).toBeNull();
    // Capture the developed mark instead of the supplied video's opening fade-in.
    await video.evaluate(element => new Promise<void>(resolve => {
      element.addEventListener('seeked', () => resolve(), { once: true });
      element.currentTime = element.duration / 2;
    }));
    await page.screenshot({ path: testInfo.outputPath('loading-light.png') });

    const favicon = page.locator('link[rel="icon"]');
    await expect(favicon).toHaveAttribute('type', 'image/svg+xml');
    const faviconUrl = await favicon.evaluate(element => (element as HTMLLinkElement).href);
    const response = await page.request.get(faviconUrl);
    expect(response.ok()).toBe(true);
    expect(await response.text()).toContain('blueYellowGrad');

    if (width < 768) await page.getByRole('button', { name: 'Open navigation menu' }).click();
    const sidebar = page.locator('#app-sidebar');
    const logo = sidebar.getByRole('img', { name: 'Skyulf ML logo' });
    await expect(logo).toBeVisible();
    await expect.poll(() => logo.evaluate(element => (element as HTMLImageElement).naturalWidth)).toBeGreaterThan(0);
    await sidebar.getByRole('button', { name: 'Switch to dark mode' }).click();
    await expect(page.locator('html')).toHaveClass(/dark/);
    await page.screenshot({ path: testInfo.outputPath('navigation-dark.png') });
    if (width < 768) await page.getByRole('button', { name: 'Close navigation menu' }).click();

    await page.emulateMedia({ reducedMotion: 'reduce' });
    await expect(status.locator('video')).toHaveCount(0);
    await expect(status.locator('img')).toBeVisible();
    finishLoading();
    await expect(status).toHaveCount(0);
  });
}
