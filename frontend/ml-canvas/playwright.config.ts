import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright config for the canvas SPA.
 *
 * Strategy: hermetic E2E. We boot Vite's dev server (so source maps and
 * the React Flow renderer behave like in development) and stub every
 * backend HTTP call inside the specs themselves via `page.route()`.
 * Backend (FastAPI / Celery / Postgres) is intentionally NOT required
 * to keep CI fast and deterministic — these specs verify UI wiring,
 * not server contracts.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 5173 --strictPort',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
