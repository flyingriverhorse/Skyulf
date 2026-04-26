import type { Page, Route } from '@playwright/test';

/**
 * Stub every backend HTTP path the canvas might hit during a smoke
 * run. Specs that need a custom response should call `page.route()`
 * AFTER `mockBackend(page)` to override a specific endpoint.
 *
 * We restrict matches to `xhr` / `fetch` requests so we don't
 * accidentally swallow Vite module scripts (e.g. anything in
 * `node_modules` whose path happens to contain `/api/`) and break
 * the SPA boot.
 */
function jsonStub(body: unknown) {
  return (route: Route): void => {
    if (!['xhr', 'fetch'].includes(route.request().resourceType())) {
      void route.continue();
      return;
    }
    void route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  };
}

export async function mockBackend(page: Page): Promise<void> {
  await page.route('**/api/**', jsonStub({}));
  await page.route('**/data/api/**', jsonStub([]));
  await page.route('**/ml-workflow/**', jsonStub({}));
}

/** Minimal dataset list fixture. */
export const sampleDatasets = [
  {
    id: 'iris-demo',
    name: 'Iris (demo)',
    rows: 150,
    columns: 5,
    file_type: 'csv',
  },
];
