import { test, expect } from '@playwright/test';
import type { Route } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/**
 * Happy-path: user has a wired pipeline (Dataset → Drop Columns) and
 * clicks Run Preview. The toolbar fires `POST /pipeline/preview`,
 * the response lands in `useGraphStore.executionResult`, and
 * `ResultsPanel` renders the rows.
 *
 * Why we seed via `window.__skyulfTest.graphStore` instead of dragging:
 * React Flow's HTML5 connection drag (handle → handle) is unreliable
 * under headless Chromium even with `dispatchEvent` — empirically less
 * than 50% reliable on CI. The dev hook (gated to `import.meta.env.DEV`)
 * lets us seed the exact graph state in O(ms) and exercise the
 * **real** Toolbar gating, converter, axios POST, and ResultsPanel
 * render path. That's where the regressions actually happen.
 */
test.describe('Run Preview happy path', () => {
  test.beforeEach(async ({ page }) => {
    await mockBackend(page);
  });

  test('Run Preview button fires /pipeline/preview and renders rows', async ({ page }) => {
    // Capture the request so we can assert the toolbar dispatched it.
    let previewRequestSeen = false;

    await page.route('**/api/pipeline/preview', (route: Route) => {
      previewRequestSeen = true;
      void route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          pipeline_id: 'test-pipeline',
          status: 'success',
          node_results: {},
          preview_data: {
            rows: [
              { sepal_length: 5.1, sepal_width: 3.5, species: 'setosa' },
              { sepal_length: 4.9, sepal_width: 3.0, species: 'setosa' },
              { sepal_length: 6.2, sepal_width: 3.4, species: 'virginica' },
            ],
            columns: ['sepal_length', 'sepal_width', 'species'],
          },
          preview_totals: { _total: 150 },
          recommendations: [],
        }),
      });
    });

    await page.goto('/canvas');
    await expect(page.locator('.react-flow')).toBeVisible({ timeout: 10_000 });

    // Wait until the dev hook has installed itself.
    await page.waitForFunction(
      () => Boolean((window as unknown as { __skyulfTest?: unknown }).__skyulfTest),
      undefined,
      { timeout: 5_000 },
    );

    // Seed: Dataset node (with a chosen datasetId) → Drop Columns node,
    // connected by an edge. Mirrors what `addNode` produces in real use.
    await page.evaluate(() => {
      type AnyRec = Record<string, unknown>;
      const hook = (window as unknown as {
        __skyulfTest: { graphStore: { getState: () => AnyRec } };
      }).__skyulfTest;
      const state = hook.graphStore.getState();
      const setGraph = state.setGraph as (n: unknown[], e: unknown[]) => void;

      const datasetId = 'dataset_node-test-1';
      const dropId = 'drop_missing_columns-test-1';

      setGraph(
        [
          {
            id: datasetId,
            type: 'custom',
            position: { x: 100, y: 100 },
            data: {
              definitionType: 'dataset_node',
              catalogType: 'dataset_node',
              datasetId: 'iris-demo',
              datasetName: 'Iris (demo)',
            },
          },
          {
            id: dropId,
            type: 'custom',
            position: { x: 400, y: 100 },
            data: {
              definitionType: 'drop_missing_columns',
              catalogType: 'drop_missing_columns',
              threshold: 0.5,
            },
          },
        ],
        [
          {
            id: 'e-1',
            source: datasetId,
            target: dropId,
            sourceHandle: 'data',
            targetHandle: 'in',
          },
        ],
      );
    });

    // Toolbar gates Run Preview on `canRunPreview` (dataset + outgoing edge).
    const runPreview = page.getByRole('button', { name: /run preview/i });
    await expect(runPreview).toBeVisible({ timeout: 5_000 });
    await runPreview.click();

    // Backend was actually called.
    await expect.poll(() => previewRequestSeen, { timeout: 5_000 }).toBe(true);

    // ResultsPanel renders the mocked rows. Assert on a unique cell value
    // rather than column header (column names also appear in the dataset
    // settings panel).
    await expect(page.getByText('setosa').first()).toBeVisible({ timeout: 5_000 });
    await expect(page.getByText('virginica').first()).toBeVisible();
  });
});
