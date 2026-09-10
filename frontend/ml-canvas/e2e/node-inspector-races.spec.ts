import { expect, test, type Page, type Route } from '@playwright/test';
import type { NodeInspectorResponse } from '../src/core/api/monitoring';
import { mockBackend } from './fixtures/mockApi';

/** Present a real Error Log entry whose node link opens the historical inspector. */
async function prepare(page: Page, width: number) {
  await mockBackend(page);
  await page.setViewportSize({ width, height: 1000 });
  await page.route('**/api/monitoring/errors?*', route => route.fulfill({ json: {
    total: 0, total_unfiltered: 0, entries: [],
    facets: { severities: [], error_types: [], job_ids: [] }, filters: {},
  } }));
  await page.route('**/api/monitoring/errors/timeline?*', route => route.fulfill({ json: [] }));
  await page.route('**/api/monitoring/errors/grouped', route => route.fulfill({ json: [] }));
  await page.route('**/api/monitoring/pipeline-logs?*', route => route.fulfill({ json: {
    total: 1, total_unfiltered: 1,
    entries: [{ id: 51, pipeline_id: 'inspection-run', node_id: 'first', node_type: 'training',
      level: 'error', logger: 'skyulf', message: 'Inspect this training step', run_at: '2026-09-10T10:00:00Z' }],
    facets: { levels: ['error'], node_types: ['training'], node_ids: ['first'], pipeline_ids: ['inspection-run'] }, filters: {},
  } }));
}

/** Both node identity and neighbor navigation come from the stored run response. */
function nodeResponse(nodeId: string, label: string): NodeInspectorResponse {
  return {
    job_id: 'inspection-job', node_id: nodeId, node_found: true,
    node: { node_id: nodeId, label, step_type: 'training', params: {},
      upstream: [{ node_id: 'upstream', step_type: 'imputer', label: 'Imputer' }], downstream: [] },
    pipeline_id: 'inspection-run', dataset_source_id: 'dataset-1', dataset_name: 'Inspection data',
    branch_index: null, run_mode: 'fixed', model_type: 'RandomForest', status: 'completed',
    started_at: '2026-09-10T10:00:00Z', finished_at: '2026-09-10T10:01:00Z',
    is_synthetic_pipeline: true, can_open_in_canvas: false, recent_logs: [],
  };
}

for (const width of [1440, 900]) {
  for (const outcome of ['success', 'error'] as const) {
    test(`inspector ignores an old opening's ${outcome} and still retries neighbor navigation at ${width}px`, async ({ page }) => {
      // Late HTTP completions must not replace a reopened modal's data, error or loading state.
      await prepare(page, width);
      const firstUrl = '**/api/monitoring/pipeline-runs/inspection-run/nodes/first';
      let oldRoute: Route | undefined;
      await page.route(firstUrl, route => {
        if (!oldRoute) { oldRoute = route; return; }
        return route.fulfill({ json: nodeResponse('first', 'Current detail') });
      });
      let upstreamAttempts = 0;
      await page.route('**/api/monitoring/pipeline-runs/inspection-run/nodes/upstream', route => {
        upstreamAttempts++;
        if (upstreamAttempts === 1) return route.fulfill({ status: 503, json: { detail: 'Temporary failure' } });
        return route.fulfill({ json: nodeResponse('upstream', 'Recovered upstream detail') });
      });
      await page.goto('/errors');
      const opener = page.getByRole('button', { name: 'Node first', exact: true });
      await opener.focus();
      await page.keyboard.press('Enter');
      const dialog = page.getByRole('dialog', { name: 'Node Inspector', exact: true });
      await expect(dialog.getByRole('status')).toContainText('Loading node detail');
      await expect.poll(() => !!oldRoute).toBe(true);
      await page.keyboard.press('Escape');
      await expect(dialog).toBeHidden();
      await expect(opener).toBeFocused();
      await opener.click();
      await expect(dialog.getByRole('heading', { name: 'Current detail', exact: true })).toBeVisible();

      const completed = page.waitForResponse(response => response.url().endsWith('/nodes/first'));
      if (outcome === 'success') await oldRoute!.fulfill({ json: nodeResponse('first', 'Obsolete detail') });
      else await oldRoute!.fulfill({ status: 503, json: { detail: 'Obsolete failure' } });
      await (await completed).finished();
      // Let the delivered response and its React updates reach a painted frame before asserting absence.
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      await expect(dialog.getByRole('heading', { name: 'Current detail', exact: true })).toBeVisible();
      await expect(dialog.getByRole('alert')).toHaveCount(0);
      await expect(dialog.getByText('Obsolete detail', { exact: true })).toHaveCount(0);

      await dialog.getByRole('button', { name: 'Imputer (upstream)', exact: true }).click();
      await expect(dialog.getByRole('alert')).toBeVisible();
      await dialog.getByRole('button', { name: 'Retry', exact: true }).click();
      await expect(dialog.getByRole('heading', { name: 'Recovered upstream detail', exact: true })).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(dialog).toBeHidden();
      await expect(opener).toBeFocused();
    });
  }
}
