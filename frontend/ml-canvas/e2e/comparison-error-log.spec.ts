import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Serve saved runs through the real jobs API routes used by both comparison views. */
async function mockComparison(page: Page) {
  await mockBackend(page);
  const jobs = [10, 20].map((estimators, index) => ({
    job_id: `comparison-${index}`, pipeline_id: `compare${index}`, node_id: 'model',
    job_type: 'training', status: 'completed', model_type: 'random_forest_classifier',
    model_family: 'classification', dataset_name: `Comparison dataset ${index + 1}`,
    created_at: '2026-09-09T10:00:00Z', start_time: '2026-09-09T10:00:00Z',
    end_time: '2026-09-09T10:01:00Z', error: null, result: {},
    hyperparameters: { hyperparameters: { n_estimators: estimators } },
    metrics: { test_accuracy: index ? 0.91 : 0.82 },
    graph: { nodes: [
      { node_id: 'data', step_type: 'dataset', params: {}, inputs: [] },
      { node_id: 'scale', step_type: 'StandardScaler', params: { method: 'standard' }, inputs: ['data'] },
      { node_id: 'model', step_type: 'training', params: { n_estimators: estimators }, inputs: ['scale'] },
    ] },
  }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: jobs }));
  for (const job of jobs) {
    await page.route(`**/api/pipeline/jobs/${job.job_id}`, route => route.fulfill({ json: job }));
  }
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
}

/** Apply server-side search facets so browser assertions cover actual request wiring. */
async function mockErrorLog(page: Page) {
  await mockBackend(page);
  const errorQueries: URLSearchParams[] = [];
  const pipelineQueries: URLSearchParams[] = [];
  const events = [
    { id: 41, severity: 'critical', status_code: 500, error_type: 'ValueError',
      message: 'Training failed', route: '/api/train', created_at: '2026-09-09T10:00:00Z',
      resolved_at: null, traceback: 'Traceback: invalid target' },
    { id: 42, severity: 'warning', status_code: 400, error_type: 'InputError',
      message: 'Missing feature', route: '/api/preview', created_at: '2026-09-09T10:01:00Z',
      resolved_at: null, traceback: '' },
  ];
  const logs = [{ id: 51, pipeline_id: null, node_id: null, node_type: 'encoder',
    level: 'error', logger: 'skyulf', message: 'Encoding failed', run_at: '2026-09-09T10:02:00' }];
  await page.route('**/api/monitoring/errors?*', route => {
    const query = new URL(route.request().url()).searchParams;
    errorQueries.push(query);
    const entries = events.filter(event => (!query.get('severity') || event.severity === query.get('severity'))
      && event.message.toLowerCase().includes((query.get('q') ?? '').toLowerCase()));
    return route.fulfill({ json: { total: entries.length, total_unfiltered: events.length, entries,
      facets: { severities: ['critical', 'warning'], error_types: ['ValueError', 'InputError'], job_ids: [] },
      filters: {},
    } });
  });
  await page.route('**/api/monitoring/pipeline-logs?*', route => {
    const query = new URL(route.request().url()).searchParams;
    pipelineQueries.push(query);
    const entries = logs.filter(log => (!query.get('level') || log.level === query.get('level'))
      && log.message.toLowerCase().includes((query.get('q') ?? '').toLowerCase()));
    return route.fulfill({ json: { total: entries.length, total_unfiltered: logs.length, entries,
      facets: { levels: ['error'], node_types: ['encoder'], node_ids: [], pipeline_ids: [] }, filters: {},
    } });
  });
  await page.route('**/api/monitoring/errors/timeline?*', route => route.fulfill({ json: [] }));
  await page.route('**/api/monitoring/errors/grouped', route => route.fulfill({ json: [{
    error_type: 'ValueError', route: '/api/train', count: 1, sample_id: 41,
    first_seen: '2026-09-09T10:00:00Z', last_seen: '2026-09-09T10:00:00Z',
  }] }));
  await page.route('**/api/monitoring/errors/41', route => route.fulfill({ json: events[0] }));
  return { errorQueries, pipelineQueries };
}

for (const width of [1440, 900]) {
  test(`comparison table and saved pipeline diff retain selection at ${width}px`, async ({ page }) => {
    // Collapsed rows survive view switches while Swap changes the saved graph roles.
    await page.setViewportSize({ width, height: 1000 });
    await mockComparison(page);
    await page.goto('/canvas');
    await page.getByRole('tab', { name: 'Experiments', exact: true }).click();
    await page.getByText('Comparison dataset 1', { exact: false }).click();
    await page.getByText('Comparison dataset 2', { exact: false }).click();
    await page.getByRole('button', { name: 'Detailed Metrics & Params', exact: true }).click();
    await expect(page.getByRole('cell', { name: '0.8200', exact: true })).toBeVisible();
    await expect(page.getByRole('cell', { name: '0.9100 ★', exact: true })).toBeVisible();
    await expect(page.getByRole('row').filter({ hasText: 'n_estimators' })).toHaveText(/10.*20/);
    const step = page.getByRole('row').filter({ hasText: 'method=standard' });
    await expect(step).toBeVisible();
    await page.getByText('Pipeline Steps', { exact: true }).click();
    await expect(step).toBeHidden();

    await page.getByRole('button', { name: 'Pipeline Diff', exact: true }).click();
    await expect(page.getByText('Diff summary:', { exact: true })).toBeVisible();
    await expect(page.getByText('1 modified', { exact: true })).toBeVisible();
    await expect(page.locator('.react-flow__node')).toHaveCount(6);
    const baselineHeader = page.getByText('Baseline', { exact: true }).locator('..');
    await expect(baselineHeader).toContainText('compare0');
    const swap = page.getByRole('button', { name: 'Swap', exact: true });
    await swap.focus();
    await swap.press('Enter');
    await expect(baselineHeader).toContainText('compare1');
    await expect(swap).toBeFocused();
    await expect(page.getByText('1 modified', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Detailed Metrics & Params', exact: true }).click();
    await expect(step).toBeHidden();
    await page.getByText('Pipeline Steps', { exact: true }).click();
    await expect(step).toBeVisible();
    await expect(page.getByRole('cell', { name: '0.9100 ★', exact: true })).toBeVisible();
  });

  test(`error log preserves server filters and issue details at ${width}px`, async ({ page }) => {
    // Native controls keep focus and send the same facets to HTTP and pipeline history.
    await page.setViewportSize({ width, height: 1000 });
    const { errorQueries, pipelineQueries } = await mockErrorLog(page);
    await page.goto('/errors');
    await expect(page.getByRole('heading', { name: 'Error Log', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Events (3)', exact: true })).toBeVisible();
    const severity = page.getByRole('combobox', { name: 'All severities', exact: true });
    await severity.selectOption('critical');
    await expect(page.getByRole('button', { name: 'Events (2)', exact: true })).toBeVisible();
    await expect.poll(() => errorQueries.at(-1)?.get('severity')).toBe('critical');
    await expect.poll(() => pipelineQueries.at(-1)?.get('level')).toBe('error');
    const search = page.getByPlaceholder('Search errors, job id, node id…');
    await search.fill('not-present');
    await expect(page.getByText('No matching errors', { exact: true })).toBeVisible();
    await expect(search).toBeFocused();
    await search.fill('');
    await expect(page.getByRole('button', { name: 'Events (2)', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Show resolved', exact: true }).click();
    await expect.poll(() => errorQueries.at(-1)?.get('show_resolved')).toBe('true');
    await page.getByRole('button', { name: 'All', exact: true }).click();
    await expect.poll(() => errorQueries.at(-1)?.has('since')).toBe(false);
    await page.getByRole('button', { name: /^Issues \(/ }).click();
    await page.getByRole('button', { name: 'View sample', exact: true }).click();
    await expect(page.getByText('Traceback: invalid target', { exact: true })).toBeVisible();
  });
}
