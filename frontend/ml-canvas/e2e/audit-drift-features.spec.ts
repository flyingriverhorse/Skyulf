import { expect, test, type Page } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';
import type { DriftAlertStatus, DriftDispositionAction } from '../src/core/api/monitoring';

/** Apply the audit API facets so the browser exercises server-filter wiring. */
async function mockAudit(page: Page) {
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  const queries: URLSearchParams[] = [];
  const entries = [
    { id: 2, version_int: 2, name: 'Updated scaler', note: 'Changed scaling', kind: 'manual',
      user_id: 7, created_at: '2026-09-10T10:00:00Z', node_count: 2, edge_count: 1,
      diff: { nodes_added: [], nodes_removed: [], nodes_modified: ['scale-node'], delta_node_count: 0 } },
    { id: 1, version_int: 1, name: 'Initial pipeline', note: null, kind: 'auto',
      user_id: null, created_at: '2026-09-10T09:00:00Z', node_count: 2, edge_count: 1,
      diff: { nodes_added: ['data-node', 'scale-node'], nodes_removed: [], nodes_modified: [], delta_node_count: 2 } },
  ];
  await page.route('**/api/pipeline/versions/iris-demo/audit?*', route => {
    const query = new URL(route.request().url()).searchParams;
    queries.push(query);
    const filtered = entries.filter(entry => (!query.has('kind') || entry.kind === query.get('kind'))
      && (!query.has('actor') || String(entry.user_id ?? '__anonymous__') === query.get('actor')));
    return route.fulfill({ json: { dataset_source_id: 'iris-demo', total: filtered.length,
      total_unfiltered: entries.length, entries: filtered,
      facets: { actors: ['7'], kinds: ['manual', 'auto'], has_anonymous_actor: true }, filters: {},
    } });
  });
  return queries;
}

/** Keep disposition state on the mocked server to exercise the real detail hook. */
async function mockDriftAlert(page: Page) {
  await mockBackend(page);
  await page.route('**/api/monitoring/jobs', route => route.fulfill({ json: [] }));
  const submissions: { action: DriftDispositionAction; actor: string; note: string | null }[] = [];
  const history: { status: DriftAlertStatus; actor: string; note: string | null; at: string }[] = [];
  let status: DriftAlertStatus = 'new';
  const detail = () => ({
    id: 41, job_id: 'drift-job', severity: 'critical', status, evaluation_status: 'completed',
    created_at: '2026-09-10T10:00:00Z', drifted_columns_count: 1, total_columns: 4,
    threshold_version: 3, threshold_psi: 0.2, threshold_ks: 0.05,
    threshold_wasserstein: 0.1, threshold_kl: 0.1,
    owner: history.at(-1)?.actor ?? null, disposition_history: history,
    column_drifts: { age: { column: 'age', drift_detected: true, suggestions: [],
      metrics: [{ metric: 'psi', value: 0.456789, threshold: 0.2, drift_detected: true }] } },
  });
  await page.route('**/api/monitoring/drift/alerts/41', route => route.fulfill({ json: detail() }));
  await page.route('**/api/monitoring/drift/alerts/41/disposition', route => {
    const body = route.request().postDataJSON() as typeof submissions[number];
    submissions.push(body);
    const next: Record<DriftDispositionAction, DriftAlertStatus> = {
      acknowledge: 'acknowledged', resolve: 'resolved', reopen: 'reopened',
    };
    status = next[body.action];
    history.push({ status, actor: body.actor, note: body.note, at: '2026-09-10T10:05:00Z' });
    return route.fulfill({ json: detail() });
  });
  return submissions;
}

for (const width of [1440, 900]) {
  test(`audit filters and expanded changes stay usable at ${width}px`, async ({ page }) => {
    // A selected facet must filter server history while retaining the other facet options.
    await page.setViewportSize({ width, height: 1000 });
    const queries = await mockAudit(page);
    await page.goto('/audit');
    await expect(page.getByRole('heading', { name: 'Pipeline Audit Log', exact: true })).toBeVisible();
    const updated = page.getByRole('button', { name: /Updated scaler/ });
    await updated.focus();
    await updated.press('Enter');
    await expect(page.getByText('Modified (1)', { exact: true })).toBeVisible();
    await expect(updated).toBeFocused();
    await expect(page.getByText('scale-node', { exact: true })).toBeVisible();
    const actor = page.getByRole('combobox', { name: 'Actor', exact: true });
    await actor.focus();
    await actor.selectOption('7');
    await expect.poll(() => queries.at(-1)?.get('actor')).toBe('7');
    await expect(page.getByText('Initial pipeline', { exact: true })).toBeHidden();
    await expect(actor.locator('option')).toHaveText(['All actors', 'user #7', 'anonymous']);
    await expect(actor).toBeFocused();
    await page.getByRole('combobox', { name: 'Action kind', exact: true }).selectOption('auto');
    await expect(page.getByText(/No audit records match the current filters/)).toBeVisible();
    await actor.selectOption('__anonymous__');
    await expect(page.getByText('Initial pipeline', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: '25', exact: true }).click();
    await expect.poll(() => queries.at(-1)?.get('limit')).toBe('25');
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await expect.poll(() => queries.at(-1)?.get('actor')).toBe('__anonymous__');
    await expect(page.getByText('Initial pipeline', { exact: true })).toBeVisible();
  });

  test(`drift investigation preserves evidence and disposition at ${width}px`, async ({ page }) => {
    // Submission must use trimmed actor/note values and update the visible lifecycle actions.
    await page.setViewportSize({ width, height: 1000 });
    const submissions = await mockDriftAlert(page);
    await page.goto('/drift?investigate=41');
    const modal = page.getByRole('dialog', { name: 'Drift alert #41', exact: true });
    await expect(modal).toBeVisible();
    await expect(modal.getByRole('cell', { name: 'age', exact: true })).toBeVisible();
    await expect(modal.getByRole('cell', { name: '0.4568', exact: true })).toBeVisible();
    await modal.getByRole('button', { name: 'Acknowledge', exact: true }).click();
    await expect(modal.getByText('Enter your name so the disposition records who made it.')).toBeVisible();
    expect(submissions).toHaveLength(0);
    const actor = modal.getByRole('textbox', { name: 'Your name', exact: true });
    await actor.fill('  Murat  ');
    await modal.getByRole('textbox', { name: 'Note', exact: true }).fill('  Investigating age  ');
    await modal.getByRole('button', { name: 'Acknowledge', exact: true }).press('Enter');
    await expect(modal.getByRole('button', { name: 'Resolve', exact: true })).toBeVisible();
    expect(submissions[0]).toEqual({ action: 'acknowledge', actor: 'Murat', note: 'Investigating age' });
    await expect(modal.getByRole('textbox', { name: 'Note', exact: true })).toHaveValue('');
    await expect(actor).toHaveValue('  Murat  ');
    await expect(modal.getByText('"Investigating age"', { exact: true })).toBeVisible();
    await modal.getByRole('button', { name: 'Resolve', exact: true }).click();
    await expect(modal.getByRole('button', { name: 'Resolve', exact: true })).toBeHidden();
    await expect(modal.getByRole('button', { name: 'Reopen', exact: true })).toBeVisible();
    await modal.getByRole('button', { name: 'Close', exact: true }).focus();
    await page.keyboard.press('Escape');
    await expect(modal).toBeHidden();
    await expect(page).not.toHaveURL(/investigate=/);
  });

  const canvasWidth = Math.max(width, 1100);
  test(`feature operations preserve edits across panel layouts at ${canvasWidth}px`, async ({ page }) => {
    // Moving between compact and expanded settings must retain operation order and typed config.
    await page.setViewportSize({ width: canvasWidth, height: 1000 });
    await mockBackend(page);
    await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
    await page.goto('/canvas');
    const search = page.getByRole('textbox', { name: 'Search nodes', exact: true });
    if (!(await search.isVisible())) {
      await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
    }
    await search.fill('Feature Generation');
    await page.getByRole('button', { name: 'Add Feature Generation node', exact: true }).click();
    await page.getByRole('button', { name: 'Arithmetic', exact: true }).click();
    const output = page.getByRole('textbox', { name: 'Output Column Name for operation 1', exact: true });
    await output.fill('total_amount');
    await expect(output).toBeFocused();
    await page.getByRole('combobox', { name: 'Method for operation 1', exact: true }).selectOption('multiply');
    await page.getByRole('button', { name: 'Collapse arithmetic operation 1', exact: true }).press('Enter');
    await expect(output).toBeHidden();
    await page.getByRole('button', { name: 'Expand arithmetic operation 1', exact: true }).press('Enter');
    await expect(output).toHaveValue('total_amount');
    await page.getByRole('button', { name: 'Date Extraction', exact: true }).click();
    await page.getByRole('checkbox', { name: /^month int$/ }).check();
    await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
    await expect(output).toHaveValue('total_amount');
    await expect(page.getByRole('combobox', { name: 'Method for operation 1', exact: true })).toHaveValue('multiply');
    await expect(page.getByRole('checkbox', { name: /^month int$/ })).toBeChecked();
    await page.getByRole('button', { name: 'Remove operation 1', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Collapse Date operation 1', exact: true })).toBeVisible();
    await expect(page.getByRole('checkbox', { name: /^month int$/ })).toBeChecked();
    await expect(page.getByRole('button', { name: 'Remove operation 2', exact: true })).toBeHidden();
  });
}
