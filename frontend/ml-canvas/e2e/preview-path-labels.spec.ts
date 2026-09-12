import { test, expect, type Page } from '@playwright/test';
import type { PipelineConfigModel } from '../src/core/api/client';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

/** Keep the real graph and toolbar while excluding backend execution from this UI regression. */
async function openPreviewBranches(page: Page, width: number) {
  await page.setViewportSize({ width, height: 1000 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: Object.fromEntries(['feature', 'outcome'].map(name => [name, {
      name, dtype: 'float64', missing_count: 0, missing_ratio: 0, unique_count: 2,
    }])), row_count: 150,
  } }));
  await page.route('**/api/pipeline/schema-preview', route => route.fulfill({ json: {
    pipeline_id: 'path-schema', predicted_schemas: {}, broken_references: [],
  } }));
  await page.goto('/canvas');
  await page.waitForFunction(() => Boolean(window.__skyulfTest));
  await page.evaluate(() => {
    window.__skyulfTest!.graphStore.getState().setGraph([
      { id: 'ds', type: 'custom', position: { x: 60, y: 230 },
        data: { definitionType: 'dataset_node', datasetId: 'iris-demo' } },
      { id: 'impute', type: 'custom', position: { x: 420, y: 80 },
        data: { definitionType: 'imputation_node',
          columns: ['feature'], method: 'simple', strategy: 'mean' } },
      { id: 'resample', type: 'custom', position: { x: 420, y: 390 },
        data: { definitionType: 'ResamplingNode', type: 'oversampling',
          method: 'smote_tomek', target_column: 'outcome', k_neighbors: 1,
          sampling_strategy: 'auto', random_state: 42 } },
      { id: 'inspect', type: 'custom', position: { x: 780, y: 80 },
        data: { definitionType: 'data_preview' } },
    ], [
      { id: 'ds-impute', source: 'ds', sourceHandle: 'data', target: 'impute', targetHandle: 'in' },
      { id: 'ds-resample', source: 'ds', sourceHandle: 'data', target: 'resample', targetHandle: 'in' },
      { id: 'impute-inspect', source: 'impute', sourceHandle: 'out', target: 'inspect', targetHandle: 'in' },
    ]);
  });
  await page.locator('.react-flow__controls-fitview').click();
}

/** Derive response paths from submitted terminal order, independently of Canvas label state. */
function previewForRequest(config: PipelineConfigModel) {
  const consumed = new Set(config.nodes.flatMap(node => node.inputs));
  const leaves = config.nodes.filter(node => !consumed.has(node.node_id));
  const branchPreviews = Object.fromEntries(leaves.map((node, index) => [
    `Path ${String.fromCharCode(65 + index)} · ${node.params._display_name}`,
    [{ result: `${node.node_id}-output` }],
  ]));
  return {
    pipeline_id: 'preview-path-labels', status: 'success', node_results: {},
    preview_data: [], branch_previews: branchPreviews, recommendations: [],
  };
}

for (const width of [1440, 390]) {
  test(`Canvas paths match preview branches beside an inspection sink at ${width}px`, async ({ page }) => {
    // An inspection-only sink must not shift another branch's letter or displayed rows.
    await openPreviewBranches(page, width);
    let submitted: PipelineConfigModel | undefined;
    await page.route('**/api/pipeline/preview?*', route => {
      submitted = route.request().postDataJSON() as PipelineConfigModel;
      return route.fulfill({ json: previewForRequest(submitted) });
    });
    const labels = page.locator('.react-flow__edgelabel-renderer');
    await expect(labels.getByText('Path A · Imputation', { exact: true })).toBeVisible();
    await expect(labels.getByText('Path B · Resampling', { exact: true })).toBeVisible();
    await expect(labels.getByText('Path A · Imputation', { exact: true })).toBeInViewport();
    await expect(labels.getByText('Path B · Resampling', { exact: true })).toBeInViewport();
    await expect(labels.getByText(/Resampling Node/)).toHaveCount(0);

    if (width < 768) await page.getByRole('button', { name: 'Read-only', exact: true }).click();
    await page.getByRole('button', { name: 'Preview data', exact: true }).click();

    await expect.poll(() => submitted?.nodes.map(node => node.node_id))
      .toEqual(['ds', 'impute', 'resample']);
    expect(submitted?.nodes.some(node => node.inputs.includes('inspect'))).toBe(false);
    const results = page.getByRole('region', { name: 'Preview results', exact: true });
    await expect(results.getByRole('button', { name: 'Path A · Imputation', exact: true })).toBeVisible();
    await expect(results.getByRole('button', { name: 'Path B · Resampling', exact: true })).toBeVisible();
    await expect(results.getByRole('button', { name: /Resampling Node/ })).toHaveCount(0);
    await expect(results.getByRole('cell', { name: 'impute-output', exact: true })).toBeVisible();
    await expect(results.getByRole('cell', { name: 'resample-output', exact: true })).toHaveCount(0);
    await results.getByRole('button', { name: 'Path B · Resampling', exact: true }).click();
    await expect(results.getByRole('cell', { name: 'resample-output', exact: true })).toBeVisible();
    await expect(results.getByRole('cell', { name: 'impute-output', exact: true })).toHaveCount(0);
    await results.getByRole('button', { name: 'Path A · Imputation', exact: true }).click();
    await expect(results.getByRole('cell', { name: 'impute-output', exact: true })).toBeVisible();
  });
}
