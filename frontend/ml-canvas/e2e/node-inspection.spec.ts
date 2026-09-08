import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';
import type { InspectionSide, InspectionTable } from '../src/core/types/nodeInspection';

/** Seed a runnable graph while preserving real selection, validation and request handlers. */
async function openInspection(page: Page, dark = false) {
  await page.setViewportSize({ width: dark ? 1100 : 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: { age: { name: 'age', dtype: 'float64', missing_count: 0, missing_ratio: 0, unique_count: 10 } }, row_count: 150,
  } }));
  await page.route('**/api/pipeline/schema-preview', route => route.fulfill({ json: {
    pipeline_id: 'schema', predicted_schemas: { scale: { columns: ['age'], dtypes: { age: 'float64' } } }, broken_references: [],
  } }));
  await page.goto('/canvas');
  await page.waitForFunction(() => Boolean(window.__skyulfTest));
  await page.evaluate(isDark => {
    document.documentElement.classList.toggle('dark', isDark);
    window.__skyulfTest!.graphStore.getState().setGraph([
      { id: 'ds', type: 'custom', position: { x: 80, y: 140 }, data: { definitionType: 'dataset_node', datasetId: 'iris-demo' } },
      { id: 'scale', type: 'custom', position: { x: 380, y: 140 }, selected: true,
        data: { definitionType: 'scale_numeric_features', columns: ['age'], method: 'standard' } },
    ], [{ id: 'edge', source: 'ds', sourceHandle: 'data', target: 'scale', targetHandle: 'in' }]);
  }, dark);
}

/** Return the wire representation of a bounded table while keeping its real total. */
function table(value: number, split: string | null = null): InspectionTable {
  return { port: 'data', split, row_count: 150, column_count: 1,
    columns: [{ name: 'age', dtype: 'float64' }], rows: [{ age: value }], truncated: true };
}

/** Distinct branches must stay distinct even though they inspect the same node. */
function preview(run: string) {
  const side = (port: 'input' | 'output', ...tables: InspectionTable[]): InspectionSide => ({
    status: 'available', reason: null, tables: tables.map(item => ({ ...item, port })),
  });
  return { pipeline_id: 'p', run_id: run, status: 'success', node_results: {}, preview_data: [], recommendations: [],
    node_inspections: [
      { node_id: 'ds', branch_id: 'a', branch_label: 'Model A', path_id: 'source-path', path_label: 'Dataset',
        input: { status: 'unavailable', reason: 'Dataset sources have no input.', tables: [] }, output: side('output', table(100)) },
      { node_id: 'scale', branch_id: 'a', branch_label: 'Model A', path_id: 'scale-a', path_label: 'Dataset → Scaler A', input: side('input', table(100), table(150, 'test')),
        output: side('output', table(0.5), table(0.75, 'test')) },
      { node_id: 'scale', branch_id: 'b', branch_label: 'Model B', path_id: 'scale-b', path_label: 'Dataset → Scaler B', input: side('input', table(200)), output: side('output', table(0.25)) },
      { node_id: 'ds', branch_id: 'b', branch_label: 'Model B', path_id: 'source-path', path_label: 'Dataset',
        input: { status: 'unavailable', reason: 'Dataset sources have no input.', tables: [] }, output: side('output', table(100)) },
    ],
  };
}

for (const dark of [false, true]) {
  test(`selected-node samples, branches and stale results (${dark ? 'dark laptop' : 'light desktop'})`, async ({ page }) => {
    // The visible panel must reflect measured data and preserve keyboard access at both widths.
    await openInspection(page, dark);
    const requestedModes: (string | null)[] = [];
    await page.route('**/api/pipeline/preview?*', route => {
      const params = new URL(route.request().url()).searchParams;
      requestedModes.push(params.get('inspect_all'));
      expect(params.has('inspect_node_id')).toBe(false);
      return route.fulfill({ json: preview(`run-${requestedModes.length}`) });
    });
    const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
    await panel.getByRole('tab', { name: 'Settings', exact: true }).press('End');
    const outputTab = panel.getByRole('tab', { name: 'Output', exact: true });
    await expect(outputTab).toBeFocused();
    await expect(panel.getByRole('table', { name: 'Predicted output schema', exact: true })).toBeVisible();
    await expect(panel.getByRole('button', { name: /preview/i })).toHaveCount(0);
    const run = page.getByRole('button', { name: 'Preview data', exact: true });
    await run.press('Enter');
    await expect(panel.getByRole('table', { name: 'Measured output sample', exact: true })).toBeVisible();
    expect(requestedModes).toEqual(['true']);
    await page.getByRole('button', { name: 'Close preview results', exact: true }).click();
    await expect(panel.getByText('150 rows × 1 columns', { exact: true })).toBeVisible();
    await expect(panel.getByText(/Preview uses up to 1,000 source rows/)).toBeVisible();
    await panel.getByRole('combobox', { name: 'Port and split', exact: true }).selectOption('1');
    await expect(panel.getByRole('cell', { name: '0.75', exact: true })).toBeVisible();
    await outputTab.press('ArrowLeft');
    await expect(panel.getByRole('combobox', { name: 'Port and split', exact: true })).toHaveValue('1');
    await expect(panel.getByRole('cell', { name: '150', exact: true })).toBeVisible();
    await panel.getByRole('tab', { name: 'Input', exact: true }).press('ArrowRight');
    await panel.getByRole('combobox', { name: 'Data path', exact: true }).selectOption('b');
    await expect(panel.getByRole('cell', { name: '0.25', exact: true })).toBeVisible();
    await outputTab.press('ArrowLeft');
    await expect(panel.getByRole('tab', { name: 'Input', exact: true })).toBeFocused();
    await expect(panel.getByRole('combobox', { name: 'Data path', exact: true })).toHaveValue('b');
    await expect(panel.getByRole('cell', { name: '200', exact: true })).toBeVisible();
    await page.evaluate(() => window.__skyulfTest!.graphStore.getState().onNodesChange([
      { id: 'scale', type: 'position', position: { x: 450, y: 180 } },
    ]));
    await expect(panel.getByText('Stale preview.', { exact: true })).toHaveCount(0);
    await panel.getByRole('tab', { name: 'Settings', exact: true }).click();
    await panel.getByRole('combobox', { name: 'Scaling Method', exact: true }).selectOption('minmax');
    await outputTab.click();
    await expect(panel.getByText('Stale preview.', { exact: true })).toBeVisible();
    await expect(panel.getByRole('status')).toContainText('Run Preview data in the toolbar');
    await run.click();
    await expect(panel.getByText('Preview run: run-2', { exact: true })).toBeVisible();
    await expect(panel.getByText('Stale preview.', { exact: true })).toHaveCount(0);
    const sample = await panel.getByRole('region', { name: 'Measured output sample scroll area', exact: true }).boundingBox();
    const bounds = await panel.boundingBox();
    expect(sample!.x + sample!.width).toBeLessThanOrEqual(bounds!.x + bounds!.width);
    const scan = await new AxeBuilder({ page }).include('[aria-label="Node settings"]')
      .withRules(['button-name', 'label', 'select-name', 'aria-valid-attr-value', 'aria-required-children']).analyze();
    expect(scan.violations).toEqual([]);
    await page.screenshot({ path: `test-results/node-inspection-${dark ? 'dark' : 'light'}.png` });
  });
}

test('selection changes during preview do not mix nodes and request failures remain visible', async ({ page }) => {
  // One slow toolbar request fills every node without allowing an inspector to submit more work.
  await openInspection(page);
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  let requests = 0;
  await page.route('**/api/pipeline/preview?*', async route => {
    requests += 1;
    if (requests === 1) { await gate; await route.fulfill({ json: preview('slow-run') }); }
    else await route.fulfill({ status: 500, json: { detail: 'Preview failed' } });
  });
  const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
  await panel.getByRole('tab', { name: 'Output', exact: true }).click();
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect(panel.getByText('Running preview…', { exact: true })).toBeVisible();
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().selectNode('ds'));
  await expect(page.getByRole('button', { name: 'Previewing data...', exact: true })).toBeDisabled();
  await expect(panel.getByRole('button', { name: /preview/i })).toHaveCount(0);
  release();
  await expect(panel.getByText('Running preview…', { exact: true })).toHaveCount(0);
  await expect(panel.getByRole('table', { name: 'Measured output sample', exact: true })).toBeVisible();
  await expect(panel.getByRole('cell', { name: '100', exact: true })).toBeVisible();
  await expect(panel.getByText('Preview run: slow-run', { exact: true })).toBeVisible();
  expect(requests).toBe(1);
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().selectNode('scale'));
  await expect(panel.getByRole('table', { name: 'Measured output sample', exact: true })).toBeVisible();
  await expect(panel.getByRole('cell', { name: '0.5', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText(/500|failed/i);
  await expect(page.getByRole('button', { name: 'Preview data', exact: true })).toBeEnabled();
  expect(requests).toBe(2);
});

test('one preview groups repeated model paths locally and browsing preserves results visibility', async ({ page }) => {
  // Shared sources have one result; differing paths remain available only at the affected node.
  await openInspection(page);
  let requests = 0;
  await page.route('**/api/pipeline/preview?*', route => {
    expect(new URL(route.request().url()).searchParams.get('inspect_all')).toBe('true');
    requests += 1;
    const result = preview(`run-${requests}`);
    result.node_inspections.push({ ...result.node_inspections[1]!, branch_id: 'c', branch_label: 'Model C' });
    return route.fulfill({ json: result });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  const closeResults = page.getByRole('button', { name: 'Close preview results', exact: true });
  await expect(closeResults).toBeVisible();
  const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
  await panel.getByRole('tab', { name: 'Output', exact: true }).click();
  await expect(panel.getByRole('cell', { name: '0.5', exact: true })).toBeVisible();
  await expect(panel.getByRole('combobox', { name: 'Data path', exact: true }).getByRole('option')).toHaveCount(2);
  await expect(panel.getByRole('combobox', { name: 'Data path', exact: true })).toHaveValue('a');
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().selectNode('ds'));
  await expect(panel.getByRole('cell', { name: '100', exact: true })).toBeVisible();
  await expect(panel.getByText('Preview run: run-1', { exact: true })).toBeVisible();
  await expect(panel.getByRole('combobox', { name: /Data path|Branch/ })).toHaveCount(0);
  await expect(panel.getByRole('button', { name: /preview/i })).toHaveCount(0);
  expect(requests).toBe(1);
  await page.getByRole('button', { name: 'Collapse results panel', exact: true }).click();
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().selectNode('scale'));
  await panel.getByRole('tab', { name: 'Input', exact: true }).click();
  await expect(panel.getByText('Preview run: run-1', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Expand results panel', exact: true })).toBeVisible();
  await closeResults.click();
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().selectNode('ds'));
  await panel.getByRole('tab', { name: 'Output', exact: true }).click();
  await expect(panel.getByText('Preview run: run-1', { exact: true })).toBeVisible();
  await expect(closeResults).toHaveCount(0);
  expect(requests).toBe(1);
});

test('captured node errors remain available when global results are closed', async ({ page }) => {
  // HTTP success can carry a failed pipeline, so the selected branch must retain its diagnostic.
  await openInspection(page);
  await page.route('**/api/pipeline/preview?*', route => {
    const result = preview('failed-run');
    result.status = 'failed';
    result.node_inspections[1]!.output = { status: 'error',
      reason: "Node execution failed: could not convert string to float: 'bad'", tables: [] };
    return route.fulfill({ json: result });
  });
  const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
  await panel.getByRole('tab', { name: 'Output', exact: true }).click();
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText("could not convert string to float: 'bad'");
  await page.getByRole('button', { name: 'Close preview results', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText("could not convert string to float: 'bad'");
  await panel.getByRole('combobox', { name: 'Data path', exact: true }).selectOption('b');
  await expect(panel.getByRole('alert')).toHaveCount(0);
  await expect(panel.getByRole('cell', { name: '0.25', exact: true })).toBeVisible();
});
