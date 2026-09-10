import { readFile } from 'node:fs/promises';
import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Exercise real node definitions and controls with a deterministic upstream schema. */
async function prepare(page: Page, width = 1440) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    age: { name: 'age', dtype: 'float64' },
    height: { name: 'height', dtype: 'int64' },
    species: { name: 'species', dtype: 'object' },
  } } }));
  await page.setViewportSize({ width, height: 900 });
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'light'));
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  return page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 0 });
    const replacement = state.addNode('InvalidValueReplacement', { x: 240, y: 0 });
    const casting = state.addNode('casting', { x: 480, y: 0 });
    const binning = state.addNode('BinningNode', { x: 720, y: 0 });
    state.updateNodeData(dataset, { datasetId: 'iris-demo', datasetName: 'Iris' });
    state.onConnect({ source: dataset, sourceHandle: 'data', target: replacement, targetHandle: 'in' });
    state.onConnect({ source: replacement, sourceHandle: 'out', target: casting, targetHandle: 'in' });
    state.onConnect({ source: casting, sourceHandle: 'out', target: binning, targetHandle: 'in' });
    state.onNodesChange(window.__skyulfTest!.graphStore.getState().nodes.map(node => ({
      id: node.id, type: 'select', selected: node.id === replacement,
    })));
    return { replacement, casting, binning };
  });
}

/** Switching selection must retain the values sent by the previous node. */
async function selectNode(page: Page, id: string) {
  await page.evaluate(id => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.onNodesChange(state.nodes.map(node => ({ id: node.id, type: 'select', selected: node.id === id })));
  }, id);
}

/** The responsive grid must settle before checking controls or taking a screenshot. */
async function expectBinningLayout(page: Page, wide: boolean) {
  const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
  await expect.poll(() => panel.evaluate(element => {
    element.getBoundingClientRect();
    return element.getAnimations().every(animation => animation.playState === 'finished');
  })).toBe(true);
  const grid = page.getByRole('tabpanel', { name: 'Settings', exact: true }).locator('.grid.gap-4');
  await expect(grid).toHaveClass(wide ? /\bgrid-cols-2\b/ : /\bgrid-cols-1\b/);
  await expect.poll(() => grid.evaluate(element => element.getBoundingClientRect().width > 450)).toBe(wide);
}

for (const width of [1440, 1100]) {
  test(`replacement, casting and binning retain controlled values and Preview payloads at ${width}px`, async ({ page }) => {
    // Real shared column controls, local custom-bin drafts and graph conversion must stay connected.
    let submitted: { nodes: { node_id: string; params: Record<string, unknown> }[] } | undefined;
    const ids = await prepare(page, width);
    await page.route('**/api/pipeline/preview?*', route => {
      submitted = route.request().postDataJSON() as typeof submitted;
      return route.fulfill({ json: { pipeline_id: 'ccn9-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
    });
    const settings = page.getByRole('tabpanel', { name: 'Settings', exact: true });
    await settings.getByRole('checkbox', { name: 'age', exact: true }).check();
    await expect(settings.getByRole('checkbox', { name: 'species', exact: true })).toHaveCount(0);
    const mode = settings.getByRole('combobox', { name: 'Replacement Mode' });
    await mode.selectOption('custom_range');
    await settings.getByRole('spinbutton', { name: 'Min Value' }).fill('0');
    await settings.getByRole('spinbutton', { name: 'Max Value' }).fill('120');
    await mode.selectOption('negative_to_nan');
    await expect(settings.getByRole('spinbutton', { name: 'Min Value' })).toHaveCount(0);
    await mode.selectOption('custom_range');
    await expect(settings.getByRole('spinbutton', { name: 'Min Value' })).toHaveValue('0');
    await selectNode(page, ids.casting);
    await settings.getByRole('button', { name: 'Add Casting Rule' }).click();
    await settings.getByRole('combobox', { name: 'Data type for age', exact: true }).selectOption('int');
    await selectNode(page, ids.binning);
    await settings.getByRole('checkbox', { name: 'height', exact: true }).check();
    await settings.getByRole('combobox', { name: 'Binning Strategy' }).selectOption('custom');
    const edges = settings.getByRole('textbox', { name: 'Bin edges for height' });
    await edges.fill('40, 0, 20');
    await edges.press('Tab');
    await settings.getByRole('combobox', { name: 'Label Format' }).selectOption('range');
    await settings.getByRole('spinbutton', { name: 'Precision (Decimals)', exact: true }).fill('0');
    await expectBinningLayout(page, false);
    await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
    await expectBinningLayout(page, true);
    await expect(settings.getByRole('checkbox', { name: 'height', exact: true })).toBeChecked();
    await expect(settings.getByRole('spinbutton', { name: 'Precision (Decimals)', exact: true })).toHaveValue('0');
    await page.screenshot({ path: `test-results/ccn9-binning-expanded-${width}.png`, animations: 'disabled' });
    await page.getByRole('button', { name: 'Collapse settings panel', exact: true }).click();
    await expectBinningLayout(page, false);
    await selectNode(page, ids.casting);
    await expect(settings.getByRole('combobox', { name: 'Data type for age', exact: true })).toHaveValue('int');
    await selectNode(page, ids.binning);
    await expect(edges).toHaveValue('0, 20, 40');
    await page.getByRole('button', { name: 'Preview data', exact: true }).click();
    await expect.poll(() => submitted?.nodes.length).toBe(4);
    const params = (id: string) => submitted!.nodes.find(node => node.node_id === id)!.params;
    expect(params(ids.replacement)).toMatchObject({ columns: ['age'], mode: 'custom_range', min_value: 0, max_value: 120 });
    expect(params(ids.casting)).toMatchObject({ column_types: { age: 'int' } });
    expect(params(ids.binning)).toMatchObject({ columns: ['height'], strategy: 'custom', custom_bins: { height: [0, 20, 40] }, label_format: 'range', precision: 0 });
  });
}

test('canvas PNG and SVG exports download real rendered nodes', async ({ page }) => {
  // Verify browser-generated files, not only calls to the image export library.
  await prepare(page);
  await expect(page.locator('.react-flow__node')).toHaveCount(4);
  for (const kind of ['png', 'svg'] as const) {
    await page.getByRole('button', { name: 'More canvas tools', exact: true }).click();
    const pending = page.waitForEvent('download');
    await page.getByRole('menuitem', { name: `Export ${kind.toUpperCase()}`, exact: true }).click();
    const download = await pending;
    expect(download.suggestedFilename()).toBe(`skyulf-canvas.${kind}`);
    expect(await download.failure()).toBeNull();
    const contents = await readFile((await download.path())!);
    if (kind === 'png') {
      expect(contents.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
      expect(contents.readUInt32BE(16)).toBeGreaterThan(0);
      expect(contents.readUInt32BE(20)).toBeGreaterThan(0);
    } else {
      const svg = contents.toString('utf8');
      expect(svg).toContain('<svg');
      expect(svg).toContain('Binning / Discretization');
      expect(svg).toContain('Invalid Value Replacement');
    }
  }
});
