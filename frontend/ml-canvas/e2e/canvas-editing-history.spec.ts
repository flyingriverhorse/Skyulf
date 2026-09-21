import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Render a real connected graph with deterministic backend responses. */
async function prepareCanvas(page: Page) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {} } }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    window.__skyulfTest!.graphStore.getState().setGraph([
      { id: 'a', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'MissingIndicator', columns: [] } },
      { id: 'b', type: 'custom', position: { x: 450, y: 100 }, data: { definitionType: 'MissingIndicator', columns: [] } },
    ], [{ id: 'a-b', type: 'custom', source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' }]);
  });
  await page.locator('.react-flow__controls-fitview').click();
  await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().clear());
}

/** Read live positions so keyboard undo is checked against the actual drag result. */
async function positions(page: Page) {
  return page.evaluate(() => window.__skyulfTest!.graphStore.getState().nodes.map(node => node.position));
}

test('hover, zoom, pan and repeated node selection leave history unchanged', async ({ page }) => {
  // Looking around a graph must not create edits or discard an available redo.
  await prepareCanvas(page);
  const before = await positions(page);
  for (const id of ['a', 'b', 'a', 'b']) {
    const header = page.locator(`.react-flow__node[data-id="${id}"] [title="Missing Indicator"]`);
    await header.hover();
    await header.click();
  }
  await page.locator('.react-flow__edge').hover();
  await page.locator('.react-flow__controls-zoomin').click();
  await page.locator('.react-flow__controls-zoomout').click();
  await page.mouse.move(750, 700);
  await page.mouse.down();
  await page.mouse.move(790, 750, { steps: 12 });
  await page.mouse.up();
  expect(await positions(page)).toEqual(before);
  expect(await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().pastStates.length)).toBe(0);
});

test('reopening tuned model settings does not record identical loaded defaults', async ({ page }) => {
  // Clicking a configured trainer repeatedly must not fill history with invisible copies.
  await prepareCanvas(page);
  let defaultRequests = 0;
  await page.route('**/api/pipeline/hyperparameters/*/defaults?*', route => {
    defaultRequests++;
    return route.fulfill({ json: { n_estimators: [10, 20] } });
  });
  const id = await page.evaluate(() => {
    const store = window.__skyulfTest!.graphStore;
    const id = store.getState().addNode('classification', { x: 100, y: 350 }, {
      run_mode: 'advanced', search_space: { n_estimators: [10, 20] },
    });
    store.getState().onNodesChange([{ id, type: 'select', selected: false }]);
    store.temporal.getState().clear();
    return id;
  });
  await page.locator('.react-flow__controls-fitview').click();
  const node = page.locator(`.react-flow__node[data-id="${id}"] [title="Classification"]`);
  for (let i = 0; i < 5; i++) {
    await node.click();
    await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  }
  expect(defaultRequests).toBe(0);
  expect(await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().pastStates.length)).toBe(0);
});

for (const grouped of [false, true]) {
  test(`pointer ${grouped ? 'group' : 'single-node'} drag creates one undo step and ignores edge selection`, async ({ page }) => {
    // Real React Flow pointer batches must undo the full gesture after selecting a connection.
    await prepareCanvas(page);
    await page.evaluate(grouped => {
      window.__skyulfTest!.graphStore.getState().onNodesChange(['a', 'b'].map(id => ({
        id, type: 'select', selected: grouped || id === 'a',
      })));
    }, grouped);
    const initial = await positions(page);
    const header = page.locator('.react-flow__node[data-id="a"] [title="Missing Indicator"]');
    const box = await header.boundingBox();
    expect(box).not.toBeNull();
    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.down();
    await page.mouse.move(box!.x + box!.width / 2 + 90, box!.y + box!.height / 2 + 70, { steps: 8 });
    await page.mouse.up();
    const moved = await positions(page);
    expect(moved[0]).not.toEqual(initial[0]);
    if (grouped) expect(moved[1]).not.toEqual(initial[1]);
    else expect(moved[1]).toEqual(initial[1]);
    await expect.poll(() => page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().pastStates.length)).toBe(1);
    const edge = page.locator('.react-flow__edge');
    await edge.focus();
    await edge.press('Enter');
    await expect(edge).toHaveClass(/selected/);
    await page.locator('.react-flow').locator('..').focus();
    await page.keyboard.press('Control+z');
    await expect.poll(() => positions(page)).toEqual(initial);
    await page.keyboard.press('Control+Shift+z');
    await expect.poll(() => positions(page)).toEqual(moved);
    expect(await page.evaluate(() => window.__skyulfTest!.graphStore.getState().nodes.some(node => node.dragging))).toBe(false);
  });
}

test('read-only copy remains available when editing resumes and paste is blocked until then', async ({ page }) => {
  // Tablet read-only mode and its override must guard the mounted clipboard listener.
  await prepareCanvas(page);
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().selectNode('a'));
  await page.setViewportSize({ width: 900, height: 800 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toBeVisible();
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');
  await expect(page.locator('.react-flow__node')).toHaveCount(2);
  await page.getByRole('button', { name: 'Read-only', exact: true }).click();
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+v');
  await expect(page.locator('.react-flow__node')).toHaveCount(3);
  await page.getByRole('button', { name: 'Editing', exact: true }).click();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+v');
  await expect(page.locator('.react-flow__node')).toHaveCount(3);
});

test('following an existing dataset source link does not insert another dataset', async ({ page }) => {
  // Client-side source navigation must recognize the real registered dataset wrapper.
  await prepareCanvas(page);
  const id = await page.evaluate(() => window.__skyulfTest!.graphStore.getState().addNode(
    'dataset_node', { x: 0, y: 300 }, { datasetId: 'existing-source' },
  ));
  await page.evaluate(() => {
    window.history.pushState(null, '', '/canvas?view=canvas&source_id=existing-source');
    window.dispatchEvent(new PopStateEvent('popstate'));
  });
  await expect(page).toHaveURL(/\/canvas\?view=canvas$/);
  await expect(page.locator('[data-node-definition-type="dataset_node"]')).toHaveCount(1);
  expect(await page.evaluate(() => window.__skyulfTest!.graphStore.getState().nodes.filter(node =>
    node.data.definitionType === 'dataset_node').map(node => node.id))).toEqual([id]);
});
