import { test, expect, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Seed one labeled connection so pointer and keyboard behavior have stable targets. */
async function seedConnection(page: Page) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([
      { id: 'input-technical-id', type: 'custom', position: { x: 0, y: 0 }, data: { definitionType: 'MissingIndicator', label: 'Prepare data', columns: [] } },
      { id: 'output-technical-id', type: 'custom', position: { x: 450, y: 100 }, data: { definitionType: 'MissingIndicator', label: 'Fill gaps', columns: [] } },
    ], [{ id: 'connection', type: 'custom', source: 'input-technical-id', target: 'output-technical-id', sourceHandle: 'out', targetHandle: 'in' }]);
  });
  await page.locator('.react-flow__controls-fitview').click();
}

test('connection controls appear on interaction, use node names, and retain undo', async ({ page }) => {
  // A clean resting canvas must still offer reliable pointer and keyboard deletion.
  await seedConnection(page);
  const edge = page.getByRole('group', { name: 'Connection from Prepare data to Fill gaps', exact: true });
  await expect(edge).toBeVisible();
  const remove = page.getByRole('button', { name: 'Remove connection from Prepare data to Fill gaps', exact: true });
  await expect(remove).toHaveCSS('opacity', '0');
  await edge.locator('.react-flow__edge-interaction').first().hover();
  await expect(remove).toHaveCSS('opacity', '1');
  const tooltip = page.getByRole('tooltip', { name: 'Prepare data → Fill gaps', exact: true });
  await expect(tooltip).toBeVisible();
  // Short endpoint names should occupy one small line, with no fixed-width card.
  await expect.poll(async () => tooltip.evaluate(element => {
    const box = element.getBoundingClientRect();
    return box.width < 220 && box.height < 30;
  })).toBe(true);
  await tooltip.hover();
  await expect(tooltip).toBeVisible();
  await page.mouse.move(70, 60);
  await expect(tooltip).toHaveCount(0);
  await edge.locator('.react-flow__edge-interaction').first().hover();
  await expect(tooltip).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(tooltip).toHaveCount(0);
  await page.mouse.move(70, 60);
  await expect(remove).toHaveCSS('opacity', '0');
  await edge.locator('.react-flow__edge-interaction').first().hover();
  await expect(tooltip).toBeVisible();
  await remove.hover();
  await expect(remove).toHaveCSS('opacity', '1');
  await remove.click();
  await expect(edge).toHaveCount(0);
  await page.keyboard.press('Control+z');
  await expect(edge).toBeVisible();
  await page.mouse.move(70, 60);
  await expect(remove).toHaveCSS('opacity', '0');
  await edge.focus();
  await expect(remove).toHaveCSS('opacity', '1');
  await edge.press('Enter');
  await expect(remove).toHaveCSS('opacity', '1');
  for (let i = 0; i < 12 && !(await remove.evaluate(element => element === document.activeElement)); i++) {
    await page.keyboard.press('Tab');
  }
  await expect(remove).toBeFocused();
  await remove.press('Space');
  await expect(edge).toHaveCount(0);
  await page.keyboard.press('Control+z');
  await expect(edge).toBeVisible();
});

test('node IDs open from the header without accumulating across node selections', async ({ page, context }) => {
  // Troubleshooting identifiers must remain available without filling the settings header.
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await seedConnection(page);
  await page.locator('.react-flow__node[data-id="output-technical-id"]').click();
  const settings = page.getByRole('complementary', { name: 'Node settings' });
  await expect(settings.getByText('output-technical-id', { exact: true })).not.toBeVisible();
  const info = settings.getByRole('button', { name: 'Node information', exact: true });
  const dialog = page.getByRole('dialog', { name: 'Node information', exact: true });
  await info.press('Enter');
  await expect(dialog.getByText('output-technical-id', { exact: true })).toBeVisible();
  await dialog.getByRole('button', { name: 'Copy node ID', exact: true }).press('Enter');
  await expect(dialog.getByRole('status')).toHaveText('Node ID copied.');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('output-technical-id');
  await page.keyboard.press('Escape');
  await expect(info).toBeFocused();
  await expect(settings.getByText('output-technical-id', { exact: true })).not.toBeVisible();
  await info.press('Space');
  await page.evaluate(() => { navigator.clipboard.writeText = () => Promise.reject(new Error('Clipboard denied')); });
  await dialog.getByRole('button', { name: 'Copy node ID', exact: true }).click();
  await expect(dialog.getByRole('status')).toHaveText('Could not copy. Select the ID to copy it manually.');
  await expect(dialog.getByText('output-technical-id', { exact: true })).toBeVisible();
  for (const nodeId of ['input-technical-id', 'output-technical-id', 'input-technical-id', 'output-technical-id']) {
    await page.evaluate(id => {
      const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
      (state.selectNode as (id: string) => void)(id);
    }, nodeId);
    await expect(dialog).toHaveCount(0);
    await expect(info).toHaveCount(1);
    await expect(settings.locator('details')).toHaveCount(0);
    await info.click();
    await expect(dialog.getByText(nodeId, { exact: true })).toBeVisible();
    await expect(dialog.getByText(/Node type/)).toHaveCount(0);
  }
  await page.keyboard.press('Escape');
  await expect(settings.getByText('output-technical-id', { exact: true })).toHaveCount(0);
});

test('read-only connections stay inspectable without exposing deletion', async ({ page }) => {
  // Inspecting an edge in read-only mode must not provide a mutation through its custom button.
  await seedConnection(page);
  await page.setViewportSize({ width: 900, height: 800 });
  const edge = page.getByRole('group', { name: 'Connection from Prepare data to Fill gaps', exact: true });
  await edge.focus();
  await edge.press('Enter');
  await expect(page.getByRole('button', { name: /Remove connection/ })).toHaveCount(0);
  await edge.press('Delete');
  await expect(edge).toBeVisible();
});

test('connected nodes can still be copied and pasted with editable connections', async ({ page }) => {
  // Presentation-only edge state must remain cloneable and must not lock pasted connections.
  await seedConnection(page);
  await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    (state.onNodesChange as (changes: unknown[]) => void)(['input-technical-id', 'output-technical-id'].map(id => ({ type: 'select', id, selected: true })));
  });
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+c');
  await page.keyboard.press('Control+v');
  await expect(page.locator('.react-flow__node')).toHaveCount(4);
  await expect(page.locator('.react-flow__edge')).toHaveCount(2);
  await expect(page.getByRole('button', { name: /Remove connection/ })).toHaveCount(2);
});

test('long connection names float above nearby nodes and clear after inspection', async ({ page }) => {
  // Edge labels must not be clipped by node stacking or remain as clutter after selection.
  await seedConnection(page);
  const sourceName = 'Prepare customer purchase history and account details for the complete training dataset';
  const targetName = 'Generate additional features and fill all missing customer values before model training';
  await page.evaluate(({ sourceName, targetName }) => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    (state.updateNodeData as (id: string, data: unknown) => void)('input-technical-id', { label: sourceName });
    (state.updateNodeData as (id: string, data: unknown) => void)('output-technical-id', { label: targetName });
    (state.onNodesChange as (changes: unknown[]) => void)([{ id: 'output-technical-id', type: 'position', position: { x: 295, y: 0 } }]);
  }, { sourceName, targetName });
  await page.locator('.react-flow__controls-fitview').click();
  const edge = page.getByRole('group', { name: `Connection from ${sourceName} to ${targetName}`, exact: true });
  await edge.focus();
  const tooltip = page.getByRole('tooltip', { name: `${sourceName} → ${targetName}`, exact: true });
  await expect(tooltip).toBeVisible();
  await expect(page.locator('.react-flow__edgelabel-renderer [role="tooltip"]')).toHaveCount(0);
  await expect(tooltip.getByText('From', { exact: true })).toHaveCount(0);
  await expect(tooltip.getByText('To', { exact: true })).toHaveCount(0);
  await expect(tooltip.getByText(sourceName, { exact: true })).toBeVisible();
  await expect(tooltip.getByText(targetName, { exact: true })).toBeVisible();
  await expect.poll(async () => tooltip.evaluate(element => {
    const box = element.getBoundingClientRect();
    const front = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
    return element.contains(front) && box.width <= 280 && box.x >= 8 && box.right <= window.innerWidth - 8;
  })).toBe(true);
  await page.screenshot({ path: 'test-results/connection-hover-card.png' });
  await page.setViewportSize({ width: 900, height: 800 });
  await edge.focus();
  await expect.poll(async () => tooltip.evaluate(element => {
    const box = element.getBoundingClientRect();
    return box.x >= 8 && box.right <= window.innerWidth - 8 && box.y >= 8 && box.bottom <= window.innerHeight - 8;
  })).toBe(true);
  await edge.press('Enter');
  await page.mouse.move(70, 60);
  await page.locator('.react-flow').locator('..').focus();
  await expect(edge).toHaveClass(/selected/);
  await expect(tooltip).toHaveCount(0);
});

for (const theme of ['light', 'dark'] as const) {
  test(`branch names and merge winners remain readable on a compact ${theme} canvas`, async ({ page }) => {
    // Removing idle controls must preserve the information that distinguishes experiment branches.
    await page.addInitScript(value => localStorage.setItem('skyulf-theme', value), theme);
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await seedConnection(page);
    await page.setViewportSize({ width: 1100, height: 800 });
    await page.evaluate(() => {
      const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
      (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([
        ...['source-a', 'source-b'].map((id, index) => ({ id, type: 'custom', position: { x: 0, y: index * 350 }, data: { definitionType: 'MissingIndicator', label: 'Features', columns: [] } })),
        ...['model-a', 'model-b'].map((id, index) => ({ id, type: 'custom', position: { x: 450, y: index * 350 }, data: { definitionType: 'classification', label: 'Model', model_type: 'random_forest_classifier', target_column: 'outcome' } })),
      ], [
        { id: 'winner', type: 'custom', source: 'source-a', target: 'model-a', sourceHandle: 'out', targetHandle: 'in' },
        { id: 'other-input', type: 'custom', source: 'source-b', target: 'model-a', sourceHandle: 'out', targetHandle: 'in' },
        { id: 'branch-b', type: 'custom', source: 'source-b', target: 'model-b', sourceHandle: 'out', targetHandle: 'in' },
      ]);
      (state.setExecutionResult as (result: unknown) => void)({ pipeline_id: 'clutter', status: 'success', node_results: {}, recommendations: [], merge_warnings: [{ kind: 'sibling_fan_in', node_id: 'model-a', winner_input: 'source-a', inputs: ['source-a', 'source-b'], overlap_columns: ['age'] }] });
    });
    await page.getByRole('button', { name: 'Collapse results panel', exact: true }).click();
    await page.locator('.react-flow__controls-fitview').click();
    const edge = page.getByRole('group', { name: 'Connection from Features (1) to Model (1)', exact: true });
    await expect(edge.locator('.react-flow__edge-path')).toHaveCSS('stroke-width', '4px');
    const labels = page.locator('.react-flow__edgelabel-renderer');
    await expect(labels.getByText('Wins merge', { exact: true })).toBeVisible();
    await expect(labels.getByText(/Path A/).first()).toBeVisible();
    await expect(labels.getByText(/Path B/).first()).toBeVisible();
    await edge.focus();
    const tooltip = page.getByRole('tooltip', { name: 'Features (1) → Model (1)', exact: true });
    await expect(tooltip).toBeVisible();
    await expect.poll(async () => {
      const tooltipBox = await tooltip.boundingBox();
      const winnerBox = await labels.getByText('Wins merge', { exact: true }).boundingBox();
      return tooltipBox && winnerBox && tooltipBox.y >= winnerBox.y + winnerBox.height;
    }).toBeTruthy();
    await page.screenshot({ path: `test-results/canvas-clutter-${theme}.png` });
  });
}
