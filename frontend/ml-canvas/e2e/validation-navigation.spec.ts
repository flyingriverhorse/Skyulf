import { test, expect, type Page } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

/** Wait for the reveal animation and verify the node clears both toolbar and results. */
async function expectNodeRevealed(page: Page) {
  await expect.poll(async () => {
    const node = await page.locator('.react-flow__node.selected').boundingBox();
    const canvas = await page.locator('.react-flow').boundingBox();
    const results = await page.getByRole('region', { name: 'Preview results', exact: true }).boundingBox();
    return !!node && !!canvas && !!results && node.x >= canvas.x + 60 && node.x + node.width <= canvas.x + canvas.width - 15 && node.y >= canvas.y + 60 && node.y + node.height <= results.y - 10;
  }).toBe(true);
}

test.beforeEach(async ({ page }) => {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {}, row_count: 150 } }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
});

test('an issue reveals a distant model, opens its tab, and retains typing focus after correction', async ({ page }) => {
  // Navigation must restore the usable canvas and reach a field hidden by another tab.
  await page.getByRole('button', { name: 'Add Classification node', exact: true }).click();
  await page.getByRole('button', { name: 'Hyperparameters', exact: true }).click();
  await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    const nodes = state.nodes as { id: string }[];
    (state.onNodesChange as (changes: unknown[]) => void)([{ type: 'position', id: nodes[0].id, position: { x: 3000, y: 2000 } }]);
    (state.setExecutionResult as (result: unknown) => void)({ pipeline_id: 'validation', status: 'success', node_results: {}, preview_data: [{ outcome: 'retained-row' }], recommendations: [] });
  });
  await page.getByRole('tab', { name: /Issues/ }).click();
  await page.getByRole('button', { name: 'Maximize results panel', exact: true }).click();
  await page.getByRole('button', { name: /Configuration Classification.*Target column is required/i }).press('Enter');
  const target = page.getByRole('textbox', { name: 'Target Column', exact: true });
  await expect(target).toBeFocused();
  await expect(target).toHaveAttribute('aria-invalid', 'true');
  await expect(target).toHaveAccessibleDescription(/Target column is required/);
  await expect(page.getByRole('button', { name: 'Maximize results panel', exact: true })).toBeVisible();
  await expectNodeRevealed(page);
  await target.fill('outcome');
  await expect(target).toBeFocused();
  await expect(target).not.toHaveAttribute('aria-invalid');
  await page.getByRole('tab', { name: 'Data', exact: true }).click();
  await expect(page.getByText('retained-row')).toBeVisible();
});

test('dataset navigation exits upload and general connection issues focus their explanation', async ({ page }) => {
  // Both a field error and a graph error need useful destinations without guessing a control.
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  await page.getByRole('button', { name: 'New Upload', exact: true }).click();
  await page.getByRole('button', { name: /Configuration Dataset.*Dataset is required/i }).click();
  const dataset = page.getByRole('combobox', { name: 'Select Dataset', exact: true });
  await expect(dataset).toBeFocused();
  await dataset.selectOption('iris-demo');
  await expect(dataset).toBeFocused();
  await expect(dataset).not.toHaveAttribute('aria-invalid');
  await page.getByRole('button', { name: /Connection Dataset.*downstream/i }).press('Space');
  await expect(page.getByRole('group', { name: 'Validation issue', exact: true })).toBeFocused();
  await expectNodeRevealed(page);
  await page.screenshot({ path: 'test-results/validation-navigation.png' });
});

test('read-only issues reveal the node without enabling editing', async ({ page }) => {
  // Browsing an error on a narrow viewport must preserve the user's editing mode.
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  await page.setViewportSize({ width: 900, height: 800 });
  await page.getByRole('button', { name: /Configuration Dataset.*Dataset is required/i }).press('Enter');
  await expect(page.locator('.react-flow').locator('..')).toBeFocused();
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('complementary', { name: 'Node settings' })).toHaveCount(0);
});

test('a closed settings panel opens from an issue and closing it does not replay old navigation', async ({ page }) => {
  // Mounting under StrictMode must keep a fresh request and discard it on real dismissal.
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  await page.getByRole('button', { name: /Configuration Dataset.*Dataset is required/i }).press('Enter');
  await expect(page.getByRole('combobox', { name: 'Select Dataset', exact: true })).toBeFocused();
  await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  await page.locator('.react-flow__node').click();
  await expect(page.getByRole('group', { name: 'Validation issue' })).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: 'Select Dataset', exact: true })).not.toBeFocused();
});

test('revealed preprocessing sections stay scoped to their node on a compact dark canvas', async ({ page }) => {
  // A local validation reveal must not expose irrelevant fields on another node of the same type.
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 1100, height: 800 });
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    const nodes = ['a', 'b'].map((id, index) => ({ id, type: 'custom', position: { x: index * 300, y: 0 }, data: { definitionType: 'BinningNode', label: `Bins ${id}`, columns: ['age'], strategy: 'custom', n_bins: index ? 5 : 1 } }));
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)(nodes, []);
  });
  await page.getByRole('button', { name: /Configuration Bins a.*at least 2/i }).press('Enter');
  const invalid = page.locator('[data-validation-field="n_bins"] input');
  await expect(invalid).toBeFocused();
  await expect(invalid).toHaveAccessibleDescription(/at least 2/);
  await expectNodeRevealed(page);
  await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore.getState();
    (state.onNodesChange as (changes: unknown[]) => void)([{ type: 'select', id: 'a', selected: false }, { type: 'select', id: 'b', selected: true }]);
  });
  await expect(page.getByRole('heading', { name: 'Bins b', exact: true })).toBeVisible();
  await expect(invalid).toHaveCount(0);
});
