import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

test.beforeEach(async ({ page }) => {
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {}, row_count: 150 } }));
});

for (const dark of [false, true]) {
  test(`dataset upload is keyboard reachable and returns focus (${dark ? 'dark laptop' : 'light desktop'})`, async ({ page }) => {
    // Upload must remain possible without a mouse and cancellation must return to a live control.
    await page.setViewportSize({ width: dark ? 1100 : 1440, height: 900 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/canvas');
    if (dark) {
      await page.evaluate(() => document.documentElement.classList.add('dark'));
      await page.getByRole('button', { name: 'Expand components sidebar', exact: true }).click();
    }
    await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
    const upload = page.getByRole('button', { name: 'New Upload', exact: true });
    await upload.press('Enter');
    const browse = page.getByLabel('Browse dataset file', { exact: true });
    await expect(browse).toBeFocused();
    const chooser = page.waitForEvent('filechooser');
    await browse.press('Enter');
    expect((await chooser).isMultiple()).toBe(false);
    const violations = await new AxeBuilder({ page }).include('[aria-label="Node settings"]')
      .withRules(['button-name', 'label', 'select-name']).analyze();
    expect(violations.violations).toEqual([]);
    await page.getByRole('button', { name: 'Close dataset upload', exact: true }).press('Enter');
    await expect(upload).toBeFocused();
    await page.getByRole('button', { name: 'Close settings panel', exact: true }).press('Enter');
    await expect(page.getByRole('region', { name: 'Pipeline canvas', exact: true })).toBeFocused();
  });
}

test('settings expansion and dismissal retain visible keyboard focus', async ({ page }) => {
  // Removing a panel must not strand focus on the document or reset the graph viewport.
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  await page.getByRole('button', { name: 'Expand settings panel', exact: true }).press('Enter');
  const collapse = page.getByRole('button', { name: 'Collapse settings panel', exact: true });
  await expect(collapse).toBeFocused();
  await expect.poll(() => collapse.evaluate(element => getComputedStyle(element).boxShadow)).not.toBe('none');
  await collapse.press('Enter');
  await expect(page.getByRole('button', { name: 'Expand settings panel', exact: true })).toBeFocused();
  const viewport = await page.locator('.react-flow__viewport').getAttribute('style');
  await page.getByRole('button', { name: 'Close settings panel', exact: true }).press('Enter');
  await expect(page.getByRole('region', { name: 'Pipeline canvas', exact: true })).toBeFocused();
  await expect(page.locator('.react-flow__viewport')).toHaveAttribute('style', viewport!);
});

test('dismissing results returns keyboard focus to the canvas', async ({ page }) => {
  // The close button disappears with results, so subsequent shortcuts need a live focus target.
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  await page.getByRole('button', { name: 'Close preview results', exact: true }).press('Enter');
  await expect(page.getByRole('region', { name: 'Pipeline canvas', exact: true })).toBeFocused();
});

test('a completed upload returns focus to the selected dataset', async ({ page }) => {
  // Completion removes the file picker, but the user must still be able to continue configuring the node.
  await page.route('**/upload', route => route.fulfill({ json: { job_id: 'iris-demo' } }));
  await page.goto('/canvas');
  await page.getByRole('button', { name: 'Add Dataset node', exact: true }).click();
  await page.getByRole('button', { name: 'New Upload', exact: true }).press('Enter');
  await page.getByLabel('Browse dataset file', { exact: true }).setInputFiles({
    name: 'iris.csv', mimeType: 'text/csv', buffer: Buffer.from('age\n10\n'),
  });
  const dataset = page.getByRole('combobox', { name: 'Select Dataset', exact: true });
  await expect(dataset).toHaveValue('iris-demo');
  await expect(dataset).toBeFocused();
});

for (const dark of [false, true]) {
  test(`repeated preprocessing controls retain keyboard context (${dark ? 'dark' : 'light'})`, async ({ page }) => {
    // Native select/delete keys must not bubble into a collapsible row and hide the active editor.
    await page.setViewportSize({ width: dark ? 1100 : 1440, height: 900 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/canvas');
    await page.evaluate((isDark) => {
      document.documentElement.classList.toggle('dark', isDark);
      const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => {
        addNode: (type: string, position: { x: number; y: number }, data: unknown) => string;
      } } } }).__skyulfTest.graphStore.getState();
      state.addNode('FeatureGenerationNode', { x: 200, y: 200 }, { operations: [
        { operation_type: 'arithmetic', method: 'add', input_columns: [], secondary_columns: [], isExpanded: true },
        { operation_type: 'ratio', method: 'ratio', input_columns: [], secondary_columns: [], isExpanded: true },
      ] });
    }, dark);
    const panel = page.getByRole('complementary', { name: 'Node settings', exact: true });
    const method = panel.getByRole('combobox', { name: 'Method for operation 1', exact: true });
    await method.press('Space');
    await method.press('Escape');
    const header = panel.getByRole('button', { name: 'Collapse arithmetic operation 1', exact: true });
    await expect(header).toHaveAttribute('aria-expanded', 'true');
    await expect(method).toBeVisible();
    await header.press('Enter');
    await expect(panel.getByRole('button', { name: 'Expand arithmetic operation 1', exact: true })).toBeFocused();
    await panel.getByRole('button', { name: 'Expand arithmetic operation 1', exact: true }).press('Space');
    await expect.poll(() => header.evaluate(element => getComputedStyle(element).boxShadow)).not.toBe('none');
    await panel.getByRole('button', { name: 'Remove operation 2', exact: true }).press('Space');
    await expect(method).toHaveValue('add');
    await expect(panel.getByRole('button', { name: 'Remove operation 2', exact: true })).toHaveCount(0);
    const scan = await new AxeBuilder({ page }).include('[aria-label="Node settings"]')
      .withRules(['button-name', 'label', 'select-name', 'nested-interactive']).analyze();
    expect(scan.violations).toEqual([]);
    await page.screenshot({ path: `test-results/preprocessing-accessibility-${dark ? 'dark' : 'light'}.png` });
  });
}
