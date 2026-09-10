import { test, expect, type Locator, type Page } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

/** Seed real canvas settings without a schema name that triggers target autofill. */
async function openResampling(page: Page) {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: Object.fromEntries(['outcome', 'feature', 'second_label'].map(name => [name, {
      name, dtype: 'float64', missing_count: 0, missing_ratio: 0, unique_count: 10,
    }])), row_count: 150,
  } }));
  await page.route('**/api/pipeline/schema-preview', route => route.fulfill({ json: {
    pipeline_id: 'resampling-schema', predicted_schemas: {}, broken_references: [],
  } }));
  await page.goto('/canvas');
  await page.waitForFunction(() => Boolean(window.__skyulfTest));
  await page.evaluate(() => {
    window.__skyulfTest!.graphStore.getState().setGraph([
      { id: 'ds', type: 'custom', position: { x: 80, y: 140 },
        data: { definitionType: 'dataset_node', datasetId: 'iris-demo' } },
      { id: 'resample', type: 'custom', position: { x: 380, y: 140 }, selected: true,
        data: { definitionType: 'ResamplingNode', type: 'oversampling', method: 'smote',
          target_column: '', sampling_strategy: 'auto', random_state: 42, k_neighbors: 5 } },
    ], [{ id: 'edge', source: 'ds', sourceHandle: 'data', target: 'resample', targetHandle: 'in' }]);
  });
  await expect(page.getByLabel('Target Column', { exact: true })).toBeVisible();
}

/** Compare rendered geometry so portal suggestions cannot drift away from their input. */
async function expectAnchored(input: Locator, suggestions: Locator) {
  await expect(suggestions).toBeVisible();
  await expect.poll(async () => {
    const field = await input.boundingBox();
    const list = await suggestions.boundingBox();
    if (!field || !list) return false;
    const gap = list.y - field.y - field.height;
    return Math.abs(list.x - field.x) <= 2 && Math.abs(list.width - field.width) <= 2
      && gap >= 0 && gap <= 10;
  }).toBe(true);
}

for (const expanded of [false, true]) {
  test(`target suggestions stay anchored and support selection in ${expanded ? 'expanded' : 'docked narrow'} settings`, async ({ page }) => {
    // Application listbox geometry is testable; the old OS-native datalist popup is outside the DOM.
    await openResampling(page);
    if (expanded) await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
    else await expect(page.getByRole('separator', { name: 'Resize settings panel' })).toHaveAttribute('aria-valuenow', '320');

    const input = page.getByLabel('Target Column', { exact: true });
    const suggestions = page.getByRole('listbox', { name: 'Target column suggestions', exact: true });
    await input.click();
    await expectAnchored(input, suggestions);
    await expect(suggestions.getByRole('option')).toHaveText(['outcome', 'feature', 'second_label']);
    await page.setViewportSize({ width: 1280, height: 1000 });
    await expectAnchored(input, suggestions);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await expectAnchored(input, suggestions);
    await suggestions.getByRole('option', { name: 'outcome', exact: true }).click();
    await expect(input).toHaveValue('outcome');
    await expect(suggestions).toBeHidden();

    await input.fill('second');
    await expectAnchored(input, suggestions);
    await input.press('ArrowDown');
    await input.press('Enter');
    await expect(input).toHaveValue('second_label');
    await expect(suggestions).toBeHidden();

    await input.fill('');
    await expectAnchored(input, suggestions);
    await input.press('Escape');
    await expect(suggestions).toBeHidden();
    await expect(page.getByRole('button', { name: 'Close settings panel', exact: true })).toBeVisible();
    await expect(input).toBeFocused();

    if (!expanded) {
      const resize = page.getByRole('separator', { name: 'Resize settings panel' });
      await resize.focus();
      await resize.press('ArrowLeft');
      await expect(resize).toHaveAttribute('aria-valuenow', '340');
      await input.click();
      await expectAnchored(input, suggestions);
    }
    await input.click();
    await expectAnchored(input, suggestions);
    await input.press('Tab');
    await expect(suggestions).toBeHidden();
    await expect(page.getByRole('combobox', { name: 'Sampling Strategy', exact: true })).toBeFocused();
    await input.click();
    await expectAnchored(input, suggestions);
    const randomState = page.getByRole('spinbutton', { name: 'Random State', exact: true });
    await randomState.click();
    await expect(suggestions).toBeHidden();
    await expect(randomState).toBeFocused();
    await input.fill('manual_target');
    await expect(input).toHaveValue('manual_target');
  });
}
