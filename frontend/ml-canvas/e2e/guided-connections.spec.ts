import { test, expect, type Page, type Locator } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Seed editable nodes without backend jobs so connection interactions can be inspected directly. */
async function seed(page: Page, types: string[], edges: unknown[] = []) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(({ types, edges }) => {
    const store = (window as unknown as { __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } } }).__skyulfTest.graphStore;
    const state = store.getState();
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([], []);
    types.forEach((type, index) => (state.addNode as (type: string, position: unknown) => string)(type, { x: index === 0 ? 0 : 400, y: index > 1 ? 240 : 0 }));
    const nodes = store.getState().nodes as Record<string, unknown>[];
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)(nodes.map((node, index) => ({ ...node, id: `node-${index}`, selected: false })), edges);
  }, { types, edges });
  const collapse = page.getByRole('button', { name: 'Collapse results panel', exact: true });
  if (await collapse.isVisible()) await collapse.click();
  await page.locator('.react-flow__controls-fitview').click();
}

/** Move directly between port centers while keeping the pointer pressed. */
async function moveTo(page: Page, handle: Locator) {
  const box = await handle.boundingBox();
  if (!box) throw new Error('Expected a visible connection handle');
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 8 });
}

test('keyboard next-step insertion groups active split ports and undoes both node and edge', async ({ page }) => {
  // The output label is the keyboard equivalent of dragging a connection to a new step.
  await seed(page, ['TrainTestSplitter']);
  const trigger = page.getByRole('button', { name: 'Next step from Test', exact: true });
  await trigger.press('Enter');
  const picker = page.getByRole('dialog', { name: 'Connect next step', exact: true });
  await expect(picker).toBeVisible();
  await expect(picker.getByRole('textbox', { name: 'Search next steps' })).toBeFocused();
  await picker.getByRole('textbox', { name: 'Search next steps' }).fill('normalize');
  const add = picker.getByRole('button', { name: 'Add Scaling using Data', exact: true });
  await add.press('Space');
  await expect(picker).toHaveCount(0);
  await expect(page.locator('.react-flow__node')).toHaveCount(2);
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
  await expect(page.locator('.react-flow').locator('..')).toBeFocused();
  const handles = await page.evaluate(() => {
    const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => { edges: { sourceHandle: string }[] } } } }).__skyulfTest.graphStore.getState();
    return state.edges.map(edge => edge.sourceHandle);
  });
  expect(handles).toEqual(['train']);
  await expect(page.locator('[data-split-handle]')).toHaveCount(2);
  await expect(page.locator('[data-split-junction]')).toHaveCount(1);
  await expect.poll(() => page.locator('.react-flow__node').evaluateAll(elements => {
    const flow = document.querySelector('.react-flow')!.getBoundingClientRect();
    return elements.every(element => {
      const box = element.getBoundingClientRect();
      return box.left >= flow.left && box.right <= flow.right && box.top >= flow.top && box.bottom <= flow.bottom;
    });
  })).toBe(true);
  await page.locator('.react-flow').locator('..').focus();
  await page.keyboard.press('Control+z');
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
  await expect(page.locator('.react-flow__edge')).toHaveCount(0);
  await page.keyboard.press('Control+Shift+z');
  await expect(page.locator('.react-flow__node')).toHaveCount(2);
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
});

test('existing-node picker connects through the named input and Escape preserves the graph', async ({ page }) => {
  // Mouse and keyboard users must be able to choose an existing endpoint without dragging.
  await seed(page, ['imputation_node', 'encoding']);
  const trigger = page.locator('[data-id="node-0"]').getByRole('button', { name: 'Next step from Cleaned Data', exact: true });
  await trigger.click();
  const picker = page.getByRole('dialog', { name: 'Connect next step', exact: true });
  await picker.getByRole('button', { name: 'Existing node', exact: true }).click();
  await picker.getByRole('button', { name: 'Connect to Encoding using Data', exact: true }).press('Enter');
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
  await expect(trigger).toBeFocused();
  await trigger.press('Space');
  await expect(picker).toBeVisible();
  const search = picker.getByRole('textbox', { name: 'Search next steps', exact: true });
  await expect(search).toBeFocused();
  await search.press('Escape');
  await expect(picker).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
});

test('drag guidance rejects model-to-data and accepts the ensemble shared input', async ({ page }) => {
  // Guidance and the committed graph must agree about model endpoints.
  await seed(page, ['classification', 'imputation_node', 'EnsembleNode']);
  const source = page.locator('[data-id="node-0"] .react-flow__handle.source');
  const invalid = page.locator('[data-id="node-1"] .react-flow__handle.target');
  const valid = page.locator('[data-id="node-2"] .react-flow__handle.target');
  await moveTo(page, source);
  await page.mouse.down();
  await moveTo(page, invalid);
  await expect(invalid).toHaveAttribute('data-connection-state', 'incompatible');
  await expect(valid).toHaveAttribute('data-connection-state', 'compatible');
  const guidance = page.getByRole('tooltip', { name: 'Connection guidance', exact: true });
  await expect(guidance).toContainText('trained model');
  await expect.poll(() => guidance.evaluate(element => {
    const box = element.getBoundingClientRect();
    return box.x >= 8 && box.right <= innerWidth - 8 && box.y >= 8 && box.bottom <= innerHeight - 8;
  })).toBe(true);
  await page.screenshot({ path: 'test-results/connection-guidance.png' });
  await moveTo(page, valid);
  await expect(guidance).toHaveCount(0);
  await page.mouse.up();
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
  await expect(valid).not.toHaveAttribute('data-connection-state');
});

test('reverse dragging explains a cycle and does not commit the rejected edge', async ({ page }) => {
  // Starting at an input must not reverse the source/target used by the validator.
  await seed(page, ['imputation_node', 'encoding'], [{ id: 'back', source: 'node-1', sourceHandle: 'out', target: 'node-0', targetHandle: 'in', type: 'custom' }]);
  const start = page.locator('[data-id="node-1"] .react-flow__handle.target');
  const end = page.locator('[data-id="node-0"] .react-flow__handle.source');
  await moveTo(page, start);
  await page.mouse.down();
  await moveTo(page, end);
  await expect(page.getByRole('tooltip', { name: 'Connection guidance', exact: true })).toContainText('loop');
  await page.mouse.up();
  await expect(page.locator('.react-flow__edge')).toHaveCount(1);
});

test('model suggestions stay bounded in a dark laptop and disappear in read-only mode', async ({ page }) => {
  // Model output offers only compatible steps, and resizing must close mutation UI.
  await page.addInitScript(() => localStorage.setItem('skyulf-theme', 'dark'));
  await seed(page, ['classification']);
  await page.setViewportSize({ width: 1100, height: 800 });
  await page.locator('.react-flow__controls-fitview').click();
  await page.getByRole('button', { name: 'Next step from Trained Model', exact: true }).click();
  const picker = page.getByRole('dialog', { name: 'Connect next step', exact: true });
  await expect(picker.getByRole('button', { name: /^Add .* using/ })).toHaveCount(1);
  await expect(picker.getByRole('button', { name: 'Add Ensemble using Data / Models', exact: true })).toBeVisible();
  await expect.poll(() => picker.evaluate(element => {
    const box = element.getBoundingClientRect();
    return box.x >= 8 && box.right <= innerWidth - 8 && box.y >= 8 && box.bottom <= innerHeight - 8;
  })).toBe(true);
  await page.screenshot({ path: 'test-results/connection-picker-dark.png' });
  await page.setViewportSize({ width: 900, height: 800 });
  await expect(picker).toHaveCount(0);
  await expect(page.getByRole('button', { name: /^Next step from/ })).toHaveCount(0);
  await expect(page.locator('.react-flow__handle.connectable')).toHaveCount(0);
  await expect(page.locator('.react-flow__node')).toHaveCount(1);
  await page.setViewportSize({ width: 1100, height: 800 });
  await expect(page.getByRole('button', { name: 'Next step from Trained Model', exact: true })).toBeVisible();
  await expect(picker).toHaveCount(0);
});

for (const split of [
  { type: 'TrainTestSplitter', start: 'test', handles: ['train', 'validation', 'test'] },
  { type: 'feature_target_split', start: 'y', handles: ['X', 'y'] },
]) {
  test(`${split.type} joins all active outputs when manually wired and retains a single delete action`, async ({ page }) => {
    // Every member of a grouped split must meet one trunk and survive delete/undo/copy together.
    page.on('dialog', dialog => { void dialog.accept(); });
    await seed(page, [split.type, 'imputation_node']);
    const updateValidation = async (value: number) => page.evaluate(value => {
      const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => { updateNodeData: (id: string, data: unknown) => void } } } }).__skyulfTest.graphStore.getState();
      state.updateNodeData('node-0', { validation_size: value });
    }, value);
    if (split.type === 'TrainTestSplitter') await updateValidation(0.1);
    // Platform fallback fonts have different widths; long output labels must still clear the summary.
    const splitCard = page.locator('[data-id="node-0"] [data-node-definition-type]');
    for (const font of ['', 'Verdana, sans-serif', 'monospace']) {
      await splitCard.evaluate((element, font) => {
        (element as HTMLElement).style.fontFamily = font;
        // Exercise wider glyph spacing locally as well as Linux's monospace fallback.
        element.querySelectorAll<HTMLElement>('.react-flow__handle.source button').forEach(button => {
          button.style.letterSpacing = font === 'monospace' ? '0.5px' : '';
        });
      }, font);
      expect(await splitCard.evaluate(element => (element as HTMLElement).offsetHeight)).toBeLessThanOrEqual(110);
      const measurements = await splitCard.evaluate(element => {
        const card = element.getBoundingClientRect();
        const summary = element.querySelector('.text-\\[11px\\]')!.getBoundingClientRect();
        const labels = Array.from(element.querySelectorAll('.react-flow__handle.source button')).map(button => ({
          text: button.textContent, bounds: button.getBoundingClientRect(),
        }));
        return labels.map((label, index) => ({
          text: label.text,
          summaryGap: label.bounds.left - summary.right,
          bottomGap: card.bottom - label.bounds.bottom,
          previousLabelGap: index === 0 ? 0 : label.bounds.top - labels[index - 1]!.bounds.bottom,
        }));
      });
      for (const label of measurements) {
        const context = `${split.type}, ${font || 'default font'}, ${label.text}`;
        expect(label.summaryGap, `${context}: summary clearance`).toBeGreaterThanOrEqual(0);
        expect(label.bottomGap, `${context}: card clearance`).toBeGreaterThanOrEqual(0);
        expect(label.previousLabelGap, `${context}: preceding label clearance`).toBeGreaterThanOrEqual(0);
      }
    }
    await splitCard.evaluate(element => {
      (element as HTMLElement).style.fontFamily = '';
      element.querySelectorAll<HTMLElement>('.react-flow__handle.source button').forEach(button => {
        button.style.letterSpacing = '';
      });
    });
    const source = page.locator(`[data-id="node-0"] .react-flow__handle.source[data-handleid="${split.start}"]`);
    const target = page.locator('[data-id="node-1"] .react-flow__handle.target');
    await moveTo(page, source);
    await page.mouse.down();
    await moveTo(page, target);
    await page.mouse.up();
    await expect(page.locator('.react-flow__edge')).toHaveCount(1);
    await expect(page.locator('[data-split-handle]')).toHaveCount(split.handles.length);
    await expect(page.locator('[data-split-junction]')).toHaveCount(1);
    expect(await page.locator('[data-split-handle]').evaluateAll(elements => elements.every(element => {
      const path = element.querySelector('path') as SVGPathElement;
      const end = path.getPointAtLength(path.getTotalLength());
      const junction = element.parentElement!.querySelector('[data-split-junction]')!;
      return Math.abs(end.x - Number(junction.getAttribute('cx'))) < 0.01 && Math.abs(end.y - Number(junction.getAttribute('cy'))) < 0.01;
    }))).toBe(true);
    if (split.type === 'TrainTestSplitter') {
      const title = await page.locator('[data-id="node-0"]').getByText('Train-Test Split', { exact: true }).boundingBox();
      const train = await page.getByRole('button', { name: 'Next step from Train', exact: true }).boundingBox();
      expect(train!.y).toBeGreaterThan(title!.y + title!.height);
      await updateValidation(0);
      await expect(page.locator('[data-split-handle="validation"]')).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Next step from Validation', exact: true })).toHaveCount(0);
      await updateValidation(0.1);
      await expect(page.locator('[data-split-handle="validation"]')).toHaveCount(1);
    }
    await page.screenshot({ path: `test-results/grouped-${split.type}.png` });
    await page.locator('.react-flow__edge').focus();
    const remove = page.getByRole('button', { name: /^Remove connection from/ });
    await expect(remove).toHaveCount(1);
    await remove.click();
    await expect(page.locator('.react-flow__edge')).toHaveCount(0);
    await page.keyboard.press('Control+z');
    await expect(page.locator('[data-split-handle]')).toHaveCount(split.handles.length);
    await page.evaluate(() => {
      const state = (window as unknown as { __skyulfTest: { graphStore: { getState: () => { onNodesChange: (changes: unknown[]) => void } } } }).__skyulfTest.graphStore.getState();
      state.onNodesChange(['node-0', 'node-1'].map(id => ({ id, type: 'select', selected: true })));
    });
    await page.locator('.react-flow').locator('..').focus();
    await page.keyboard.press('Control+c');
    await page.keyboard.press('Control+v');
    await expect(page.locator('.react-flow__node')).toHaveCount(4);
    await expect(page.locator('.react-flow__edge')).toHaveCount(2);
    await expect(page.locator('[data-split-junction]')).toHaveCount(2);
  });
}
