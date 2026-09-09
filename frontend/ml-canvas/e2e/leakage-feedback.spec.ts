import { expect, test, type Locator, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

type GraphState = Record<string, unknown>;
type TestWindow = Window & { __skyulfTest: { graphStore: { getState: () => GraphState } } };

test.use({ viewport: { width: 1440, height: 900 } });

/** Auto-selected missing indicators learn from raw data and therefore belong after the row split. */
async function seedLeakageGraph(page: Page) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([
      { id: 'raw', type: 'custom', position: { x: 0, y: 140 }, data: { definitionType: 'dataset_node', label: 'Raw data', datasetId: 'leakage-demo' } },
      { id: 'missing', type: 'custom', position: { x: 400, y: 140 }, data: { definitionType: 'MissingIndicator', label: 'Missing flags', columns: [] } },
      { id: 'row-split', type: 'custom', position: { x: 800, y: 140 }, data: { definitionType: 'TrainTestSplitter', label: 'Row split', test_size: 0.2, validation_size: 0, random_state: 42 } },
    ], [
      { id: 'raw-missing', type: 'custom', source: 'raw', target: 'missing', sourceHandle: 'data', targetHandle: 'in' },
      { id: 'missing-split', type: 'custom', source: 'missing', target: 'row-split', sourceHandle: 'out', targetHandle: 'in' },
    ]);
  });
  await expect(page.locator('.react-flow__node')).toHaveCount(3);
  await page.locator('.react-flow__controls-fitview').click();
}

/** Browser geometry catches clipped portals and horizontal overflow that jsdom cannot measure. */
async function expectWithinViewport(locator: Locator) {
  await expect.poll(() => locator.evaluate(element => {
    const box = element.getBoundingClientRect();
    return box.x >= -1 && box.y >= -1 && box.right <= window.innerWidth + 1 && box.bottom <= window.innerHeight + 1;
  })).toBe(true);
}

test('shows persistent node and edge advice, opens the guide, and clears fixed-column warnings without breaking deletion', async ({ page }) => {
  /** Leakage-only feedback must remain small, actionable, and independent of graph serialization. */
  await seedLeakageGraph(page);
  const node = page.locator('.react-flow__node[data-id="missing"]');
  const marker = node.getByRole('button', { name: /^Data leakage error:/ });
  const edgeMarkers = page.locator('.react-flow__edgelabel-renderer').getByRole('button', { name: /^Data leakage error:/ });
  await page.mouse.move(15, 15);
  await expect(marker).toBeVisible();
  await expect(node.locator('[data-leakage-severity="error"]')).toHaveClass(/border-red-500/);
  await expect(edgeMarkers.first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Collapse results panel', exact: true })).toHaveCount(0);

  await marker.focus();
  const details = page.getByRole('dialog', { name: 'Data leakage details', exact: true });
  await expect(details).toBeVisible();
  await expect(details).toContainText(/split|training/i);
  await page.keyboard.press('Escape');
  await expect(details).toHaveCount(0);
  await edgeMarkers.first().click();
  await details.getByRole('button', { name: 'Close data leakage details', exact: true }).click();
  await page.getByTestId('navbar-help').click();
  const guide = page.getByRole('dialog', { name: 'How pipelines work', exact: true });
  await guide.getByRole('tab', { name: 'Preprocessing & Leakage', exact: true }).click();
  await expect(guide.getByRole('tab', { name: 'Preprocessing & Leakage', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Escape');
  await expect(guide).toHaveCount(0);

  await page.evaluate(() => {
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    (state.updateNodeData as (id: string, data: unknown) => void)('missing', { columns: ['age'] });
  });
  await expect(page.getByRole('button', { name: /^Data leakage error:/ })).toHaveCount(0);
  await expect(node.locator('[data-leakage-severity]')).toHaveCount(0);
  expect(await page.evaluate(() => {
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    const graph = structuredClone({ nodes: state.nodes, edges: state.edges }) as {
      nodes: { data: Record<string, unknown> }[]; edges: { data?: Record<string, unknown> }[];
    };
    return [...graph.nodes, ...graph.edges].every(item =>
      !Object.prototype.hasOwnProperty.call(item.data ?? {}, 'leakageIssues') &&
      !Object.prototype.hasOwnProperty.call(item.data ?? {}, 'onOpenLeakageGuide'),
    );
  })).toBe(true);

  const remove = page.getByRole('button', { name: 'Remove connection from Missing flags to Row split', exact: true });
  await remove.focus();
  await expect(remove).toHaveCSS('opacity', '1');
  await remove.press('Enter');
  await expect(page.locator('.react-flow__edge[data-id="missing-split"]')).toHaveCount(0);
  await page.keyboard.press('Control+z');
  // A horizontal SVG group has zero layout height even though its stroked path is visible.
  await expect(page.locator('.react-flow__edge[data-id="missing-split"]')).toHaveCount(1);
  await remove.focus();
  await expect(remove).toHaveCSS('opacity', '1');
});

test('clears an open warning when the learned operation is rewired onto the training branch', async ({ page }) => {
  /** Rewiring must update feedback immediately without keeping the old warning portal alive. */
  await seedLeakageGraph(page);
  await page.locator('.react-flow__node[data-id="missing"]').getByRole('button', { name: /^Data leakage error:/ }).click();
  await expect(page.getByRole('dialog', { name: 'Data leakage details', exact: true })).toBeVisible();
  await page.evaluate(() => {
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)(state.nodes as unknown[], [
      { id: 'raw-split', type: 'custom', source: 'raw', target: 'row-split', sourceHandle: 'data', targetHandle: 'in' },
      { id: 'train-missing', type: 'custom', source: 'row-split', target: 'missing', sourceHandle: 'train', targetHandle: 'in' },
    ]);
  });

  await expect(page.getByRole('button', { name: /^Data leakage (error|warning):/ })).toHaveCount(0);
  await expect(page.getByRole('dialog', { name: 'Data leakage details', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Collapse results panel', exact: true })).toHaveCount(0);
});

test('keeps a backend leakage notice compact through dragging and clears it after configuration changes', async ({ page }) => {
  /** A blocked run needs useful guidance without opening Preview Results or surviving semantic edits. */
  await seedLeakageGraph(page);
  const message = 'Per-fold preprocessing refit skipped: payload reconstruction failed; CV/tuning scores may be optimistically biased.';
  await page.evaluate(async message => {
    const storeModule = '/src/core/store/useViewStore.ts';
    const feedbackModule = '/src/core/utils/leakageFeedback.ts';
    const { useViewStore } = await import(storeModule);
    const { graphSemanticSignature } = await import(feedbackModule);
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    useViewStore.getState().setLeakageNotice({
      message, graphSignature: graphSemanticSignature(state.nodes, state.edges),
    });
  }, message);
  const notice = page.getByRole('alert', { name: 'Leakage safety notice', exact: true });
  await expect(notice).toBeVisible();
  await expect(notice).toContainText('Leakage check blocked this run');
  await expect.poll(() => notice.evaluate(element => element.getBoundingClientRect().height < window.innerHeight / 3)).toBe(true);
  await expect(page.getByRole('button', { name: 'Collapse results panel', exact: true })).toHaveCount(0);

  const node = page.locator('.react-flow__node[data-id="missing"]');
  const transform = await node.evaluate(element => getComputedStyle(element).transform);
  await page.evaluate(() => {
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    (state.onNodesChange as (changes: unknown[]) => void)([{ id: 'missing', type: 'position', position: { x: 450, y: 220 } }]);
  });
  await expect(node).not.toHaveCSS('transform', transform);
  await expect(notice).toBeVisible();
  await notice.getByText('Details', { exact: true }).click();
  await expect(notice.getByText(message, { exact: true })).toBeVisible();
  await notice.getByRole('button', { name: 'Preprocessing guide', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'How pipelines work', exact: true })
    .getByRole('tab', { name: 'Preprocessing & Leakage', exact: true })).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Escape');
  await page.evaluate(() => {
    const state = (window as TestWindow).__skyulfTest.graphStore.getState();
    (state.updateNodeData as (id: string, data: unknown) => void)('missing', { columns: ['age'] });
  });

  await expect(notice).toHaveCount(0);
});

test.describe('phone leakage guidance', () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true });

  test('keeps touch advice and searchable guidance inside the viewport in read-only mode', async ({ page }) => {
    /** Phone users must be able to understand leakage and search the guide without editing or horizontal scrolling. */
    await seedLeakageGraph(page);
    const marker = page.locator('.react-flow__node[data-id="missing"]').getByRole('button', { name: /^Data leakage error:/ });
    await expect(marker).toBeVisible();
    await expect(page.getByRole('button', { name: /Remove connection/ })).toHaveCount(0);
    await marker.tap();
    const details = page.getByRole('dialog', { name: 'Data leakage details', exact: true });
    await expect(details).toBeVisible();
    await expectWithinViewport(details);
    await details.getByRole('button', { name: 'Close data leakage details', exact: true }).tap();
    await page.getByTestId('navbar-help').tap();
    const guide = page.getByRole('dialog', { name: 'How pipelines work', exact: true });
    await expect(guide).toBeVisible();
    await expectWithinViewport(guide);
    await guide.getByRole('tab', { name: 'Preprocessing & Leakage', exact: true }).tap();
    const search = guide.getByRole('searchbox', { name: 'Search preprocessing nodes', exact: true });
    await search.fill('MissingIndicator');
    await expectWithinViewport(search);
    await expect(guide.getByRole('article', { name: 'Missing Indicator', exact: true })).toBeVisible();
    await expect(guide.getByRole('article')).toHaveCount(1);
    const placementFilter = guide.getByRole('combobox', { name: 'Filter by placement', exact: true });
    await placementFilter.scrollIntoViewIfNeeded();
    await expectWithinViewport(placementFilter);

    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  });
});

test('opens node and edge advice on mouse hover without requiring a click', async ({ page }) => {
  /** Moving into the portaled advice must not close it or require keyboard focus. */
  await seedLeakageGraph(page);
  const nodeMarker = page.locator('.react-flow__node[data-id="missing"]')
    .getByRole('button', { name: /^Data leakage error:/ });
  const details = page.getByRole('dialog', { name: 'Data leakage details', exact: true });
  await nodeMarker.hover();
  await expect(details).toBeVisible();
  await expect(nodeMarker).not.toBeFocused();
  await details.hover();
  await expect(details).toBeVisible();
  await page.mouse.move(15, 15);
  await expect(details).toHaveCount(0);

  await page.locator('.react-flow__edgelabel-renderer')
    .getByRole('button', { name: /^Data leakage error:/ }).first().hover();
  await expect(details).toBeVisible();
  await page.mouse.move(15, 15);
  await expect(details).toHaveCount(0);
});

/** Measure against the browser's actual SVG curve, not the component's anchor calculation. */
async function expectControlsOnConnection(page: Page) {
  await expect.poll(() => page.evaluate(() => {
    const path = document.querySelector<SVGPathElement>(
      '.react-flow__edge[data-id="missing-split"] path.react-flow__edge-path',
    );
    const labels = document.querySelector('.react-flow__edgelabel-renderer');
    const marker = labels?.querySelector<HTMLButtonElement>('button[aria-label^="Data leakage error:"]');
    const remove = labels?.querySelector<HTMLButtonElement>(
      'button[aria-label="Remove connection from Missing flags to Row split"]',
    );
    const matrix = path?.getScreenCTM();
    if (!path || !marker || !remove || !matrix) return { onPath: false, separated: false };
    const boxes = [remove, marker].map(button => button.getBoundingClientRect());
    const centers = boxes.map(box => ({ x: box.x + box.width / 2, y: box.y + box.height / 2 }));
    const minimumDistances = [Infinity, Infinity];
    const length = path.getTotalLength();
    for (let index = 0; index <= 1200; index += 1) {
      const point = path.getPointAtLength(length * index / 1200);
      const screenPoint = new DOMPoint(point.x, point.y).matrixTransform(matrix);
      centers.forEach((center, control) => {
        minimumDistances[control] = Math.min(
          minimumDistances[control], Math.hypot(center.x - screenPoint.x, center.y - screenPoint.y),
        );
      });
    }
    return {
      onPath: minimumDistances.every(distance => distance <= 2),
      separated: Math.hypot(centers[0].x - centers[1].x, centers[0].y - centers[1].y)
        >= (boxes[0].width + boxes[1].width) / 2,
    };
  })).toEqual({ onPath: true, separated: true });
}

test('keeps both connection controls on horizontal and bent paths after rerouting and zooming', async ({ page }) => {
  /** Independent curve measurements catch floating controls that offset-based snapshots miss. */
  await seedLeakageGraph(page);
  await page.getByRole('button', {
    name: 'Remove connection from Missing flags to Row split', exact: true,
  }).focus();
  await expectControlsOnConnection(page);
  for (const position of [{ x: 550, y: 500 }, { x: 50, y: -150 }]) {
    await page.evaluate(position => {
      const state = (window as TestWindow).__skyulfTest.graphStore.getState();
      (state.onNodesChange as (changes: unknown[]) => void)([
        { id: 'row-split', type: 'position', position },
      ]);
    }, position);
    await page.locator('.react-flow__controls-fitview').click();
    await expectControlsOnConnection(page);
  }
  await page.locator('.react-flow__controls-zoomin').click();
  await expectControlsOnConnection(page);
});

test('keeps autosave behind the responsive guide and all three tabs accessible', async ({ page }, testInfo) => {
  /** Modal layering and tab access must survive desktop-to-phone resizing and long content. */
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.addInitScript(() => {
    localStorage.setItem('skyulf:canvas:autosave:v1', JSON.stringify({
      version: 1, savedAt: new Date().toISOString(), edges: [],
      nodes: [{ id: 'saved', type: 'custom', position: { x: 0, y: 0 },
        data: { definitionType: 'dataset_node', label: 'Saved dataset' } }],
    }));
  });
  await page.goto('/canvas');
  const restore = page.getByRole('button', { name: 'Restore', exact: true });
  await expect(restore).toBeVisible();
  await page.getByTestId('navbar-help').click();
  const guide = page.getByRole('dialog', { name: 'How pipelines work', exact: true });
  await expectWithinViewport(guide);
  await expect.poll(() => guide.evaluate(element => element.getBoundingClientRect().width)).toBeGreaterThan(900);
  await expect(guide.getByRole('tab')).toHaveCount(3);
  await guide.getByRole('tab', { name: 'Split & Merge', exact: true }).click();
  await expect(guide.getByRole('tabpanel')).toContainText('age');
  expect(await restore.evaluate(button => {
    const box = button.getBoundingClientRect();
    const top = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
    return top !== null && !button.closest('[role="status"]')?.contains(top);
  })).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('guide-desktop.png') });

  await page.setViewportSize({ width: 390, height: 844 });
  await expectWithinViewport(guide);
  await guide.getByRole('tabpanel').evaluate(panel => {
    let parent = panel.parentElement;
    while (parent) {
      if (['auto', 'scroll'].includes(getComputedStyle(parent).overflowY)) {
        parent.scrollTop = parent.scrollHeight;
        break;
      }
      parent = parent.parentElement;
    }
  });
  for (const tab of await guide.getByRole('tab').all()) await expectWithinViewport(tab);
  await page.screenshot({ path: testInfo.outputPath('guide-phone.png') });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
