import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

interface GraphFixture {
  nodes: Array<{ id: string; label: string }>;
  edges: Array<{ source: string; target: string; type: string }>;
}

/** Keep report data complete so graph failures cannot come from unrelated page fixtures. */
function report(id: number, graph: GraphFixture | null, target = 'species', categorical = false) {
  return {
    id, status: 'COMPLETED', created_at: '2026-09-11T12:00:00Z',
    profile_data: {
      row_count: 100, column_count: 2, duplicate_rows: 0,
      missing_cells_percentage: 0, memory_usage_mb: 0.01,
      columns: Object.fromEntries(['species', 'empty_target'].map(name => [name, {
        name, dtype: 'Categorical', missing_count: 0, missing_percentage: 0,
        categorical_stats: { unique_count: 2, top_k: [], rare_labels_count: 0 },
      }])),
      alerts: [], target_col: target, causal_graph: graph,
      correlations: null, correlations_with_target: null,
      causal_target_exclusion_reason: categorical ? 'categorical' : null,
    },
  };
}

/** Serve graph reports through the real EDA request and history lifecycle. */
async function openGraph(page: Page, graph: GraphFixture | null, mobile: boolean, categorical = false) {
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '101', source_id: 'causal-graph', name: 'Causal graph', type: 'file', format: 'csv',
    rows: 100, columns: 2, created_at: '2026-09-11T10:00:00Z',
  }] } }));
  await page.route('**/api/eda/101/latest', route => route.fulfill({ json: report(1, graph, 'species', categorical) }));
  await page.route('**/api/eda/101/history', route => route.fulfill({ json: [{
    id: 2, status: 'COMPLETED', target_col: 'empty_target', created_at: '2026-09-11T11:00:00Z',
  }] }));
  await page.route('**/api/eda/reports/2', route => route.fulfill({
    json: report(2, { nodes: [], edges: [] }, 'empty_target', true),
  }));
  await page.goto('/eda?dataset_id=101');
  await expect(page.getByRole('button', { name: 'Causal Graph', exact: true })).toBeVisible();
  if (mobile) await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
  await page.getByRole('button', { name: 'Causal Graph', exact: true }).click();
}

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test.describe(viewport.name, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('opens the target omission explanation when neither correlation matrix exists', async ({ page }) => {
      /** Both navigation and selected content must admit omission metadata without a numeric matrix. */
      await openGraph(page, null, viewport.name === 'mobile', true);
      const correlations = page.getByRole('button', { name: 'Correlations', exact: true });
      await expect(correlations).toBeVisible();
      await correlations.click();
      await expect(page.getByText('No Correlation Data', { exact: true })).toBeVisible();
      await expect(page.getByText(/species.*categorical.*omitted/i)).toBeVisible();
      await expect(page.getByText(/Pearson correlations.*Target Analysis/i)).toBeVisible();
    });

    test('preserves distinct column names and their edge connections', async ({ page }) => {
      /** Safe renderer IDs must not merge punctuation, Unicode, or prototype-like column names. */
      const sources = ['petal width', 'petal_width', '花', '__proto__', 'constructor', 'toString'];
      await openGraph(page, {
        nodes: [...sources, 'species_encoded'].map(id => ({ id, label: id })),
        edges: sources.map(source => ({ source, target: 'species_encoded', type: 'directed' })),
      }, viewport.name === 'mobile');
      const nodes = page.locator('#causal-chart .react-flow__node');
      await expect(nodes).toHaveCount(7);
      expect((await nodes.allTextContents()).sort()).toEqual([...sources, 'species_encoded'].sort());
      const renderedNodes = await nodes.evaluateAll(elements => elements.map(element => ({
        id: element.getAttribute('data-id'), label: element.textContent,
      })));
      expect(new Set(renderedNodes.map(node => node.id)).size).toBe(7);
      const targetId = renderedNodes.find(node => node.label === 'species_encoded')?.id;
      const connections = sources.map(source => {
        const sourceId = renderedNodes.find(node => node.label === source)?.id;
        return `Edge from ${sourceId} to ${targetId}`;
      });
      await expect.poll(() => page.locator('#causal-chart .react-flow__edge').evaluateAll(
        elements => elements.map(element => element.getAttribute('aria-label')),
      )).toEqual(connections);
      const paths = page.locator('#causal-chart .react-flow__edge-path');
      await expect(paths).toHaveCount(6);
      await expect.poll(async () => new Set(await paths.evaluateAll(elements => elements.map(element => element.getAttribute('d')))).size).toBe(6);
    });

    test('renders both bidirected arrowheads and explains categorical target omission', async ({ page }) => {
      /** A categorical target stays out of numeric inference while bidirected edges retain both endpoints. */
      await openGraph(page, {
        nodes: ['A', 'B', 'C'].map(id => ({ id, label: id })),
        edges: [
          { source: 'A', target: 'B', type: 'bidirected' },
          { source: 'B', target: 'C', type: 'directed' },
          { source: 'A', target: 'C', type: 'undirected' },
        ],
      }, viewport.name === 'mobile', true);
      const edges = page.locator('#causal-chart .react-flow__edge-path');
      await expect(edges).toHaveCount(3);
      await expect.poll(() => edges.evaluateAll(elements => elements.map(element => ({
        start: !!element.getAttribute('marker-start'), end: !!element.getAttribute('marker-end'),
      })))).toEqual([
        { start: true, end: true }, { start: false, end: true }, { start: false, end: false },
      ]);
      await expect(page.getByText(/species.*categorical.*omitted/i)).toBeVisible();
      await expect(page.getByText(/numeric variables/i)).toBeVisible();
      await expect(page.locator('#causal-chart')).not.toContainText('species_encoded');
    });

    test('clears the previous graph when an empty historical report loads', async ({ page }) => {
      /** A newly loaded report must never display causal relationships from the previous report. */
      await openGraph(page, {
        nodes: ['old_feature', 'species_encoded'].map(id => ({ id, label: id })),
        edges: [{ source: 'old_feature', target: 'species_encoded', type: 'directed' }],
      }, viewport.name === 'mobile');
      await expect(page.locator('#causal-chart .react-flow__node')).toHaveCount(2);
      const loaded = page.waitForResponse('**/api/eda/reports/2');
      await page.getByRole('button', { name: 'empty_target', exact: true }).click();
      expect((await (await loaded).json()).profile_data.causal_graph.nodes).toEqual([]);
      await expect(page.locator('#eda-toolbar-target-column')).toHaveValue('empty_target');
      await expect(page.getByText(/No causal graph available/i)).toBeVisible();
      await expect(page.locator('.react-flow__node')).toHaveCount(0);
      await expect(page.locator('.react-flow__edge-path')).toHaveCount(0);
      await expect(page.getByText(/empty_target.*categorical.*omitted/i)).toBeVisible();
    });
  });
}
