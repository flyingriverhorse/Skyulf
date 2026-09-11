import { writeFile } from 'node:fs/promises';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

// Exercise the actual Plotly WebGL renderer in headless Chromium.
test.use({ launchOptions: { args: ['--enable-unsafe-swiftshader'] } });

const missingColor = '#6b7280';
const missingLabel = 'Unlabeled (missing) 2';
const categoryLabels = ['__proto__', 'constructor', 'Other', 'toString', 'Unlabeled', 'Unlabeled (missing)'];
const labels = ['toString', null, 'Other', '__proto__', 'Unlabeled', 'constructor', 'Unlabeled (missing)'];

interface Point {
  x: number;
  y: number;
  z: number;
  lat: number;
  lon: number;
  label: string | null;
}

interface LegendEntry {
  label: string;
  color: string;
  shape: string;
  symbol: string;
}

interface PlotlyTrace {
  name: string;
  type: string;
  x: number[];
  y: number[];
  z: number[];
  marker: { color: string; symbol: string };
  renderedSymbol?: string;
}

/** Keep coordinates unique so point loss and label/coordinate misalignment are observable. */
function pointsFor(pointLabels: Array<string | null>): Point[] {
  return pointLabels.map((label, index) => ({
    x: index + 1, y: index + 2, z: index + 3,
    lat: 54.6 + index / 100, lon: 25.2 + index / 100, label,
  }));
}

/** Serve complete EDA API fixtures while allowing a subsequent reload to reverse row order. */
async function mockLabelProfile(page: Page, initialPoints: Point[]) {
  let points = initialPoints;
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '101', source_id: 'label-groups', name: 'Label groups', type: 'file',
    format: 'csv', rows: points.length, columns: 6, created_at: '2026-09-11T10:00:00Z',
  }] } }));
  await page.route('**/api/eda/101/history', route => route.fulfill({ json: [] }));
  await page.route('**/api/eda/101/latest', route => route.fulfill({ json: {
    id: 41, status: 'COMPLETED', profile_data: {
      row_count: points.length, column_count: 6, columns: {}, alerts: [], target_col: 'group',
      pca_data: points,
      geospatial: {
        lat_col: 'lat', lon_col: 'lon', min_lat: 54.59, max_lat: 54.7,
        min_lon: 25.19, max_lon: 25.3, centroid_lat: 54.645, centroid_lon: 25.245,
        sample_points: points,
      },
    },
  } }));
  await page.route('https://*.tile.openstreetmap.org/**', route => route.fulfill({
    contentType: 'image/svg+xml',
    body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="#f8fafc"/></svg>',
  }));
  return () => { points = [...points].reverse(); };
}

/** Use the existing sidebar control to leave chart space on narrow viewports. */
async function openAnalysis(page: Page, name: 'PCA & Clusters' | 'Geospatial', mobile: boolean) {
  await expect(page.getByRole('button', { name, exact: true })).toBeVisible();
  if (mobile && await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).count()) {
    await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
    const sidebar = page.locator('div.w-14').filter({
      has: page.getByRole('button', { name: 'Expand Sidebar', exact: true }),
    });
    await expect(sidebar).toHaveCSS('width', '56px');
  }
  await page.getByRole('button', { name, exact: true }).click();
}

/** Read the user-visible legend, including the actual SVG swatch color and shape. */
async function readLegend(page: Page, anchorLabel: string): Promise<LegendEntry[]> {
  const legend = page.locator('ul').filter({ has: page.locator(`span[title="${anchorLabel}"]`) });
  await expect(legend).toBeVisible();
  return legend.locator('li').evaluateAll(items => items.map(item => {
    const icon = item.querySelector('svg > *');
    let symbol = icon?.tagName.toLowerCase() ?? '';
    if (icon instanceof SVGRectElement && icon.width.baseVal.value === icon.height.baseVal.value) symbol = 'square';
    if (icon instanceof SVGPolygonElement && icon.points.numberOfItems === 4) symbol = 'diamond';
    if (icon instanceof SVGPathElement) {
      const samples = [0.125, 0.375, 0.625, 0.875].map(fraction => icon.getPointAtLength(icon.getTotalLength() * fraction));
      const isPlus = samples.every(point => Math.abs(point.x - 6) < 0.01 || Math.abs(point.y - 6) < 0.01);
      const isDiagonal = samples.every(point => Math.abs(Math.abs(point.x - 6) - Math.abs(point.y - 6)) < 0.01);
      symbol = isPlus ? 'cross' : isDiagonal ? 'x' : 'unknown-path';
    }
    return {
      label: item.querySelector('span[title]')?.textContent ?? '',
      color: icon?.getAttribute('fill') ?? icon?.getAttribute('stroke') ?? '',
      shape: icon?.outerHTML ?? '',
      symbol,
    };
  }));
}

/** Resolve marker identity through Plotly itself and compare it with the drawn legend geometry. */
function expectRenderedLegendSymbols(traces: PlotlyTrace[], entries: LegendEntry[]) {
  expect(traces.map(trace => trace.renderedSymbol)).toEqual(traces.map(trace => trace.marker.symbol));
  expect(entries.map(entry => ({ label: entry.label, symbol: entry.symbol })))
    .toEqual(traces.map(trace => ({ label: trace.name, symbol: trace.renderedSymbol })));
}

/** Assert that missing values cannot merge with real category names or object prototype keys. */
function expectDistinctGroups(entries: LegendEntry[]) {
  expect(entries.map(entry => entry.label)).toEqual([...categoryLabels, missingLabel]);
  expect(entries.find(entry => entry.label === missingLabel)?.color).toBe(missingColor);
  expect(entries.filter(entry => entry.color === missingColor)).toHaveLength(1);
}

/** Inspect the real Plotly trace arrays rather than a mocked renderer or a module export. */
async function readTraces(plot: Locator): Promise<PlotlyTrace[]> {
  await expect(plot.locator('.gl-container canvas')).toBeVisible();
  return plot.evaluate(element => {
    const chart = element as HTMLElement & { data: PlotlyTrace[]; _fullData?: PlotlyTrace[] };
    return chart.data.map(({ name, type, x, y, z, marker }, index) => ({
      name, type, x, y, z, marker: { color: marker.color, symbol: marker.symbol },
      renderedSymbol: chart._fullData?.[index]?.marker.symbol ?? '',
    }));
  });
}

/** Match each SVG marker to its real popup coordinates, independent of map render order. */
async function readMapPoints(page: Page) {
  const markers = page.locator('.leaflet-overlay-pane path.leaflet-interactive');
  const renderedPoints = [];
  for (const marker of await markers.all()) {
    const color = await marker.getAttribute('stroke');
    expect(await marker.getAttribute('fill')).toBe(color);
    await marker.click();
    const popup = page.locator('.leaflet-popup-content');
    await expect(popup).toBeVisible();
    const text = await popup.innerText();
    renderedPoints.push({
      color,
      lat: text.match(/Lat:\s*([\d.-]+)/)?.[1] ?? '',
      lon: text.match(/Lon:\s*([\d.-]+)/)?.[1] ?? '',
      label: text.match(/group:\s*(.*)/)?.[1]?.trim() ?? '',
    });
    await page.locator('.leaflet-popup-close-button').click();
    await expect(page.locator('.leaflet-popup-content')).toHaveCount(0);
  }
  return renderedPoints.sort((left, right) => left.lat.localeCompare(right.lat));
}

/** Every input row must retain its coordinates within the matching displayed group. */
function expectTracePoints(traces: PlotlyTrace[], points: Point[], entries: LegendEntry[]) {
  expect(traces.map(trace => trace.name)).toEqual(entries.map(entry => entry.label));
  for (const trace of traces) {
    const expected = points.filter(point => (point.label ?? missingLabel) === trace.name);
    expect(trace.type).toBe('scatter3d');
    expect(trace.x).toEqual(expected.map(point => point.x));
    expect(trace.y).toEqual(expected.map(point => point.y));
    expect(trace.z).toEqual(expected.map(point => point.z));
    expect(trace.marker.color).toBe(entries.find(entry => entry.label === trace.name)?.color);
  }
  expect(traces.reduce((count, trace) => count + trace.x.length, 0)).toBe(points.length);
}

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000, mobile: false },
  { name: 'mobile', width: 390, height: 844, mobile: true },
]) {
  test.describe(`EDA label groups on ${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });
    test.setTimeout(60_000);

    test('PCA keeps distinct groups, colors, and every row across 2D, 3D, and row reversal', async ({ page }, testInfo) => {
      // Reserved object keys and missing-label collisions must survive real rendering.
      const points = pointsFor(labels);
      const reverseRows = await mockLabelProfile(page, points);
      const pageErrors: string[] = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.goto('/eda?dataset_id=101');
      await openAnalysis(page, 'PCA & Clusters', viewport.mobile);
      await expect(page.locator('#pca-chart canvas')).toBeVisible();
      const initialLegend = await readLegend(page, 'Other');
      expectDistinctGroups(initialLegend);
      await page.getByRole('button', { name: 'View data table', exact: true }).click();
      await expect(page.getByRole('region', { name: 'PCA projection data' }).locator('tbody tr')).toHaveCount(points.length);
      await page.getByRole('button', { name: 'Switch to 3D', exact: true }).click();
      const initialTraces = await readTraces(page.locator('#pca-chart .js-plotly-plot'));
      expectTracePoints(initialTraces, points, initialLegend);
      expectRenderedLegendSymbols(initialTraces, initialLegend);
      expect(initialLegend.map(entry => entry.symbol)).toEqual(['circle', 'square', 'diamond', 'cross', 'x', 'circle', 'cross']);
      const symbolPath = testInfo.outputPath('pca-marker-symbols.json');
      await writeFile(symbolPath, JSON.stringify(initialTraces.map(trace => ({
          label: trace.name, requested: trace.marker.symbol, rendered: trace.renderedSymbol,
        }))));
      await testInfo.attach('pca-marker-symbols', {
        path: symbolPath,
        contentType: 'application/json',
      });
      expect(await readLegend(page, 'Other')).toEqual(initialLegend);
      await page.getByRole('button', { name: 'Switch to 2D', exact: true }).click();
      await expect(page.locator('#pca-chart .js-plotly-plot')).toHaveCount(0);
      await expect(page.locator('#pca-chart canvas')).toBeVisible();
      expect(await readLegend(page, 'Other')).toEqual(initialLegend);

      reverseRows();
      await page.reload();
      await openAnalysis(page, 'PCA & Clusters', viewport.mobile);
      await expect(page.locator('#pca-chart canvas')).toBeVisible();
      expect(await readLegend(page, 'Other')).toEqual(initialLegend);
      await page.getByRole('button', { name: 'Switch to 3D', exact: true }).click();
      const reversedTraces = await readTraces(page.locator('#pca-chart .js-plotly-plot'));
      expectTracePoints(reversedTraces, [...points].reverse(), initialLegend);
      expectRenderedLegendSymbols(reversedTraces, await readLegend(page, 'Other'));
      expect(reversedTraces.map(trace => ({ name: trace.name, ...trace.marker })))
        .toEqual(initialTraces.map(trace => ({ name: trace.name, ...trace.marker })));
      await page.getByRole('button', { name: 'Switch to 2D', exact: true }).click();
      await expect(page.locator('#pca-chart .js-plotly-plot')).toHaveCount(0);
      await expect(page.locator('#pca-chart canvas')).toBeVisible();
      expect(await readLegend(page, 'Other')).toEqual(initialLegend);
      expect(pageErrors).toEqual([]);
    });

    test('map circles and legend retain category colors after row reversal', async ({ page }) => {
      // Map points must use the same group identity as PCA, including an explicit missing group.
      const points = pointsFor(labels);
      const reverseRows = await mockLabelProfile(page, points);
      await page.goto('/eda?dataset_id=101');
      await openAnalysis(page, 'Geospatial', viewport.mobile);
      const circles = page.locator('.leaflet-overlay-pane path.leaflet-interactive');
      await expect(circles).toHaveCount(points.length);
      const initialLegend = await readLegend(page, 'Other');
      expectDistinctGroups(initialLegend);
      expect(initialLegend.every(entry => entry.shape.startsWith('<circle '))).toBe(true);
      const renderedPoints = await readMapPoints(page);
      expect(renderedPoints).toEqual(points.map(point => ({
        color: initialLegend.find(entry => entry.label === (point.label ?? missingLabel))?.color,
        label: point.label ?? missingLabel,
        lat: point.lat.toFixed(4), lon: point.lon.toFixed(4),
      })));

      reverseRows();
      await page.reload();
      await openAnalysis(page, 'Geospatial', viewport.mobile);
      await expect(circles).toHaveCount(points.length);
      expect(await readLegend(page, 'Other')).toEqual(initialLegend);
      expect(await readMapPoints(page)).toEqual(renderedPoints);
    });

    test('all-missing PCA labels remain visible with an Unlabeled legend', async ({ page }) => {
      // A one-group missing-label legend must remain available in both chart modes.
      const points = pointsFor([null, null, null]);
      await mockLabelProfile(page, points);
      await page.goto('/eda?dataset_id=101');
      await openAnalysis(page, 'PCA & Clusters', viewport.mobile);
      await expect(page.locator('#pca-chart canvas')).toBeVisible();
      const entries = await readLegend(page, 'Unlabeled');
      expect(entries).toHaveLength(1);
      expect(entries[0]?.color).toBe(missingColor);
      await page.getByRole('button', { name: 'Switch to 3D', exact: true }).click();
      const traces = await readTraces(page.locator('#pca-chart .js-plotly-plot'));
      expect(traces).toHaveLength(1);
      expect(traces[0]?.name).toBe('Unlabeled');
      expect(traces[0]?.x).toEqual([1, 2, 3]);
      expect(traces[0]?.marker.color).toBe(missingColor);
      expectRenderedLegendSymbols(traces, entries);
      expect(await readLegend(page, 'Unlabeled')).toEqual(entries);
      await page.getByRole('button', { name: 'Switch to 2D', exact: true }).click();
      await expect(page.locator('#pca-chart canvas')).toBeVisible();
      expect(await readLegend(page, 'Unlabeled')).toEqual(entries);
    });
  });
}
