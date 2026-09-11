import type { ReactNode } from 'react';
import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { CanvasScatterPlot } from './CanvasScatterPlot';
import { ThreeDScatterPlot, type ScatterPoint } from './ThreeDScatterPlot';
import { GeospatialTab } from './tabs/GeospatialTab';
import { PCATab } from './tabs/PCATab';

interface Dataset {
  label: string;
  backgroundColor: string;
  borderColor: string;
  pointStyle: string;
  data: Array<{ x: number; y: number; raw: ScatterPoint }>;
}

interface Trace {
  name: string;
  x: number[];
  marker: { color: string; symbol: string };
}

let datasets: Dataset[] = [];
let traces: Trace[] = [];

vi.mock('react-chartjs-2', () => ({
  Scatter: ({ data }: { data: { datasets: Dataset[] } }) => {
    datasets = data.datasets;
    return null;
  },
}));

vi.mock('../../core/plotly', () => ({
  Plot: ({ data }: { data: Trace[] }) => {
    traces = data;
    return null;
  },
}));

vi.mock('react-leaflet', () => ({
  MapContainer: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  TileLayer: () => null,
  Popup: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  CircleMarker: ({ center, pathOptions, children }: {
    center: [number, number];
    pathOptions: { color: string; fillColor: string };
    children: ReactNode;
  }) => (
    <div data-testid="map-point" data-x={center[0]} data-color={pathOptions.color} data-fill={pathOptions.fillColor}>
      {children}
    </div>
  ),
}));

interface Point {
  x: number;
  y: number;
  z: number;
  label: string | null | undefined;
  [key: string]: string | number | null | undefined;
}

function pointsFor(labels: Array<string | null | undefined>): Point[] {
  return labels.map((label, index) => ({ x: index + 1, y: index + 2, z: index + 3, label }));
}

function Charts({ points }: { points: Point[] }) {
  const profile = {
    target_col: 'label',
    geospatial: {
      min_lat: 0, min_lon: 0, max_lat: 50, max_lon: 50,
      sample_points: points.map(({ x, y, label }) => ({
        lat: x, lon: y, ...(label === undefined ? {} : { label }),
      })),
    },
  };
  return (
    <>
      <CanvasScatterPlot data={points} xKey="x" yKey="y" labelKey="label" />
      <ThreeDScatterPlot data={points} xKey="x" yKey="y" zKey="z" labelKey="label" />
      <GeospatialTab profile={profile} />
    </>
  );
}

function chartStyles() {
  return new Map(datasets.map((dataset) => [dataset.label, {
    color: dataset.backgroundColor,
    shape: dataset.pointStyle,
  }]));
}

function traceStyles() {
  return new Map(traces.map((trace) => [trace.name, trace.marker]));
}

function mapColors() {
  return new Map(screen.getAllByTestId('map-point').map((point) => [
    Number(point.getAttribute('data-x')), point.getAttribute('data-color'),
  ]));
}

function expectMatchingColors(points: Point[]) {
  const colors = mapColors();
  expect(colors.size).toBe(points.length);
  for (const dataset of datasets) {
    const trace = traces.find((candidate) => candidate.name === dataset.label);
    expect(trace?.marker.color).toBe(dataset.backgroundColor);
    expect(trace?.x).toEqual(dataset.data.map((point) => point.x));
    for (const point of dataset.data) expect(colors.get(point.x)).toBe(dataset.backgroundColor);
  }
  for (const marker of screen.getAllByTestId('map-point')) {
    expect(marker.getAttribute('data-fill')).toBe(marker.getAttribute('data-color'));
  }
  expect(datasets.flatMap((dataset) => dataset.data.map((point) => point.x)).sort((a, b) => a - b))
    .toEqual(points.map((point) => point.x).sort((a, b) => a - b));
}

beforeEach(() => {
  datasets = [];
  traces = [];
});

afterEach(cleanup);

describe('EDA category labels at chart and map boundaries', () => {
  it('uses the same five supported marker shapes in both scatter renderers', () => {
    // Unsupported 3D symbols must not silently collapse distinct category markers to circles.
    render(<Charts points={pointsFor(['f', 'c', 'e', 'a', 'd', 'b'])} />);

    expect(datasets.map((dataset) => dataset.label)).toEqual(['a', 'b', 'c', 'd', 'e', 'f']);
    expect(datasets.map((dataset) => dataset.pointStyle))
      .toEqual(['circle', 'rect', 'rectRot', 'cross', 'crossRot', 'circle']);
    expect(traces.map((trace) => trace.marker.symbol))
      .toEqual(['circle', 'square', 'diamond', 'cross', 'x', 'circle']);
  });

  it('keeps category colors and marker shapes stable when point order reverses', () => {
    // Switching views or row order must not change the meaning of a category color.
    const points = pointsFor(['zeta', 'alpha', null, 'beta', 'zeta']);
    const { rerender } = render(<Charts points={points} />);
    const original2D = chartStyles();
    const original3D = traceStyles();
    const originalMap = mapColors();

    rerender(<Charts points={[...points].reverse()} />);

    expect(chartStyles()).toEqual(original2D);
    expect(traceStyles()).toEqual(original3D);
    expect(mapColors()).toEqual(originalMap);
    expect(new Set(datasets.map((dataset) => dataset.pointStyle)).size).toBeGreaterThan(1);
    expectMatchingColors(points);
  });

  it.each([
    { realLabels: ['Other'], missingLabel: 'Unlabeled' },
    { realLabels: ['Other', 'Unlabeled'], missingLabel: 'Unlabeled (missing)' },
    { realLabels: ['Other', 'Unlabeled', 'Unlabeled (missing)'], missingLabel: undefined },
  ])('keeps missing points separate from real labels $realLabels', ({ realLabels, missingLabel }) => {
    // Missing targets must not impersonate observed values, even fallback-like names.
    const points = pointsFor([...realLabels, null, undefined]);
    render(<Charts points={points} />);

    const missing = datasets.find((dataset) => dataset.data.some((point) => point.raw.label == null));
    expect(datasets).toHaveLength(realLabels.length + 1);
    expect(new Set(datasets.map((dataset) => dataset.label)).size).toBe(datasets.length);
    expect(missing?.data).toHaveLength(2);
    expect(missing?.backgroundColor).toBe('#6b7280');
    expect(realLabels).not.toContain(missing?.label);
    if (missingLabel !== undefined) expect(missing?.label).toBe(missingLabel);
    for (const label of realLabels) {
      const category = datasets.find((dataset) => dataset.label === label);
      expect(category?.data).toHaveLength(1);
      expect(category?.backgroundColor).not.toBe('#6b7280');
    }
    expectMatchingColors(points);
  });

  it('renders prototype-like labels, nan, inf and the empty string as real categories', () => {
    // Arbitrary category text must remain visible without throwing or losing points.
    const labels = ['__proto__', 'constructor', 'toString', 'nan', 'inf', ''];
    const points = pointsFor([...labels, null]);
    render(<Charts points={points} />);

    expect(datasets).toHaveLength(labels.length + 1);
    for (const label of labels) {
      const category = datasets.find((dataset) => dataset.label === label);
      expect(category?.data).toHaveLength(1);
      expect(category?.data[0]?.raw.label).toBe(label);
      expect(category?.backgroundColor).not.toBe('#6b7280');
    }
    expectMatchingColors(points);
  });

  it('shows the gray Unlabeled legend when every PCA target value is missing', () => {
    // A single missing group still needs a visible explanation beside its chart.
    render(<PCATab profile={{ pca_data: pointsFor([null, undefined]) }} isPCA3D={false} setIsPCA3D={vi.fn()} downloadChart={vi.fn()} />);

    const legendLabel = screen.getByTitle('Unlabeled');
    expect(legendLabel).toBeVisible();
    const swatch = legendLabel.closest('li')?.querySelector('svg [fill], svg [stroke]');
    expect(swatch?.getAttribute('fill') ?? swatch?.getAttribute('stroke')).toBe('#6b7280');
    expect(datasets[0]?.backgroundColor).toBe('#6b7280');
    expect(datasets[0]?.borderColor).toBe('#6b7280');
    expect(datasets[0]?.data).toHaveLength(2);
  });

  it('keeps map category colors and a filterable legend beyond twenty categories', () => {
    // Dense maps must keep their category meanings instead of silently switching to blue.
    const points = pointsFor(Array.from({ length: 21 }, (_, index) => `category-${index}`));
    render(<Charts points={points} />);

    expect(screen.getByRole('searchbox', { name: 'Filter legend groups' })).toBeVisible();
    expect(new Set([...mapColors().values()]).size).toBeGreaterThan(1);
    expectMatchingColors(points);
  });

  it('keeps an uncolored map blue and coordinate-only when no target is selected', () => {
    // Absence of a target must not be presented as missing target values.
    render(<GeospatialTab profile={{ geospatial: {
      min_lat: 1, min_lon: 2, max_lat: 3, max_lon: 4,
      sample_points: [{ lat: 1, lon: 2 }, { lat: 3, lon: 4 }],
    } }} />);

    expect(screen.getAllByTestId('map-point')).toHaveLength(2);
    expect(screen.queryByTitle('Unlabeled')).not.toBeInTheDocument();
    expect(screen.queryByText('Label:')).not.toBeInTheDocument();
    expect([...mapColors().values()]).toEqual(['#3b82f6', '#3b82f6']);
  });
});
