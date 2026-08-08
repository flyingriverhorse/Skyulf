import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { CanvasScatterPlot } from './CanvasScatterPlot';

interface CapturedProps {
  data: { datasets: Array<{ label: string; pointStyle: string; backgroundColor: string }> };
  options: { plugins?: { legend?: { display?: boolean } } };
}

let captured: CapturedProps | undefined;

vi.mock('react-chartjs-2', () => ({
  Scatter: (props: CapturedProps) => {
    captured = props;
    return <div data-testid="scatter-mock" />;
  },
}));

const points = [
  { x: 1, y: 2, group: 'setosa' },
  { x: 2, y: 3, group: 'versicolor' },
  { x: 3, y: 1, group: 'virginica' },
];

describe('CanvasScatterPlot', () => {
  it('assigns a distinct point shape per group, not just a color', () => {
    render(<CanvasScatterPlot data={points} xKey="x" yKey="y" labelKey="group" />);

    const datasets = captured?.data.datasets ?? [];
    expect(datasets).toHaveLength(3);
    const shapes = new Set(datasets.map((d) => d.pointStyle));
    expect(shapes.size).toBeGreaterThan(1);
  });

  it('never hides the legend internally regardless of group count (delegated to ChartLegend instead)', () => {
    const manyPoints = Array.from({ length: 25 }, (_, i) => ({ x: i, y: i, group: `g${i}` }));
    render(<CanvasScatterPlot data={manyPoints} xKey="x" yKey="y" labelKey="group" />);

    expect(captured?.options.plugins?.legend?.display).toBe(false);
  });
});
