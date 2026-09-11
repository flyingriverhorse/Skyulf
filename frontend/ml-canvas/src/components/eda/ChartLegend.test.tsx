import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ChartLegend, type ChartLegendEntry } from './ChartLegend';

const makeEntries = (count: number): ChartLegendEntry[] => {
  const shapes: ChartLegendEntry['shape'][] = ['circle', 'square', 'diamond', 'cross', 'x'];
  return Array.from({ length: count }, (_, i) => ({
    label: `Group ${i}`,
    color: '#8884d8',
    shape: shapes[i % shapes.length]!,
  }));
};

describe('ChartLegend', () => {
  it('renders nothing when there are no entries', () => {
    const { container } = render(<ChartLegend entries={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders a distinct shape per entry so color is never the only encoding', () => {
    const entries = makeEntries(3);
    const { container } = render(<ChartLegend entries={entries} />);

    const svgs = container.querySelectorAll('svg');
    expect(svgs).toHaveLength(3);
    // Each shape renders a different SVG primitive (circle/polygon/rect/path).
    const shapeTags = Array.from(svgs).map((svg) => svg.firstElementChild?.tagName);
    expect(new Set(shapeTags).size).toBeGreaterThan(1);
  });

  it('distinguishes the upright cross from the diagonal x in the legend', () => {
    // Plus and diagonal markers must mean the same thing in the legend and either chart engine.
    const entries: ChartLegendEntry[] = [
      { label: 'Plus', color: '#8884d8', shape: 'cross' },
      { label: 'Diagonal', color: '#82ca9d', shape: 'x' },
    ];
    render(<ChartLegend entries={entries} />);

    const plus = screen.getByTitle('Plus').closest('li')?.querySelector('path');
    const diagonal = screen.getByTitle('Diagonal').closest('li')?.querySelector('path');
    expect(plus).toHaveAttribute('d', 'M6 1 L6 11 M1 6 L11 6');
    expect(diagonal).toHaveAttribute('d', 'M2 2 L10 10 M10 2 L2 10');
    expect(plus).toHaveAttribute('stroke', '#8884d8');
    expect(diagonal).toHaveAttribute('stroke', '#82ca9d');
  });

  it('stays visible (never hides) once the group count is large, and becomes filterable instead', () => {
    const entries = makeEntries(25);
    render(<ChartLegend entries={entries} />);

    expect(screen.getAllByText(/^Group \d+$/)).toHaveLength(25);
    const filterInput = screen.getByLabelText(/filter legend groups/i);
    expect(filterInput).toBeInTheDocument();

    fireEvent.change(filterInput, { target: { value: 'Group 1' } });
    // "Group 1" itself plus "Group 10".."Group 19" all match the substring.
    expect(screen.getAllByText(/^Group 1\d*$/).length).toBeGreaterThan(0);
    expect(screen.queryByText('Group 2')).not.toBeInTheDocument();
  });

  it('shows a no-match message rather than silently rendering nothing', () => {
    render(<ChartLegend entries={makeEntries(20)} />);
    fireEvent.change(screen.getByLabelText(/filter legend groups/i), { target: { value: 'zzz-no-match' } });
    expect(screen.getByText(/no groups match/i)).toBeInTheDocument();
  });
});
