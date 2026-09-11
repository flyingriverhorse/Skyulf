import { describe, expect, it } from 'vitest';

import { buildScatterLegendEntries, groupScatterPoints } from './scatterGrouping';

describe('scatter label grouping', () => {
  it('orders mixed-case and accented category labels alphabetically', () => {
    // Letter case and accents must not move categories behind unrelated letters.
    const points = ['zebra', 'Banana', 'apple', '\u00c9clair'].map((label, x) => ({ x, label }));

    expect(groupScatterPoints(points, 'label').map((group) => group.label))
      .toEqual(['apple', 'Banana', '\u00c9clair', 'zebra']);
  });

  it('keeps distinct Unicode spellings styled consistently when collation considers them equal', () => {
    // Canonically equivalent labels stay distinct without assigning styles by input row order.
    const points = ['\u00e9', 'e\u0301'].map((label, x) => ({ x, label }));
    const original = buildScatterLegendEntries(groupScatterPoints(points, 'label'));
    const reordered = buildScatterLegendEntries(groupScatterPoints([...points].reverse(), 'label'));

    expect(original.map((entry) => entry.label)).toEqual(['e\u0301', '\u00e9']);
    expect(reordered).toEqual(original);
    expect(original[0]?.color).not.toBe(original[1]?.color);
  });

  it('keeps category colors and shapes stable when rows are reordered or missing labels are added', () => {
    // Group identity must survive row order changes and unlabeled observations.
    const points = ['Gold', 'Silver', 'Bronze', '10', '2'].map((label, x) => ({ x, label }));
    const original = buildScatterLegendEntries(groupScatterPoints(points, 'label'));
    const reordered = buildScatterLegendEntries(groupScatterPoints([...points].reverse(), 'label'));
    const withMissing = groupScatterPoints([{ x: -1, label: null }, ...points], 'label');

    expect(reordered).toEqual(original);
    expect(buildScatterLegendEntries(withMissing.filter((group) => group.value !== null))).toEqual(original);
    expect(original.map((entry) => entry.label)).toEqual(['10', '2', 'Bronze', 'Gold', 'Silver']);
  });

  it('keeps missing labels separate from real fallback names, empty strings and prototype names', () => {
    // Only null and undefined may merge; every observed category must retain its rows.
    const labels = [null, 'Other', 'Unlabeled', 'Unlabeled (missing)', '', undefined,
      '__proto__', 'constructor', 'toString', 'nan', 'inf', '-inf', '1e309'];
    const points = labels.map((label, x) => Object.freeze({ x, label }));
    const groups = groupScatterPoints(points, 'label');
    const missing = groups.find((group) => group.value === null);

    expect(groups).toHaveLength(labels.length - 1);
    expect(missing).toMatchObject({ color: '#6b7280', shape: 'cross' });
    expect(missing?.points.map((point) => point.x)).toEqual([0, 5]);
    expect(new Set(groups.map((group) => group.label)).size).toBe(groups.length);
    expect(groups.flatMap((group) => group.points).map((point) => point.x).sort((a, b) => a - b))
      .toEqual(points.map((point) => point.x));
    for (const label of labels.filter((value) => value != null)) {
      expect(groups.find((group) => group.value === label)?.points.map((point) => point.label)).toEqual([label]);
    }
    expect(points.map((point) => point.label)).toEqual(labels);
  });

  it('explains an entirely missing target with one neutral legend entry', () => {
    // An all-missing target must still display all coordinates and identify their meaning.
    const points = [{ x: 1, label: null }, { x: 2, label: undefined }];
    const groups = groupScatterPoints(points, 'label');

    expect(groups).toHaveLength(1);
    expect(groups[0]?.points).toEqual(points);
    expect(buildScatterLegendEntries(groups)).toEqual([
      { label: 'Unlabeled', color: '#6b7280', shape: 'cross' },
    ]);
  });

  it('preserves the single uncolored group when no target is selected', () => {
    // Bivariate charts without a color column must not infer groups from unrelated labels.
    const points = [{ x: 1, label: 'a' }, { x: 2, label: null }];
    const groups = groupScatterPoints(points);

    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({ label: 'Data Points', points });
    expect(buildScatterLegendEntries(groups)[0]?.color).not.toBe('#6b7280');
  });

  it('does not invent a missing group when a selected target has no rows', () => {
    // Empty chart input must not produce a misleading unlabeled legend entry.
    expect(groupScatterPoints([], 'label')).toEqual([]);
  });
});
