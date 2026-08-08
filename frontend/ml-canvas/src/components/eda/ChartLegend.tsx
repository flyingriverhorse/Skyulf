import React, { useMemo, useState } from 'react';
import type { ChartMarkerShape } from './chartMarkerShapes';

export interface ChartLegendEntry {
  /** Group label shown next to the swatch. */
  label: string;
  /** Fill color for the swatch. */
  color: string;
  /** Shape used so the group is distinguishable without relying on color. */
  shape: ChartMarkerShape;
}

interface ChartLegendProps {
  entries: ChartLegendEntry[];
  /** Above this count the legend becomes a scrollable, filterable list
   * instead of being hidden outright (see DAT-007: `CanvasScatterPlot`
   * previously hid its legend entirely once a chart had 20+ groups). */
  filterThreshold?: number;
}

const ShapeIcon: React.FC<{ shape: ChartMarkerShape; color: string }> = ({ shape, color }) => {
  const common = { width: 12, height: 12, 'aria-hidden': true as const };
  switch (shape) {
    case 'triangle':
      return (
        <svg {...common} viewBox="0 0 12 12">
          <polygon points="6,1 11,11 1,11" fill={color} />
        </svg>
      );
    case 'square':
      return (
        <svg {...common} viewBox="0 0 12 12">
          <rect x="1" y="1" width="10" height="10" fill={color} />
        </svg>
      );
    case 'diamond':
      return (
        <svg {...common} viewBox="0 0 12 12">
          <polygon points="6,0 12,6 6,12 0,6" fill={color} />
        </svg>
      );
    case 'star':
      return (
        <svg {...common} viewBox="0 0 12 12">
          <polygon points="6,0 7.4,4.2 12,4.2 8.3,6.9 9.7,11.1 6,8.4 2.3,11.1 3.7,6.9 0,4.2 4.6,4.2" fill={color} />
        </svg>
      );
    case 'cross':
      return (
        <svg {...common} viewBox="0 0 12 12">
          <path d="M2 2 L10 10 M10 2 L2 10" stroke={color} strokeWidth={2} />
        </svg>
      );
    case 'circle':
    default:
      return (
        <svg {...common} viewBox="0 0 12 12">
          <circle cx="6" cy="6" r="5" fill={color} />
        </svg>
      );
  }
};

/**
 * Persistent, always-visible legend for grouped charts. Every entry pairs a
 * color swatch with a distinct shape so group identity survives grayscale
 * printing, color-vision deficiency, or dense hover-only tooltips. Renders as
 * a scrollable/filterable list rather than being hidden once the group count
 * gets large, so legend coverage never silently disappears.
 */
export const ChartLegend: React.FC<ChartLegendProps> = ({ entries, filterThreshold = 12 }) => {
  const [query, setQuery] = useState('');

  const visibleEntries = useMemo(() => {
    if (!query.trim()) return entries;
    const needle = query.trim().toLowerCase();
    return entries.filter((entry) => entry.label.toLowerCase().includes(needle));
  }, [entries, query]);

  if (entries.length === 0) return null;

  const isFilterable = entries.length > filterThreshold;

  return (
    <div className="mt-3 border border-gray-200 dark:border-gray-700 rounded-md p-2">
      {isFilterable && (
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Filter ${entries.length} legend groups…`}
          aria-label="Filter legend groups"
          className="mb-2 w-full text-xs rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800 dark:text-white px-2 py-1"
        />
      )}
      <ul
        className={`flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-gray-700 dark:text-gray-300 ${
          isFilterable ? 'max-h-32 overflow-y-auto' : ''
        }`}
      >
        {visibleEntries.map((entry) => (
          <li key={entry.label} className="flex items-center gap-1.5">
            <ShapeIcon shape={entry.shape} color={entry.color} />
            <span className="truncate max-w-[10rem]" title={entry.label}>
              {entry.label}
            </span>
          </li>
        ))}
        {visibleEntries.length === 0 && (
          <li className="italic text-gray-400 dark:text-gray-500">No groups match “{query}”.</li>
        )}
      </ul>
    </div>
  );
};
