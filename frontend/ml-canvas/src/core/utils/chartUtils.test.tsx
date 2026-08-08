import { beforeEach, describe, expect, it, vi } from 'vitest';

const toPng = vi.fn(async () => 'data:image/png;base64,stub');

vi.mock('html-to-image', () => ({
  toPng: (...args: unknown[]) => toPng(...(args as [])),
}));

vi.mock('../plotly', () => ({
  Plotly: { downloadImage: vi.fn() },
}));

vi.mock('../theme/chartTheme', () => ({
  isDarkModeActive: () => false,
  getChartTheme: () => ({}),
}));

import { downloadChart } from './chartUtils';

/**
 * The Clustering tab's `#clustering-chart` container holds only styled `div`s and
 * text — no `svg`/`canvas`/Plotly root — yet its PNG export is a supported feature.
 * A renderable-chart precondition inside `downloadChart` would silently break it.
 */
describe('downloadChart', () => {
  beforeEach(() => {
    toPng.mockClear();
    document.body.innerHTML = '';
  });

  it('exports a container built from plain HTML, as the Clustering tab uses', async () => {
    document.body.innerHTML = `
      <div id="clustering-chart">
        <div>Algorithm Summary</div>
        <div>Cluster 0 — 62.5% (5)</div>
      </div>
    `;

    await downloadChart('clustering-chart', 'clustering-analysis', 'Clustering Segmentation');

    expect(toPng).toHaveBeenCalledTimes(1);
  });

  it('does nothing when the target container is absent', async () => {
    await downloadChart('missing-chart', 'nope');

    expect(toPng).not.toHaveBeenCalled();
  });
});
