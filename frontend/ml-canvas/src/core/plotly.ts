// Centralized Plotly entry point.
//
// We ship the slim `plotly.js-gl3d-dist-min` build (~542 KB gzip vs
// ~2.85 MB gzip for the full bundle). It covers every trace type the
// app actually uses (audited 2026-04-27): scatter / scattergl / bar /
// histogram / heatmap / scatter3d / surface / mesh3d. We have ZERO
// usages of geo/mapbox, sankey, treemap, sunburst, parcoords, or
// finance traces, so the slim build is a complete swap.
//
// package.json aliases `plotly.js` to the official GL3D bundle. This
// satisfies the React wrapper's peer dependency without installing
// the unused full Plotly package and its mapping dependencies.
// The factory entry point uses this same slim instance for rendering.
//
// If a future feature needs an unsupported trace type, swap this
// alias to an appropriate official bundle and verify rendering/export
// and bundle budgets. The default React wrapper entry requires full
// Plotly's dist/plotly path, so keep the factory entry point below.
import Plotly from 'plotly.js';
import createPlotlyComponent from 'react-plotly.js/factory';

export const Plot = createPlotlyComponent(Plotly);
export { Plotly };
export default Plot;
