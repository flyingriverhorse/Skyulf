// Centralized Plotly entry point.
//
// We ship the slim `plotly.js-gl3d-dist-min` build (~542 KB gzip vs
// ~2.85 MB gzip for the full bundle). It covers every trace type the
// app actually uses (audited 2026-04-27): scatter / scattergl / bar /
// histogram / heatmap / scatter3d / surface / mesh3d. We have ZERO
// usages of geo/mapbox, sankey, treemap, sunburst, parcoords, or
// finance traces, so the slim build is a complete swap.
//
// `react-plotly.js` is wired through its `factory` entry point so it
// uses our slim Plotly instance instead of pulling its own copy of
// the full `plotly.js` package transitively.
//
// If a future feature needs an unsupported trace type, swap this
// single import to `plotly.js-dist-min` and the rest of the app keeps
// working unchanged.
import Plotly from 'plotly.js-gl3d-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';

export const Plot = createPlotlyComponent(Plotly);
export { Plotly };
export default Plot;
