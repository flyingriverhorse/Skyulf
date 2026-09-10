# ML Canvas

This is the Frontend for the Skyulf.

## Getting Started
(Instructions to be added after project initialization)

## Plotly bundle

`package.json` installs the official `plotly.js-gl3d-dist-min` distribution under
the `plotly.js` name using an [npm alias](https://docs.npmjs.com/cli/v11/using-npm/package-spec/#aliases).
This satisfies the React wrapper's peer dependency with the same slim bundle
used for rendering and PNG export, avoiding the unused full Plotly dependencies.

Import `Plot` and `Plotly` through `src/core/plotly.ts`. Keep the wrapper's
`react-plotly.js/factory` entry in both that module and Vite's `manualChunks`;
the default wrapper entry requires a `dist/plotly` file from the full package.
For trace types outside GL3D, update the alias to a suitable official bundle
and verify rendering, export and bundle budgets. `e2e/plotly-bundle.spec.ts`
covers a real 3D PCA chart, its PNG download and the return to 2D.
