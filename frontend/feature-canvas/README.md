# ML workflow canvas frontend

This package contains the React micro-frontend that powers the interactive ML workflow canvas.

## Getting started

```bash
cd frontend/feature-canvas
npm install
npm run dev
```

The development server defaults to <http://localhost:5173>. During development you can either:

- open the dev server directly, or
- configure FastAPI to proxy `/ml-workflow` to the dev server (recommended for testing authenticated flows).

## Production build

```bash
cd frontend/feature-canvas
npm run build
```

The build command outputs static assets into `static/feature_canvas/`. When you restart FastAPI, the `/ml-workflow` route will automatically serve the fresh bundle.

## Canvas interactions

- Drag from the right handle of a node to the left handle of another to create connections manually.
- Automatic "Proximity Connect" links have been removed, so edges are only created on demand by the user.
- Use the × button that appears mid-connection to quickly delete any edge.

## Project structure

- `src/` – React components and styling for the canvas
- `src/api.ts` – Fetch helpers that communicate with the FastAPI ML workflow endpoints
- `vite.config.ts` – Vite setup that writes compiled files straight into `static/feature_canvas`

Feel free to evolve this package (state management, routing, etc.) as the canvas becomes more capable.
