/**
 * Dev-only test hooks. Imported for side effects from `main.tsx` and
 * tree-shaken out of production builds via the `import.meta.env.DEV`
 * guard (Vite replaces this at build time, so the entire block becomes
 * dead code when minified for production).
 *
 * Used by Playwright (`e2e/`) to seed the zustand graph store
 * deterministically without simulating drag-and-drop / handle
 * connection through the React Flow renderer (which is famously
 * fragile under headless automation).
 *
 * Public surface:
 *   window.__skyulfTest.graphStore.getState() / setGraph(...)
 *
 * NOT a public API. Do not use from production code.
 */
import { useGraphStore } from '../core/store/useGraphStore';

declare global {
  interface Window {
    __skyulfTest?: {
      graphStore: typeof useGraphStore;
    };
  }
}

if (import.meta.env.DEV) {
  window.__skyulfTest = { graphStore: useGraphStore };
}

export {};
