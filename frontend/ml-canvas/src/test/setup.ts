// Global test setup. Loaded before each Vitest worker.
// - `@testing-library/jest-dom` adds DOM-aware matchers (`toBeInTheDocument`, etc.).
// - `cleanup()` after each test prevents leaking renders between cases.
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

// Stub `matchMedia` and `ResizeObserver` so components that probe them
// (Radix UI, React Flow internals) don't blow up under jsdom.
if (!window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}

if (!('ResizeObserver' in window)) {
  // @ts-expect-error — minimal jsdom shim, not a full ResizeObserver impl.
  window.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
