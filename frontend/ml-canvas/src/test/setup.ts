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

// jsdom 24 + newer Node (which ships its own experimental global
// `localStorage` gated behind `--localstorage-file`) leaves
// `window.localStorage` undefined in this environment instead of a working
// Storage implementation. Several utils (`recentPipelines`,
// `canvasPersistence`) rely on `getItem`/`setItem`/`removeItem`/`clear`, so
// provide a minimal in-memory polyfill — plenty for tests, which never
// need persistence across process restarts.
if (!window.localStorage) {
  class MemoryStorage implements Storage {
    private store = new Map<string, string>();

    get length(): number {
      return this.store.size;
    }

    clear(): void {
      this.store.clear();
    }

    getItem(key: string): string | null {
      return this.store.has(key) ? this.store.get(key)! : null;
    }

    key(index: number): string | null {
      return Array.from(this.store.keys())[index] ?? null;
    }

    removeItem(key: string): void {
      this.store.delete(key);
    }

    setItem(key: string, value: string): void {
      this.store.set(key, String(value));
    }
  }

  Object.defineProperty(window, 'localStorage', {
    value: new MemoryStorage(),
    writable: true,
    configurable: true,
  });
}
