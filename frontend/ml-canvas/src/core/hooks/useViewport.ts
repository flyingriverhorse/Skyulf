import { useEffect, useState } from 'react';

// Tailwind's `lg` breakpoint. Below this we collapse to read-only because
// the canvas + two 256/320 px sidebars + Toolbar can't all fit comfortably.
export const TABLET_MAX_WIDTH = 1024;
// Tailwind's `md` breakpoint. Reserved for future mobile-specific tweaks.
export const MOBILE_MAX_WIDTH = 768;

export interface ViewportInfo {
  width: number;
  isMobile: boolean;
  isTablet: boolean; // true for both phone- and tablet-sized viewports
  isDesktop: boolean;
}

const readViewport = (): ViewportInfo => {
  // SSR / vitest jsdom safety: assume desktop when window is missing.
  const w = typeof window === 'undefined' ? 1280 : window.innerWidth;
  return {
    width: w,
    isMobile: w < MOBILE_MAX_WIDTH,
    isTablet: w < TABLET_MAX_WIDTH,
    isDesktop: w >= TABLET_MAX_WIDTH,
  };
};

/**
 * Track the current viewport width and derive coarse breakpoint flags.
 * Listens to `resize` once and rAF-throttles updates so a long drag
 * across the window edge doesn't trigger one re-render per pixel.
 */
export function useViewport(): ViewportInfo {
  const [info, setInfo] = useState<ViewportInfo>(readViewport);

  useEffect(() => {
    let raf = 0;
    const onResize = (): void => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setInfo(readViewport()));
    };
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(raf);
    };
  }, []);

  return info;
}
