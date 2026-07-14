import { useEffect, useRef, useState } from 'react';

/**
 * Shared responsive-layout hook for node settings panels: reports whether
 * the panel's own container is wider than `threshold` px, so the caller can
 * switch between a stacked single-column layout (narrow properties panel)
 * and a 2-column layout (panel expanded/undocked). Replaces ~14 copies of
 * the same `containerRef` + `ResizeObserver` boilerplate that had drifted
 * across node settings components.
 */
export function useIsWideContainer<T extends HTMLElement = HTMLDivElement>(
  threshold = 450,
): [React.RefObject<T>, boolean] {
  const containerRef = useRef<T>(null);
  const [isWide, setIsWide] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setIsWide(entry.contentRect.width > threshold);
      }
    });
    observer.observe(containerRef.current);
    return () => { observer.disconnect(); };
  }, [threshold]);

  return [containerRef, isWide];
}
