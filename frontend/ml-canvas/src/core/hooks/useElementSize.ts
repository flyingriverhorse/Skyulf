import { useState, useEffect, useCallback } from 'react';

export function useElementSize<T extends HTMLElement = HTMLDivElement>(): [
  (node: T | null) => void,
  { width: number; height: number }
] {
  const [ref, setRef] = useState<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  const handleResize = useCallback((entries: ResizeObserverEntry[]) => {
    const first = entries[0];
    if (!first) return;
    const { width, height } = first.contentRect;
    setSize({ width, height });
  }, []);

  useEffect(() => {
    if (!ref) return;
    const observer = new ResizeObserver(handleResize);
    observer.observe(ref);
    return () => { observer.disconnect(); };
  }, [ref, handleResize]);

  return [setRef, size];
}
