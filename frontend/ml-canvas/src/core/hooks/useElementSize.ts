import { useState, useEffect, useCallback } from 'react';

export function useElementSize<T extends HTMLElement = HTMLDivElement>(): [
  (node: T | null) => void,
  { width: number; height: number }
] {
  const [ref, setRef] = useState<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  const handleResize = useCallback((entries: ResizeObserverEntry[]) => {
    if (!entries || entries.length === 0) return;
    const { width, height } = entries[0].contentRect;
    setSize({ width, height });
  }, []);

  useEffect(() => {
    if (!ref) return;
    const observer = new ResizeObserver(handleResize);
    observer.observe(ref);
    return () => observer.disconnect();
  }, [ref, handleResize]);

  return [setRef, size];
}
