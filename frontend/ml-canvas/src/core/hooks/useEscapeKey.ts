import { useEffect, useRef } from 'react';

export function useEscapeKey(onEscape: () => void, enabled: boolean = true): void {
  const callbackRef = useRef(onEscape);
  callbackRef.current = onEscape;

  useEffect(() => {
    if (!enabled) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') callbackRef.current();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [enabled]);
}
