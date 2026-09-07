import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';

/** Keep the action available while reserving the full explanation for the end of settings. */
export function TrainingActionFooter({ children, details }: { children: ReactNode; details: ReactNode }) {
  const sentinel = useRef<HTMLDivElement>(null);
  const detailsRef = useRef<HTMLDivElement>(null);
  const [nearEnd, setNearEnd] = useState(false);
  const [detailsFocused, setDetailsFocused] = useState(false);
  const expanded = nearEnd || detailsFocused;

  useEffect(() => {
    const marker = sentinel.current;
    if (!marker) return;
    if (typeof IntersectionObserver === 'undefined') {
      setNearEnd(true);
      return;
    }
    // Observe the natural footer position, since the sticky action is always visible.
    let root = marker.parentElement;
    while (root && !/auto|scroll/.test(getComputedStyle(root).overflowY)) root = root.parentElement;
    const observer = new IntersectionObserver(([entry]) => { if (entry) setNearEnd(entry.isIntersecting); }, { root });
    observer.observe(marker);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    if (detailsRef.current) detailsRef.current.inert = !expanded;
  }, [expanded]);

  return <>
    <div ref={sentinel} className="h-px shrink-0" aria-hidden="true" />
    <div data-testid="training-action-footer" data-expanded={expanded}
      className="sticky bottom-0 z-10 flex flex-col items-center gap-2 border-t border-border bg-background py-2">
      {children}
      <div className={`grid w-full transition-[grid-template-rows,opacity] duration-200 motion-reduce:transition-none ${expanded ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0 -mt-2'}`}
        aria-hidden={!expanded} ref={detailsRef}
        onFocusCapture={() => setDetailsFocused(true)}
        onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget)) setDetailsFocused(false); }}>
        <div className="min-h-0 overflow-hidden">
          <div className="flex flex-col items-center gap-2">{details}</div>
        </div>
      </div>
    </div>
  </>;
}
