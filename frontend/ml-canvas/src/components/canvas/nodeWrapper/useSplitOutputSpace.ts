import { useLayoutEffect, useRef } from 'react';
import type { NodeDefinition } from '../../../core/types/nodes';

/** Reserve output-label width in canvas pixels, independent of zoom. */
export function useSplitOutputSpace(definition: NodeDefinition<unknown> | undefined) {
  const splitBodyRef = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const body = splitBodyRef.current;
    if (!body || (definition?.outputs.length ?? 0) < 2) return;
    const labels = Array.from(body.querySelectorAll<HTMLElement>('[data-output-label]'));
    const reserveLabelSpace = () => {
      // offsetWidth excludes canvas zoom and the selected-card transform.
      // Include the labels' right-4 inset and 8px of clearance for the summary.
      const width = Math.max(0, ...labels.map(label => label.offsetWidth)) + 24;
      body.style.setProperty('--split-output-space', `${width}px`);
    };
    reserveLabelSpace();
    const observer = new ResizeObserver(reserveLabelSpace);
    labels.forEach(label => observer.observe(label));
    return () => observer.disconnect();
  }, [definition]);
  return splitBodyRef;
}
