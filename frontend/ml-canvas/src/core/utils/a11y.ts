/**
 * keyboard-equivalent helper for clickable non-button elements.
 *
 * Spreading `clickableProps(handler)` onto a `<div>` / `<li>` etc. gives it:
 *   - `role="button"` so AT announces it as actionable,
 *   - `tabIndex={0}` so keyboard users can focus it,
 *   - `onClick={handler}`,
 *   - `onKeyDown` that fires `handler` on Enter / Space.
 *
 * Use on real interactive cards (table rows, list items, tree nodes).
 * Backdrops, dismiss-zones, and stopPropagation wrappers should NOT use
 * this — disable the lint rule at those sites with a justification.
 */
import type { KeyboardEvent } from 'react';

export function onActivateKey<T extends Element>(
  handler: (e: KeyboardEvent<T>) => void,
) {
  return (e: KeyboardEvent<T>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handler(e);
    }
  };
}

export function clickableProps<T extends Element>(handler: () => void) {
  return {
    role: 'button' as const,
    tabIndex: 0,
    onClick: handler,
    onKeyDown: (e: KeyboardEvent<T>) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handler();
      }
    },
  };
}
