import { useEffect, useRef, type RefObject } from 'react';

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]',
].join(',');

const getFocusable = (root: HTMLElement): HTMLElement[] => {
  const nodes = Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
  return nodes.filter((el) => {
    if (el.hasAttribute('disabled')) return false;
    if (el.getAttribute('aria-hidden') === 'true') return false;
    if (el.tabIndex < 0) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    return true;
  });
};

interface UseModalFocusOptions {
  isOpen: boolean;
  containerRef: RefObject<HTMLElement | null>;
  initialFocusRef?: RefObject<HTMLElement | null>;
  returnFocusRef?: RefObject<HTMLElement | null>;
}

/**
 * Traps focus inside an open modal and restores it to the opener on close.
 */
export function useModalFocus({
  isOpen,
  containerRef,
  initialFocusRef,
  returnFocusRef,
}: UseModalFocusOptions): void {
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return undefined;
    previouslyFocusedRef.current = (document.activeElement as HTMLElement | null) ?? null;

    const raf = window.requestAnimationFrame(() => {
      const container = containerRef.current;
      if (!container) return;
      const target = initialFocusRef?.current ?? getFocusable(container)[0] ?? container;
      target.focus();
    });
    const fallback = returnFocusRef?.current ?? null;

    return () => {
      window.cancelAnimationFrame(raf);
      const previous = previouslyFocusedRef.current;
      const next =
        previous && document.contains(previous) && previous !== document.body
          ? previous
          : fallback && document.contains(fallback)
            ? fallback
            : null;

      try {
        next?.focus();
      } catch {
        // Ignore detached or non-focusable elements during teardown.
      }
    };
  }, [isOpen, containerRef, initialFocusRef, returnFocusRef]);

  useEffect(() => {
    if (!isOpen) return undefined;

    const handler = (e: KeyboardEvent): void => {
      if (e.key !== 'Tab') return;
      const container = containerRef.current;
      if (!container) return;

      const focusables = getFocusable(container);
      if (focusables.length === 0) {
        e.preventDefault();
        container.focus();
        return;
      }

      const first = focusables[0]!;
      const last = focusables[focusables.length - 1]!;
      const active = document.activeElement as HTMLElement | null;

      if (e.shiftKey) {
        if (!container.contains(active) || active === first) {
          e.preventDefault();
          last.focus();
        }
        return;
      }

      if (!container.contains(active) || active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [isOpen, containerRef]);
}
