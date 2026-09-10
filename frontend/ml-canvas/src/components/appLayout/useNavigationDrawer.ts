import { useEffect, useRef, useState } from 'react';

/** Own the responsive drawer lifetime and its opener/first-link focus behavior. */
export function useNavigationDrawer(pathname: string, isMobile: boolean) {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement | null>(null);
  const asideRef = useRef<HTMLElement | null>(null);

  // Close the drawer on route change (a nav click just navigated away, so
  // there's nothing left to show it for) and whenever the viewport grows
  // back to a size where the sidebar is a persistent column again.
  useEffect(() => {
    setIsDrawerOpen(false);
  }, [pathname]);
  useEffect(() => {
    if (!isMobile) setIsDrawerOpen(false);
  }, [isMobile]);

  // Escape closes the drawer and returns focus to the button that opened
  // it; focus moves into the drawer's first link on open so keyboard and
  // screen-reader users land somewhere useful. This is a minimal, self-
  // contained version of the containment behavior FND-002 will later make
  // a shared contract across every shell overlay.
  useEffect(() => {
    if (!isDrawerOpen) return;
    const raf = window.requestAnimationFrame(() => {
      asideRef.current?.querySelector<HTMLAnchorElement>('a[href]')?.focus();
    });
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsDrawerOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    const openerButton = menuButtonRef.current;
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener('keydown', onKeyDown);
      openerButton?.focus();
    };
  }, [isDrawerOpen]);

  return { isDrawerOpen, setIsDrawerOpen, menuButtonRef, asideRef };
}
