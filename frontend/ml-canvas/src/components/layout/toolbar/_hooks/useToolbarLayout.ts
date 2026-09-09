import { useEffect, useRef, useState } from 'react';
import { useSidebarOpen } from '../../../../core/hooks/useSidebarOpen';

/** Measure the canvas pane, including space consumed by adjacent panels. */
export function useToolbarLayout() {
  const isSidebarOpen = useSidebarOpen();
  // CAN-005: collapse secondary actions based on the *actual* Flow-pane
  // width (the toolbar's absolute-positioned containing block), not a
  // Tailwind viewport breakpoint. `xl:` media queries only track the
  // browser window, so at a wide viewport (e.g. 1440px) with the
  // Properties panel open — which narrows this same pane without
  // changing the window width — the old `hidden xl:flex` buttons stayed
  // visible and rendered on top of the left cluster (Undo/Clear).
  // Measuring the real container closes that gap at every panel
  // combination instead of only at the one viewport width devs tested.
  const toolbarRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const COMPACT_WIDTH = 1280;
  const isCompact = containerWidth < COMPACT_WIDTH;
  const isNarrow = containerWidth < 720;
  const hideUndoRedo = containerWidth < 380;
  useEffect(() => {
    const container = toolbarRef.current?.parentElement;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setContainerWidth(entry.contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return { isSidebarOpen, toolbarRef, isCompact, isNarrow, hideUndoRedo };
}
