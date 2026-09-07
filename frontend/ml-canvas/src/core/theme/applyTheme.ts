let pendingFrame: number | undefined;

/** Apply all theme colors in one paint without leaving component transitions disabled. */
export function applyTheme(dark: boolean): void {
  const root = document.documentElement;
  if (pendingFrame !== undefined) cancelAnimationFrame(pendingFrame);
  root.classList.add('theme-switching');
  root.classList.toggle('dark', dark);
  // Commit the new styles while transitions are suppressed, including portals.
  void getComputedStyle(root).color;
  pendingFrame = requestAnimationFrame(() => {
    pendingFrame = requestAnimationFrame(() => {
      root.classList.remove('theme-switching');
      pendingFrame = undefined;
    });
  });
  try {
    localStorage.setItem('skyulf-theme', dark ? 'dark' : 'light');
  } catch { /* Theme switching also works when storage is unavailable. */ }
}
