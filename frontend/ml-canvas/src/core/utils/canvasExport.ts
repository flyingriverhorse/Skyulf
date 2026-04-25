/**
 * Canvas → PNG / SVG export. Uses `html-to-image` against the live
 * React Flow viewport so existing nodes / edges / styles render
 * exactly as on screen (no separate SVG re-emit needed).
 *
 * Two helpers:
 *  - `exportCanvasToPng(filename)` — high-DPI raster (2x pixel ratio).
 *  - `exportCanvasToSvg(filename)` — vector, smaller for line-only graphs.
 *
 * Both rely on React Flow's standard `.react-flow__viewport` element
 * being present in the DOM when called. If it isn't (e.g. the user
 * triggered the export from a non-canvas view), the function
 * resolves to `null` and the caller can surface a toast.
 */
import { toPng, toSvg } from 'html-to-image';

const VIEWPORT_SELECTOR = '.react-flow__viewport';

const findViewport = (): HTMLElement | null =>
  document.querySelector<HTMLElement>(VIEWPORT_SELECTOR);

const triggerDownload = (dataUrl: string, filename: string): void => {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

/**
 * Common html-to-image options. White background so dark-mode users
 * still get a screenshot that works on light docs/issue trackers.
 */
const baseOptions = {
  backgroundColor: '#ffffff',
  // Skip nodes the user marked .no-export (e.g. selection rings,
  // floating toolbars) if any get added later.
  filter: (node: HTMLElement) => !node.classList?.contains?.('no-export'),
  cacheBust: true,
};

export async function exportCanvasToPng(
  filename = 'skyulf-canvas.png',
): Promise<string | null> {
  const viewport = findViewport();
  if (!viewport) return null;
  const dataUrl = await toPng(viewport, {
    ...baseOptions,
    pixelRatio: 2,
  });
  triggerDownload(dataUrl, filename);
  return dataUrl;
}

export async function exportCanvasToSvg(
  filename = 'skyulf-canvas.svg',
): Promise<string | null> {
  const viewport = findViewport();
  if (!viewport) return null;
  const dataUrl = await toSvg(viewport, baseOptions);
  triggerDownload(dataUrl, filename);
  return dataUrl;
}
