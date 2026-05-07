/**
 * Canvas → PNG / SVG export. Uses `html-to-image` against the live
 * React Flow canvas so all nodes render exactly as on screen.
 *
 * Two helpers:
 *  - `exportCanvasToPng(filename)` — high-DPI raster (2x pixel ratio).
 *  - `exportCanvasToSvg(filename)` — vector, smaller for line-only graphs.
 *
 * Both target the outer `.react-flow` container (which has fixed pixel
 * dimensions) rather than the inner `.react-flow__viewport` (which is
 * clipped by `overflow: hidden`). Before each capture we temporarily
 * apply a "fit-all-nodes" transform so off-screen nodes are scrolled
 * into view, then restore the original transform in a `finally` block.
 */
import { toPng, toSvg } from 'html-to-image';

const CANVAS_SELECTOR = '.react-flow';
const VIEWPORT_SELECTOR = '.react-flow__viewport';
const NODE_SELECTOR = '.react-flow__node';

/** Padding fraction: content fills at most this fraction of the canvas. */
const EXPORT_PADDING = 0.9;

const findCanvasAndViewport = (): [HTMLElement, HTMLElement] | null => {
  const canvas = document.querySelector<HTMLElement>(CANVAS_SELECTOR);
  if (!canvas) return null;
  const vp = canvas.querySelector<HTMLElement>(VIEWPORT_SELECTOR);
  if (!vp) return null;
  return [canvas, vp];
};

const triggerDownload = (dataUrl: string, filename: string): void => {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

/**
 * Compute a CSS transform string that fits every node in the canvas
 * container. Reads the current viewport transform to convert screen
 * coordinates back to flow coordinates, then emits a new
 * `translate(…) scale(…)` that centres all nodes with padding.
 *
 * Returns null when there are no nodes or the bounds are degenerate.
 */
function computeFitTransform(
  canvas: HTMLElement,
  vpEl: HTMLElement,
): string | null {
  const nodes = vpEl.querySelectorAll<HTMLElement>(NODE_SELECTOR);
  if (nodes.length === 0) return null;

  // Parse the current viewport transform produced by React Flow:
  // "translate(vpXpx, vpYpx) scale(vpZoom)"
  const ct = vpEl.style.transform;
  const tMatch = ct.match(/translate\(([^,]+)px,\s*([^)]+)px\)/);
  const sMatch = ct.match(/scale\(([^)]+)\)/);
  const vpX = tMatch ? parseFloat(tMatch[1] ?? '0') : 0;
  const vpY = tMatch ? parseFloat(tMatch[2] ?? '0') : 0;
  const vpZoom = sMatch ? parseFloat(sMatch[1] ?? '1') : 1;

  const canvasRect = canvas.getBoundingClientRect();

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const node of nodes) {
    const r = node.getBoundingClientRect();
    // getBoundingClientRect is in viewport (screen) space; subtract
    // canvas origin to get coords relative to the canvas container.
    const relLeft = r.left - canvasRect.left;
    const relTop  = r.top  - canvasRect.top;
    // Convert screen-relative position back to React Flow coordinates.
    const fx = (relLeft - vpX) / vpZoom;
    const fy = (relTop  - vpY) / vpZoom;
    const fw = r.width  / vpZoom;
    const fh = r.height / vpZoom;
    minX = Math.min(minX, fx);
    minY = Math.min(minY, fy);
    maxX = Math.max(maxX, fx + fw);
    maxY = Math.max(maxY, fy + fh);
  }

  if (!isFinite(minX)) return null;

  const cW = canvas.offsetWidth;
  const cH = canvas.offsetHeight;
  const contentW = maxX - minX;
  const contentH = maxY - minY;
  if (contentW <= 0 || contentH <= 0) return null;

  const zoom = Math.min(
    (cW * EXPORT_PADDING) / contentW,
    (cH * EXPORT_PADDING) / contentH,
    2,
  );
  const newX = (cW - contentW * zoom) / 2 - minX * zoom;
  const newY = (cH - contentH * zoom) / 2 - minY * zoom;
  return `translate(${newX}px, ${newY}px) scale(${zoom})`;
}

/**
 * Common html-to-image options. White background so dark-mode users
 * still get a screenshot that works on light docs/issue trackers.
 * Control panels (.react-flow__panel, .react-flow__controls) are
 * excluded — they're UI chrome, not part of the pipeline diagram.
 */
const isExportPanel = (el: HTMLElement): boolean =>
  el.classList?.contains?.('react-flow__panel') ||
  el.classList?.contains?.('react-flow__controls') ||
  el.classList?.contains?.('react-flow__attribution') ||
  el.classList?.contains?.('no-export');

const baseOptions = {
  backgroundColor: '#ffffff',
  filter: (node: HTMLElement) => !isExportPanel(node),
  cacheBust: true,
};

export async function exportCanvasToPng(
  filename = 'skyulf-canvas.png',
): Promise<string | null> {
  const pair = findCanvasAndViewport();
  if (!pair) return null;
  const [canvas, vpEl] = pair;

  const originalTransform = vpEl.style.transform;
  const fitTransform = computeFitTransform(canvas, vpEl);
  if (fitTransform) {
    vpEl.style.transform = fitTransform;
    // One rAF so the browser commits the new transform before capture.
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
  try {
    const dataUrl = await toPng(canvas, { ...baseOptions, pixelRatio: 2 });
    triggerDownload(dataUrl, filename);
    return dataUrl;
  } finally {
    if (fitTransform) vpEl.style.transform = originalTransform;
  }
}

export async function exportCanvasToSvg(
  filename = 'skyulf-canvas.svg',
): Promise<string | null> {
  const pair = findCanvasAndViewport();
  if (!pair) return null;
  const [canvas, vpEl] = pair;

  const originalTransform = vpEl.style.transform;
  const fitTransform = computeFitTransform(canvas, vpEl);
  if (fitTransform) {
    vpEl.style.transform = fitTransform;
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  }
  try {
    const dataUrl = await toSvg(canvas, baseOptions);
    triggerDownload(dataUrl, filename);
    return dataUrl;
  } finally {
    if (fitTransform) vpEl.style.transform = originalTransform;
  }
}
