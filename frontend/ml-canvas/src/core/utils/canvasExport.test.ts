import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { toPng, toSvg } from 'html-to-image';
import { exportCanvasToPng, exportCanvasToSvg } from './canvasExport';

vi.mock('html-to-image', () => ({ toPng: vi.fn(), toSvg: vi.fn() }));

/** Supply screen bounds matching a translated, zoomed React Flow viewport. */
function mountCanvas(): HTMLElement {
  document.body.innerHTML = '<div class="react-flow"><div class="react-flow__viewport"><div class="react-flow__node"></div></div></div>';
  const canvas = document.querySelector<HTMLElement>('.react-flow')!;
  const viewport = canvas.firstElementChild as HTMLElement;
  viewport.style.transform = 'translate(20px, 40px) scale(2)';
  Object.defineProperties(canvas, { offsetWidth: { value: 1000 }, offsetHeight: { value: 600 } });
  vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 100, top: 200 } as DOMRect);
  vi.spyOn(viewport.firstElementChild!, 'getBoundingClientRect').mockReturnValue({ left: 140, top: 280, width: 200, height: 100 } as DOMRect);
  return viewport;
}

beforeEach(() => {
  vi.mocked(toPng).mockResolvedValue('data:image/png;base64,test');
  vi.mocked(toSvg).mockResolvedValue('data:image/svg+xml;base64,test');
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { callback(0); return 1; });
});
afterEach(() => { document.body.innerHTML = ''; vi.restoreAllMocks(); vi.clearAllMocks(); vi.unstubAllGlobals(); });

it.each([exportCanvasToPng, exportCanvasToSvg])('returns null without a complete canvas', async (exportCanvas) => {
  // Missing DOM must not trigger image generation or a download.
  expect(await exportCanvas()).toBeNull();
  document.body.innerHTML = '<div class="react-flow"></div>';
  expect(await exportCanvas()).toBeNull();
  expect(toPng).not.toHaveBeenCalled();
  expect(toSvg).not.toHaveBeenCalled();
});

it.each([
  [exportCanvasToPng, toPng, 'skyulf-canvas.png', 'data:image/png;base64,test'],
  [exportCanvasToSvg, toSvg, 'skyulf-canvas.svg', 'data:image/svg+xml;base64,test'],
] as const)('fits, filters and restores the viewport while downloading with default options', async (exportCanvas, render, filename, url) => {
  // Export needs stable geometry and cleanup while preserving the download contract.
  const viewport = mountCanvas();
  const original = viewport.style.transform;
  const downloads: string[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { downloads.push(this.download); });
  vi.mocked(render).mockImplementation(async (_canvas, options) => {
    expect(viewport.style.transform).toBe('translate(380px, 210px) scale(2)');
    expect(options).toMatchObject({ backgroundColor: '#ffffff', cacheBust: true });
    for (const className of ['react-flow__panel', 'react-flow__controls', 'react-flow__attribution', 'no-export']) {
      const panel = document.createElement('div');
      panel.className = className;
      expect(options!.filter!(panel)).toBe(false);
    }
    expect(options!.filter!(document.createElement('div'))).toBe(true);
    return url;
  });
  expect(await exportCanvas()).toBe(url);
  expect(downloads).toEqual([filename]);
  expect(viewport.style.transform).toBe(original);
  expect(document.querySelector('a')).toBeNull();
  if (render === toPng) expect(vi.mocked(toPng).mock.calls[0]![1]).toMatchObject({ pixelRatio: 2 });
});

it.each([toPng, toSvg])('restores the viewport after capture rejects', async (render) => {
  // Image failures must not leave the interactive canvas repositioned.
  const viewport = mountCanvas();
  const original = viewport.style.transform;
  vi.mocked(render).mockRejectedValueOnce(new Error('capture failed'));
  const exportCanvas = render === toPng ? exportCanvasToPng : exportCanvasToSvg;
  await expect(exportCanvas('custom')).rejects.toThrow('capture failed');
  expect(viewport.style.transform).toBe(original);
  expect(document.querySelector('a')).toBeNull();
});

it('captures a node-free canvas without changing its transform and keeps custom filenames', async () => {
  // Empty bounds still export the visible canvas without requesting a fit frame.
  const viewport = mountCanvas();
  viewport.innerHTML = '';
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { expect(this.download).toBe('custom.svg'); });
  expect(await exportCanvasToSvg('custom.svg')).toBe('data:image/svg+xml;base64,test');
  expect(viewport.style.transform).toBe('translate(20px, 40px) scale(2)');
  expect(click).toHaveBeenCalledOnce();
});
