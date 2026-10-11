import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { LoadingLogo } from './LoadingLogo';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('animates the transparent scene only while reduced motion is off', () => {
  // Preference changes must stop frame work and release the media listener on unmount.
  const frames = new Map<number, FrameRequestCallback>();
  let nextFrame = 0;
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => {
    frames.set(++nextFrame, callback);
    return nextFrame;
  });
  const cancel = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(id => { frames.delete(id); });
  const advance = (time: number) => {
    const [id, callback] = [...frames.entries()][0]!;
    frames.delete(id);
    act(() => callback(time));
  };
  const preference = Object.assign(new EventTarget(), { matches: true });
  const remove = vi.spyOn(preference, 'removeEventListener');
  vi.spyOn(window, 'matchMedia').mockReturnValue(preference as MediaQueryList);
  const { container, unmount } = render(<LoadingLogo />);
  expect(container.querySelector('video')).toBeNull();
  expect(frames.size).toBe(0);
  expect(screen.getByAltText('')).toHaveAttribute('src', expect.stringContaining('logo.png'));

  act(() => {
    preference.matches = false;
    preference.dispatchEvent(new Event('change'));
  });
  const scene = container.querySelector('svg');
  expect(scene).toHaveAttribute('viewBox', '420 0 1080 1080');
  expect(container.querySelector('video')).toBeNull();
  advance(1000);
  advance(3600);
  expect(scene!.querySelectorAll('line').length).toBeGreaterThan(0);

  act(() => {
    preference.matches = true;
    preference.dispatchEvent(new Event('change'));
  });
  expect(container.querySelector('svg')).toBeNull();
  expect(screen.getByAltText('')).toHaveAttribute('src', expect.stringContaining('logo.png'));
  expect(frames.size).toBe(0);
  expect(cancel).toHaveBeenCalled();
  unmount();
  expect(remove).toHaveBeenCalledWith('change', expect.any(Function));
});

it('keeps SVG definition references unique for concurrent loading indicators', () => {
  // Multiple loaders must not steal each other's gradient and filter definitions.
  vi.spyOn(window, 'matchMedia').mockReturnValue(Object.assign(new EventTarget(), { matches: false }) as MediaQueryList);
  const { container } = render(<><LoadingLogo /><LoadingLogo /></>);
  const definitions = [...container.querySelectorAll('defs [id]')].map(element => element.id);
  expect(definitions).toHaveLength(8);
  expect(new Set(definitions).size).toBe(8);
  for (const scene of container.querySelectorAll('svg')) {
    const ids = new Set([...scene.querySelectorAll('defs [id]')].map(element => element.id));
    for (const element of scene.querySelectorAll('[fill^="url"], [filter^="url"]')) {
      const reference = [element.getAttribute('fill'), element.getAttribute('filter')].find(value => value?.startsWith('url'));
      expect(ids.has(reference!.slice(5, -1))).toBe(true);
    }
  }
});
