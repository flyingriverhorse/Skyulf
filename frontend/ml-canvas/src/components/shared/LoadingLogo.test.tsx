import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { LoadingLogo } from './LoadingLogo';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

it('plays the supplied animation only while reduced motion is off', () => {
  // Preference changes must stop playback and release the media listener on unmount.
  const preference = Object.assign(new EventTarget(), { matches: true });
  const remove = vi.spyOn(preference, 'removeEventListener');
  vi.spyOn(window, 'matchMedia').mockReturnValue(preference as MediaQueryList);
  const { container, unmount } = render(<LoadingLogo />);
  expect(container.querySelector('video')).toBeNull();
  expect(screen.getByAltText('')).toHaveAttribute('src', expect.stringContaining('swirl_yellow'));

  act(() => {
    preference.matches = false;
    preference.dispatchEvent(new Event('change'));
  });
  const video = container.querySelector('video');
  expect(decodeURI(video!.getAttribute('src')!)).toContain('Swirl Logo Animation.mp4');
  expect(video).toHaveProperty('muted', true);
  expect(video).toHaveAttribute('autoplay');
  expect(video).toHaveAttribute('loop');

  act(() => {
    preference.matches = true;
    preference.dispatchEvent(new Event('change'));
  });
  expect(container.querySelector('video')).toBeNull();
  unmount();
  expect(remove).toHaveBeenCalledWith('change', expect.any(Function));
});
