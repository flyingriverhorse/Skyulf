import { beforeEach, afterEach, describe, expect, it } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';
import { ShortcutsOverlay } from './ShortcutsOverlay';

let originalRect: typeof Element.prototype.getBoundingClientRect;

beforeEach(() => {
  originalRect = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function () {
    return { width: 100, height: 20, top: 0, left: 0, bottom: 20, right: 100, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
  };
});

afterEach(() => {
  Element.prototype.getBoundingClientRect = originalRect;
});

const Harness: React.FC = () => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open shortcuts
      </button>
      <ShortcutsOverlay open={open} onClose={() => setOpen(false)} />
    </>
  );
};

describe('ShortcutsOverlay', () => {
  it('focuses the close control on open, traps Tab, and returns focus on close', async () => {
    render(<Harness />);

    const opener = screen.getByRole('button', { name: 'Open shortcuts' });
    opener.focus();
    fireEvent.click(opener);

    const closeButtons = await screen.findAllByRole('button', { name: 'Close shortcuts overlay' });
    const closeButton = closeButtons.find((button) => button.getAttribute('tabindex') !== '-1')!;
    await waitFor(() => expect(closeButton).toHaveFocus());

    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    window.dispatchEvent(tabEvent);
    expect(tabEvent.defaultPrevented).toBe(true);

    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(opener).toHaveFocus());
  });
});
