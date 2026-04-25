import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ModalShell } from './ModalShell';

// jsdom returns a 0x0 rect for every element, which causes the
// production `getFocusable()` filter to reject buttons as offscreen.
// Stub a non-zero rect so the focus trap behaves like a real browser.
let originalGetBoundingClientRect: typeof Element.prototype.getBoundingClientRect;
beforeEach(() => {
  originalGetBoundingClientRect = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function () {
    return { width: 100, height: 20, top: 0, left: 0, bottom: 20, right: 100, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
  };
});
afterEach(() => {
  Element.prototype.getBoundingClientRect = originalGetBoundingClientRect;
});

describe('ModalShell', () => {
  it('renders nothing when closed', () => {
    render(
      <ModalShell isOpen={false} onClose={() => {}} title="Hidden">
        <p>body</p>
      </ModalShell>,
    );
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('renders the dialog with title and body when open', () => {
    render(
      <ModalShell isOpen onClose={() => {}} title="My modal">
        <p>body content</p>
      </ModalShell>,
    );
    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText('My modal')).toBeInTheDocument();
    expect(screen.getByText('body content')).toBeInTheDocument();
  });

  it('moves focus to the first focusable element on open', async () => {
    render(
      <ModalShell isOpen onClose={() => {}} title="Focusable" hideCloseButton>
        <button>first</button>
        <button>second</button>
      </ModalShell>,
    );
    // Focus is moved on the next animation frame, so wait for it.
    await waitFor(() => {
      expect(screen.getByText('first')).toHaveFocus();
    });
  });

  it('Escape triggers onClose by default', () => {
    const onClose = vi.fn();
    render(
      <ModalShell isOpen onClose={onClose} title="Esc">
        <p>x</p>
      </ModalShell>,
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('Escape does NOT trigger onClose when dismissOnEscape={false}', () => {
    const onClose = vi.fn();
    render(
      <ModalShell isOpen onClose={onClose} dismissOnEscape={false} title="No-esc">
        <p>x</p>
      </ModalShell>,
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('clicking the backdrop dismisses by default', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ModalShell isOpen onClose={onClose} title="Backdrop">
        <p>body</p>
      </ModalShell>,
    );
    // The backdrop is the dialog's parent — click it directly.
    const dialog = screen.getByRole('dialog');
    const backdrop = dialog.parentElement!;
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('clicking inside the dialog does NOT dismiss', async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ModalShell isOpen onClose={onClose} title="Inside">
        <p>body</p>
      </ModalShell>,
    );
    await user.click(screen.getByText('body'));
    expect(onClose).not.toHaveBeenCalled();
  });

  it('Tab from the last focusable wraps back to the first (focus trap)', async () => {
    const user = userEvent.setup();
    render(
      <ModalShell isOpen onClose={() => {}} title="Trap" hideCloseButton>
        <button>alpha</button>
        <button>beta</button>
      </ModalShell>,
    );
    const alpha = screen.getByText('alpha') as HTMLButtonElement;
    const beta = screen.getByText('beta') as HTMLButtonElement;

    await waitFor(() => expect(alpha).toHaveFocus());

    // Move to last, then Tab should wrap to first.
    beta.focus();
    expect(beta).toHaveFocus();
    await user.tab();
    expect(alpha).toHaveFocus();
  });
});
