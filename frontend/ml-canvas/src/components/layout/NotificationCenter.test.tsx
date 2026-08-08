import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { NotificationCenter } from './NotificationCenter';
import { useNotificationsStore } from '../../core/store/useNotificationsStore';

vi.mock('../../core/toast', () => ({
  toast: {
    dismissAll: vi.fn(),
  },
}));

let originalRect: typeof Element.prototype.getBoundingClientRect;

beforeEach(() => {
  originalRect = Element.prototype.getBoundingClientRect;
  Element.prototype.getBoundingClientRect = function () {
    return { width: 100, height: 20, top: 0, left: 0, bottom: 20, right: 100, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
  };
  act(() => {
    useNotificationsStore.getState().clear();
    useNotificationsStore.getState().addMany([
      {
        node_id: 'node-1',
        node_type: 'TrainingNode',
        level: 'warning',
        logger: 'pipeline',
        message: 'Training took too long',
      },
    ]);
  });
});

afterEach(() => {
  act(() => {
    useNotificationsStore.getState().clear();
  });
  Element.prototype.getBoundingClientRect = originalRect;
});

describe('NotificationCenter detail modal', () => {
  it('focuses the modal on open, traps Tab, and returns focus to the bell button', async () => {
    render(
      <MemoryRouter>
        <NotificationCenter />
      </MemoryRouter>,
    );

    const bell = screen.getByRole('button', { name: /Notifications/ });
    bell.focus();
    fireEvent.click(bell);

    const row = await screen.findByRole('button', { name: /Training took too long/ });
    fireEvent.click(row);

    const closeButton = await screen.findByRole('button', { name: 'Close detail' });
    await waitFor(() => expect(closeButton).toHaveFocus());

    const modal = screen.getByRole('dialog', { name: /TrainingNode details/ });
    const modalButtons = within(modal).getAllByRole('button');
    const lastButton = modalButtons[modalButtons.length - 1]!;
    lastButton.focus();
    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    window.dispatchEvent(tabEvent);
    expect(tabEvent.defaultPrevented).toBe(true);

    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /TrainingNode details/ })).toBeNull());
    await waitFor(() => expect(bell).toHaveFocus());
  });

  it('keeps the row and Dismiss controls separate and independent', async () => {
    const user = userEvent.setup();

    const { container } = render(
      <MemoryRouter>
        <NotificationCenter />
      </MemoryRouter>,
    );

    const bell = screen.getByRole('button', { name: /Notifications/ });
    await act(async () => {
      await user.click(bell);
    });

    const row = await screen.findByRole('button', { name: /Training took too long/ });
    expect(container.querySelector('button button')).toBeNull();

    await act(async () => {
      await user.click(row);
    });
    await screen.findByRole('dialog', { name: /TrainingNode details/ });
    await act(async () => {
      await user.keyboard('{Escape}');
    });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: /TrainingNode details/ })).toBeNull());

    await act(async () => {
      await user.click(bell);
    });
    const dismiss = await screen.findByRole('button', { name: 'Dismiss' });
    await act(async () => {
      await user.click(dismiss);
    });

    await waitFor(() => expect(screen.queryByText('Training took too long')).toBeNull());
    expect(screen.queryByRole('dialog', { name: /TrainingNode details/ })).toBeNull();
  });
});
