import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { useState } from 'react';
import { ConfirmProvider, useConfirm } from './ConfirmDialog';

// Match ModalShell.test.tsx — jsdom returns 0x0 rects, which the
// focus-trap utilities treat as "not focusable".
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

const Trigger: React.FC<{ onResult: (ok: boolean) => void; danger?: boolean }> = ({ onResult, danger }) => {
  const confirm = useConfirm();
  const [busy, setBusy] = useState(false);
  return (
    <button
      disabled={busy}
      onClick={async () => {
        setBusy(true);
        const ok = await confirm({
          title: 'Delete it?',
          message: 'This is permanent.',
          ...(danger ? { variant: 'danger' as const } : {}),
        });
        onResult(ok);
        setBusy(false);
      }}
    >
      go
    </button>
  );
};

describe('ConfirmDialog', () => {
  it('resolves true when the confirm button is clicked', async () => {
    const results: boolean[] = [];
    render(
      <ConfirmProvider>
        <Trigger onResult={(ok) => results.push(ok)} />
      </ConfirmProvider>,
    );

    fireEvent.click(screen.getByText('go'));
    await waitFor(() => expect(screen.getByText('Delete it?')).toBeTruthy());

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /confirm|ok/i }));
    });

    await waitFor(() => expect(results).toEqual([true]));
  });

  it('resolves false when cancel is clicked', async () => {
    const results: boolean[] = [];
    render(
      <ConfirmProvider>
        <Trigger onResult={(ok) => results.push(ok)} />
      </ConfirmProvider>,
    );

    fireEvent.click(screen.getByText('go'));
    await waitFor(() => expect(screen.getByText('Delete it?')).toBeTruthy());

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /cancel/i }));
    });

    await waitFor(() => expect(results).toEqual([false]));
  });

  it('throws if useConfirm is used outside ConfirmProvider', () => {
    // Render without provider; the hook should throw on call.
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const Bad: React.FC = () => {
      useConfirm();
      return null;
    };
    expect(() => render(<Bad />)).toThrow(/ConfirmProvider/i);
    errSpy.mockRestore();
  });
});
