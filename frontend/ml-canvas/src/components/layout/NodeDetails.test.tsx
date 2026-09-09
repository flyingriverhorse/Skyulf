import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import * as Popover from '@radix-ui/react-popover';
import { NodeDetails } from './NodeDetails';

afterEach(() => { vi.unstubAllGlobals(); });

describe('node information', () => {
  it('returns focus on Escape even when a background tooltip opens later', async () => {
    // A pointer left over the canvas must not steal dismissal from keyboard users.
    const content = (tooltipOpen: boolean) => <>
      <NodeDetails nodeId="classification-example" />
      <Popover.Root open={tooltipOpen}>
        <Popover.Portal>
          <Popover.Content role="tooltip" onOpenAutoFocus={event => event.preventDefault()}>
            Background connection
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </>;
    const { rerender } = render(content(false));
    const trigger = screen.getByRole('button', { name: 'Node information' });
    fireEvent.click(trigger);
    const copy = screen.getByRole('button', { name: 'Copy node ID' });
    copy.focus();
    rerender(content(true));
    expect(screen.getByRole('tooltip')).toBeVisible();
    expect(copy).toHaveFocus();
    fireEvent.keyDown(copy, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Node information' })).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it.each([true, false])('reports clipboard success=%s and keeps the identifier available', async success => {
    // A denied clipboard write must leave the ID readable and explain how to copy it manually.
    const writeText = success ? vi.fn().mockResolvedValue(undefined) : vi.fn().mockRejectedValue(new Error('Denied'));
    vi.stubGlobal('navigator', { clipboard: { writeText } });
    render(<NodeDetails nodeId="classification-example" />);
    fireEvent.click(screen.getByRole('button', { name: 'Node information' }));
    fireEvent.click(screen.getByRole('button', { name: 'Copy node ID' }));
    expect(writeText).toHaveBeenCalledWith('classification-example');
    expect(await screen.findByText(success ? 'Node ID copied.' : 'Could not copy. Select the ID to copy it manually.')).toBeVisible();
    expect(screen.getByText('classification-example')).toBeVisible();
  });
});
