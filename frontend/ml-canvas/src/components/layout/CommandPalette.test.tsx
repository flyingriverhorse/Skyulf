import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { useState } from 'react';
import { CommandPalette } from './CommandPalette';
import { SHOW_PALETTE_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { registry } from '../../core/registry/NodeRegistry';
import type { NodeDefinition } from '../../core/types/nodes';

let originalRect: typeof Element.prototype.getBoundingClientRect;
let originalScrollIntoView: typeof HTMLElement.prototype.scrollIntoView | undefined;
let getAllSpy: ReturnType<typeof vi.spyOn> | undefined;

const stubNode = (type: string, label: string): NodeDefinition => ({
  type,
  label,
  category: 'Utility',
  description: `${label} description`,
  inputs: [],
  outputs: [],
  settings: () => null,
  validate: () => ({ isValid: true }),
  getDefaultConfig: () => ({}),
});

beforeEach(() => {
  originalRect = Element.prototype.getBoundingClientRect;
  originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
  Element.prototype.getBoundingClientRect = function () {
    return { width: 100, height: 20, top: 0, left: 0, bottom: 20, right: 100, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
  };
  HTMLElement.prototype.scrollIntoView = vi.fn();
  getAllSpy = vi.spyOn(registry, 'getAll').mockReturnValue([
    stubNode('alpha-node', 'Alpha node'),
    stubNode('beta-node', 'Beta node'),
  ]);
});

afterEach(() => {
  getAllSpy?.mockRestore();
  HTMLElement.prototype.scrollIntoView = originalScrollIntoView ?? (() => {});
  Element.prototype.getBoundingClientRect = originalRect;
});

const Harness: React.FC = () => {
  const [opened, setOpened] = useState(0);
  return (
    <>
      <button
        type="button"
        onClick={() => {
          setOpened((value) => value + 1);
          window.dispatchEvent(new CustomEvent(SHOW_PALETTE_EVENT));
        }}
      >
        Open palette
      </button>
      <span data-testid="opened-count">{opened}</span>
      <CommandPalette />
    </>
  );
};

describe('CommandPalette', () => {
  it('keeps the search input focused, preserves arrow navigation, and returns focus on close', async () => {
    render(<Harness />);

    const opener = screen.getByRole('button', { name: 'Open palette' });
    opener.focus();
    fireEvent.click(opener);

    const input = await screen.findByRole('textbox', { name: 'Search nodes' });
    await waitFor(() => expect(input).toHaveFocus());

    const options = await screen.findAllByRole('option');
    expect(options.length).toBeGreaterThan(1);
    expect(options[0]).toHaveAttribute('aria-selected', 'true');

    fireEvent.keyDown(window, { key: 'ArrowDown' });
    await waitFor(() => expect(options[1]).toHaveAttribute('aria-selected', 'true'));

    const lastOption = options[options.length - 1]!;
    const lastButton = within(lastOption).getByRole('button');
    lastButton.focus();
    const tabEvent = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    window.dispatchEvent(tabEvent);
    expect(tabEvent.defaultPrevented).toBe(true);

    fireEvent.keyDown(window, { key: 'Escape' });
    await waitFor(() => expect(opener).toHaveFocus());
  });
});
