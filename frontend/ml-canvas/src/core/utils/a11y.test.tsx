import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { clickableProps, onActivateKey } from './a11y';

describe('clickableProps', () => {
  it('returns the standard a11y props for a non-button clickable', () => {
    const handler = vi.fn();
    const props = clickableProps(handler);
    expect(props.role).toBe('button');
    expect(props.tabIndex).toBe(0);
    expect(props.onClick).toBe(handler);
    expect(typeof props.onKeyDown).toBe('function');
  });

  it('fires the handler on Enter and Space, calling preventDefault', async () => {
    const handler = vi.fn();
    const user = userEvent.setup();
    render(
      <div data-testid="row" {...clickableProps<HTMLDivElement>(handler)}>
        click me
      </div>,
    );
    const row = screen.getByTestId('row');
    row.focus();
    await user.keyboard('{Enter}');
    expect(handler).toHaveBeenCalledTimes(1);
    await user.keyboard(' ');
    expect(handler).toHaveBeenCalledTimes(2);
  });

  it('does not fire the handler on other keys', async () => {
    const handler = vi.fn();
    const user = userEvent.setup();
    render(
      <div data-testid="row" {...clickableProps<HTMLDivElement>(handler)}>
        x
      </div>,
    );
    screen.getByTestId('row').focus();
    await user.keyboard('a');
    await user.keyboard('{Tab}');
    await user.keyboard('{Escape}');
    expect(handler).not.toHaveBeenCalled();
  });
});

describe('onActivateKey', () => {
  it('forwards Enter/Space and ignores everything else', async () => {
    const handler = vi.fn();
    const user = userEvent.setup();
    render(
      <button data-testid="b" onKeyDown={onActivateKey<HTMLButtonElement>(handler)}>
        b
      </button>,
    );
    screen.getByTestId('b').focus();
    await user.keyboard('{Enter}');
    await user.keyboard(' ');
    await user.keyboard('a');
    expect(handler).toHaveBeenCalledTimes(2);
  });
});
