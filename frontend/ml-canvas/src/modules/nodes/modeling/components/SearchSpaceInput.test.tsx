import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { SearchSpaceInput } from './SearchSpaceInput';
import type { HyperparameterDef } from './types';

/** Overflow candidates must invalidate the stored search, not leave a stale runnable value. */
it.each(['Infinity', '1e400', 'NaN'])('blocks a nonfinite numeric draft %s', draft => {
  const onChange = vi.fn();
  render(<SearchSpaceInput def={{ name: 'p', label: 'Parameter', type: 'number', default: 1 }} value={[1]} onChange={onChange} />);
  fireEvent.change(screen.getByRole('textbox'), { target: { value: draft } });
  fireEvent.blur(screen.getByRole('textbox'));
  expect(screen.getByRole('alert')).toHaveTextContent('not a valid number');
  expect(onChange).toHaveBeenLastCalledWith([1], draft);
});

/** Reloaded draft errors remain visible, and an intentional None edit clears the persisted error. */
it('restores an invalid draft and allows repair to an unchanged None candidate', () => {
  const onChange = vi.fn();
  render(<SearchSpaceInput def={{ name: 'p', label: 'Parameter', type: 'number', default: null }} value={[null]} invalidDraft="1e400" onChange={onChange} />);
  const input = screen.getByRole('textbox');
  expect(input).toHaveValue('1e400');
  expect(input).toHaveAttribute('aria-invalid', 'true');
  fireEvent.change(input, { target: { value: 'None' } });
  fireEvent.blur(input);
  expect(onChange).toHaveBeenLastCalledWith([null]);
});

/** A defaults reload must clear the old draft's visible error along with persisted state. */
it('clears stale error styling when valid external defaults replace the draft', () => {
  const def: HyperparameterDef = { name: 'p', label: 'Parameter', type: 'number', default: 1 };
  const onChange = vi.fn();
  const { rerender } = render(<SearchSpaceInput def={def} value={[1]} invalidDraft="1e400" onChange={onChange} />);
  expect(screen.getByRole('textbox')).toHaveAttribute('aria-invalid', 'true');
  rerender(<SearchSpaceInput def={def} value={[2]} onChange={onChange} />);
  expect(screen.getByRole('textbox')).toHaveValue('2');
  expect(screen.getByRole('textbox')).not.toHaveAttribute('aria-invalid');
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

/** Comma parsing keeps order, duplicates, nulls and the existing Number coercion. */
it.each<{ type: HyperparameterDef['type']; draft: string; expected: unknown[] }>([
  { type: 'number', draft: '0, None, 0x10, , 0', expected: [0, null, 16, 0] },
  { type: 'boolean', draft: 'False, TRUE, none', expected: [false, true, null] },
  { type: 'select', draft: ' a, b, None ', expected: ['a', 'b', null] },
  { type: 'number', draft: ' ', expected: [] },
])('parses $type candidates on blur', ({ type, draft, expected }) => {
  const onChange = vi.fn();
  render(<SearchSpaceInput def={{ name: 'p', label: 'Parameter', type, default: 1 }} value={[1]} onChange={onChange} />);
  const input = screen.getByRole('textbox', { name: 'Parameter' });
  fireEvent.change(input, { target: { value: draft } });
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.blur(input);
  expect(onChange).toHaveBeenCalledExactlyOnceWith(expected);
});

/** Invalid candidates retain the draft; numeric errors also block the stored search. */
it.each([
  { type: 'number' as const, draft: '1, bad, worse', message: '"bad" is not a valid number' },
  { type: 'boolean' as const, draft: 'true, bad', message: '"bad" must be true or false' },
])('reports $type parse failures and clears errors while editing', ({ type, draft, message }) => {
  const onChange = vi.fn();
  render(<SearchSpaceInput def={{ name: 'p', label: 'Parameter', type, default: 1 }} value={[1]} onChange={onChange} />);
  const input = screen.getByRole('textbox');
  fireEvent.change(input, { target: { value: draft } });
  fireEvent.blur(input);
  expect(screen.getByRole('alert')).toHaveTextContent(message);
  expect(input).toHaveValue(draft);
  expect(onChange).toHaveBeenCalledExactlyOnceWith([1], draft);
  fireEvent.change(input, { target: { value: '' } });
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

/** Identical parsed candidates must not trigger an upstream identity-only update. */
it('suppresses unchanged lists and enforces exclusive chip selection', () => {
  const onChange = vi.fn();
  const def: HyperparameterDef = { name: 'p', label: 'Parameter', type: 'select', default: 'a',
    options: [{ label: 'A', value: 'a' }, { label: 'B', value: 'b' }, { label: 'Only', value: 'only' }], exclusive_options: ['only'] };
  const { rerender } = render(<SearchSpaceInput def={def} value={['a']} onChange={onChange} />);
  fireEvent.blur(screen.getByRole('textbox'));
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Only' }));
  expect(onChange).toHaveBeenLastCalledWith(['only']);
  rerender(<SearchSpaceInput def={def} value={['only']} onChange={onChange} />);
  fireEvent.click(screen.getByRole('button', { name: 'B' }));
  expect(onChange).toHaveBeenLastCalledWith(['b']);
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'only, a' } });
  fireEvent.blur(screen.getByRole('textbox'));
  expect(screen.getByRole('alert')).toHaveTextContent('"only" can\'t be combined with other options here');
});
