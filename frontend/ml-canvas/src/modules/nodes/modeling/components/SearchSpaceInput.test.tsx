import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { SearchSpaceInput } from './SearchSpaceInput';
import type { HyperparameterDef } from './types';

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

/** Invalid candidates must retain the draft and report the first bad entry without emitting data. */
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
  expect(onChange).not.toHaveBeenCalled();
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
