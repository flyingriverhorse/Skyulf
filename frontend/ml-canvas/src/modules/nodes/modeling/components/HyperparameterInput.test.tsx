import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { HyperparameterInput } from './HyperparameterInput';

/** Drafts commit only on blur and preserve supported falsy/null numeric values. */
it.each([
  ['number', '0', 0], ['number', '  None ', null], ['number', '0x10', 16],
  ['boolean', 'FALSE', false], ['boolean', ' TrUe ', true], ['text', ' value ', 'value'],
])('commits %s draft %s on blur', (type, draft, expected) => {
  const onChange = vi.fn();
  render(<HyperparameterInput value={2} type={type as string} onChange={onChange} />);
  const input = screen.getByRole('textbox');
  fireEvent.change(input, { target: { value: draft } });
  expect(onChange).not.toHaveBeenCalled();
  fireEvent.blur(input);
  expect(onChange).toHaveBeenCalledExactlyOnceWith(expected);
});

/** Invalid drafts revert, while clearing a numeric draft leaves it empty without committing. */
it.each([['number', 'bad', '4'], ['number', '', ''], ['boolean', 'bad', '4']])(
  'retains the original %s rejection policy for %s', (type, draft, displayed) => {
    const onChange = vi.fn();
    render(<HyperparameterInput value={4} type={type} onChange={onChange} />);
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: draft } });
    fireEvent.blur(input);
    expect(onChange).not.toHaveBeenCalled();
    expect(input).toHaveValue(displayed);
  },
);

/** Incoming values replace local drafts and numeric bounds remain input attributes only. */
it('synchronizes changed values and keeps numeric attributes', () => {
  const onChange = vi.fn();
  const { rerender } = render(<HyperparameterInput value={null} type="number" min={1} max={5} step={2} onChange={onChange} />);
  const input = screen.getByRole('textbox');
  expect(input).toHaveValue('None');
  expect(input).toHaveAttribute('min', '1');
  fireEvent.change(input, { target: { value: '10' } });
  fireEvent.blur(input);
  expect(onChange).toHaveBeenLastCalledWith(10);
  rerender(<HyperparameterInput value={false} type="boolean" onChange={onChange} />);
  expect(input).toHaveValue('false');
  expect(input).not.toHaveAttribute('min');
});
