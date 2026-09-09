import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { SplitMergeGuide } from './SplitMergeGuide';

describe('SplitMergeGuide', () => {
  /** The example must not imply that resolving one conflict discards the losing branch. */
  it('retains both distinct features under either displayed merge strategy', () => {
    render(<SplitMergeGuide />);
    const firstResult = screen.getByText('First wins', { selector: 'strong' }).parentElement;
    const lastResult = screen.getByText('Last wins (default)', { selector: 'strong' }).parentElement;

    expect(firstResult).toHaveTextContent(/age\s*=\s*20/);
    expect(lastResult).toHaveTextContent(/age\s*=\s*0\.4/);
    for (const result of [firstResult, lastResult]) {
      expect(result).toHaveTextContent(/income\s*=\s*900/);
      expect(result).toHaveTextContent(/city_code\s*=\s*2/);
    }
  });

  /** A one-row illustration must state the assumptions that make its winners valid. */
  it('places alignment and ownership conditions alongside the example', () => {
    render(<SplitMergeGuide />);
    const heading = screen.getByRole('heading', { name: /First wins or last wins/ });
    const section = heading.closest('section');
    if (!section) throw new Error('The merge example must have its own section.');

    expect(section).toHaveTextContent(/same row/);
    expect(section).toHaveTextContent(/both branches have changed age/);
    expect(within(section).getByText('Ownership exception')).toBeInTheDocument();
    expect(section).toHaveTextContent(/same aligned target/);
  });
});
