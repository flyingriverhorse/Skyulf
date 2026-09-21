import axios from 'axios';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DecompositionTab } from './DecompositionTab';

describe('decomposition measure selection', () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, 'scrollTo', { configurable: true, value: vi.fn() });
  });
  afterEach(() => vi.restoreAllMocks());

  it('sends the visible aggregation on the first metric selection and restores row count', async () => {
    // An invalid hidden count aggregation used to disagree with the displayed Sum option.
    const post = vi.spyOn(axios, 'post').mockImplementation(async (_url, body) => {
      const { measure_col, measure_agg } = body as { measure_col: string | null; measure_agg: string };
      const value = measure_col === null || measure_agg === 'count' ? 2 : measure_agg === 'sum' ? 30 : 15;
      return { data: [{ name: 'Total', value, ratio: 1 }] };
    });
    render(<DecompositionTab datasetId={6161} columns={['revenue', 'cost']} />);
    await screen.findByRole('button', { name: 'Total 2 (100%)' });
    fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'revenue' } });
    await screen.findByRole('button', { name: 'Total 30 (100%)' });
    expect(screen.getAllByRole('combobox')[1]!).toHaveValue('sum');
    expect(post).toHaveBeenLastCalledWith('/api/eda/6161/decomposition', {
      measure_col: 'revenue', measure_agg: 'sum', split_col: '', filters: [],
    });
    fireEvent.change(screen.getAllByRole('combobox')[1]!, { target: { value: 'mean' } });
    await screen.findByRole('button', { name: 'Total 15 (100%)' });
    expect(post).toHaveBeenLastCalledWith('/api/eda/6161/decomposition', {
      measure_col: 'revenue', measure_agg: 'mean', split_col: '', filters: [],
    });
    fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'count' } });
    await screen.findByRole('button', { name: 'Total 2 (100%)' });
    expect(screen.getAllByRole('combobox')).toHaveLength(1);
    // Count can be restored from the existing tree cache; a new metric must retain Average.
    fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'cost' } });
    await screen.findByRole('button', { name: 'Total 15 (100%)' });
    expect(screen.getAllByRole('combobox')[1]!).toHaveValue('mean');
    expect(post.mock.calls.map(([, body]) => { const request = body as { measure_col: string | null; measure_agg: string }; return [request.measure_col, request.measure_agg]; })).toEqual([
      [null, 'count'], ['revenue', 'sum'], ['revenue', 'mean'], ['cost', 'mean'],
    ]);
  });
});
