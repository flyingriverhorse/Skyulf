import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { CountVectorizerNode, TfidfVectorizerNode, HashingVectorizerNode } from './VectorizerNodes';

vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [{ datasetId: 'data' }] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: { columns: {
  text: { name: 'text', dtype: 'STRING' }, category: { name: 'category', dtype: 'category' },
  dropped: { name: 'dropped', dtype: 'object' }, numeric: { name: 'numeric', dtype: 'float' },
} } }) }));
vi.mock('../../../core/hooks/useUpstreamDroppedColumns', () => ({ useUpstreamDroppedColumns: () => new Set(['dropped']) }));

/** The actual picker excludes dropped/nontext columns and emits a complete unchanged config patch. */
it('filters text columns and preserves count numeric null/zero/ngram policies', () => {
  const Settings = CountVectorizerNode.settings!;
  const config = { ...CountVectorizerNode.getDefaultConfig(), max_features: 20, ngram_range: [2, 3] as [number, number] };
  const onChange = vi.fn();
  render(<Settings config={config} onChange={onChange} />);
  expect(screen.queryByRole('checkbox', { name: 'numeric' })).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox', { name: 'dropped' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('checkbox', { name: 'text' }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, columns: ['text'] });
  fireEvent.change(screen.getByLabelText('Max features'), { target: { value: '0' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, max_features: 0 });
  fireEvent.change(screen.getByLabelText('Max features'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, max_features: null });
  fireEvent.change(screen.getByLabelText('Min document freq'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, min_df: 1 });
  fireEvent.change(screen.getByLabelText('N-gram min'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, ngram_range: [1, 3] });
  fireEvent.click(screen.getByRole('checkbox', { name: /Binary counts/ }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, binary: true });
});

/** Variant controls preserve settings visibility, zero buckets and info-box local state. */
it('renders hashing fields and keeps info expansion independent of later panel width', () => {
  const Settings = HashingVectorizerNode.settings!;
  const config = HashingVectorizerNode.getDefaultConfig();
  const onChange = vi.fn();
  const { rerender } = render(<Settings config={config} onChange={onChange} isExpanded />);
  expect(screen.getByText(/Stateless hashing vectorizer/)).toBeVisible();
  expect(screen.queryByLabelText('Max features')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('Number of features (hash buckets)'), { target: { value: '0' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, n_features: 0 });
  fireEvent.change(screen.getByLabelText('Number of features (hash buckets)'), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, n_features: 1024 });
  fireEvent.change(screen.getByLabelText('Normalization'), { target: { value: 'none' } });
  expect(onChange).toHaveBeenLastCalledWith({ ...config, norm: 'none' });
  rerender(<Settings config={config} onChange={onChange} isExpanded={false} />);
  expect(screen.getByText(/Stateless hashing vectorizer/)).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'About this vectorizer' }));
  expect(screen.queryByText(/Stateless hashing vectorizer/)).not.toBeInTheDocument();
});

/** TF-IDF keeps its own flag and exported definitions keep the original validation and preview strings. */
it('preserves TF-IDF callbacks and vectorizer definition output', () => {
  const Settings = TfidfVectorizerNode.settings!;
  const config = TfidfVectorizerNode.getDefaultConfig();
  const onChange = vi.fn();
  render(<Settings config={config} onChange={onChange} />);
  fireEvent.click(screen.getByRole('checkbox', { name: /Sublinear TF scaling/ }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, sublinear_tf: true });
  fireEvent.click(screen.getByRole('checkbox', { name: /Remove English stop words/ }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, stop_words: 'english' });
  expect(TfidfVectorizerNode.validate(config)).toEqual({ isValid: false, field: 'columns', message: 'Select at least one text column.' });
  expect(CountVectorizerNode.bodyPreview!({ ...CountVectorizerNode.getDefaultConfig(), columns: ['text'], max_features: 0 })).toBe('1 col · all feats');
  expect(HashingVectorizerNode.bodyPreview!({ ...HashingVectorizerNode.getDefaultConfig(), columns: ['text'], n_features: 0 })).toBe('1 col · 0 buckets');
});
