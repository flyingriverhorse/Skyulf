import { fireEvent, render, screen, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { FeatureTargetSplitNode } from './FeatureTargetSplitNode';
import { TrainTestSplitNode } from './TrainTestSplitNode';

vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: undefined }) }));
vi.mock('../../../core/hooks/useUpstreamData', () => ({ useUpstreamData: () => [] }));

/** Split settings need labeled ratios and per-instance checkboxes to avoid editing another node. */
it('labels split controls independently in repeated settings panels', () => {
  const Settings = TrainTestSplitNode.settings!;
  const onChange = vi.fn();
  const config = { test_size: 0.2, validation_size: 0.1, random_state: 42, stratify: true, shuffle: true };
  render(<>
    <section aria-label="First split"><Settings config={config} onChange={vi.fn()} /></section>
    <section aria-label="Second split"><Settings config={config} onChange={onChange} /></section>
  </>);
  const first = within(screen.getByRole('region', { name: 'First split' }));
  const second = within(screen.getByRole('region', { name: 'Second split' }));
  for (const name of ['Test Size (0.0 - 1.0)', 'Validation Size (0.0 - 1.0)', 'Random State', 'Target Column', 'Shuffle Data', 'Stratify by Target']) {
    expect(first.getByLabelText(name).id).not.toBe(second.getByLabelText(name).id);
  }
  fireEvent.click(second.getByText('Shuffle Data'));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, shuffle: false });
});

/** Feature selection must expose the target dropdown independently of validation errors. */
it('labels the feature-target split dropdown', () => {
  const Settings = FeatureTargetSplitNode.settings!;
  render(<Settings config={{ target_column: '' }} onChange={vi.fn()} />);
  expect(screen.getByRole('combobox', { name: 'Target Column' })).toBeVisible();
});
