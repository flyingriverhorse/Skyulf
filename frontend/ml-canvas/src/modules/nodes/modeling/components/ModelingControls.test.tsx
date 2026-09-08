import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { jobsApi } from '../../../../core/api/jobs';
import { BaseModelParamsEditor } from './BaseModelParamsEditor';
import { BestParamsModal } from './BestParamsModal';
import { HelpTooltip } from './HelpTooltip';
import { MultiSelectChips } from './MultiSelectChips';
import { SearchSpaceInput } from './SearchSpaceInput';
import { StrategySettingsModal } from './StrategySettingsModal';

vi.mock('../../../../core/api/jobs', () => ({ jobsApi: { getHyperparameters: vi.fn(), getTuningHistory: vi.fn() } }));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(jobsApi.getTuningHistory).mockResolvedValue([]);
});

/** Invalid candidate values must be explained by the field that needs correction. */
it('labels search spaces and associates parse errors', () => {
  render(<SearchSpaceInput def={{ name: 'depth', label: 'Max Depth', type: 'number', default: 3 }} value={[3]} onChange={vi.fn()} />);
  const input = screen.getByRole('textbox', { name: 'Max Depth' });
  fireEvent.change(input, { target: { value: 'wrong' } });
  fireEvent.blur(input);
  expect(input).toHaveAttribute('aria-invalid', 'true');
  expect(input).toHaveAccessibleDescription('"wrong" is not a valid number');
});

/** Option chips must expose selection without relying on their colors. */
it('announces selected search candidates and model chips', () => {
  render(<>
    <SearchSpaceInput def={{ name: 'solver', label: 'Solver', type: 'select', default: 'a', options: [{ label: 'A', value: 'a' }, { label: 'B', value: 'b' }] }} value={['a']} onChange={vi.fn()} />
    <MultiSelectChips options={[{ label: 'Forest', value: 'forest' }]} selected={['forest']} onChange={vi.fn()} />
  </>);
  expect(screen.getByRole('button', { name: 'A', pressed: true })).toBeVisible();
  expect(screen.getByRole('button', { name: 'B', pressed: false })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Forest', pressed: true })).toBeVisible();
});

/** Repeated model parameters must be distinguishable by their model section. */
it('labels every base and final estimator parameter variant with unique ids', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue([
    { name: 'depth', label: 'Depth', type: 'number', default: 3 },
    { name: 'bootstrap', label: 'Bootstrap', type: 'boolean', default: true },
    { name: 'criterion', label: 'Criterion', type: 'select', default: 'gini', options: [{ label: 'Gini', value: 'gini' }] },
  ]);
  await act(async () => render(<BaseModelParamsEditor task="classification" baseEstimators={['random_forest']} finalEstimator="random_forest" optionLabels={{ random_forest: 'Random Forest' }} baseParams={{}} finalParams={{}} onChange={vi.fn()} />));
  fireEvent.click(screen.getByRole('button', { name: 'Random Forest' }));
  fireEvent.click(screen.getByRole('button', { name: /Final.*Random Forest/ }));
  for (const name of ['Depth', 'Bootstrap', 'Criterion']) {
    const controls = screen.getAllByLabelText(name);
    expect(controls).toHaveLength(2);
    expect(controls[0]?.id).not.toBe(controls[1]?.id);
  }
  expect(within(screen.getByRole('group', { name: 'Random Forest' })).getByRole('spinbutton', { name: 'Depth' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Random Forest' })).toHaveAttribute('aria-expanded', 'true');
});

/** Every strategy's conditional fields must remain reachable by their visible captions. */
it.each([
  { strategy: 'halving_random', labels: ['Factor', 'Min Resources', 'Resource'] },
  { strategy: 'optuna', labels: ['Sampler', 'Pruner', 'Timeout (Seconds)'] },
])('labels $strategy controls', ({ strategy, labels }) => {
  render(<StrategySettingsModal isOpen onClose={vi.fn()} onSave={vi.fn()} strategy={strategy} />);
  for (const label of labels) expect(screen.getByLabelText(label)).toBeVisible();
});

/** Keyboard users need the same parameter help available on hover. */
it('provides focusable help with an accessible description', () => {
  render(<HelpTooltip text="Choose a stable seed." />);
  const help = screen.getByRole('button', { name: 'Help' });
  act(() => help.focus());
  expect(help).toHaveFocus();
  expect(help).toHaveAccessibleDescription('Choose a stable seed.');
});

/** Dismissing contextual help must leave the settings dialog and focused trigger available. */
it('consumes the first Escape on open help before dismissing its modal', async () => {
  const onClose = vi.fn();
  await act(async () => render(<StrategySettingsModal isOpen onClose={onClose} onSave={vi.fn()} strategy="optuna" />));
  const help = screen.getAllByRole('button', { name: 'Help' })[0]!;
  act(() => help.focus());
  expect(help).toHaveAccessibleDescription(/TPE provides smart Bayesian learning/);
  fireEvent.keyDown(help, { key: 'Escape' });
  expect(onClose).not.toHaveBeenCalled();
  expect(help).toHaveFocus();
  fireEvent.keyDown(help, { key: 'Escape' });
  expect(onClose).toHaveBeenCalledOnce();
});

/** History must identify its modal and controls, and allow keyboard dismissal. */
it('labels history actions and closes with Escape', async () => {
  const onClose = vi.fn();
  await act(async () => render(<BestParamsModal isOpen onClose={onClose} modelType="random_forest_classifier" />));
  expect(screen.getByRole('dialog', { name: 'Best Parameters History' })).toBeVisible();
  expect(screen.getByRole('combobox', { name: 'View parameters for:' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Refresh history' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Close history' })).toBeVisible();
  fireEvent.keyDown(window, { key: 'Escape' });
  expect(onClose).toHaveBeenCalledOnce();
});
