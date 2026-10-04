import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { expect, it } from 'vitest';
import { modelNumericIssue } from '../../../../core/utils/numericValidation';
import { CVPolicySettings, policyChange, type CVPolicyConfig } from './CVPolicySettings';

const configuredTime: CVPolicyConfig = { cv_type: 'time_series_split', cv_time_column: 'event_time', cv_group_column: 'customer', cv_shuffle: false, cv_gap: 2, cv_test_size: 8, cv_max_train_size: 30 };

function Harness({ initial }: { initial: CVPolicyConfig }) {
  const [config, setConfig] = useState(initial);
  return <><label>Method<select value={config.cv_type} onChange={e => setConfig({ ...config, ...policyChange('cv_type', e.target.value, config) })}>
    <option value="time_series_split">Time Series</option><option value="group_k_fold">Group</option><option value="k_fold">K-Fold</option><option value="nested_cv">Nested</option>
  </select></label><CVPolicySettings config={config} fieldId="policy" columns={[{ name: 'event_time' }, { name: 'customer' }]} update={patch => setConfig({ ...config, ...patch })} />
    <output aria-label="Policy config">{JSON.stringify(config)}</output></>;
}

/** Hidden temporal windows must never remain active after a method change. */
it.each(['group_k_fold', 'k_fold', 'nested_cv'])('clears temporal windows when changing the method to %s', value => {
  render(<Harness initial={configuredTime} />);
  fireEvent.change(screen.getByRole('combobox', { name: 'Method' }), { target: { value } });
  const config = JSON.parse(screen.getByLabelText('Policy config').textContent!);
  expect(config).toMatchObject({ cv_type: value, cv_gap: 0, cv_test_size: null, cv_max_train_size: null, cv_shuffle: false, cv_group_column: 'customer' });
  expect(screen.queryByRole('spinbutton', { name: 'Gap (rows)' })).not.toBeInTheDocument();
  expect(modelNumericIssue(config)).toBeUndefined();
});

/** Changing the nested policy also clears fields which lose their controls. */
it.each(['auto', 'k_fold', 'group_k_fold'])('clears windows when leaving nested chronology for %s', value => {
  render(<Harness initial={{ ...configuredTime, cv_type: 'nested_cv', cv_nested_type: 'time_series_split' }} />);
  fireEvent.change(screen.getByRole('combobox', { name: 'Nested split policy' }), { target: { value } });
  expect(JSON.parse(screen.getByLabelText('Policy config').textContent!)).toMatchObject({ cv_nested_type: value, cv_gap: 0, cv_test_size: null, cv_max_train_size: null });
});

/** Ordinary custom windows require named timestamps just like Core's strict policy path. */
it('requires an explicit time column with ordinary custom windows', () => {
  const config = { ...configuredTime, cv_time_column: '' };
  render(<Harness initial={config} />);
  expect(screen.getByRole('combobox', { name: 'Time column' })).toBeVisible();
  expect(screen.queryByRole('option', { name: 'Auto-detect' })).not.toBeInTheDocument();
  expect(modelNumericIssue(config)).toMatchObject({ field: 'cv_time_column', isValid: false });
});

/** A method change that retains chronology must preserve its visible window controls. */
it('preserves windows when switching to an existing nested temporal policy', () => {
  render(<Harness initial={{ ...configuredTime, cv_nested_type: 'time_series_split' }} />);
  fireEvent.change(screen.getByRole('combobox', { name: 'Method' }), { target: { value: 'nested_cv' } });
  expect(JSON.parse(screen.getByLabelText('Policy config').textContent!)).toMatchObject({ cv_type: 'nested_cv', cv_nested_type: 'time_series_split', cv_gap: 2, cv_test_size: 8, cv_max_train_size: 30 });
  expect(screen.getByRole('spinbutton', { name: 'Gap (rows)' })).toHaveValue(2);
});
