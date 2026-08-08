import { describe, expect, it } from 'vitest';

import {
  isSelectionMissingFromDatasets,
  resolveEdaDatasetSelection,
  shouldSyncDatasetParam,
} from './edaDatasetSelection';

const datasets = [{ id: 3 }, { id: 68 }, { id: 212 }];

describe('resolveEdaDatasetSelection', () => {
  it('lets an explicit dataset_id override an existing selection', () => {
    expect(resolveEdaDatasetSelection('212', 68, datasets)).toBe(212);
  });

  it('uses an explicit dataset_id when nothing is selected yet', () => {
    expect(resolveEdaDatasetSelection('212', null, datasets)).toBe(212);
  });

  it('accepts an explicit dataset_id before the dataset list has loaded', () => {
    expect(resolveEdaDatasetSelection('212', null, [])).toBe(212);
  });

  it('is idempotent when the param already matches the selection', () => {
    expect(resolveEdaDatasetSelection('68', 68, datasets)).toBe(68);
  });

  it('keeps the current selection when no param is present', () => {
    expect(resolveEdaDatasetSelection(null, 68, datasets)).toBe(68);
  });

  it('falls back to the first dataset only when nothing is selected', () => {
    expect(resolveEdaDatasetSelection(null, null, datasets)).toBe(3);
  });

  it('coerces a string dataset id from the API to a number', () => {
    expect(resolveEdaDatasetSelection(null, null, [{ id: '77' }])).toBe(77);
  });

  it('returns null when there is nothing to select', () => {
    expect(resolveEdaDatasetSelection(null, null, [])).toBeNull();
  });

  it.each(['', '  ', 'abc', '1.5', '0', '-4', 'NaN'])(
    'ignores the malformed dataset_id %o and keeps the current selection',
    (param) => {
      expect(resolveEdaDatasetSelection(param, 68, datasets)).toBe(68);
    },
  );

  it('ignores a malformed dataset_id and still defaults when nothing is selected', () => {
    expect(resolveEdaDatasetSelection('abc', null, datasets)).toBe(3);
  });
});

describe('shouldSyncDatasetParam', () => {
  it('requests a sync when the URL carries no dataset_id', () => {
    expect(shouldSyncDatasetParam(null, 68)).toBe(true);
  });

  it.each(['', '   ', 'abc', '1.5', '0', '-4'])(
    'requests a sync when the URL carries the unusable dataset_id %o',
    (param) => {
      expect(shouldSyncDatasetParam(param, 68)).toBe(true);
    },
  );

  // The URL is authoritative whenever it names a usable dataset. Rewriting it
  // from a not-yet-reconciled selection would clobber the incoming deep link.
  it('never overwrites a usable dataset_id that disagrees with the selection', () => {
    expect(shouldSyncDatasetParam('212', 68)).toBe(false);
  });

  it('does not request a sync when the URL already matches', () => {
    expect(shouldSyncDatasetParam('68', 68)).toBe(false);
  });

  it('does not request a sync when nothing is selected', () => {
    expect(shouldSyncDatasetParam(null, null)).toBe(false);
  });
});

describe('isSelectionMissingFromDatasets', () => {
  it('flags a deep-linked dataset that the usable list does not contain', () => {
    expect(isSelectionMissingFromDatasets(999, datasets, true)).toBe(true);
  });

  it('does not flag a selection that is present in the list', () => {
    expect(isSelectionMissingFromDatasets(68, datasets, true)).toBe(false);
  });

  it('matches ids across the string/number boundary the API returns', () => {
    expect(isSelectionMissingFromDatasets(77, [{ id: '77' }], true)).toBe(false);
  });

  it('stays quiet while the dataset list is still loading', () => {
    expect(isSelectionMissingFromDatasets(999, [], false)).toBe(false);
  });

  it('flags a selection when the loaded list is genuinely empty', () => {
    expect(isSelectionMissingFromDatasets(999, [], true)).toBe(true);
  });

  it('stays quiet when nothing is selected', () => {
    expect(isSelectionMissingFromDatasets(null, datasets, true)).toBe(false);
  });
});
