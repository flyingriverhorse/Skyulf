export type KeepStrategy = 'first' | 'last' | 'none';

export const DEFAULT_KEEP_STRATEGY: KeepStrategy = 'first';

export const DEDUP_KEEP_OPTIONS: Array<{ value: KeepStrategy; label: string }> = [
  { value: 'first', label: 'Keep first occurrence' },
  { value: 'last', label: 'Keep last occurrence' },
  { value: 'none', label: 'Drop all duplicates' },
];
