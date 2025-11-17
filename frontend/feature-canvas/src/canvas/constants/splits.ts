export const SPLIT_DEFINITIONS = [
  { key: 'train', label: 'Train' },
  { key: 'test', label: 'Test' },
  { key: 'validation', label: 'Validation' },
] as const;
export type SplitTypeKey = (typeof SPLIT_DEFINITIONS)[number]['key'];

export const SPLIT_TYPE_ORDER: SplitTypeKey[] = SPLIT_DEFINITIONS.map((definition) => definition.key);

export const SPLIT_LABEL_MAP: Record<SplitTypeKey, string> = SPLIT_DEFINITIONS.reduce(
  (accumulator, definition) => ({
    ...accumulator,
    [definition.key]: definition.label,
  }),
  {} as Record<SplitTypeKey, string>
);

export const isValidSplitKey = (value: unknown): value is SplitTypeKey =>
  typeof value === 'string' && SPLIT_TYPE_ORDER.includes(value as SplitTypeKey);

export const sanitizeSplitList = (value?: unknown): SplitTypeKey[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const uniqueKeys = new Set<SplitTypeKey>();
  value.forEach((entry) => {
    if (isValidSplitKey(entry)) {
      uniqueKeys.add(entry);
    }
  });

  return SPLIT_TYPE_ORDER.filter((key) => uniqueKeys.has(key));
};

export const areSplitArraysEqual = (
  a?: SplitTypeKey[] | null,
  b?: SplitTypeKey[] | null
): boolean => {
  const first = sanitizeSplitList(a);
  const second = sanitizeSplitList(b);

  if (first.length !== second.length) {
    return false;
  }

  return first.every((value, index) => value === second[index]);
};

export const getSplitKeyFromHandle = (handleId?: string | null): SplitTypeKey | null => {
  if (!handleId || typeof handleId !== 'string') {
    return null;
  }

  return SPLIT_TYPE_ORDER.find((key) => handleId.endsWith(`-${key}`)) ?? null;
};

export const getSplitHandlePosition = (index: number, total: number): string => {
  if (total <= 1) {
    return '50%';
  }

  const step = 100 / (total + 1);
  const position = Math.round((index + 1) * step);
  return `${position}%`;
};
