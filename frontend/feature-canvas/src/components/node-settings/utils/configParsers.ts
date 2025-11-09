// Shared config parsing helpers used by NodeSettingsModal and its hooks across multiple node types.
const TRUE_CONFIG_TOKENS = new Set(['true', '1', 'yes', 'on']);
const FALSE_CONFIG_TOKENS = new Set(['false', '0', 'no', 'off']);
const AUTO_DETECT_CONFIG_KEYS = ['auto_detect', 'autoDetect', 'auto_detect_columns', 'auto', 'auto_detect_text'];

export const BOOLEAN_TRUE_TOKENS = new Set(['true', '1', 'yes', 'y', 'on', 't']);
export const BOOLEAN_FALSE_TOKENS = new Set(['false', '0', 'no', 'n', 'off', 'f']);

export const normalizeConfigBoolean = (value: unknown): boolean | null => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return null;
    }
    if (value === 1) {
      return true;
    }
    if (value === 0) {
      return false;
    }
    return null;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (!normalized) {
      return null;
    }
    if (TRUE_CONFIG_TOKENS.has(normalized)) {
      return true;
    }
    if (FALSE_CONFIG_TOKENS.has(normalized)) {
      return false;
    }
  }
  return null;
};

export const pickAutoDetectValue = (source: Record<string, unknown> | null | undefined): unknown => {
  if (!source || typeof source !== 'object') {
    return undefined;
  }
  for (const key of AUTO_DETECT_CONFIG_KEYS) {
    if (Object.prototype.hasOwnProperty.call(source, key)) {
      return source[key];
    }
  }
  return undefined;
};

export const cloneConfig = (value: any) => {
  if (value === undefined || value === null) {
    return {};
  }
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (error) {
    if (Array.isArray(value)) {
      return [...value];
    }
    if (typeof value === 'object') {
      return { ...value };
    }
    return value;
  }
};

const sortValue = (value: any): any => {
  if (Array.isArray(value)) {
    return value.map((item) => sortValue(item));
  }
  if (value && typeof value === 'object') {
    return Object.keys(value)
      .sort()
      .reduce<Record<string, any>>((acc, key) => {
        acc[key] = sortValue(value[key]);
        return acc;
      }, {});
  }
  return value;
};

export const stableStringify = (value: any): string => {
  if (value === undefined) {
    return '';
  }
  return JSON.stringify(sortValue(value));
};

export const arraysAreEqual = (first: string[], second: string[]) => {
  if (first.length !== second.length) {
    return false;
  }
  for (let index = 0; index < first.length; index += 1) {
    if (first[index] !== second[index]) {
      return false;
    }
  }
  return true;
};

export const inferColumnSuggestions = (dtype: string | null, values: any[]): string[] => {
  const suggestions: string[] = [];
  const pushSuggestion = (candidate: string) => {
    if (!suggestions.includes(candidate)) {
      suggestions.push(candidate);
    }
  };
  const normalizedDtype = (dtype ?? '').toLowerCase();
  if (!values.length) {
    return suggestions;
  }

  const presentValues = values.filter((value) => value !== undefined && value !== null);
  if (!presentValues.length) {
    return suggestions;
  }

  const numericConvertibleCount = presentValues.reduce((count, value) => {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return count + 1;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (!trimmed) {
        return count;
      }
      const numeric = Number(trimmed);
      if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
        return count + 1;
      }
    }
    return count;
  }, 0);

  const booleanConvertibleCount = presentValues.reduce((count, value) => {
    const token = String(value).trim().toLowerCase();
    if (!token) {
      return count;
    }
    if (BOOLEAN_TRUE_TOKENS.has(token) || BOOLEAN_FALSE_TOKENS.has(token)) {
      return count + 1;
    }
    return count;
  }, 0);

  const uniqueTokenCount = (() => {
    const tokens = new Set<string>();
    presentValues.forEach((value) => {
      tokens.add(String(value).trim().toLowerCase());
    });
    return tokens.size;
  })();

  const numericShare = numericConvertibleCount / presentValues.length;
  const booleanShare = booleanConvertibleCount / presentValues.length;

  if (normalizedDtype.includes('object') || normalizedDtype.includes('string')) {
    if (booleanShare >= 0.7) {
      pushSuggestion('boolean');
    }
    if (numericShare >= 0.7) {
      pushSuggestion('float64');
    }
    if (uniqueTokenCount > 0 && uniqueTokenCount <= 20) {
      pushSuggestion('category');
    }
    return suggestions;
  }

  if (normalizedDtype.includes('float')) {
    const integerConvertibleCount = presentValues.reduce((count, value) => {
      if (typeof value === 'number') {
        return Number.isInteger(value) ? count + 1 : count;
      }
      if (typeof value === 'string') {
        const trimmed = value.trim();
        if (!trimmed) {
          return count;
        }
        const numeric = Number(trimmed);
        if (!Number.isNaN(numeric) && Number.isFinite(numeric) && Number.isInteger(numeric)) {
          return count + 1;
        }
      }
      return count;
    }, 0);

    if (booleanShare >= 0.8) {
      pushSuggestion('boolean');
    }

    if (presentValues.length > 0 && integerConvertibleCount / presentValues.length >= 0.9) {
      pushSuggestion('Int64');
    }
    return suggestions;
  }

  if (normalizedDtype.includes('int')) {
    if (booleanShare >= 0.7) {
      pushSuggestion('boolean');
    }
  }

  return suggestions;
};
