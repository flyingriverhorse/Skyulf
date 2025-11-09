import { ensureArrayOfString } from '../../sharedUtils';

export type BinningStrategy = 'equal_width' | 'equal_frequency' | 'custom' | 'kbins';
export type BinningLabelFormat = 'range' | 'bin_index' | 'ordinal' | 'column_suffix';
export type BinningMissingStrategy = 'keep' | 'label';
export type KBinsStrategy = 'uniform' | 'quantile' | 'kmeans';
export type KBinsEncode = 'ordinal' | 'onehot' | 'onehot-dense';

export type BinningColumnOverride = {
  strategy?: BinningStrategy;
  equalWidthBins?: number;
  equalFrequencyBins?: number;
  kbinsNBins?: number;
  kbinsEncode?: KBinsEncode;
  kbinsStrategy?: KBinsStrategy;
  customBins?: number[];
  customLabels?: string[];
};

export type NormalizedBinningConfig = {
  strategy: BinningStrategy;
  columns: string[];
  equalWidthBins: number;
  equalFrequencyBins: number;
  includeLowest: boolean;
  precision: number;
  duplicates: 'raise' | 'drop';
  outputSuffix: string;
  dropOriginal: boolean;
  labelFormat: BinningLabelFormat;
  missingStrategy: BinningMissingStrategy;
  missingLabel: string;
  customBins: Record<string, number[]>;
  customLabels: Record<string, string[]>;
  columnStrategies: Record<string, BinningColumnOverride>;
  // KBinsDiscretizer options
  kbinsNBins: number;
  kbinsEncode: KBinsEncode;
  kbinsStrategy: KBinsStrategy;
};

export type BinnedSamplePresetValue = '500' | '10000' | 'all';

export type BinnedSamplePreset = {
  value: BinnedSamplePresetValue;
  label: string;
};

export type BinnedDistributionBin = {
  label: string;
  count: number;
  percentage: number;
  isMissing: boolean;
};

export type BinnedDistributionCard = {
  column: string;
  sourceColumn: string | null;
  totalRows: number;
  missingRows: number;
  distinctBins: number;
  topLabel: string | null;
  topCount: number | null;
  topPercentage: number | null;
  bins: BinnedDistributionBin[];
  hasMoreBins: boolean;
  totalBinCount: number;
};

export const BINNING_DEFAULT_EQUAL_WIDTH_BINS = 5;
export const BINNING_DEFAULT_EQUAL_FREQUENCY_BINS = 4;
export const BINNING_DEFAULT_PRECISION = 3;
export const BINNING_DEFAULT_SUFFIX = '_binned';
export const BINNING_DEFAULT_MISSING_LABEL = 'Missing';
export const KBINS_DEFAULT_N_BINS = 5;
export const KBINS_DEFAULT_ENCODE = 'ordinal';
export const KBINS_DEFAULT_STRATEGY = 'quantile';

export type BinningStrategyOption = {
  value: BinningStrategy;
  label: string;
  description: string;
};

export const BINNING_STRATEGY_OPTIONS: BinningStrategyOption[] = [
  { value: 'equal_width', label: 'Equal-width', description: 'Uniform bin widths based on min/max range.' },
  {
    value: 'equal_frequency',
    label: 'Equal-frequency',
    description: 'Quantile-based bins with similar row counts in each bucket.',
  },
  { value: 'custom', label: 'Custom edges', description: 'Provide explicit thresholds and optional labels per column.' },
  { value: 'kbins', label: 'KBins (sklearn)', description: 'Use sklearn KBinsDiscretizer with uniform, quantile, or k-means strategies.' },
];

export type BinningLabelFormatOption = {
  value: BinningLabelFormat;
  label: string;
};

export const BINNING_LABEL_FORMAT_OPTIONS: BinningLabelFormatOption[] = [
  { value: 'range', label: 'Numeric interval (default)' },
  { value: 'bin_index', label: 'Index (bin_0, bin_1, …)' },
  { value: 'ordinal', label: 'Ordinal (Bin 1, Bin 2, …)' },
  { value: 'column_suffix', label: 'Column suffix (col_binned_1, …)' },
];

export type BinningDuplicateOption = {
  value: 'raise' | 'drop';
  label: string;
};

export const BINNING_DUPLICATE_OPTIONS: BinningDuplicateOption[] = [
  { value: 'raise', label: 'Raise error on duplicate edges' },
  { value: 'drop', label: 'Drop duplicate edges' },
];

export const BINNING_STRATEGY_LABELS: Record<BinningStrategy, string> = {
  equal_width: 'Equal-width',
  equal_frequency: 'Equal-frequency',
  custom: 'Custom edges',
  kbins: 'KBins (sklearn)',
};

export type BinningMissingOption = {
  value: BinningMissingStrategy;
  label: string;
};

export const BINNING_MISSING_OPTIONS: BinningMissingOption[] = [
  { value: 'keep', label: 'Leave as missing' },
  { value: 'label', label: 'Replace with label' },
];

export const BINNED_SAMPLE_PRESETS: BinnedSamplePreset[] = [
  { value: '500', label: '500 rows (default)' },
  { value: '10000', label: '10000 rows (detailed)' },
  { value: 'all', label: 'Full dataset' },
];

export const normalizeBinningConfigValue = (value: any): NormalizedBinningConfig => {
  const rawStrategy = typeof value?.strategy === 'string' ? value.strategy.trim().toLowerCase() : '';
  const strategy = (['equal_width', 'equal_frequency', 'custom', 'kbins'] as BinningStrategy[]).includes(
    rawStrategy as BinningStrategy,
  )
    ? (rawStrategy as BinningStrategy)
    : 'equal_width';

  const columns = ensureArrayOfString(value?.columns);

  const clampInteger = (input: any, fallback: number, minimum: number, maximum: number) => {
    const numeric = Number(input);
    if (!Number.isFinite(numeric)) {
      return fallback;
    }
    const rounded = Math.round(numeric);
    return Math.max(minimum, Math.min(maximum, rounded));
  };

  const equalWidthBins = clampInteger(value?.equal_width_bins, BINNING_DEFAULT_EQUAL_WIDTH_BINS, 2, 200);
  const equalFrequencyBins = clampInteger(value?.equal_frequency_bins, BINNING_DEFAULT_EQUAL_FREQUENCY_BINS, 2, 200);
  const precision = clampInteger(value?.precision, BINNING_DEFAULT_PRECISION, 0, 8);

  const duplicates = value?.duplicates === 'drop' ? 'drop' : 'raise';
  const includeLowest = Boolean(value?.include_lowest ?? true);
  const dropOriginal = Boolean(value?.drop_original ?? false);

  const outputSuffixRaw = typeof value?.output_suffix === 'string' ? value.output_suffix.trim() : '';
  const outputSuffix = outputSuffixRaw || BINNING_DEFAULT_SUFFIX;

  const rawLabelFormat = typeof value?.label_format === 'string' ? value.label_format.trim().toLowerCase() : '';
  const labelFormat = (['range', 'bin_index', 'ordinal', 'column_suffix'] as BinningLabelFormat[]).includes(
    rawLabelFormat as BinningLabelFormat,
  )
    ? (rawLabelFormat as BinningLabelFormat)
    : 'range';

  const rawMissingStrategy =
    typeof value?.missing_strategy === 'string' ? value.missing_strategy.trim().toLowerCase() : '';
  const missingStrategy: BinningMissingStrategy = rawMissingStrategy === 'label' ? 'label' : 'keep';
  const missingLabelCandidate = typeof value?.missing_label === 'string' ? value.missing_label.trim() : '';
  const missingLabel =
    missingStrategy === 'label' ? missingLabelCandidate || BINNING_DEFAULT_MISSING_LABEL : BINNING_DEFAULT_MISSING_LABEL;

  // KBins parameters
  const kbinsNBins = clampInteger(value?.kbins_n_bins, KBINS_DEFAULT_N_BINS, 2, 200);
  
  const rawKbinsEncode = typeof value?.kbins_encode === 'string' ? value.kbins_encode.trim().toLowerCase() : '';
  const kbinsEncode: KBinsEncode = (['ordinal', 'onehot', 'onehot-dense'] as KBinsEncode[]).includes(
    rawKbinsEncode as KBinsEncode,
  )
    ? (rawKbinsEncode as KBinsEncode)
    : KBINS_DEFAULT_ENCODE;
  
  const rawKbinsStrategy = typeof value?.kbins_strategy === 'string' ? value.kbins_strategy.trim().toLowerCase() : '';
  const kbinsStrategy: KBinsStrategy = (['uniform', 'quantile', 'kmeans'] as KBinsStrategy[]).includes(
    rawKbinsStrategy as KBinsStrategy,
  )
    ? (rawKbinsStrategy as KBinsStrategy)
    : KBINS_DEFAULT_STRATEGY;

  const parseNumberList = (rawBins: any): number[] => {
    if (!Array.isArray(rawBins)) {
      return [];
    }
    const numericBins = rawBins
      .map((entry) => {
        if (typeof entry === 'number') {
          return entry;
        }
        if (entry === null || entry === undefined) {
          return Number.NaN;
        }
        const trimmed = String(entry).trim();
        if (!trimmed) {
          return Number.NaN;
        }
        return Number(trimmed);
      })
      .filter((entry) => Number.isFinite(entry))
      .sort((a, b) => a - b);
    const uniqueBins = numericBins.filter((entry, index, array) => index === 0 || entry !== array[index - 1]);
    return uniqueBins.length >= 2 ? uniqueBins : [];
  };

  const parseLabelList = (rawLabels: any): string[] => {
    if (Array.isArray(rawLabels)) {
      return rawLabels.map((entry) => String(entry).trim()).filter(Boolean);
    }
    if (typeof rawLabels === 'string') {
      return rawLabels
        .split(/[\,\n]/)
        .map((segment) => segment.trim())
        .filter(Boolean);
    }
    return [];
  };

  const coerceOptionalInteger = (input: any, minimum: number, maximum: number): number | null => {
    if (input === null || input === undefined) {
      return null;
    }
    const numeric = Number(input);
    if (!Number.isFinite(numeric)) {
      return null;
    }
    const rounded = Math.round(numeric);
    return Math.max(minimum, Math.min(maximum, rounded));
  };

  const customBins: Record<string, number[]> = {};
  if (value?.custom_bins && typeof value.custom_bins === 'object' && !Array.isArray(value.custom_bins)) {
    Object.entries(value.custom_bins as Record<string, any>).forEach(([key, rawBins]) => {
      const column = String(key ?? '').trim();
      if (!column) {
        return;
      }
      const parsed = parseNumberList(rawBins);
      if (parsed.length >= 2) {
        customBins[column] = parsed;
      }
    });
  }

  const customLabels: Record<string, string[]> = {};
  if (value?.custom_labels && typeof value.custom_labels === 'object' && !Array.isArray(value.custom_labels)) {
    Object.entries(value.custom_labels as Record<string, any>).forEach(([key, rawLabels]) => {
      const column = String(key ?? '').trim();
      if (!column) {
        return;
      }
      const labels = parseLabelList(rawLabels);
      if (labels.length) {
        customLabels[column] = labels;
      }
    });
  }

  const columnStrategies: Record<string, BinningColumnOverride> = {};
  const rawColumnStrategies = (() => {
    if (value?.column_strategies && typeof value.column_strategies === 'object' && !Array.isArray(value.column_strategies)) {
      return value.column_strategies as Record<string, any>;
    }
    if (value?.column_overrides && typeof value.column_overrides === 'object' && !Array.isArray(value.column_overrides)) {
      return value.column_overrides as Record<string, any>;
    }
    return null;
  })();

  if (rawColumnStrategies) {
    Object.entries(rawColumnStrategies).forEach(([key, rawOverride]) => {
      const column = String(key ?? '').trim();
      if (!column) {
        return;
      }
      if (!rawOverride || typeof rawOverride !== 'object' || Array.isArray(rawOverride)) {
        return;
      }

      const overridePayload = rawOverride as Record<string, any>;
      const override: BinningColumnOverride = {};

      const rawOverrideStrategy = typeof overridePayload.strategy === 'string' ? overridePayload.strategy.trim().toLowerCase() : '';
      if ((['equal_width', 'equal_frequency', 'custom', 'kbins'] as BinningStrategy[]).includes(rawOverrideStrategy as BinningStrategy)) {
        override.strategy = rawOverrideStrategy as BinningStrategy;
      }

      const overrideEqualWidth = coerceOptionalInteger(overridePayload.equal_width_bins, 2, 200);
      if (overrideEqualWidth !== null) {
        override.equalWidthBins = overrideEqualWidth;
      }

      const overrideEqualFrequency = coerceOptionalInteger(overridePayload.equal_frequency_bins, 2, 200);
      if (overrideEqualFrequency !== null) {
        override.equalFrequencyBins = overrideEqualFrequency;
      }

      const overrideKbinsNBins = coerceOptionalInteger(overridePayload.kbins_n_bins, 2, 200);
      if (overrideKbinsNBins !== null) {
        override.kbinsNBins = overrideKbinsNBins;
      }

      const rawOverrideEncode = typeof overridePayload.kbins_encode === 'string' ? overridePayload.kbins_encode.trim().toLowerCase() : '';
      if ((['ordinal', 'onehot', 'onehot-dense'] as KBinsEncode[]).includes(rawOverrideEncode as KBinsEncode)) {
        override.kbinsEncode = rawOverrideEncode as KBinsEncode;
      }

      const rawOverrideKbinsStrategy = typeof overridePayload.kbins_strategy === 'string' ? overridePayload.kbins_strategy.trim().toLowerCase() : '';
      if ((['uniform', 'quantile', 'kmeans'] as KBinsStrategy[]).includes(rawOverrideKbinsStrategy as KBinsStrategy)) {
        override.kbinsStrategy = rawOverrideKbinsStrategy as KBinsStrategy;
      }

      const overrideCustomBins = parseNumberList(overridePayload.custom_bins);
      if (overrideCustomBins.length >= 2) {
        override.customBins = overrideCustomBins;
      }

      const overrideCustomLabels = parseLabelList(overridePayload.custom_labels);
      if (overrideCustomLabels.length > 0) {
        override.customLabels = overrideCustomLabels;
      }

      if (Object.keys(override).length > 0) {
        columnStrategies[column] = override;
      }
    });
  }

  return {
    strategy,
    columns,
    equalWidthBins,
    equalFrequencyBins,
    includeLowest,
    precision,
    duplicates,
    outputSuffix,
    dropOriginal,
    labelFormat,
    missingStrategy,
    missingLabel,
    customBins,
    customLabels,
    columnStrategies,
    kbinsNBins,
    kbinsEncode,
    kbinsStrategy,
  };
};

export const buildBinningOverrideSummary = (
  overrideColumns: string[],
  columnStrategies: Record<string, BinningColumnOverride>,
  overrideCount: number,
  strategyLabels: Record<BinningStrategy, string> = BINNING_STRATEGY_LABELS,
): string | null => {
  if (!overrideColumns.length) {
    return null;
  }

  const parts = overrideColumns.map((column) => {
    const override = columnStrategies[column];
    const strategy = override?.strategy ?? null;
    if (!strategy) {
      return column;
    }
    const label = strategyLabels[strategy] ?? strategy;
    const detailParts: string[] = [];

    if (strategy === 'equal_width' && override?.equalWidthBins) {
      detailParts.push(`${override.equalWidthBins} bins`);
    } else if (strategy === 'equal_frequency' && override?.equalFrequencyBins) {
      detailParts.push(`${override.equalFrequencyBins} bins`);
    } else if (strategy === 'kbins') {
      if (override?.kbinsNBins) {
        detailParts.push(`${override.kbinsNBins} bins`);
      }
      if (override?.kbinsStrategy) {
        detailParts.push(override.kbinsStrategy);
      }
      if (override?.kbinsEncode) {
        detailParts.push(override.kbinsEncode);
      }
    }

    const detail = detailParts.length ? ` (${detailParts.join(', ')})` : '';
    return `${column} -> ${label}${detail}`;
  });

  if (overrideCount > overrideColumns.length) {
    parts.push('…');
  }

  return parts.join(', ');
};
