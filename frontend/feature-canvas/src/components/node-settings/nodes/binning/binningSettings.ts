import { ensureArrayOfString } from '../../sharedUtils';

export type BinningStrategy = 'equal_width' | 'equal_frequency' | 'custom' | 'kbins';
export type BinningLabelFormat = 'range' | 'bin_index' | 'ordinal' | 'column_suffix';
export type BinningMissingStrategy = 'keep' | 'label';
export type KBinsStrategy = 'uniform' | 'quantile' | 'kmeans';
export type KBinsEncode = 'ordinal' | 'onehot' | 'onehot-dense';

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

  const customBins: Record<string, number[]> = {};
  if (value?.custom_bins && typeof value.custom_bins === 'object' && !Array.isArray(value.custom_bins)) {
    Object.entries(value.custom_bins as Record<string, any>).forEach(([key, rawBins]) => {
      const column = String(key ?? '').trim();
      if (!column) {
        return;
      }
      if (!Array.isArray(rawBins)) {
        return;
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
      if (uniqueBins.length >= 2) {
        customBins[column] = uniqueBins;
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
      const labels = Array.isArray(rawLabels)
        ? rawLabels.map((entry) => String(entry).trim()).filter(Boolean)
        : typeof rawLabels === 'string'
          ? rawLabels
              .split(/[\,\n]/)
              .map((segment) => segment.trim())
              .filter(Boolean)
          : [];
      if (labels.length) {
        customLabels[column] = labels;
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
    kbinsNBins,
    kbinsEncode,
    kbinsStrategy,
  };
};
