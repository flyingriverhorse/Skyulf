import { useCallback } from 'react';
import {
  type BinningStrategy,
  type KBinsEncode,
  type KBinsStrategy,
  type BinningLabelFormat,
  type BinningMissingStrategy,
} from '../../nodes/binning/binningSettings';
import { ensureArrayOfString } from '../../sharedUtils';

type UseBinningHandlersProps = {
  setConfigState: (updater: any) => void;
  setBinningCustomEdgeDrafts: (updater: any) => void;
  setBinningCustomLabelDrafts: (updater: any) => void;
  binningConfig: any;
  binningInsightsRecommendations: any[];
  binningAllNumericColumns: string[];
};

export const useBinningHandlers = ({
  setConfigState,
  setBinningCustomEdgeDrafts,
  setBinningCustomLabelDrafts,
  binningConfig,
  binningInsightsRecommendations,
  binningAllNumericColumns,
}: UseBinningHandlersProps) => {
  const handleBinningIntegerChange = useCallback(
    (
      field: 'equal_width_bins' | 'equal_frequency_bins' | 'precision' | 'kbins_n_bins',
      rawValue: string,
      min: number,
      max: number
    ) => {
      setConfigState((previous: any) => {
        const previousValue =
          typeof previous?.[field] === 'number' && Number.isFinite(previous[field]) ? previous[field] : undefined;
        if (rawValue === '') {
          if (previousValue === undefined) {
            return previous;
          }
          const next: Record<string, any> = { ...previous };
          delete next[field];
          return next;
        }
        const numeric = Number(rawValue);
        if (!Number.isFinite(numeric)) {
          return previous;
        }
        const clamped = Math.min(Math.max(Math.round(numeric), min), max);
        if (previousValue === clamped) {
          return previous;
        }
        return {
          ...previous,
          [field]: clamped,
        };
      });
    },
    [setConfigState]
  );

  const handleBinningBooleanToggle = useCallback((field: 'include_lowest' | 'drop_original', value: boolean) => {
    setConfigState((previous: any) => {
      const current = Boolean(previous?.[field]);
      if (current === value) {
        return previous;
      }
      return {
        ...previous,
        [field]: value,
      };
    });
  }, [setConfigState]);

  const handleBinningSuffixChange = useCallback((value: string) => {
    const trimmed = value.trim();
    setConfigState((previous: any) => {
      const current = typeof previous?.output_suffix === 'string' ? previous.output_suffix : '';
      if (!trimmed) {
        if (!current) {
          return previous;
        }
        const next: Record<string, any> = { ...previous };
        delete next.output_suffix;
        return next;
      }
      if (current === trimmed) {
        return previous;
      }
      return {
        ...previous,
        output_suffix: trimmed,
      };
    });
  }, [setConfigState]);

  const handleBinningLabelFormatChange = useCallback((value: BinningLabelFormat) => {
    setConfigState((previous: any) => {
      const raw = typeof previous?.label_format === 'string' ? previous.label_format : '';
      const current = (['range', 'bin_index', 'ordinal', 'column_suffix'] as BinningLabelFormat[]).includes(
        raw as BinningLabelFormat,
      )
        ? (raw as BinningLabelFormat)
        : 'range';
      if (current === value) {
        return previous;
      }
      return {
        ...previous,
        label_format: value,
      };
    });
  }, [setConfigState]);

  const handleBinningMissingStrategyChange = useCallback((value: BinningMissingStrategy) => {
    setConfigState((previous: any) => {
      const current = previous?.missing_strategy === 'label' ? 'label' : 'keep';
      if (current === value) {
        return previous;
      }
      const next: Record<string, any> = {
        ...previous,
        missing_strategy: value,
      };
      if (value !== 'label' && Object.prototype.hasOwnProperty.call(next, 'missing_label')) {
        delete next.missing_label;
      }
      return next;
    });
  }, [setConfigState]);

  const handleBinningMissingLabelChange = useCallback((value: string) => {
    const trimmed = value.trim();
    setConfigState((previous: any) => {
      const current = typeof previous?.missing_label === 'string' ? previous.missing_label : '';
      if (!trimmed) {
        if (!current) {
          return previous;
        }
        const next: Record<string, any> = { ...previous };
        delete next.missing_label;
        return next;
      }
      if (current === trimmed) {
        return previous;
      }
      return {
        ...previous,
        missing_label: trimmed,
      };
    });
  }, [setConfigState]);

  const handleBinningCustomBinsChange = useCallback((column: string, rawValue: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setBinningCustomEdgeDrafts((previous: any) => ({
      ...previous,
      [normalizedColumn]: rawValue,
    }));
    setConfigState((previous: any) => {
      const existingRaw = previous?.custom_bins;
      const existing =
        existingRaw && typeof existingRaw === 'object' && !Array.isArray(existingRaw)
          ? (existingRaw as Record<string, number[]>)
          : {};
      const existingEntry = Array.isArray(existing[normalizedColumn]) ? existing[normalizedColumn] : undefined;
      const parsed = rawValue
        .split(/[,\n]/)
        .map((segment) => segment.trim())
        .filter(Boolean)
        .map((segment) => Number(segment))
        .filter((value) => Number.isFinite(value))
        .sort((a, b) => a - b);
      const unique = parsed.filter((value, index, array) => index === 0 || value !== array[index - 1]);
      const hasValidEntry = unique.length >= 2;
      if (hasValidEntry && existingEntry) {
        if (existingEntry.length === unique.length && existingEntry.every((value, index) => value === unique[index])) {
          return previous;
        }
      }
      if (!hasValidEntry && !existingEntry) {
        return previous;
      }
      const nextState: Record<string, any> = { ...previous };
      const nextBins: Record<string, number[]> = { ...existing };
      if (hasValidEntry) {
        nextBins[normalizedColumn] = unique;
      } else {
        delete nextBins[normalizedColumn];
      }
      if (Object.keys(nextBins).length > 0) {
        nextState.custom_bins = nextBins;
      } else {
        delete nextState.custom_bins;
      }
      return nextState;
    });
  }, [setBinningCustomEdgeDrafts, setConfigState]);

  const handleBinningCustomLabelsChange = useCallback((column: string, rawValue: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setBinningCustomLabelDrafts((previous: any) => ({
      ...previous,
      [normalizedColumn]: rawValue,
    }));
    setConfigState((previous: any) => {
      const existingRaw = previous?.custom_labels;
      const existing =
        existingRaw && typeof existingRaw === 'object' && !Array.isArray(existingRaw)
          ? (existingRaw as Record<string, string[]>)
          : {};
      const existingEntry = Array.isArray(existing[normalizedColumn]) ? existing[normalizedColumn] : undefined;
      const parsed = rawValue
        .split(/[,\n]/)
        .map((segment) => segment.trim())
        .filter(Boolean);
      if (parsed.length > 0 && existingEntry) {
        if (existingEntry.length === parsed.length && existingEntry.every((value, index) => value === parsed[index])) {
          return previous;
        }
      }
      if (parsed.length === 0 && !existingEntry) {
        return previous;
      }
      const nextState: Record<string, any> = { ...previous };
      const nextLabels: Record<string, string[]> = { ...existing };
      if (parsed.length > 0) {
        nextLabels[normalizedColumn] = parsed;
      } else {
        delete nextLabels[normalizedColumn];
      }
      if (Object.keys(nextLabels).length > 0) {
        nextState.custom_labels = nextLabels;
      } else {
        delete nextState.custom_labels;
      }
      return nextState;
    });
  }, [setBinningCustomLabelDrafts, setConfigState]);

  const handleBinningClearCustomColumn = useCallback((column: string) => {
    const normalizedColumn = String(column ?? '').trim();
    if (!normalizedColumn) {
      return;
    }
    setBinningCustomEdgeDrafts((previous: any) => {
      if (!Object.prototype.hasOwnProperty.call(previous, normalizedColumn)) {
        return previous;
      }
      const next = { ...previous };
      delete next[normalizedColumn];
      return next;
    });
    setBinningCustomLabelDrafts((previous: any) => {
      if (!Object.prototype.hasOwnProperty.call(previous, normalizedColumn)) {
        return previous;
      }
      const next = { ...previous };
      delete next[normalizedColumn];
      return next;
    });
    setConfigState((previous: any) => {
      let changed = false;
      const nextState: Record<string, any> = { ...previous };
      if (
        previous?.custom_bins &&
        typeof previous.custom_bins === 'object' &&
        !Array.isArray(previous.custom_bins) &&
        Object.prototype.hasOwnProperty.call(previous.custom_bins, normalizedColumn)
      ) {
        const nextBins = { ...(previous.custom_bins as Record<string, number[]>) };
        delete nextBins[normalizedColumn];
        if (Object.keys(nextBins).length > 0) {
          nextState.custom_bins = nextBins;
        } else {
          delete nextState.custom_bins;
        }
        changed = true;
      }
      if (
        previous?.custom_labels &&
        typeof previous.custom_labels === 'object' &&
        !Array.isArray(previous.custom_labels) &&
        Object.prototype.hasOwnProperty.call(previous.custom_labels, normalizedColumn)
      ) {
        const nextLabels = { ...(previous.custom_labels as Record<string, string[]>) };
        delete nextLabels[normalizedColumn];
        if (Object.keys(nextLabels).length > 0) {
          nextState.custom_labels = nextLabels;
        } else {
          delete nextState.custom_labels;
        }
        changed = true;
      }
      if (!changed) {
        return previous;
      }
      return nextState;
    });
  }, [setBinningCustomEdgeDrafts, setBinningCustomLabelDrafts, setConfigState]);

  const updateBinningColumnOverride = useCallback(
    (
      column: string,
      mutator: (current: Record<string, any>) => Record<string, any> | null,
    ) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous: any) => {
        const sourceOverrides = (() => {
          if (
            previous?.column_strategies &&
            typeof previous.column_strategies === 'object' &&
            !Array.isArray(previous.column_strategies)
          ) {
            return previous.column_strategies as Record<string, any>;
          }
          if (
            previous?.column_overrides &&
            typeof previous.column_overrides === 'object' &&
            !Array.isArray(previous.column_overrides)
          ) {
            return previous.column_overrides as Record<string, any>;
          }
          return {};
        })();

        const workingOverrides: Record<string, any> = { ...sourceOverrides };
        const rawExisting = workingOverrides[normalizedColumn];
        const existingOverride =
          rawExisting && typeof rawExisting === 'object' && !Array.isArray(rawExisting)
            ? { ...rawExisting }
            : {};

        const mutated = mutator(existingOverride);
        const hasOverride = Boolean(mutated && Object.keys(mutated).length);

        if (hasOverride) {
          workingOverrides[normalizedColumn] = mutated as Record<string, any>;
        } else {
          delete workingOverrides[normalizedColumn];
        }

        const nextState: Record<string, any> = { ...previous };

        if (hasOverride) {
          const currentColumns = new Set(ensureArrayOfString(previous?.columns));
          if (!currentColumns.has(normalizedColumn)) {
            currentColumns.add(normalizedColumn);
            nextState.columns = Array.from(currentColumns).sort((a, b) => a.localeCompare(b));
          }
        }

        if (Object.keys(workingOverrides).length) {
          nextState.column_overrides = workingOverrides;
          nextState.column_strategies = workingOverrides;
        } else {
          if (Object.prototype.hasOwnProperty.call(nextState, 'column_overrides')) {
            delete nextState.column_overrides;
          }
          if (Object.prototype.hasOwnProperty.call(nextState, 'column_strategies')) {
            delete nextState.column_strategies;
          }
        }

        return nextState;
      });
    },
    [setConfigState],
  );

  const handleBinningOverrideStrategyChange = useCallback(
    (
      column: string,
      value: BinningStrategy | '__default__',
      options?: { recommendedBins?: number | null },
    ) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }

      if (value === '__default__') {
        handleBinningClearCustomColumn(normalizedColumn);
        updateBinningColumnOverride(normalizedColumn, () => null);
        return;
      }

      const recommendedBins = Number.isFinite(options?.recommendedBins)
        ? Math.round(options!.recommendedBins as number)
        : null;

      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;

        next.strategy = value;

        const resolveBins = (fallback: number) => {
          const candidate = recommendedBins ?? fallback;
          return Math.min(200, Math.max(2, Math.round(candidate)));
        };

        if (value === 'equal_width') {
          next.equal_width_bins = resolveBins(binningConfig.equalWidthBins);
          delete next.equal_frequency_bins;
          delete next.kbins_n_bins;
          delete next.kbins_encode;
          delete next.kbins_strategy;
        } else if (value === 'equal_frequency') {
          next.equal_frequency_bins = resolveBins(binningConfig.equalFrequencyBins);
          delete next.equal_width_bins;
          delete next.kbins_n_bins;
          delete next.kbins_encode;
          delete next.kbins_strategy;
        } else if (value === 'kbins') {
          next.kbins_n_bins = resolveBins(binningConfig.kbinsNBins);
          next.kbins_encode = binningConfig.kbinsEncode;
          next.kbins_strategy = binningConfig.kbinsStrategy;
          delete next.equal_width_bins;
          delete next.equal_frequency_bins;
        } else if (value === 'custom') {
          delete next.equal_width_bins;
          delete next.equal_frequency_bins;
          delete next.kbins_n_bins;
          delete next.kbins_encode;
          delete next.kbins_strategy;
        }

        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });

      if (value !== 'custom') {
        handleBinningClearCustomColumn(normalizedColumn);
      }
    },
    [
      binningConfig.equalFrequencyBins,
      binningConfig.equalWidthBins,
      binningConfig.kbinsEncode,
      binningConfig.kbinsNBins,
      binningConfig.kbinsStrategy,
      handleBinningClearCustomColumn,
      updateBinningColumnOverride,
    ],
  );

  const handleBinningOverrideNumberChange = useCallback(
    (column: string, field: 'equal_width_bins' | 'equal_frequency_bins' | 'kbins_n_bins', rawValue: string) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }

      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;

        if (!rawValue.trim()) {
          delete next[field];
        } else {
          const numeric = Number(rawValue);
          if (!Number.isFinite(numeric)) {
            return current;
          }
          const clamped = Math.min(200, Math.max(2, Math.round(numeric)));
          next[field] = clamped;
          if (field === 'equal_width_bins') {
            next.strategy = 'equal_width';
            delete next.equal_frequency_bins;
            delete next.kbins_n_bins;
            delete next.kbins_encode;
            delete next.kbins_strategy;
          } else if (field === 'equal_frequency_bins') {
            next.strategy = 'equal_frequency';
            delete next.equal_width_bins;
            delete next.kbins_n_bins;
            delete next.kbins_encode;
            delete next.kbins_strategy;
          } else if (field === 'kbins_n_bins') {
            next.strategy = 'kbins';
            delete next.equal_width_bins;
            delete next.equal_frequency_bins;
          }
        }

        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });

      handleBinningClearCustomColumn(normalizedColumn);
    },
    [handleBinningClearCustomColumn, updateBinningColumnOverride],
  );

  const handleBinningOverrideKbinsEncodeChange = useCallback(
    (column: string, value: KBinsEncode | null) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;
        if (value) {
          next.strategy = 'kbins';
          next.kbins_encode = value;
        } else {
          delete next.kbins_encode;
        }
        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });
    },
    [updateBinningColumnOverride],
  );

  const handleBinningOverrideKbinsStrategyChange = useCallback(
    (column: string, value: KBinsStrategy | null) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      updateBinningColumnOverride(normalizedColumn, (current) => {
        const next = { ...current } as Record<string, any>;
        if (value) {
          next.strategy = 'kbins';
          next.kbins_strategy = value;
        } else {
          delete next.kbins_strategy;
        }
        if (!Object.keys(next).length) {
          return null;
        }
        return next;
      });
    },
    [updateBinningColumnOverride],
  );

  const handleBinningClearOverride = useCallback(
    (column: string) => {
      const normalizedColumn = String(column ?? '').trim();
      if (!normalizedColumn) {
        return;
      }
      handleBinningClearCustomColumn(normalizedColumn);
      updateBinningColumnOverride(normalizedColumn, () => null);
    },
    [handleBinningClearCustomColumn, updateBinningColumnOverride],
  );

  const handleBinningClearOverrides = useCallback(() => {
    setConfigState((previous: any) => {
      const hasOverrides =
        (previous?.column_strategies &&
          typeof previous.column_strategies === 'object' &&
          !Array.isArray(previous.column_strategies) &&
          Object.keys(previous.column_strategies).length > 0) ||
        (previous?.column_overrides &&
          typeof previous.column_overrides === 'object' &&
          !Array.isArray(previous.column_overrides) &&
          Object.keys(previous.column_overrides).length > 0);

      if (!hasOverrides) {
        return previous;
      }

      const nextState = { ...previous } as Record<string, any>;
      if (Object.prototype.hasOwnProperty.call(nextState, 'column_overrides')) {
        delete nextState.column_overrides;
      }
      if (Object.prototype.hasOwnProperty.call(nextState, 'column_strategies')) {
        delete nextState.column_strategies;
      }
      return nextState;
    });
  }, [setConfigState]);

  const handleBinningApplyStrategies = useCallback(() => {
    if (!binningInsightsRecommendations.length) {
      return;
    }
    binningInsightsRecommendations.forEach((entry) => {
      const columnName = typeof entry?.column === 'string' ? entry.column.trim() : '';
      if (!columnName) {
        return;
      }
      const strategy = (entry.recommended_strategy as BinningStrategy) ?? 'equal_width';
      handleBinningOverrideStrategyChange(columnName, strategy, {
        recommendedBins: Number.isFinite(entry?.recommended_bins) ? entry.recommended_bins : null,
      });
    });
  }, [binningInsightsRecommendations, handleBinningOverrideStrategyChange]);

  const handleBinningApplyColumns = useCallback((columns: Iterable<string>) => {
    if (!columns) {
      return;
    }
    const normalizedColumns: string[] = [];
    for (const column of columns) {
      const name = typeof column === 'string' ? column.trim() : '';
      if (!name) {
        continue;
      }
      normalizedColumns.push(name);
    }
    if (!normalizedColumns.length) {
      return;
    }
    setConfigState((previous: any) => {
      const base = previous && typeof previous === 'object' ? previous : {};
      const currentColumns = new Set(ensureArrayOfString(base.columns));
      let changed = false;
      normalizedColumns.forEach((column) => {
        if (currentColumns.has(column)) {
          return;
        }
        currentColumns.add(column);
        changed = true;
      });
      if (!changed) {
        return previous;
      }
      const nextState: Record<string, any> = { ...base };
      nextState.columns = Array.from(currentColumns).sort((a, b) => a.localeCompare(b));
      return nextState;
    });
  }, [setConfigState]);

  const handleApplyAllBinningNumeric = useCallback(() => {
    if (!binningAllNumericColumns.length) {
      return;
    }
    handleBinningApplyColumns(binningAllNumericColumns);
  }, [binningAllNumericColumns, handleBinningApplyColumns]);

  return {
    handleBinningIntegerChange,
    handleBinningBooleanToggle,
    handleBinningSuffixChange,
    handleBinningLabelFormatChange,
    handleBinningMissingStrategyChange,
    handleBinningMissingLabelChange,
    handleBinningCustomBinsChange,
    handleBinningCustomLabelsChange,
    handleBinningClearCustomColumn,
    updateBinningColumnOverride,
    handleBinningOverrideStrategyChange,
    handleBinningOverrideNumberChange,
    handleBinningOverrideKbinsEncodeChange,
    handleBinningOverrideKbinsStrategyChange,
    handleBinningClearOverride,
    handleBinningClearOverrides,
    handleBinningApplyStrategies,
    handleBinningApplyColumns,
    handleApplyAllBinningNumeric,
  };
};
