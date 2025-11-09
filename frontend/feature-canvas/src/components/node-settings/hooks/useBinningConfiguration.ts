import { useEffect, useMemo, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import {
  BINNING_STRATEGY_LABELS,
  buildBinningOverrideSummary,
  normalizeBinningConfigValue,
  type NormalizedBinningConfig,
} from '../nodes/binning/binningSettings';

export type UseBinningConfigurationArgs = {
  configState: Record<string, any>;
  nodeId: string;
};

export type UseBinningConfigurationResult = {
  binningConfig: NormalizedBinningConfig;
  binningSelectedCount: number;
  binningDefaultLabel: string;
  binningOverrideColumns: string[];
  binningOverrideCount: number;
  binningOverrideSummary: string | null;
  fieldIds: {
    equalWidth: string;
    equalFrequency: string;
    precision: string;
    suffix: string;
    missingLabel: string;
    includeLowest: string;
    dropOriginal: string;
    labelFormat: string;
    duplicates: string;
    kbinsNBins: string;
    kbinsEncode: string;
    kbinsStrategy: string;
  };
  customEdgeDrafts: Record<string, string>;
  setCustomEdgeDrafts: Dispatch<SetStateAction<Record<string, string>>>;
  customLabelDrafts: Record<string, string>;
  setCustomLabelDrafts: Dispatch<SetStateAction<Record<string, string>>>;
};

export const useBinningConfiguration = ({
  configState,
  nodeId,
}: UseBinningConfigurationArgs): UseBinningConfigurationResult => {
  const binningConfig = useMemo<NormalizedBinningConfig>(
    () => normalizeBinningConfigValue(configState),
    [configState],
  );

  const binningSelectedCount = binningConfig.columns.length;
  const binningDefaultLabel = BINNING_STRATEGY_LABELS[binningConfig.strategy] ?? binningConfig.strategy;

  const binningOverrideColumns = useMemo(
    () =>
      Object.keys(binningConfig.columnStrategies)
        .map((column) => column.trim())
        .filter(Boolean)
        .sort((a, b) => a.localeCompare(b)),
    [binningConfig.columnStrategies],
  );

  const binningOverrideCount = binningOverrideColumns.length;

  const binningOverrideSummary = useMemo(() => {
    if (!binningOverrideColumns.length) {
      return null;
    }
    const preview = binningOverrideColumns.slice(0, 4);
    return buildBinningOverrideSummary(preview, binningConfig.columnStrategies, binningOverrideCount);
  }, [binningConfig.columnStrategies, binningOverrideColumns, binningOverrideCount]);

  const fieldPrefix = useMemo(() => `binning-${nodeId || 'node'}`, [nodeId]);

  const fieldIds = useMemo(
    () => ({
      equalWidth: `${fieldPrefix}-equal-width-bins`,
      equalFrequency: `${fieldPrefix}-equal-frequency-bins`,
      precision: `${fieldPrefix}-precision`,
      suffix: `${fieldPrefix}-suffix`,
      missingLabel: `${fieldPrefix}-missing-label`,
      includeLowest: `${fieldPrefix}-include-lowest`,
      dropOriginal: `${fieldPrefix}-drop-original`,
      labelFormat: `${fieldPrefix}-label-format`,
      duplicates: `${fieldPrefix}-duplicates`,
      kbinsNBins: `${fieldPrefix}-kbins-n-bins`,
      kbinsEncode: `${fieldPrefix}-kbins-encode`,
      kbinsStrategy: `${fieldPrefix}-kbins-strategy`,
    }),
    [fieldPrefix],
  );

  const [customEdgeDrafts, setCustomEdgeDrafts] = useState<Record<string, string>>({});
  const [customLabelDrafts, setCustomLabelDrafts] = useState<Record<string, string>>({});

  useEffect(() => {
    setCustomEdgeDrafts((previous) => {
      const activeColumns = new Set<string>();
      const next: Record<string, string> = { ...previous };

      binningConfig.columns.forEach((column) => {
        const normalized = String(column ?? '').trim();
        if (!normalized) {
          return;
        }
        activeColumns.add(normalized);
        if (Object.prototype.hasOwnProperty.call(next, normalized)) {
          return;
        }
        const configured = binningConfig.customBins[normalized] ?? [];
        next[normalized] = configured.length >= 2 ? configured.join(', ') : '';
      });

      Object.keys(next).forEach((key) => {
        if (!activeColumns.has(key)) {
          delete next[key];
        }
      });

      return next;
    });
  }, [binningConfig.columns, binningConfig.customBins]);

  useEffect(() => {
    setCustomLabelDrafts((previous) => {
      const activeColumns = new Set<string>();
      const next: Record<string, string> = { ...previous };

      binningConfig.columns.forEach((column) => {
        const normalized = String(column ?? '').trim();
        if (!normalized) {
          return;
        }
        activeColumns.add(normalized);
        if (Object.prototype.hasOwnProperty.call(next, normalized)) {
          return;
        }
        const configured = binningConfig.customLabels[normalized] ?? [];
        next[normalized] = configured.length > 0 ? configured.join(', ') : '';
      });

      Object.keys(next).forEach((key) => {
        if (!activeColumns.has(key)) {
          delete next[key];
        }
      });

      return next;
    });
  }, [binningConfig.columns, binningConfig.customLabels]);

  return {
    binningConfig,
    binningSelectedCount,
    binningDefaultLabel,
    binningOverrideColumns,
    binningOverrideCount,
    binningOverrideSummary,
    fieldIds,
    customEdgeDrafts,
    setCustomEdgeDrafts,
    customLabelDrafts,
    setCustomLabelDrafts,
  };
};
