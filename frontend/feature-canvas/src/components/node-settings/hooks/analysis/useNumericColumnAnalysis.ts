// Used by NodeSettingsModal when configuring scaling and binning nodes.
import { useMemo } from 'react';
import {
  BOOLEAN_FALSE_TOKENS,
  BOOLEAN_TRUE_TOKENS,
} from '../../utils/configParsers';
import { type CatalogFlagMap } from '../core/useCatalogFlags';

// Type tokens for numeric column detection
const NUMERIC_TYPE_TOKENS = [
  'int',
  'float',
  'double',
  'decimal',
  'number',
  'numeric',
  'long',
  'short',
] as const;

const TEMPORAL_TYPE_TOKENS = ['datetime', 'date', 'time', 'timestamp'] as const;

const TEXTUAL_TYPE_TOKENS = ['object', 'string', 'text', 'category', 'categorical', 'char'] as const;

type UseNumericColumnAnalysisParams = {
  catalogFlags: CatalogFlagMap;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  previewSampleRows: Array<Record<string, any>>;
};

type NumericColumnAnalysisResult = {
  numericExcludedColumns: Set<string>;
  numericExclusionReasons: Map<string, string>;
};

export const useNumericColumnAnalysis = ({
  catalogFlags,
  availableColumns,
  columnTypeMap,
  previewSampleRows,
}: UseNumericColumnAnalysisParams): NumericColumnAnalysisResult => {
  const { isBinningNode, isScalingNode, isOutlierNode } = catalogFlags;
  const shouldAnalyze = isBinningNode || isScalingNode || isOutlierNode;

  const analysis = useMemo(() => {
    if (!shouldAnalyze) {
      return {
        excluded: new Set<string>(),
        reasons: new Map<string, string>(),
      };
    }

    const excluded = new Set<string>();
    const reasons = new Map<string, string>();

    availableColumns.forEach((column) => {
      const rawType = (columnTypeMap[column] ?? '').toLowerCase();
      const hasNumericTypeToken = NUMERIC_TYPE_TOKENS.some((token) => rawType.includes(token));
      const isTemporalType = TEMPORAL_TYPE_TOKENS.some((token) => rawType.includes(token));
      const isTextualType = TEXTUAL_TYPE_TOKENS.some((token) => rawType.includes(token));

      if (rawType.includes('bool')) {
        excluded.add(column);
        reasons.set(column, 'Boolean dtype');
        return;
      }

      if ((isTemporalType || isTextualType) && !hasNumericTypeToken) {
        excluded.add(column);
        reasons.set(column, 'Non-numeric dtype');
        return;
      }

      const values = previewSampleRows
        .map((row) => (row && Object.prototype.hasOwnProperty.call(row, column) ? row[column] : undefined))
        .filter((value) => value !== null && value !== undefined);

      if (!values.length) {
        if (!hasNumericTypeToken) {
          excluded.add(column);
          reasons.set(column, 'No numeric samples');
        }
        return;
      }

      let numericConvertible = true;
      let hasNumericSample = false;
      let encounteredNonBinaryNumeric = false;
      const binaryTokens = new Set<string>();

      for (const value of values) {
        if (typeof value === 'boolean') {
          binaryTokens.add(value ? '1' : '0');
          continue;
        }

        if (typeof value === 'number') {
          if (!Number.isFinite(value)) {
            numericConvertible = false;
            break;
          }
          hasNumericSample = true;
          if (Math.abs(value) < 1e-9) {
            binaryTokens.add('0');
          } else if (Math.abs(value - 1) < 1e-9) {
            binaryTokens.add('1');
          } else {
            encounteredNonBinaryNumeric = true;
          }
          continue;
        }

        if (typeof value === 'string') {
          const trimmed = value.trim();
          if (!trimmed) {
            continue;
          }
          const lower = trimmed.toLowerCase();
          if (BOOLEAN_TRUE_TOKENS.has(lower)) {
            binaryTokens.add('1');
            continue;
          }
          if (BOOLEAN_FALSE_TOKENS.has(lower)) {
            binaryTokens.add('0');
            continue;
          }
          const numeric = Number(trimmed);
          if (!Number.isNaN(numeric) && Number.isFinite(numeric)) {
            hasNumericSample = true;
            if (Math.abs(numeric) < 1e-9) {
              binaryTokens.add('0');
            } else if (Math.abs(numeric - 1) < 1e-9) {
              binaryTokens.add('1');
            } else {
              encounteredNonBinaryNumeric = true;
            }
            continue;
          }
          numericConvertible = false;
          break;
        }

        numericConvertible = false;
        break;
      }

      if (!numericConvertible || (!hasNumericSample && !hasNumericTypeToken)) {
        excluded.add(column);
        reasons.set(column, 'Non-numeric values');
        return;
      }

      if (!encounteredNonBinaryNumeric && binaryTokens.size > 0) {
        const isBinary = Array.from(binaryTokens).every((token) => token === '0' || token === '1');
        if (isBinary) {
          excluded.add(column);
          reasons.set(column, 'Binary/bool-like values');
        }
      }
    });

    return {
      excluded,
      reasons,
    };
  }, [availableColumns, columnTypeMap, previewSampleRows, shouldAnalyze]);

  return {
    numericExcludedColumns: analysis.excluded,
    numericExclusionReasons: analysis.reasons,
  };
};
