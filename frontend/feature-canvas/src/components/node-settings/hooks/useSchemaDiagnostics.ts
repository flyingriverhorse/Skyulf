import { useMemo } from 'react';
import type { PipelinePreviewSchema } from '../../../api';
import type { ResamplingSchemaGuard } from '../nodes/resampling/ClassResamplingSection';
import {
  isNumericImputationMethod,
  type ImputationStrategyConfig,
} from '../nodes/imputation/imputationSettings';
import type { ImputationSchemaDiagnostics } from '../nodes/imputation/ImputationStrategiesSection';

interface UseSchemaDiagnosticsOptions {
  cachedPreviewSchema: PipelinePreviewSchema | null;
  isClassOversamplingNode: boolean;
  resamplingTargetColumn: string | null;
  isImputerNode: boolean;
  imputerStrategies: ImputationStrategyConfig[];
}

interface UseSchemaDiagnosticsResult {
  cachedSchemaColumns: NonNullable<PipelinePreviewSchema['columns']>;
  oversamplingSchemaGuard: ResamplingSchemaGuard | null;
  imputationSchemaDiagnostics: ImputationSchemaDiagnostics | null;
  skipPreview: boolean;
}

export const useSchemaDiagnostics = ({
  cachedPreviewSchema,
  isClassOversamplingNode,
  resamplingTargetColumn,
  isImputerNode,
  imputerStrategies,
}: UseSchemaDiagnosticsOptions): UseSchemaDiagnosticsResult => {
  const cachedSchemaColumns = useMemo(() => cachedPreviewSchema?.columns ?? [], [cachedPreviewSchema]);

  const oversamplingSchemaGuard = useMemo<ResamplingSchemaGuard | null>(() => {
    if (!isClassOversamplingNode) {
      return null;
    }
    if (!cachedSchemaColumns.length) {
      return null;
    }
    if (!resamplingTargetColumn) {
      return null;
    }

    const allowedFamilies = new Set(['numeric', 'integer', 'boolean']);
    const blockedDetails = cachedSchemaColumns
      .filter((column) => column && column.name !== resamplingTargetColumn)
      .filter((column) => !allowedFamilies.has(String(column.logical_family ?? 'unknown')))
      .map((column) => ({
        name: column.name,
        logical_family: String(column.logical_family ?? 'unknown'),
      }));

    if (!blockedDetails.length) {
      return null;
    }

    return {
      blocked: true,
      message:
        'Class oversampling requires numeric feature columns. Encode or cast the listed fields before refreshing the preview.',
      columns: blockedDetails.map((detail) => detail.name),
      details: blockedDetails,
    };
  }, [cachedSchemaColumns, isClassOversamplingNode, resamplingTargetColumn]);

  const imputationSchemaDiagnostics = useMemo<ImputationSchemaDiagnostics | null>(() => {
    if (!isImputerNode) {
      return null;
    }
    if (!cachedSchemaColumns.length) {
      return null;
    }
    if (!imputerStrategies.length) {
      return null;
    }

    const numericFamilies = new Set(['numeric', 'integer']);
    const columnLookup = new Map(cachedSchemaColumns.map((column) => [column.name, column]));
    const entries: ImputationSchemaDiagnostics['entries'] = [];

    imputerStrategies.forEach((strategy, index) => {
      if (!isNumericImputationMethod(strategy.method)) {
        return;
      }
      if (!Array.isArray(strategy.columns) || !strategy.columns.length) {
        return;
      }

      const invalidDetails = strategy.columns
        .map((columnName) => columnLookup.get(columnName))
        .filter((column): column is NonNullable<(typeof cachedSchemaColumns)[number]> => Boolean(column))
        .filter((column) => !numericFamilies.has(String(column.logical_family ?? 'unknown')))
        .map((column) => ({
          name: column.name,
          logical_family: String(column.logical_family ?? 'unknown'),
        }));

      if (invalidDetails.length) {
        entries.push({
          index,
          columns: invalidDetails.map((detail) => detail.name),
          details: invalidDetails,
        });
      }
    });

    if (!entries.length) {
      return null;
    }

    return {
      blocked: true,
      message:
        'Numeric imputation methods can only target numeric columns. Recast or adjust the highlighted fields before running the preview.',
      entries,
    };
  }, [cachedSchemaColumns, imputerStrategies, isImputerNode]);

  const skipPreview = Boolean(oversamplingSchemaGuard?.blocked || imputationSchemaDiagnostics?.blocked);

  return {
    cachedSchemaColumns,
    oversamplingSchemaGuard,
    imputationSchemaDiagnostics,
    skipPreview,
  };
};
