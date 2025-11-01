import React, { useCallback, useMemo } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { ensureArrayOfString } from '../../sharedUtils';

const COMMON_DTYPE_OPTIONS = ['float64', 'Int64', 'boolean', 'string', 'category', 'datetime64[ns]', 'object'];

export type CastColumnTypesSectionProps = {
  configState: Record<string, any>;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  columnSuggestions: Record<string, string[]>;
  selectedColumns: string[];
  previewColumns: string[];
  previewStatus: 'idle' | 'loading' | 'success' | 'error';
  sourceId?: string | null;
  hasReachableSource: boolean;
};

type CastTableRow = {
  column: string;
  originalType: string;
  recommended: string | null;
  currentTarget: string;
  suggestions: string[];
  options: string[];
  hasRecommendationGap: boolean;
  hasOverride: boolean;
  usesDefaultTarget: boolean;
  isRecommendationApplied: boolean;
  isSelected: boolean;
};

export const CastColumnTypesSection: React.FC<CastColumnTypesSectionProps> = ({
  configState,
  setConfigState,
  availableColumns,
  columnTypeMap,
  columnSuggestions,
  selectedColumns,
  previewColumns,
  previewStatus,
  sourceId,
  hasReachableSource,
}) => {
  const columnOverrides = useMemo(() => {
    const raw = configState?.column_overrides;
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      return {} as Record<string, string>;
    }

    const entries: Record<string, string> = {};
    Object.entries(raw).forEach(([name, value]) => {
      const key = String(name ?? '').trim();
      if (!key) {
        return;
      }
      if (typeof value !== 'string') {
        return;
      }
      const dtype = value.trim();
      if (!dtype) {
        return;
      }
      entries[key] = dtype;
    });
    return entries;
  }, [configState?.column_overrides]);

  const castDefaultTarget = useMemo(() => {
    const raw = configState?.target_dtype;
    return typeof raw === 'string' ? raw.trim() : '';
  }, [configState?.target_dtype]);

  const castTableRows = useMemo<CastTableRow[]>(() => {
    const columnSet = new Set<string>();
    availableColumns.forEach((column) => columnSet.add(column));
    Object.keys(columnTypeMap).forEach((column) => columnSet.add(column));
    Object.keys(columnSuggestions).forEach((column) => columnSet.add(column));
    Object.keys(columnOverrides).forEach((column) => columnSet.add(column));
    selectedColumns.forEach((column) => columnSet.add(column));
    previewColumns.forEach((column) => {
      if (typeof column === 'string') {
        columnSet.add(column);
      }
    });

    const orderedColumns = Array.from(columnSet).sort((a, b) => a.localeCompare(b));

    return orderedColumns.map((column) => {
      const originalType = columnTypeMap[column] ?? 'Unknown';
      const suggestions = columnSuggestions[column] ?? [];
      const recommended = suggestions.length ? suggestions[0] : null;
      const hasOverride = Object.prototype.hasOwnProperty.call(columnOverrides, column);
      const overrideValue = hasOverride ? columnOverrides[column] : null;
      const normalizedOverride = typeof overrideValue === 'string' ? overrideValue.trim() : '';
      const currentTarget = hasOverride ? normalizedOverride : castDefaultTarget;
      const optionSet = new Set<string>();

      if (currentTarget) {
        optionSet.add(currentTarget);
      }
      suggestions.forEach((dtype) => optionSet.add(dtype));
      if (originalType && originalType !== 'Unknown') {
        optionSet.add(originalType);
      }
      if (castDefaultTarget) {
        optionSet.add(castDefaultTarget);
      }
      COMMON_DTYPE_OPTIONS.forEach((dtype) => optionSet.add(dtype));

      const effectiveTarget = currentTarget ?? '';
      const hasRecommendationGap = Boolean(recommended && recommended !== effectiveTarget);
      const isRecommendationApplied = Boolean(recommended && recommended === effectiveTarget);
      const isSelected = selectedColumns.includes(column);

      const options = Array.from(optionSet).sort((a, b) => a.localeCompare(b));

      return {
        column,
        originalType,
        recommended,
        currentTarget,
        suggestions,
        options,
        hasRecommendationGap,
        hasOverride,
        usesDefaultTarget: !hasOverride,
        isRecommendationApplied,
        isSelected,
      };
    });
  }, [
    availableColumns,
    castDefaultTarget,
    columnOverrides,
    columnSuggestions,
    columnTypeMap,
    previewColumns,
    selectedColumns,
  ]);

  const castTableGroups = useMemo(() => {
    const needsAttention = castTableRows.filter((row) => row.hasRecommendationGap);
    const aligned = castTableRows.filter((row) => !row.hasRecommendationGap);

    const groups: Array<{ key: string; label: string; rows: CastTableRow[] }> = [];
    if (needsAttention.length) {
      groups.push({ key: 'needs-attention', label: 'Needs attention', rows: needsAttention });
    }
    if (aligned.length) {
      groups.push({ key: 'all-columns', label: needsAttention.length ? 'Other columns' : 'All columns', rows: aligned });
    }
    return groups;
  }, [castTableRows]);

  const castAttentionCount = useMemo(
    () => castTableRows.filter((row) => row.hasRecommendationGap).length,
    [castTableRows]
  );
  const hasCastRecommendations = useMemo(() => castAttentionCount > 0, [castAttentionCount]);
  const castSelectedCount = useMemo(
    () => castTableRows.filter((row) => row.isSelected).length,
    [castTableRows]
  );

  const handleSetColumnOverride = useCallback(
    (column: string, dtype: string | null) => {
      const normalizedColumn = column.trim();
      if (!normalizedColumn) {
        return;
      }
      setConfigState((previous) => {
        const currentOverrides =
          previous.column_overrides &&
          typeof previous.column_overrides === 'object' &&
          !Array.isArray(previous.column_overrides)
            ? { ...previous.column_overrides }
            : {};
        const trimmedDtype = typeof dtype === 'string' ? dtype.trim() : '';
        const nextColumns = new Set(ensureArrayOfString(previous.columns));
        const skippedColumns = new Set(ensureArrayOfString(previous.skipped_columns));
        const hadSkip = skippedColumns.delete(normalizedColumn);

        if (!trimmedDtype) {
          delete currentOverrides[normalizedColumn];
        } else {
          currentOverrides[normalizedColumn] = trimmedDtype;
          nextColumns.add(normalizedColumn);
        }

        const result: Record<string, any> = {
          ...previous,
          columns: Array.from(nextColumns).sort((a, b) => a.localeCompare(b)),
          column_overrides: currentOverrides,
        };
        if (hadSkip) {
          result.skipped_columns = Array.from(skippedColumns).sort((a, b) => a.localeCompare(b));
        }
        return result;
      });
    },
    [setConfigState]
  );

  const handleApplyAllColumnRecommendations = useCallback(() => {
    const recommendedEntries = Object.entries(columnSuggestions).filter(([, suggestions]) => suggestions.length > 0);
    if (!recommendedEntries.length) {
      return;
    }
    setConfigState((previous) => {
      const nextColumns = new Set(ensureArrayOfString(previous.columns));
      const currentOverrides =
        previous.column_overrides &&
        typeof previous.column_overrides === 'object' &&
        !Array.isArray(previous.column_overrides)
          ? { ...previous.column_overrides }
          : {};

      recommendedEntries.forEach(([column, suggestions]) => {
        const normalizedColumn = column.trim();
        if (!normalizedColumn) {
          return;
        }
        nextColumns.add(normalizedColumn);
        if (suggestions.length) {
          currentOverrides[normalizedColumn] = suggestions[0];
        }
      });

      return {
        ...previous,
        columns: Array.from(nextColumns).sort((a, b) => a.localeCompare(b)),
        column_overrides: currentOverrides,
      };
    });
  }, [columnSuggestions, setConfigState]);

  const handleCastTargetDropdownChange = useCallback(
    (column: string, value: string) => {
      if (value === '__default__') {
        handleSetColumnOverride(column, null);
      } else {
        handleSetColumnOverride(column, value);
      }
    },
    [handleSetColumnOverride]
  );

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Column dtype alignment</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleApplyAllColumnRecommendations}
            disabled={!hasCastRecommendations}
          >
            Apply recommendations
          </button>
        </div>
      </div>
      {!sourceId && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Select a dataset to analyse current column dtypes and suggestions.
        </p>
      )}
      {sourceId && !hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Connect this step to an upstream output to load dtype insights.
        </p>
      )}
      {castDefaultTarget ? (
        <p className="canvas-modal__note">
          Global target dtype: <strong>{castDefaultTarget}</strong>. Columns inherit this unless an override is set.
        </p>
      ) : (
        <p className="canvas-modal__note">Set a global target dtype or override individual columns below.</p>
      )}
      {castTableRows.length > 0 && (
        <p className="canvas-modal__note">
          Tracking {castTableRows.length} column{castTableRows.length === 1 ? '' : 's'} · {castSelectedCount} mapped in node selection.
        </p>
      )}
      {hasCastRecommendations ? (
        <p className="canvas-modal__note">
          {castAttentionCount} column{castAttentionCount === 1 ? '' : 's'} need attention. Apply them in bulk or adjust individually.
        </p>
      ) : (
        castTableRows.length > 0 && (
          <p className="canvas-modal__note">All recommendations are in sync. Fine-tune overrides per column as needed.</p>
        )
      )}
      {previewStatus === 'loading' && !castTableRows.length && (
        <p className="canvas-modal__note">Loading column type insights…</p>
      )}
      {castTableRows.length === 0 ? (
        <p className="canvas-modal__note">Add columns or refresh the dataset preview to surface dtype intelligence.</p>
      ) : (
        <div className="canvas-cast__body">
          {castTableGroups.map((group) => (
            <div key={`cast-group-${group.key}`} className="canvas-cast__group">
              {group.label && (
                <div className="canvas-cast__group-header">
                  <h4>{group.label}</h4>
                  <span>
                    {group.rows.length} column{group.rows.length === 1 ? '' : 's'}
                  </span>
                </div>
              )}
              <div className="canvas-cast__table-wrapper">
                <table className="canvas-cast__table">
                  <thead>
                    <tr>
                      <th scope="col">Column</th>
                      <th scope="col">Original dtype</th>
                      <th scope="col">Recommended</th>
                      <th scope="col">Target dtype</th>
                    </tr>
                  </thead>
                  <tbody>
                    {group.rows.map((row) => {
                      const rowClasses = ['canvas-cast__row'];
                      if (row.hasRecommendationGap) {
                        rowClasses.push('canvas-cast__row--attention');
                      }
                      if (row.hasOverride) {
                        rowClasses.push('canvas-cast__row--override');
                      }
                      const selectValue = row.hasOverride ? row.currentTarget || '' : '__default__';
                      const recommendationApplied = row.isRecommendationApplied;
                      const additionalSuggestions = row.suggestions.slice(1, 4);
                      const helperText = row.hasOverride
                        ? row.currentTarget
                          ? `Override → ${row.currentTarget}`
                          : 'Override cleared (uses original dtype)'
                        : castDefaultTarget
                          ? `Inheriting ${castDefaultTarget}`
                          : 'Retains original dtype until a target is set';
                      return (
                        <tr key={`cast-row-${row.column}`} className={rowClasses.join(' ')}>
                          <th scope="row">
                            <div className="canvas-cast__column-cell">
                              <span className="canvas-cast__column-name">{row.column}</span>
                              <div className="canvas-cast__column-meta">
                                {row.isSelected && (
                                  <span className="canvas-cast__badge canvas-cast__badge--selected">In node</span>
                                )}
                                {row.hasOverride && (
                                  <span className="canvas-cast__badge canvas-cast__badge--override">Override</span>
                                )}
                                {!row.isSelected && !row.hasOverride && (
                                  <span className="canvas-cast__muted">Not yet in node</span>
                                )}
                              </div>
                            </div>
                          </th>
                          <td>
                            <span className="canvas-cast__muted">{row.originalType}</span>
                          </td>
                          <td>
                            {row.recommended ? (
                              <div className="canvas-cast__recommendation">
                                <span
                                  className={`canvas-cast__chip${row.hasRecommendationGap ? ' canvas-cast__chip--attention' : ''}${recommendationApplied ? ' canvas-cast__chip--applied' : ''}`}
                                >
                                  {row.recommended}
                                </span>
                                <span className="canvas-cast__recommendation-note">
                                  {recommendationApplied ? 'Aligned with target' : 'Not applied yet'}
                                </span>
                                {additionalSuggestions.length > 0 && (
                                  <span className="canvas-cast__suggestions">
                                    Other options: {additionalSuggestions.join(', ')}
                                  </span>
                                )}
                              </div>
                            ) : (
                              <span className="canvas-cast__muted">No recommendation</span>
                            )}
                          </td>
                          <td>
                            <div className="canvas-cast__target">
                              <select
                                value={selectValue}
                                onChange={(event) => handleCastTargetDropdownChange(row.column, event.target.value)}
                              >
                                <option value="__default__">
                                  {castDefaultTarget ? `Inherit global (${castDefaultTarget})` : 'No override (use original)'}
                                </option>
                                {row.options.map((option) => (
                                  <option key={`${row.column}-${option}`} value={option}>
                                    {option}
                                  </option>
                                ))}
                              </select>
                              <span className="canvas-cast__target-note">{helperText}</span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
};
