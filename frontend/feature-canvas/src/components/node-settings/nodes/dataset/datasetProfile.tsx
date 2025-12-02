import React, { useCallback, useEffect, useMemo, useState } from 'react';
import type { Node } from 'react-flow-renderer';
import { fetchQuickProfile, type QuickProfileResponse } from '../../../../api';
import { PendingConfigurationDetail } from '../../../types';
import { extractPendingConfigurationDetails } from '../../../utils/extractPendingConfigurationDetails';

export type ProfilingSamplePresetValue = '200' | '500' | '1000' | 'all';

export const PROFILING_SAMPLE_PRESETS: Array<{ value: ProfilingSamplePresetValue; label: string }> = [
  { value: '200', label: '200 rows' },
  { value: '500', label: '500 rows' },
  { value: '1000', label: '1,000 rows' },
  { value: 'all', label: 'Full dataset' },
];

export type ProfilingState = {
  status: 'idle' | 'loading' | 'success' | 'error';
  data: QuickProfileResponse | null;
  error: string | null;
};

type DatasetProfileArgs = {
  node: Node;
  isDatasetProfileNode: boolean;
  sourceId?: string | null;
  hasReachableSource: boolean;
  graphContext: { nodes: any[]; edges: any[] } | null;
  profilingGraphSignature: string;
  formatRelativeTime: (value?: string | null) => string | null;
};

export type DatasetProfileController = {
  profileState: ProfilingState;
  profileData: QuickProfileResponse | null;
  profilingSamplePreset: ProfilingSamplePresetValue;
  setProfilingSamplePreset: (value: ProfilingSamplePresetValue) => void;
  profilingActiveTab: 'summary' | 'correlations';
  setProfilingActiveTab: (tab: 'summary' | 'correlations') => void;
  isProfilingWarningsOpen: boolean;
  setIsProfilingWarningsOpen: (value: boolean | ((value: boolean) => boolean)) => void;
  handleRefreshDatasetProfile: () => void;
  activeProfilingSamplePreset: { value: ProfilingSamplePresetValue; label: string };
  profilingWarningsListId: string;
  profilingSummaryPanelId: string;
  profilingCorrelationsPanelId: string;
  profilingSummaryTabId: string;
  profilingCorrelationsTabId: string;
  profilingCorrelationCount: number;
  relativeProfilingGeneratedAt: string | null;
};

export const useDatasetProfileController = ({
  node,
  isDatasetProfileNode,
  sourceId,
  hasReachableSource,
  graphContext,
  profilingGraphSignature,
  formatRelativeTime,
}: DatasetProfileArgs): DatasetProfileController => {
  const [profileState, setProfileState] = useState<ProfilingState>({
    status: 'idle',
    data: null,
    error: null,
  });
  const [profileRequestId, setProfileRequestId] = useState(0);
  const [profileCache, setProfileCache] = useState<Record<string, QuickProfileResponse>>({});
  const [profilingSamplePreset, setProfilingSamplePreset] = useState<ProfilingSamplePresetValue>('500');
  const [profilingActiveTab, setProfilingActiveTab] = useState<'summary' | 'correlations'>('summary');
  const [isProfilingWarningsOpen, setIsProfilingWarningsOpen] = useState(false);

  const profilingCacheKey = useMemo(() => {
    if (!isDatasetProfileNode || !sourceId) {
      return null;
    }
    return `${sourceId}::${node?.id ?? ''}::${profilingSamplePreset}::${profilingGraphSignature}`;
  }, [isDatasetProfileNode, node?.id, profilingGraphSignature, profilingSamplePreset, sourceId]);

  const cachedProfileResult = useMemo(() => {
    if (!profilingCacheKey) {
      return null;
    }
    return profileCache[profilingCacheKey] ?? null;
  }, [profileCache, profilingCacheKey]);

  useEffect(() => {
    setProfilingSamplePreset('500');
    setProfileState({ status: 'idle', data: null, error: null });
  }, [isDatasetProfileNode, node?.id]);

  useEffect(() => {
    if (!isDatasetProfileNode) {
      setProfileState({ status: 'idle', data: null, error: null });
      return;
    }

    if (!sourceId) {
      setProfileState({
        status: 'error',
        data: null,
        error: 'Select a dataset to generate a lightweight profile.',
      });
      return;
    }

    if (!hasReachableSource) {
      setProfileState({
        status: 'error',
        data: null,
        error: 'Connect this step to an upstream output to generate a profile.',
      });
      return;
    }

    if (cachedProfileResult) {
      setProfileState({
        status: 'success',
        data: cachedProfileResult,
        error: null,
      });
      return;
    }

    let isActive = true;
    setProfileState({ status: 'loading', data: null, error: null });

    const selectedPreset =
      PROFILING_SAMPLE_PRESETS.find((preset) => preset.value === profilingSamplePreset) ??
      PROFILING_SAMPLE_PRESETS[1] ??
      PROFILING_SAMPLE_PRESETS[0];

    const requestedSampleSize: number | 'all' =
      selectedPreset.value === 'all' ? 'all' : Number(selectedPreset.value);

    fetchQuickProfile(sourceId, {
      sampleSize: requestedSampleSize,
      graph: graphContext,
      targetNodeId: node?.id ?? null,
    })
      .then((result) => {
        if (!isActive) {
          return;
        }
        if (profilingCacheKey && result) {
          setProfileCache((previous) => ({
            ...previous,
            [profilingCacheKey]: result,
          }));
        }
        setProfileState({
          status: 'success',
          data: result ?? null,
          error: null,
        });
      })
      .catch((error: any) => {
        if (!isActive) {
          return;
        }
        setProfileState({
          status: 'error',
          data: null,
          error: error?.message ?? 'Unable to generate dataset profile',
        });
      });

    return () => {
      isActive = false;
    };
  }, [
    cachedProfileResult,
    graphContext,
    hasReachableSource,
    isDatasetProfileNode,
    node?.id,
    profileRequestId,
    profilingCacheKey,
    profilingSamplePreset,
    sourceId,
  ]);

  const profileData = profileState.data;
  const profileGeneratedAtKey = profileData?.generated_at ?? null;
  const profilingCorrelationCount = profileData?.correlations?.length ?? 0;

  useEffect(() => {
    setIsProfilingWarningsOpen(false);
    setProfilingActiveTab('summary');
  }, [isDatasetProfileNode, profileGeneratedAtKey]);

  useEffect(() => {
    if (profilingActiveTab === 'correlations' && profilingCorrelationCount === 0) {
      setProfilingActiveTab('summary');
    }
  }, [profilingActiveTab, profilingCorrelationCount]);

  const handleRefreshDatasetProfile = useCallback(() => {
    if (profilingCacheKey) {
      setProfileCache((previous) => {
        if (!Object.prototype.hasOwnProperty.call(previous, profilingCacheKey)) {
          return previous;
        }
        const next = { ...previous };
        delete next[profilingCacheKey];
        return next;
      });
    }
    setProfileRequestId((value) => value + 1);
  }, [profilingCacheKey]);

  const profilingWarningsListId = useMemo(
    () => (node?.id ? `profiling-warnings-${node.id}` : 'profiling-warnings'),
    [node?.id],
  );

  const profilingSummaryPanelId = useMemo(
    () => (node?.id ? `profiling-summary-${node.id}` : 'profiling-summary'),
    [node?.id],
  );

  const profilingCorrelationsPanelId = useMemo(
    () => (node?.id ? `profiling-correlations-${node.id}` : 'profiling-correlations'),
    [node?.id],
  );

  const profilingSummaryTabId = useMemo(
    () => (node?.id ? `profiling-summary-tab-${node.id}` : 'profiling-summary-tab'),
    [node?.id],
  );

  const profilingCorrelationsTabId = useMemo(
    () => (node?.id ? `profiling-correlations-tab-${node.id}` : 'profiling-correlations-tab'),
    [node?.id],
  );

  const activeProfilingSamplePreset = useMemo(
    () =>
      PROFILING_SAMPLE_PRESETS.find((preset) => preset.value === profilingSamplePreset) ??
      PROFILING_SAMPLE_PRESETS[0],
    [profilingSamplePreset],
  );

  const relativeProfilingGeneratedAt = useMemo(
    () => formatRelativeTime(profileData?.generated_at ?? null),
    [formatRelativeTime, profileData?.generated_at],
  );

  return {
    profileState,
    profileData,
    profilingSamplePreset,
    setProfilingSamplePreset,
    profilingActiveTab,
    setProfilingActiveTab,
    isProfilingWarningsOpen,
    setIsProfilingWarningsOpen,
    handleRefreshDatasetProfile,
    activeProfilingSamplePreset,
    profilingWarningsListId,
    profilingSummaryPanelId,
    profilingCorrelationsPanelId,
    profilingSummaryTabId,
    profilingCorrelationsTabId,
    profilingCorrelationCount,
    relativeProfilingGeneratedAt,
  };
};

export type DatasetProfileSectionProps = {
  isDatasetProfileNode: boolean;
  controller: DatasetProfileController;
  formatCellValue: (value: any) => string;
  formatNumericStat: (value?: number | null) => string;
  formatMissingPercentage: (value: number) => string;
  previewState?: any;
  onPendingConfigurationWarning?: (details: PendingConfigurationDetail[]) => void;
};

export const DatasetProfileSection: React.FC<DatasetProfileSectionProps> = ({
  isDatasetProfileNode,
  controller,
  formatCellValue,
  formatNumericStat,
  formatMissingPercentage,
  previewState,
  onPendingConfigurationWarning,
}) => {
  if (!isDatasetProfileNode) {
    return null;
  }

  const {
    profileState,
    profileData,
    profilingSamplePreset,
    setProfilingSamplePreset,
    profilingActiveTab,
    setProfilingActiveTab,
    isProfilingWarningsOpen,
    setIsProfilingWarningsOpen,
    handleRefreshDatasetProfile,
    activeProfilingSamplePreset,
    profilingWarningsListId,
    profilingSummaryPanelId,
    profilingCorrelationsPanelId,
    profilingSummaryTabId,
    profilingCorrelationsTabId,
    profilingCorrelationCount,
    relativeProfilingGeneratedAt,
  } = controller;

  const hasProfileData = profileState.status === 'success' && Boolean(profileData);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Dataset profile</h3>
        <div className="canvas-modal__section-actions">
          <div className="canvas-sample__group">
            <span className="canvas-sample__label">Sample</span>
            <div className="canvas-skewness__segmented" role="group" aria-label="Profile sampling presets">
              {PROFILING_SAMPLE_PRESETS.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  className="canvas-skewness__segmented-button"
                  data-active={preset.value === profilingSamplePreset}
                  onClick={() => setProfilingSamplePreset(preset.value)}
                  disabled={profileState.status === 'loading'}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={handleRefreshDatasetProfile}
            disabled={profileState.status === 'loading'}
          >
            {profileState.status === 'loading' ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>
      {relativeProfilingGeneratedAt && (
        <span className="canvas-modal__meta">Updated {relativeProfilingGeneratedAt}</span>
      )}
      <p className="canvas-modal__note">
        Sampling preset: <strong>{activeProfilingSamplePreset.label}</strong>.
      </p>
      {profileState.status === 'loading' && (
        <p className="canvas-modal__note">Generating dataset profile…</p>
      )}
      {profileState.status === 'error' && (
        <p className="canvas-modal__note canvas-modal__note--error">{profileState.error}</p>
      )}
      {hasProfileData && profileData && (
        <>
          <div className="canvas-profiling__meta">
            <span>Rows analysed: {profileData.rows_analyzed.toLocaleString()}</span>
            <span>Columns: {profileData.columns_analyzed.toLocaleString()}</span>
            <span>Sample size: {profileData.sample_size.toLocaleString()} rows</span>
            <span>Missing cells: {profileData.metrics.missing_cells.toLocaleString()}</span>
            <span>Duplicate rows: {profileData.metrics.duplicate_rows.toLocaleString()}</span>
          </div>
          <div className="canvas-profiling__tabs" role="tablist" aria-label="Dataset profile views">
            <button
              type="button"
              className="canvas-profiling__tab"
              id={profilingSummaryTabId}
              role="tab"
              data-active={profilingActiveTab === 'summary'}
              aria-selected={profilingActiveTab === 'summary'}
              aria-controls={profilingSummaryPanelId}
              onClick={() => setProfilingActiveTab('summary')}
            >
              Summary
            </button>
            <button
              type="button"
              className="canvas-profiling__tab"
              id={profilingCorrelationsTabId}
              role="tab"
              data-active={profilingActiveTab === 'correlations'}
              aria-selected={profilingActiveTab === 'correlations'}
              aria-controls={profilingCorrelationsPanelId}
              onClick={() => setProfilingActiveTab('correlations')}
              disabled={!profilingCorrelationCount}
            >
              Correlations
            </button>
          </div>
          {profilingActiveTab === 'summary' && (
            <div
              className="canvas-profiling__panel"
              id={profilingSummaryPanelId}
              role="tabpanel"
              aria-labelledby={profilingSummaryTabId}
            >
              {profileData.warnings && profileData.warnings.length > 0 ? (
                <div className="canvas-profiling__warnings">
                  <button
                    type="button"
                    className="canvas-profiling__warnings-toggle"
                    onClick={() => setIsProfilingWarningsOpen((value) => !value)}
                    aria-expanded={isProfilingWarningsOpen}
                    aria-controls={profilingWarningsListId}
                  >
                    {isProfilingWarningsOpen
                      ? 'Hide warnings'
                      : `Show warnings (${profileData.warnings.length})`}
                  </button>
                  {isProfilingWarningsOpen && (
                    <ul className="canvas-profiling__warnings-list" id={profilingWarningsListId}>
                      {profileData.warnings.map((warning, index) => (
                        <li key={`profiling-warning-${index}`}>{warning}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : null}
              <div className="canvas-quick-profile__columns">
                {profileData.columns.map((column) => {
                  const numericSummary = column.numeric_summary;
                  const hasNumericSummary = Boolean(numericSummary);
                  const topValues = Array.isArray(column.top_values) ? column.top_values : [];
                  const sampleValues = Array.isArray(column.sample_values) ? column.sample_values : [];

                  return (
                    <article
                      key={column.name}
                      className="canvas-quick-profile__column"
                      aria-label={`${column.name} summary`}
                    >
                      <header className="canvas-quick-profile__column-header">
                        <h4>{column.name}</h4>
                        <span className="canvas-quick-profile__dtype">
                          {column.semantic_type ?? column.dtype ?? 'Unknown'}
                        </span>
                      </header>
                      <dl className="canvas-quick-profile__stats">
                        <div>
                          <dt>Missing</dt>
                          <dd>
                            {`${formatMissingPercentage(column.missing_percentage)} (${column.missing_count.toLocaleString()})`}
                          </dd>
                        </div>
                        <div>
                          <dt>Distinct</dt>
                          <dd>{column.distinct_count !== null ? column.distinct_count.toLocaleString() : '—'}</dd>
                        </div>
                      </dl>
                      {hasNumericSummary ? (
                        <div className="canvas-quick-profile__numeric">
                          <h5>Distribution</h5>
                          <div className="canvas-quick-profile__numeric-grid">
                            <span>
                              <strong>Mean</strong> {formatNumericStat(numericSummary?.mean)}
                            </span>
                            <span>
                              <strong>Std dev</strong> {formatNumericStat(numericSummary?.std)}
                            </span>
                            <span>
                              <strong>Min</strong> {formatNumericStat(numericSummary?.minimum)}
                            </span>
                            <span>
                              <strong>Max</strong> {formatNumericStat(numericSummary?.maximum)}
                            </span>
                            <span>
                              <strong>P25</strong> {formatNumericStat(numericSummary?.percentile_25)}
                            </span>
                            <span>
                              <strong>Median</strong> {formatNumericStat(numericSummary?.percentile_50)}
                            </span>
                            <span>
                              <strong>P75</strong> {formatNumericStat(numericSummary?.percentile_75)}
                            </span>
                          </div>
                        </div>
                      ) : null}
                      {topValues.length > 0 ? (
                        <div className="canvas-quick-profile__top-values">
                          <h5>Top values</h5>
                          <ul>
                            {topValues.map((entry, index) => (
                              <li key={`${column.name}-top-${index}`}>
                                <span className="canvas-quick-profile__value">{formatCellValue(entry.value)}</span>
                                <span className="canvas-quick-profile__count">
                                  {entry.count.toLocaleString()} ({entry.percentage.toFixed(1)}%)
                                </span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      {sampleValues.length > 0 ? (
                        <div className="canvas-quick-profile__samples">
                          <h5>Sample values</h5>
                          <ul>
                            {sampleValues.slice(0, 5).map((value, index) => (
                              <li key={`${column.name}-sample-${index}`}>{formatCellValue(value)}</li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            </div>
          )}
          {profilingActiveTab === 'correlations' && (
            <div
              className="canvas-profiling__panel"
              id={profilingCorrelationsPanelId}
              role="tabpanel"
              aria-labelledby={profilingCorrelationsTabId}
            >
              <div className="canvas-quick-profile__correlations">
                <h4>Top correlations</h4>
                {profileData.correlations && profileData.correlations.length > 0 ? (
                  <table className="canvas-quick-profile__correlation-table">
                    <thead>
                      <tr>
                        <th scope="col">Column A</th>
                        <th scope="col">Column B</th>
                        <th scope="col">Coefficient</th>
                      </tr>
                    </thead>
                    <tbody>
                      {profileData.correlations.map((entry, index) => {
                        const magnitude = Math.abs(entry.coefficient);
                        let strength: 'low' | 'moderate' | 'strong' | 'extreme' = 'low';
                        if (magnitude >= 0.85) {
                          strength = 'extreme';
                        } else if (magnitude >= 0.7) {
                          strength = 'strong';
                        } else if (magnitude >= 0.55) {
                          strength = 'moderate';
                        }
                        const sign = entry.coefficient >= 0 ? 'positive' : 'negative';
                        return (
                          <tr key={`profile-correlation-${index}`}>
                            <td>{entry.column_a}</td>
                            <td>{entry.column_b}</td>
                            <td
                              className="canvas-quick-profile__correlation-coefficient"
                              data-strength={strength}
                              data-sign={sign}
                            >
                              {entry.coefficient.toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <p className="canvas-quick-profile__correlations-empty">
                    Not enough numeric columns to compute correlations yet.
                  </p>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
};
