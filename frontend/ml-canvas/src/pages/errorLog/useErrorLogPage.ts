import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { monitoringApi, type ErrorEvent, type ErrorSeverity, type GroupedIssue, type PipelineRunLog } from '../../core/api/monitoring';
import { useConfirm } from '../../components/shared';
import type { OperationalTimeRange } from '../../core/utils/operationalContext';
import { toast } from '../../core/toast';
import { sinceIso, SEVERITY_TO_PIPELINE_LEVEL, type TimeRange } from './errorLogFormatting';
import { groupPipelineIssues, mergeTimeline } from './errorLogAggregation';

/** Keep the latest HTTP snapshot and its nonblocking pipeline follow-up authoritative. */
export function useErrorLogPage() {
  const loadGeneration = useRef(0);
  const [events, setEvents] = useState<ErrorEvent[]>([]);
  const [eventsTotal, setEventsTotal] = useState(0);
  const [eventsTotalUnfiltered, setEventsTotalUnfiltered] = useState(0);
  const [errorFacets, setErrorFacets] = useState<{ severities: ErrorSeverity[]; error_types: string[]; job_ids: string[] }>(
    { severities: [], error_types: [], job_ids: [] },
  );
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [showResolved, setShowResolved] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<'' | ErrorSeverity>('');
  const [errorTypeFilter, setErrorTypeFilter] = useState('');
  const [jobIdFilter, setJobIdFilter] = useState('');
  const [nodeIdFilter, setNodeIdFilter] = useState('');
  const [modal, setModal] = useState<ErrorEvent | null>(null);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<{ hour: string; count: number }[]>([]);
  const [view, setView] = useState<'events' | 'issues'>('events');
  const [grouped, setGrouped] = useState<GroupedIssue[]>([]);

  // --- Backend-persisted pipeline run log ---
  const [pipelineLogs, setPipelineLogs] = useState<PipelineRunLog[]>([]);
  const [pipelineFacets, setPipelineFacets] = useState<{ node_types: string[]; node_ids: string[] }>(
    { node_types: [], node_ids: [] },
  );

  // Operational context this view hands to every contextual RecordLink, so a
  // followed link and its return preserve the active time/facet scope.
  const operationalTimeRange = (timeRange === 'all' ? 'all' : timeRange) as OperationalTimeRange;
  const linkFilters = useMemo(() => {
    const f: Record<string, string> = {};
    if (showResolved) f.showResolved = 'true';
    if (severityFilter) f.severity = severityFilter;
    if (errorTypeFilter) f.errorType = errorTypeFilter;
    if (jobIdFilter) f.jobId = jobIdFilter;
    if (nodeIdFilter) f.nodeId = nodeIdFilter;
    if (search) f.q = search;
    return f;
  }, [showResolved, severityFilter, errorTypeFilter, jobIdFilter, nodeIdFilter, search]);

  const fetchPipelineLogs = useCallback(async (generation: number) => {
    try {
      const resp = await monitoringApi.getPipelineLogs(200, sinceIso(timeRange), undefined, {
        ...(search ? { q: search } : {}),
        ...(severityFilter ? { level: SEVERITY_TO_PIPELINE_LEVEL[severityFilter] } : {}),
        ...(nodeIdFilter ? { nodeId: nodeIdFilter } : {}),
      });
      if (generation !== loadGeneration.current) return;
      setPipelineLogs(resp.entries);
      setPipelineFacets({ node_types: resp.facets.node_types, node_ids: resp.facets.node_ids });
    } catch (err) {
      if (generation !== loadGeneration.current) return;
      // backend may be unavailable; log for diagnostics but don't block the page
      console.debug('[error-log] failed to fetch pipeline logs', err);
    }
  }, [timeRange, search, severityFilter, nodeIdFilter]);

  const handleClearPipelineLogs = useCallback(async () => {
    try {
      await monitoringApi.clearPipelineLogs();
      setPipelineLogs([]);
    } catch (err) {
      console.error('Failed to clear pipeline logs', err);
      toast.error('Failed to clear pipeline logs', 'Please try again.');
    }
  }, []);

  // Keep the HTTP snapshot request separate from the lifetime of its state writes.
  const fetchErrorSnapshot = useCallback(() => Promise.all([
    monitoringApi.getErrors(500, sinceIso(timeRange), showResolved, {
      ...(search ? { q: search } : {}),
      ...(severityFilter ? { severity: severityFilter } : {}),
      ...(errorTypeFilter ? { errorType: errorTypeFilter } : {}),
      ...(jobIdFilter ? { jobId: jobIdFilter } : {}),
    }),
    monitoringApi.getTimeline(24),
    monitoringApi.getGrouped(),
  ]), [timeRange, showResolved, search, severityFilter, errorTypeFilter, jobIdFilter]);

  const load = useCallback(async () => {
    const generation = loadGeneration.current + 1;
    loadGeneration.current = generation;
    setLoading(true);
    setError(null);
    try {
      const [data, tl, grp] = await fetchErrorSnapshot();
      if (generation !== loadGeneration.current) return;
      setEvents(data.entries);
      setEventsTotal(data.total);
      setEventsTotalUnfiltered(data.total_unfiltered);
      setErrorFacets(data.facets);
      setTimeline(tl);
      setGrouped(grp);
      // also refresh pipeline logs so they appear in Events tab
      void fetchPipelineLogs(generation);
    } catch {
      if (generation !== loadGeneration.current) return;
      setError('Could not reach the backend. Is the server running?');
    } finally {
      if (generation === loadGeneration.current) setLoading(false);
    }
  }, [fetchErrorSnapshot, fetchPipelineLogs]);

  useEffect(() => {
    void load();
    // Invalidate HTTP and detached pipeline continuations on filter changes/unmount.
    return () => { loadGeneration.current++; };
  }, [load]);

  const confirm = useConfirm();

  const handleClear = async () => {
    const ok = await confirm({
      title: 'Delete all error events?',
      message: `Delete all ${events.length} error events? This cannot be undone.`,
      confirmLabel: 'Delete all',
      variant: 'danger',
    });
    if (!ok) return;
    setClearing(true);
    try {
      await monitoringApi.clearErrors();
      setEvents([]);
    } finally {
      setClearing(false);
    }
  };

  const handleViewSample = async (id: number) => {
    try {
      const ev = await monitoringApi.getError(id);
      setModal(ev);
    } catch (err) {
      console.error('Failed to load error sample', err);
      toast.error('Failed to load error sample', 'Please try again.');
    }
  };

  const handleResolve = async (ev: ErrorEvent) => {
    const updated = ev.resolved_at
      ? await monitoringApi.unresolveError(ev.id)
      : await monitoringApi.resolveError(ev.id);
    setEvents(prev => prev.map(e => e.id === updated.id ? updated : e));
  };

  // Filters (time range, resolved state, severity, error type, job/node id, and
  // the generic search box) are all applied server-side across the full stored
  // history — see `monitoringApi.getErrors`/`getPipelineLogs` — so these lists
  // are already the matching set, not a client-side narrowing of one page.
  const hasActiveFilters =
    !!search || !!severityFilter || !!errorTypeFilter || !!jobIdFilter || !!nodeIdFilter;

  const pipelineIssues = useMemo(() => groupPipelineIssues(pipelineLogs), [pipelineLogs]);

  const mergedTimeline = useMemo(() => mergeTimeline(timeline, pipelineLogs), [timeline, pipelineLogs]);

  return {
    events,
    eventsTotal,
    eventsTotalUnfiltered,
    errorFacets,
    loading,
    search,
    setSearch,
    timeRange,
    setTimeRange,
    showResolved,
    setShowResolved,
    severityFilter,
    setSeverityFilter,
    errorTypeFilter,
    setErrorTypeFilter,
    jobIdFilter,
    setJobIdFilter,
    nodeIdFilter,
    setNodeIdFilter,
    modal,
    setModal,
    clearing,
    error,
    view,
    setView,
    grouped,
    pipelineLogs,
    pipelineFacets,
    operationalTimeRange,
    linkFilters,
    load,
    handleClear,
    handleClearPipelineLogs,
    handleViewSample,
    handleResolve,
    hasActiveFilters,
    pipelineIssues,
    mergedTimeline,
  };
}

export type ErrorLogPageState = ReturnType<typeof useErrorLogPage>;
