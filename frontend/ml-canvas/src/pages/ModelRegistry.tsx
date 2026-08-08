import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Archive, Box, CheckCircle, ChevronRight, Play, Folder, FileText, Cloud, HardDrive } from 'lucide-react';
import { LoadingState, ErrorState, EmptyState, ModalShell, RecordLink, useConfirm } from '../components/shared';
import { toast } from '../core/toast';
import { parseOperationalContext } from '../core/utils/operationalContext';
import {
  useRegistryStats,
  useRegistryModels,
  useArtifacts,
  useDeployModel,
  type ModelRegistryEntry,
  type ModelVersion,
} from '../core/hooks/useModelRegistry';

/** A dataset/model_type identity the backend could not resolve (see OPS-002 evidence). */
const isUnknownIdentity = (value: string | null | undefined): boolean =>
  !value || value.toLowerCase() === 'unknown';

/** Bound on how many extra pages Registry deep-link resolution will fetch before giving up. */
const MAX_DEEP_LINK_FETCH_PAGES = 20;

export const ModelRegistry: React.FC = () => {
  const confirm = useConfirm();
  const [searchParams] = useSearchParams();
  // Server state -> React Query. Cache invalidation lives in the hook module.
  const statsQuery = useRegistryStats();
  const modelsQuery = useRegistryModels();
  const deployMutation = useDeployModel();

  const stats = statsQuery.data ?? null;
  const models: ModelRegistryEntry[] = useMemo(
    () => modelsQuery.data?.pages.flat() ?? [],
    [modelsQuery.data],
  );
  const loading = modelsQuery.isFetching;
  const error = modelsQuery.error ? (modelsQuery.error as Error).message : null;
  const hasMore = modelsQuery.hasNextPage ?? false;

  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);
  const selectedModel = useMemo(
    () =>
      selectedModelKey
        ? models.find((m) => `${m.model_type}-${m.dataset_id}` === selectedModelKey) ?? null
        : null,
    [selectedModelKey, models],
  );
  const deployingId = deployMutation.isPending ? deployMutation.variables ?? null : null;

  // Deep link support: a `modelVersion` context (e.g. followed from Deployments
  // or Jobs) opens the matching version's dialog directly on load/refresh
  // rather than dropping the operator back at the bare list.
  const versionContext = useMemo(() => parseOperationalContext(searchParams), [searchParams]);
  const deepLinkJobId = versionContext?.ref.kind === 'modelVersion' ? versionContext.ref.jobId : null;
  const deepLinkAttemptedRef = useRef<string | null>(null);
  const [deepLinkNotFound, setDeepLinkNotFound] = useState(false);

  useEffect(() => {
    if (!deepLinkJobId || deepLinkAttemptedRef.current === deepLinkJobId) return;
    const match = models.find((m) => m.versions.some((v) => v.job_id === deepLinkJobId));
    if (match) {
      deepLinkAttemptedRef.current = deepLinkJobId;
      setSelectedModelKey(`${match.model_type}-${match.dataset_id}`);
      setDeepLinkNotFound(false);
      return;
    }
    if (hasMore && !modelsQuery.isFetchingNextPage) {
      const pagesFetched = modelsQuery.data?.pages.length ?? 0;
      if (pagesFetched < MAX_DEEP_LINK_FETCH_PAGES) {
        void modelsQuery.fetchNextPage();
        return;
      }
    }
    if (!hasMore) {
      deepLinkAttemptedRef.current = deepLinkJobId;
      setDeepLinkNotFound(true);
    }
  }, [deepLinkJobId, models, hasMore, modelsQuery]);

  // Artifacts: fetched lazily once a row is expanded.
  const [viewingArtifacts, setViewingArtifacts] = useState<string | null>(null);
  const artifactsQuery = useArtifacts(viewingArtifacts);
  const artifacts = artifactsQuery.data ?? null;
  const loadingArtifacts = artifactsQuery.isFetching;

  // Filters (client-side; the API endpoint does not accept them yet).
  const [datasetFilter, setDatasetFilter] = useState('');
  const [modelTypeFilter, setModelTypeFilter] = useState('');

  // Scoped to the version whose deploy is in flight/failed, so a failure on
  // one version never blocks or misattributes to another row's action.
  const [deployError, setDeployError] = useState<{ jobId: string; version: number | string; message: string } | null>(null);

  const handleViewArtifacts = (jobId: string) => {
    setViewingArtifacts(jobId);
  };

  // Infinite scroll sentinel -> trigger React Query's `fetchNextPage` when it comes into view.
  const loadMoreSentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sentinel = loadMoreSentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting && !modelsQuery.isFetchingNextPage && hasMore) {
        void modelsQuery.fetchNextPage();
      }
    });

    observer.observe(sentinel);
    return () => { observer.disconnect(); };
  }, [modelsQuery, hasMore]);

  /** Deploys `version`, naming the exact before/after model in the confirmation and any failure. */
  const requestDeploy = async (version: ModelVersion, activeVersion: ModelVersion | null) => {
    const replaces = activeVersion && activeVersion.job_id !== version.job_id
      ? ` This replaces the currently active version ${activeVersion.version} (job ${activeVersion.job_id}).`
      : '';
    const ok = await confirm({
      title: 'Deploy model version?',
      message: `Deploy version ${version.version} (job ${version.job_id}) of ${version.model_type}?${replaces}`,
      confirmLabel: 'Deploy',
      variant: 'danger',
    });
    if (!ok) return;
    await runDeploy(version);
  };

  /** Runs the deploy mutation without re-confirming, used for both the initial attempt and retry-in-place. */
  const runDeploy = async (version: ModelVersion) => {
    setDeployError(null);
    try {
      await deployMutation.mutateAsync(version.job_id);
      // Cache invalidation in the hook will refetch models+stats; the
      // selected-model panel is derived from `models`, so it updates
      // automatically once the new data arrives.
      toast.success('Model deployed', `Version ${version.version} (job ${version.job_id}) is now active.`);
    } catch (err: unknown) {
      const message = (err as Error).message || 'Deploy failed.';
      setDeployError({ jobId: version.job_id, version: version.version, message });
      toast.error(`Failed to deploy version ${version.version}`, message);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed': return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
      case 'failed': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'running': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
      default: return 'bg-gray-100 text-gray-800 dark:bg-slate-700 dark:text-slate-300';
    }
  };

  const formatMetrics = (metrics: Record<string, unknown>) => {
    // Try to find common metrics
    const score = metrics.score || metrics.accuracy || metrics.f1_score || metrics.rmse || metrics.mse;
    if (score !== undefined) {
      // Format to 4 decimal places if number
      return typeof score === 'number' ? score.toFixed(4) : String(score);
    }
    // Fallback: first key
    const keys = Object.keys(metrics);
    const firstKey = keys[0];
    if (firstKey) {
      const val = metrics[firstKey];
      return `${firstKey}: ${typeof val === 'number' ? val.toFixed(4) : val}`;
    }
    return '-';
  };

  const formatSource = (source: string) => {
    if (source.toLowerCase() === 'tuning') return 'Advanced Training';
    if (source.toLowerCase() === 'training') return 'Standard Training';
    return source.charAt(0).toUpperCase() + source.slice(1);
  };

  const filteredModels = models.filter(model => {
    const matchDataset = (model.dataset_name || '').toLowerCase().includes(datasetFilter.toLowerCase()) ||
                         (model.dataset_id || '').toLowerCase().includes(datasetFilter.toLowerCase());
    const matchModel = model.model_type.toLowerCase().includes(modelTypeFilter.toLowerCase());
    return matchDataset && matchModel;
  });

  if (loading && !stats && models.length === 0) return (
    <div className="p-8">
      <LoadingState message="Loading registry..." />
    </div>
  );

  if (error) return (
    <div className="p-8">
      <ErrorState error={error} onRetry={() => modelsQuery.refetch()} />
    </div>
  );

  return (
    <div className="p-8 min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 transition-colors duration-200">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">Model Registry</h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">Manage versions, track metrics, and deploy your best models.</p>
        </div>
        <button
          onClick={() => {
            void modelsQuery.refetch();
            void statsQuery.refetch();
          }}
          className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors shadow-sm"
        >
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <StatsCard
          title="Total Versions"
          value={stats?.total_versions || 0}
          icon={<Box className="text-blue-500" />}
        />
        <StatsCard
          title="Active Deployments"
          value={stats?.active_deployments || 0}
          icon={<RocketIcon className="text-green-500" />}
          valueColor="text-green-600 dark:text-green-400"
        />
        <StatsCard
          title="Model Types"
          value={models.length}
          icon={<Archive className="text-purple-500" />}
          valueColor="text-purple-600 dark:text-purple-400"
        />
      </div>

      {deepLinkNotFound && (
        <div className="mb-6">
          <ErrorState error="The linked model version could not be found in the registry. It may have been removed, or is on a page not yet loaded." />
        </div>
      )}

      {/* Filters */}
      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <span className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Filter by Model Type</span>
          <input
            type="text"
            placeholder="e.g. RandomForest"
            value={modelTypeFilter}
            onChange={(e) => { setModelTypeFilter(e.target.value); }}
            className="w-full px-4 py-2 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div className="flex-1">
          <span className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Filter by Dataset</span>
          <input
            type="text"
            placeholder="e.g. Iris Dataset"
            value={datasetFilter}
            onChange={(e) => { setDatasetFilter(e.target.value); }}
            className="w-full px-4 py-2 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Models List */}
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center bg-slate-50/50 dark:bg-slate-800/50">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Archive size={20} className="text-slate-400" />
            Registered Models
          </h2>
          <span className="text-sm text-slate-500 dark:text-slate-400">
            Showing {filteredModels.length} of {models.length} models
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700">
            <thead className="bg-slate-50 dark:bg-slate-900/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Model Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Dataset</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Latest Version</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Source</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Created At</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Deployments</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-slate-800 divide-y divide-slate-200 dark:divide-slate-700">
              {filteredModels.length === 0 ? (
                <tr>
                  <td colSpan={8}>
                    <EmptyState
                      icon={<Box size={48} className="text-slate-300 dark:text-slate-600" />}
                      title="No models found matching your filters."
                    />
                  </td>
                </tr>
              ) : (
                filteredModels.map((model) => {
                  const latest = model.latest_version;
                  if (!latest) return null;

                  const rowKey = `${model.model_type}-${model.dataset_id}`;
                  const deployedVersion = model.versions.find((v) => v.is_deployed) ?? null;
                  const datasetLinkable = !isUnknownIdentity(model.dataset_id);

                  return (
                    <tr key={rowKey} className="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-white">
                        {model.model_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        <div className="flex flex-col">
                          <div className="flex items-center gap-2">
                            {datasetLinkable ? (
                              <RecordLink
                                recordRef={{ kind: 'dataset', datasetId: model.dataset_id }}
                                label={model.dataset_name}
                                className="font-medium"
                              />
                            ) : (
                              <span className="font-medium text-slate-700 dark:text-slate-300" title="No target available">
                                {model.dataset_name || 'Unknown dataset'}
                              </span>
                            )}
                            {model.dataset_type && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-600 uppercase">
                                {model.dataset_type}
                              </span>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        <span className="font-mono bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded text-xs">v{latest.version}</span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        {formatSource(latest.source)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        {new Date(latest.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(latest.status)}`}>
                          {latest.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        {deployedVersion ? (
                          <div className="flex items-center gap-2">
                            <span className="inline-flex items-center gap-1 text-green-600 dark:text-green-400 font-medium bg-green-50 dark:bg-green-900/20 px-2 py-1 rounded-full text-xs">
                              <CheckCircle size={12} /> Active
                            </span>
                            {deployedVersion.deployment_id !== undefined ? (
                              <RecordLink
                                recordRef={{ kind: 'deployment', deploymentId: deployedVersion.deployment_id }}
                                label={`v${deployedVersion.version}`}
                                origin="/registry"
                              />
                            ) : (
                              <span className="text-xs text-slate-400 italic" title="No target available">
                                v{deployedVersion.version}
                              </span>
                            )}
                          </div>
                        ) : (
                          <span className="text-slate-400 dark:text-slate-600 text-xs">None</span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={() => { setSelectedModelKey(`${model.model_type}-${model.dataset_id}`); }}
                          className="text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 flex items-center justify-end gap-1 ml-auto"
                        >
                          View Versions <ChevronRight size={16} />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sentinel observed by the IntersectionObserver above to trigger
          `fetchNextPage()` when it scrolls into view. */}
      <div ref={loadMoreSentinelRef} className="h-1" aria-hidden="true" />

      {loading && models.length > 0 && (
        <div className="py-4 flex justify-center text-slate-500 dark:text-slate-400">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-2"></div>
          Loading more models...
        </div>
      )}

      {/* Versions Modal/Drawer */}
      <ModalShell
        isOpen={!!selectedModel}
        onClose={() => { setSelectedModelKey(null); setDeployError(null); }}
        title={selectedModel?.model_type}
        size="4xl"
        footer={
          <div className="flex justify-end">
            <button
              onClick={() => { setSelectedModelKey(null); setDeployError(null); }}
              className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
            >
              Close
            </button>
          </div>
        }
      >
        {selectedModel && (
          <div className="p-6">
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">Version History</p>

            {deployError && (
              <div className="mb-4">
                <ErrorState
                  error={`Failed to deploy version ${deployError.version} (job ${deployError.jobId}): ${deployError.message}`}
                  onRetry={() => {
                    const failedVersion = selectedModel.versions.find((v) => v.job_id === deployError.jobId);
                    return failedVersion ? runDeploy(failedVersion) : undefined;
                  }}
                />
              </div>
            )}

            <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700">
              <thead>
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Version</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Job</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Date</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Metrics</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Status</th>
                  <th className="px-4 py-2 text-right text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {selectedModel.versions.map((version) => {
                  const activeVersion = selectedModel.versions.find((v) => v.is_deployed) ?? null;
                  const isDeploying = deployingId === version.job_id;
                  return (
                    <tr key={version.job_id} className={`hover:bg-slate-50 dark:hover:bg-slate-700/30 ${version.is_deployed ? 'bg-green-50/50 dark:bg-green-900/10' : ''}`}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-white">
                        <div className="flex items-center gap-2">
                          <span className="font-mono">v{version.version}</span>
                          {version.is_deployed && (
                            <span className="text-xs bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 px-1.5 py-0.5 rounded border border-green-200 dark:border-green-800">
                              Deployed
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <RecordLink
                          recordRef={{ kind: 'job', jobId: version.job_id }}
                          label={version.job_id.slice(0, 8)}
                          origin="/registry"
                        />
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        {new Date(version.created_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-600 dark:text-slate-300 font-mono">
                        {formatMetrics(version.metrics)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span className={`px-2 py-0.5 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(version.status)}`}>
                          {version.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => { handleViewArtifacts(version.job_id); }}
                            className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                            title="View Artifacts"
                          >
                            <Folder size={16} />
                          </button>

                          {version.status === 'completed' && !version.is_deployed && (
                            <button
                              onClick={() => { void requestDeploy(version, activeVersion); }}
                              disabled={isDeploying}
                              aria-label={`Deploy version ${version.version} (job ${version.job_id})`}
                              className="text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 disabled:opacity-50 flex items-center gap-1"
                            >
                              {isDeploying ? (
                                <span className="animate-spin h-3 w-3 border-b-2 border-current rounded-full"></span>
                              ) : (
                                <Play size={14} />
                              )}
                              Deploy
                            </button>
                          )}
                          {version.is_deployed && (
                            version.deployment_id !== undefined ? (
                              <RecordLink
                                recordRef={{ kind: 'deployment', deploymentId: version.deployment_id }}
                                label={<span className="inline-flex items-center gap-1"><CheckCircle size={14} /> Active</span>}
                                origin="/registry"
                                className="text-xs"
                              />
                            ) : (
                              <span className="text-green-600 dark:text-green-400 text-xs flex items-center gap-1">
                                <CheckCircle size={14} /> Active
                              </span>
                            )
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </ModalShell>

      {/* Artifacts Modal */}
      <ModalShell
        isOpen={!!viewingArtifacts}
        onClose={() => { setViewingArtifacts(null); }}
        title="Artifacts"
        size="2xl"
        zIndex="z-[60]"
        footer={
          <div className="flex justify-end">
            <button
              onClick={() => { setViewingArtifacts(null); }}
              className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
            >
              Close
            </button>
          </div>
        }
      >
        <div className="p-6">              {loadingArtifacts ? (
                <div className="flex justify-center py-8 text-slate-500 dark:text-slate-400">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-2"></div>
                  Loading artifacts...
                </div>
              ) : !artifacts || artifacts.files.length === 0 ? (
                <div className="text-center py-8 text-slate-500 dark:text-slate-400">
                  No artifacts found.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800">
                    {artifacts.storage_type === 's3' ? (
                      <Cloud size={20} className="text-blue-600 dark:text-blue-400 flex-shrink-0" />
                    ) : (
                      <HardDrive size={20} className="text-slate-600 dark:text-slate-400 flex-shrink-0" />
                    )}
                    <div className="flex flex-col min-w-0">
                      <span className="text-xs font-semibold uppercase tracking-wider text-blue-700 dark:text-blue-300">
                        {artifacts.storage_type === 's3' ? 'S3 Bucket Storage' : 'Local Storage'}
                      </span>
                      <span className="text-xs font-mono text-slate-600 dark:text-slate-400 truncate" title={artifacts.base_uri}>
                        {artifacts.base_uri}
                      </span>
                    </div>
                  </div>

                  <ul className="space-y-2">
                    {artifacts.files.map((artifact, idx) => (
                      <li key={idx} className="flex items-center gap-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-700/30 border border-slate-100 dark:border-slate-700">
                        <FileText size={18} className="text-slate-400" />
                        <span className="text-sm font-mono text-slate-700 dark:text-slate-300 break-all">
                          {artifact}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
        </div>
      </ModalShell>
    </div>
  );
};

const StatsCard = ({ title, value, icon, valueColor = "text-slate-900 dark:text-white" }: { title: string, value: number, icon: React.ReactNode, valueColor?: string }) => (
  <div className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-sm border border-slate-200 dark:border-slate-700 flex items-start justify-between">
    <div>
      <div className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">{title}</div>
      <div className={`mt-2 text-3xl font-bold ${valueColor}`}>{value}</div>
    </div>
    <div className="p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
      {icon}
    </div>
  </div>
);

const RocketIcon = ({ className }: { className?: string }) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/></svg>
);
