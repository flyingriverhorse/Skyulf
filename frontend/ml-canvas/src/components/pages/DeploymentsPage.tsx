import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { deploymentApi, DeploymentInfo } from '../../core/api/deployment';
import { Rocket, Power, Clock, CheckCircle, RefreshCw, Box } from 'lucide-react';
import { ErrorState, EmptyState, LoadingState, RecordLink, useConfirm } from '../shared';
import { toast } from '../../core/toast';
import { parseOperationalContext } from '../../core/utils/operationalContext';

/** A dataset/model_type identity the backend could not resolve (see OPS-002 evidence). */
const isUnknownIdentity = (value: string | null | undefined): boolean =>
  !value || value.toLowerCase() === 'unknown';

/** Names the exact model version a deployment record identifies, for confirmations and toasts. */
const describeDeployment = (deployment: DeploymentInfo): string =>
  deployment.version !== undefined && deployment.version !== null
    ? `${deployment.model_type} v${deployment.version} (job ${deployment.job_id})`
    : `${deployment.model_type} (job ${deployment.job_id})`;

/** Renders the model-version/dataset lineage for one deployment record as RecordLinks, or an
 * explicit "no target available" note when the backing job/dataset could not be resolved. */
const DeploymentLineage: React.FC<{ deployment: DeploymentInfo }> = ({ deployment }) => {
  const hasVersion = deployment.version !== undefined && deployment.version !== null;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      <RecordLink
        recordRef={{ kind: 'job', jobId: deployment.job_id }}
        label={`Job ${deployment.job_id.slice(0, 8)}`}
        origin="/deployments"
      />
      {hasVersion ? (
        <RecordLink
          recordRef={{ kind: 'modelVersion', jobId: deployment.job_id, version: String(deployment.version) }}
          label={`Version ${deployment.version}`}
          origin="/deployments"
        />
      ) : (
        <span className="text-slate-400 italic" title="No target available">Version unavailable</span>
      )}
      {!isUnknownIdentity(deployment.dataset_id) ? (
        <RecordLink
          recordRef={{ kind: 'dataset', datasetId: deployment.dataset_id as string }}
          label="Dataset"
          origin="/deployments"
        />
      ) : (
        <span className="text-slate-400 italic" title="No target available">No dataset available</span>
      )}
      {deployment.previous_deployment_id !== undefined && deployment.previous_deployment_id !== null ? (
        <RecordLink
          recordRef={{ kind: 'deployment', deploymentId: deployment.previous_deployment_id }}
          label={`Replaced deployment #${deployment.previous_deployment_id}`}
          origin="/deployments"
        />
      ) : (
        <span className="text-slate-400 italic">No prior deployment</span>
      )}
    </div>
  );
};

export const DeploymentsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const [activeDeployment, setActiveDeployment] = useState<DeploymentInfo | null>(null);
  const [history, setHistory] = useState<DeploymentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDeactivating, setIsDeactivating] = useState(false);
  const [redeployingJobId, setRedeployingJobId] = useState<string | null>(null);
  // Scoped to the action that failed, so retrying one doesn't clear or block another.
  const [deactivateError, setDeactivateError] = useState<string | null>(null);
  const [redeployError, setRedeployError] = useState<{ jobId: string; message: string } | null>(null);
  const confirm = useConfirm();

  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [active, hist] = await Promise.all([
        deploymentApi.getActive(),
        deploymentApi.getHistory(50, 0)
      ]);
      setActiveDeployment(active);
      setHistory(hist);
    } catch (err: unknown) {
      console.error("Failed to load deployments", err);
      setError("Failed to load deployment data.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  // Deep link support: a `deployment` or `modelVersion` context (e.g. followed
  // from Registry or Jobs) highlights the matching record on load/refresh
  // instead of leaving the operator to re-find it in the history table.
  const deepLinkContext = useMemo(() => parseOperationalContext(searchParams), [searchParams]);
  const highlightedDeploymentId = deepLinkContext?.ref.kind === 'deployment' ? deepLinkContext.ref.deploymentId : null;
  const highlightedJobId = deepLinkContext?.ref.kind === 'modelVersion' ? deepLinkContext.ref.jobId : null;
  const highlightRowRef = useRef<HTMLTableRowElement | HTMLDivElement | null>(null);
  const isHighlighted = useCallback(
    (deployment: DeploymentInfo): boolean =>
      (highlightedDeploymentId !== null && deployment.id === highlightedDeploymentId) ||
      (highlightedJobId !== null && deployment.job_id === highlightedJobId),
    [highlightedDeploymentId, highlightedJobId],
  );
  useEffect(() => {
    if ((highlightedDeploymentId === null && highlightedJobId === null) || isLoading) return;
    highlightRowRef.current?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, [highlightedDeploymentId, highlightedJobId, isLoading, history, activeDeployment]);

  const handleDeactivate = async () => {
    if (!activeDeployment) return;
    const ok = await confirm({
      title: 'Deactivate deployment?',
      message: `Deactivate ${describeDeployment(activeDeployment)}? It will no longer serve predictions.`,
      confirmLabel: 'Deactivate',
      variant: 'danger',
    });
    if (!ok) return;
    await runDeactivate(activeDeployment);
  };

  /** Deactivates without re-confirming, used for both the initial attempt and retry-in-place. */
  const runDeactivate = async (deployment: DeploymentInfo) => {
    setDeactivateError(null);
    setIsDeactivating(true);
    try {
      await deploymentApi.deactivate();
      await loadData();
      toast.success('Deployment deactivated', describeDeployment(deployment));
    } catch (e) {
      console.error("Failed to deactivate", e);
      const message = (e as Error).message || 'Failed to deactivate deployment.';
      setDeactivateError(message);
      toast.error(`Failed to deactivate ${describeDeployment(deployment)}`, message);
    } finally {
      setIsDeactivating(false);
    }
  };

  const handleRedeploy = async (deployment: DeploymentInfo) => {
    const ok = await confirm({
      title: 'Redeploy model?',
      message: `Redeploy ${describeDeployment(deployment)}? This will replace the currently active deployment${activeDeployment ? ` (${describeDeployment(activeDeployment)})` : ''}.`,
      confirmLabel: 'Redeploy',
    });
    if (!ok) return;
    await runRedeploy(deployment);
  };

  /** Redeploys without re-confirming, used for both the initial attempt and retry-in-place. */
  const runRedeploy = async (deployment: DeploymentInfo) => {
    setRedeployError(null);
    setRedeployingJobId(deployment.job_id);
    try {
      await deploymentApi.deployModel(deployment.job_id);
      await loadData();
      toast.success('Model redeployed', describeDeployment(deployment));
    } catch (e) {
      console.error("Failed to redeploy", e);
      const message = (e as Error).message || 'Failed to redeploy model.';
      setRedeployError({ jobId: deployment.job_id, message });
      toast.error(`Failed to redeploy ${describeDeployment(deployment)}`, message);
    } finally {
      setRedeployingJobId(null);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 overflow-hidden">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-6 flex justify-between items-center shrink-0">
        <div>
          <h1 className="text-2xl font-semibold text-gray-800 dark:text-gray-100">Model Deployments</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Manage active models and view deployment history</p>
        </div>
        <button
          onClick={() => { void loadData(); }}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 transition-colors"
          title="Refresh"
          aria-label="Refresh deployments"
        >
          <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {error && (
          <ErrorState error={error} onRetry={() => loadData()} />
        )}

        {/* Active Deployment Section */}
        <section>
          <h2 className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
            <Rocket className="w-5 h-5 text-green-500" />
            Active Deployment
          </h2>

          {activeDeployment ? (
            <div
              ref={isHighlighted(activeDeployment) ? (highlightRowRef as React.RefObject<HTMLDivElement>) : null}
              className={`bg-white dark:bg-gray-800 rounded-xl border shadow-sm overflow-hidden ${isHighlighted(activeDeployment) ? 'border-blue-400 dark:border-blue-500 ring-2 ring-blue-200 dark:ring-blue-900' : 'border-green-200 dark:border-green-900'}`}
            >
              <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-start">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
                    <Box className="w-8 h-8 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                        {activeDeployment.model_type}
                      </h3>
                      <span className="px-2 py-0.5 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 text-xs font-medium rounded-full flex items-center gap-1">
                        <CheckCircle className="w-3 h-3" /> Active
                      </span>
                    </div>
                    <div className="mt-1">
                      <DeploymentLineage deployment={activeDeployment} />
                    </div>
                    <p className="text-xs text-gray-400 mt-2">
                      Deployed: {new Date(activeDeployment.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => { void handleDeactivate(); }}
                  disabled={isDeactivating}
                  className="flex items-center gap-2 px-4 py-2 bg-red-50 hover:bg-red-100 text-red-600 dark:bg-red-900/20 dark:hover:bg-red-900/30 dark:text-red-400 rounded-lg transition-colors text-sm font-medium disabled:opacity-50"
                >
                  {isDeactivating ? (
                    <div className="w-4 h-4 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Power className="w-4 h-4" />
                  )}
                  Deactivate
                </button>
              </div>
              {deactivateError && (
                <div className="border-b border-gray-100 dark:border-gray-700">
                  <ErrorState
                    error={`Failed to deactivate ${describeDeployment(activeDeployment)}: ${deactivateError}`}
                    onRetry={() => runDeactivate(activeDeployment)}
                  />
                </div>
              )}
              <div className="bg-gray-50 dark:bg-gray-900/50 p-4 text-xs font-mono text-gray-500 dark:text-gray-400 break-all">
                Artifact URI: {activeDeployment.artifact_uri}
              </div>
            </div>
          ) : isLoading ? (
            <LoadingState message="Loading active deployment..." />
          ) : (
            <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-300 dark:border-gray-700 p-8 text-center">
              <div className="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
                <Power className="w-6 h-6 text-gray-400" />
              </div>
              <h3 className="text-gray-900 dark:text-gray-100 font-medium">No Active Model</h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Deploy a model from the Experiments page to see it here.
              </p>
            </div>
          )}
        </section>

        {/* History Section */}
        <section>
          <h2 className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-blue-500" />
            Deployment History
          </h2>

          {redeployError && (
            <div className="mb-4">
              <ErrorState
                error={`Failed to redeploy job ${redeployError.jobId}: ${redeployError.message}`}
                onRetry={() => {
                  const target = history.find((d) => d.job_id === redeployError.jobId);
                  return target ? runRedeploy(target) : undefined;
                }}
              />
            </div>
          )}

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-gray-50 dark:bg-gray-900/50 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="px-6 py-3">Status</th>
                    <th className="px-6 py-3">Model Type</th>
                    <th className="px-6 py-3">Lineage</th>
                    <th className="px-6 py-3">Deployed At</th>
                    <th className="px-6 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                  {history.length === 0 ? (
                    <tr>
                      <td colSpan={5}>
                        {isLoading ? (
                          <LoadingState message="Loading deployment history..." />
                        ) : (
                          <EmptyState title="No deployment history found." />
                        )}
                      </td>
                    </tr>
                  ) : (
                    history.map((deployment) => (
                      <tr
                        key={deployment.id}
                        ref={isHighlighted(deployment) ? (highlightRowRef as React.RefObject<HTMLTableRowElement>) : null}
                        className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors ${isHighlighted(deployment) ? 'bg-blue-50 dark:bg-blue-900/20 ring-1 ring-inset ring-blue-300 dark:ring-blue-700' : ''}`}
                      >
                        <td className="px-6 py-4">
                          {deployment.is_active ? (
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                              Active
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-gray-400"></span>
                              Inactive
                            </span>
                          )}
                        </td>
                        <td className="px-6 py-4 font-medium text-gray-900 dark:text-gray-100">
                          {deployment.model_type}
                        </td>
                        <td className="px-6 py-4">
                          <DeploymentLineage deployment={deployment} />
                        </td>
                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                          {new Date(deployment.created_at).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-right">
                          {!deployment.is_active && (
                            <button
                              onClick={() => { void handleRedeploy(deployment); }}
                              disabled={redeployingJobId === deployment.job_id}
                              aria-label={`Redeploy ${describeDeployment(deployment)}`}
                              className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium text-xs disabled:opacity-50"
                            >
                              {redeployingJobId === deployment.job_id ? 'Redeploying...' : 'Redeploy'}
                            </button>
                          )}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
