import React, { useEffect, useState } from 'react';
import { deploymentApi, DeploymentInfo } from '../../core/api/deployment';
import { Rocket, Power, Clock, CheckCircle, AlertCircle, RefreshCw, Box } from 'lucide-react';

export const DeploymentsPage: React.FC = () => {
  const [activeDeployment, setActiveDeployment] = useState<DeploymentInfo | null>(null);
  const [history, setHistory] = useState<DeploymentInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [active, hist] = await Promise.all([
        deploymentApi.getActive(),
        deploymentApi.getHistory(50, 0)
      ]);
      setActiveDeployment(active);
      setHistory(hist);
    } catch (err: any) {
      console.error("Failed to load deployments", err);
      setError("Failed to load deployment data.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadData();
  }, []);

  const handleDeactivate = async () => {
    if (!confirm("Are you sure you want to deactivate the current deployment?")) return;
    try {
      await deploymentApi.deactivate();
      await loadData();
    } catch (e) {
      console.error("Failed to deactivate", e);
      alert("Failed to deactivate deployment.");
    }
  };

  const handleRedeploy = async (jobId: string) => {
    if (!confirm(`Are you sure you want to redeploy job ${jobId}?`)) return;
    try {
      await deploymentApi.deployModel(jobId);
      await loadData();
    } catch (e) {
      console.error("Failed to redeploy", e);
      alert("Failed to redeploy model.");
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
          onClick={loadData}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-8">
        {error && (
          <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg border border-red-200 dark:border-red-800 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {/* Active Deployment Section */}
        <section>
          <h2 className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
            <Rocket className="w-5 h-5 text-green-500" />
            Active Deployment
          </h2>
          
          {activeDeployment ? (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-green-200 dark:border-green-900 shadow-sm overflow-hidden">
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
                    <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                      Job ID: {activeDeployment.job_id}
                    </p>
                    <p className="text-xs text-gray-400 mt-1">
                      Deployed: {new Date(activeDeployment.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleDeactivate}
                  className="flex items-center gap-2 px-4 py-2 bg-red-50 hover:bg-red-100 text-red-600 dark:bg-red-900/20 dark:hover:bg-red-900/30 dark:text-red-400 rounded-lg transition-colors text-sm font-medium"
                >
                  <Power className="w-4 h-4" />
                  Deactivate
                </button>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900/50 p-4 text-xs font-mono text-gray-500 dark:text-gray-400 break-all">
                Artifact URI: {activeDeployment.artifact_uri}
              </div>
            </div>
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
          
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="bg-gray-50 dark:bg-gray-900/50 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700">
                  <tr>
                    <th className="px-6 py-3">Status</th>
                    <th className="px-6 py-3">Model Type</th>
                    <th className="px-6 py-3">Job ID</th>
                    <th className="px-6 py-3">Deployed At</th>
                    <th className="px-6 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                  {history.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="px-6 py-8 text-center text-gray-500 dark:text-gray-400">
                        No deployment history found.
                      </td>
                    </tr>
                  ) : (
                    history.map((deployment) => (
                      <tr key={deployment.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
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
                        <td className="px-6 py-4 font-mono text-xs text-gray-500 dark:text-gray-400">
                          {deployment.job_id.slice(0, 8)}...
                        </td>
                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                          {new Date(deployment.created_at).toLocaleString()}
                        </td>
                        <td className="px-6 py-4 text-right">
                          {!deployment.is_active && (
                            <button
                              onClick={() => handleRedeploy(deployment.job_id)}
                              className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium text-xs"
                            >
                              Redeploy
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
