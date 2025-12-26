import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Archive, Box, CheckCircle, ChevronRight, X, Play, Folder, FileText, Cloud, HardDrive } from 'lucide-react';

interface ArtifactResponse {
  storage_type: string;
  base_uri: string;
  files: string[];
}

interface ModelVersion {
  job_id: string;
  pipeline_id: string;
  node_id: string;
  model_type: string;
  version: number | string;
  source: string;
  status: string;
  metrics: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
  created_at: string;
  artifact_uri: string;
  is_deployed: boolean;
  deployment_id?: number;
}

interface ModelRegistryEntry {
  model_type: string;
  dataset_id: string;
  dataset_name: string;
  dataset_type?: string;
  latest_version: ModelVersion | null;
  versions: ModelVersion[];
  deployment_count: number;
}

interface RegistryStats {
  total_models: number;
  total_versions: number;
  active_deployments: number;
}

export const ModelRegistry: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<RegistryStats | null>(null);
  const [models, setModels] = useState<ModelRegistryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelRegistryEntry | null>(null);
  const [deployingId, setDeployingId] = useState<string | null>(null);
  
  // Artifacts viewing
  const [viewingArtifacts, setViewingArtifacts] = useState<string | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactResponse | null>(null);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  
  // Pagination
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const LIMIT = 10;
  
  // Filters
  const [datasetFilter, setDatasetFilter] = useState('');
  const [modelTypeFilter, setModelTypeFilter] = useState('');
  
  // Manual deployment tracking (Local Storage)
  const [manualDeployments, setManualDeployments] = useState<Record<string, boolean>>(() => {
    try {
      const saved = localStorage.getItem('skyulf_manual_deployments');
      return saved ? JSON.parse(saved) : {};
    } catch (e) {
      return {};
    }
  });

  const toggleManualDeployment = (key: string) => {
    setManualDeployments(prev => {
      const next = { ...prev, [key]: !prev[key] };
      localStorage.setItem('skyulf_manual_deployments', JSON.stringify(next));
      return next;
    });
  };

  const handleViewArtifacts = async (jobId: string) => {
    setViewingArtifacts(jobId);
    setLoadingArtifacts(true);
    setArtifacts(null);
    try {
      const res = await fetch(`/api/registry/artifacts/${jobId}`);
      if (res.ok) {
        const data: ArtifactResponse = await res.json();
        setArtifacts(data);
      } else {
        const err = await res.json();
        // Fallback for error display
        setArtifacts({ 
          storage_type: 'error', 
          base_uri: '', 
          files: [`Failed to load artifacts: ${err.detail || 'Unknown error'}`] 
        });
      }
    } catch (e) {
      setArtifacts({ 
        storage_type: 'error', 
        base_uri: '', 
        files: ['Error loading artifacts'] 
      });
    } finally {
      setLoadingArtifacts(false);
    }
  };

  const fetchStats = async () => {
    try {
      const statsRes = await fetch('/api/registry/stats');
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
    } catch (e) {
      console.error("Failed to fetch stats", e);
    }
  };

  const fetchModels = async (pageNum: number, reset: boolean = false) => {
    if (loading) return;
    try {
      setLoading(true);
      const skip = pageNum * LIMIT;
      const params = new URLSearchParams({
        skip: skip.toString(),
        limit: LIMIT.toString()
      });
      const modelsRes = await fetch(`/api/registry/models?${params.toString()}`);
      if (!modelsRes.ok) throw new Error('Failed to fetch models');
      const modelsData = await modelsRes.json();
      
      if (modelsData.length < LIMIT) {
        setHasMore(false);
      } else {
        setHasMore(true);
      }

      setModels(prev => reset ? modelsData : [...prev, ...modelsData]);
      return modelsData;
    } catch (err: unknown) {
      setError((err as Error).message);
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    void fetchStats();
    void fetchModels(0, true);
  }, []);

  // Infinite scroll listener
  useEffect(() => {
    const handleScroll = (e: Event) => {
      const target = e.target as HTMLElement;
      if (
        target.scrollTop + target.clientHeight >= target.scrollHeight - 100 &&
        !loading &&
        hasMore
      ) {
        setPage(prev => prev + 1);
      }
    };

    const mainElement = document.querySelector('main');
    if (mainElement) {
      mainElement.addEventListener('scroll', handleScroll);
      return () => { mainElement.removeEventListener('scroll', handleScroll); };
    } else {
      // Fallback to window if main not found (though it should be there)
      window.addEventListener('scroll', handleScroll as unknown as EventListener);
      return () => { window.removeEventListener('scroll', handleScroll as unknown as EventListener); };
    }
  }, [loading, hasMore]);

  // Fetch on page change
  useEffect(() => {
    if (page > 0) {
      void fetchModels(page, false);
    }
  }, [page]);

  const handleDeploy = async (jobId: string) => {
    if (!confirm('Are you sure you want to deploy this model version? This will replace the currently active deployment.')) return;
    
    try {
      setDeployingId(jobId);
      const res = await fetch(`/api/deployment/deploy/${jobId}`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Deployment failed');
      }
      
      // Refresh data to update UI (reset to page 0)
      setPage(0);
      const updatedModels = await fetchModels(0, true);
      
      // Also update selected model if open
      if (selectedModel && updatedModels) {
        const updatedModel = updatedModels.find((m: ModelRegistryEntry) => 
          m.model_type === selectedModel.model_type && m.dataset_id === selectedModel.dataset_id
        );
        if (updatedModel) setSelectedModel(updatedModel);
      }

      // Ask to go to inference page
      if (confirm('Model deployed successfully! Do you want to go to the inference page?')) {
        navigate('/deployments');
      }
    } catch (err: unknown) {
      alert(`Error deploying model: ${(err as Error).message}`);
    } finally {
      setDeployingId(null);
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
    // metrics is typed as Record<string, unknown>, so it's always truthy if it exists on the type.
    // Assuming strict null checks, if metrics is not optional, this check is redundant.
    // However, if it can be null/undefined, the check is valid.
    // Based on the error "Unnecessary optional chain on a non-nullish value" elsewhere, maybe metrics is guaranteed.
    // But here it's "Unnecessary conditional, value is always falsy" which is weird if metrics is an object.
    // Wait, the error was:
    // frontend/ml-canvas/src/pages/ModelRegistry.tsx:194: if (!metrics) return '-';
    // Unnecessary conditional, value is always falsy.
    // This means metrics is NEVER falsy, i.e. it's always an object.
    
    // Try to find common metrics
    const score = metrics.score || metrics.accuracy || metrics.f1_score || metrics.rmse || metrics.mse;
    if (score !== undefined) {
      // Format to 4 decimal places if number
      return typeof score === 'number' ? score.toFixed(4) : String(score);
    }
    // Fallback: first key
    const keys = Object.keys(metrics);
    if (keys.length > 0) {
      const val = metrics[keys[0]];
      return `${keys[0]}: ${typeof val === 'number' ? val.toFixed(4) : val}`;
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
    <div className="p-8 flex justify-center items-center h-full text-slate-500 dark:text-slate-400">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3"></div>
      Loading registry...
    </div>
  );
  
  if (error) return <div className="p-8 text-center text-red-600 dark:text-red-400">Error: {error}</div>;

  return (
    <div className="p-8 min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 transition-colors duration-200">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">Model Registry</h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">Manage versions, track metrics, and deploy your best models.</p>
        </div>
        <button 
          onClick={() => { setPage(0); void fetchModels(0, true); fetchStats(); }} 
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

      {/* Filters */}
      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Filter by Model Type</label>
          <input 
            type="text" 
            placeholder="e.g. RandomForest" 
            value={modelTypeFilter}
            onChange={(e) => { setModelTypeFilter(e.target.value); }}
            className="w-full px-4 py-2 rounded-md border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Filter by Dataset</label>
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
                  <td colSpan={8} className="px-6 py-12 text-center text-slate-500 dark:text-slate-400">
                    <div className="flex flex-col items-center justify-center">
                      <Box size={48} className="mb-4 opacity-20" />
                      <p>No models found matching your filters.</p>
                    </div>
                  </td>
                </tr>
              ) : (
                filteredModels.map((model) => {
                  const latest = model.latest_version;
                  if (!latest) return null;
                  
                  const rowKey = `${model.model_type}-${model.dataset_id}`;
                  const isSystemDeployed = model.deployment_count > 0;
                  const isManuallyDeployed = manualDeployments[rowKey] || false;
                  const isDeployed = isSystemDeployed || isManuallyDeployed;

                  return (
                    <tr key={rowKey} className="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-white">
                        {model.model_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        <div className="flex flex-col">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-slate-700 dark:text-slate-300">{model.dataset_name}</span>
                            {model.dataset_type && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400 border border-slate-200 dark:border-slate-600 uppercase">
                                {model.dataset_type}
                              </span>
                            )}
                          </div>
                          <span className="text-xs text-slate-400 font-mono mt-0.5">{model.dataset_id}</span>
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
                        <div className="flex items-center gap-2" onClick={(e) => { e.stopPropagation(); }}>
                          <input 
                            type="checkbox" 
                            checked={isDeployed} 
                            onChange={() => { toggleManualDeployment(rowKey); }}
                            disabled={isSystemDeployed}
                            className={`w-4 h-4 rounded border-gray-300 focus:ring-green-500 ${isSystemDeployed ? 'text-green-600 opacity-50 cursor-not-allowed' : 'text-blue-600 cursor-pointer'}`} 
                          />
                          {isDeployed ? (
                            <span className="inline-flex items-center gap-1 text-green-600 dark:text-green-400 font-medium bg-green-50 dark:bg-green-900/20 px-2 py-1 rounded-full text-xs">
                              <CheckCircle size={12} /> {isSystemDeployed ? 'Active' : 'Manual'}
                            </span>
                          ) : (
                            <span className="text-slate-400 dark:text-slate-600 text-xs">None</span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button 
                          onClick={() => { setSelectedModel(model); }}
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
      
      {loading && models.length > 0 && (
        <div className="py-4 flex justify-center text-slate-500 dark:text-slate-400">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-2"></div>
          Loading more models...
        </div>
      )}

      {/* Versions Modal/Drawer */}
      {selectedModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm" onClick={() => { setSelectedModel(null); }}>
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden border border-slate-200 dark:border-slate-700" onClick={e => { e.stopPropagation(); }}>
            <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center bg-slate-50 dark:bg-slate-900/50">
              <div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-white">{selectedModel.model_type}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">Version History</p>
              </div>
              <button onClick={() => { setSelectedModel(null); }} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors">
                <X size={24} />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700">
                <thead>
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Version</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Date</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Metrics</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Status</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-slate-500 dark:text-slate-400 uppercase">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                  {selectedModel.versions.map((version) => (
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
                            onClick={() => { void handleViewArtifacts(version.job_id); }}
                            className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                            title="View Artifacts"
                          >
                            <Folder size={16} />
                          </button>
                          
                          {version.status === 'completed' && !version.is_deployed && (
                            <button 
                              onClick={() => { void handleDeploy(version.job_id); }}
                              disabled={deployingId === version.job_id}
                              className="text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 disabled:opacity-50 flex items-center gap-1"
                            >
                              {deployingId === version.job_id ? (
                                <span className="animate-spin h-3 w-3 border-b-2 border-current rounded-full"></span>
                              ) : (
                                <Play size={14} />
                              )}
                              Deploy
                            </button>
                          )}
                          {version.is_deployed && (
                            <span className="text-green-600 dark:text-green-400 text-xs flex items-center gap-1">
                              <CheckCircle size={14} /> Active
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="px-6 py-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50 flex justify-end">
              <button 
                onClick={() => { setSelectedModel(null); }}
                className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Artifacts Modal */}
      {viewingArtifacts && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm" onClick={() => { setViewingArtifacts(null); }}>
          <div className="bg-white dark:bg-slate-800 rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden border border-slate-200 dark:border-slate-700" onClick={e => { e.stopPropagation(); }}>
            <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center bg-slate-50 dark:bg-slate-900/50">
              <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <Folder size={20} className="text-blue-500" />
                Artifacts
              </h3>
              <button onClick={() => { setViewingArtifacts(null); }} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors">
                <X size={24} />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              {loadingArtifacts ? (
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
            
            <div className="px-6 py-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/50 flex justify-end">
              <button 
                onClick={() => { setViewingArtifacts(null); }}
                className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
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
