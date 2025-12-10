import React, { useState, useEffect } from 'react';
import { deploymentApi, DeploymentInfo } from '../../core/api/deployment';
import { jobsApi } from '../../core/api/jobs';
import { DatasetService } from '../../core/api/datasets';
import { Play, AlertCircle, CheckCircle, Box, Power } from 'lucide-react';

export const InferencePage: React.FC = () => {
  const [activeDeployment, setActiveDeployment] = useState<DeploymentInfo | null>(null);
  const [inputData, setInputData] = useState<string>('[\n  {\n    "feature1": 0.5,\n    "feature2": 1.2\n  }\n]');
  const [predictions, setPredictions] = useState<any[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [autoFilterInfo, setAutoFilterInfo] = useState<string | null>(null);

  useEffect(() => {
    loadActiveDeployment();
  }, []);

  const loadActiveDeployment = async () => {
    try {
      const deployment = await deploymentApi.getActive();
      setActiveDeployment(deployment);

      // Auto-fill input data from dataset sample if available
      if (deployment && deployment.job_id) {
        try {
          const job = await jobsApi.getJob(deployment.job_id);
          if (job.dataset_id) {
            // 1. Use Target and Dropped Columns from Job Info
            const targetColumn = job.target_column;
            const droppedColumns = job.dropped_columns || [];

            // 2. Fetch Sample Data
            const sample = await DatasetService.getSample(job.dataset_id, 1);
            
            if (sample && sample.length > 0) {
              // 3. Filter the sample data
              const filteredSample = sample.map(row => {
                  const newRow = { ...row };
                  
                  // Remove target column
                  if (targetColumn && targetColumn in newRow) {
                      delete newRow[targetColumn];
                  }
                  
                  // Remove dropped columns
                  droppedColumns.forEach(col => {
                      if (col in newRow) delete newRow[col];
                  });
                  
                  return newRow;
              });

              setInputData(JSON.stringify(filteredSample, null, 2));
              
              // Set info message
              const droppedInfo = [];
              if (targetColumn) droppedInfo.push(`Target: ${targetColumn}`);
              if (droppedColumns.length > 0) droppedInfo.push(`Dropped: ${droppedColumns.length} cols`);
              
              if (droppedInfo.length > 0) {
                  setAutoFilterInfo(`Auto-filtered: ${droppedInfo.join(', ')}`);
              }
            }
          }
        } catch (err) {
          console.warn("Failed to fetch dataset sample for inference auto-fill", err);
        }
      }
    } catch (e) {
      setActiveDeployment(null);
    }
  };

  const handleDeactivate = async () => {
    if (!confirm("Are you sure you want to undeploy the current model?")) return;
    try {
      await deploymentApi.deactivate();
      setActiveDeployment(null);
      setPredictions(null);
    } catch (e) {
      console.error("Failed to deactivate", e);
    }
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setPredictions(null);
    try {
      const data = JSON.parse(inputData);
      if (!Array.isArray(data)) {
        throw new Error("Input must be a JSON array of objects");
      }
      const response = await deploymentApi.predict(data);
      setPredictions(response.predictions);
    } catch (e: any) {
      setError(e.message || "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 p-6 overflow-hidden">
      <h1 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 mb-6 shrink-0">Model Inference</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
        {/* Left Column: Input */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col h-full">
          <div className="flex justify-between items-start mb-4">
            <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Input Data (JSON)</h3>
            {autoFilterInfo && (
                <div className="flex flex-col items-end">
                    <span className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-2 py-1 rounded border border-amber-200 dark:border-amber-800">
                        {autoFilterInfo}
                    </span>
                    <span className="text-[10px] text-gray-400 mt-1">Please verify fields before running</span>
                </div>
            )}
          </div>
          <textarea
            className="flex-1 w-full p-4 font-mono text-sm bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none resize-none"
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder='[{"col1": 1, "col2": "A"}]'
          />
          <div className="mt-4 flex justify-end">
            <button
              onClick={handlePredict}
              disabled={!activeDeployment || isLoading}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                !activeDeployment || isLoading
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-800 dark:text-gray-600'
                  : 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'
              }`}
            >
              {isLoading ? 'Running...' : (
                <>
                  <Play className="w-4 h-4" /> Run Prediction
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Deployment & Output */}
        <div className="flex flex-col gap-6 h-full min-h-0">
            {/* Active Deployment Status */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 shrink-0">
                <h2 className="text-lg font-medium text-gray-800 dark:text-gray-100 mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Box className="w-5 h-5 text-blue-500" />
                    Active Deployment
                </div>
                {activeDeployment && (
                    <button
                    onClick={handleDeactivate}
                    className="flex items-center gap-1 text-xs px-2 py-1 bg-red-100 text-red-600 rounded hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50 transition-colors"
                    title="Undeploy Model"
                    >
                    <Power className="w-3 h-3" />
                    Undeploy
                    </button>
                )}
                </h2>
                {activeDeployment ? (
                <div className="grid grid-cols-2 gap-4">
                    <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Model Type</div>
                    <div className="font-medium text-gray-800 dark:text-gray-200">{activeDeployment.model_type}</div>
                    </div>
                    <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Job ID</div>
                    <div className="font-mono text-sm text-gray-800 dark:text-gray-200">{activeDeployment.job_id.slice(0, 8)}</div>
                    </div>
                    <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Deployed At</div>
                    <div className="text-sm text-gray-800 dark:text-gray-200">{new Date(activeDeployment.created_at).toLocaleDateString()}</div>
                    </div>
                    <div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">Status</div>
                    <div className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium text-sm">
                        <CheckCircle className="w-4 h-4" /> Active
                    </div>
                    </div>
                </div>
                ) : (
                <div className="text-gray-500 dark:text-gray-400 italic flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    No model is currently deployed. Go to Experiments page to deploy one.
                </div>
                )}
            </div>

            {/* Output */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col flex-1 min-h-0">
            <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100 mb-4">Prediction Results</h3>
            <div className="flex-1 bg-gray-50 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 p-4 overflow-auto">
                {error ? (
                <div className="text-red-600 dark:text-red-400 flex items-start gap-2">
                    <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                    <pre className="whitespace-pre-wrap font-mono text-sm">{error}</pre>
                </div>
                ) : predictions ? (
                <div className="space-y-2">
                    {predictions.map((pred, i) => (
                    <div key={i} className="flex items-center gap-3 p-2 bg-white dark:bg-gray-800 rounded border border-gray-100 dark:border-gray-700">
                        <span className="text-xs text-gray-500 w-8">#{i + 1}</span>
                        <span className="font-mono font-medium text-blue-600 dark:text-blue-400">
                        {typeof pred === 'object' ? JSON.stringify(pred) : String(pred)}
                        </span>
                    </div>
                    ))}
                </div>
                ) : (
                <div className="h-full flex items-center justify-center text-gray-400 text-sm italic">
                    Run a prediction to see results here...
                </div>
                )}
            </div>
            </div>
        </div>
      </div>
    </div>
  );
};
