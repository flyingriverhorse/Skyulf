import React, { useState, useEffect } from 'react';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { NodeDefinition } from '../../../core/types/nodes';
import { Table, Activity, CheckCircle, AlertCircle, Play } from 'lucide-react';
import { jobsApi, JobInfo } from '../../../core/api/jobs';
import { convertGraphToPipelineConfig } from '../../../core/utils/pipelineConverter';

interface DataPreviewConfig {
  lastRunJobId?: string;
}

// Helper to render a mini table
const renderTable = (summary: unknown) => {
  if (!summary || !(summary as Record<string, any>).sample) return <div className="text-xs text-muted-foreground italic">No data available</div>;
  
  const cols = Object.keys((summary as Record<string, any>).sample[0] || {}).slice(0, 5); // Show max 5 cols
  
  return (
    <div className="overflow-x-auto rounded border border-border">
      <div className="text-xs font-semibold mb-1 flex justify-between p-2 bg-muted/50 border-b border-border">
          <span>{(summary as Record<string, any>).name}</span>
          <span className="text-muted-foreground">{(summary as Record<string, any>).shape[0]} rows x {(summary as Record<string, any>).shape[1]} cols</span>
      </div>
      <table className="w-full text-[10px] border-collapse">
        <thead>
          <tr className="bg-muted/30">
            {cols.map(c => <th key={c} className="p-2 border-b border-r border-border text-left font-medium text-muted-foreground last:border-r-0">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {(summary as Record<string, any>).sample.slice(0, 5).map((row: unknown, i: number) => (
            <tr key={i} className="border-b border-border last:border-0 hover:bg-muted/20">
              {cols.map(c => (
                <td key={c} className="p-2 border-r border-border truncate max-w-[80px] last:border-r-0">{String((row as Record<string, unknown>)[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const DataPreviewSettings: React.FC<{ config: DataPreviewConfig; onChange: (c: DataPreviewConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId
}) => {
  const { nodes, edges } = useGraphStore();
  const [isRunning, setIsRunning] = useState(false);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [activeTab, setActiveTab] = useState<'train' | 'test' | 'validation'>('train');

  // Poll for job status if we have a job ID
  useEffect(() => {
    const jobId = config.lastRunJobId;
    if (!jobId) return;

    const fetchJob = async () => {
      try {
        const j = await jobsApi.getJob(jobId);
        setJob(j);
        if (j.status !== 'completed' && j.status !== 'failed') {
           // Keep polling if not done
        }
      } catch (e) {
        console.error("Failed to fetch job", e);
      }
    };

    void fetchJob();
    const interval = setInterval(fetchJob, 2000);
    return () => { clearInterval(interval); };
  }, [config.lastRunJobId]);

  const handleRunPreview = async () => {
    if (!nodeId) return;
    setIsRunning(true);
    try {
      const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
      
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        target_node_id: nodeId,
        job_type: 'preview'
      });
      
      onChange({ ...config, lastRunJobId: response.job_id });
    } catch (error) {
      console.error("Failed to run preview:", error);
      alert("Failed to start preview job.");
    } finally {
      setIsRunning(false);
    }
  };

  const result = job?.result?.metrics || job?.result;
  const dataSummary = (result as Record<string, any>)?.data_summary;
  const operationMode = (result as Record<string, any>)?.operation_mode;

  return (
    <div className="p-4 space-y-4 h-full overflow-y-auto">
      <div className="text-sm text-muted-foreground">
        Connect this node to any data output to inspect the data state, schema, and applied transformations.
      </div>
      
      <button
        onClick={handleRunPreview}
        disabled={isRunning}
        className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-[#38bdf8] via-[#6366f1] to-[#a855f7] text-white px-3 py-1.5 rounded-md hover:opacity-90 disabled:opacity-50 transition-all text-xs font-medium shadow-md"
      >
        {isRunning ? <Activity size={14} className="animate-spin" /> : <Play size={14} />}
        {isRunning ? 'Running...' : 'Run Preview'}
      </button>
      
      {config.lastRunJobId && job && job.status !== 'completed' && (
        <div className="text-xs text-muted-foreground flex items-center gap-2 p-2 bg-muted/30 rounded border border-border">
          <span>Status:</span>
          <span className={`ml-auto font-medium px-1.5 py-0.5 rounded text-[10px] ${
            job.status === 'failed' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : 
            'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
          }`}>
            {job.status}
          </span>
        </div>
      )}

      {/* Results Area */}
      {job?.status === 'completed' && result && (
        <div className="space-y-4 border-t border-border pt-4 animate-in fade-in slide-in-from-top-2 duration-300">
             {/* Operation Mode */}
             <div className="text-[10px] bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300 p-2 rounded border border-blue-100 dark:border-blue-900/50">
                <strong className="block mb-0.5">Operation Mode</strong>
                {operationMode || 'Unknown'}
            </div>

            {/* Tabs */}
            {dataSummary && (
                <div className="space-y-2">
                    <div className="flex border-b border-border">
                        {['train', 'test', 'validation'].map(t => (
                            dataSummary[t] && (
                                <button
                                    key={t}
                                    className={`text-[10px] px-3 py-1.5 border-b-2 transition-colors ${
                                      activeTab === t 
                                        ? 'border-primary font-medium text-primary' 
                                        : 'border-transparent text-muted-foreground hover:text-foreground'
                                    }`}
                                    onClick={() => { setActiveTab(t as 'train' | 'test' | 'validation'); }}
                                >
                                    {t.charAt(0).toUpperCase() + t.slice(1)}
                                </button>
                            )
                        ))}
                    </div>
                    
                    {dataSummary[activeTab] ? renderTable(dataSummary[activeTab]) : (
                        <div className="text-xs text-muted-foreground p-4 text-center border border-dashed border-border rounded">No {activeTab} data available</div>
                    )}
                </div>
            )}
            
            {/* Transformations */}
            {(result as Record<string, any>).applied_transformations && (result as Record<string, any>).applied_transformations.length > 0 && (
                <div className="pt-2 border-t border-border">
                    <div className="text-[10px] font-semibold mb-2 text-foreground">Applied Steps</div>
                    <div className="space-y-1.5">
                        {(result as Record<string, any>).applied_transformations.map((t: unknown, i: number) => (
                            <div key={i} className="text-[10px] flex items-center gap-2 text-muted-foreground p-1.5 bg-muted/30 rounded border border-border">
                                <span className="w-4 h-4 rounded-full bg-muted flex items-center justify-center text-[8px] font-medium text-foreground">{i+1}</span>
                                <div className="flex flex-col">
                                  <span className="font-medium text-foreground">{(t as Record<string, unknown>).transformer_name as string}</span>
                                  <span className="text-[9px] opacity-80">{(t as Record<string, unknown>).transformer_type as string}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
      )}
    </div>
  );
};

const DataPreviewComponent: React.FC<{ data: DataPreviewConfig }> = ({ data }) => {
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    const jobId = data.lastRunJobId;
    if (!jobId) return;
    
    const checkStatus = async () => {
        try {
            const job = await jobsApi.getJob(jobId);
            setStatus(job.status);
        } catch (e) {
            setStatus('error');
        }
    };
    
    void checkStatus();
    // Poll if not final
    const interval = setInterval(() => {
        void checkStatus();
    }, 5000);
    
    return () => { clearInterval(interval); };
  }, [data.lastRunJobId]);

  return (
    <div className="text-xs flex items-center gap-2">
       {!data.lastRunJobId ? (
         <span className="text-muted-foreground italic">Not run yet</span>
       ) : status === 'completed' ? (
         <span className="text-green-600 dark:text-green-400 font-medium flex items-center gap-1"><CheckCircle size={12}/> Ready</span>
       ) : status === 'failed' ? (
         <span className="text-red-600 dark:text-red-400 font-medium flex items-center gap-1"><AlertCircle size={12}/> Failed</span>
       ) : (
         <span className="text-blue-600 dark:text-blue-400 font-medium flex items-center gap-1"><Activity size={12} className="animate-spin"/> Running</span>
       )}
    </div>
  );
};

export const DataPreviewNode: NodeDefinition<DataPreviewConfig> = {
  type: 'data_preview',
  label: 'Data Preview',
  category: 'Evaluation', // Or Utility
  description: 'Inspect data state and transformations at this point.',
  icon: Table,
  inputs: [{ id: 'in', label: 'Input Data', type: 'any' }],
  outputs: [], // Sink node
  component: DataPreviewComponent,
  settings: DataPreviewSettings,
  validate: () => ({ isValid: true }),
  getDefaultConfig: () => ({}),
};
