import React, { useState, useMemo } from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { ChevronUp, ChevronDown, Table, Layers } from 'lucide-react';

export const ResultsPanel: React.FC = () => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeTab, setActiveTab] = useState<string | null>(null);

  // Determine available datasets (tabs)
  const datasets = useMemo(() => {
    if (!executionResult?.preview_data) return {};
    
    if (Array.isArray(executionResult.preview_data)) {
      return { 'Result': executionResult.preview_data };
    }
    
    if (typeof executionResult.preview_data === 'object') {
      // It's a dictionary of datasets (e.g. { train: [...], test: [...] } or { X: [...], y: [...] })
      return executionResult.preview_data;
    }
    
    return {};
  }, [executionResult]);

  const tabNames = Object.keys(datasets);
  
  // Set default tab if none selected or current selection is invalid
  React.useEffect(() => {
    if (tabNames.length > 0 && (!activeTab || !tabNames.includes(activeTab))) {
      // Prefer 'train' or 'X' if available, otherwise first one
      if (tabNames.includes('train')) setActiveTab('train');
      else if (tabNames.includes('X')) setActiveTab('X');
      else setActiveTab(tabNames[0]);
    }
  }, [tabNames, activeTab]);

  if (!executionResult) return null;

  const currentRows = activeTab && datasets[activeTab] ? datasets[activeTab] : [];
  const columns = currentRows.length > 0 ? Object.keys(currentRows[0]) : [];
  const applied_steps = executionResult.node_results ? Object.keys(executionResult.node_results) : [];

  // Check for errors
  const errorNodeId = Object.keys(executionResult.node_results || {}).find(
    nodeId => executionResult.node_results[nodeId].status === 'failed'
  );
  const error = errorNodeId ? executionResult.node_results[errorNodeId].error : null;

  return (
    <div 
      className={`absolute bottom-0 left-0 right-0 bg-background border-t shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] transition-all duration-300 z-20 flex flex-col ${
        isExpanded ? 'h-96' : 'h-10'
      }`}
    >
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 bg-muted/10 cursor-pointer hover:bg-muted/20 border-b select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Table className="w-4 h-4 text-primary" />
          <span className="font-semibold text-sm">Preview Results</span>
          <span className="text-xs text-muted-foreground ml-2">
            {currentRows.length} rows shown
          </span>
          {executionResult.status === 'failed' && (
            <span className="text-xs text-red-600 font-bold ml-2">
              (Failed)
            </span>
          )}
        </div>
        <button className="p-1 hover:bg-muted rounded">
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </button>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Error Message */}
          {executionResult.status === 'failed' && (
             <div className="p-4 bg-red-50 border-b text-red-800 text-sm overflow-auto max-h-32">
                <div className="font-bold mb-1">Pipeline Execution Failed</div>
                <div className="font-mono text-xs whitespace-pre-wrap">{error || 'Unknown error occurred during execution.'}</div>
             </div>
          )}

          {/* Tabs for Multiple Outputs */}
          {tabNames.length > 1 && (
            <div className="flex items-center gap-1 px-2 pt-2 border-b bg-muted/5">
              <Layers className="w-3 h-3 text-muted-foreground mr-1" />
              {tabNames.map(name => (
                <button
                  key={name}
                  onClick={() => setActiveTab(name)}
                  className={`px-3 py-1 text-xs font-medium rounded-t-md border-t border-l border-r transition-colors ${
                    activeTab === name 
                      ? 'bg-background text-primary border-b-background translate-y-[1px]' 
                      : 'bg-muted/30 text-muted-foreground hover:bg-muted/50 border-transparent'
                  }`}
                >
                  {name}
                </button>
              ))}
            </div>
          )}

          {/* Signals / Warnings */}
          {applied_steps.length > 0 && executionResult.status !== 'failed' && (
             <div className="p-2 bg-blue-50 border-b flex gap-2 overflow-x-auto">
                {applied_steps.map((step: string, idx: number) => (
                  <div key={idx} className="text-xs text-blue-800 bg-blue-100 px-2 py-1 rounded border border-blue-200 whitespace-nowrap">
                    {step}
                  </div>
                ))}
             </div>
          )}

          {/* Data Table */}
          <div className="flex-1 overflow-auto">
            <table className="w-full text-sm text-left border-collapse">
              <thead className="text-xs text-muted-foreground uppercase bg-muted sticky top-0 z-10 shadow-sm">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-4 py-2 font-medium border-b whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {currentRows.map((row: any, idx: number) => (
                  <tr key={idx} className="border-b hover:bg-muted/10">
                    {columns.map((col) => (
                      <td key={`${idx}-${col}`} className="px-4 py-2 whitespace-nowrap font-mono text-xs">
                        {row[col] !== null ? String(row[col]) : <span className="text-muted-foreground italic">null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {currentRows.length === 0 && (
              <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
                No preview data available {activeTab ? `for ${activeTab}` : ''}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
