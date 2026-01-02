import React, { useState } from 'react';
import { X, Clock, CheckCircle, AlertCircle, Loader2, ArrowLeft, Database, Columns, FileText, EyeOff, Play, Hash, AlignLeft, Calendar } from 'lucide-react';

interface JobsHistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: any[];
  onSelect: (report: any) => void;
  onFetchReport: (id: number) => Promise<any>;
}

export const JobsHistoryModal: React.FC<JobsHistoryModalProps> = ({ isOpen, onClose, history, onSelect, onFetchReport }) => {
  const [selectedJob, setSelectedJob] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleClose = () => {
    setSelectedJob(null);
    onClose();
  };

  const handleJobClick = async (job: any) => {
    setLoading(true);
    try {
        const fullReport = await onFetchReport(job.id);
        setSelectedJob(fullReport);
    } catch (err) {
        console.error("Failed to load report details", err);
    } finally {
        setLoading(false);
    }
  };

  const getIconForType = (dtype: string) => {
    switch (dtype) {
      case 'Numeric': return <Hash className="w-4 h-4 text-blue-500" />;
      case 'Categorical': return <AlignLeft className="w-4 h-4 text-purple-500" />;
      case 'DateTime': return <Calendar className="w-4 h-4 text-green-500" />;
      default: return <FileText className="w-4 h-4 text-gray-500" />;
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full flex flex-col transition-all duration-300 ${selectedJob ? 'max-w-5xl h-[90vh]' : 'max-w-2xl max-h-[80vh]'}`}>
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            {selectedJob ? (
              <button 
                onClick={() => setSelectedJob(null)}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-500 dark:text-gray-400" />
              </button>
            ) : (
              <Clock className="w-5 h-5 text-blue-500" />
            )}
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              {selectedJob ? `Analysis #${selectedJob.id} Details` : 'Analysis History'}
            </h2>
          </div>
          <button 
            onClick={handleClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-6">
          {!selectedJob ? (
            history.length === 0 ? (
              <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                <Clock className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p>No history available for this dataset.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {history.map((job) => (
                  <div 
                    key={job.id}
                    onClick={() => handleJobClick(job)}
                    className="flex items-center justify-between p-4 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-all group"
                  >
                    <div className="flex items-center gap-4">
                      <div className={`p-2 rounded-full ${
                        job.status === 'COMPLETED' ? 'bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400' :
                        job.status === 'FAILED' ? 'bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400' :
                        'bg-yellow-100 text-yellow-600 dark:bg-yellow-900/30 dark:text-yellow-400'
                      }`}>
                        {job.status === 'COMPLETED' ? <CheckCircle className="w-5 h-5" /> :
                         job.status === 'FAILED' ? <AlertCircle className="w-5 h-5" /> :
                         <Loader2 className="w-5 h-5 animate-spin" />}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400">
                          Analysis #{job.id}
                        </div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                          {new Date(job.created_at).toLocaleString()}
                        </div>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      {/* Note: excluded_columns is not available in history list summary */}
                    </div>
                  </div>
                ))}
              </div>
            )
          ) : loading ? (
            <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <Loader2 className="w-12 h-12 mb-4 animate-spin text-blue-500" />
                <p>Loading analysis details...</p>
            </div>
          ) : (
            <div className="space-y-8">
                {/* Overview Stats */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
                            <Database className="w-4 h-4" />
                            <span className="text-xs uppercase font-medium">Rows</span>
                        </div>
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                            {selectedJob.profile_data?.row_count?.toLocaleString() || '-'}
                        </div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
                            <Columns className="w-4 h-4" />
                            <span className="text-xs uppercase font-medium">Columns</span>
                        </div>
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                            {selectedJob.profile_data?.column_count?.toLocaleString() || '-'}
                        </div>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 mb-1">
                            <FileText className="w-4 h-4" />
                            <span className="text-xs uppercase font-medium">Missing Cells</span>
                        </div>
                        <div className="text-2xl font-bold text-gray-900 dark:text-white">
                            {selectedJob.profile_data?.missing_cells_percentage?.toFixed(1)}%
                        </div>
                    </div>
                </div>

                {/* Excluded Columns */}
                {selectedJob.profile_data?.excluded_columns && selectedJob.profile_data.excluded_columns.length > 0 && (
                    <div>
                        <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3 flex items-center gap-2">
                            <EyeOff className="w-4 h-4 text-gray-500" />
                            Excluded Columns ({selectedJob.profile_data.excluded_columns.length})
                        </h3>
                        <div className="flex flex-wrap gap-2">
                            {selectedJob.profile_data.excluded_columns.map((col: string) => (
                                <span key={col} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded text-sm line-through">
                                    {col}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

                {/* Variables List */}
                <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">Variables Overview</h3>
                    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-gray-50 dark:bg-gray-900/50 text-gray-500 dark:text-gray-400">
                                <tr>
                                    <th className="px-4 py-3 font-medium">Name</th>
                                    <th className="px-4 py-3 font-medium">Type</th>
                                    <th className="px-4 py-3 font-medium">Missing</th>
                                    <th className="px-4 py-3 font-medium">Distinct</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                {selectedJob.profile_data?.columns && Object.values(selectedJob.profile_data.columns).map((v: any) => (
                                    <tr key={v.name} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                                        <td className="px-4 py-3 font-medium text-gray-900 dark:text-white">{v.name}</td>
                                        <td className="px-4 py-3">
                                            <div className="flex items-center gap-2">
                                                {getIconForType(v.dtype)}
                                                <span>{v.dtype}</span>
                                            </div>
                                        </td>
                                        <td className="px-4 py-3">
                                            {v.missing_percentage > 0 ? (
                                                <span className="text-amber-600">{v.missing_percentage.toFixed(1)}%</span>
                                            ) : (
                                                <span className="text-green-600">0%</span>
                                            )}
                                        </td>
                                        <td className="px-4 py-3 text-gray-500 dark:text-gray-400">
                                            {v.categorical_stats?.unique_count || '-'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
          )}
        </div>
        
        <div className="p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 rounded-b-xl flex justify-between items-center">
            {selectedJob ? (
                <button
                    onClick={() => {
                        onSelect(selectedJob);
                        handleClose();
                    }}
                    className="ml-auto flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors shadow-sm"
                >
                    <Play className="w-4 h-4 mr-2" />
                    Load this Report
                </button>
            ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400 text-center w-full">
                    Select a previous analysis to view details.
                </p>
            )}
        </div>
      </div>
    </div>
  );
};
