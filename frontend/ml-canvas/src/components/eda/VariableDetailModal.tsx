import React from 'react';
import { Download, X } from 'lucide-react';
import { DistributionChart } from './DistributionChart';
import { InfoTooltip } from '../ui/InfoTooltip';

interface VariableDetailModalProps {
  selectedVariable: any;
  onClose: () => void;
  downloadChart: (elementId: string, filename: string, title?: string, subtitle?: string, extraInfo?: string) => void;
  filters: any[];
  setFilters: (filters: any[]) => void;
  runAnalysis: (overrideExcluded?: string[] | any, overrideFilters?: any[]) => void;
  handleAddFilter: (column: string, value: any, operator?: string) => void;
}

export const VariableDetailModal: React.FC<VariableDetailModalProps> = ({
  selectedVariable,
  onClose,
  downloadChart,
  filters,
  setFilters,
  runAnalysis,
  handleAddFilter
}) => {
  if (!selectedVariable) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4 backdrop-blur-sm">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto flex flex-col animate-in fade-in zoom-in duration-200">
        <div className="flex justify-between items-center p-6 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-800 z-10">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
              {selectedVariable.name}
              <span className={`text-sm font-normal px-2 py-0.5 rounded-full ${
                selectedVariable.dtype === 'Numeric' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300' :
                selectedVariable.dtype === 'Categorical' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300' :
                'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
              }`}>
                {selectedVariable.dtype}
              </span>
            </h2>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>
        
        <div className="p-6 space-y-6">
          {/* Distribution Chart */}
          <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center">
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mr-2">Distribution</h3>
                    <InfoTooltip text="Click on any bar to filter the entire dataset by that range or category." />
                </div>
                <button
                    onClick={() => {
                        let extraInfo = undefined;
                        if (selectedVariable.normality_test) {
                            extraInfo = `Normality Test (${selectedVariable.normality_test.test_name})\np-value: ${selectedVariable.normality_test.p_value.toExponential(2)}\n${selectedVariable.normality_test.is_normal ? 'Likely Normal' : 'Likely Not Normal'}`;
                        }
                        downloadChart('distribution-chart', `distribution-${selectedVariable.name}`, `Distribution: ${selectedVariable.name}`, undefined, extraInfo);
                    }}
                    className="p-1.5 rounded-md bg-white border border-gray-200 text-gray-500 hover:text-gray-700 hover:bg-gray-50 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-700 shadow-sm"
                    title="Download Chart"
                >
                    <Download className="w-3 h-3" />
                </button>
            </div>
            <div id="distribution-chart">
                <DistributionChart 
                    profile={selectedVariable} 
                    onBarClick={(data) => {
                        if ((selectedVariable.dtype === 'Numeric' || selectedVariable.dtype === 'DateTime') && data.rawBin) {
                            // Add range filter (>= start AND < end)
                            const newFilters = [
                                ...filters,
                                { column: selectedVariable.name, operator: '>=', value: data.rawBin.start },
                                { column: selectedVariable.name, operator: '<', value: data.rawBin.end }
                            ];
                            setFilters(newFilters);
                            runAnalysis(undefined, newFilters);
                            onClose();
                        } else if (selectedVariable.dtype === 'Categorical') {
                            handleAddFilter(selectedVariable.name, data.value, '==');
                            onClose();
                        }
                    }}
                />
            </div>
          </div>

          {/* Detailed Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* General Stats */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">General Statistics</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                  <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Missing</span>
                  <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.missing_count} ({selectedVariable.missing_percentage.toFixed(2)}%)</span>
                </div>
              </div>
            </div>

            {/* Type Specific Stats */}
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white border-b border-gray-200 dark:border-gray-700 pb-2">
                {selectedVariable.dtype} Statistics
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                {selectedVariable.numeric_stats && (
                  <>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Mean</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.mean?.toFixed(4)}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Median</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.median?.toFixed(4)}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Std Dev</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.std?.toFixed(4)}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.min}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.max}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Zeros</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.zeros_count}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                        <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Skewness</span> 
                        <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.skewness?.toFixed(4)}</span>
                        <span className="block text-[10px] text-gray-500 mt-1">
                            {Math.abs(selectedVariable.numeric_stats.skewness) < 0.5 ? "Symmetric (Normal-like)" : 
                             Math.abs(selectedVariable.numeric_stats.skewness) < 1 ? "Moderately Skewed" : "Highly Skewed"}
                        </span>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                        <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Kurtosis</span> 
                        <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.numeric_stats.kurtosis?.toFixed(4)}</span>
                        <span className="block text-[10px] text-gray-500 mt-1">
                            {Math.abs(selectedVariable.numeric_stats.kurtosis) < 0.5 ? "Mesokurtic (Normal-like)" : 
                             selectedVariable.numeric_stats.kurtosis > 0 ? "Leptokurtic (Heavy Tails)" : "Platykurtic (Light Tails)"}
                        </span>
                    </div>
                    {selectedVariable.vif !== undefined && (
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                            <div className="flex items-center gap-1 mb-1">
                                <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">VIF</span>
                                <InfoTooltip text="Variance Inflation Factor measures multicollinearity. VIF > 5 indicates high correlation with other variables." />
                            </div>
                            <span className={`font-medium ${selectedVariable.vif > 5 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                                {selectedVariable.vif.toFixed(2)}
                            </span>
                            <span className="block text-[10px] text-gray-500 mt-1">
                                {selectedVariable.vif > 10 ? "Severe Multicollinearity" : 
                                 selectedVariable.vif > 5 ? "High Multicollinearity" : "Low Multicollinearity"}
                            </span>
                        </div>
                    )}
                    {selectedVariable.normality_test && (
                      <>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                            <div className="flex items-center gap-1 mb-1">
                                <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Normality ({selectedVariable.normality_test.test_name})</span>
                                <InfoTooltip text="Normality tests are performed on a sample of up to 5,000 rows. Shapiro-Wilk is used for smaller datasets (<5,000), and Kolmogorov-Smirnov for larger ones. For very large datasets, rely on Skewness and Kurtosis." />
                            </div>
                            <span className={`font-medium ${selectedVariable.normality_test.is_normal ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}`}>
                                {selectedVariable.normality_test.is_normal ? 'Normal' : 'Not Normal'}
                            </span>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                            <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">P-Value</span>
                            <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.normality_test.p_value.toFixed(4)}</span>
                        </div>
                      </>
                    )}
                  </>
                )}
                {selectedVariable.categorical_stats && (
                  <>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Unique Values</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.categorical_stats.unique_count}</span></div>
                    <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Rare Labels</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.categorical_stats.rare_labels_count}</span></div>
                  </>
                )}
                {selectedVariable.text_stats && (
                  <>
                     <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Avg Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.avg_length?.toFixed(2)}</span></div>
                     <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.min_length}</span></div>
                     <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max Length</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.text_stats.max_length}</span></div>
                     {selectedVariable.text_stats.sentiment_distribution && (
                        <div className="col-span-2 bg-gray-50 dark:bg-gray-900/50 p-3 rounded">
                            <span className="text-gray-500 dark:text-gray-400 block text-xs uppercase mb-2">Sentiment Distribution</span>
                            <div className="flex h-4 rounded-full overflow-hidden">
                                <div className="bg-green-500 h-full" style={{ width: `${(selectedVariable.text_stats.sentiment_distribution.positive || 0) * 100}%` }} title={`Positive: ${(selectedVariable.text_stats.sentiment_distribution.positive || 0).toFixed(2)}`}></div>
                                <div className="bg-gray-400 h-full" style={{ width: `${(selectedVariable.text_stats.sentiment_distribution.neutral || 0) * 100}%` }} title={`Neutral: ${(selectedVariable.text_stats.sentiment_distribution.neutral || 0).toFixed(2)}`}></div>
                                <div className="bg-red-500 h-full" style={{ width: `${(selectedVariable.text_stats.sentiment_distribution.negative || 0) * 100}%` }} title={`Negative: ${(selectedVariable.text_stats.sentiment_distribution.negative || 0).toFixed(2)}`}></div>
                            </div>
                            <div className="flex justify-between text-xs mt-1 text-gray-600 dark:text-gray-400">
                                <span>Pos: {((selectedVariable.text_stats.sentiment_distribution.positive || 0) * 100).toFixed(1)}%</span>
                                <span>Neu: {((selectedVariable.text_stats.sentiment_distribution.neutral || 0) * 100).toFixed(1)}%</span>
                                <span>Neg: {((selectedVariable.text_stats.sentiment_distribution.negative || 0) * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                     )}
                  </>
                )}
                {selectedVariable.date_stats && (
                  <>
                     <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Min Date</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.min_date}</span></div>
                     <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Max Date</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.max_date}</span></div>
                     <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded"><span className="text-gray-500 dark:text-gray-400 block text-xs uppercase">Duration (Days)</span> <span className="font-medium text-gray-900 dark:text-white">{selectedVariable.date_stats.duration_days?.toFixed(1)}</span></div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Additional Lists (Top K / Common Words) */}
          {selectedVariable.categorical_stats?.top_k && (
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3 uppercase tracking-wider">Top Categories</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {selectedVariable.categorical_stats.top_k.map((item: any, idx: number) => (
                          <div key={idx} className="flex justify-between items-center p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-sm">
                              <span className="truncate font-medium text-gray-700 dark:text-gray-300" title={item.value}>{item.value}</span>
                              <span className="text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-900 px-2 py-0.5 rounded-full text-xs">{item.count}</span>
                          </div>
                      ))}
                  </div>
              </div>
          )}

          {selectedVariable.text_stats?.common_words && (
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg border border-gray-100 dark:border-gray-800">
                  <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3 uppercase tracking-wider">Most Common Words</h3>
                  <div className="flex flex-wrap gap-2">
                      {selectedVariable.text_stats.common_words.map((item: any, idx: number) => (
                          <div key={idx} className="flex items-center p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 text-sm">
                              <span className="font-medium text-blue-600 dark:text-blue-400 mr-2">{item.word}</span>
                              <span className="text-gray-500 dark:text-gray-400 text-xs">({item.count})</span>
                          </div>
                      ))}
                  </div>
              </div>
          )}
        </div>
      </div>
    </div>
  );
};
