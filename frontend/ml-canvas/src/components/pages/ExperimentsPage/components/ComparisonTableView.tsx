import React, { useMemo } from 'react';
import { GitBranch, Trophy } from 'lucide-react';
import type { JobInfo } from '../../../../core/api/jobs';
import { shortRunId } from '../utils/jobMeta';
import { getParameterKeys } from './comparisonTable/config';
import { preparePipelineData } from './comparisonTable/pipeline';
import { EnsembleRows, prepareEnsembleData } from './comparisonTable/EnsembleRows';
import { PipelineRows } from './comparisonTable/PipelineRows';
import { MetricRows } from './comparisonTable/MetricRows';
import { ParameterRows } from './comparisonTable/ParameterRows';
import { TrainingRows } from './comparisonTable/TrainingRows';
import { SectionHeader } from './comparisonTable/SectionHeader';

interface Props {
  selectedJobs: JobInfo[];
  metricKeys: string[];
  isPipelineExpanded: boolean;
  setIsPipelineExpanded: (v: boolean) => void;
  isMetricsExpanded: boolean;
  setIsMetricsExpanded: (v: boolean) => void;
  isParamsExpanded: boolean;
  setIsParamsExpanded: (v: boolean) => void;
  isTuningExpanded: boolean;
  setIsTuningExpanded: (v: boolean) => void;
}

export const ComparisonTableView: React.FC<Props> = ({
  selectedJobs,
  metricKeys,
  isPipelineExpanded,
  setIsPipelineExpanded,
  isMetricsExpanded,
  setIsMetricsExpanded,
  isParamsExpanded,
  setIsParamsExpanded,
  isTuningExpanded,
  setIsTuningExpanded,
}) => {
  // Graph traversal, ensemble structure and parameter unions depend only on selection.
  const ensembleData = useMemo(() => prepareEnsembleData(selectedJobs), [selectedJobs]);
  const pipelineData = useMemo(() => preparePipelineData(selectedJobs), [selectedJobs]);
  const paramsAllKeys = useMemo(() => getParameterKeys(selectedJobs), [selectedJobs]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-100">Detailed Comparison</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead className="text-xs text-gray-700 uppercase bg-gray-50 dark:bg-gray-900/50 dark:text-gray-400">
            <tr>
              <th className="px-4 py-2">Parameter / Metric</th>
              {selectedJobs.map(job => (
                <th key={job.job_id} className="px-4 py-2 font-mono break-all min-w-[100px]">
                  {shortRunId(job)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {/* Model Type */}
            <tr className="bg-white dark:bg-gray-800">
              <td className="px-4 py-2 font-medium text-gray-900 dark:text-gray-100">Model Type</td>
              {selectedJobs.map(job => (
                <td key={job.job_id} className="px-4 py-2 text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-1.5">
                    {job.model_type}
                    {job.branch_index != null && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 text-[10px] font-semibold">
                        <GitBranch className="w-2.5 h-2.5" /> Path {String.fromCharCode(65 + (job.branch_index ?? 0))}
                      </span>
                    )}
                    {job.promoted_at && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-[10px] font-semibold">
                        <Trophy className="w-2.5 h-2.5" /> Winner
                      </span>
                    )}
                  </div>
                </td>
              ))}
            </tr>
            <EnsembleRows selectedJobs={selectedJobs} ensembleData={ensembleData} />
            <SectionHeader label="Pipeline Steps" columnCount={selectedJobs.length + 1} expanded={isPipelineExpanded} onExpandedChange={setIsPipelineExpanded} />
            {isPipelineExpanded && <PipelineRows selectedJobs={selectedJobs} pipelineData={pipelineData} />}
            <SectionHeader label="Key Metrics" columnCount={selectedJobs.length + 1} expanded={isMetricsExpanded} onExpandedChange={setIsMetricsExpanded} />
            {isMetricsExpanded && <MetricRows selectedJobs={selectedJobs} metricKeys={metricKeys} />}
            <SectionHeader label="Hyperparameters" columnCount={selectedJobs.length + 1} expanded={isParamsExpanded} onExpandedChange={setIsParamsExpanded} />
            {isParamsExpanded && <ParameterRows selectedJobs={selectedJobs} paramsAllKeys={paramsAllKeys} />}
            <SectionHeader label="Training Configuration" columnCount={selectedJobs.length + 1} expanded={isTuningExpanded} onExpandedChange={setIsTuningExpanded} />
            {isTuningExpanded && <TrainingRows selectedJobs={selectedJobs} />}
          </tbody>
        </table>
      </div>
    </div>
  );
};
