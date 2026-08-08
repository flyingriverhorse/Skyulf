import React, { useMemo, useState } from 'react';
import { shortRunId } from '../utils/jobMeta';
import { ShapSummaryView, type ShapSummaryCoverageInput } from './ShapSummaryView';
import { ShapBeeswarmView } from './ShapBeeswarmView';
import { ShapDependenceView } from './ShapDependenceView';
import { ShapWaterfallView } from './ShapWaterfallView';
import { ShapForceView } from './ShapForceView';
import { ShapInteractionView } from './ShapInteractionView';
import type { ShapExplanationData } from '../types';

export interface ShapExplanationEntry {
  jobId: string;
  pipeline_id: string;
  parent_pipeline_id?: string | null;
  modelType: string;
  shapExplanation: ShapExplanationData | null;
}

interface Props {
  shapExplanationByJob: ShapExplanationEntry[];
  coverageInputs: ShapSummaryCoverageInput[];
  handleDownload: (elementId: string, fileName: string) => void | Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
}

type ShapSubView = 'summary' | 'beeswarm' | 'dependence' | 'waterfall' | 'force' | 'interaction';

const SUB_TABS: { key: ShapSubView; label: string }[] = [
  { key: 'summary', label: 'Summary' },
  { key: 'beeswarm', label: 'Beeswarm' },
  { key: 'dependence', label: 'Dependence' },
  { key: 'waterfall', label: 'Waterfall' },
  { key: 'force', label: 'Force Plot' },
  { key: 'interaction', label: 'Interaction' },
];

const SUB_TAB_BASE = 'px-3 py-1.5 text-sm font-medium rounded-md transition-colors';
const subTabClass = (active: boolean) => `${SUB_TAB_BASE} ${
  active
    ? 'bg-blue-600 text-white'
    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
}`;

// Beeswarm/Dependence/Waterfall each explain a *single* run's predictions
// (they need per-row SHAP values, which aren't meaningfully comparable
// across different models/runs). Only the Summary bar chart aggregates
// across every selected run, mirroring Feature Importance.
export const ShapExplainabilityView: React.FC<Props> = ({
  shapExplanationByJob,
  coverageInputs,
  handleDownload,
  downloadingChart,
  doneChart,
}) => {
  const [subView, setSubView] = useState<ShapSubView>('summary');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const jobsWithData = useMemo(
    () => shapExplanationByJob.filter(
      (j): j is ShapExplanationEntry & { shapExplanation: ShapExplanationData } => j.shapExplanation !== null
    ),
    [shapExplanationByJob]
  );

  const activeJob = jobsWithData.find(j => j.jobId === selectedJobId) ?? jobsWithData[0];

  if (jobsWithData.length === 0) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex gap-1.5">
          {SUB_TABS.map(tab => (
            <button
              key={tab.key}
              className={subTabClass(subView === tab.key)}
              onClick={() => { setSubView(tab.key); }}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {subView !== 'summary' && jobsWithData.length > 1 && (
          <select
            className="ml-auto bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg p-2"
            value={activeJob?.jobId ?? ''}
            onChange={(e) => { setSelectedJobId(e.target.value); }}
          >
            {jobsWithData.map(j => (
              <option key={j.jobId} value={j.jobId}>
                {j.modelType !== 'unknown' ? j.modelType : 'model'} ({shortRunId(j)})
              </option>
            ))}
          </select>
        )}
      </div>

      {subView === 'summary' && (
        <ShapSummaryView
          shapSummaryByJob={shapExplanationByJob.map(j => ({
            jobId: j.jobId,
            pipeline_id: j.pipeline_id,
            parent_pipeline_id: j.parent_pipeline_id ?? null,
            modelType: j.modelType,
            shapSummary: j.shapExplanation?.mean_abs_importance ?? null,
          }))}
          coverageInputs={coverageInputs}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
        />
      )}

      {subView === 'beeswarm' && activeJob && (
        <ShapBeeswarmView
          jobId={activeJob.jobId}
          pipeline_id={activeJob.pipeline_id}
          parent_pipeline_id={activeJob.parent_pipeline_id ?? null}
          modelType={activeJob.modelType}
          shapExplanation={activeJob.shapExplanation}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
        />
      )}

      {subView === 'dependence' && activeJob && (
        <ShapDependenceView
          jobId={activeJob.jobId}
          pipeline_id={activeJob.pipeline_id}
          parent_pipeline_id={activeJob.parent_pipeline_id ?? null}
          modelType={activeJob.modelType}
          shapExplanation={activeJob.shapExplanation}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
        />
      )}

      {subView === 'waterfall' && activeJob && (
        <ShapWaterfallView
          jobId={activeJob.jobId}
          pipeline_id={activeJob.pipeline_id}
          parent_pipeline_id={activeJob.parent_pipeline_id ?? null}
          modelType={activeJob.modelType}
          shapExplanation={activeJob.shapExplanation}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
        />
      )}

      {subView === 'force' && activeJob && (
        <ShapForceView
          jobId={activeJob.jobId}
          pipeline_id={activeJob.pipeline_id}
          parent_pipeline_id={activeJob.parent_pipeline_id ?? null}
          modelType={activeJob.modelType}
          shapExplanation={activeJob.shapExplanation}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
        />
      )}

      {subView === 'interaction' && activeJob && (
        <ShapInteractionView
          jobId={activeJob.jobId}
          pipeline_id={activeJob.pipeline_id}
          parent_pipeline_id={activeJob.parent_pipeline_id ?? null}
          modelType={activeJob.modelType}
          shapExplanation={activeJob.shapExplanation}
          handleDownload={handleDownload}
          downloadingChart={downloadingChart}
          doneChart={doneChart}
        />
      )}
    </div>
  );
};
