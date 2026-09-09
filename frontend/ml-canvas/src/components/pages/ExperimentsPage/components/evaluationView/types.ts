import type { EvaluationData } from '../../types';
import type { ThresholdMetric } from '../../utils/jobMeta';
import type { ThresholdPreviewResult } from '../../../../../core/api/thresholdTuning';

interface BestMetricInfo {
  threshold: number;
  value: number;
  splitLabel: string;
  metricName: string;
}

export interface EvaluationViewProps {
  /** Selected runs this tab can actually render, in selection order. */
  eligibleJobIds: string[];
  /** Selected runs with pipeline metadata for stable display labels. */
  eligibleJobs?: Array<{
    jobId: string;
    pipeline_id: string;
    parent_pipeline_id?: string | null;
  }>;
  evalJobId: string | null;
  fetchEvaluationData: (jobId: string) => void | Promise<void>;
  isEvalLoading: boolean;
  evalError: string | null;
  evaluationData: EvaluationData | null;
  selectedRegressionSplit: string | null;
  setSelectedRegressionSplit: (v: string | null) => void;
  showTrainMetrics: boolean;
  setShowTrainMetrics: (v: boolean) => void;
  showTestMetrics: boolean;
  setShowTestMetrics: (v: boolean) => void;
  showValMetrics: boolean;
  setShowValMetrics: (v: boolean) => void;
  threshold: number;
  setThreshold: (v: number) => void;
  selectedRocClass: string | null;
  setSelectedRocClass: (v: string) => void;
  cmView: 'overall' | 'per-class';
  setCmView: (v: 'overall' | 'per-class') => void;
  activeTab: 'slider' | 'tuning';
  setActiveTab: (v: 'slider' | 'tuning') => void;
  selectedMetric: ThresholdMetric;
  setSelectedMetric: (v: ThresholdMetric) => void;
  bestMetricInfos: BestMetricInfo[];
  handleDownload: (elementId: string, fileName: string) => Promise<void>;
  downloadingChart: string | null;
  doneChart: string | null;
  selectedTuningMetric: string;
  onSelectedTuningMetricChange: (v: string) => void;
  tuningPreview: ThresholdPreviewResult | null;
  tuningError: string | null;
  useTunedThresholds: boolean;
  hasSavedThresholds: boolean;
  onPreviewThresholds: () => void | Promise<void>;
  onSaveThresholds: () => void | Promise<void>;
  onToggleThresholds: (enabled: boolean) => void | Promise<void>;
  onClearThresholds: () => void | Promise<void>;
}

export type ChartEvaluationProps = EvaluationViewProps & {
  evaluationData: Extract<EvaluationData, { problem_type: 'classification' | 'regression' }>;
};
