import { JobInfo } from '../../../../core/api/jobs';

export const getScoringMetric = (job: JobInfo): string | undefined => {
  const result = job.result as Record<string, unknown> | undefined;
  if (result?.scoring_metric) return result.scoring_metric as string;
  const config = job.config as Record<string, unknown> | undefined;
  const tuning = config?.tuning_config as Record<string, unknown> | undefined;
  return tuning?.metric as string | undefined;
};
