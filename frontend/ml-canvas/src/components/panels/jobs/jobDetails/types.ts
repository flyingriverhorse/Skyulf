import { JobInfo } from '../../../../core/api/jobs';

export interface JobRecordContext {
  job: JobInfo;
  origin?: string;
  filters?: Record<string, string>;
}
