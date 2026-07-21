/**
 * The four ML task types the canvas's task-scoped training nodes cover.
 * Shared across the Job History drawer (`useJobStore`/`JobsDrawer`) and the
 * Experiments page (`jobMeta.ts`'s `getTaskForModelType`) so both group jobs
 * by task rather than by which training engine (`basic_training`/
 * `advanced_tuning`) ran them (plan §0.5: "Job History tabs become
 * task-type tabs").
 *
 * Defined here (a neutral, foundational location) rather than in
 * `ExperimentsPage/utils/jobMeta.ts` to avoid a page-utility module being
 * imported by the more foundational `core/store/useJobStore.ts`.
 */
export type TaskType = 'classification' | 'regression' | 'text_classification' | 'segmentation' | 'ensemble';
