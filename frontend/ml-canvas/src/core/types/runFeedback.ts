/** A client receipt that keeps background feedback tied to the submitted job IDs. */
export interface SubmittedRun {
  label: string;
  jobIds: string[];
}

/** Submission state survives the keyed node settings editor being closed or replaced. */
export interface NodeSubmission {
  pending: boolean;
  message: string;
  run: SubmittedRun | null;
}
