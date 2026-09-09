export type Config = Record<string, unknown>;

export type GraphNode = {
  node_id: string;
  step_type?: string;
  params?: Config;
  inputs?: string[];
};

export interface PipelineRow {
  nid: string;
  idx: number;
  cells: (string | null)[];
  allSame: boolean;
}

export interface PipelineData {
  hasSteps: boolean;
  rows: PipelineRow[];
}
