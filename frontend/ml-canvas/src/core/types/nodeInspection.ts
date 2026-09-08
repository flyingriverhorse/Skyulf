/** A bounded sample of one measured input/output port and dataset split. */
export interface InspectionTable {
  port: string;
  split: string | null;
  row_count: number;
  column_count: number;
  columns: { name: string; dtype: string }[];
  rows: Record<string, unknown>[];
  truncated: boolean;
}

/** Availability is separate from an empty, successfully measured table. */
export interface InspectionSide {
  status: 'available' | 'unavailable' | 'error';
  reason: string | null;
  tables: InspectionTable[];
}

/** One execution of the requested node within a particular preview branch. */
export interface NodeInspection {
  node_id: string;
  branch_id: string;
  branch_label: string;
  path_id?: string | null;
  path_label?: string | null;
  input: InspectionSide;
  output: InspectionSide;
}
