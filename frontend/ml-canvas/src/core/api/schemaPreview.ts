// C7: client for `POST /api/pipeline/schema-preview`. Pure read endpoint
// — predicts each node's output schema and surfaces broken column refs
// without running the pipeline. Used by `useSchemaPreview` to paint
// "↳ N cols" badges and red borders on the canvas.

import { apiClient, PipelineConfigModel } from './client';

export interface PredictedSchema {
  columns: string[];
  dtypes: Record<string, string>;
}

export interface BrokenReference {
  node_id: string;
  field: string;
  column: string;
  upstream_node_id: string | null;
}

export interface SchemaPreviewResponse {
  pipeline_id: string;
  /** Per-node predicted schema; `null` when the calculator is
   * data-dependent or upstream prediction was unknown. */
  predicted_schemas: Record<string, PredictedSchema | null>;
  broken_references: BrokenReference[];
}

export const previewPipelineSchema = async (
  payload: PipelineConfigModel,
): Promise<SchemaPreviewResponse> => {
  const response = await apiClient.post<SchemaPreviewResponse>(
    '/pipeline/schema-preview',
    payload,
  );
  return response.data;
};
