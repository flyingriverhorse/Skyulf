import { create } from 'zustand';
import { runPipelinePreview, type PipelineConfigModel, type PreviewResponse } from '../api/client';
import { previewConfigurationKey } from '../utils/previewConfiguration';

interface InspectionReceipt {
  configurationKey: string;
  response: PreviewResponse;
}

interface NodeInspectionState {
  receipt: InspectionReceipt | null;
  isLoading: boolean;
  error: string | null;
  runPreview: (config: PipelineConfigModel) => Promise<PreviewResponse>;
}

/** Keep the latest graph inspection in memory and serialize toolbar/panel preview requests. */
export const useNodeInspectionStore = create<NodeInspectionState>((set, get) => ({
  receipt: null,
  isLoading: false,
  error: null,
  runPreview: async (config) => {
    if (get().isLoading) throw new Error('A data preview is already running.');
    // Match the wire payload and detach nested params from editable graph objects.
    const submitted = JSON.parse(JSON.stringify(config)) as PipelineConfigModel;
    const configurationKey = previewConfigurationKey(submitted);
    set({ isLoading: true, error: null, receipt: null });
    try {
      const response = await runPipelinePreview(submitted, { inspectAll: true });
      set({ receipt: { configurationKey, response } });
      return response;
    } catch (error) {
      set({ error: error instanceof Error ? error.message : 'Data preview failed. Try again.' });
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },
}));
