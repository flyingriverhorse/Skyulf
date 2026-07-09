import { apiClient } from './client';

export interface RegistryItem {
    id: string;
    name: string;
    category: string;
    description: string;
    params: Record<string, unknown>;
    tags?: string[];
}

export const registryApi = {
    getAllNodes: async (): Promise<RegistryItem[]> => {
        // Node catalog lives under the pipeline router (`/api/pipeline/registry`,
        // see backend/ml_pipeline/_internal/_routers/meta.py). `/api/ml/registry`
        // is the *model* registry (backend/ml_pipeline/model_registry/api.py) and
        // has no bare `/registry` route, so calling it here always 404s.
        const response = await apiClient.get<RegistryItem[]>('/pipeline/registry');
        return response.data;
    },
};
