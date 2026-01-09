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
        const response = await apiClient.get<RegistryItem[]>('/ml/registry');
        return response.data;
    },
};
