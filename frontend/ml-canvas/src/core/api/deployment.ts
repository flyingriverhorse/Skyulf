import { apiClient } from './client';

export interface DeploymentInfo {
  id: number;
  job_id: string;
  model_type: string;
  artifact_uri: string;
  is_active: boolean;
  created_at: string;
}

export interface PredictionResponse {
  predictions: any[];
  model_version: string;
}

export const deploymentApi = {
  deployModel: async (jobId: string): Promise<DeploymentInfo> => {
    const response = await apiClient.post<DeploymentInfo>(`/deployment/deploy/${jobId}`);
    return response.data;
  },

  getActiveDeployment: async (): Promise<DeploymentInfo> => {
    const response = await apiClient.get<DeploymentInfo>('/deployment/active');
    return response.data;
  },

  deactivateDeployment: async (): Promise<void> => {
    await apiClient.post('/deployment/deactivate');
  },

  predict: async (data: any[]): Promise<PredictionResponse> => {
    const response = await apiClient.post<PredictionResponse>('/deployment/predict', { data });
    return response.data;
  }
};
