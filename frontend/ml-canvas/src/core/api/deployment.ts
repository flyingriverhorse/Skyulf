import { apiClientV2 } from './client';

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
  deploy: async (jobId: string): Promise<DeploymentInfo> => {
    const response = await apiClientV2.post<DeploymentInfo>(`/deployment/deploy/${jobId}`);
    return response.data;
  },

  getActive: async (): Promise<DeploymentInfo> => {
    const response = await apiClientV2.get<DeploymentInfo>('/deployment/active');
    return response.data;
  },

  deactivate: async (): Promise<void> => {
    await apiClientV2.post('/deployment/deactivate');
  },

  predict: async (data: any[]): Promise<PredictionResponse> => {
    const response = await apiClientV2.post<PredictionResponse>('/deployment/predict', { data });
    return response.data;
  }
};
