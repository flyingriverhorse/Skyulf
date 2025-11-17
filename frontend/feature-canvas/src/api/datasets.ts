// @ts-nocheck
import { DatasetSourceSummary } from './types';

export async function fetchDatasets(limit = 8): Promise<DatasetSourceSummary[]> {
  const response = await fetch(`/ml-workflow/api/datasets?limit=${encodeURIComponent(String(limit))}`);

  if (!response.ok) {
    throw new Error('Failed to load datasets');
  }

  return response.json();
}
