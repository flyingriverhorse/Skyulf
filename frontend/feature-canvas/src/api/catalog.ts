// @ts-nocheck
import { FeatureNodeCatalogEntry } from './types';

export async function fetchNodeCatalog(): Promise<FeatureNodeCatalogEntry[]> {
  const response = await fetch('/ml-workflow/api/node-catalog');

  if (!response.ok) {
    throw new Error('Failed to load node catalog');
  }

  return response.json();
}
