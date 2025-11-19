import { useMemo } from 'react';
import type { FeatureNodeParameter } from '../../../api';

export type ParameterLookup = (name: string) => FeatureNodeParameter | null;

export const useParameterCatalog = (parameters: FeatureNodeParameter[]): ParameterLookup => {
  return useMemo(() => {
    const byName = new Map<string, FeatureNodeParameter>();
    parameters.forEach((parameter) => {
      if (parameter?.name) {
        byName.set(parameter.name, parameter);
      }
    });
    return (name: string): FeatureNodeParameter | null => byName.get(name) ?? null;
  }, [parameters]);
};
