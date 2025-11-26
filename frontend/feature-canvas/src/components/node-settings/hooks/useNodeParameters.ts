import { useCallback, useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import type { FeatureNodeParameter } from '../../../api';
import { useParameterCatalog } from './useParameterCatalog';

interface UseNodeParametersResult {
  parameters: FeatureNodeParameter[];
  getParameter: ReturnType<typeof useParameterCatalog>;
  getParameterIf: (condition: boolean, name: string) => FeatureNodeParameter | null;
  requiresColumnCatalog: boolean;
  dropColumnParameter: FeatureNodeParameter | null;
}

export const useNodeParameters = (node: Node | null | undefined): UseNodeParametersResult => {
  const parameters = useMemo<FeatureNodeParameter[]>(() => {
    const raw = node?.data?.parameters;
    if (!Array.isArray(raw)) {
      return [];
    }
    return raw
      .filter((parameter) => Boolean(parameter?.name))
      .map((parameter) => ({ ...parameter }));
  }, [node]);

  const getParameter = useParameterCatalog(parameters);

  const getParameterIf = useCallback(
    (condition: boolean, name: string) => (condition ? getParameter(name) : null),
    [getParameter],
  );

  const requiresColumnCatalog = useMemo(
    () =>
      parameters.some(
        (parameter) =>
          parameter?.type === 'multi_select' && parameter?.source?.type !== 'drop_column_recommendations',
      ),
    [parameters],
  );

  const dropColumnParameter = useMemo(
    () => parameters.find((parameter) => parameter?.source?.type === 'drop_column_recommendations') ?? null,
    [parameters],
  );

  return {
    parameters,
    getParameter,
    getParameterIf,
    requiresColumnCatalog,
    dropColumnParameter,
  };
};
