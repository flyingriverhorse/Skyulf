import { useCallback, useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import type { FeatureNodeParameter, FeatureNodeCatalogEntry } from '../../../../api';
import { useParameterCatalog } from './useParameterCatalog';

interface UseNodeParametersResult {
  parameters: FeatureNodeParameter[];
  getParameter: ReturnType<typeof useParameterCatalog>;
  getParameterIf: (condition: boolean, name: string) => FeatureNodeParameter | null;
  requiresColumnCatalog: boolean;
  dropColumnParameter: FeatureNodeParameter | null;
}

export const useNodeParameters = (
  node: Node | null | undefined,
  catalogEntry?: FeatureNodeCatalogEntry | null,
): UseNodeParametersResult => {
  const parameters = useMemo<FeatureNodeParameter[]>(() => {
    const raw = node?.data?.parameters;
    let sourceParams: FeatureNodeParameter[] = [];

    if (Array.isArray(raw) && raw.length > 0) {
      sourceParams = raw;
    } else if (catalogEntry?.parameters) {
      sourceParams = catalogEntry.parameters;
    }

    return sourceParams
      .filter((parameter) => Boolean(parameter?.name))
      .map((parameter) => ({ ...parameter }));
  }, [node, catalogEntry]);

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
