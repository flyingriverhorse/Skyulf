import { useCallback, useMemo, type Dispatch, type SetStateAction } from 'react';
import {
  dedupeSkewnessTransformations,
  normalizeSkewnessTransformations,
  type SkewnessTransformationConfig,
} from '../../nodes/skewness/skewnessSettings';

export const useSkewnessState = (
  configState: Record<string, any>,
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>
) => {
  const skewnessTransformations = useMemo(
    () => dedupeSkewnessTransformations(normalizeSkewnessTransformations(configState?.transformations)),
    [configState?.transformations],
  );

  const updateSkewnessTransformations = useCallback(
    (updater: (current: SkewnessTransformationConfig[]) => SkewnessTransformationConfig[]) => {
      setConfigState((previous: any) => {
        const currentTransformations = dedupeSkewnessTransformations(
          normalizeSkewnessTransformations(previous?.transformations),
        );
        const nextTransformations = dedupeSkewnessTransformations(updater(currentTransformations));
        return {
          ...previous,
          transformations: nextTransformations.map((entry) => ({ ...entry })),
        };
      });
    },
    [setConfigState]
  );

  return {
    skewnessTransformations,
    updateSkewnessTransformations,
  };
};
