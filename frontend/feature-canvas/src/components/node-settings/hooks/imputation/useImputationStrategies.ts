import { useMemo } from 'react';
import {
  IMPUTATION_METHOD_OPTIONS,
  normalizeImputationStrategies,
} from '../../nodes/imputation/imputationSettings';

export const useImputationStrategies = (configState: Record<string, any>) => {
  const imputationMethodOptions = useMemo(() => IMPUTATION_METHOD_OPTIONS, []);

  const imputationMethodValues = useMemo(
    () => imputationMethodOptions.map((option) => option.value),
    [imputationMethodOptions]
  );

  const imputerStrategies = useMemo(
    () => normalizeImputationStrategies(configState?.strategies, imputationMethodValues),
    [configState?.strategies, imputationMethodValues]
  );
  const imputerStrategyCount = imputerStrategies.length;

  return {
    imputationMethodOptions,
    imputationMethodValues,
    imputerStrategies,
    imputerStrategyCount,
  };
};
