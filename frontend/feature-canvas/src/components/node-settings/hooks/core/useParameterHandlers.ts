import { useCallback, type Dispatch, type SetStateAction } from 'react';

interface UseParameterHandlersArgs {
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
}

export const useParameterHandlers = ({ setConfigState }: UseParameterHandlersArgs) => {
  const handleParameterChange = useCallback(
    (name: string, value: any) => {
      if (!name) {
        return;
      }
      setConfigState((previous) => {
        const next = { ...previous };
        if (value === undefined) {
          delete next[name];
        } else {
          next[name] = value;
        }
        return next;
      });
    },
    [setConfigState],
  );

  const handleNumberChange = useCallback(
    (name: string, rawValue: string) => {
      if (!name) {
        return;
      }
      if (rawValue === '') {
        handleParameterChange(name, undefined);
        return;
      }
      const numericValue = Number(rawValue);
      if (Number.isNaN(numericValue)) {
        return;
      }
      handleParameterChange(name, numericValue);
    },
    [handleParameterChange],
  );

  const handlePercentileChange = useCallback(
    (name: 'lower_percentile' | 'upper_percentile', rawValue: string) => {
      if (!name) {
        return;
      }
      if (rawValue === '') {
        handleParameterChange(name, undefined);
        return;
      }
      const numericValue = Number(rawValue);
      if (!Number.isFinite(numericValue)) {
        return;
      }
      const clamped = Math.min(Math.max(numericValue, 0), 100);
      handleParameterChange(name, clamped);
    },
    [handleParameterChange],
  );

  const handleBooleanChange = useCallback(
    (name: string, checked: boolean) => {
      handleParameterChange(name, checked);
    },
    [handleParameterChange],
  );

  const handleTextChange = useCallback(
    (name: string, nextValue: string) => {
      handleParameterChange(name, nextValue);
    },
    [handleParameterChange],
  );

  return {
    handleParameterChange,
    handleNumberChange,
    handlePercentileChange,
    handleBooleanChange,
    handleTextChange,
  };
};
