import { useCallback } from 'react';
import {
  createFeatureMathOperation,
  FeatureMathOperationDraft,
  FeatureMathOperationType,
  getMethodOptions,
  normalizeFeatureMathOperations,
  serializeFeatureMathOperations,
} from '../../nodes/feature_math/featureMathSettings';
import {
  sanitizeConstantsList,
  sanitizeDatetimeFeaturesList,
  sanitizeIntegerValue,
  sanitizeNumberValue,
  sanitizeStringList,
  sanitizeTimezoneValue,
} from '../../utils/sanitizers';
import type { CatalogFlagMap } from '../core/useCatalogFlags';

interface UseFeatureMathHandlersProps {
  catalogFlags: CatalogFlagMap;
  setConfigState: React.Dispatch<React.SetStateAction<any>>;
  setCollapsedFeatureMath: React.Dispatch<React.SetStateAction<Set<string>>>;
}

export const useFeatureMathHandlers = ({
  catalogFlags,
  setConfigState,
  setCollapsedFeatureMath,
}: UseFeatureMathHandlersProps) => {
  const { isFeatureMathNode } = catalogFlags;

  const updateFeatureMathOperations = useCallback(
    (updater: (operations: FeatureMathOperationDraft[]) => FeatureMathOperationDraft[]) => {
      setConfigState((previous: any) => {
        if (!isFeatureMathNode) {
          return previous;
        }
        const baseConfig =
          previous && typeof previous === 'object' && !Array.isArray(previous) ? { ...previous } : {};
        const existingOperations = normalizeFeatureMathOperations(
          Array.isArray((previous as any)?.operations) ? (previous as any).operations : [],
        );
        const nextOperations = updater(existingOperations);
        baseConfig.operations = serializeFeatureMathOperations(nextOperations);
        return baseConfig;
      });
    },
    [isFeatureMathNode, setConfigState],
  );

  const handleAddFeatureMathOperation = useCallback(
    (operationType: FeatureMathOperationType) => {
      if (!isFeatureMathNode) {
        return;
      }
      let createdOperationId = '';
      updateFeatureMathOperations((current) => {
        const existingIds = new Set(current.map((operation) => operation.id));
        const draft = createFeatureMathOperation(operationType, existingIds);
        createdOperationId = draft.id;
        return [...current, draft];
      });
      if (createdOperationId) {
        setCollapsedFeatureMath((previous) => {
          const next = new Set(previous);
          next.delete(createdOperationId);
          return next;
        });
      }
    },
    [isFeatureMathNode, setCollapsedFeatureMath, updateFeatureMathOperations],
  );

  const handleDuplicateFeatureMathOperation = useCallback(
    (operationId: string) => {
      if (!isFeatureMathNode) {
        return;
      }
      let createdOperationId = '';
      updateFeatureMathOperations((current) => {
        const index = current.findIndex((operation) => operation.id === operationId);
        if (index === -1) {
          return current;
        }
        const source = current[index];
        const existingIds = new Set(current.map((operation) => operation.id));
        const seed = createFeatureMathOperation(source.type, existingIds);
        createdOperationId = seed.id;
        const duplicate: FeatureMathOperationDraft = {
          ...source,
          id: seed.id,
          inputColumns: [...source.inputColumns],
          secondaryColumns: [...source.secondaryColumns],
          constants: [...source.constants],
          datetimeFeatures: [...source.datetimeFeatures],
          outputColumn: source.outputColumn ? `${source.outputColumn}_copy` : '',
        };
        const next = [...current];
        next.splice(index + 1, 0, duplicate);
        return next;
      });
      if (createdOperationId) {
        setCollapsedFeatureMath((previous) => {
          const next = new Set(previous);
          next.delete(createdOperationId);
          return next;
        });
      }
    },
    [isFeatureMathNode, setCollapsedFeatureMath, updateFeatureMathOperations],
  );

  const handleRemoveFeatureMathOperation = useCallback(
    (operationId: string) => {
      if (!isFeatureMathNode) {
        return;
      }
      updateFeatureMathOperations((current) => current.filter((operation) => operation.id !== operationId));
      setCollapsedFeatureMath((previous) => {
        const next = new Set(previous);
        next.delete(operationId);
        return next;
      });
    },
    [isFeatureMathNode, setCollapsedFeatureMath, updateFeatureMathOperations],
  );

  const handleReorderFeatureMathOperation = useCallback(
    (operationId: string, direction: 'up' | 'down') => {
      if (!isFeatureMathNode) {
        return;
      }
      updateFeatureMathOperations((current) => {
        const index = current.findIndex((operation) => operation.id === operationId);
        if (index === -1) {
          return current;
        }
        const targetIndex = direction === 'up' ? index - 1 : index + 1;
        if (targetIndex < 0 || targetIndex >= current.length) {
          return current;
        }
        const next = [...current];
        const [moved] = next.splice(index, 1);
        next.splice(targetIndex, 0, moved);
        return next;
      });
    },
    [isFeatureMathNode, updateFeatureMathOperations],
  );

  const handleToggleFeatureMathOperation = useCallback((operationId: string) => {
    setCollapsedFeatureMath((previous) => {
      const next = new Set(previous);
      if (next.has(operationId)) {
        next.delete(operationId);
      } else {
        next.add(operationId);
      }
      return next;
    });
  }, [setCollapsedFeatureMath]);

  const handleFeatureMathOperationChange = useCallback(
    (operationId: string, updates: Partial<FeatureMathOperationDraft>) => {
      if (!isFeatureMathNode) {
        return;
      }
      updateFeatureMathOperations((current) =>
        current.map((operation) => {
          if (operation.id !== operationId) {
            return operation;
          }

          let next: FeatureMathOperationDraft = { ...operation };

          if (updates.type) {
            next.type = updates.type;
          }

          if (updates.method !== undefined) {
            next.method = updates.method;
          }

          if (updates.inputColumns !== undefined) {
            next.inputColumns = sanitizeStringList(updates.inputColumns);
          }

          if (updates.secondaryColumns !== undefined) {
            next.secondaryColumns = sanitizeStringList(updates.secondaryColumns);
          }

          if (updates.constants !== undefined) {
            next.constants = sanitizeConstantsList(updates.constants);
          }

          if (updates.outputColumn !== undefined) {
            next.outputColumn = updates.outputColumn.trim();
          }

          if (updates.outputPrefix !== undefined) {
            next.outputPrefix = updates.outputPrefix.trim();
          }

          if (updates.datetimeFeatures !== undefined) {
            next.datetimeFeatures = sanitizeDatetimeFeaturesList(updates.datetimeFeatures);
          }

          if (updates.timezone !== undefined) {
            next.timezone = sanitizeTimezoneValue(updates.timezone);
          }

          if (updates.fillna !== undefined) {
            next.fillna = sanitizeNumberValue(updates.fillna);
          }

          if (updates.roundDigits !== undefined) {
            next.roundDigits = sanitizeIntegerValue(updates.roundDigits);
          }

          if (updates.normalize !== undefined) {
            next.normalize = updates.normalize;
          }

          if (updates.epsilon !== undefined) {
            next.epsilon = sanitizeNumberValue(updates.epsilon);
          }

          if (updates.allowOverwrite !== undefined) {
            next.allowOverwrite = updates.allowOverwrite;
          }

          if (updates.description !== undefined) {
            const trimmed = typeof updates.description === 'string' ? updates.description.trim() : '';
            next.description = trimmed ? trimmed : undefined;
          }

          const resolvedType = next.type;
          const methodChoices = getMethodOptions(resolvedType);
          if (methodChoices.length) {
            const allowed = new Set(methodChoices.map((option) => option.value));
            if (!allowed.has(next.method)) {
              next.method = methodChoices[0].value;
            }
          } else if (resolvedType === 'ratio') {
            next.method = 'ratio';
          } else if (resolvedType === 'datetime_extract') {
            next.method = 'datetime_extract';
          }

          next.inputColumns = sanitizeStringList(next.inputColumns);
          if (resolvedType === 'ratio' || resolvedType === 'similarity') {
            next.secondaryColumns = sanitizeStringList(next.secondaryColumns);
          } else {
            next.secondaryColumns = [];
          }

          next.constants = sanitizeConstantsList(next.constants);
          next.fillna = sanitizeNumberValue(next.fillna);
          next.roundDigits = sanitizeIntegerValue(next.roundDigits);
          next.epsilon = sanitizeNumberValue(next.epsilon);

          if (resolvedType === 'datetime_extract') {
            next.datetimeFeatures = sanitizeDatetimeFeaturesList(next.datetimeFeatures);
            next.timezone = sanitizeTimezoneValue(next.timezone);
          } else {
            next.datetimeFeatures = [];
            next.timezone = 'UTC';
          }

          if (resolvedType !== 'similarity') {
            next.normalize = false;
          } else {
            next.normalize = Boolean(next.normalize);
          }

          if (typeof next.allowOverwrite !== 'boolean') {
            next.allowOverwrite = null;
          }

          next.outputColumn = next.outputColumn.trim();
          next.outputPrefix = next.outputPrefix.trim();

          return next;
        }),
      );
    },
    [isFeatureMathNode, updateFeatureMathOperations],
  );

  return {
    updateFeatureMathOperations,
    handleAddFeatureMathOperation,
    handleDuplicateFeatureMathOperation,
    handleRemoveFeatureMathOperation,
    handleReorderFeatureMathOperation,
    handleToggleFeatureMathOperation,
    handleFeatureMathOperationChange,
  };
};
