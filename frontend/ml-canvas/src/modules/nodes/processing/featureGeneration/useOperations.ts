import { useState } from 'react';
import type React from 'react';
import { useValidationReveal } from '../../../../components/shared/ValidationField';
import type { FeatureGenerationConfig, MathOperation } from './types';

export function useOperations(config: FeatureGenerationConfig, onChange: (config: FeatureGenerationConfig) => void) {
  const [revealedOperations, setRevealedOperations] = useState<number[]>([]);
  useValidationReveal((field) => {
    const match = /^operations\.(\d+)\./.exec(field);
    if (!match) return;
    const index = Number(match[1]);
    setRevealedOperations(previous => previous.includes(index) ? previous : [...previous, index]);
  });
  const addOperation = (type: MathOperation['operation_type']) => {
    const newOp: MathOperation = {
      operation_type: type,
      method: type === 'arithmetic' ? 'add' :
              type === 'similarity' ? 'ratio' :
              type === 'group_agg' ? 'mean' : 'year',
      input_columns: [],
      output_column: '',
      datetime_features: type === 'datetime_extract' ? ['year'] : undefined,
      isExpanded: true
    };
    onChange({ operations: [...(config.operations || []), newOp] });
  };

  const updateOperation = (index: number, updates: Partial<MathOperation>) => {
    const newOps = [...(config.operations || [])];
    const existing = newOps[index];
    if (!existing) return;
    newOps[index] = { ...existing, ...updates };
    onChange({ operations: newOps });
  };

  const removeOperation = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const newOps = [...(config.operations || [])];
    newOps.splice(index, 1);
    setRevealedOperations(previous => previous.filter(i => i !== index).map(i => i > index ? i - 1 : i));
    onChange({ operations: newOps });
  };

  const toggleExpand = (index: number) => {
    const newOps = [...(config.operations || [])];
    const existing = newOps[index];
    if (!existing) return;
    newOps[index] = { ...existing, isExpanded: !(existing.isExpanded || revealedOperations.includes(index)) };
    setRevealedOperations(previous => previous.filter(i => i !== index));
    onChange({ operations: newOps });
  };

  return { revealedOperations, addOperation, updateOperation, removeOperation, toggleExpand };
}
