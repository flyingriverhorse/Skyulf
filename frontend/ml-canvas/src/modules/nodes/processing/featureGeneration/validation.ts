import type { ValidationResult } from '../../../../core/types/nodes';
import type { FeatureGenerationConfig, MathOperation } from './types';

function validateArithmetic(op: MathOperation, index: number): ValidationResult | undefined {
  if (op.input_columns.length === 0 || (!op.secondary_columns?.length && !op.constants?.length)) {
    return { isValid: false, field: `operations.${index}.${op.input_columns.length === 0 ? 'input_columns' : 'secondary_columns'}`, message: 'Arithmetic requires two operands.' };
  }
}

function validateGroupAggregation(op: MathOperation, index: number): ValidationResult | undefined {
  if (op.input_columns.length === 0 || !op.secondary_columns?.length) {
    return { isValid: false, field: `operations.${index}.${op.input_columns.length === 0 ? 'input_columns' : 'secondary_columns'}`, message: 'Select both Group By and Target columns.' };
  }
}

function validateOperation(op: MathOperation, index: number): ValidationResult | undefined {
  if (op.operation_type === 'arithmetic') return validateArithmetic(op, index);
  if (op.operation_type === 'group_agg') return validateGroupAggregation(op, index);
  if (op.operation_type === 'datetime_extract' && op.input_columns.length === 0) {
    return { isValid: false, field: `operations.${index}.input_columns`, message: 'Select a date column.' };
  }
}

export function validateFeatureGeneration(config: FeatureGenerationConfig): ValidationResult {
  if (!config.operations?.length) {
    return { isValid: false, field: 'operations', message: 'Add at least one operation.' };
  }
  for (const [index, op] of config.operations.entries()) {
    const result = validateOperation(op, index);
    if (result) return result;
  }
  return { isValid: true };
}
