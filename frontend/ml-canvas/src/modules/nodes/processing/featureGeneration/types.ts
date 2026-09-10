export interface MathOperation {
  operation_type: 'arithmetic' | 'datetime_extract' | 'ratio' | 'similarity' | 'group_agg';
  method: string;
  input_columns: string[];
  secondary_columns?: string[] | undefined; // For arithmetic (second operand)
  constants?: number[] | undefined;
  output_column?: string | undefined;
  datetime_features?: string[] | undefined; // For datetime_extract

  isExpanded?: boolean | undefined; // UI state
}

export interface FeatureGenerationConfig {
  operations: MathOperation[];
}

export interface OperationEditorProps {
  op: MathOperation;
  idx: number;
  updateOperation: (index: number, updates: Partial<MathOperation>) => void;
  allColumns: string[];
  numericColumns: string[];
  dateColumns: string[];
  stringColumns: string[];
}
