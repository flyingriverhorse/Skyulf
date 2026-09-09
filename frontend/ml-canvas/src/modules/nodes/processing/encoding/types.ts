export interface EncodingConfig {
  method: 'onehot' | 'ordinal' | 'label' | 'target' | 'hash' | 'dummy' | 'woe';
  columns: string[];
  // OneHot/Dummy specific
  drop_first?: boolean | undefined;
  drop_original?: boolean | undefined;
  handle_unknown?: 'error' | 'ignore' | 'use_encoded_value' | undefined;
  // OneHot specific
  max_categories?: number | undefined;
  include_missing?: boolean | undefined;
  // Hash specific
  n_features?: number | undefined;
  // Target specific
  target_column?: string | undefined;
  smooth?: number | 'auto' | undefined;
  target_type?: 'auto' | 'continuous' | 'multiclass' | 'binary' | undefined;
  // WOE specific
  regularization?: number | undefined;
  // Label/Ordinal specific
  unknown_value?: number | undefined;
  missing_code?: number | undefined;
  // Ordinal specific: user-defined category order
  categories_order?: string | undefined;
}

export interface EncodingSettingsProps {
  config: EncodingConfig;
  onChange: (config: EncodingConfig) => void;
}
