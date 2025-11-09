// Guidance text surfaced by NodeSettingsModal for data consistency nodes.
export type GuidanceEntry = {
  title: string;
  body: string;
};

export const DATA_CONSISTENCY_GUIDANCE: Record<string, GuidanceEntry> = {
  trim_whitespace: {
    title: 'Use after Dataset snapshot or Missing indicator nodes',
    body: 'Great for cleaning string columns before alias replacement or case normalization. Auto-detects text columns when left blank.',
  },
  normalize_text_case: {
    title: 'Use after trimming and alias cleanup',
    body: 'Keeps categorical values consistent for joins and grouping. Pair with Dataset profile or Preview to confirm column impacts.',
  },
  replace_aliases_typos: {
    title: 'Requires Dataset profile insights',
    body: 'Apply after profiling so you can confirm value distributions. Combine with custom pairs to align business-specific labels.',
  },
  standardize_date_formats: {
    title: 'Run after parsing-friendly nodes',
    body: 'Normalizes mixed date strings into the format you choose. Works best once whitespace and punctuation cleanup nodes have run.',
  },
  remove_special_characters: {
    title: 'Pair with Trim whitespace or Regex cleanup',
    body: 'Strips punctuation or noise from text fields before alias grouping. Preview to ensure replacements retain the intent of each field.',
  },
  replace_invalid_values: {
    title: 'Use after Outlier diagnostics or Dataset profile',
    body: 'Flags negative, zero, or out-of-range numeric placeholders and converts them to missing for downstream handling.',
  },
  regex_replace_fix: {
    title: 'Configure after reviewing Dataset snapshot',
    body: 'Use presets for quick fixes or supply a custom regex. Ideal before date standardization or alias cleanup passes.',
  },
};
