# Node Settings Module

This directory contains the React components, hooks, and helpers that power the feature-canvas node settings modal. The structure groups shared infrastructure from node-specific UI so future refactors can locate functionality quickly.

## Directory layout

```
node-settings/
├── formatting.ts              # Common string and formatting helpers used by multiple nodes
├── sharedUtils.ts             # Low-level utilities for config normalization and data guards
├── utils/                     # Cross-cutting helpers organised by concern
│   ├── catalogTypes.ts
│   ├── configParsers.ts
│   ├── formatters.ts
│   └── guidance.ts
├── layout/                    # Reusable chrome for the modal shell
│   ├── NodeSettingsFooter.tsx
│   └── NodeSettingsHeader.tsx
├── hooks/                     # State orchestration extracted from the modal
│   ├── useAliasConfiguration.ts
│   ├── useBinnedDistribution.ts
│   ├── useBinningConfiguration.ts
│   ├── useCatalogFlags.ts
│   ├── useDropColumnRecommendations.ts
│   ├── useImputationConfiguration.ts
│   ├── useNumericColumnAnalysis.ts
│   ├── useNumericRangeSummaries.ts
│   ├── useOutlierConfiguration.ts
│   ├── useOutlierDiagnostics.ts
│   ├── usePipelinePreview.ts
│   ├── usePruneColumnSelections.ts
│   ├── useReplaceInvalidConfiguration.ts
│   ├── useScalingConfiguration.ts
│   ├── useScalingInsights.ts
│   ├── useSkewnessConfiguration.ts
│   ├── useSkewnessInsights.ts
│   ├── useStandardizeDatesConfiguration.ts
│   └── useTextCleanupConfiguration.ts
├── nodes/                     # Node-specific UI elements grouped by node type
│   ├── binning/
│   │   ├── BinnedDistributionSection.tsx
│   │   ├── BinNumericColumnsSection.tsx
│   │   └── binningSettings.ts
│   ├── cast_column/
│   ├── dataset/
│   ├── drop_col_rows/
│   ├── imputation/
│   ├── missing_indicator/
│   ├── normalize_text/
│   ├── outlier/
│   ├── regex_node/
│   ├── remove_duplicates/
│   ├── remove_special_char/
│   ├── replace_aliases/
│   ├── replace_invalid_values/
│   ├── scaling/
│   ├── skewness/
│   ├── standardize_date/
│   └── trim_white_space/
└── layout/                    # Imported from NodeSettingsModal for modal framing
```

Each subdirectory under `nodes/` contains UI pieces, configuration models, or helper logic that apply to a specific pipeline node. The binning example above illustrates the typical pattern (settings schema + one or more sections rendered by the modal).

## Conventions

- **Hooks**: All hooks live in `hooks/` and expose a single responsibility (configuration state, insights data, etc.). They should accept plain arguments and return serialisable state slices that the modal can consume.
- **Node modules**: Keep node-specific sections and config models together. Prefer colocating React sections, TypeScript models, and helpers under a dedicated node folder.
- **Shared helpers**: Use `sharedUtils.ts`, `formatting.ts`, or `utils/` when behaviour is reused across multiple nodes. Avoid importing between node folders to prevent hidden coupling.
- **Naming**: File and folder names follow snake_case for node keys (matching backend identifiers) and PascalCase for React components.
- **Imports**: Components outside the node root should import via `./node-settings/...` so moves remain localised.

## Tips for future updates

1. When adding a new node type, mirror the existing folder pattern inside `nodes/` and register the accompanying hook or parser in `hooks/` or `utils/` as needed.
2. If multiple nodes start sharing behaviour, extract the logic into `hooks/` or `utils/` instead of cross-linking node folders.
3. Keep the README updated when moving or renaming folders so contributors can navigate the module quickly.
