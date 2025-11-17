import type { CSSProperties } from 'react';
import type { Node } from 'react-flow-renderer';
import type { FeatureNodeCatalogEntry } from '../../api';
import type { FeatureNodeData } from '../types/nodes';
import {
  cloneConfig,
  sanitizeDefaultConfigForNode,
  PENDING_CONFIRMATION_FLAG,
  normalizeDescription,
  normalizeParameters,
  isAutoConfirmedCatalogType,
} from './configSanitizer';
import { DEFAULT_NODE_MIN_HEIGHT, DEFAULT_NODE_WIDTH } from '../constants/defaults';
import { sanitizeSplitList, SPLIT_TYPE_ORDER } from '../constants/splits';
import { NODE_HANDLE_CONFIG, formatConnectionDescriptor } from '../constants/nodeHandles';

const RESETTABLE_NODE_TYPES = new Set([
  'binned_distribution',
  'skewness_distribution',
  'skewness_transform',
  'binning_discretization',
  'label_encoding',
  'target_encoding',
  'ordinal_encoding',
  'dummy_encoding',
  'one_hot_encoding',
  'hash_encoding',
  'scale_numeric_features',
  'polynomial_features',
  'feature_selection',
  'feature_math',
  'missing_value_indicator',
  'trim_whitespace',
  'normalize_text_case',
  'replace_aliases_typos',
  'standardize_date_formats',
  'remove_special_characters',
  'replace_invalid_values',
  'regex_replace_fix',
  'class_undersampling',
  'class_oversampling',
  'feature_target_split',
  'train_test_split',
  'train_model_draft',
  'model_evaluation',
  'hyperparameter_tuning',
]);

export const isResettableCatalogEntry = (
  catalogType?: string | null,
  catalogEntryMap?: Map<string, FeatureNodeCatalogEntry>
): boolean => {
  if (!catalogType) {
    return false;
  }
  if (RESETTABLE_NODE_TYPES.has(catalogType)) {
    return true;
  }
  const catalogEntry = catalogEntryMap?.get(catalogType);
  if (!catalogEntry) {
    return false;
  }
  const category = typeof catalogEntry.category === 'string' ? catalogEntry.category.toLowerCase() : '';
  return category.includes('preprocess');
};

export type NodeInteractionDeps = {
  handleOpenSettings: (nodeId: string) => void;
  handleRemoveNode: (nodeId: string) => void;
  catalogEntryMap: Map<string, FeatureNodeCatalogEntry>;
};

export const registerNodeInteractions = (rawNode: Node, deps: NodeInteractionDeps): Node => {
  const { handleOpenSettings, handleRemoveNode, catalogEntryMap } = deps;
  const baseData = rawNode.data ?? {};
  const isDataset = baseData.isDataset ?? rawNode.id === 'dataset-source';
  const normalizedStyle = { ...(rawNode.style ?? {}) } as CSSProperties;
  const catalogType = baseData.catalogType ?? baseData.type ?? rawNode.id;
  const rawParameterSource = Array.isArray(baseData.parameters) ? baseData.parameters : null;
  let effectiveParameterSource = rawParameterSource;

  if ((!effectiveParameterSource || effectiveParameterSource.length === 0) && typeof catalogType === 'string') {
    const fallbackEntry = catalogEntryMap.get(catalogType) ?? null;
    if (fallbackEntry?.parameters?.length) {
      effectiveParameterSource = fallbackEntry.parameters;
    }
  }

  const parameterDefs = normalizeParameters(effectiveParameterSource);

  let resolvedConfig: Record<string, any> | undefined;
  if (parameterDefs.length || (baseData.config && typeof baseData.config === 'object')) {
    const nextConfig = cloneConfig(baseData.config);
    parameterDefs.forEach((parameter) => {
      if (!parameter?.name) {
        return;
      }
      if (nextConfig[parameter.name] !== undefined) {
        return;
      }
      if (parameter.default === undefined) {
        return;
      }
      if (Array.isArray(parameter.default)) {
        nextConfig[parameter.name] = [...parameter.default];
      } else if (parameter.type === 'boolean') {
        nextConfig[parameter.name] = Boolean(parameter.default);
      } else {
        nextConfig[parameter.name] = parameter.default;
      }
    });
    resolvedConfig = nextConfig;
  }

  if (!normalizedStyle.width) {
    normalizedStyle.width = isDataset ? DEFAULT_NODE_WIDTH + 20 : DEFAULT_NODE_WIDTH;
  }

  if (!normalizedStyle.minHeight) {
    normalizedStyle.minHeight = DEFAULT_NODE_MIN_HEIGHT;
  }

  if (!normalizedStyle.padding) {
    normalizedStyle.padding = 12;
  }

  if (!normalizedStyle.borderRadius) {
    normalizedStyle.borderRadius = 16;
  }

  normalizedStyle.boxSizing = normalizedStyle.boxSizing ?? 'border-box';
  normalizedStyle.whiteSpace = normalizedStyle.whiteSpace ?? 'pre-line';

  if (!normalizedStyle.background) {
    normalizedStyle.background = isDataset
      ? 'linear-gradient(165deg, rgba(30, 64, 175, 0.9), rgba(59, 130, 246, 0.85))'
      : 'linear-gradient(160deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.92))';
  }

  if (!normalizedStyle.border) {
    normalizedStyle.border = isDataset
      ? '1px solid rgba(59, 130, 246, 0.55)'
      : '1px solid rgba(148, 163, 184, 0.22)';
  }

  if (!normalizedStyle.boxShadow) {
    normalizedStyle.boxShadow = isDataset
      ? '0 24px 54px rgba(59, 130, 246, 0.28)'
      : '0 18px 42px rgba(2, 6, 23, 0.45)';
  }

  if (!normalizedStyle.color) {
    normalizedStyle.color = isDataset ? '#e0f2fe' : '#e2e8f0';
  }

  if (!normalizedStyle.fontWeight) {
    normalizedStyle.fontWeight = isDataset ? 600 : 500;
  }

  const normalizedDescription = normalizeDescription(baseData.description, catalogType as string);
  const catalogTypeAutoConfirmed = isAutoConfirmedCatalogType(catalogType);
  const clonedConfig =
    resolvedConfig ?? (baseData.config && typeof baseData.config === 'object' ? cloneConfig(baseData.config) : undefined);
  if (clonedConfig && typeof clonedConfig === 'object' && catalogTypeAutoConfirmed) {
    delete (clonedConfig as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
  }

  const normalizedData: FeatureNodeData = {
    ...baseData,
    label: baseData.label ?? rawNode.id,
    description: normalizedDescription,
    isDataset,
    isRemovable: baseData.isRemovable ?? !isDataset,
    isConfigured: (() => {
      if (isDataset) {
        return true;
      }
      if (baseData.isConfigured === true) {
        return true;
      }
      if (baseData.config && typeof baseData.config === 'object') {
        if (catalogTypeAutoConfirmed) {
          return true;
        }
        return !(baseData.config as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
      }
      return false;
    })(),
    inputs: Array.isArray(baseData.inputs) ? [...baseData.inputs] : baseData.inputs,
    outputs: Array.isArray(baseData.outputs) ? [...baseData.outputs] : baseData.outputs,
    category: baseData.category,
    parameters: parameterDefs,
    config: clonedConfig,
    catalogType: catalogType as string,
    onRemoveNode: handleRemoveNode,
    onOpenSettings: handleOpenSettings,
  };

  const handleConfig = typeof catalogType === 'string' ? NODE_HANDLE_CONFIG[catalogType] : undefined;
  if (handleConfig) {
    const formattedInputs = handleConfig.inputs?.map((definition) => formatConnectionDescriptor(definition)) ?? [];
    const formattedOutputs = handleConfig.outputs?.map((definition) => definition.label) ?? [];

    if (formattedInputs.length) {
      normalizedData.inputs = formattedInputs;
    } else {
      delete normalizedData.inputs;
    }

    if (formattedOutputs.length) {
      normalizedData.outputs = formattedOutputs;
    } else {
      delete normalizedData.outputs;
    }

    normalizedData.connectionInfo = {
      inputs: handleConfig.inputs?.map((definition) => ({ ...definition })) ?? [],
      outputs: handleConfig.outputs?.map((definition) => ({ ...definition })) ?? [],
    };
  } else {
    if (!normalizedData.inputs || normalizedData.inputs.length === 0) {
      delete normalizedData.inputs;
    }
    if (!normalizedData.outputs || normalizedData.outputs.length === 0) {
      delete normalizedData.outputs;
    }
    delete normalizedData.connectionInfo;
  }

  const sanitizedSplits = sanitizeSplitList(normalizedData.activeSplits);
  if (sanitizedSplits.length) {
    normalizedData.activeSplits = sanitizedSplits;
  } else {
    delete normalizedData.activeSplits;
  }

  const sanitizedConnections = sanitizeSplitList(normalizedData.connectedSplits);
  if (sanitizedConnections.length) {
    normalizedData.connectedSplits = sanitizedConnections;
  } else {
    delete normalizedData.connectedSplits;
  }

  if (catalogType === 'train_test_split') {
    normalizedData.activeSplits = [...SPLIT_TYPE_ORDER];
    if (!sanitizedConnections.length) {
      delete normalizedData.connectedSplits;
    }
  }

  if (!normalizedData.parameters?.length) {
    delete normalizedData.parameters;
  }

  if (normalizedData.config === undefined) {
    delete normalizedData.config;
  }

  return {
    ...rawNode,
    type: 'featureNode',
    position: rawNode.position ?? { x: 0, y: 0 },
    data: normalizedData,
    style: normalizedStyle,
  };
};

export const createNewNode = (
  catalogNode: FeatureNodeCatalogEntry | null | undefined,
  nodeId: string,
  position: { x: number; y: number },
  fallbackLabel: string,
  deps: NodeInteractionDeps
): Node => {
  const rawLabel = catalogNode?.label ?? catalogNode?.type ?? fallbackLabel;
  const label = typeof rawLabel === 'string' && rawLabel.trim() ? rawLabel : fallbackLabel;
  const normalizedConfig = sanitizeDefaultConfigForNode(catalogNode);
  const isAutoConfirmed = isAutoConfirmedCatalogType(catalogNode?.type ?? null);

  return registerNodeInteractions(
    {
      id: nodeId,
      type: 'featureNode',
      position,
      data: {
        label,
        description: catalogNode?.description,
        inputs: catalogNode?.inputs,
        outputs: catalogNode?.outputs,
        category: catalogNode?.category,
        parameters: catalogNode?.parameters,
        config: normalizedConfig,
        catalogType: catalogNode?.type,
        isConfigured: isAutoConfirmed ? true : false,
      },
    },
    deps
  );
};
