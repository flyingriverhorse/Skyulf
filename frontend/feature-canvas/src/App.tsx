// @ts-nocheck
import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import ReactFlow, {
  Background,
  Connection,
  ConnectionMode,
  Controls,
  Edge,
  Handle,
  Node,
  NodeProps,
  Position,
  ReactFlowInstance,
  addEdge,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useUpdateNodeInternals,
} from 'react-flow-renderer';
import 'react-flow-renderer/dist/style.css';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import AnimatedEdge from './components/edges/AnimatedEdge';
import ConnectionLine from './components/edges/ConnectionLine';
import { FeatureCanvasSidebar } from './components/FeatureCanvasSidebar';
import { NodeSettingsModal } from './components/NodeSettingsModal';
import {
  DatasetSourceSummary,
  FeatureNodeCatalogEntry,
  FeatureNodeParameter,
  FeaturePipelinePayload,
  FeaturePipelineResponse,
  fetchDatasets,
  fetchNodeCatalog,
  fetchPipeline,
  fetchPipelineHistory,
  savePipeline,
} from './api';
import './styles.css';

const SPLIT_DEFINITIONS = [
  { key: 'train', label: 'Train' },
  { key: 'test', label: 'Test' },
  { key: 'validation', label: 'Validation' },
] as const;

type SplitTypeKey = (typeof SPLIT_DEFINITIONS)[number]['key'];

const SPLIT_TYPE_ORDER: SplitTypeKey[] = SPLIT_DEFINITIONS.map((definition) => definition.key);

const SPLIT_LABEL_MAP: Record<SplitTypeKey, string> = SPLIT_DEFINITIONS.reduce(
  (accumulator, definition) => ({
    ...accumulator,
    [definition.key]: definition.label,
  }),
  {} as Record<SplitTypeKey, string>
);

const isValidSplitKey = (value: unknown): value is SplitTypeKey =>
  typeof value === 'string' && SPLIT_TYPE_ORDER.includes(value as SplitTypeKey);

type FeatureNodeData = {
  label?: string;
  description?: string;
  isDataset?: boolean;
  isRemovable?: boolean;
  onRemoveNode?: (nodeId: string) => void;
  onOpenSettings?: (nodeId: string) => void;
  inputs?: string[];
  outputs?: string[];
  category?: string;
  parameters?: FeatureNodeParameter[];
  config?: Record<string, any>;
  catalogType?: string;
  backgroundExecutionStatus?: 'idle' | 'loading' | 'success' | 'error';
  activeSplits?: SplitTypeKey[];
  connectedSplits?: SplitTypeKey[];
  hasRequiredConnections?: boolean;
  connectionInfo?: NodeConnectionInfo;
};

const HANDLE_BASE_SIZE = 28;
const DEFAULT_NODE_WIDTH = 320;
const DEFAULT_NODE_MIN_HEIGHT = 170;
const DROP_MISSING_LEGACY_DESCRIPTION =
  'Removes columns whose missing percentage exceeds a configurable threshold and applies EDA-backed column drop recommendations.';
const DROP_MISSING_DISPLAY_DESCRIPTION = 'Drop columns';

const cloneConfig = (value: any) => {
  if (value === undefined || value === null) {
    return {};
  }
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (error) {
    return { ...value };
  }
};

const PENDING_CONFIRMATION_FLAG = '__pending_confirmation__';

const isPlainObject = (value: any) => value && typeof value === 'object' && !Array.isArray(value);

const COLUMN_ARRAY_KEY_PATTERNS = [
  /^columns?$/i,
  /_columns$/i,
  /_features$/i,
  /_list$/i,
  /_ids$/i,
  /^selected_/i,
  /^target_/i,
  /^include_/i,
  /^exclude_/i,
  /^skipped_/i,
  /_rules$/i,
  /_transformations$/i,
  /^strategies$/i,
  /^recommendations$/i,
];

const COLUMN_MAP_KEY_PATTERNS = [/methods$/i, /overrides$/i, /mapping$/i, /weights$/i, /assignments$/i];

type ConnectionMatcherKey = 'train' | 'validation' | 'test' | 'model' | 'params';

type NodeHandleDefinition = {
  key: string;
  label: string;
  position?: number;
  required?: boolean;
  accepts?: ConnectionMatcherKey[];
};

type NodeConnectionInfo = {
  inputs?: NodeHandleDefinition[];
  outputs?: NodeHandleDefinition[];
};

const CONNECTION_ACCEPT_MATCHERS: Record<ConnectionMatcherKey, (handleId: string) => boolean> = {
  train: (handleId) => handleId.endsWith('-train'),
  validation: (handleId) => handleId.endsWith('-validation'),
  test: (handleId) => handleId.endsWith('-test'),
  model: (handleId) => handleId.endsWith('-model-out'),
  params: (handleId) => handleId.endsWith('-best-params-out'),
};

const NODE_HANDLE_CONFIG: Record<string, NodeConnectionInfo> = {
  hyperparameter_tuning: {
    inputs: [
      { key: 'train-in', label: 'Train Split', position: 35, required: true, accepts: ['train'] },
      { key: 'validation-in', label: 'Validation Split', position: 65, required: true, accepts: ['validation'] },
    ],
    outputs: [{ key: 'best-params-out', label: 'Best Params', position: 50 }],
  },
  train_model_draft: {
    inputs: [
      { key: 'train-in', label: 'Train Split', position: 40, required: true, accepts: ['train'] },
      { key: 'params-in', label: 'Best Params', position: 60, required: false, accepts: ['params'] },
    ],
    outputs: [{ key: 'model-out', label: 'Model', position: 50 }],
  },
  model_evaluation: {
    inputs: [
      { key: 'validation-in', label: 'Validation Split', position: 30, required: false, accepts: ['validation'] },
      { key: 'test-in', label: 'Test Split', position: 50, required: false, accepts: ['test'] },
      { key: 'models-in', label: 'Model', position: 70, required: true, accepts: ['model'] },
    ],
  },
  model_registry_overview: {
    inputs: [{ key: 'models-in', label: 'Model', position: 50, required: true, accepts: ['model'] }],
  },
};

const sanitizeSplitList = (value?: unknown): SplitTypeKey[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const uniqueKeys = new Set<SplitTypeKey>();
  value.forEach((entry) => {
    if (isValidSplitKey(entry)) {
      uniqueKeys.add(entry);
    }
  });

  return SPLIT_TYPE_ORDER.filter((key) => uniqueKeys.has(key));
};

const areSplitArraysEqual = (a?: SplitTypeKey[] | null, b?: SplitTypeKey[] | null): boolean => {
  const first = sanitizeSplitList(a);
  const second = sanitizeSplitList(b);

  if (first.length !== second.length) {
    return false;
  }

  return first.every((value, index) => value === second[index]);
};

const getSplitKeyFromHandle = (handleId?: string | null): SplitTypeKey | null => {
  if (!handleId || typeof handleId !== 'string') {
    return null;
  }

  return SPLIT_TYPE_ORDER.find((key) => handleId.endsWith(`-${key}`)) ?? null;
};

const getSplitHandlePosition = (index: number, total: number): string => {
  if (total <= 1) {
    return '50%';
  }

  const step = 100 / (total + 1);
  const position = Math.round((index + 1) * step);
  return `${position}%`;
};

const resolveHandleTopPosition = (definition: NodeHandleDefinition, index: number, total: number): string => {
  if (typeof definition.position === 'number' && Number.isFinite(definition.position)) {
    const clamped = Math.min(Math.max(definition.position, 5), 95);
    return `${clamped}%`;
  }
  return getSplitHandlePosition(index, total);
};

const formatHandleDisplayLabel = (definition: NodeHandleDefinition): string => {
  if (definition.required === false) {
    return `${definition.label} (optional)`;
  }
  return `${definition.label} (required)`;
};

const formatConnectionDescriptor = (definition: NodeHandleDefinition): string => {
  return definition.required === false ? `${definition.label} (optional)` : `${definition.label} (required)`;
};

const extractHandleKey = (nodeId: string, handleId?: string | null): string | null => {
  if (!handleId || typeof handleId !== 'string') {
    return null;
  }
  const prefix = `${nodeId}-`;
  if (handleId.startsWith(prefix)) {
    return handleId.slice(prefix.length);
  }
  const parts = handleId.split('-');
  if (parts.length >= 2) {
    return parts.slice(-2).join('-');
  }
  return handleId;
};

const SPLIT_PROPAGATION_BLOCKED_TYPES = new Set([
  'train_model_draft',
  'model_registry_overview',
  'model_evaluation',
  'hyperparameter_tuning',
]);

const computeActiveSplitMap = (nodes: Node[], edges: Edge[]): Map<string, SplitTypeKey[]> => {
  const assignments = new Map<string, Set<SplitTypeKey>>();
  const blockedNodeIds = new Set<string>();

  // Prepare split-enabled and blocked nodes
  nodes.forEach((node) => {
    const catalogType = node?.data?.catalogType;
    if (catalogType && SPLIT_PROPAGATION_BLOCKED_TYPES.has(catalogType)) {
      blockedNodeIds.add(node.id);
      return;
    }

    if (catalogType === 'train_test_split') {
      assignments.set(node.id, new Set(SPLIT_TYPE_ORDER));
    }
  });

  // Propagate splits through the graph using iterative approach
  let changed = true;
  let iterations = 0;
  const maxIterations = nodes.length * 3; // Allow more iterations for complex graphs

  while (changed && iterations < maxIterations) {
    changed = false;
    iterations++;

    edges.forEach((edge) => {
      if (blockedNodeIds.has(edge.source) || blockedNodeIds.has(edge.target)) {
        return;
      }

      const sourceSplits = assignments.get(edge.source);
      const splitKey = getSplitKeyFromHandle(edge.sourceHandle);
      
      if (!sourceSplits || sourceSplits.size === 0) {
        return; // Source has no splits to propagate
      }

      // Initialize target if needed
      if (!assignments.has(edge.target)) {
        assignments.set(edge.target, new Set());
      }
      
      const targetSplits = assignments.get(edge.target)!;
      const initialSize = targetSplits.size;
      
      if (splitKey) {
        // Split-specific edge: only propagate the specific split
        if (sourceSplits.has(splitKey)) {
          targetSplits.add(splitKey);
        }
      } else {
        // Regular edge: propagate ALL splits from source
        sourceSplits.forEach((split) => targetSplits.add(split));
      }
      
      if (targetSplits.size > initialSize) {
        changed = true;
      }
    });
  }

  const orderedAssignments = new Map<string, SplitTypeKey[]>();
  assignments.forEach((splitSet, nodeId) => {
    if (!splitSet.size || blockedNodeIds.has(nodeId)) {
      return;
    }

    orderedAssignments.set(
      nodeId,
      SPLIT_TYPE_ORDER.filter((key) => splitSet.has(key))
    );
  });

  return orderedAssignments;
};

const computeSplitConnectionMap = (edges: Edge[]): Map<string, SplitTypeKey[]> => {
  const connections = new Map<string, Set<SplitTypeKey>>();

  edges.forEach((edge) => {
    const splitKey = getSplitKeyFromHandle(edge.sourceHandle);
    if (!splitKey) {
      return;
    }

    if (!connections.has(edge.source)) {
      connections.set(edge.source, new Set());
    }

    connections.get(edge.source)!.add(splitKey);
  });

  const orderedConnections = new Map<string, SplitTypeKey[]>();
  connections.forEach((splitSet, nodeId) => {
    orderedConnections.set(
      nodeId,
      SPLIT_TYPE_ORDER.filter((key) => splitSet.has(key))
    );
  });

  return orderedConnections;
};

const getNodeRequiredConnections = (catalogType: string): string[] => {
  const config = catalogType ? NODE_HANDLE_CONFIG[catalogType] : undefined;
  if (!config?.inputs?.length) {
    return [];
  }
  return config.inputs
    .filter((definition) => definition.required !== false)
    .map((definition) => definition.key);
};

const checkNodeConnectionStatus = (nodeId: string, catalogType: string, edges: Edge[]): boolean => {
  const required = getNodeRequiredConnections(catalogType);
  if (required.length === 0) {
    return true; // No specific requirements
  }

  const incomingEdges = edges.filter(edge => edge.target === nodeId);
  const connectedHandles = new Set(
    incomingEdges
      .map((edge) => extractHandleKey(nodeId, edge.targetHandle))
      .filter((value): value is string => Boolean(value))
  );

  return required.every(req => connectedHandles.has(req));
};

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

const AUTO_CONFIRMED_NODE_TYPES = new Set(['transformer_audit']);

const isAutoConfirmedCatalogType = (catalogType?: string | null) =>
  typeof catalogType === 'string' && AUTO_CONFIRMED_NODE_TYPES.has(catalogType);

const sanitizeDefaultConfigForNode = (catalogNode?: FeatureNodeCatalogEntry | null) => {
  const defaultConfig = catalogNode?.default_config;
  const nodeType = catalogNode?.type ?? '';
  const requiresConfirmation = !AUTO_CONFIRMED_NODE_TYPES.has(nodeType);

  const applyPendingState = (config: Record<string, any>) => {
    if (requiresConfirmation) {
      config[PENDING_CONFIRMATION_FLAG] = true;
    } else {
      delete config[PENDING_CONFIRMATION_FLAG];
    }
    return config;
  };

  if (defaultConfig === undefined || defaultConfig === null) {
    return applyPendingState({});
  }

  const cloned = cloneConfig(defaultConfig);
  if (!isPlainObject(cloned)) {
    if (cloned && typeof cloned === 'object') {
      return applyPendingState(cloned as Record<string, any>);
    }
    return applyPendingState({});
  }

  Object.entries(cloned).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      const shouldReset = COLUMN_ARRAY_KEY_PATTERNS.some((pattern) => pattern.test(key));
      if (shouldReset) {
        cloned[key] = [];
      }
    } else if (isPlainObject(value)) {
      const shouldResetMap = COLUMN_MAP_KEY_PATTERNS.some((pattern) => pattern.test(key));
      if (shouldResetMap) {
        cloned[key] = {};
      }
    }
  });

  if ('auto_detect' in cloned) {
    cloned.auto_detect = false;
  }

  switch (catalogNode?.type) {
    case 'scale_numeric_features':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      cloned.column_methods = isPlainObject(cloned.column_methods) ? cloned.column_methods : {};
      cloned.skipped_columns = Array.isArray(cloned.skipped_columns) ? cloned.skipped_columns : [];
      cloned.default_method =
        typeof cloned.default_method === 'string' && cloned.default_method.trim() ? cloned.default_method : 'standard';
      break;
    case 'polynomial_features':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      cloned.auto_detect = Boolean(cloned.auto_detect);
      cloned.include_bias = Boolean(cloned.include_bias);
      cloned.interaction_only = Boolean(cloned.interaction_only);
      cloned.include_input_features = Boolean(cloned.include_input_features);
      cloned.degree = typeof cloned.degree === 'number' && Number.isFinite(cloned.degree) ? cloned.degree : 2;
      cloned.output_prefix =
        typeof cloned.output_prefix === 'string' && cloned.output_prefix.trim()
          ? cloned.output_prefix
          : 'poly';
      break;
    case 'drop_missing_columns':
      if ('missing_threshold' in cloned) {
        cloned.missing_threshold = null;
      }
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      break;
    case 'drop_missing_rows':
      if ('missing_threshold' in cloned) {
        cloned.missing_threshold = null;
      }
      if ('drop_if_any_missing' in cloned) {
        cloned.drop_if_any_missing = false;
      }
      break;
    case 'outlier_removal':
      cloned.method = 'manual';
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      cloned.manual_bounds = isPlainObject(cloned.manual_bounds) ? cloned.manual_bounds : {};
      break;
    case 'missing_value_indicator':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      break;
    case 'remove_duplicates':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      if (typeof cloned.keep === 'string') {
        const normalized = cloned.keep.trim().toLowerCase();
        cloned.keep = normalized === 'last' || normalized === 'none' ? normalized : 'first';
      } else {
        cloned.keep = 'first';
      }
      break;
    default:
      break;
  }

  return applyPendingState(cloned as Record<string, any>);
};

const buildHandleStyle = (type: 'target' | 'source'): React.CSSProperties => ({
  width: `${HANDLE_BASE_SIZE}px`,
  height: `${HANDLE_BASE_SIZE}px`,
  borderRadius: '999px',
  border: type === 'target' ? '1px solid rgba(148, 163, 184, 0.55)' : '1px solid rgba(45, 212, 191, 0.55)',
  background: type === 'target' ? 'rgba(148, 163, 184, 0.22)' : 'rgba(45, 212, 191, 0.18)',
  boxShadow: type === 'target' ? '0 0 0 3px rgba(148, 163, 184, 0.18)' : '0 0 0 3px rgba(45, 212, 191, 0.18)',
  backdropFilter: 'blur(2px)',
});

type HandleConfig = {
  position: Position;
  id: string;
  style?: React.CSSProperties;
  label?: string;
};

const FeatureCanvasNode: React.FC<NodeProps<FeatureNodeData>> = ({ id, data, selected }) => {
  const label = data?.label ?? id;
  const isDataset = Boolean(data?.isDataset ?? id === 'dataset-source');
  const isRemovable = data?.isRemovable ?? !isDataset;
  const isSplitNode = data?.catalogType === 'train_test_split';
  const defaultDescription = isDataset ? 'Primary dataset input' : undefined;
  const rawDescription = data?.description ?? defaultDescription;
  const shouldDisplayDropMissingShort =
    data?.catalogType === 'drop_missing_columns' || rawDescription === DROP_MISSING_LEGACY_DESCRIPTION;
  const description = shouldDisplayDropMissingShort ? DROP_MISSING_DISPLAY_DESCRIPTION : rawDescription;
  const backgroundStatus = data?.backgroundExecutionStatus ?? 'idle';
  const hasRequiredConnections = data?.hasRequiredConnections ?? true;
  const needsConnection = !hasRequiredConnections;
  const catalogType = data?.catalogType ?? '';
  const isModelingNode = ['hyperparameter_tuning', 'train_model_draft', 'model_evaluation', 'model_registry_overview'].includes(catalogType);

  const handleRemove = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation();
      data?.onRemoveNode?.(id);
    },
    [data, id]
  );

  const targetHandles = useMemo(() => {
    if (isDataset) {
      return [];
    }

    const catalogType = data?.catalogType ?? '';
    const handleConfig = catalogType ? NODE_HANDLE_CONFIG[catalogType] : undefined;

    if (handleConfig?.inputs && handleConfig.inputs.length) {
      const total = handleConfig.inputs.length;
      return handleConfig.inputs.map((definition, index) => ({
        position: Position.Left,
        id: `${id}-${definition.key}`,
        label: formatHandleDisplayLabel(definition),
        style: { top: resolveHandleTopPosition(definition, index, total) },
      }));
    }

    return [
      { position: Position.Left, id: `${id}-target` },
      { position: Position.Top, id: `${id}-target-top` },
    ];
  }, [data?.catalogType, id, isDataset]);

  const sourceHandles = useMemo(() => {
    if (isDataset) {
      return [
        { position: Position.Left, id: `${id}-source-left` },
        { position: Position.Right, id: `${id}-source` },
        { position: Position.Top, id: `${id}-source-top` },
        { position: Position.Bottom, id: `${id}-source-bottom` },
      ];
    }

    const catalogType = data?.catalogType ?? '';
    const handleConfig = catalogType ? NODE_HANDLE_CONFIG[catalogType] : undefined;

    if (handleConfig?.outputs && handleConfig.outputs.length) {
      const total = handleConfig.outputs.length;
      return handleConfig.outputs.map((definition, index) => ({
        position: Position.Right,
        id: `${id}-${definition.key}`,
        label: definition.label,
        style: { top: resolveHandleTopPosition(definition, index, total) },
      }));
    }

    const activeSplits = isSplitNode ? SPLIT_TYPE_ORDER : sanitizeSplitList(data?.activeSplits ?? []);
    const connectedSplits = sanitizeSplitList(data?.connectedSplits ?? []);

    if (activeSplits.length) {
      const total = activeSplits.length;
      return activeSplits.map((splitKey, index) => ({
        position: Position.Right,
        id: `${id}-${splitKey}`,
        style: { top: getSplitHandlePosition(index, total) },
        label: (() => {
          const baseLabel = SPLIT_LABEL_MAP[splitKey];
          if (isSplitNode) {
            if (splitKey === 'validation') {
              const validationSize = Number(data?.config?.validation_size ?? 0);
              if (!(validationSize > 0)) {
                return `${baseLabel} (disabled)`;
              }
            }
            if (!connectedSplits.includes(splitKey)) {
              return `${baseLabel} (not set)`;
            }
          }
          return baseLabel;
        })(),
      }));
    }

    return [
      { position: Position.Right, id: `${id}-source` },
      { position: Position.Bottom, id: `${id}-source-bottom` },
    ];
  }, [data?.activeSplits, data?.catalogType, data?.connectedSplits, data?.config?.validation_size, id, isDataset, isSplitNode]);

  const renderHandle = useCallback((handleConfig: HandleConfig, type: 'target' | 'source') => {
    const { position, id: handleId, style, label } = handleConfig;
    const isSplitHandle = handleId.includes('-train') || handleId.includes('-test') || handleId.includes('-validation');
    
    return (
      <React.Fragment key={handleId}>
        <Handle
          type={type}
          position={position}
          id={handleId}
          className={`feature-node__handle feature-node__handle--${type}`}
          style={{ 
            ...buildHandleStyle(type), 
            ...(style ?? {}),
            // Ensure split handles are visible
            ...(isSplitHandle ? {
              opacity: 1,
              visibility: 'visible',
              pointerEvents: 'all',
            } : {})
          }}
        />
        {label && type === 'source' && (
          <div 
            className="feature-node__handle-label"
            style={{ 
              position: 'absolute',
              right: '-8px',
              top: style?.top || '50%',
              transform: 'translate(100%, calc(-100% - 8px))', // Position above the handle consistently
              fontSize: '0.7rem',
              fontWeight: '600',
              color: 'rgba(148, 163, 184, 0.9)',
              background: 'rgba(15, 23, 42, 0.95)',
              padding: '3px 8px',
              borderRadius: '4px',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              zIndex: 100,
              border: '1px solid rgba(71, 85, 105, 0.5)',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
            }}
          >
            {label}
          </div>
        )}
        {label && type === 'target' && (
          <div 
            className="feature-node__handle-label"
            style={{ 
              position: 'absolute',
              left: '-8px',
              top: style?.top || '50%',
              transform: 'translate(-100%, calc(-100% - 8px))', // Position above the handle on the left
              fontSize: '0.7rem',
              fontWeight: '600',
              color: 'rgba(148, 163, 184, 0.9)',
              background: 'rgba(15, 23, 42, 0.95)',
              padding: '3px 8px',
              borderRadius: '4px',
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              zIndex: 100,
              border: '1px solid rgba(71, 85, 105, 0.5)',
              boxShadow: '0 2px 4px rgba(0, 0, 0, 0.2)',
            }}
          >
            {label}
          </div>
        )}
      </React.Fragment>
    );
  }, []);

  return (
    <div
      className={`feature-node${selected ? ' feature-node--selected' : ''}${
        isDataset ? ' feature-node--dataset' : ''
      }`}
    >
  {targetHandles.map((handle) => renderHandle(handle, 'target'))}
      <div className="feature-node__header">
        <div className="feature-node__title-group">
          <span className="feature-node__drag-handle" aria-hidden="true">
            ⠿
          </span>
          <span className="feature-node__title">{label}</span>
        </div>
        <div className="feature-node__controls">
          {needsConnection && isModelingNode && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--warning"
              title="Required connections missing"
              style={{
                background: 'rgba(251, 191, 36, 0.2)',
                color: 'rgb(245, 158, 11)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
              }}
            >
              ⚠
            </span>
          )}
          {backgroundStatus === 'loading' && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--loading"
              title="Loading full dataset in background..."
            >
              <svg className="feature-node__spinner" viewBox="0 0 24 24">
                <circle
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="3"
                  fill="none"
                  strokeDasharray="60"
                  strokeDashoffset="30"
                  strokeLinecap="round"
                />
              </svg>
            </span>
          )}
          {backgroundStatus === 'success' && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--success"
              title="Full dataset ready"
            >
              ✓
            </span>
          )}
          {backgroundStatus === 'error' && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--error"
              title="Background execution failed"
            >
              !
            </span>
          )}
          {isRemovable && (
            <button
              className="feature-node__control feature-node__control--danger"
              type="button"
              onClick={handleRemove}
              title="Remove node"
            >
              ×
            </button>
          )}
        </div>
      </div>
      {description && <p className="feature-node__description">{description}</p>}
      {sourceHandles.map((handle) => renderHandle(handle, 'source'))}
    </div>
  );
};

const initialNodes: Node[] = [
  {
    id: 'dataset-source',
    type: 'featureNode',
    position: { x: -200, y: 0 },
    data: {
      label: 'Dataset input',
      description: 'Start from the currently selected dataset',
      isDataset: true,
      isRemovable: false,
      isConfigured: true,
    },
    style: {
      background: 'linear-gradient(165deg, rgba(30, 64, 175, 0.9), rgba(59, 130, 246, 0.85))',
      border: '1px solid rgba(59, 130, 246, 0.55)',
      color: '#e0f2fe',
      fontWeight: 600,
    },
  },
];

const initialEdges: Edge[] = [];

const cloneNode = (node: Node): Node => ({
  ...node,
  data: {
    ...(node.data ?? {}),
  },
});

const getDefaultNodes = () => initialNodes.map((node) => cloneNode(node));

const getDefaultEdges = () => initialEdges.map((edge) => ({ ...edge }));

type PipelineHydrationContext = 'sample' | 'stored' | 'reset';

interface PipelineHydrationPayload {
  nodes: Node[];
  edges: Edge[];
  pipeline?: FeaturePipelineResponse | null;
  context: PipelineHydrationContext;
}

const HISTORY_LIMIT = 12;

const getSamplePipelineGraph = (datasetLabel?: string | null) => {
  const displayName = datasetLabel && datasetLabel.trim() ? datasetLabel.trim() : 'Demo dataset';
  const datasetNodeLabel = `Dataset input\n(${displayName})`;

  const nodes: Node[] = [
    {
      id: 'dataset-source',
      type: 'featureNode',
      position: { x: -200, y: 40 },
      data: {
        label: datasetNodeLabel,
        isDataset: true,
        isRemovable: false,
      },
      style: {
        whiteSpace: 'pre-line',
        background: '#eef2ff',
        border: '1px solid rgba(99, 102, 241, 0.35)',
        color: '#312e81',
        fontWeight: 600,
      },
    },
    {
      id: 'node-1',
      type: 'featureNode',
      position: { x: 260, y: 40 },
      data: {
        label: 'Profile columns\n(assess types & nulls)',
      },
      style: {
        whiteSpace: 'pre-line',
        background: 'linear-gradient(165deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.85))',
      },
    },
    {
      id: 'node-4',
      type: 'featureNode',
      position: { x: 600, y: 60 },
      data: {
        label: 'Feature preview\n(sample rows)',
      },
      style: {
        whiteSpace: 'pre-line',
        background: 'linear-gradient(165deg, rgba(8, 145, 178, 0.42), rgba(2, 132, 199, 0.32))',
      },
    },
  ];

  const edges: Edge[] = [
    {
      id: 'edge-dataset-profile',
      source: 'dataset-source',
      target: 'node-1',
      animated: true,
      type: 'animatedEdge',
      sourceHandle: 'dataset-source-source',
      targetHandle: 'node-1-target',
    },
    {
      id: 'edge-profile-preview',
      source: 'node-1',
      target: 'node-4',
      animated: true,
      type: 'animatedEdge',
      sourceHandle: 'node-1-source',
      targetHandle: 'node-4-target',
    },
  ];

  return { nodes, edges };
};

const parseServerTimestamp = (value?: string | null): Date | null => {
  if (!value) {
    return null;
  }

  let normalized = value.trim();
  if (!normalized) {
    return null;
  }

  if (/^\d{4}-\d{2}-\d{2}\s+\d/.test(normalized)) {
    normalized = normalized.replace(' ', 'T');
  }

  if (!/(Z|z|[+-]\d{2}:?\d{2})$/.test(normalized)) {
    normalized = `${normalized}Z`;
  }

  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return parsed;
};

const formatTimestamp = (value?: string | null) => {
  const date = parseServerTimestamp(value);
  if (!date) {
    return 'Unknown time';
  }

  return date.toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  });
};

const formatRelativeTime = (value?: string | null) => {
  const parsed = parseServerTimestamp(value);
  if (!parsed) {
    return null;
  }

  const timestamp = parsed.getTime();

  const diffMs = Date.now() - timestamp;
  const diffMinutes = Math.round(diffMs / 60000);

  if (diffMinutes < 1) {
    return 'just now';
  }
  if (diffMinutes === 1) {
    return '1 minute ago';
  }
  if (diffMinutes < 60) {
    return `${diffMinutes} minutes ago`;
  }

  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours === 1) {
    return '1 hour ago';
  }
  if (diffHours < 24) {
    return `${diffHours} hours ago`;
  }

  const diffDays = Math.round(diffHours / 24);
  if (diffDays === 1) {
    return '1 day ago';
  }
  if (diffDays < 7) {
    return `${diffDays} days ago`;
  }

  const diffWeeks = Math.round(diffDays / 7);
  if (diffWeeks === 1) {
    return '1 week ago';
  }

  return `${diffWeeks} weeks ago`;
};

interface CanvasShellProps {
  sourceId?: string | null;
  datasetName?: string | null;
  onGraphChange?: (nodes: Node[], edges: Edge[]) => void;
  onPipelineHydrated?: (payload: PipelineHydrationPayload) => void;
  onPipelineError?: (error: Error) => void;
}

type CanvasShellHandle = {
  openCatalog: () => void;
  closeCatalog: () => void;
  clearGraph: () => void;
};

const CanvasShell = forwardRef<CanvasShellHandle, CanvasShellProps>(({ sourceId, datasetName, onGraphChange, onPipelineHydrated, onPipelineError }, ref) => {
  const nodeTypes = useMemo(() => ({ featureNode: FeatureCanvasNode }), []);
  const edgeTypes = useMemo(() => ({ animatedEdge: AnimatedEdge }), []);
  const [nodes, setNodes, onNodesChange] = useNodesState(
    getDefaultNodes().map((node) => ({
      ...node,
      type: 'featureNode',
    }))
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(getDefaultEdges());
  const hasInitialSampleHydratedRef = useRef(false);
  const [isCatalogOpen, setIsCatalogOpen] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const nodeIdRef = useRef(0);
  const reactFlowInstanceRef = useRef<ReactFlowInstance | null>(null);
  const canvasViewportRef = useRef<HTMLDivElement | null>(null);
  const shouldFitViewRef = useRef(false);
  const catalogEntryMapRef = useRef<Map<string, FeatureNodeCatalogEntry>>(new Map());
  const datasetDisplayLabel = datasetName ?? sourceId ?? 'Demo dataset';
  const updateNodeInternals = useUpdateNodeInternals();

  const scheduleNodeInternalsUpdate = useCallback((nodeIds: string | string[]) => {
    const ids = (Array.isArray(nodeIds) ? nodeIds : [nodeIds]).filter(Boolean);
    if (!ids.length) {
      return;
    }

    const uniqueIds = Array.from(new Set(ids));
    console.debug('📐 Scheduling node internals update for:', uniqueIds);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        uniqueIds.forEach((nodeId) => {
          updateNodeInternals(nodeId);
        });
      });
    });
  }, [updateNodeInternals]);

  const updateNodeCounter = useCallback((list: Node[]) => {
    const highestId = list.reduce((max, node) => {
      if (typeof node.id === 'string' && node.id.startsWith('node-')) {
        const parsed = Number(node.id.replace('node-', ''));
        if (!Number.isNaN(parsed)) {
          return Math.max(max, parsed);
        }
      }
      return max;
    }, 0);

    if (highestId > nodeIdRef.current) {
      nodeIdRef.current = highestId;
    }
  }, []);

  useEffect(() => {
    updateNodeCounter(nodes);
  }, [nodes, updateNodeCounter]);

  useEffect(() => {
    const nodesWithHandleChanges: string[] = [];

    setNodes((currentNodes) => {
      const activeSplitMap = computeActiveSplitMap(currentNodes, edges);
      const splitConnectionMap = computeSplitConnectionMap(edges);
      let didChange = false;

      const nextNodes = currentNodes.map((node) => {
        const isSplitNodeType = node?.data?.catalogType === 'train_test_split';
        const desiredSplits = isSplitNodeType
          ? [...SPLIT_TYPE_ORDER]
          : activeSplitMap.get(node.id) ?? [];
        const currentSplits = sanitizeSplitList(node.data?.activeSplits);
        const desiredConnections = sanitizeSplitList(splitConnectionMap.get(node.id));
        const currentConnections = sanitizeSplitList(node.data?.connectedSplits);

        // Check if node has required connections
        const catalogType = node?.data?.catalogType ?? '';
        const hasRequiredConnections = checkNodeConnectionStatus(node.id, catalogType, edges);
        const currentHasRequiredConnections = node.data?.hasRequiredConnections ?? true;

        const splitsChanged = !areSplitArraysEqual(currentSplits, desiredSplits);
        const connectionsChanged = !areSplitArraysEqual(currentConnections, desiredConnections);
        const connectionStatusChanged = hasRequiredConnections !== currentHasRequiredConnections;

        if (!splitsChanged && !connectionsChanged && !connectionStatusChanged) {
          return node;
        }

        const nextData = {
          ...node.data,
          hasRequiredConnections,
        } as FeatureNodeData;

        if (desiredSplits.length) {
          nextData.activeSplits = desiredSplits;
        } else {
          delete nextData.activeSplits;
        }

        if (desiredConnections.length) {
          nextData.connectedSplits = desiredConnections;
        } else {
          delete nextData.connectedSplits;
        }

        didChange = true;
        if (splitsChanged || connectionsChanged) {
          nodesWithHandleChanges.push(node.id);
        }

        return {
          ...node,
          data: nextData,
        };
      });

      return didChange ? nextNodes : currentNodes;
    });

    if (nodesWithHandleChanges.length) {
      scheduleNodeInternalsUpdate(nodesWithHandleChanges);
    }
  }, [edges, scheduleNodeInternalsUpdate, setNodes]);

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId]
  );

  const graphSnapshot = useMemo(() => {
    const sanitizedNodes = nodes.map((node) => {
      const { data, ...rest } = node;
      const sanitizedData = data ? { ...data } : undefined;
      if (sanitizedData) {
        delete sanitizedData.onRemoveNode;
        delete sanitizedData.onOpenSettings;
      }
      return JSON.parse(
        JSON.stringify({
          ...rest,
          data: sanitizedData,
        })
      );
    });

    const sanitizedEdges = edges.map((edge) => JSON.parse(JSON.stringify(edge)));

    return {
      nodes: sanitizedNodes,
      edges: sanitizedEdges,
    };
  }, [edges, nodes]);

  const handleOpenSettings = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    setIsSettingsModalOpen(true);
  }, []);

  const handleCloseSettings = useCallback(() => {
    setIsSettingsModalOpen(false);
    setSelectedNodeId(null);
  }, []);

  const scheduleFitView = useCallback(() => {
    shouldFitViewRef.current = true;
  }, []);

  useEffect(() => {
    if (!shouldFitViewRef.current) {
      return;
    }
    shouldFitViewRef.current = false;
    requestAnimationFrame(() => {
      reactFlowInstanceRef.current?.fitView({ padding: 0.25, duration: 350 });
    });
  }, [edges, nodes]);

  const handleRemoveNode = useCallback(
    (nodeId: string) => {
      if (nodeId === 'dataset-source') {
        return;
      }

      setNodes((current) => current.filter((node) => node.id !== nodeId));
      setEdges((current) => current.filter((edge) => edge.source !== nodeId && edge.target !== nodeId));
      setSelectedNodeId((currentSelected) => {
        if (currentSelected === nodeId) {
          setIsSettingsModalOpen(false);
          return null;
        }
        return currentSelected;
      });
    },
    [setEdges, setIsSettingsModalOpen, setNodes]
  );

  const registerNodeInteractions = useCallback(
    (rawNode: Node): Node => {
      const baseData = rawNode.data ?? {};
      const isDataset = baseData.isDataset ?? rawNode.id === 'dataset-source';
      const normalizedStyle = { ...(rawNode.style ?? {}) } as React.CSSProperties;
      const catalogType = baseData.catalogType ?? baseData.type ?? rawNode.id;
      const rawParameterSource = Array.isArray(baseData.parameters) ? baseData.parameters : null;
      let effectiveParameterSource: FeatureNodeParameter[] | null = rawParameterSource;

      if ((!effectiveParameterSource || effectiveParameterSource.length === 0) && typeof catalogType === 'string') {
        const fallbackEntry = catalogEntryMapRef.current.get(catalogType) ?? null;
        if (fallbackEntry?.parameters?.length) {
          effectiveParameterSource = fallbackEntry.parameters;
        }
      }

      const parameterDefs: FeatureNodeParameter[] = Array.isArray(effectiveParameterSource)
        ? effectiveParameterSource.map((param: FeatureNodeParameter) => ({ ...param }))
        : [];

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

      const normalizedDescription =
        catalogType === 'drop_missing_columns' || baseData.description === DROP_MISSING_LEGACY_DESCRIPTION
          ? DROP_MISSING_DISPLAY_DESCRIPTION
          : baseData.description;
      const catalogTypeAutoConfirmed = isAutoConfirmedCatalogType(catalogType);
      const clonedConfig =
        resolvedConfig ??
        (baseData.config && typeof baseData.config === 'object' ? cloneConfig(baseData.config) : undefined);
      if (clonedConfig && typeof clonedConfig === 'object' && catalogTypeAutoConfirmed) {
        delete (clonedConfig as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
      }

      const normalizedData: Record<string, any> = {
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
        catalogType,
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
    },
    [handleOpenSettings, handleRemoveNode]
  );

  const clearGraph = useCallback(() => {
    setNodes((current) => {
      const existingDataset = current.find((node) => node.id === 'dataset-source');

      const datasetNode = registerNodeInteractions(
        existingDataset
          ? {
              ...existingDataset,
              data: {
                ...(existingDataset.data ?? {}),
                label: `Dataset input\n(${datasetDisplayLabel})`,
                isDataset: true,
                isRemovable: false,
                isConfigured: true,
              },
            }
          : {
              id: 'dataset-source',
              type: 'featureNode',
              position: { x: -200, y: 40 },
              data: {
                label: `Dataset input\n(${datasetDisplayLabel})`,
                isDataset: true,
                isRemovable: false,
                isConfigured: true,
              },
            }
      );

      return [datasetNode];
    });

    setEdges([]);
    setSelectedNodeId(null);
    setIsSettingsModalOpen(false);
    scheduleFitView();
  }, [
    datasetDisplayLabel,
    registerNodeInteractions,
    scheduleFitView,
    setEdges,
    setNodes,
    setSelectedNodeId,
    setIsSettingsModalOpen,
  ]);

  useImperativeHandle(
    ref,
    () => ({
      openCatalog: () => setIsCatalogOpen(true),
      closeCatalog: () => setIsCatalogOpen(false),
      clearGraph,
    }),
    [clearGraph]
  );

  const prepareNodes = useCallback(
    (rawNodes: Node[]) =>
      rawNodes.map((node) => {
        const baseData = {
          ...(node.data ?? {}),
        };
        if (node.id === 'dataset-source') {
          baseData.label = `Dataset input\n(${datasetDisplayLabel})`;
          baseData.isDataset = true;
          baseData.isRemovable = false;
          baseData.isConfigured = true;
        }
        const catalogType = baseData.catalogType ?? baseData.type ?? node.id;
        if (baseData.config && typeof baseData.config === 'object' && isAutoConfirmedCatalogType(catalogType)) {
          delete (baseData.config as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
        }
        if (baseData.isConfigured === undefined) {
          if (baseData.config && typeof baseData.config === 'object') {
            baseData.isConfigured = isAutoConfirmedCatalogType(catalogType)
              ? true
              : !(baseData.config as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
          } else {
            baseData.isConfigured = true;
          }
        }
        return registerNodeInteractions({
          ...node,
          data: baseData,
        });
      }),
    [datasetDisplayLabel, registerNodeInteractions]
  );

  const createNewNode = useCallback(
    (catalogNode?: FeatureNodeCatalogEntry | null, position?: { x: number; y: number }) => {
      const nextNumericId = nodeIdRef.current + 1;
      const nodeId = `node-${nextNumericId}`;
      nodeIdRef.current = nextNumericId;

      const basePosition =
        position ?? {
          x: 160 + ((nextNumericId - 1) % 4) * 220,
          y: 40 + Math.floor((nextNumericId - 1) / 4) * 160,
        };

      const fallbackLabel = `Step ${nextNumericId}`;
      const rawLabel = catalogNode?.label ?? catalogNode?.type ?? fallbackLabel;
      const label = typeof rawLabel === 'string' && rawLabel.trim() ? rawLabel : fallbackLabel;

      const normalizedConfig = sanitizeDefaultConfigForNode(catalogNode);
      const isAutoConfirmed = isAutoConfirmedCatalogType(catalogNode?.type ?? null);

      return registerNodeInteractions({
        id: nodeId,
        type: 'featureNode',
        position: basePosition,
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
      });
    },
    [registerNodeInteractions]
  );

  const resolveDropPosition = useCallback((currentNodes: Node[]): { x: number; y: number } | undefined => {
    const instance = reactFlowInstanceRef.current;
    const viewportEl = canvasViewportRef.current;
    if (!instance || typeof instance.project !== 'function' || !viewportEl) {
      return undefined;
    }

    const bounds = viewportEl.getBoundingClientRect();
    const projectedCenter = instance.project({ x: bounds.width / 2, y: bounds.height / 2 });
    if (!projectedCenter) {
      return undefined;
    }

    const nonDatasetCount = currentNodes.filter((node) => node.id !== 'dataset-source').length;
    const pattern = [
      { x: 0, y: 0 },
      { x: 220, y: 0 },
      { x: -220, y: 0 },
      { x: 0, y: 180 },
      { x: 220, y: 180 },
      { x: -220, y: 180 },
      { x: 0, y: -180 },
      { x: 220, y: -180 },
      { x: -220, y: -180 },
    ];

    const patternIndex = nonDatasetCount % pattern.length;
    const ring = Math.floor(nonDatasetCount / pattern.length);
    const offset = pattern[patternIndex];
    const scale = ring + 1;

    return {
      x: projectedCenter.x + offset.x * scale,
      y: projectedCenter.y + offset.y * scale,
    };
  }, []);

  const nodeCatalogQuery = useQuery({
    queryKey: ['feature-canvas', 'node-catalog'],
    queryFn: fetchNodeCatalog,
    staleTime: 5 * 60 * 1000,
  });

  const nodeCatalog = nodeCatalogQuery.data ?? [];
  const isCatalogLoading = nodeCatalogQuery.isLoading || nodeCatalogQuery.isFetching;
  const catalogErrorMessage = nodeCatalogQuery.error
    ? (nodeCatalogQuery.error as Error)?.message ?? 'Unable to load node catalog'
    : null;
  const catalogEntryMap = useMemo(() => {
    const map = new Map<string, FeatureNodeCatalogEntry>();
    nodeCatalog.forEach((entry) => {
      if (entry && typeof entry.type === 'string' && entry.type.trim()) {
        map.set(entry.type, entry);
      }
    });
    return map;
  }, [nodeCatalog]);

  useEffect(() => {
    catalogEntryMapRef.current = catalogEntryMap;
  }, [catalogEntryMap]);

  const isResettableEntry = useCallback(
    (catalogType?: string | null) => {
      if (!catalogType) {
        return false;
      }
      if (RESETTABLE_NODE_TYPES.has(catalogType)) {
        return true;
      }
      const catalogEntry = catalogEntryMap.get(catalogType);
      if (!catalogEntry) {
        return false;
      }
      const category = typeof catalogEntry.category === 'string' ? catalogEntry.category.toLowerCase() : '';
      return category.includes('preprocess');
    },
    [catalogEntryMap]
  );

  const shouldResetNode = useCallback(
    (node: Node) => {
      if (!node || node.id === 'dataset-source') {
        return false;
      }
      const catalogType = node?.data?.catalogType;
      return isResettableEntry(typeof catalogType === 'string' ? catalogType : null);
    },
    [isResettableEntry]
  );

  const hasResettableNodes = useMemo(() => nodes.some((node) => shouldResetNode(node)), [nodes, shouldResetNode]);

  const pipelineQuery = useQuery({
    queryKey: ['feature-canvas', 'pipeline', sourceId],
    queryFn: () => fetchPipeline(sourceId as string),
    enabled: Boolean(sourceId),
    staleTime: 60 * 1000,
    retry: 1,
  });

  const isPipelineLoading = pipelineQuery.isLoading || pipelineQuery.isFetching;

  const handleAddNode = useCallback(
    (catalogNode: FeatureNodeCatalogEntry) => {
      setNodes((current) => {
        const dropPosition = resolveDropPosition(current);
        const newNode = createNewNode(catalogNode, dropPosition);
        return [...current, newNode];
      });
      scheduleFitView();
    },
    [createNewNode, resolveDropPosition, scheduleFitView, setNodes]
  );

  const handleUpdateNodeConfig = useCallback(
    (nodeId: string, nextConfig: Record<string, any>) => {
      setNodes((current) =>
        current.map((node) => {
          if (node.id !== nodeId) {
            return node;
          }

          const sanitizedConfig = cloneConfig(nextConfig);
          if (sanitizedConfig && typeof sanitizedConfig === 'object') {
            delete (sanitizedConfig as Record<string, any>)[PENDING_CONFIRMATION_FLAG];
          }

          const baseData = {
            ...(node.data ?? {}),
            config: sanitizedConfig,
            isConfigured: true,
          };

          return registerNodeInteractions({
            ...node,
            data: baseData,
          });
        })
      );
    },
    [registerNodeInteractions, setNodes]
  );

  const handleUpdateNodeData = useCallback(
    (nodeId: string, dataUpdates: Partial<FeatureNodeData>) => {
      setNodes((current) =>
        current.map((node) => {
          if (node.id !== nodeId) {
            return node;
          }

          const baseData = {
            ...(node.data ?? {}),
            ...dataUpdates,
          };

          return registerNodeInteractions({
            ...node,
            data: baseData,
          });
        })
      );
    },
    [registerNodeInteractions, setNodes]
  );

  const handleResetNodeConfig = useCallback(
    (nodeId: string, template?: Record<string, any> | null) => {
      setNodes((current) =>
        current.map((node) => {
          if (node.id !== nodeId) {
            return node;
          }

          const catalogType = node?.data?.catalogType;
          const catalogEntry = catalogType ? catalogEntryMap.get(catalogType) ?? null : null;
          const resolvedTemplate = template && typeof template === 'object' ? cloneConfig(template) : sanitizeDefaultConfigForNode(catalogEntry ?? null);

          const baseData = {
            ...(node.data ?? {}),
            config: resolvedTemplate,
            isConfigured: false,
            backgroundExecutionStatus: 'idle', // Reset background execution status
          };

          return registerNodeInteractions({
            ...node,
            data: baseData,
          });
        })
      );
    },
    [catalogEntryMap, registerNodeInteractions, setNodes]
  );

  const handleResetAllNodes = useCallback(() => {
    if (!hasResettableNodes) {
      return;
    }

    if (typeof window !== 'undefined') {
      const confirmed = window.confirm('Reset all preprocessing nodes to their default settings?');
      if (!confirmed) {
        return;
      }
    }

    setNodes((current) =>
      current.map((node) => {
        if (!shouldResetNode(node)) {
          return node;
        }

        const catalogType = node?.data?.catalogType;
        const catalogEntry = catalogType ? catalogEntryMap.get(catalogType) ?? null : null;
        const sanitizedConfig = sanitizeDefaultConfigForNode(catalogEntry ?? null);

        const baseData = {
          ...(node.data ?? {}),
          config: sanitizedConfig,
          isConfigured: false,
          backgroundExecutionStatus: 'idle', // Reset background execution status
        };

        return registerNodeInteractions({
          ...node,
          data: baseData,
        });
      })
    );
  }, [catalogEntryMap, hasResettableNodes, registerNodeInteractions, setNodes, shouldResetNode]);

  const isValidConnection = useCallback(
    (connection: Connection) => {
      const { source, target, sourceHandle, targetHandle } = connection;

      if (!source || !target || !targetHandle) {
        return false;
      }

      if (source === target) {
        return false;
      }

      const targetNode = nodes.find((node) => node.id === target);
      if (!targetNode) {
        return false;
      }

      const targetCatalogType = targetNode?.data?.catalogType ?? '';
      const handleConfig = targetCatalogType ? NODE_HANDLE_CONFIG[targetCatalogType] : undefined;

      if (handleConfig?.inputs?.length) {
        const handleKey = extractHandleKey(target, targetHandle);
        if (!handleKey) {
          return false;
        }

        const inputDefinition = handleConfig.inputs.find((definition) => definition.key === handleKey);
        if (!inputDefinition) {
          return false;
        }

        if (!sourceHandle) {
          return false;
        }

        if (inputDefinition.accepts && inputDefinition.accepts.length > 0) {
          return inputDefinition.accepts.some((matcherKey) => {
            const matcher = CONNECTION_ACCEPT_MATCHERS[matcherKey];
            return matcher ? matcher(sourceHandle) : false;
          });
        }

        return true;
      }

      return true;
    },
    [nodes]
  );

  const onConnect = useCallback(
    (params: Edge | Connection) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            type: 'animatedEdge',
            animated: true,
          },
          eds
        )
      );

      const impactedNodes = [params.source as string | undefined, params.target as string | undefined].filter(
        (value): value is string => Boolean(value)
      );
      if (impactedNodes.length) {
        scheduleNodeInternalsUpdate(impactedNodes);
      }
    },
    [scheduleNodeInternalsUpdate, setEdges]
  );

  useEffect(() => {
    if (!sourceId) {
      const datasetNodeLabel = `Dataset input\n(${datasetDisplayLabel})`;

      if (!hasInitialSampleHydratedRef.current) {
        const sample = getSamplePipelineGraph(datasetDisplayLabel);
        const preparedNodes = prepareNodes(sample.nodes);
        hasInitialSampleHydratedRef.current = true;
        setNodes(preparedNodes);
        setEdges(sample.edges);
        updateNodeCounter(preparedNodes);
        scheduleFitView();
        onPipelineHydrated?.({ nodes: preparedNodes, edges: sample.edges, pipeline: null, context: 'sample' });
      } else {
        setNodes((existing) => {
          let changed = false;
          const next = existing.map((node) => {
            if (node.id !== 'dataset-source') {
              return node;
            }
            const currentLabel = node?.data?.label ?? '';
            if (currentLabel === datasetNodeLabel) {
              return node;
            }
            changed = true;
            return registerNodeInteractions({
              ...node,
              data: {
                ...(node.data ?? {}),
                label: datasetNodeLabel,
                isDataset: true,
                isRemovable: false,
              },
            });
          });
          return changed ? next : existing;
        });
      }

      return;
    }

    hasInitialSampleHydratedRef.current = false;

    if (pipelineQuery.isLoading) {
      const datasetNodeLabel = `Dataset input\n(${datasetDisplayLabel})`;
      setNodes((existing) =>
        existing.map((node) =>
          node.id === 'dataset-source'
            ? registerNodeInteractions({
                ...node,
                data: {
                  ...(node.data ?? {}),
                  label: datasetNodeLabel,
                  isDataset: true,
                  isRemovable: false,
                },
              })
            : node
        )
      );
      return;
    }

    if (pipelineQuery.isError) {
      const pipelineError = (pipelineQuery.error as Error) ?? new Error('Failed to load saved pipeline');
      console.error('Failed to load saved pipeline', pipelineError);
      onPipelineError?.(pipelineError);
      const sample = getSamplePipelineGraph(datasetDisplayLabel);
      const preparedNodes = prepareNodes(sample.nodes);
      setNodes(preparedNodes);
      setEdges(sample.edges);
      updateNodeCounter(preparedNodes);
      scheduleFitView();
      onPipelineHydrated?.({ nodes: preparedNodes, edges: sample.edges, pipeline: null, context: 'reset' });
      return;
    }

    if (pipelineQuery.data) {
      const graph = pipelineQuery.data.graph ?? {};
      const rawNodes = Array.isArray(graph?.nodes) && graph.nodes.length ? (graph.nodes as Node[]) : getDefaultNodes();
      const hydratedNodes = prepareNodes(rawNodes);
      const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
      const hydratedEdges = rawEdges.map((edge: any) => {
        const existingType = edge?.type;
        const type = !existingType || existingType === 'smoothstep' || existingType === 'default'
          ? 'animatedEdge'
          : existingType;
        const sourceId = edge?.source;
        const targetId = edge?.target;
        const targetNode = hydratedNodes.find((node) => node.id === targetId);
        const normalizedSourceHandle = edge?.sourceHandle ?? (sourceId ? `${sourceId}-source` : undefined);
        const normalizedTargetHandle = edge?.targetHandle ?? (targetNode?.data?.isDataset ? undefined : targetId ? `${targetId}-target` : undefined);
        return {
          ...edge,
          animated: edge?.animated ?? true,
          type,
          sourceHandle: normalizedSourceHandle,
          targetHandle: normalizedTargetHandle,
        };
      });

      setNodes(hydratedNodes);
      setEdges(hydratedEdges);
      updateNodeCounter(hydratedNodes);
      scheduleFitView();
      onPipelineHydrated?.({
        nodes: hydratedNodes,
        edges: hydratedEdges,
        pipeline: pipelineQuery.data ?? null,
        context: 'stored',
      });
      return;
    }

    if (pipelineQuery.isFetched && !pipelineQuery.data) {
      const sample = getSamplePipelineGraph(datasetDisplayLabel);
      const preparedNodes = prepareNodes(sample.nodes);
      setNodes(preparedNodes);
      setEdges(sample.edges);
      updateNodeCounter(preparedNodes);
      scheduleFitView();
      onPipelineHydrated?.({ nodes: preparedNodes, edges: sample.edges, pipeline: null, context: 'sample' });
    }
  }, [
    datasetDisplayLabel,
    onPipelineHydrated,
    onPipelineError,
    pipelineQuery.data,
    pipelineQuery.error,
    pipelineQuery.isError,
    pipelineQuery.isFetched,
    pipelineQuery.isLoading,
    prepareNodes,
    registerNodeInteractions,
    scheduleFitView,
    setEdges,
    setNodes,
    sourceId,
    updateNodeCounter,
  ]);

  useEffect(() => {
    onGraphChange?.(nodes, edges);
  }, [edges, nodes, onGraphChange]);

  const selectedNodeDefaultConfig = useMemo(() => {
    if (!selectedNode) {
      return null;
    }
    const catalogType = selectedNode?.data?.catalogType;
    if (!catalogType) {
      return null;
    }
    const catalogEntry = catalogEntryMap.get(catalogType);
    if (!catalogEntry) {
      return null;
    }
    return sanitizeDefaultConfigForNode(catalogEntry);
  }, [catalogEntryMap, selectedNode]);

  const canResetSelectedNode = useMemo(() => {
    if (!selectedNode) {
      return false;
    }
    if (selectedNode.data?.isDataset) {
      return false;
    }
    const catalogType = selectedNode?.data?.catalogType;
    return isResettableEntry(typeof catalogType === 'string' ? catalogType : null);
  }, [isResettableEntry, selectedNode]);

  const sidebarContent = useMemo(() => {
    if (isCatalogLoading) {
      return <p className="text-muted">Loading node catalog…</p>;
    }
    if (catalogErrorMessage) {
      return <p className="text-danger">{catalogErrorMessage}</p>;
    }
    if (!nodeCatalog.length) {
      return <p className="text-muted">Node catalog unavailable. Define nodes in the backend to continue.</p>;
    }
    return <FeatureCanvasSidebar nodes={nodeCatalog} onAddNode={handleAddNode} />;
  }, [catalogErrorMessage, handleAddNode, isCatalogLoading, nodeCatalog]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      handleOpenSettings(node.id);
    },
    [handleOpenSettings]
  );

  return (
    <div className="canvas-stage">
      <div
        className="canvas-stage__viewport"
        data-pipeline-loading={isPipelineLoading ? 'true' : 'false'}
        aria-busy={isPipelineLoading}
        ref={canvasViewportRef}
      >
        <ReactFlow
          style={{ width: '100%', height: '100%' }}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          onNodeClick={handleNodeClick}
          onInit={(instance) => {
            reactFlowInstanceRef.current = instance;
            requestAnimationFrame(() => instance.fitView({ padding: 0.25 }));
          }}
          minZoom={0.5}
          maxZoom={1.75}
          connectionRadius={180}
          connectionMode={ConnectionMode.Loose}
          connectOnClick
          nodeDragHandle=".feature-node__drag-handle"
          proOptions={{ hideAttribution: true }}
          defaultEdgeOptions={{ 
            type: 'animatedEdge', 
            animated: true,
            style: { strokeWidth: 3 }
          }}
          connectionLineComponent={ConnectionLine}
          elevateEdgesOnSelect={true}
          edgesUpdatable={false}
        >
          <Controls position="bottom-left" />
          <Background gap={16} />
        </ReactFlow>

        <button
          type="button"
          className="canvas-fab canvas-fab--reset"
          onClick={handleResetAllNodes}
          disabled={!hasResettableNodes}
          aria-label="Reset preprocessing nodes"
          title={hasResettableNodes ? 'Reset all preprocessing nodes to defaults' : 'No preprocessing nodes to reset'}
        >
          ↺
        </button>

        <button
          type="button"
          className="canvas-fab"
          onClick={() => setIsCatalogOpen(true)}
          aria-label="Open node catalog"
        >
          +
        </button>
      </div>

      <div className={`canvas-drawer${isCatalogOpen ? ' canvas-drawer--open' : ''}`}>
        <div className="canvas-drawer__header">
          <h2>Node catalog</h2>
          <button
            type="button"
            className="canvas-drawer__close"
            onClick={() => setIsCatalogOpen(false)}
            aria-label="Close catalog"
          >
            ×
          </button>
        </div>
        <div className="canvas-drawer__body">{sidebarContent}</div>
      </div>

      {isCatalogOpen && <div className="canvas-drawer__backdrop" onClick={() => setIsCatalogOpen(false)} />}

      {isSettingsModalOpen && selectedNode && (
        <NodeSettingsModal
          node={selectedNode}
          sourceId={sourceId ?? null}
          graphSnapshot={graphSnapshot}
          onClose={handleCloseSettings}
          onUpdateConfig={handleUpdateNodeConfig}
          onUpdateNodeData={handleUpdateNodeData}
          onResetConfig={handleResetNodeConfig}
          defaultConfigTemplate={selectedNodeDefaultConfig}
          isResetAvailable={canResetSelectedNode}
        />
      )}
    </div>
  );
});

type SaveFeedback = {
  message: string;
  tone: 'info' | 'success' | 'error';
};

const App: React.FC<AppProps> = ({ sourceId }) => {
  const queryClient = useQueryClient();
  const canvasShellRef = useRef<CanvasShellHandle | null>(null);
  const [activeSourceId, setActiveSourceId] = useState<string | null>(sourceId ?? null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetSourceSummary | null>(null);
  const [isSidepanelExpanded, setIsSidepanelExpanded] = useState(true);
  const [isDirty, setIsDirty] = useState(false);
  const [saveFeedback, setSaveFeedback] = useState<SaveFeedback | null>(null);
  const [activePipelineId, setActivePipelineId] = useState<number | null>(null);
  const [activePipelineUpdatedAt, setActivePipelineUpdatedAt] = useState<string | null>(null);
  const [activePipelineName, setActivePipelineName] = useState<string | null>(null);
  const [canClearCanvas, setCanClearCanvas] = useState(false);

  const pipelineQueryKey = useMemo(
    () => ['feature-canvas', 'pipeline', activeSourceId],
    [activeSourceId]
  );
  const pendingHistoryIdRef = useRef<number | null>(null);
  const graphSnapshotRef = useRef<{ nodes: any[]; edges: any[] }>({ nodes: [], edges: [] });
  const isHydratingRef = useRef(true);

  const datasetsQuery = useQuery({
    queryKey: ['feature-canvas', 'datasets'],
    queryFn: () => fetchDatasets(12),
    staleTime: 5 * 60 * 1000,
  });
  const datasets = datasetsQuery.data ?? [];
  const isDatasetLoading = datasetsQuery.isLoading;
  const datasetErrorMessage = datasetsQuery.error
    ? (datasetsQuery.error as Error)?.message ?? 'Unable to load datasets'
    : null;
  const ownedDatasets = useMemo(
    () => datasets.filter((item) => item?.is_owned !== false),
    [datasets]
  );

  useEffect(() => {
    if (!datasets.length) {
      setSelectedDataset(null);
      return;
    }

    if (activeSourceId) {
      const match = datasets.find((item) => item.source_id === activeSourceId);
      if (match) {
        setSelectedDataset((previous) => (previous?.id === match.id ? previous : match));
        return;
      }
    }

    const fallback = ownedDatasets[0] ?? datasets[0];
    if (!fallback) {
      return;
    }

    if (!activeSourceId) {
      setActiveSourceId(fallback.source_id ?? String(fallback.id));
    }
    setSelectedDataset(fallback);
  }, [activeSourceId, datasets, ownedDatasets]);

  const pipelineHistoryQuery = useQuery({
    queryKey: ['feature-canvas', 'pipeline-history', activeSourceId],
    queryFn: () => fetchPipelineHistory(activeSourceId as string, HISTORY_LIMIT),
    enabled: Boolean(activeSourceId),
    staleTime: 30 * 1000,
    retry: 1,
  });

  const historyItems = pipelineHistoryQuery.data ?? [];
  const isHistoryLoading = pipelineHistoryQuery.isLoading || pipelineHistoryQuery.isFetching;
  const historyErrorMessage = pipelineHistoryQuery.error
    ? (pipelineHistoryQuery.error as Error)?.message ?? 'Unable to load history'
    : null;

  const { mutate: triggerSave, isPending: isSaving } = useMutation({
    mutationFn: async ({
      sourceId: datasetSourceId,
      payload,
    }: {
      sourceId: string;
      payload: FeaturePipelinePayload;
    }) => savePipeline(datasetSourceId, payload),
    onSuccess: (response, variables) => {
      setSaveFeedback({ message: 'Draft saved successfully.', tone: 'success' });
      setIsDirty(false);
      setActivePipelineId(response.id ?? null);
      setActivePipelineName(response.name ?? null);
      setActivePipelineUpdatedAt(response.updated_at ?? null);
      pendingHistoryIdRef.current = response.id ?? null;
      queryClient.setQueryData(['feature-canvas', 'pipeline', variables.sourceId], response);
      queryClient.invalidateQueries({
        queryKey: ['feature-canvas', 'pipeline-history', variables.sourceId],
        exact: false,
      });
    },
    onError: (error: Error) => {
      setSaveFeedback({ message: error?.message ?? 'Failed to save pipeline.', tone: 'error' });
    },
  });

  useEffect(() => {
    if (!saveFeedback || saveFeedback.tone === 'error' || typeof window === 'undefined') {
      return;
    }
    const timeout = window.setTimeout(
      () => setSaveFeedback(null),
      saveFeedback.tone === 'success' ? 4000 : 2500
    );
    return () => window.clearTimeout(timeout);
  }, [saveFeedback]);

  const handleGraphChange = useCallback((nodes: Node[], edges: Edge[]) => {
    const nextSnapshot = {
      nodes: JSON.parse(JSON.stringify(nodes ?? [])),
      edges: JSON.parse(JSON.stringify(edges ?? [])),
    };
    const previousSnapshot = graphSnapshotRef.current;

    const isSameSnapshot =
      JSON.stringify(previousSnapshot.nodes ?? []) === JSON.stringify(nextSnapshot.nodes ?? []) &&
      JSON.stringify(previousSnapshot.edges ?? []) === JSON.stringify(nextSnapshot.edges ?? []);

    graphSnapshotRef.current = nextSnapshot;

    const hasRemovableNodes = Array.isArray(nextSnapshot.nodes)
      ? nextSnapshot.nodes.some((node: any) => node?.id && node.id !== 'dataset-source')
      : false;
    const hasEdges = Array.isArray(nextSnapshot.edges) ? nextSnapshot.edges.length > 0 : false;
    setCanClearCanvas(hasRemovableNodes || hasEdges);

    if (isHydratingRef.current || isSameSnapshot) {
      return;
    }

    setIsDirty(true);
    setSaveFeedback(null);
  }, []);

  const handlePipelineHydrated = useCallback(
    (payload: PipelineHydrationPayload) => {
      const { nodes, edges, pipeline, context } = payload;
      const snapshot = {
        nodes: JSON.parse(JSON.stringify(nodes ?? [])),
        edges: JSON.parse(JSON.stringify(edges ?? [])),
      };

      graphSnapshotRef.current = snapshot;

      const hasCustomNodes = (snapshot.nodes ?? []).some((node) => node.id !== 'dataset-source');
      const hasEdges = Array.isArray(snapshot.edges) ? snapshot.edges.length > 0 : false;
      setCanClearCanvas(hasCustomNodes || hasEdges);
      const hydratedPipelineId = pipeline?.id ?? null;

      setActivePipelineId(hydratedPipelineId);
      setActivePipelineName(pipeline?.name ?? null);
      setActivePipelineUpdatedAt(pipeline?.updated_at ?? null);

      if (context === 'stored' && pipeline) {
        if (pendingHistoryIdRef.current === pipeline.id) {
          const relative = formatRelativeTime(pipeline.updated_at);
          const timeLabel = relative ?? formatTimestamp(pipeline.updated_at);
          setSaveFeedback({
            message: `Loaded revision “${pipeline.name ?? `#${pipeline.id}`}” (${timeLabel})`,
            tone: 'info',
          });
          pendingHistoryIdRef.current = null;
        } else if (isHydratingRef.current) {
          setSaveFeedback((previous) => {
            if (previous?.tone === 'error') {
              return previous;
            }
            if (hasCustomNodes) {
              return { message: 'Loaded saved pipeline', tone: 'info' };
            }
            return previous;
          });
        }
      } else if (context === 'sample') {
        pendingHistoryIdRef.current = null;
        setSaveFeedback((previous) => {
          if (previous?.tone === 'error') {
            return previous;
          }
          if (!hasCustomNodes) {
            const datasetLabel = selectedDataset?.name ?? selectedDataset?.source_id ?? activeSourceId ?? 'demo dataset';
            return {
              message: `Showing starter pipeline for ${datasetLabel}`,
              tone: 'info',
            };
          }
          return previous;
        });
      } else if (context === 'reset') {
        pendingHistoryIdRef.current = null;
      } else if (!pipeline) {
        pendingHistoryIdRef.current = null;
        setSaveFeedback((previous) => (previous?.tone === 'error' ? previous : null));
      }

      if (isHydratingRef.current) {
        setIsDirty(false);
      }

      if (pipeline && pendingHistoryIdRef.current && pendingHistoryIdRef.current !== pipeline.id) {
        pendingHistoryIdRef.current = null;
      }

      isHydratingRef.current = false;
    },
    [activeSourceId, selectedDataset, setCanClearCanvas]
  );

  const handlePipelineError = useCallback((error: Error) => {
    setSaveFeedback({
      message: error?.message ?? 'Unable to load saved pipeline. Starting fresh.',
      tone: 'error',
    });
  }, []);

  const handleHistorySelection = useCallback(
    (pipeline: FeaturePipelineResponse) => {
      if (!pipeline?.graph) {
        setSaveFeedback({ message: 'Selected revision is missing pipeline data.', tone: 'error' });
        return;
      }

      if (isDirty && typeof window !== 'undefined') {
        const confirmReplace = window.confirm(
          'You have unsaved changes. Replace the canvas with the selected revision?'
        );
        if (!confirmReplace) {
          return;
        }
      }

      pendingHistoryIdRef.current = pipeline.id ?? null;
      isHydratingRef.current = true;
      setSaveFeedback({
        message: `Loading revision “${pipeline.name ?? `#${pipeline.id}`}”…`,
        tone: 'info',
      });
      queryClient.setQueryData(pipelineQueryKey, pipeline);
    },
    [isDirty, pipelineQueryKey, queryClient]
  );

  const handleSaveClick = useCallback(() => {
    if (!activeSourceId) {
      setSaveFeedback({ message: 'Select a dataset before saving.', tone: 'error' });
      return;
    }

    const snapshot = graphSnapshotRef.current;
    if (!snapshot.nodes || !snapshot.nodes.length) {
      setSaveFeedback({ message: 'Add nodes to the canvas before saving.', tone: 'error' });
      return;
    }

    const payload: FeaturePipelinePayload = {
      name: selectedDataset?.name
        ? `Draft pipeline for ${selectedDataset.name}`
        : `Draft pipeline for ${activeSourceId}`,
      graph: {
        nodes: snapshot.nodes,
        edges: (snapshot.edges ?? []).map((edge) => {
          const existingType = edge?.type;
          const type = !existingType || existingType === 'smoothstep' || existingType === 'default'
            ? 'animatedEdge'
            : existingType;
          return {
            ...edge,
            animated: edge?.animated ?? true,
            type,
          };
        }),
      },
      metadata: {
        lastClientSave: new Date().toISOString(),
        nodeCount: snapshot.nodes.length,
        edgeCount: snapshot.edges?.length ?? 0,
      },
    };

    setSaveFeedback({ message: 'Saving draft…', tone: 'info' });
    triggerSave({ sourceId: activeSourceId, payload });
  }, [activeSourceId, selectedDataset?.name, triggerSave]);

  const handleClearCanvas = useCallback(() => {
    if (!canClearCanvas) {
      return;
    }

    if (typeof window !== 'undefined') {
      const confirmClear = window.confirm(
        'Clear all nodes and connections from the canvas? This cannot be undone.'
      );
      if (!confirmClear) {
        return;
      }
    }

    canvasShellRef.current?.clearGraph();
    setSaveFeedback((previous) => {
      if (previous?.tone === 'error') {
        return previous;
      }
      return { message: 'Canvas cleared. Unsaved edits pending.', tone: 'info' };
    });
  }, [canClearCanvas, canvasShellRef, setSaveFeedback]);

  const feedbackIcon = saveFeedback?.tone === 'success' ? '✅' : saveFeedback?.tone === 'error' ? '⚠️' : '💬';
  const handleToggleSidepanel = useCallback(() => {
    setIsSidepanelExpanded((previous) => !previous);
  }, []);

  const handleDatasetSelection = useCallback(
    (value: string) => {
      const match = datasets.find((item) => item.source_id === value) ?? null;
      if (!match) {
        setSaveFeedback({ message: 'Dataset unavailable. Choose another option.', tone: 'error' });
        return;
      }

      if (match.is_owned === false) {
        setSaveFeedback({ message: 'Only datasets you own can be selected.', tone: 'error' });
        return;
      }

      setActiveSourceId(value);
      setSelectedDataset(match);
      isHydratingRef.current = true;
      setIsDirty(false);
      setSaveFeedback(null);
    },
    [datasets]
  );

  const saveButtonLabel = isSaving ? 'Saving…' : isDirty ? 'Save draft*' : 'Save draft';
  const isSaveDisabled = !activeSourceId || isSaving;
  const feedbackClass = saveFeedback
    ? saveFeedback.tone === 'error'
      ? 'text-danger'
      : saveFeedback.tone === 'success'
      ? 'text-success'
      : 'text-muted'
    : 'text-muted';
  const datasetOptions = datasets.map((item) => ({
    value: item.source_id,
    label: item.name ?? item.source_id,
    isOwned: item.is_owned !== false,
  }));
  const canSelectDatasets = ownedDatasets.length > 1;

  return (
    <div
      className="feature-canvas-app"
      data-sidepanel-expanded={isSidepanelExpanded ? 'true' : 'false'}
    >
      <button
        type="button"
        className="canvas-sidepanel__toggle"
        onClick={handleToggleSidepanel}
        aria-label={isSidepanelExpanded ? 'Collapse details panel' : 'Expand details panel'}
        aria-expanded={isSidepanelExpanded}
        aria-controls="feature-canvas-sidepanel"
      >
        {isSidepanelExpanded ? '⟨' : '⟩'}
      </button>

      {!isSidepanelExpanded && saveFeedback && (
        <div
          className={`canvas-sidepanel__feedback canvas-sidepanel__feedback--floating ${feedbackClass}`}
        >
          <span className="canvas-sidepanel__feedback-icon" aria-hidden="true">{feedbackIcon}</span>
          <span className="canvas-sidepanel__feedback-text">{saveFeedback.message}</span>
        </div>
      )}

      {isSidepanelExpanded && (
        <aside
          id="feature-canvas-sidepanel"
          className="canvas-sidepanel"
          role="complementary"
          aria-label="Canvas details"
        >
          <div className="canvas-sidepanel__content" role="group" aria-label="Canvas controls">
            <section className="canvas-sidepanel__section">
              <div className="canvas-sidepanel__section-heading">
                <h2>Dataset</h2>
                {activePipelineUpdatedAt && (
                  <span className="canvas-sidepanel__meta">
                    {formatRelativeTime(activePipelineUpdatedAt) ?? formatTimestamp(activePipelineUpdatedAt)}
                  </span>
                )}
              </div>
              {datasetErrorMessage ? (
                <p className="text-danger">{datasetErrorMessage}</p>
              ) : ownedDatasets.length ? (
                <div className="canvas-sidepanel__select-wrapper">
                  <select
                    className="canvas-sidepanel__select"
                    value={activeSourceId ?? ''}
                    onChange={(event) => handleDatasetSelection(event.target.value)}
                    disabled={isDatasetLoading || !canSelectDatasets}
                    aria-label="Select dataset"
                  >
                    {datasetOptions.map((option) => (
                      <option key={option.value} value={option.value} disabled={!option.isOwned}>
                        {`${option.label}${option.isOwned ? '' : ' (locked)'}`}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <p className="text-muted">Dataset switching is limited to collections you own.</p>
              )}
              {isDatasetLoading && <p className="text-muted">Loading datasets…</p>}
              {!isDatasetLoading && !datasets.length && !datasetErrorMessage && (
                <p className="text-muted">No datasets available. Send one from EDA to get started.</p>
              )}
              {selectedDataset?.description && (
                <p className="canvas-sidepanel__description">{selectedDataset.description}</p>
              )}
              {activeSourceId && (
                <p className="canvas-sidepanel__meta">
                  Working with <strong>{selectedDataset?.name ?? activeSourceId}</strong>
                </p>
              )}
              {!canSelectDatasets && ownedDatasets.length > 0 && (
                <p className="canvas-sidepanel__hint">Add more owned datasets to switch between them.</p>
              )}
            </section>
            <section className="canvas-sidepanel__section">
              <div className="canvas-sidepanel__section-action">
                <button
                  type="button"
                  className="canvas-sidepanel__action-button"
                  onClick={handleClearCanvas}
                  disabled={!canClearCanvas}
                  aria-label="Clean all nodes"
                  title={canClearCanvas ? 'Remove all nodes and edges from the canvas' : 'Nothing to clean yet'}
                >
                  <span aria-hidden="true">🧹</span>
                  <span>Clean all nodes</span>
                </button>
              </div>
              <div className="canvas-sidepanel__section-heading">
                <h2>History</h2>
                <div className="canvas-sidepanel__section-controls">
                  <button
                    type="button"
                    className="canvas-sidepanel__icon-button"
                    onClick={() => pipelineHistoryQuery.refetch()}
                    disabled={isHistoryLoading || !activeSourceId}
                    aria-label="Refresh history"
                    title="Refresh history"
                  >
                    ⟳
                  </button>
                  <button
                    type="button"
                    className="canvas-sidepanel__icon-button"
                    onClick={handleSaveClick}
                    disabled={isSaveDisabled}
                    aria-label={saveButtonLabel}
                    title={saveButtonLabel}
                  >
                    💾
                  </button>
                </div>
              </div>
              <div className="canvas-sidepanel__history">
                {isHistoryLoading ? (
                  <p className="text-muted">Loading history…</p>
                ) : historyErrorMessage ? (
                  <p className="text-danger">{historyErrorMessage}</p>
                ) : historyItems.length ? (
                  historyItems.map((item) => {
                    const relative = formatRelativeTime(item.updated_at);
                    const timestampLabel = relative ?? formatTimestamp(item.updated_at);
                    const matchesActive = item.id === activePipelineId;
                    const disableSelection = matchesActive && !isDirty;

                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`canvas-sidepanel__history-item${
                          matchesActive ? ' canvas-sidepanel__history-item--active' : ''
                        }`}
                        onClick={() => handleHistorySelection(item)}
                        disabled={disableSelection}
                      >
                        <span className="canvas-sidepanel__history-title">
                          {item.name || `Pipeline #${item.id}`}
                        </span>
                        <span className="canvas-sidepanel__history-meta">{timestampLabel}</span>
                        {matchesActive && isDirty && (
                          <span className="canvas-sidepanel__history-badge">Unsaved edits</span>
                        )}
                      </button>
                    );
                  })
                ) : (
                  <p className="text-muted">No saved revisions yet. Save a draft to populate history.</p>
                )}
              </div>
            </section>
          </div>
          {saveFeedback && (
            <p className={`canvas-sidepanel__feedback ${feedbackClass}`}>
              <span className="canvas-sidepanel__feedback-icon" aria-hidden="true">{feedbackIcon}</span>
              <span className="canvas-sidepanel__feedback-text">{saveFeedback.message}</span>
            </p>
          )}
        </aside>
      )}

      <div className="feature-canvas-app__viewport">
        <ReactFlowProvider>
          <CanvasShell
            ref={canvasShellRef}
            sourceId={activeSourceId}
            datasetName={selectedDataset?.name ?? selectedDataset?.source_id ?? null}
            onGraphChange={handleGraphChange}
            onPipelineHydrated={handlePipelineHydrated}
            onPipelineError={handlePipelineError}
          />
        </ReactFlowProvider>
      </div>
    </div>
  );
};

export default App;
