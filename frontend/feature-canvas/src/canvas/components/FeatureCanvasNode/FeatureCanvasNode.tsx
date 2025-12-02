// @ts-nocheck
import React, { useCallback, useMemo } from 'react';
import { Handle, NodeProps, Position } from 'react-flow-renderer';
import { AlertTriangle, Check, AlertCircle, X, Loader2 } from 'lucide-react';
import type { FeatureNodeData } from '../../types/nodes';
import {
  NODE_HANDLE_CONFIG,
  formatHandleDisplayLabel,
  resolveHandleTopPosition,
} from '../../constants/nodeHandles';
import {
  SPLIT_LABEL_MAP,
  SPLIT_TYPE_ORDER,
  sanitizeSplitList,
  getSplitHandlePosition,
} from '../../constants/splits';
import { HANDLE_BASE_SIZE, DROP_MISSING_DISPLAY_DESCRIPTION, DROP_MISSING_LEGACY_DESCRIPTION } from '../../constants/defaults';

const buildHandleStyle = (type: 'target' | 'source'): React.CSSProperties => ({
  width: `${HANDLE_BASE_SIZE}px`,
  height: `${HANDLE_BASE_SIZE}px`,
  borderRadius: '999px',
  border: type === 'target' ? '1px solid rgba(148, 163, 184, 0.55)' : '1px solid rgba(45, 212, 191, 0.55)',
  background: type === 'target' ? 'rgba(148, 163, 184, 0.22)' : 'rgba(45, 212, 191, 0.18)',
  boxShadow: type === 'target' ? '0 0 0 3px rgba(148, 163, 184, 0.18)' : '0 0 0 3px rgba(45, 212, 191, 0.18)',
  backdropFilter: 'blur(2px)',
});

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
  const isModelingNode = ['hyperparameter_tuning', 'train_model_draft', 'model_evaluation', 'model_registry_overview'].includes(
    catalogType
  );
  const pendingWarningReason = data?.pendingWarningReason ?? null;
  const hasPendingWarning = Boolean(data?.pendingWarningActive);
  const isPendingHighlight = Boolean(data?.pendingHighlight);
  const pendingWarningMessage = pendingWarningReason ?? 'This node still needs configuration.';

  const nodeClassName = [
    'feature-node',
    selected ? 'feature-node--selected' : '',
    isDataset ? 'feature-node--dataset' : '',
    hasPendingWarning ? 'feature-node--pending-warning' : '',
    isPendingHighlight ? 'feature-node--pending-highlight' : '',
  ]
    .filter(Boolean)
    .join(' ');

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

  const renderHandle = useCallback((handleConfig, type: 'target' | 'source') => {
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
            ...(isSplitHandle
              ? {
                  opacity: 1,
                  visibility: 'visible',
                  pointerEvents: 'all',
                }
              : {}),
          }}
        />
        {label && type === 'source' && (
          <div
            className="feature-node__handle-label"
            style={{
              position: 'absolute',
              right: '-8px',
              top: style?.top || '50%',
              transform: 'translate(100%, calc(-100% - 8px))',
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
              transform: 'translate(-100%, calc(-100% - 8px))',
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
    <div className={nodeClassName}>
      {targetHandles.map((handle) => renderHandle(handle, 'target'))}
      <div className="feature-node__header">
        <div className="feature-node__title-group">
          <span className="feature-node__drag-handle" aria-hidden="true">
            ⠿
          </span>
          <span className="feature-node__title">{label}</span>
        </div>
        <div className="feature-node__controls">
          {hasPendingWarning && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--pending"
              title={pendingWarningMessage}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              !
            </span>
          )}
          {needsConnection && isModelingNode && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--warning"
              title="Required connections missing"
              style={{
                background: 'rgba(251, 191, 36, 0.2)',
                color: 'rgb(245, 158, 11)',
                border: '1px solid rgba(245, 158, 11, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <AlertTriangle size={12} />
            </span>
          )}
          {backgroundStatus === 'loading' && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--loading"
              title="Loading full dataset in background..."
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <Loader2 size={12} className="feature-node__spinner" />
            </span>
          )}
          {backgroundStatus === 'success' && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--success"
              title="Full dataset ready"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <Check size={12} />
            </span>
          )}
          {backgroundStatus === 'error' && (
            <span
              className="feature-node__status-indicator feature-node__status-indicator--error"
              title="Background execution failed"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <AlertCircle size={12} />
            </span>
          )}
          {isRemovable && (
            <button
              className="feature-node__control feature-node__control--danger"
              type="button"
              onClick={handleRemove}
              title="Remove node"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <X size={12} />
            </button>
          )}
        </div>
      </div>
      {description && <p className="feature-node__description">{description}</p>}
      {hasPendingWarning && <p className="feature-node__pending-hint">{pendingWarningMessage}</p>}
      {sourceHandles.map((handle) => renderHandle(handle, 'source'))}
    </div>
  );
};

export default FeatureCanvasNode;
