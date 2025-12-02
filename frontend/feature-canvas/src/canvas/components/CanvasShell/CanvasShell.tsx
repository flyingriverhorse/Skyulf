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
import { X } from 'lucide-react';
import {
  Node,
  ReactFlowInstance,
  useEdgesState,
  useNodesState,
  useUpdateNodeInternals,
} from 'react-flow-renderer';
import AnimatedEdge from '../../../components/edges/AnimatedEdge';
import { NodeSettingsModal } from '../../../components/NodeSettingsModal';
import FeatureCanvasNode from '../FeatureCanvasNode/FeatureCanvasNode';
import type { PendingConfigurationDetail } from '../../../components/node-settings/utils/pendingConfiguration';
import { usePendingConfigurationToast } from '../../../components/node-settings/hooks';
import PendingConfigurationDock from '../../../components/PendingConfigurationToast/PendingConfigurationDock';
import {
  sanitizeDefaultConfigForNode,
  PENDING_CONFIRMATION_FLAG,
  isAutoConfirmedCatalogType,
} from '../../services/configSanitizer';
import { getDefaultEdges, getDefaultNodes } from '../../services/layout';
import { isResettableCatalogEntry } from '../../services/nodeFactory';
import { buildGraphSnapshot } from '../../services/graphSerialization';
import type { CanvasShellHandle, CanvasShellProps } from '../../types/pipeline';
import { useNodeCatalogDrawer } from '../../hooks';
import { useCanvasSidebar } from '../../hooks';
import { useNodeEditor } from '../../hooks';
import { usePipelineLoader } from '../../hooks';
import { useSplitPropagation } from '../../hooks';
import { useConnectionHandlers } from '../../hooks';
import { CanvasViewport } from '../CanvasViewport/CanvasViewport';
import { fetchTrainingJobs, fetchHyperparameterTuningJobs } from '../../../api';

const normalizeLabel = (value?: string | null): string => {
  if (!value || typeof value !== 'string') {
    return '';
  }
  return value.replace(/\s+/g, ' ').trim().toLowerCase();
};

const CanvasShell = forwardRef<CanvasShellHandle, CanvasShellProps>(
    ({ sourceId, datasetName, onGraphChange, onPipelineHydrated, onPipelineError }, ref) => {
      const nodeTypes = useMemo(() => ({ featureNode: FeatureCanvasNode }), []);
      const edgeTypes = useMemo(() => ({ animatedEdge: AnimatedEdge }), []);
      const [nodes, setNodes, onNodesChange] = useNodesState(
        getDefaultNodes().map((node) => ({
          ...node,
          type: 'featureNode',
        }))
      );
      const [edges, setEdges, onEdgesChange] = useEdgesState(getDefaultEdges());
      const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
      const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
      const [pendingConfigurationDetails, setPendingConfigurationDetails] = useState<PendingConfigurationDetail[]>([]);
      const [pendingHighlightLabels, setPendingHighlightLabels] = useState<string[]>([]);
      const {
        pendingToastDetails,
        isPendingToastVisible,
        showPendingToast,
        dismissPendingToast,
        clearPendingToast,
      } = usePendingConfigurationToast();
      const reactFlowInstanceRef = useRef<ReactFlowInstance | null>(null);
      const canvasViewportRef = useRef<HTMLDivElement | null>(null);
      const shouldFitViewRef = useRef(false);
      const datasetDisplayLabel = datasetName ?? sourceId ?? 'Demo dataset';
      const updateNodeInternals = useUpdateNodeInternals();
      const {
        nodeCatalog,
        catalogEntryMapRef,
        isCatalogOpen,
        openCatalog,
        closeCatalog,
        isCatalogLoading,
        catalogErrorMessage,
      } = useNodeCatalogDrawer();

      const scheduleNodeInternalsUpdate = useCallback(
        (nodeIds: string | string[]) => {
          const ids = (Array.isArray(nodeIds) ? nodeIds : [nodeIds]).filter(Boolean);
          if (!ids.length) {
            return;
          }

          const uniqueIds = Array.from(new Set(ids));
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              uniqueIds.forEach((nodeId) => {
                updateNodeInternals(nodeId);
              });
            });
          });
        },
        [updateNodeInternals]
      );

      useSplitPropagation({ edges, setNodes, scheduleNodeInternalsUpdate });

      const selectedNode = useMemo(
        () => nodes.find((node) => node.id === selectedNodeId) ?? null,
        [nodes, selectedNodeId]
      );

      const graphSnapshot = useMemo(() => buildGraphSnapshot(nodes, edges), [edges, nodes]);

      const handleOpenSettings = useCallback((nodeId: string) => {
        setSelectedNodeId(nodeId);
        setIsSettingsModalOpen(true);
      }, []);

      const handleCloseSettings = useCallback(() => {
        setIsSettingsModalOpen(false);
        setSelectedNodeId(null);
      }, []);

      const handlePendingConfigurationUpdate = useCallback((details: PendingConfigurationDetail[]) => {
        setPendingConfigurationDetails(details);
      }, []);

      const handlePendingConfigurationCleared = useCallback(() => {
        setPendingConfigurationDetails([]);
        setPendingHighlightLabels([]);
      }, []);

      const handleHighlightPendingNodes = useCallback(
        (labels: string[]) => {
          const normalized = Array.from(
            new Set(
              labels
                .map((label) => normalizeLabel(label))
                .filter((label): label is string => Boolean(label))
            )
          );
          if (!normalized.length) {
            return;
          }
          setIsSettingsModalOpen(false);
          setSelectedNodeId(null);
          setPendingHighlightLabels(normalized);
          requestAnimationFrame(() => {
            const targetNodes = nodes.filter((node) => {
              const data = node.data ?? {};
              const candidateLabels = [
                typeof data.label === 'string' ? data.label : null,
                typeof data.title === 'string' ? data.title : null,
                typeof data.name === 'string' ? data.name : null,
                typeof data.displayName === 'string' ? data.displayName : null,
                typeof data.display_label === 'string' ? data.display_label : null,
                typeof data.catalogType === 'string' ? data.catalogType : null,
                typeof data.type === 'string' ? data.type : null,
                node.id,
              ];
              return candidateLabels.some((value) => {
                const normalizedLabel = normalizeLabel(value);
                return normalizedLabel && normalized.includes(normalizedLabel);
              });
            });
            if (targetNodes.length) {
              reactFlowInstanceRef.current?.fitView({ nodes: targetNodes, padding: 0.35, duration: 400 });
            } else {
              reactFlowInstanceRef.current?.fitView({ padding: 0.3, duration: 350 });
            }
          });
        },
        [nodes, reactFlowInstanceRef]
      );

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

      useEffect(() => {
        if (!pendingHighlightLabels.length) {
          return;
        }
        const timer = setTimeout(() => {
          setPendingHighlightLabels([]);
        }, 4000);
        return () => clearTimeout(timer);
      }, [pendingHighlightLabels]);

      useEffect(() => {
        if (!pendingConfigurationDetails.length) {
          clearPendingToast();
          return;
        }
        showPendingToast(pendingConfigurationDetails);
      }, [pendingConfigurationDetails, showPendingToast, clearPendingToast]);

      useEffect(() => {
        if (!pendingConfigurationDetails.length) {
          setPendingHighlightLabels((current) => (current.length ? [] : current));
          return;
        }
        const pendingLabelSet = new Set(
          pendingConfigurationDetails
            .map((detail) => normalizeLabel(detail.label))
            .filter((label): label is string => Boolean(label))
        );
        setPendingHighlightLabels((current) => {
          if (!current.length) {
            return current;
          }
          const filtered = current.filter((label) => pendingLabelSet.has(label));
          return filtered.length === current.length ? current : filtered;
        });
      }, [pendingConfigurationDetails]);

      const {
        registerNode,
        handleAddNode,
        handleUpdateNodeConfig,
        handleUpdateNodeData,
        handleResetNodeConfig,
        handleResetAllNodes,
        updateNodeCounter,
      } = useNodeEditor({
        setNodes,
        setEdges,
        setSelectedNodeId,
        setIsSettingsModalOpen,
        catalogEntryMapRef,
        handleOpenSettings,
        reactFlowInstanceRef,
        canvasViewportRef,
        scheduleFitView,
      });

      useEffect(() => {
        updateNodeCounter(nodes);
      }, [nodes, updateNodeCounter]);

      // Poll for active training/tuning jobs to update progress even when modal is closed
      useEffect(() => {
        const activeNodes = nodes.filter((node) => {
          const status = node.data?.backgroundExecutionStatus;
          const type = node.data?.catalogType;
          return (
            status === 'loading' &&
            (type === 'train_model_draft' || type === 'hyperparameter_tuning')
          );
        });

        if (activeNodes.length === 0) {
          return;
        }

        const timer = setTimeout(async () => {
          const updates = new Map<string, { backgroundExecutionStatus: string; progress?: number }>();

          await Promise.all(
            activeNodes.map(async (node) => {
              try {
                const catalogType = node.data?.catalogType;
                let latestJob = null;

                if (catalogType === 'train_model_draft') {
                  const res = await fetchTrainingJobs({
                    datasetSourceId: sourceId || undefined,
                    nodeId: node.id,
                    limit: 1,
                  });
                  latestJob = res.jobs?.[0];
                } else if (catalogType === 'hyperparameter_tuning') {
                  const res = await fetchHyperparameterTuningJobs({
                    datasetSourceId: sourceId || undefined,
                    nodeId: node.id,
                    limit: 1,
                  });
                  latestJob = res.jobs?.[0];
                }

                if (latestJob) {
                  const status = latestJob.status?.toLowerCase();
                  let nodeStatus = 'idle';
                  if (status === 'running' || status === 'queued') {
                    nodeStatus = 'loading';
                  } else if (status === 'succeeded') {
                    nodeStatus = 'success';
                  } else if (status === 'failed' || status === 'cancelled') {
                    nodeStatus = 'error';
                  }

                  const progress = typeof latestJob.progress === 'number' ? latestJob.progress : undefined;
                  updates.set(node.id, { backgroundExecutionStatus: nodeStatus, progress });
                }
              } catch (error) {
                // Ignore polling errors
              }
            })
          );

          if (updates.size > 0) {
            setNodes((currentNodes) =>
              currentNodes.map((node) => {
                if (updates.has(node.id)) {
                  const update = updates.get(node.id)!;
                  const currentData = node.data || {};
                  if (
                    currentData.backgroundExecutionStatus !== update.backgroundExecutionStatus ||
                    currentData.progress !== update.progress
                  ) {
                    return { ...node, data: { ...currentData, ...update } };
                  }
                }
                return node;
              })
            );
          }
        }, 1000);

        return () => clearTimeout(timer);
      }, [nodes, sourceId, setNodes]);

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
            return registerNode({
              ...node,
              data: baseData,
            });
          }),
        [datasetDisplayLabel, registerNode]
      );

      const { isPipelineLoading } = usePipelineLoader({
        sourceId,
        datasetDisplayLabel,
        registerNode,
        prepareNodes,
        setNodes,
        setEdges,
        updateNodeCounter,
        scheduleFitView,
        onPipelineHydrated,
        onPipelineError,
      });

      // Rehydration check: When pipeline finishes loading, check for active jobs that might have been saved as 'idle'
      const [hasCheckedInitialJobs, setHasCheckedInitialJobs] = useState(false);

      useEffect(() => {
        setHasCheckedInitialJobs(false);
      }, [sourceId]);

      useEffect(() => {
        if (isPipelineLoading || hasCheckedInitialJobs || !sourceId || nodes.length === 0) {
          return;
        }

        // Ensure we have at least one node other than dataset-source to consider it loaded
        const hasContent = nodes.some((n) => n.id !== 'dataset-source');
        if (!hasContent && nodes.length > 1) {
           // If we have nodes but they are all dataset-source (unlikely), skip
           return;
        }

        const trainingNodes = nodes.filter((node) => {
          const type = node.data?.catalogType;
          return type === 'train_model_draft' || type === 'hyperparameter_tuning';
        });

        if (trainingNodes.length === 0) {
          setHasCheckedInitialJobs(true);
          return;
        }

        const checkJobs = async () => {
          const updates = new Map<string, { backgroundExecutionStatus: string; progress?: number }>();

          await Promise.all(
            trainingNodes.map(async (node) => {
              try {
                const catalogType = node.data?.catalogType;
                let latestJob = null;

                if (catalogType === 'train_model_draft') {
                  const res = await fetchTrainingJobs({
                    datasetSourceId: sourceId || undefined,
                    nodeId: node.id,
                    limit: 1,
                  });
                  latestJob = res.jobs?.[0];
                } else if (catalogType === 'hyperparameter_tuning') {
                  const res = await fetchHyperparameterTuningJobs({
                    datasetSourceId: sourceId || undefined,
                    nodeId: node.id,
                    limit: 1,
                  });
                  latestJob = res.jobs?.[0];
                }

                if (latestJob) {
                  const status = latestJob.status?.toLowerCase();
                  let nodeStatus = 'idle';
                  if (status === 'running' || status === 'queued') {
                    nodeStatus = 'loading';
                  } else if (status === 'succeeded') {
                    nodeStatus = 'success';
                  } else if (status === 'failed' || status === 'cancelled') {
                    nodeStatus = 'error';
                  }

                  const progress = typeof latestJob.progress === 'number' ? latestJob.progress : undefined;
                  
                  // Only update if status is active or different from current
                  if (nodeStatus === 'loading' || nodeStatus !== node.data?.backgroundExecutionStatus) {
                     updates.set(node.id, { backgroundExecutionStatus: nodeStatus, progress });
                  }
                }
              } catch (error) {
                // Ignore errors
              }
            })
          );

          if (updates.size > 0) {
            setNodes((currentNodes) =>
              currentNodes.map((node) => {
                if (updates.has(node.id)) {
                  const update = updates.get(node.id)!;
                  return { ...node, data: { ...(node.data || {}), ...update } };
                }
                return node;
              })
            );
          }
          setHasCheckedInitialJobs(true);
        };

        checkJobs();
      }, [isPipelineLoading, hasCheckedInitialJobs, sourceId, nodes, setNodes]);

      const clearGraph = useCallback(() => {
        setNodes((current) => {
          const existingDataset = current.find((node) => node.id === 'dataset-source');

          const datasetNode = registerNode(
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
        registerNode,
        scheduleFitView,
        setEdges,
        setNodes,
        setSelectedNodeId,
        setIsSettingsModalOpen,
      ]);

      useImperativeHandle(
        ref,
        () => ({
          openCatalog,
          closeCatalog,
          clearGraph,
        }),
        [clearGraph, closeCatalog, openCatalog]
      );

      const { isValidConnection, onConnect } = useConnectionHandlers({
        nodes,
        setEdges,
        scheduleNodeInternalsUpdate,
      });

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
        const catalogEntry = catalogEntryMapRef.current.get(catalogType);
        if (!catalogEntry) {
          return null;
        }
        return sanitizeDefaultConfigForNode(catalogEntry);
      }, [selectedNode]);

      const canResetSelectedNode = useMemo(() => {
        if (!selectedNode) {
          return false;
        }
        if (selectedNode.data?.isDataset) {
          return false;
        }
        const catalogType = selectedNode?.data?.catalogType;
        return isResettableCatalogEntry(typeof catalogType === 'string' ? catalogType : null, catalogEntryMapRef.current);
      }, [selectedNode]);

      const { sidebarContent } = useCanvasSidebar({
        nodeCatalog,
        isCatalogLoading,
        catalogErrorMessage,
        handleAddNode,
      });

      const handleNodeClick = useCallback(
        (_: React.MouseEvent, node: Node) => {
          handleOpenSettings(node.id);
        },
        [handleOpenSettings]
      );

      const pendingWarningMap = useMemo(() => {
        const map = new Map<string, string | null>();
        pendingConfigurationDetails.forEach(({ label, reason }) => {
          const normalized = normalizeLabel(label);
          if (!normalized) {
            return;
          }
          map.set(normalized, reason ?? null);
        });
        return map;
      }, [pendingConfigurationDetails]);

      const pendingHighlightSet = useMemo(() => new Set(pendingHighlightLabels), [pendingHighlightLabels]);

      useEffect(() => {
        if (!pendingWarningMap.size && !pendingHighlightSet.size) {
          setNodes((current) => {
            let changed = false;
            const next = current.map((node) => {
              const hasWarning = Boolean(node.data?.pendingWarningActive);
              const hasHighlight = Boolean(node.data?.pendingHighlight);
              if (!hasWarning && !hasHighlight) {
                return node;
              }
              changed = true;
              return {
                ...node,
                data: {
                  ...(node.data ?? {}),
                  pendingWarningActive: false,
                  pendingWarningReason: null,
                  pendingHighlight: false,
                },
              };
            });
            return changed ? next : current;
          });
          return;
        }

        setNodes((current) => {
          let changed = false;
          const next = current.map((node) => {
            const candidateLabels = [
              typeof node.data?.label === 'string' ? node.data?.label : null,
              typeof node.data?.title === 'string' ? node.data?.title : null,
              node.id,
            ];
            const normalizedLabel = candidateLabels.map((value) => normalizeLabel(value)).find(Boolean) ?? '';
            const hasPendingWarning = normalizedLabel ? pendingWarningMap.has(normalizedLabel) : false;
            const warningReason = hasPendingWarning && normalizedLabel ? pendingWarningMap.get(normalizedLabel) ?? null : null;
            const highlight = normalizedLabel ? pendingHighlightSet.has(normalizedLabel) : false;
            const existingReason = node.data?.pendingWarningReason ?? null;
            const existingHighlight = Boolean(node.data?.pendingHighlight);
            const existingWarningFlag = Boolean(node.data?.pendingWarningActive);
            if (
              existingReason === warningReason &&
              existingHighlight === highlight &&
              existingWarningFlag === hasPendingWarning
            ) {
              return node;
            }
            changed = true;
            return {
              ...node,
              data: {
                ...(node.data ?? {}),
                pendingWarningActive: hasPendingWarning,
                pendingWarningReason: warningReason,
                pendingHighlight: highlight,
              },
            };
          });
          return changed ? next : current;
        });
      }, [pendingHighlightSet, pendingWarningMap, setNodes]);

      return (
        <div
          className="canvas-stage"
          data-pipeline-loading={isPipelineLoading ? 'true' : 'false'}
          aria-busy={isPipelineLoading}
        >
          <CanvasViewport
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            onNodeClick={handleNodeClick}
            reactFlowInstanceRef={reactFlowInstanceRef}
            canvasViewportRef={canvasViewportRef}
            handleResetAllNodes={handleResetAllNodes}
            openCatalog={openCatalog}
          />

          <div className={`canvas-drawer${isCatalogOpen ? ' canvas-drawer--open' : ''}`}>
            <div className="canvas-drawer__header">
              <h2>Node catalog</h2>
              <button
                type="button"
                className="canvas-drawer__close"
                onClick={closeCatalog}
                aria-label="Close catalog"
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <X size={20} />
              </button>
            </div>
            <div className="canvas-drawer__body">{sidebarContent}</div>
          </div>

          {isCatalogOpen && <div className="canvas-drawer__backdrop" onClick={closeCatalog} />}

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
              onPendingConfigurationUpdate={handlePendingConfigurationUpdate}
              onPendingConfigurationCleared={handlePendingConfigurationCleared}
              onHighlightPendingNodes={handleHighlightPendingNodes}
              pendingConfigurationDetails={pendingConfigurationDetails}
            />
          )}

          {isPendingToastVisible && pendingToastDetails.length > 0 && (
            <PendingConfigurationDock
              details={pendingToastDetails}
              onDismiss={dismissPendingToast}
              onHighlight={handleHighlightPendingNodes}
            />
          )}
        </div>
      );
    }
  );

  export default CanvasShell;
