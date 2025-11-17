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
  Node,
  ReactFlowInstance,
  addEdge,
  useEdgesState,
  useNodesState,
  useUpdateNodeInternals,
} from 'react-flow-renderer';
import { useQuery } from '@tanstack/react-query';
import AnimatedEdge from '../../../components/edges/AnimatedEdge';
import ConnectionLine from '../../../components/edges/ConnectionLine';
import { FeatureCanvasSidebar } from '../../../components/FeatureCanvasSidebar';
import { NodeSettingsModal } from '../../../components/NodeSettingsModal';
import FeatureCanvasNode from '../FeatureCanvasNode/FeatureCanvasNode';
import { fetchNodeCatalog, fetchPipeline } from '../../../api';
import type { FeatureNodeCatalogEntry } from '../../../api';
import {
  cloneConfig,
  sanitizeDefaultConfigForNode,
  PENDING_CONFIRMATION_FLAG,
  isAutoConfirmedCatalogType,
} from '../../services/configSanitizer';
import {
  getDefaultEdges,
  getDefaultNodes,
  getSamplePipelineGraph,
  resolveDropPosition,
} from '../../services/layout';
import {
  registerNodeInteractions,
  createNewNode,
  isResettableCatalogEntry,
} from '../../services/nodeFactory';
import {
  computeActiveSplitMap,
  computeSplitConnectionMap,
  checkNodeConnectionStatus,
} from '../../services/splitPropagation';
import {
  areSplitArraysEqual,
  sanitizeSplitList,
  SPLIT_TYPE_ORDER,
} from '../../constants/splits';
import {
  CONNECTION_ACCEPT_MATCHERS,
  NODE_HANDLE_CONFIG,
  extractHandleKey,
} from '../../constants/nodeHandles';
import { buildGraphSnapshot } from '../../services/graphSerialization';
import type { FeatureNodeData } from '../../types/nodes';
import type { CanvasShellHandle, CanvasShellProps } from '../../types/pipeline';

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

            if (splitsChanged || connectionsChanged) {
              nodesWithHandleChanges.push(node.id);
            }

            didChange = true;
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

      const graphSnapshot = useMemo(() => buildGraphSnapshot(nodes, edges), [edges, nodes]);

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

      const pipelineQuery = useQuery({
        queryKey: ['feature-canvas', 'pipeline', sourceId],
        queryFn: () => fetchPipeline(sourceId as string),
        enabled: Boolean(sourceId),
        staleTime: 60 * 1000,
        retry: 1,
      });

      const isPipelineLoading = pipelineQuery.isLoading || pipelineQuery.isFetching;

      useEffect(() => {
        const map = new Map<string, FeatureNodeCatalogEntry>();
        nodeCatalog.forEach((entry) => {
          if (entry && typeof entry.type === 'string' && entry.type.trim()) {
            map.set(entry.type, entry);
          }
        });
        catalogEntryMapRef.current = map;
      }, [nodeCatalog]);

      const registerNode = useCallback(
        (node: Node) =>
          registerNodeInteractions(node, {
            handleOpenSettings,
            handleRemoveNode,
            catalogEntryMap: catalogEntryMapRef.current,
          }),
        [handleOpenSettings, handleRemoveNode]
      );

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
      }, [datasetDisplayLabel, registerNode, scheduleFitView, setEdges, setNodes, setSelectedNodeId, setIsSettingsModalOpen]);

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
            return registerNode({
              ...node,
              data: baseData,
            });
          }),
        [datasetDisplayLabel, registerNode]
      );

      const createNode = useCallback(
        (catalogNode?: FeatureNodeCatalogEntry | null, position?: { x: number; y: number }) => {
          const nextNumericId = nodeIdRef.current + 1;
          const nodeId = `node-${nextNumericId}`;
          nodeIdRef.current = nextNumericId;

          const basePosition =
            position ?? {
              x: 160 + ((nextNumericId - 1) % 4) * 220,
              y: 40 + Math.floor((nextNumericId - 1) / 4) * 160,
            };

          return registerNode(
            createNewNode(catalogNode ?? null, nodeId, basePosition, `Step ${nextNumericId}`, {
              handleOpenSettings,
              handleRemoveNode,
              catalogEntryMap: catalogEntryMapRef.current,
            })
          );
        },
        [handleOpenSettings, handleRemoveNode, registerNode]
      );

      const handleAddNode = useCallback(
        (catalogNode: FeatureNodeCatalogEntry) => {
          setNodes((current) => {
            const dropPosition = resolveDropPosition(current, reactFlowInstanceRef.current, canvasViewportRef.current);
            const newNode = createNode(catalogNode, dropPosition);
            return [...current, newNode];
          });
          scheduleFitView();
        },
        [createNode, scheduleFitView, setNodes]
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

              return registerNode({
                ...node,
                data: baseData,
              });
            })
          );
        },
        [registerNode, setNodes]
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

              return registerNode({
                ...node,
                data: baseData,
              });
            })
          );
        },
        [registerNode, setNodes]
      );

      const handleResetNodeConfig = useCallback(
        (nodeId: string, template?: Record<string, any> | null) => {
          setNodes((current) =>
            current.map((node) => {
              if (node.id !== nodeId) {
                return node;
              }

              const catalogType = node?.data?.catalogType;
              const catalogEntry = catalogType ? catalogEntryMapRef.current.get(catalogType) ?? null : null;
              const resolvedTemplate =
                template && typeof template === 'object' ? cloneConfig(template) : sanitizeDefaultConfigForNode(catalogEntry ?? null);

              const baseData = {
                ...(node.data ?? {}),
                config: resolvedTemplate,
                isConfigured: false,
                backgroundExecutionStatus: 'idle',
              };

              return registerNode({
                ...node,
                data: baseData,
              });
            })
          );
        },
        [registerNode, setNodes]
      );

      const handleResetAllNodes = useCallback(() => {
        setNodes((current) =>
          current.map((node) => {
            if (!isResettableCatalogEntry(node?.data?.catalogType, catalogEntryMapRef.current)) {
              return node;
            }

            const catalogType = node?.data?.catalogType;
            const catalogEntry = catalogType ? catalogEntryMapRef.current.get(catalogType) ?? null : null;
            const sanitizedConfig = sanitizeDefaultConfigForNode(catalogEntry ?? null);

            const baseData = {
              ...(node.data ?? {}),
              config: sanitizedConfig,
              isConfigured: false,
              backgroundExecutionStatus: 'idle',
            };

            return registerNode({
              ...node,
              data: baseData,
            });
          })
        );
      }, [registerNode, setNodes]);

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
        const datasetNodeLabel = `Dataset input\n(${datasetDisplayLabel})`;

        if (!sourceId) {
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
                return registerNode({
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
          setNodes((existing) =>
            existing.map((node) =>
              node.id === 'dataset-source'
                ? registerNode({
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
          const defaultNodes = getDefaultNodes().map((n) =>
            n.id === 'dataset-source'
              ? {
                  ...n,
                  data: {
                    ...(n.data ?? {}),
                    label: datasetNodeLabel,
                    isDataset: true,
                    isRemovable: false,
                  },
                }
              : n
          );
          const preparedNodes = prepareNodes(defaultNodes);
          setNodes(preparedNodes);
          setEdges([]);
          updateNodeCounter(preparedNodes);
          scheduleFitView();
          onPipelineHydrated?.({ nodes: preparedNodes, edges: [], pipeline: null, context: 'reset' });
          return;
        }

        if (pipelineQuery.data) {
          const graph = pipelineQuery.data.graph ?? {};
          const rawNodes = Array.isArray(graph?.nodes) && graph.nodes.length ? (graph.nodes as Node[]) : getDefaultNodes();
          const hydratedNodes = prepareNodes(rawNodes);
          const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
          const hydratedEdges = rawEdges.map((edge: any) => {
            const existingType = edge?.type;
            const type = !existingType || existingType === 'smoothstep' || existingType === 'default' ? 'animatedEdge' : existingType;
            const sourceNodeId = edge?.source;
            const targetNodeId = edge?.target;
            const targetNode = hydratedNodes.find((node) => node.id === targetNodeId);
            const normalizedSourceHandle = edge?.sourceHandle ?? (sourceNodeId ? `${sourceNodeId}-source` : undefined);
            const normalizedTargetHandle =
              edge?.targetHandle ?? (targetNode?.data?.isDataset ? undefined : targetNodeId ? `${targetNodeId}-target` : undefined);
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
          const defaultNodes = getDefaultNodes().map((n) =>
            n.id === 'dataset-source'
              ? {
                  ...n,
                  data: {
                    ...(n.data ?? {}),
                    label: datasetNodeLabel,
                    isDataset: true,
                    isRemovable: false,
                  },
                }
              : n
          );
          const preparedNodes = prepareNodes(defaultNodes);
          setNodes(preparedNodes);
          setEdges([]);
          updateNodeCounter(preparedNodes);
          scheduleFitView();
          onPipelineHydrated?.({ nodes: preparedNodes, edges: [], pipeline: null, context: 'sample' });
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
        registerNode,
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
        <div
          className="canvas-stage"
          data-pipeline-loading={isPipelineLoading ? 'true' : 'false'}
          aria-busy={isPipelineLoading}
        >
          <div
            className="canvas-stage__viewport"
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
                style: { strokeWidth: 3 },
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
              aria-label="Reset preprocessing nodes"
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
    }
  );

  export default CanvasShell;
