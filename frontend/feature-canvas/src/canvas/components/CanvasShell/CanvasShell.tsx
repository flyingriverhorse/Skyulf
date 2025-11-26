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
              >
                ×
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
            />
          )}
        </div>
      );
    }
  );

  export default CanvasShell;
