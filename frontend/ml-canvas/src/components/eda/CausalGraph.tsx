import React, { useEffect, useCallback, useState } from 'react';
import dagre from 'dagre';
import {
    ReactFlow,
    Background,
    Controls,
    useNodesState,
    useEdgesState,
    MarkerType,
    Node,
    Edge,
    Position
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { CausalGraphData } from '../../core/types/edaProfile';

interface CausalGraphProps {
    graph: CausalGraphData;
}

const nodeWidth = 150;
const nodeHeight = 40;

const getNodeStyle = (): React.CSSProperties => {
    const isDark = document.documentElement.classList.contains('dark');
    return {
        borderRadius: '5px',
        padding: '10px',
        width: nodeWidth,
        textAlign: 'center',
        borderWidth: '1px',
        borderStyle: 'solid',
        backgroundColor: isDark ? '#222529' : '#ffffff',
        color: isDark ? '#f3f4f6' : '#17102b',
        borderColor: isDark ? '#666d77' : '#b9b3c8',
    };
};

const getEdgeStrokeColor = (type: string): string => {
    const isDark = document.documentElement.classList.contains('dark');
    if (type === 'directed') return isDark ? '#9acbfa' : '#3b4bc4';
    return isDark ? '#abafb7' : '#6b6485';
};

/** Translate edge orientation to arrowheads at the corresponding endpoints. */
const getEdgeMarkers = (type: string, color: string): Pick<Edge, 'markerStart' | 'markerEnd'> => {
    const arrow = { type: MarkerType.ArrowClosed, color };
    if (type === 'bidirected') return { markerStart: arrow, markerEnd: arrow };
    if (type === 'directed') return { markerEnd: arrow };
    return {};
};

export const CausalGraph: React.FC<CausalGraphProps> = ({ graph }) => {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
    // Bumped whenever the document's `dark` class toggles, so the graph
    // styling effect below re-runs even when `graph` itself hasn't changed.
    const [themeVersion, setThemeVersion] = useState(0);

    useEffect(() => {
        const observer = new MutationObserver(() => {
            setThemeVersion((v) => v + 1);
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
        return () => observer.disconnect();
    }, []);

    const getLayoutedElements = useCallback((nodes: Node[], edges: Edge[]) => {
        try {
            const dagreGraph = new dagre.graphlib.Graph();
            dagreGraph.setDefaultEdgeLabel(() => ({}));

            // Switch to Left-Right layout for better flow
            // Increase separation significantly to avoid edge overlaps
            dagreGraph.setGraph({ rankdir: 'LR', nodesep: 80, ranksep: 200 });

            nodes.forEach((node) => {
                dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
            });

            edges.forEach((edge) => {
                dagreGraph.setEdge(edge.source, edge.target);
            });

            dagre.layout(dagreGraph);

            const newNodes = nodes.map((node) => {
                const nodeWithPosition = dagreGraph.node(node.id);
                // Fallback if dagre fails to position a node
                if (!nodeWithPosition) {
                    // nosemgrep: insecure-random-generator -- cosmetic fallback layout
                    // position only, not security-sensitive.
                    return {
                        ...node,
                        position: { x: Math.random() * 500, y: Math.random() * 500 }
                    };
                }
                return {
                    ...node,
                    targetPosition: Position.Left,
                    sourcePosition: Position.Right,
                    position: {
                        x: nodeWithPosition.x - nodeWidth / 2,
                        y: nodeWithPosition.y - nodeHeight / 2,
                    },
                };
            });

            return { nodes: newNodes, edges };
        } catch (error) {
            console.error("Dagre layout failed:", error);
            // Fallback: return nodes with random positions or grid
            const newNodes = nodes.map((node, index) => ({
                ...node,
                position: { x: (index % 3) * 250, y: Math.floor(index / 3) * 100 }
            }));
            return { nodes: newNodes, edges };
        }
    }, []);

    useEffect(() => {
        if (!graph || !graph.nodes || graph.nodes.length === 0) {
            setNodes([]);
            setEdges([]);
            return;
        }

        // Keep arbitrary column names distinct while giving both graph libraries opaque IDs.
        const nodeIds = new Map(graph.nodes.map((node, index) => [node.id, `causal-node-${index}`]));

        const initialNodes: Node[] = graph.nodes.map((n, index) => ({
            id: `causal-node-${index}`,
            data: { label: n.label },
            position: { x: 0, y: 0 },
            style: getNodeStyle(),
        }));

        const initialEdges: Edge[] = graph.edges.flatMap((e, i) => {
            const source = nodeIds.get(e.source);
            const target = nodeIds.get(e.target);
            if (source === undefined || target === undefined) return [];
            const stroke = getEdgeStrokeColor(e.type);
            return [{
                id: `e${i}`,
                source,
                target,
                animated: true,
                type: 'default',
                label: e.type === 'directed' ? 'causes' : (e.type === 'bidirected' ? 'confounded' : 'related'),
                labelStyle: { fill: 'hsl(var(--foreground))', fontSize: 12 },
                labelBgStyle: { fill: 'hsl(var(--card))' },
                style: { stroke, strokeDasharray: e.type === 'directed' ? '0' : '5 5' },
                ...getEdgeMarkers(e.type, stroke),
            }];
        });

        const layouted = getLayoutedElements(initialNodes, initialEdges);
        setNodes(layouted.nodes);
        setEdges(layouted.edges);

    }, [graph, getLayoutedElements, setNodes, setEdges, themeVersion]);

    return (
        <div className="space-y-2">
            <div style={{ width: '100%', height: '500px' }} className="border rounded-lg bg-gray-50 dark:bg-gray-900 relative">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    fitView
                    className="dark:bg-gray-900"
                >
                    <Background className="dark:bg-gray-900" />
                    <Controls className="dark:bg-gray-800 dark:text-white dark:border-gray-700" />
                </ReactFlow>
            </div>
        </div>
    );
};
