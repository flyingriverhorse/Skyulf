import React, { useEffect, useCallback } from 'react';
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
import { Info } from 'lucide-react';

interface CausalGraphProps {
    graph: {
        nodes: { id: string; label: string }[];
        edges: { source: string; target: string; type: string }[];
    };
}

const nodeWidth = 150;
const nodeHeight = 40;

export const CausalGraph: React.FC<CausalGraphProps> = ({ graph }) => {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

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
        if (!graph || !graph.nodes || graph.nodes.length === 0) return;

        // Sanitize IDs to ensure they are safe strings
        const sanitizeId = (id: string) => id.replace(/[^a-zA-Z0-9-_]/g, '_');

        const initialNodes: Node[] = graph.nodes.map(n => ({
            id: sanitizeId(n.id),
            data: { label: n.label },
            position: { x: 0, y: 0 }, // Will be calculated
            className: 'dark:bg-gray-800 dark:text-white dark:border-gray-600 bg-white text-gray-900 border-gray-300',
            style: { 
                borderRadius: '5px',
                padding: '10px',
                width: nodeWidth,
                textAlign: 'center',
                borderWidth: '1px',
                borderStyle: 'solid'
            }
        }));

        const initialEdges: Edge[] = graph.edges.map((e, i) => ({
            id: `e${i}`,
            source: sanitizeId(e.source),
            target: sanitizeId(e.target),
            animated: true,
            type: 'default', // Bezier curves are smoother for LR layout
            label: e.type === 'directed' ? 'causes' : (e.type === 'bidirected' ? 'confounded' : 'related'),
            style: { stroke: e.type === 'directed' ? '#333' : '#999', strokeDasharray: e.type === 'directed' ? '0' : '5 5' },
            markerEnd: e.type === 'directed' ? { type: MarkerType.ArrowClosed } : undefined,
            markerStart: e.type === 'bidirected' ? { type: MarkerType.ArrowClosed } : undefined,
        }));

        const layouted = getLayoutedElements(initialNodes, initialEdges);
        setNodes(layouted.nodes);
        setEdges(layouted.edges);

    }, [graph, getLayoutedElements, setNodes, setEdges]);

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
            <p className="text-xs text-gray-500 dark:text-gray-400 italic flex items-center gap-1">
                <Info className="w-3 h-3" />
                To ensure fast performance, this graph displays the <strong>Target</strong> and the top <strong>14 features</strong> most correlated with it.
            </p>
        </div>
    );
};
