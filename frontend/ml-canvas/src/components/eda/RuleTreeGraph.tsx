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
    Position,
    Handle
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Info, GitFork, CheckCircle } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip';

interface RuleNodeData {
    id: number;
    feature?: string;
    threshold?: number;
    impurity: number;
    samples: number;
    value: number[];
    class_name?: string;
    is_leaf: boolean;
    children: number[];
}

interface RuleTreeGraphProps {
    tree: {
        nodes: RuleNodeData[];
        accuracy?: number;
    };
}

const nodeWidth = 220;
const nodeHeight = 100;

// Custom Node Component
const DecisionNode = ({ data }: { data: any }) => {
    const isLeaf = data.is_leaf;
    
    // Calculate class distribution percentage
    const total = data.value.reduce((a: number, b: number) => a + b, 0);
    const maxVal = Math.max(...data.value);
    const purity = total > 0 ? (maxVal / total) * 100 : 0;

    return (
        <div className={`px-4 py-3 shadow-md rounded-md border-2 ${isLeaf ? 'border-green-500 bg-green-50 dark:bg-green-900/20' : 'border-blue-500 bg-white dark:bg-gray-800'} min-w-[200px]`}>
            <Handle type="target" position={Position.Left} className="w-3 h-3 !bg-gray-400" />
            
            <div className="flex flex-col gap-1">
                {/* Header: Decision Rule */}
                <div className="font-bold text-sm text-gray-900 dark:text-white border-b pb-1 mb-1">
                    {isLeaf ? (
                        <span className="flex items-center gap-2 text-green-700 dark:text-green-400">
                            <CheckCircle className="w-4 h-4" />
                            Prediction: {data.class_name}
                        </span>
                    ) : (
                        <span className="flex items-center gap-2 text-blue-700 dark:text-blue-400">
                            <GitFork className="w-4 h-4" />
                            {data.feature} {data.threshold !== undefined ? `<= ${data.threshold.toFixed(2)}` : ''}
                        </span>
                    )}
                </div>

                {/* Body: Stats */}
                <div className="text-xs text-gray-600 dark:text-gray-300 space-y-1">
                    <TooltipProvider>
                        <div className="flex justify-between">
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <span className="cursor-help border-b border-dotted border-gray-400">Samples:</span>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p>The number of data points that fall into this rule bucket.</p>
                                </TooltipContent>
                            </Tooltip>
                            <span className="font-mono">{data.samples}</span>
                        </div>
                        <div className="flex justify-between">
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <span className="cursor-help border-b border-dotted border-gray-400">Purity:</span>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <p className="max-w-xs">The percentage of samples in this node that belong to the majority class. Higher purity means a clearer separation.</p>
                                </TooltipContent>
                            </Tooltip>
                            <span className={`font-mono ${purity > 80 ? 'text-green-600 font-bold' : ''}`}>
                                {purity.toFixed(1)}%
                            </span>
                        </div>
                    </TooltipProvider>
                    {/* Class Distribution Bar */}
                    <div className="w-full h-1.5 bg-gray-200 rounded-full mt-1 overflow-hidden flex">
                        {data.value.map((val: number, idx: number) => (
                            <div 
                                key={idx}
                                style={{ width: `${(val / total) * 100}%`, backgroundColor: `hsl(${idx * 137.5}, 70%, 50%)` }}
                                title={`Class ${idx}: ${val}`}
                            />
                        ))}
                    </div>
                </div>
            </div>

            <Handle type="source" position={Position.Right} className="w-3 h-3 !bg-gray-400" />
        </div>
    );
};

const nodeTypes = {
    decision: DecisionNode,
};

export const RuleTreeGraph: React.FC<RuleTreeGraphProps> = ({ tree }) => {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

    const getLayoutedElements = useCallback((nodes: Node[], edges: Edge[]) => {
        try {
            const dagreGraph = new dagre.graphlib.Graph();
            dagreGraph.setDefaultEdgeLabel(() => ({}));

            // Left-Right layout (Decomposition Tree style)
            dagreGraph.setGraph({ rankdir: 'LR', nodesep: 50, ranksep: 150 });

            nodes.forEach((node) => {
                dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
            });

            edges.forEach((edge) => {
                dagreGraph.setEdge(edge.source, edge.target);
            });

            dagre.layout(dagreGraph);

            const newNodes = nodes.map((node) => {
                const nodeWithPosition = dagreGraph.node(node.id);
                return {
                    ...node,
                    targetPosition: Position.Left,
                    sourcePosition: Position.Right,
                    position: {
                        x: nodeWithPosition ? nodeWithPosition.x - nodeWidth / 2 : 0,
                        y: nodeWithPosition ? nodeWithPosition.y - nodeHeight / 2 : 0,
                    },
                };
            });

            return { nodes: newNodes, edges };
        } catch (error) {
            console.error("Dagre layout failed:", error);
            return { nodes, edges };
        }
    }, []);

    useEffect(() => {
        if (!tree || !tree.nodes || tree.nodes.length === 0) return;

        // Convert RuleNodes to ReactFlow Nodes
        const flowNodes: Node[] = tree.nodes.map(n => ({
            id: n.id.toString(),
            type: 'decision',
            data: { ...n },
            position: { x: 0, y: 0 }, // Calculated by layout
        }));

        // Create Edges based on children
        const flowEdges: Edge[] = [];
        tree.nodes.forEach(n => {
            if (n.children && n.children.length > 0) {
                // Left Child (True path usually)
                if (n.children[0] !== undefined) {
                    flowEdges.push({
                        id: `e${n.id}-${n.children[0]}`,
                        source: n.id.toString(),
                        target: n.children[0].toString(),
                        label: 'True',
                        type: 'smoothstep',
                        style: { stroke: '#22c55e', strokeWidth: 2 },
                        labelStyle: { fill: '#22c55e', fontWeight: 700 },
                        markerEnd: { type: MarkerType.ArrowClosed, color: '#22c55e' },
                    });
                }
                // Right Child (False path usually)
                if (n.children[1] !== undefined) {
                    flowEdges.push({
                        id: `e${n.id}-${n.children[1]}`,
                        source: n.id.toString(),
                        target: n.children[1].toString(),
                        label: 'False',
                        type: 'smoothstep',
                        style: { stroke: '#ef4444', strokeWidth: 2 },
                        labelStyle: { fill: '#ef4444', fontWeight: 700 },
                        markerEnd: { type: MarkerType.ArrowClosed, color: '#ef4444' },
                    });
                }
            }
        });

        const layouted = getLayoutedElements(flowNodes, flowEdges);
        setNodes(layouted.nodes);
        setEdges(layouted.edges);

    }, [tree, getLayoutedElements, setNodes, setEdges]);

    return (
        <div className="space-y-2">
            <div style={{ width: '100%', height: '600px' }} className="border rounded-lg bg-gray-50 dark:bg-gray-900 relative">
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    nodeTypes={nodeTypes}
                    fitView
                    className="dark:bg-gray-900"
                >
                    <Background className="dark:bg-gray-900" gap={16} size={1} />
                    <Controls className="dark:bg-gray-800 dark:text-white dark:border-gray-700" />
                </ReactFlow>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 italic flex items-center gap-1">
                <Info className="w-3 h-3" />
                This tree is a <strong>surrogate (explanatory) model</strong> trained on your data to surface readable rules.
                Do not treat it as a production-grade predictor. Green paths mean the condition is met (True); red paths mean it is not (False).
            </p>
        </div>
    );
};
