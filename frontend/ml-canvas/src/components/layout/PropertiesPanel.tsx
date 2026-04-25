import React from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { registry } from '../../core/registry/NodeRegistry';
import {
  ExecutionMode,
  getExecutionMode,
  isAutoParallelType,
  supportsExecutionModeToggle,
} from '../../core/types/executionMode';
import { getMergeStrategy } from '../../core/types/nodeData';
import { X, Maximize2, Minimize2, Settings2, Merge } from 'lucide-react';
import { Node } from '@xyflow/react';

export const PropertiesPanel: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const { isSidebarOpen, isPropertiesPanelExpanded, setPropertiesPanelExpanded } = useViewStore();
  
  // Find the currently selected node
  const selectedNode = nodes.find((n) => n.selected);

  // Reset expansion when selection clears (optional, but good UX)
  React.useEffect(() => {
    if (!selectedNode) setPropertiesPanelExpanded(false);
  }, [selectedNode, setPropertiesPanelExpanded]);

  // Calculate width based on sidebar state
  // Sidebar is w-64 (256px). We leave some buffer.
  const expandedWidth = isSidebarOpen ? 'w-[calc(100vw-300px)]' : 'w-[calc(100vw-50px)]';

  return (
    <aside 
      className={`border-l bg-background shrink-0 transition-all duration-300 ease-in-out overflow-hidden ${
        selectedNode ? (isPropertiesPanelExpanded ? `${expandedWidth} opacity-100` : 'w-80 opacity-100') : 'w-0 opacity-0'
      }`}
    >
      {selectedNode && (
        <PropertiesContent 
          selectedNode={selectedNode} 
          isExpanded={isPropertiesPanelExpanded} 
          toggleExpand={() => setPropertiesPanelExpanded(!isPropertiesPanelExpanded)} 
        />
      )}
    </aside>
  );
};

const PropertiesContent: React.FC<{ 
  selectedNode: Node; 
  isExpanded: boolean; 
  toggleExpand: () => void; 
}> = ({ selectedNode, isExpanded, toggleExpand }) => {
  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);

  const handleClose = () => {
    onNodesChange([{ id: selectedNode.id, type: 'select', selected: false }]);
  };

  const definitionType = selectedNode.data.definitionType as string;
  const definition = registry.get(definitionType);

  if (!definition) {
    return (
      <div className="p-4">
        <div className="text-destructive flex items-center gap-2">
          <X className="w-4 h-4" />
          Error: Node definition &apos;{definitionType}&apos; not found.
        </div>
      </div>
    );
  }

  const SettingsComponent = definition.settings;

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b flex items-center justify-between bg-muted/30">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-primary/10 rounded-md">
            <Settings2 className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-sm">{String(selectedNode.data.label || definition.label)}</h2>
            <div className="text-xs text-muted-foreground font-mono">ID: {selectedNode.id}</div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button 
            onClick={toggleExpand}
            className="p-1.5 hover:bg-accent rounded-md text-muted-foreground hover:text-foreground transition-colors"
          >
            {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
          <button 
            onClick={handleClose}
            className="p-1.5 hover:bg-accent rounded-md text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-6">
          <SettingsComponent
            config={selectedNode.data}
            onChange={(data: unknown) => updateNodeData(selectedNode.id, data)}
            nodeId={selectedNode.id}
          />
          <MultiInputModeSection selectedNode={selectedNode} />
          <MergeStrategySection selectedNode={selectedNode} />
        </div>
      </div>
    </div>
  );
};

/**
 * Modeling nodes can either merge their multiple upstream inputs into one
 * dataset or fan out and run each input as a separate experiment. This toggle
 * lives in the properties panel so it sits right above the related Merge
 * Strategy dropdown instead of being buried in each settings panel's footer.
 */
const MultiInputModeSection: React.FC<{ selectedNode: Node }> = ({ selectedNode }) => {
  const edges = useGraphStore((state) => state.edges);
  const setExecutionMode = useGraphStore((state) => state.setExecutionMode);

  const definitionType = selectedNode.data.definitionType as string;

  // Only modeling nodes opt in to this toggle today.
  const supportsToggle = supportsExecutionModeToggle(definitionType);
  const incomingSourceCount = new Set(
    edges.filter((e) => e.target === selectedNode.id).map((e) => e.source)
  ).size;

  if (!supportsToggle || incomingSourceCount < 2) return null;

  const current: ExecutionMode = getExecutionMode(selectedNode.data);

  return (
    <div className="border-t pt-4">
      <div className="flex items-center gap-2 mb-2">
        <Settings2 className="w-4 h-4 text-muted-foreground" />
        <h3 className="text-sm font-semibold">Multi-Input Mode</h3>
      </div>
      <p className="text-xs text-muted-foreground mb-2">
        Merge combines all inputs into one dataset. Parallel runs each input as a separate experiment.
      </p>
      <div className="flex rounded-md overflow-hidden border border-slate-300 dark:border-slate-600 text-xs font-medium w-fit">
        <button
          onClick={() => setExecutionMode(selectedNode.id, 'merge')}
          className={`px-3 py-1.5 transition-colors ${
            current === 'merge'
              ? 'bg-purple-500 text-white'
              : 'bg-white dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-600'
          }`}
        >
          Merge
        </button>
        <button
          onClick={() => setExecutionMode(selectedNode.id, 'parallel')}
          className={`px-3 py-1.5 transition-colors ${
            current === 'parallel'
              ? 'bg-blue-500 text-white'
              : 'bg-white dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-600'
          }`}
        >
          Parallel
        </button>
      </div>
    </div>
  );
};

const MergeStrategySection: React.FC<{ selectedNode: Node }> = ({ selectedNode }) => {
  const edges = useGraphStore((state) => state.edges);
  const updateNodeData = useGraphStore((state) => state.updateNodeData);

  const definitionType = selectedNode.data.definitionType as string;
  const definition = registry.get(definitionType);
  const canMerge = (definition?.inputs?.length ?? 0) > 0;
  const incomingSourceCount = new Set(
    edges.filter((e) => e.target === selectedNode.id).map((e) => e.source)
  ).size;

  // Auto-parallel terminals (data_preview) render each input in its own
  // tab instead of merging columns, so the merge-strategy dropdown is
  // meaningless for them. Sourced from `core/types/executionMode` so the
  // canvas / engine / UI all agree on which types are auto-parallel.
  const isAutoParallel = isAutoParallelType(definitionType);

  // Modeling nodes expose an explicit Multi-Input Mode toggle (merge / parallel).
  // When the user picks "parallel", merging is skipped at runtime, so the
  // strategy dropdown would be misleading. Hide it in that case.
  const isParallelMode = getExecutionMode(selectedNode.data) === 'parallel';

  // Only expose the strategy when the node actually merges: multi-input
  // node with 2+ distinct upstream sources, not an auto-parallel terminal,
  // and not explicitly set to parallel execution.
  if (!canMerge || incomingSourceCount < 2 || isAutoParallel || isParallelMode) return null;

  const current = getMergeStrategy(selectedNode.data);

  return (
    <div className="border-t pt-4">
      <div className="flex items-center gap-2 mb-2">
        <Merge className="w-4 h-4 text-muted-foreground" />
        <h3 className="text-sm font-semibold">Merge Strategy</h3>
      </div>
      <p className="text-xs text-muted-foreground mb-2">
        How to resolve columns present in more than one input.
      </p>
      <select
        value={current}
        onChange={(e) => updateNodeData(selectedNode.id, { merge_strategy: e.target.value })}
        className="w-full px-2 py-1.5 text-sm bg-background border rounded-md"
      >
        <option value="last_wins">Last wins (default) - downstream input overwrites</option>
        <option value="first_wins">First wins - earlier input kept</option>
      </select>
    </div>
  );
};
