import React from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { registry } from '../../core/registry/NodeRegistry';
import { X, Maximize2, Minimize2, Settings2 } from 'lucide-react';

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
  selectedNode: any; 
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
          Error: Node definition '{definitionType}' not found.
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
            <h2 className="font-semibold text-sm">{selectedNode.data.label || definition.label}</h2>
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
            onChange={(data: any) => updateNodeData(selectedNode.id, data)}
            nodeId={selectedNode.id}
          />
        </div>
      </div>
    </div>
  );
};
