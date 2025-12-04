import React from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { registry } from '../../core/registry/NodeRegistry';
import { X, Settings2 } from 'lucide-react';

export const PropertiesPanel: React.FC = () => {
  const nodes = useGraphStore((state) => state.nodes);
  const updateNodeData = useGraphStore((state) => state.updateNodeData);
  const onNodesChange = useGraphStore((state) => state.onNodesChange);
  
  // Find the currently selected node
  const selectedNode = nodes.find((n) => n.selected);

  if (!selectedNode) {
    return (
      <aside className="w-80 border-l bg-background p-8 flex flex-col items-center justify-center text-muted-foreground text-center">
        <div className="p-4 bg-muted/50 rounded-full mb-4">
          <Settings2 className="w-8 h-8 opacity-50" />
        </div>
        <h3 className="font-medium mb-1">No Node Selected</h3>
        <p className="text-sm">Select a node on the canvas to configure its properties.</p>
      </aside>
    );
  }

  const handleClose = () => {
    onNodesChange([{ id: selectedNode.id, type: 'select', selected: false }]);
  };

  const definitionType = selectedNode.data.definitionType as string;
  const definition = registry.get(definitionType);

  if (!definition) {
    return (
      <aside className="w-80 border-l bg-background p-4">
        <div className="text-destructive flex items-center gap-2">
          <X className="w-4 h-4" />
          Error: Node definition '{definitionType}' not found.
        </div>
      </aside>
    );
  }

  const SettingsComponent = definition.settings;

  return (
    <aside className="w-80 border-l bg-background flex flex-col h-full shadow-xl z-20">
      <div className="p-4 border-b flex items-center justify-between bg-muted/10">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-md">
            {definition.icon && <definition.icon className="w-4 h-4 text-primary" />}
          </div>
          <div>
            <h2 className="font-semibold text-sm">{definition.label}</h2>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{definition.category}</p>
          </div>
        </div>
        <button 
          onClick={handleClose}
          className="p-1 hover:bg-muted rounded-md transition-colors text-muted-foreground hover:text-foreground"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {SettingsComponent ? (
          <SettingsComponent 
            config={selectedNode.data} 
            onChange={(newData) => updateNodeData(selectedNode.id, newData)} 
            nodeId={selectedNode.id}
          />
        ) : (
          <div className="p-8 text-center text-sm text-muted-foreground italic">
            No configuration options available for this node.
          </div>
        )}
      </div>
      
      <div className="p-3 border-t bg-muted/5 text-[10px] text-muted-foreground font-mono flex justify-between">
        <span>ID: {selectedNode.id}</span>
        <span>v1.0</span>
      </div>
    </aside>
  );
};
