import React, { useState } from 'react';
import { registry } from '../../core/registry/NodeRegistry';
import { useGraphStore } from '../../core/store/useGraphStore';
import { Search } from 'lucide-react';

export const Sidebar: React.FC = () => {
  const nodes = registry.getAll();
  const addNode = useGraphStore((state) => state.addNode);
  const [searchTerm, setSearchTerm] = useState('');

  const handleDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const filteredNodes = nodes.filter(n => 
    n.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    n.category.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const categories = ['Data Source', 'Preprocessing', 'Modeling', 'Evaluation', 'Utility'];

  return (
    <aside className="w-64 shrink-0 border-r bg-background flex flex-col h-full shadow-sm z-10">
      <div className="p-4 border-b space-y-3">
        <div>
          <h2 className="font-semibold tracking-tight">Components</h2>
          <p className="text-xs text-muted-foreground">Drag and drop to canvas</p>
        </div>
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            placeholder="Search nodes..."
            className="w-full pl-8 pr-3 py-2 text-sm border rounded-md bg-muted/50 focus:bg-background focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin">
        {categories.map((category) => {
          const categoryNodes = filteredNodes.filter(n => n.category === category);
          if (categoryNodes.length === 0) return null;

          return (
            <div key={category}>
              <h3 className="text-[10px] font-bold text-muted-foreground mb-2 uppercase tracking-wider px-1">
                {category}
              </h3>
              <div className="space-y-2">
                {categoryNodes.map((node) => (
                  <div
                    key={node.type}
                    className="group flex items-center p-3 border rounded-lg bg-card hover:border-primary/50 hover:shadow-sm cursor-grab active:cursor-grabbing transition-all"
                    draggable
                    onDragStart={(e) => handleDragStart(e, node.type)}
                    onClick={() => addNode(node.type, { x: 100, y: 100 })}
                  >
                    <div className="p-2 bg-primary/5 group-hover:bg-primary/10 rounded-md mr-3 transition-colors">
                      {node.icon && <node.icon className="w-4 h-4 text-primary" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{node.label}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {node.description}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
};
