import React, { useId, useRef, useState } from 'react';
import { registry } from '../../core/registry/NodeRegistry';
import { useGraphStore } from '../../core/store/useGraphStore';
import { useViewStore } from '../../core/store/useViewStore';
import { FOCUS_NODE_EVENT } from '../../core/hooks/useKeyboardShortcuts';
import { Search, PanelLeftClose, PanelLeftOpen, ChevronDown, ChevronRight } from 'lucide-react';

export const Sidebar: React.FC = () => {
  // Legacy node types (e.g. the old Basic Training / Advanced Tuning nodes,
  // superseded by the unified `TrainingNode`) stay registered for backward
  // compatibility but are excluded from the drag-and-drop palette.
  const nodes = registry.getAll().filter((n) => !n.hidden);
  const addNode = useGraphStore((state) => state.addNode);
  const { isSidebarOpen, setSidebarOpen } = useViewStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({});
  const categoryListId = useId();
  // Cascades click-to-add nodes so repeated clicks don't stack them on top of each other.
  const placementCounterRef = useRef(0);

  const handleDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  // CAN-001: the previous 30 px diagonal step was smaller than a node
  // card's `min-w-[200px]`, so consecutive click-adds landed mostly on
  // top of each other and the older card intercepted clicks meant for
  // the new one. 260/220 px steps clear a card's width/typical height;
  // wrapping into a new column every 5 nodes and a new "sheet" every 4
  // rows keeps the cascade from drifting arbitrarily far off-screen
  // before FlowCanvas pans/zooms the latest node into view below.
  const COLUMNS = 5;
  const ROWS = 4;
  const STEP_X = 260;
  const STEP_Y = 220;

  const handleAddNodeClick = (nodeType: string) => {
    const step = placementCounterRef.current % (COLUMNS * ROWS);
    placementCounterRef.current += 1;
    const col = step % COLUMNS;
    const row = Math.floor(step / COLUMNS);
    const id = addNode(nodeType, { x: 100 + col * STEP_X, y: 100 + row * STEP_Y });
    if (id) {
      window.dispatchEvent(new CustomEvent(FOCUS_NODE_EVENT, { detail: { id } }));
    }
  };


  if (!isSidebarOpen) {
    return (
      <div className="absolute left-4 top-4 z-10">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-2 bg-background border shadow-md rounded-md text-muted-foreground hover:text-foreground transition-colors"
          title="Expand Components"
          aria-label="Expand components sidebar"
        >
          <PanelLeftOpen className="w-5 h-5" />
        </button>
      </div>
    );
  }

  const searchQuery = searchTerm.trim().toLowerCase();
  const filteredNodes = nodes.filter(n =>
    n.label.toLowerCase().includes(searchQuery) ||
    n.category.toLowerCase().includes(searchQuery) ||
    n.description.toLowerCase().includes(searchQuery)
  );

  const categories = ['Data Source', 'Preprocessing', 'Modeling', 'Evaluation', 'Utility'];

  return (
    <aside className="w-64 shrink-0 border-r bg-background flex flex-col h-full shadow-sm z-10 transition-all duration-300">
      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold tracking-tight">Components</h2>
            <p className="text-xs text-muted-foreground">Click or drag to add a node</p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 hover:bg-accent rounded-md text-muted-foreground hover:text-foreground transition-colors"
            title="Collapse Sidebar"
            aria-label="Collapse sidebar"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        </div>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground pointer-events-none" aria-hidden="true" />
          <input
            placeholder="Search nodes..."
            aria-label="Search nodes"
            className="w-full pl-9 pr-3 py-2 text-sm border rounded-md bg-muted/50 focus:bg-background focus:outline-none focus:ring-1 focus:ring-primary transition-colors"
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); }}
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin">
        {filteredNodes.length === 0 && (
          <div className="text-center py-8 px-2">
            <p className="text-sm font-medium text-muted-foreground">No components found</p>
            <p className="text-xs text-muted-foreground mt-1">Try a different search term.</p>
          </div>
        )}
        {categories.map((category, index) => {
          const categoryNodes = filteredNodes.filter(n => n.category === category);
          if (categoryNodes.length === 0) return null;
          // Search reveals every matching group without changing the browsing layout.
          const isSearching = searchQuery.length > 0;
          const isExpanded = isSearching || !collapsedCategories[category];
          const contentId = `${categoryListId}-${index}`;

          return (
            <div key={category}>
              <h3 className="mb-2">
                <button
                  type="button"
                  aria-expanded={isExpanded}
                  aria-controls={contentId}
                  disabled={isSearching}
                  title={isSearching ? 'Clear search to collapse categories' : undefined}
                  onClick={() => setCollapsedCategories(current => ({ ...current, [category]: !current[category] }))}
                  className="flex w-full items-center gap-1.5 rounded px-1 py-1 text-left text-[10px] font-bold uppercase tracking-wider text-muted-foreground enabled:hover:bg-accent enabled:hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                >
                  {isExpanded
                    ? <ChevronDown aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
                    : <ChevronRight aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />}
                  <span className="flex-1">{category}</span>
                  <span aria-hidden="true" className="rounded bg-muted px-1.5 py-0.5 tabular-nums">{categoryNodes.length}</span>
                </button>
              </h3>
              <div id={contentId} hidden={!isExpanded} className="space-y-2">
                {categoryNodes.map((node) => (
                  <button
                    type="button"
                    key={node.type}
                    data-testid={`sidebar-node-${node.type}`}
                    aria-label={`Add ${node.label} node`}
                    className="group flex w-full items-center p-3 border rounded-lg bg-card text-left hover:border-primary/50 hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background cursor-grab active:cursor-grabbing transition-all"
                    draggable
                    onDragStart={(e) => { handleDragStart(e, node.type); }}
                    onClick={() => { handleAddNodeClick(node.type); }}
                  >
                    <span className="p-2 bg-primary/5 group-hover:bg-primary/10 rounded-md mr-3 transition-colors">
                      {node.icon && <node.icon className="w-4 h-4 text-primary" />}
                    </span>
                    <span className="flex-1 min-w-0">
                      <span className="block text-sm font-medium truncate">{node.label}</span>
                      <span className="block text-xs text-muted-foreground truncate">
                        {node.description}
                      </span>
                    </span>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
};
