import React, { useState, useEffect, useRef } from 'react';
import { EDAService, Filter } from '../../core/api/eda';
import { Plus, X, Loader2 } from 'lucide-react';

interface TreeItem {
    name: string;
    value: number;
    ratio: number;
}

interface TreeLevel {
    id: string;
    splitColumn: string | null; // The column used to generate this level's items
    items: TreeItem[];
    selectedItemName: string | null;
    filters: Filter[]; // Filters applied to reach this level
}

interface DecompositionTreeProps {
    datasetId: number;
    measureCol: string;
    measureAgg: string;
    columns: string[];
    initialFilters: Filter[];
}

// Module-level cache to persist state across tab switches
const treeCache: Record<string, { levels: TreeLevel[], splitPath: (string|null)[] }> = {};

export const DecompositionTree: React.FC<DecompositionTreeProps> = ({
    datasetId,
    measureCol,
    measureAgg,
    columns,
    initialFilters
}) => {
    const [levels, setLevels] = useState<TreeLevel[]>([]);
    const [loading, setLoading] = useState(false);
    const [splitMenuOpen, setSplitMenuOpen] = useState<{ levelIndex: number, item: TreeItem, x: number, y: number } | null>(null);
    
    // We need to store the "structure" of the tree (the sequence of split columns)
    // so we can automatically refresh subsequent levels when a parent selection changes.
    const [splitPath, setSplitPath] = useState<(string | null)[]>([null]); // Root is null

    const containerRef = useRef<HTMLDivElement>(null);
    const innerRef = useRef<HTMLDivElement>(null);
    const [paths, setPaths] = useState<{ d: string }[]>([]);

    // Auto-scroll to right when levels change
    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTo({ left: containerRef.current.scrollWidth, behavior: 'smooth' });
        }
    }, [levels.length]);

    const updatePaths = () => {
        if (!innerRef.current) return;

        const newPaths: { d: string }[] = [];
        const innerRect = innerRef.current.getBoundingClientRect();

        levels.forEach((level, index) => {
            if (index >= levels.length - 1) return;
            
            const selectedItemName = level.selectedItemName;
            if (!selectedItemName) return;

            const startId = `node-${index}-${selectedItemName}`;
            const startEl = document.getElementById(startId);

            if (startEl) {
                const startRect = startEl.getBoundingClientRect();
                
                // Calculate start point (right side of selected item)
                const x1 = startRect.right - innerRect.left;
                const y1 = startRect.top + (startRect.height / 2) - innerRect.top;
                
                // Determine end point
                let x2, y2;
                
                const nextLevel = levels[index + 1];
                if (nextLevel.selectedItemName) {
                    // If next level has a selection, connect to that item
                    const endId = `node-${index + 1}-${nextLevel.selectedItemName}`;
                    const endEl = document.getElementById(endId);
                    
                    if (endEl) {
                        const endRect = endEl.getBoundingClientRect();
                        x2 = endRect.left - innerRect.left;
                        y2 = endRect.top + (endRect.height / 2) - innerRect.top;
                    } else {
                        // Fallback if element not found yet
                        return;
                    }
                } else {
                    // If no selection in next level, connect to the left-center of the list container
                    // We need an ID for the list container
                    const listId = `list-${index + 1}`;
                    const listEl = document.getElementById(listId);
                    
                    if (listEl) {
                        const listRect = listEl.getBoundingClientRect();
                        x2 = listRect.left - innerRect.left;
                        // Point to the vertical center of the visible list area, or just the top area?
                        // Let's point to the top-ish area so it looks like it feeds into the list
                        // Or maybe the center of the container height?
                        y2 = listRect.top + (listRect.height / 2) - innerRect.top;
                    } else {
                        return;
                    }
                }

                if (x2 !== undefined && y2 !== undefined) {
                    // Draw a sigmoid curve
                    const controlPointX = (x1 + x2) / 2;
                    const d = `M ${x1} ${y1} C ${controlPointX} ${y1}, ${controlPointX} ${y2}, ${x2} ${y2}`;
                    newPaths.push({ d });
                }
            }
        });

        setPaths(newPaths);
    };

    // Calculate connection paths
    React.useLayoutEffect(() => {
        updatePaths();
    }, [levels, splitMenuOpen]); // Recalculate when levels change or menu opens (layout might shift)

    // Update cache when state changes
    useEffect(() => {
        if (levels.length > 0) {
            const cacheKey = `${datasetId}-${measureCol}-${measureAgg}`;
            treeCache[cacheKey] = { levels, splitPath };
        }
    }, [levels, splitPath, datasetId, measureCol, measureAgg]);

    // Initialize Root or Load from Cache
    useEffect(() => {
        const cacheKey = `${datasetId}-${measureCol}-${measureAgg}`;
        if (treeCache[cacheKey]) {
            setLevels(treeCache[cacheKey].levels);
            setSplitPath(treeCache[cacheKey].splitPath);
        } else {
            loadRoot();
        }
    }, [datasetId, measureCol, measureAgg]);

    const loadRoot = async () => {
        setLoading(true);
        try {
            const res = await EDAService.getDecomposition(
                datasetId,
                measureCol === 'count' ? null : measureCol,
                measureAgg,
                '',
                initialFilters
            );
            
            const rootLevel: TreeLevel = {
                id: 'root',
                splitColumn: null,
                items: res,
                selectedItemName: null,
                filters: initialFilters
            };
            
            setLevels([rootLevel]);
            setSplitPath([null]);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleHeaderSplitClick = (levelIndex: number, event: React.MouseEvent) => {
        // Only allow splitting if an item is selected in this level
        const level = levels[levelIndex];
        if (!level.selectedItemName) {
            // Maybe show a toast or tooltip? For now just return.
            return;
        }

        const item = level.items.find(i => i.name === level.selectedItemName);
        if (!item) return;

        // Open Split Menu
        const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
        const containerRect = containerRef.current?.getBoundingClientRect();
        
        if (containerRect) {
            let x = rect.right - containerRect.left + 10;
            let y = rect.top - containerRect.top;

            const wrapperWidth = containerRef.current?.parentElement?.clientWidth || window.innerWidth;
            if (x + 260 > wrapperWidth) {
                x = rect.left - containerRect.left - 270; 
            }

            const wrapperHeight = containerRef.current?.parentElement?.clientHeight || window.innerHeight;
            if (y + 300 > wrapperHeight) {
                y = wrapperHeight - 310;
            }

            setSplitMenuOpen({
                levelIndex,
                item,
                x,
                y
            });
        }
    };

    const handleItemClick = async (levelIndex: number, item: TreeItem) => {
        // 1. Update selection in current level
        const newLevels = [...levels];
        newLevels[levelIndex].selectedItemName = item.name;
        
        // Check if we have a next level defined in our split path
        const nextSplitCol = splitPath[levelIndex + 1];
        
        if (nextSplitCol) {
            // AUTOMATIC UPDATE: If next level exists, refresh it instead of opening menu
            setLoading(true);
            try {
                // Construct filters for the next level
                const currentLevel = newLevels[levelIndex];
                const newFilters = [...currentLevel.filters];
                if (currentLevel.splitColumn) {
                    newFilters.push({
                        column: currentLevel.splitColumn,
                        operator: '==',
                        value: item.name
                    });
                } else if (levelIndex === 0) {
                    // Root level
                }

                // Fetch data for next level
                const res = await EDAService.getDecomposition(
                    datasetId,
                    measureCol === 'count' ? null : measureCol,
                    measureAgg,
                    nextSplitCol,
                    newFilters
                );

                // Update next level
                const nextLevel: TreeLevel = {
                    id: `level-${levelIndex + 1}`,
                    splitColumn: nextSplitCol,
                    items: res,
                    selectedItemName: null, // Reset selection in next level
                    filters: newFilters
                };
                
                // Replace next level and remove any levels AFTER it (because selection in next level is reset)
                const finalLevels = [...newLevels.slice(0, levelIndex + 1), nextLevel];
                setLevels(finalLevels);
                
                // Update split path to match (remove anything after next level)
                // Actually, if we reset selection in next level, we lose the path after it.
                // So we should truncate splitPath too?
                // Yes, because if Level 2 selection is gone, Level 3 cannot exist.
                setSplitPath(splitPath.slice(0, levelIndex + 2));

            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        } else {
            // No next level defined, just update selection
            setLevels(newLevels);
            // Do NOT open menu automatically anymore. User must click header + button.
        }
    };

    const handleSplit = async (column: string) => {
        if (!splitMenuOpen) return;
        
        const { levelIndex, item } = splitMenuOpen;
        setSplitMenuOpen(null);
        setLoading(true);

        try {
            const currentLevel = levels[levelIndex];
            const newFilters = [...currentLevel.filters];
            if (currentLevel.splitColumn) {
                newFilters.push({
                    column: currentLevel.splitColumn,
                    operator: '==',
                    value: item.name
                });
            }

            const res = await EDAService.getDecomposition(
                datasetId,
                measureCol === 'count' ? null : measureCol,
                measureAgg,
                column,
                newFilters
            );

            const newLevel: TreeLevel = {
                id: `level-${levelIndex + 1}`,
                splitColumn: column,
                items: res,
                selectedItemName: null,
                filters: newFilters
            };

            const newLevels = [...levels.slice(0, levelIndex + 1), newLevel];
            setLevels(newLevels);
            
            // Update Split Path
            const newSplitPath = [...splitPath.slice(0, levelIndex + 1), column];
            setSplitPath(newSplitPath);

        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const closeSplit = (levelIndex: number) => {
        setLevels(levels.slice(0, levelIndex));
        setSplitPath(splitPath.slice(0, levelIndex)); 
    };

    return (
        <div className="relative w-full h-[600px] bg-slate-50 dark:bg-slate-900 overflow-hidden flex flex-col">
            {loading && (
                <div className="absolute top-2 right-2 z-50">
                    <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                </div>
            )}
            <div className="flex-1 overflow-x-auto overflow-y-hidden" ref={containerRef}>
                <div className="flex h-full p-8 min-w-max gap-12 relative" ref={innerRef}>
                    {levels.map((level, index) => (
                        <div key={index} className="flex flex-col w-[200px] h-full relative z-10">
                            {/* Header */}
                            <div 
                                id={`header-${index}`}
                                className="flex items-center justify-between mb-4 bg-white dark:bg-slate-800 p-2 rounded shadow-sm border border-slate-200 dark:border-slate-700"
                            >
                                <span className="font-semibold text-sm text-slate-700 dark:text-slate-200">
                                    {level.splitColumn || "Total"}
                                </span>
                                <div className="flex items-center gap-1">
                                    {/* Add Split Button (Only on last level) */}
                                    {index === levels.length - 1 && (
                                        <button 
                                            onClick={(e) => handleHeaderSplitClick(index, e)}
                                            className={`
                                                p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors
                                                ${level.selectedItemName ? 'text-blue-500' : 'text-slate-300 cursor-not-allowed'}
                                            `}
                                            title={level.selectedItemName ? "Split further" : "Select an item to split"}
                                            disabled={!level.selectedItemName}
                                        >
                                            <Plus className="w-4 h-4" />
                                        </button>
                                    )}
                                    
                                    {index > 0 && (
                                        <button onClick={() => closeSplit(index)} className="text-slate-400 hover:text-red-500">
                                            <X className="w-4 h-4" />
                                        </button>
                                    )}
                                </div>
                            </div>

                            {/* List */}
                            <div 
                                id={`list-${index}`}
                                className="flex-1 overflow-y-auto pr-2 space-y-2 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-600"
                                onScroll={updatePaths}
                            >
                                {level.items.map((item) => {
                                    const isSelected = level.selectedItemName === item.name;
                                    const isMax = Math.max(...level.items.map(i => i.value));
                                    const barWidth = (item.value / isMax) * 100;

                                    return (
                                        <div 
                                            key={item.name}
                                            id={`node-${index}-${item.name}`}
                                            onClick={() => handleItemClick(index, item)}
                                            className={`
                                                relative p-3 rounded-md border cursor-pointer transition-all group
                                                ${isSelected 
                                                    ? 'bg-blue-50 border-blue-500 dark:bg-blue-900/20 dark:border-blue-500 shadow-md' 
                                                    : 'bg-white border-slate-200 dark:bg-slate-800 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-700'
                                                }
                                            `}
                                        >
                                            <div className="flex justify-between items-center mb-1 relative z-10 gap-2">
                                                <span className="font-medium text-sm truncate flex-1 min-w-0" title={item.name}>
                                                    {item.name}
                                                </span>
                                                <span className="text-xs font-mono text-slate-500 dark:text-slate-400 whitespace-nowrap">
                                                    {item.value.toLocaleString()} ({Math.round(item.ratio * 100)}%)
                                                </span>
                                            </div>
                                            
                                            {/* Bar Background */}
                                            <div className="absolute bottom-0 left-0 h-1 bg-blue-500/20 transition-all duration-500" style={{ width: `${barWidth}%` }} />
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                
                {/* SVG Connections Layer */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
                    {paths.map((path, i) => (
                        <path
                            key={i}
                            d={path.d}
                            fill="none"
                            stroke="#3b82f6" // blue-500
                            strokeWidth="2"
                            strokeOpacity="0.5"
                        />
                    ))}
                </svg>
                </div>
            </div>

            {/* Split Menu */}
            {splitMenuOpen && (
                <div 
                    className="absolute z-50 bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 w-64 max-h-80 overflow-y-auto flex flex-col"
                    style={{ left: splitMenuOpen.x, top: splitMenuOpen.y }}
                >
                    <div className="p-3 border-b border-slate-100 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 sticky top-0">
                        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Split by</span>
                    </div>
                    <div className="p-1">
                        {columns.map(col => (
                            <button
                                key={col}
                                onClick={() => handleSplit(col)}
                                className="w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-200 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-md transition-colors"
                            >
                                {col}
                            </button>
                        ))}
                    </div>
                </div>
            )}
            
            {/* Backdrop for menu */}
            {splitMenuOpen && (
                <div className="fixed inset-0 z-40" onClick={() => setSplitMenuOpen(null)} />
            )}
        </div>
    );
};
