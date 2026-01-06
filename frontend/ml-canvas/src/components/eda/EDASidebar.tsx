import React, { useState } from 'react';
import { 
    LayoutDashboard, 
    Table, 
    BarChart2, 
    AlertTriangle, 
    Target, 
    GitMerge, 
    ScatterChart, 
    Network, 
    GitBranch, 
    Split, 
    Calendar, 
    Map, 
    ChevronRight,
    Lightbulb,
    Plus,
    X,
    EyeOff,
    ChevronDown,
    ChevronUp,
    PanelLeftClose,
    PanelLeftOpen,
} from 'lucide-react';

interface FilterItem {
    column: string;
    operator: string;
    value: any;
}

interface EDASidebarProps {
    activeTab: string;
    setActiveTab: (tab: string) => void;
    profile: any;
    filters: FilterItem[];
    columns: string[];
    excludedCols: string[];
    excludedDirty: boolean;
    analyzing: boolean;
    onAddFilter: (column: string, value: any, operator: string) => void;
    onRemoveFilter: (index: number) => void;
    onClearFilters: () => void;
    onToggleExclude: (column: string, exclude: boolean) => void;
    onApplyExcluded: () => void;
}

export const EDASidebar: React.FC<EDASidebarProps> = ({ 
    activeTab, 
    setActiveTab, 
    profile,
    filters,
    columns,
    excludedCols,
    excludedDirty,
    analyzing,
    onAddFilter,
    onRemoveFilter,
    onClearFilters,
    onToggleExclude,
    onApplyExcluded
}) => {
    const [showFilters, setShowFilters] = useState(true);
    const [showExclusions, setShowExclusions] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);
    
    // Filter Form State
    const [isAddingFilter, setIsAddingFilter] = useState(false);
    const [newFilterCol, setNewFilterCol] = useState('');
    const [newFilterOp, setNewFilterOp] = useState('==');
    const [newFilterVal, setNewFilterVal] = useState('');

    // Exclusion Form State
    const [isAddingExclusion, setIsAddingExclusion] = useState(false);

    const handleAddFilterSubmit = () => {
        if (newFilterCol && newFilterVal) {
            onAddFilter(
                newFilterCol, 
                isNaN(Number(newFilterVal)) ? newFilterVal : Number(newFilterVal), 
                newFilterOp
            );
            setIsAddingFilter(false);
            setNewFilterCol('');
            setNewFilterVal('');
        }
    };
    
    const groups = [
        {
            title: "Overview",
            items: [
                { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, show: true },
                { id: 'sample', label: 'Sample Data', icon: Table, show: !!profile.sample_data },
                { id: 'insights', label: 'Smart Insights', icon: Lightbulb, show: true },
            ]
        },
        {
            title: "Univariate Analysis",
            items: [
                { id: 'variables', label: 'Variables', icon: BarChart2, show: true },
                { id: 'outliers', label: 'Outliers', icon: AlertTriangle, show: !!profile.outliers },
                { id: 'target', label: 'Target Analysis', icon: Target, show: !!(profile.target_col && profile.target_correlations) },
            ]
        },
        {
            title: "Multivariate Analysis",
            items: [
                { id: 'correlations', label: 'Correlations', icon: GitMerge, show: !!(profile.correlations || profile.correlations_with_target) },
                { id: 'bivariate', label: 'Bivariate', icon: ScatterChart, show: true },
                { id: 'pca', label: 'PCA & Clusters', icon: Network, show: !!profile.pca_data || !!profile.clustering },
            ]
        },
        {
            title: "Structure & Causal",
            items: [
                { id: 'causal', label: 'Causal Graph', icon: Network, show: !!profile.causal_graph },
                { id: 'rules', label: 'Decision Tree', icon: GitBranch, show: !!profile.rule_tree },
                { id: 'decomposition', label: 'Decomposition', icon: Split, show: true }, // Always show, handles its own empty state
            ]
        },
        {
            title: "Specialized",
            items: [
                { id: 'timeseries', label: 'Time Series', icon: Calendar, show: !!profile.timeseries },
                { id: 'geospatial', label: 'Geospatial', icon: Map, show: !!profile.geospatial },
            ]
        }
    ];

    return (
        <div className={`${isCollapsed ? 'w-14' : 'w-60'} bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex-shrink-0 h-full overflow-y-auto flex flex-col transition-all duration-300`}>
            
            {/* Data Controls Section */}
            {!isCollapsed && (
            <div className="p-3 border-b border-gray-200 dark:border-gray-700 space-y-3 bg-gray-50/50 dark:bg-gray-900/20">
                
                {/* Filters */}
                <div>
                    <div className="flex items-center justify-between w-full mb-2">
                        <button 
                            onClick={() => setShowFilters(!showFilters)}
                            className="flex items-center text-xs font-semibold text-gray-500 uppercase tracking-wider hover:text-gray-700"
                        >
                            <span>Active Filters ({filters.length})</span>
                            {showFilters ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />}
                        </button>
                        {filters.length > 0 && (
                            <button 
                                onClick={onClearFilters}
                                className="text-[10px] text-red-500 hover:text-red-700 hover:underline"
                            >
                                Clear All
                            </button>
                        )}
                    </div>
                    
                    {showFilters && (
                        <div className="space-y-2">
                            {filters.map((filter, idx) => (
                                <div key={idx} className="flex items-center justify-between bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded px-2 py-1 text-xs">
                                    <div className="truncate max-w-[140px]">
                                        <span className="font-medium text-blue-700 dark:text-blue-300">{filter.column}</span>
                                        <span className="mx-1 text-gray-500">{filter.operator}</span>
                                        <span className="text-gray-600 dark:text-gray-400">{String(filter.value)}</span>
                                    </div>
                                    <button onClick={() => onRemoveFilter(idx)} className="text-gray-400 hover:text-red-500">
                                        <X className="w-3 h-3" />
                                    </button>
                                </div>
                            ))}
                            
                            {isAddingFilter ? (
                                <div className="bg-white dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-700 space-y-2 shadow-sm">
                                    <select 
                                        value={newFilterCol}
                                        onChange={(e) => setNewFilterCol(e.target.value)}
                                        className="w-full text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                                    >
                                        <option value="" disabled>Column</option>
                                        {columns.map(col => <option key={col} value={col}>{col}</option>)}
                                    </select>
                                    <div className="flex gap-1">
                                        <select 
                                            value={newFilterOp}
                                            onChange={(e) => setNewFilterOp(e.target.value)}
                                            className="w-1/3 text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                                        >
                                            <option value="==">==</option>
                                            <option value="!=">!=</option>
                                            <option value=">">&gt;</option>
                                            <option value="<">&lt;</option>
                                            <option value=">=">&gt;=</option>
                                            <option value="<=">&lt;=</option>
                                        </select>
                                        <input 
                                            type="text"
                                            value={newFilterVal}
                                            onChange={(e) => setNewFilterVal(e.target.value)}
                                            placeholder="Value"
                                            className="w-2/3 text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                                            onKeyDown={(e) => e.key === 'Enter' && handleAddFilterSubmit()}
                                        />
                                    </div>
                                    <div className="flex justify-end gap-2">
                                        <button onClick={() => setIsAddingFilter(false)} className="text-xs text-gray-500 hover:text-gray-700">Cancel</button>
                                        <button onClick={handleAddFilterSubmit} className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700">Add</button>
                                    </div>
                                </div>
                            ) : (
                                <button 
                                    onClick={() => setIsAddingFilter(true)}
                                    className="w-full flex items-center justify-center px-2 py-1 text-xs border border-dashed border-gray-300 dark:border-gray-600 rounded text-gray-500 hover:text-blue-600 hover:border-blue-300 transition-colors"
                                >
                                    <Plus className="w-3 h-3 mr-1" /> Add Filter
                                </button>
                            )}
                        </div>
                    )}
                </div>

                {/* Excluded Columns */}
                <div>
                    <button 
                        onClick={() => setShowExclusions(!showExclusions)}
                        className="flex items-center justify-between w-full text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 hover:text-gray-700"
                    >
                        <span>Excluded ({excludedCols.length})</span>
                        {showExclusions ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>

                    {showExclusions && (
                        <div className="space-y-2">
                            {excludedCols.map((col, idx) => (
                                <div key={idx} className="flex items-center justify-between bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 text-xs">
                                    <span className="font-medium text-gray-500 line-through truncate max-w-[140px]">{col}</span>
                                    <button 
                                        onClick={() => onToggleExclude(col, false)}
                                        className="text-gray-400 hover:text-green-500"
                                        title="Include back"
                                    >
                                        <Plus className="w-3 h-3" />
                                    </button>
                                </div>
                            ))}

                            <button
                                onClick={onApplyExcluded}
                                disabled={!excludedDirty || analyzing}
                                className={`w-full flex items-center justify-center px-2 py-1 text-xs rounded border transition-colors ${
                                    excludedDirty && !analyzing
                                        ? 'bg-blue-600 text-white border-blue-700 hover:bg-blue-700'
                                        : 'bg-gray-100 dark:bg-gray-800 text-gray-400 border-gray-200 dark:border-gray-700 cursor-not-allowed'
                                }`}
                                title={excludedDirty ? 'Apply excluded columns to re-run analysis' : 'No pending exclusion changes'}
                            >
                                Apply changes
                            </button>

                            {isAddingExclusion ? (
                                <div className="bg-white dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-700 space-y-2 shadow-sm">
                                    <select 
                                        className="w-full text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                                        onChange={(e) => {
                                            if (e.target.value) {
                                                onToggleExclude(e.target.value, true);
                                                setIsAddingExclusion(false);
                                            }
                                        }}
                                        defaultValue=""
                                    >
                                        <option value="" disabled>Select Column</option>
                                        {columns.filter(c => !excludedCols.includes(c)).map(col => (
                                            <option key={col} value={col}>{col}</option>
                                        ))}
                                    </select>
                                    <button onClick={() => setIsAddingExclusion(false)} className="w-full text-xs text-gray-500 hover:text-gray-700 text-center">Cancel</button>
                                </div>
                            ) : (
                                <button 
                                    onClick={() => setIsAddingExclusion(true)}
                                    className="w-full flex items-center justify-center px-2 py-1 text-xs border border-dashed border-gray-300 dark:border-gray-600 rounded text-gray-500 hover:text-red-600 hover:border-red-300 transition-colors"
                                >
                                    <EyeOff className="w-3 h-3 mr-1" /> Exclude Column
                                </button>
                            )}
                        </div>
                    )}
                </div>
            </div>
            )}

            <div className={`flex-1 overflow-y-auto ${isCollapsed ? 'p-2' : 'p-3'}`}>
                <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'} mb-3`}>
                    {!isCollapsed && (
                    <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                        Analysis Modules
                    </h2>
                    )}
                    <button 
                        onClick={() => setIsCollapsed(!isCollapsed)}
                        className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded text-gray-500"
                        title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
                    >
                        {isCollapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
                    </button>
                </div>
                <div className="space-y-6">
                    {groups.map((group, groupIdx) => {
                        const visibleItems = group.items.filter(item => item.show);
                        if (visibleItems.length === 0) return null;

                        return (
                            <div key={groupIdx}>
                                {!isCollapsed && <h3 className="px-2 text-xs font-medium text-gray-500 mb-2">{group.title}</h3>}
                                <div className="space-y-1">
                                    {visibleItems.map((item) => (
                                        <button
                                            key={item.id}
                                            onClick={() => setActiveTab(item.id)}
                                            title={isCollapsed ? item.label : undefined}
                                            className={`w-full flex items-center ${isCollapsed ? 'justify-center py-2' : 'px-2 py-2'} text-sm font-medium rounded-md transition-colors ${
                                                activeTab === item.id
                                                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400'
                                                    : 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700'
                                            }`}
                                        >
                                            <item.icon className={`${isCollapsed ? 'h-5 w-5' : 'mr-3 h-4 w-4'} ${
                                                activeTab === item.id ? 'text-blue-500' : 'text-gray-400'
                                            }`} />
                                            {!isCollapsed && item.label}
                                            {!isCollapsed && activeTab === item.id && (
                                                <ChevronRight className="ml-auto h-4 w-4 text-blue-400" />
                                            )}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};
