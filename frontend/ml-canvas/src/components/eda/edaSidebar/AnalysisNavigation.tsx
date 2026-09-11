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
    Shield,
    PanelLeftClose,
    PanelLeftOpen,
} from 'lucide-react';

import type { EDAProfile } from '../../../core/types/edaProfile';
import type { EDASidebarProps } from './types';

type NavigationProps = Pick<EDASidebarProps, 'activeTab' | 'setActiveTab' | 'profile'> & {
    isCollapsed: boolean;
    setIsCollapsed: (value: boolean) => void;
};

function analysisGroups(profile: EDAProfile) {
    const groups = [
        {
            title: "Overview",
            items: [
                { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard, show: true },
                { id: 'pii', label: 'PII Review', icon: Shield, show: true },
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
                { id: 'correlations', label: 'Correlations', icon: GitMerge, show: !!(profile.correlations || profile.correlations_with_target || profile.causal_target_exclusion_reason) },
                { id: 'bivariate', label: 'Bivariate', icon: ScatterChart, show: true },
                { id: 'pca', label: 'PCA & Clusters', icon: Network, show: !!profile.pca_data || !!profile.clustering },
            ]
        },
        {
            title: "Structure & Causal",
            items: [
                { id: 'causal', label: 'Causal Graph', icon: Network, show: !!(profile.causal_graph || profile.causal_target_exclusion_reason) },
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

    return groups;
}

export function AnalysisNavigation({ activeTab, setActiveTab, profile, isCollapsed, setIsCollapsed }: NavigationProps) {
    const groups = analysisGroups(profile);
    return (
        <div className={`flex-1 overflow-y-auto ${isCollapsed ? 'p-2' : 'p-3'}`}>
            <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'} mb-3`}>
                {!isCollapsed && (
                    <h2 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Analysis Modules
                    </h2>
                )}
                <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded text-gray-500 dark:text-gray-400"
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
                            {!isCollapsed && <h3 className="px-2 text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">{group.title}</h3>}
                            <div className="space-y-1">
                                {visibleItems.map((item) => (
                                    <NavigationItem key={item.id} item={item} activeTab={activeTab} setActiveTab={setActiveTab} isCollapsed={isCollapsed} />
                                ))}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function NavigationItem({
    item,
    activeTab,
    setActiveTab,
    isCollapsed
}: Pick<NavigationProps, 'activeTab' | 'setActiveTab' | 'isCollapsed'> & { item: ReturnType<typeof analysisGroups>[number]['items'][number]; }) {
    const selection = activeTab === item.id
        ? { button: 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400', icon: 'text-blue-500', current: 'page' as const }
        : { button: 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700', icon: 'text-gray-400', current: undefined };
    return (
        <button
            onClick={() => setActiveTab(item.id)}
            aria-label={item.label}
            aria-current={selection.current}
            title={isCollapsed ? item.label : undefined}
            className={`w-full flex items-center ${isCollapsed ? 'justify-center py-2' : 'px-2 py-2'} text-sm font-medium rounded-md transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-gray-800 ${selection.button
                }`}
        >
            <item.icon className={`${isCollapsed ? 'h-5 w-5' : 'mr-3 h-4 w-4'} ${selection.icon
                }`} />
            {!isCollapsed && <>
                {item.label}
                {activeTab === item.id && <ChevronRight className="ml-auto h-4 w-4 text-blue-400" />}
            </>}
        </button>
    );
}
