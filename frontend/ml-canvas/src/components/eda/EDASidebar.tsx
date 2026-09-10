import React, { useState } from 'react';
import type { EDASidebarProps } from './edaSidebar/types';
import { useFilterForm } from './edaSidebar/useFilterForm';
import { FilterControls } from './edaSidebar/FilterControls';
import { ExclusionControls } from './edaSidebar/ExclusionControls';
import { AnalysisNavigation } from './edaSidebar/AnalysisNavigation';

export const EDASidebar: React.FC<EDASidebarProps> = (props) => {
    const [showFilters, setShowFilters] = useState(true);
    const [showExclusions, setShowExclusions] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);
    const form = useFilterForm(props.onAddFilter);
    const [isAddingExclusion, setIsAddingExclusion] = useState(false);

    return (
        <div className={`${isCollapsed ? 'w-14' : 'w-60'} bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex-shrink-0 h-full overflow-y-auto flex flex-col transition-all duration-300`}>
            {!isCollapsed && props.activeTab !== 'pii' && (
                <div className="p-3 border-b border-gray-200 dark:border-gray-700 space-y-3 bg-gray-50/50 dark:bg-gray-900/20">
                    <FilterControls {...props} showFilters={showFilters} setShowFilters={setShowFilters} form={form} />
                    <ExclusionControls {...props} showExclusions={showExclusions} setShowExclusions={setShowExclusions}
                        isAddingExclusion={isAddingExclusion} setIsAddingExclusion={setIsAddingExclusion} />
                </div>
            )}
            <AnalysisNavigation activeTab={props.activeTab} setActiveTab={props.setActiveTab} profile={props.profile}
                isCollapsed={isCollapsed} setIsCollapsed={setIsCollapsed} />
        </div>
    );
};
