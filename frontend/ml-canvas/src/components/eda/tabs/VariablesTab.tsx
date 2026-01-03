import React from 'react';
import { VariableCard } from '../VariableCard';

interface VariablesTabProps {
    profile: any;
    setSelectedVariable: (variable: any) => void;
    handleToggleExclude: (colName: string, exclude: boolean) => void;
}

export const VariablesTab: React.FC<VariablesTabProps> = ({
    profile,
    setSelectedVariable,
    handleToggleExclude
}) => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {/* Active Columns */}
            {Object.values(profile.columns).map((col: any) => (
                <VariableCard 
                    key={col.name} 
                    profile={col} 
                    onClick={() => setSelectedVariable(col)} 
                    onToggleExclude={handleToggleExclude}
                    isExcluded={false}
                />
            ))}
            
            {/* Excluded Columns (Ghost Cards) */}
            {profile.excluded_columns && profile.excluded_columns.map((colName: string) => (
                <VariableCard 
                    key={colName}
                    profile={{ name: colName, dtype: 'Excluded', missing_percentage: 0 }}
                    onClick={() => {}}
                    onToggleExclude={handleToggleExclude}
                    isExcluded={true}
                />
            ))}
        </div>
    );
};