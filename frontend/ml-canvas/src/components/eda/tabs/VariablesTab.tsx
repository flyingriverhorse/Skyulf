import React, { useState } from 'react';
import { VariableRow } from '../VariableRow';
import { Input } from '../../ui/input';
import { Search } from 'lucide-react';
import { Button } from '../../ui/button';
import { Badge } from '../../ui/badge';

interface VariablesTabProps {
    profile: any;
    setSelectedVariable: (variable: any) => void;
    handleToggleExclude: (colName: string, exclude: boolean) => void;
    handleAddFilter: (column: string, value: any, operator: string) => void;
}

export const VariablesTab: React.FC<VariablesTabProps> = ({
    profile,
    handleToggleExclude,
    handleAddFilter
}) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [expandedVars, setExpandedVars] = useState<Set<string>>(new Set());

    const toggleExpand = (colName: string) => {
        const newExpanded = new Set(expandedVars);
        if (newExpanded.has(colName)) {
            newExpanded.delete(colName);
        } else {
            newExpanded.add(colName);
        }
        setExpandedVars(newExpanded);
    };

    const expandAll = () => {
        setExpandedVars(new Set(Object.keys(profile.columns)));
    };

    const collapseAll = () => {
        setExpandedVars(new Set());
    };

    const filteredColumns = Object.values(profile.columns).filter((col: any) => 
        col.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="space-y-4">
             {/* Toolbar */}
             <div className="flex items-center gap-4 bg-card p-2 rounded-lg border">
                <div className="relative flex-1 max-w-sm">
                    <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input 
                        placeholder="Search variables..." 
                        className="pl-8 h-9" 
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={expandAll} className="h-9 text-xs">
                        Expand All
                    </Button>
                    <Button variant="outline" size="sm" onClick={collapseAll} className="h-9 text-xs">
                        Collapse All
                    </Button>
                </div>
                <div className="ml-auto text-sm text-muted-foreground px-2">
                    Showing {filteredColumns.length} of {Object.keys(profile.columns).length} variables
                </div>
             </div>

            {/* List */}
            <div className="space-y-2">
                {/* Active Columns */}
                {filteredColumns.map((col: any) => (
                    <VariableRow 
                        key={col.name} 
                        profile={col} 
                        isExpanded={expandedVars.has(col.name)}
                        onToggleExpand={() => toggleExpand(col.name)}
                        onToggleExclude={handleToggleExclude}
                        isExcluded={false}
                        handleAddFilter={handleAddFilter}
                    />
                ))}
                
                {/* Excluded Section if any */}
                {profile.excluded_columns && profile.excluded_columns.length > 0 && (
                    <div className="mt-8">
                        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-muted-foreground">
                            Excluded Variables <Badge variant="secondary">{profile.excluded_columns.length}</Badge>
                        </h3>
                        {profile.excluded_columns.map((colName: string) => (
                            <VariableRow 
                                key={colName}
                                profile={{ name: colName, dtype: 'Excluded', missing_percentage: 0 }}
                                isExpanded={expandedVars.has(colName)}
                                onToggleExpand={() => toggleExpand(colName)}
                                onToggleExclude={handleToggleExclude}
                                isExcluded={true}
                                handleAddFilter={handleAddFilter}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};