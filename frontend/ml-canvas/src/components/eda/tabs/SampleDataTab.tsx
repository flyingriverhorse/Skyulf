import React from 'react';
import { EyeOff } from 'lucide-react';

interface SampleDataTabProps {
    profile: any;
    excludedCols: string[];
    handleToggleExclude: (colName: string, exclude: boolean) => void;
}

export const SampleDataTab: React.FC<SampleDataTabProps> = ({
    profile,
    excludedCols,
    handleToggleExclude
}) => {
    // Filter out excluded columns from display
    // Also filter out columns ending in '_encoded' (e.g. target_encoded)
    const visibleColumns = Object.keys(profile.sample_data[0] || {}).filter(col => 
        !excludedCols.includes(col) && !col.endsWith('_encoded')
    );

    return (
        <div className="mt-4 overflow-x-auto bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                    <tr>
                        {visibleColumns.map((col) => (
                            <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider group">
                                <div className="flex items-center gap-2">
                                    {col}
                                    <button 
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleToggleExclude(col, true);
                                        }}
                                        className="p-1 rounded transition-colors opacity-0 group-hover:opacity-100 hover:bg-red-100 text-gray-400 hover:text-red-500 dark:hover:bg-red-900/30"
                                        title="Exclude from analysis"
                                    >
                                        <EyeOff className="w-3 h-3" />
                                    </button>
                                </div>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {profile.sample_data.slice(0, 50).map((row: any, i: number) => (
                        <tr key={i}>
                            {visibleColumns.map((key: string, j: number) => (
                                <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                                    {row[key] !== null ? String(row[key]) : <span className="italic text-gray-400">null</span>}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
            {profile.sample_data.length > 50 && (
                <div className="p-4 text-center text-sm text-gray-500 border-t border-gray-200 dark:border-gray-700">
                    Showing first 50 rows of {profile.sample_data.length} sample rows.
                </div>
            )}
        </div>
    );
};