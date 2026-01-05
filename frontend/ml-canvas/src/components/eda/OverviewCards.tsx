import React from 'react';

interface OverviewCardsProps {
    profile: any;
}

export const OverviewCards: React.FC<OverviewCardsProps> = ({ profile }) => {
    if (!profile) return null;

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-500">Rows</div>
                <div className="text-2xl font-bold">{profile.row_count.toLocaleString()}</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-500">Columns</div>
                <div className="text-2xl font-bold">{profile.column_count}</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-500">Missing Cells</div>
                <div className="text-2xl font-bold">{profile.missing_cells_percentage.toFixed(1)}%</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-500">Duplicates</div>
                <div className="text-2xl font-bold">{profile.duplicate_rows}</div>
            </div>
            {profile.vif && (
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                    <div className="text-sm text-gray-500">High VIF Features</div>
                    <div className="text-2xl font-bold text-amber-600">
                        {Object.values(profile.vif).filter((v: any) => v > 5).length}
                    </div>
                </div>
            )}
        </div>
    );
};
