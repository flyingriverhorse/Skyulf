import React from 'react';
import type { EDAProfile } from '../../core/types/edaProfile';

interface OverviewCardsProps {
    profile: EDAProfile;
}

export const OverviewCards: React.FC<OverviewCardsProps> = ({ profile }) => {
    if (!profile) return null;

    const missingPct = profile.missing_cells_percentage ?? 0;
    const duplicates = profile.duplicate_rows ?? 0;

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
                <div className="text-2xl font-bold">{missingPct.toFixed(1)}%</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-sm text-gray-500">Duplicates</div>
                <div className="text-2xl font-bold">{duplicates}</div>
            </div>
            {profile.vif && (
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                    <div className="text-sm text-gray-500">High VIF Features</div>
                    <div className="text-2xl font-bold text-amber-600">
                        {Object.values(profile.vif).filter((v) => v > 5).length}
                    </div>
                </div>
            )}
        </div>
    );
};
