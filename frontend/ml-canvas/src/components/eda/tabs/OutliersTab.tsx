import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { EmptyState } from '../../shared/EmptyState';
import type { EDAProfile, OutlierAnalysis } from '../../../core/types/edaProfile';
import { formatStatistic } from '../formatStatistic';

interface OutliersTabProps {
    profile: EDAProfile;
}

/** Describe only the analyzed population recorded with these outlier results. */
function outlierPopulation(data: OutlierAnalysis) {
    const { analyzed_rows: analyzedRows, total_rows: totalRows } = data;
    if (analyzedRows == null || totalRows == null) {
        return {
            description: 'Analyzed row count is unavailable for this saved report.',
            countLabel: 'Reported outliers',
            percentageLabel: 'Reported percentage',
        };
    }
    if (analyzedRows < totalRows) {
        return {
            description: `Analyzed ${analyzedRows.toLocaleString('en-US')} sampled rows out of ${totalRows.toLocaleString('en-US')} rows. Counts and percentages describe this sample.`,
            countLabel: 'Outliers in sample',
            percentageLabel: 'Percentage of sampled rows',
        };
    }
    return {
        description: `Analyzed all ${analyzedRows.toLocaleString('en-US')} rows.`,
        countLabel: 'Outliers detected',
        percentageLabel: 'Percentage of analyzed rows',
    };
}

/** Display outlier results with the measured row population and row-level explanations. */
export const OutliersTab: React.FC<OutliersTabProps> = ({ profile }) => {
    if (!profile?.outliers?.top_outliers?.length) {
        return <EmptyState icon={<AlertTriangle className="w-12 h-12 text-slate-300 dark:text-slate-600" />} title="No Outlier Data" description="Outlier analysis did not return any results." />;
    }
    const population = outlierPopulation(profile.outliers);

    return (
        <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-red-500" />
                Outlier Analysis ({profile.outliers.method})
                <InfoTooltip text="Detects anomalous rows using Isolation Forest. Lower scores indicate higher anomaly." />
            </h3>
            <p className="mb-4 text-sm text-gray-600 dark:text-gray-300">{population.description}</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-100 dark:border-red-800">
                    <span className="text-sm text-red-600 dark:text-red-400 block mb-1">{population.countLabel}</span>
                    <span className="text-2xl font-bold text-red-700 dark:text-red-300">{profile.outliers.total_outliers}</span>
                </div>
                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-100 dark:border-red-800">
                    <span className="text-sm text-red-600 dark:text-red-400 block mb-1">{population.percentageLabel}</span>
                    <span className="text-2xl font-bold text-red-700 dark:text-red-300">{profile.outliers.outlier_percentage.toFixed(2)}%</span>
                </div>
            </div>

            <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">Top Anomalous Rows</h4>
            <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-lg">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-900">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Row Index</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Anomaly Score</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Why is this an outlier?</th>
                            {profile.outliers.top_outliers[0] && Object.keys(profile.outliers.top_outliers[0].values ?? {}).slice(0, 5).map(key => (
                                <th key={key} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{key}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                        {profile.outliers.top_outliers.map((outlier) => (
                            <tr key={outlier.index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                                <td className="px-4 py-3 text-sm font-mono text-gray-500">{outlier.index}</td>
                                <td className="px-4 py-3 text-sm font-mono text-red-600">{outlier.score?.toFixed(4) ?? '—'}</td>
                                <td className="px-4 py-3 text-sm">
                                    {outlier.explanation ? (
                                        <div className="space-y-1">
                                            {outlier.explanation.map((exp, i) => (
                                                <div key={i} className="text-xs">
                                                    <span className="font-semibold text-gray-700 dark:text-gray-300">{exp.feature}:</span>{' '}
                                                    <span className="text-red-600 dark:text-red-400">{formatStatistic(exp.value, 2)}</span>{' '}
                                                    <span className="text-gray-400">
                                                        (Median: {formatStatistic(exp.median, 2)}, Diff: {formatStatistic(exp.diff_pct, 0, '%')})
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    ) : (
                                        <span className="text-gray-400 text-xs">No explanation available</span>
                                    )}
                                </td>
                                {Object.entries(outlier.values ?? {}).slice(0, 5).map(([key, val]) => (
                                    <td key={key} className="px-4 py-3 text-sm text-gray-900 dark:text-gray-300">
                                        {typeof val === 'number' ? val.toFixed(2) : String(val)}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
