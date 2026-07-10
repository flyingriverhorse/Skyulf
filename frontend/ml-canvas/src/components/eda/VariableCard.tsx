import React from 'react';
import { BarChart, Bar, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Type, EyeOff, Eye } from 'lucide-react';
import { clickableProps } from '../../core/utils/a11y';
import { getDtypeIcon, getDtypeIconColorClass, getDtypeBadgeClass, getDtypeHexColor } from '../../core/utils/dtypeVisuals';
import type { ColumnProfile } from '../../core/types/edaProfile';

interface VariableCardProps {
  profile: ColumnProfile;
  onClick: () => void;
  onToggleExclude?: (colName: string, exclude: boolean) => void;
  isExcluded?: boolean;
}

export const VariableCard: React.FC<VariableCardProps> = ({ profile, onClick, onToggleExclude, isExcluded = false }) => {
  const DtypeIcon = getDtypeIcon(profile.dtype);
  const icon = isExcluded
    ? <Type className="w-4 h-4 text-gray-400" />
    : <DtypeIcon className={`w-4 h-4 ${getDtypeIconColorClass(profile.dtype)}`} />;

  // Mini histogram data for the card preview.
  type MiniDatum = { name: string; count: number };
  let miniChartData: MiniDatum[] = [];

  if (profile.dtype === 'Numeric' && profile.histogram) {
      miniChartData = profile.histogram.map((b) => ({ name: b.start.toFixed(1), count: b.count }));
  } else if (profile.dtype === 'Text' && profile.histogram) {
      miniChartData = profile.histogram.map((b) => ({ name: b.start.toFixed(0), count: b.count }));
  } else if (profile.dtype === 'DateTime' && profile.histogram) {
      miniChartData = profile.histogram.map((b) => ({ name: new Date(b.start).toLocaleDateString(), count: b.count }));
  } else if (profile.dtype === 'Categorical' && profile.categorical_stats?.top_k) {
      miniChartData = profile.categorical_stats.top_k.slice(0, 5).map((k) => ({ name: String(k.value), count: k.count }));
  }

  return (
    <div
      className={`rounded-lg border p-4 transition-all cursor-pointer flex flex-col h-full ${
        isExcluded
          ? 'bg-gray-50 border-gray-200 dark:bg-gray-900 dark:border-gray-800 opacity-75'
          : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-md'
      }`}
      {...clickableProps(onClick)}
    >
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2 w-full">
          {icon}
          <h3 className={`font-medium text-sm truncate flex-1 ${isExcluded ? 'text-gray-500 line-through' : ''}`} title={profile.name}>
            {profile.name}
          </h3>
        </div>
      </div>

      <div className="flex items-center gap-2 mb-3">
            {!isExcluded && (
                <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${getDtypeBadgeClass(profile.dtype)}`}>
                {profile.dtype}
                </span>
            )}
            {onToggleExclude && (
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onToggleExclude(profile.name, !isExcluded);
                    }}
                    className={`p-1 rounded transition-colors ml-auto ${
                        isExcluded
                            ? 'hover:bg-green-100 text-gray-400 hover:text-green-600 dark:hover:bg-green-900/30'
                            : 'hover:bg-red-100 text-gray-400 hover:text-red-500 dark:hover:bg-red-900/30'
                    }`}
                    title={isExcluded ? "Include in analysis" : "Exclude from analysis"}
                    aria-label={isExcluded ? "Include in analysis" : "Exclude from analysis"}
                >
                    {isExcluded ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                </button>
            )}
      </div>

      {!isExcluded && (
      <div className="flex justify-between items-end mt-auto">
        <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
          {profile.missing_percentage > 0 && (
            <div className="flex items-center text-amber-600">
              <AlertTriangle className="w-3 h-3 mr-1" />
              {profile.missing_percentage.toFixed(1)}% null
            </div>
          )}
          {profile.is_unique && <div className="text-blue-500">Unique ID</div>}
          {profile.is_constant && <div className="text-red-500">Constant</div>}
          {profile.normality_test && profile.normality_test.is_normal && (
             <div className="text-purple-600" title={`Normal Distribution (p=${profile.normality_test.p_value.toFixed(3)})`}>Normal Dist.</div>
          )}
          {profile.normality_test && !profile.normality_test.is_normal && (
             <div className="text-amber-600" title={`Not Normal Distribution (p=${profile.normality_test.p_value.toFixed(3)})`}>Not Normal Dist.</div>
          )}
          {!profile.is_unique && !profile.is_constant && profile.missing_percentage === 0 && !profile.normality_test && (
             <div className="text-green-600">Healthy</div>
          )}
        </div>

        {/* Mini Chart */}
        {miniChartData.length > 0 && (
          <div className="h-12 w-24">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={miniChartData}>
                <Bar
                    dataKey="count"
                    fill={getDtypeHexColor(profile.dtype)}
                    radius={[2, 2, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
      )}
      {isExcluded && (
          <div className="mt-4 text-xs text-center text-gray-400 italic">
              Excluded from analysis
          </div>
      )}
    </div>
  );
};
