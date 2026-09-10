import React from 'react';
import { Type, EyeOff, Eye } from 'lucide-react';
import { clickableProps } from '../../core/utils/a11y';
import { getDtypeIcon, getDtypeIconColorClass, getDtypeBadgeClass } from '../../core/utils/dtypeVisuals';
import type { ColumnProfile } from '../../core/types/edaProfile';
import { getMiniChartData, MiniHistogram } from './variableCard/miniChart';
import { VariableCardStatus } from './variableCard/VariableCardStatus';

interface VariableCardProps {
  profile: ColumnProfile;
  onClick: () => void;
  onToggleExclude?: (colName: string, exclude: boolean) => void;
  isExcluded?: boolean;
}

/** Toggle the controlled exclusion flag without opening the card's details. */
function ExclusionButton({ name, isExcluded, onToggleExclude }: {
  name: string;
  isExcluded: boolean;
  onToggleExclude: NonNullable<VariableCardProps['onToggleExclude']>;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onToggleExclude(name, !isExcluded);
      }}
      className={`p-1 rounded transition-colors ml-auto ${
        isExcluded
          ? 'hover:bg-green-100 text-gray-400 hover:text-green-600 dark:hover:bg-green-900/30'
          : 'hover:bg-red-100 text-gray-400 hover:text-red-500 dark:hover:bg-red-900/30'
      }`}
      title={isExcluded ? 'Include in analysis' : 'Exclude from analysis'}
      aria-label={isExcluded ? 'Include in analysis' : 'Exclude from analysis'}
    >
      {isExcluded ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
    </button>
  );
}

export const VariableCard: React.FC<VariableCardProps> = ({ profile, onClick, onToggleExclude, isExcluded = false }) => {
  const DtypeIcon = getDtypeIcon(profile.dtype);
  const icon = isExcluded
    ? <Type className="w-4 h-4 text-gray-400" />
    : <DtypeIcon className={`w-4 h-4 ${getDtypeIconColorClass(profile.dtype)}`} />;

  const miniChartData = getMiniChartData(profile);

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
                <ExclusionButton name={profile.name} isExcluded={isExcluded} onToggleExclude={onToggleExclude} />
            )}
      </div>

      {!isExcluded && (
      <div className="flex justify-between items-end mt-auto">
        <VariableCardStatus profile={profile} />
        <MiniHistogram data={miniChartData} dtype={profile.dtype} />
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
