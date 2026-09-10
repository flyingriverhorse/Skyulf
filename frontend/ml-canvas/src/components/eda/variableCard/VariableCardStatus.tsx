import { AlertTriangle } from 'lucide-react';
import type { ColumnProfile, NormalityTestResult } from '../../../core/types/edaProfile';

/** Use the original normality label, color and three-decimal p-value. */
function NormalityStatus({ result }: { result: NormalityTestResult }) {
  return (
    <div
      className={result.is_normal ? 'text-purple-600' : 'text-amber-600'}
      title={`${result.is_normal ? 'Normal' : 'Not Normal'} Distribution (p=${result.p_value.toFixed(3)})`}
    >
      {result.is_normal ? 'Normal Dist.' : 'Not Normal Dist.'}
    </div>
  );
}

/** Show all applicable flags; Healthy retains its exact zero-missingness condition. */
export function VariableCardStatus({ profile }: { profile: ColumnProfile }) {
  return (
    <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
      {profile.missing_percentage > 0 && (
        <div className="flex items-center text-amber-600">
          <AlertTriangle className="w-3 h-3 mr-1" />
          {profile.missing_percentage.toFixed(1)}% null
        </div>
      )}
      {profile.is_unique && <div className="text-blue-500">Unique ID</div>}
      {profile.is_constant && <div className="text-red-500">Constant</div>}
      {profile.normality_test && <NormalityStatus result={profile.normality_test} />}
      {!profile.is_unique && !profile.is_constant && profile.missing_percentage === 0 && !profile.normality_test && (
        <div className="text-green-600">Healthy</div>
      )}
    </div>
  );
}
