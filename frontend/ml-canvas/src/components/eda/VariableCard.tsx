import React from 'react';
import { BarChart, Bar, ResponsiveContainer } from 'recharts';
import { AlertTriangle, Hash, Type, Calendar, AlignLeft } from 'lucide-react';

interface VariableCardProps {
  profile: any;
  onClick: () => void;
}

export const VariableCard: React.FC<VariableCardProps> = ({ profile, onClick }) => {
  const getIcon = () => {
    switch (profile.dtype) {
      case 'Numeric': return <Hash className="w-4 h-4 text-blue-500" />;
      case 'Categorical': return <AlignLeft className="w-4 h-4 text-purple-500" />;
      case 'DateTime': return <Calendar className="w-4 h-4 text-green-500" />;
      default: return <Type className="w-4 h-4 text-gray-500" />;
    }
  };

  // Prepare mini histogram data if available
  let miniChartData: any[] = [];
  
  if (profile.dtype === 'Numeric' && profile.histogram) {
      miniChartData = profile.histogram.map((b: any) => ({ name: b.start.toFixed(1), count: b.count }));
  } else if (profile.dtype === 'Text' && profile.histogram) {
      miniChartData = profile.histogram.map((b: any) => ({ name: b.start.toFixed(0), count: b.count }));
  } else if (profile.dtype === 'DateTime' && profile.histogram) {
      miniChartData = profile.histogram.map((b: any) => ({ name: new Date(b.start).toLocaleDateString(), count: b.count }));
  } else if (profile.dtype === 'Categorical' && profile.categorical_stats?.top_k) {
      miniChartData = profile.categorical_stats.top_k.slice(0, 5).map((k: any) => ({ name: k.value, count: k.count }));
  }

  return (
    <div 
      className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 hover:shadow-md transition-shadow cursor-pointer"
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          {getIcon()}
          <h3 className="font-medium text-sm truncate max-w-[120px]" title={profile.name}>
            {profile.name}
          </h3>
        </div>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {profile.dtype}
        </span>
      </div>

      <div className="flex justify-between items-end mt-2">
        <div className="text-xs text-gray-500 space-y-1">
          {profile.missing_percentage > 0 && (
            <div className="flex items-center text-amber-600">
              <AlertTriangle className="w-3 h-3 mr-1" />
              {profile.missing_percentage.toFixed(1)}% null
            </div>
          )}
          {profile.is_unique && <div className="text-blue-500">Unique ID</div>}
          {profile.is_constant && <div className="text-red-500">Constant</div>}
          {!profile.is_unique && !profile.is_constant && profile.missing_percentage === 0 && (
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
                    fill={profile.dtype === 'Numeric' ? '#3b82f6' : profile.dtype === 'Categorical' ? '#8b5cf6' : '#10b981'} 
                    radius={[2, 2, 0, 0]} 
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};
