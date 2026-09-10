import { BarChart, Bar, ResponsiveContainer } from 'recharts';
import type { ColumnProfile } from '../../../core/types/edaProfile';
import { getDtypeHexColor } from '../../../core/utils/dtypeVisuals';

interface MiniDatum { name: string; count: number }

/** Format the existing dtype-specific bins without reordering or dropping zeros. */
export function getMiniChartData(profile: ColumnProfile): MiniDatum[] {
  if (profile.dtype === 'Categorical' && profile.categorical_stats?.top_k) {
    return profile.categorical_stats.top_k.slice(0, 5).map(k => ({ name: String(k.value), count: k.count }));
  }
  if (!profile.histogram) return [];
  switch (profile.dtype) {
    case 'Numeric':
      return profile.histogram.map(b => ({ name: b.start.toFixed(1), count: b.count }));
    case 'Text':
      return profile.histogram.map(b => ({ name: b.start.toFixed(0), count: b.count }));
    case 'DateTime':
      return profile.histogram.map(b => ({ name: new Date(b.start).toLocaleDateString(), count: b.count }));
    default:
      return [];
  }
}

/** Keep the miniature histogram absent when the report has no supported bins. */
export function MiniHistogram({ data, dtype }: { data: MiniDatum[]; dtype: string }) {
  if (data.length === 0) return null;
  return (
    <div className="h-12 w-24">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <Bar dataKey="count" fill={getDtypeHexColor(dtype)} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
