import { Eye, EyeOff } from 'lucide-react';
import { BarChart, Bar, ResponsiveContainer } from 'recharts';
import { getDtypeBadgeClass } from '../../../core/utils/dtypeVisuals';
import type { ColumnProfile } from '../../../core/types/edaProfile';
import type { VariableRowProps } from '../VariableRow';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { Badge } from '../../ui/badge';
import { Button } from '../../ui/button';

export function VariableSummary({ profile }: { profile: ColumnProfile; }) {
    // Mini histogram data for the collapsed-row preview.
    type MiniDatum = { name: string; count: number; };
    let miniChartData: MiniDatum[] = [];
    if (profile.histogram) {
        miniChartData = profile.histogram.map((b) => ({ name: String(b.start), count: b.count }));
    } else if (profile.categorical_stats?.top_k) {
        miniChartData = profile.categorical_stats.top_k.slice(0, 10).map((k) => ({ name: String(k.value), count: k.count }));
    }

    return (
        <div className="flex items-center justify-between w-full pr-4">
            <div className="flex gap-4 text-xs text-muted-foreground mr-auto">
                <SummaryValues profile={profile} />

                <NormalityBadge profile={profile} />
            </div>

            {/* Mini Histogram */}
            {miniChartData.length > 0 && (
                <div className="h-8 w-24 hidden sm:block opacity-70">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={miniChartData}>
                            <Bar
                                dataKey="count"
                                fill={profile.dtype === 'Numeric' ? '#3b82f6' : profile.dtype === 'Categorical' ? '#8b5cf6' : '#10b981'}
                                radius={[1, 1, 0, 0]}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
}

function SummaryValues({ profile }: { profile: ColumnProfile; }) {
    return <>
        {profile.numeric_stats ? (
            <>
                <span>Mean: <span className="font-mono text-foreground">{profile.numeric_stats.mean?.toFixed(2)}</span></span>
                <span>Std: <span className="font-mono text-foreground">{profile.numeric_stats.std?.toFixed(2)}</span></span>
                <span>Min: <span className="font-mono text-foreground">{profile.numeric_stats.min?.toFixed(2)}</span></span>
                <span>Max: <span className="font-mono text-foreground">{profile.numeric_stats.max?.toFixed(2)}</span></span>
            </>
        ) : profile.categorical_stats ? (
            <>
                <span>Unique: <span className="font-mono text-foreground">{profile.categorical_stats.unique_count}</span></span>
                {profile.categorical_stats.top_k?.[0] && (
                    <span>Top: <span className="font-mono text-foreground">{String(profile.categorical_stats.top_k[0].value).slice(0, 20)}</span></span>
                )}
            </>
        ) : null}
    </>;
}

function NormalityBadge({ profile }: { profile: ColumnProfile; }) {
    return <>
        {/* Normality / Status Indicators */}
        {profile.normality_test && profile.normality_test.is_normal && (
            <Badge variant="outline" className="text-[10px] h-4 text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/20">
                Normal Dist
            </Badge>
        )}
        {profile.normality_test && !profile.normality_test.is_normal && (
            <Badge variant="outline" className="text-[10px] h-4 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20">
                Not Normal
            </Badge>
        )}
    </>;
}

export function VariableIdentity({ profile }: { profile: ColumnProfile; }) {
    return (
        <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-sm truncate">{profile.name}</span>
            <Badge variant="outline" className={`text-[10px] h-5 px-1 ${getDtypeBadgeClass(profile.dtype)}`}>
                {profile.dtype}
            </Badge>
            {profile.is_unique && <Badge variant="secondary" className="text-[10px] h-5">ID</Badge>}
            {profile.is_constant && <Badge variant="destructive" className="text-[10px] h-5">Constant</Badge>}
        </div>
    );
}

export function VariableControls({ profile, isExcluded, onToggleExclude }: Pick<VariableRowProps, "profile" | "isExcluded" | "onToggleExclude">) {
    return (
        <div className="flex items-center gap-4 px-2">
            {/* Missing Bar */}
            <div className="flex flex-col items-end w-24">
                <div className="flex items-center gap-1 mb-1">
                    <span className="text-[10px] text-muted-foreground">
                        {profile.missing_percentage > 0 ? `${profile.missing_percentage.toFixed(1)}% Missing` : '100% Present'}
                    </span>
                    <InfoTooltip text="Green bar indicates the percentage of valid (non-null) data. Full green means no missing values." align="end" size="sm" />
                </div>
                <div className="w-full h-1.5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                    <div
                        className={`h-full ${profile.missing_percentage > 0 ? 'bg-amber-400' : 'bg-green-500'}`}
                        style={{ width: `${Math.max(100 - profile.missing_percentage, 0)}%` }}
                    />
                </div>
            </div>

            <div className="h-8 w-[1px] bg-border mx-2" />

            <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 text-muted-foreground"
                onClick={(e) => {
                    e.stopPropagation();
                    onToggleExclude(profile.name, !isExcluded);
                }}
                title={isExcluded ? "Include in analysis" : "Exclude from analysis"}
                aria-label={isExcluded ? "Include in analysis" : "Exclude from analysis"}
            >
                {isExcluded ? <EyeOff size={16} /> : <Eye size={16} />}
            </Button>
        </div>
    );
}
