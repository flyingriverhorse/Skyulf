import type { ColumnProfile, NumericStats, TextStats, CategoricalStats } from '../../../core/types/edaProfile';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { Badge } from '../../ui/badge';

export function VariableStatistics({ profile }: { profile: ColumnProfile; }) {
    return (
        <div className="space-y-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Statistics</h4>
            <div className="bg-background rounded-md border p-3 text-sm space-y-2 font-mono">
                <div className="flex justify-between">
                    <span className="text-muted-foreground">Missing</span>
                    <span>{profile.missing_count} ({profile.missing_percentage.toFixed(2)}%)</span>
                </div>
                {profile.numeric_stats && <NumericStatistics stats={profile.numeric_stats} vif={profile.vif} />}
                {profile.text_stats && <TextStatistics stats={profile.text_stats} />}
                {profile.categorical_stats && <CategoricalStatistics stats={profile.categorical_stats} />}
            </div>
        </div>
    );
}
function NumericStatistics({ stats, vif }: { stats: NumericStats; vif: ColumnProfile["vif"]; }) {
    return <>
        <div className="flex justify-between">
            <span className="text-muted-foreground">Mean</span>
            <span>{stats.mean?.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
            <span className="text-muted-foreground">Std Dev</span>
            <span>{stats.std?.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
            <span className="text-muted-foreground">Variance</span>
            <span>{stats.variance?.toFixed(4) || (Math.pow(stats.std || 0, 2)).toFixed(4)}</span>
        </div>
        <div className="my-2 border-t border-dashed" />
        <div className="flex justify-between">
            <span className="text-muted-foreground">Min</span>
            <span>{stats.min?.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
            <span className="text-muted-foreground">25% (Q1)</span>
            <span>{stats.q25?.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
            <span className="text-muted-foreground">Median</span>
            <span>{stats.median?.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
            <span className="text-muted-foreground">75% (Q3)</span>
            <span>{stats.q75?.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
            <span className="text-muted-foreground">Max</span>
            <span>{stats.max?.toFixed(4)}</span>
        </div>
        <div className="my-2 border-t border-dashed" />
        <NumericShape stats={stats} />
        <div className="my-2 border-t border-dashed" />
        <NumericCounts stats={stats} vif={vif} />
    </>;
}

function NumericShape({ stats }: { stats: NumericStats; }) {
    return <>

        <div className="flex justify-between">
            <span className="text-muted-foreground">Skewness</span>
            <span>{stats.skewness?.toFixed(4)}</span>
        </div>
        {stats.skewness != null && (
            <div className="text-[10px] text-muted-foreground pl-2">
                {Math.abs(stats.skewness) < 0.5 ? '↳ Symmetric' :
                    Math.abs(stats.skewness) < 1 ? '↳ Moderately Skewed' : '↳ Highly Skewed'}
            </div>
        )}
        <div className="flex justify-between">
            <span className="text-muted-foreground">Kurtosis</span>
            <span>{stats.kurtosis?.toFixed(4)}</span>
        </div>
        {stats.kurtosis != null && (
            <div className="text-[10px] text-muted-foreground pl-2">
                {Math.abs(stats.kurtosis) < 0.5 ? '↳ Mesokurtic (Normal-like)' :
                    stats.kurtosis > 0 ? '↳ Leptokurtic (Heavy Tails)' : '↳ Platykurtic (Light Tails)'}
            </div>
        )}
    </>;
}

function NumericCounts({ stats, vif }: { stats: NumericStats; vif: ColumnProfile["vif"]; }) {
    return <>

        {stats.zeros_count != null && (
            <div className="flex justify-between">
                <span className="text-muted-foreground">Zeros</span>
                <span>{stats.zeros_count}</span>
            </div>
        )}
        {stats.negatives_count != null && (
            <div className="flex justify-between">
                <span className="text-muted-foreground">Negatives</span>
                <span>{stats.negatives_count}</span>
            </div>
        )}
        {vif != null && (
            <div className="flex justify-between items-center">
                <span className="text-muted-foreground flex items-center gap-1">
                    VIF
                    <InfoTooltip text="Variance Inflation Factor. VIF > 5 = high multicollinearity, VIF > 10 = severe." size="sm" />
                </span>
                <span className={vif > 5 ? 'text-red-500 font-semibold' : vif > 2 ? 'text-amber-500' : 'text-green-500'}>
                    {vif.toFixed(2)}
                </span>
            </div>
        )}
    </>;
}

function TextStatistics({ stats }: { stats: TextStats; }) {
    return (
        <>
            <div className="flex justify-between">
                <span className="text-muted-foreground">Avg Length</span>
                <span>{stats.avg_length?.toFixed(1)} chars</span>
            </div>
            <div className="flex justify-between">
                <span className="text-muted-foreground">Min Length</span>
                <span>{stats.min_length}</span>
            </div>
            <div className="flex justify-between">
                <span className="text-muted-foreground">Max Length</span>
                <span>{stats.max_length}</span>
            </div>
            {stats.sentiment_distribution && <SentimentStatistics distribution={stats.sentiment_distribution} />}
            {stats.common_words && stats.common_words.length > 0 && (
                <>
                    <div className="my-2 border-t border-dashed" />
                    <div className="text-xs font-semibold mb-1">Common Words</div>
                    <div className="flex flex-wrap gap-1">
                        {stats.common_words.slice(0, 10).map((w, idx) => (
                            <Badge key={idx} variant="secondary" className="text-[10px] h-4">
                                {w.word ?? w.value} ({w.count})
                            </Badge>
                        ))}
                    </div>
                </>
            )}
        </>
    );
}

function CategoricalStatistics({ stats }: { stats: CategoricalStats; }) {
    return (
        <>
            <div className="flex justify-between">
                <span className="text-muted-foreground">Unique</span>
                <span>{stats.unique_count}</span>
            </div>
            <div className="flex justify-between">
                <span className="text-muted-foreground">Mode</span>
                <span className="truncate max-w-[100px]" title={String(stats.top_k?.[0]?.value)}>
                    {String(stats.top_k?.[0]?.value || '-')}
                </span>
            </div>
            {stats.rare_labels_count != null && (
                <div className="flex justify-between">
                    <span className="text-muted-foreground">Rare Labels</span>
                    <span>{stats.rare_labels_count}</span>
                </div>
            )}
            {stats.top_k && stats.top_k.length > 0 && (
                <>
                    <div className="my-2 border-t border-dashed" />
                    <div className="text-xs font-semibold mb-1">Top Categories</div>
                    <div className="space-y-1">
                        {stats.top_k.slice(0, 5).map((item, i) => (
                            <div key={i} className="flex justify-between text-xs">
                                <span className="truncate max-w-[120px]" title={String(item.value)}>{String(item.value)}</span>
                                <span className="text-muted-foreground">{item.count}</span>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </>
    );
}

function SentimentStatistics({ distribution }: { distribution: NonNullable<TextStats["sentiment_distribution"]>; }) {
    const positive = distribution.positive || 0;
    const neutral = distribution.neutral || 0;
    const negative = distribution.negative || 0;
    return (
        <>
            <div className="my-2 border-t border-dashed" />
            <div className="text-xs font-semibold mb-1">Sentiment</div>
            <div className="flex h-3 rounded-full overflow-hidden w-full bg-secondary">
                <div className="bg-green-500 h-full" style={{ width: `${positive * 100}%` }} title={`Positive: ${(positive * 100).toFixed(1)}%`}></div>
                <div className="bg-gray-400 h-full" style={{ width: `${neutral * 100}%` }} title={`Neutral: ${(neutral * 100).toFixed(1)}%`}></div>
                <div className="bg-red-500 h-full" style={{ width: `${negative * 100}%` }} title={`Negative: ${(negative * 100).toFixed(1)}%`}></div>
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1 px-1">
                <span>Pos: {(positive * 100).toFixed(0)}%</span>
                <span>Neu: {(neutral * 100).toFixed(0)}%</span>
                <span>Neg: {(negative * 100).toFixed(0)}%</span>
            </div>
        </>
    );
}
