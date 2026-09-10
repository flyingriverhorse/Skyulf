import { CheckCircle2, Zap } from 'lucide-react';
import type { RunProgress } from './history';

/** Present the active run only when its receipt overlaps the current inspection. */
export function ParallelRunProgress({ progress }: { progress: RunProgress | null }) {
  if (!progress) return null;
  const { total, doneCount, pct, isDone } = progress;
  return (
    <div className={`px-4 py-2.5 border-b flex items-center gap-3 ${isDone
        ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700/50'
        : 'bg-amber-50 dark:bg-amber-900/30 border-amber-200 dark:border-amber-700/50'
      }`}>
      {isDone
        ? <CheckCircle2 className="w-4 h-4 text-green-600 dark:text-green-400 shrink-0" />
        : <Zap className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
      }
      <span className={`text-sm font-medium ${isDone
          ? 'text-green-800 dark:text-green-200'
          : 'text-amber-800 dark:text-amber-200'
        }`}>
        {isDone ? 'All branches complete!' : `Parallel Run: ${doneCount}/${total} branches complete`}
      </span>
      <div className={`flex-1 h-2 rounded-full overflow-hidden ${isDone ? 'bg-green-200 dark:bg-green-800' : 'bg-amber-200 dark:bg-amber-800'
        }`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ${isDone ? 'bg-green-500 dark:bg-green-400' : 'bg-amber-500 dark:bg-amber-400'
            }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs font-mono shrink-0 ${isDone ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'
        }`}>{pct}%</span>
    </div>
  );
}
