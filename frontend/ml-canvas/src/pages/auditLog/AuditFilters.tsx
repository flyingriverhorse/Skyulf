import React from 'react';
import { ANONYMOUS_ACTOR } from '../../core/api/pipelineVersions';
import { resolveDatasetSourceId } from './auditHistoryModel';
import type { AuditLogState } from './useAuditLog';

const LIMIT_OPTIONS: ReadonlyArray<number> = [25, 50, 100, 200];

/** Present the dataset and server-side history filter controls. */
export const AuditFilters: React.FC<{ state: AuditLogState }> = ({ state }) => {
    const { actorFilter, setActorFilter, kindFilter, setKindFilter, fromTime, setFromTime,
        toTime, setToTime, limit, setLimit, data, actorOptions, kindOptions } = state;
    return (
        <div className="flex flex-wrap items-end gap-4 mb-4">
            <AuditDatasetPicker state={state} />
            <div className="min-w-[220px]">
                <label
                    htmlFor="audit-actor"
                    className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1"
                >
                    Actor
                </label>
                <select
                    id="audit-actor"
                    value={actorFilter}
                    onChange={e => setActorFilter(e.target.value)}
                    disabled={!data}
                    className="w-full px-3 py-2 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                >
                    <option value="all">All actors</option>
                    {actorOptions.userIds.map(userId => (
                        <option key={userId} value={userId}>
                            user #{userId}
                        </option>
                    ))}
                    {actorOptions.hasAnonymous && (
                        <option value={ANONYMOUS_ACTOR}>anonymous</option>
                    )}
                </select>
            </div>
            <div className="min-w-[180px]">
                <label
                    htmlFor="audit-kind"
                    className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1"
                >
                    Action kind
                </label>
                <select
                    id="audit-kind"
                    value={kindFilter}
                    onChange={e => setKindFilter(e.target.value)}
                    disabled={!data}
                    className="w-full px-3 py-2 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                >
                    <option value="all">All kinds</option>
                    {kindOptions.map(kind => (
                        <option key={kind} value={kind}>
                            {kind}
                        </option>
                    ))}
                </select>
            </div>
            <div className="min-w-[190px]">
                <label
                    htmlFor="audit-from"
                    className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1"
                >
                    From time
                </label>
                <input
                    id="audit-from"
                    type="datetime-local"
                    value={fromTime}
                    onChange={e => setFromTime(e.target.value)}
                    disabled={!data}
                    className="w-full px-3 py-2 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                />
            </div>
            <div className="min-w-[190px]">
                <label
                    htmlFor="audit-to"
                    className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1"
                >
                    To time
                </label>
                <input
                    id="audit-to"
                    type="datetime-local"
                    value={toTime}
                    onChange={e => setToTime(e.target.value)}
                    disabled={!data}
                    className="w-full px-3 py-2 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                />
            </div>
            <div>
                <span className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                    Limit
                </span>
                <div className="inline-flex rounded border border-gray-300 dark:border-gray-600 overflow-hidden">
                    {LIMIT_OPTIONS.map(opt => (
                        <button
                            key={opt}
                            type="button"
                            onClick={() => setLimit(opt)}
                            className={`px-3 py-2 text-xs font-medium ${
                                limit === opt
                                    ? 'bg-violet-500 text-white'
                                    : 'bg-white dark:bg-slate-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-slate-700'
                            }`}
                        >
                            {opt}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

/** Coerce legacy dataset IDs consistently with canvas saves. */
const AuditDatasetPicker: React.FC<{ state: AuditLogState }> = ({ state }) => {
    const { datasets, datasetsLoading, datasetId, setDatasetId } = state;
    return (
        <div className="flex-1 min-w-[240px]">
            <label
                htmlFor="audit-dataset"
                className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1"
            >
                Dataset
            </label>
            <select
                id="audit-dataset"
                value={datasetId}
                onChange={e => setDatasetId(e.target.value)}
                disabled={datasetsLoading || !datasets || datasets.length === 0}
                className="w-full px-3 py-2 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100 disabled:opacity-50"
            >
                {!datasets || datasets.length === 0 ? (
                    <option value="">
                        {datasetsLoading ? 'Loading…' : 'No datasets'}
                    </option>
                ) : (
                    datasets.map(d => {
                        const sid = resolveDatasetSourceId(d);
                        const shortSid = sid.length > 8 ? sid.slice(0, 8) : sid;
                        return (
                            <option key={sid} value={sid}>
                                {d.name} ({shortSid})
                            </option>
                                );
                            })
                        )}
                    </select>
                </div>

    );
};
