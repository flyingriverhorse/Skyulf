import {
    AlertCircle, CheckCircle, Loader2, Play, RotateCcw, Sparkles, Trash2, Upload, Wand2,
    X, XCircle,
} from 'lucide-react';
import type { SchemaCheck } from './inferenceData';
import { SampleSizeSegmented } from './InferenceVisuals';
import { ThresholdOverrides } from './ThresholdOverrides';
import type { InferenceController } from './useInferenceController';

/** Compose the input editor, schema checks, sample actions, and threshold overrides. */
export function InferenceInputPanel({ controller }: {
    controller: Pick<InferenceController,
        | 'csvInputRef' | 'handleCsvChange' | 'sampleSize' | 'setSampleSize' | 'datasetId'
        | 'handleReloadSample' | 'isReloadingSample' | 'handleFormatJson' | 'inputStatus'
        | 'handleClearInput' | 'schemaChips' | 'excludedColumns' | 'showBanner'
        | 'autoFilterInfo' | 'setBannerDismissed' | 'editorWrapRef' | 'handleDragOver'
        | 'handleDragLeave' | 'handleDrop' | 'inputData' | 'setInputData'
        | 'handleTextareaKeyDown' | 'isDragging' | 'hasSchemaIssues' | 'schemaCheck'
        | 'handleFixMissingFields' | 'hasBlockingMissingFields' | 'acknowledgeMissingFields'
        | 'setAcknowledgeMissingFields' | 'hasUnknownTypedFields' | 'activeDeployment'
        | 'activeRun' | 'canRunPrediction' | 'handleCancelRun' | 'handlePredict'
        | 'overrideThresholdsEnabled' | 'setOverrideThresholdsEnabled'
        | 'overrideThresholdsValue' | 'handleOverrideThresholdChange'
        | 'handleRemoveOverrideEntry' | 'newOverrideClass' | 'setNewOverrideClass'
        | 'newOverrideThreshold' | 'setNewOverrideThreshold' | 'handleAddOverrideEntry'
        | 'singleProbMap' | 'handlePrefillFromLastPrediction' | 'savedThresholds'
        | 'handlePrefillFromSavedThresholds'
    >
}) {
    return (<div className="bg-white dark:bg-gray-800 p-4 sm:p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col lg:h-full lg:min-h-0">
        <InputToolbar controller={controller} />

        {/* Schema chips: scrollable strip showing expected fields. */}
        <SchemaFeatures controller={controller} />

        {/* Excluded columns reminder — only if any are configured. */}
        <ExcludedColumns controller={controller} />

        <SampleNotice controller={controller} />

        {/* Editor wrapper handles drag-and-drop CSV. */}
        <InputEditor controller={controller} />

        {/* Schema validation badges + one-click fix. */}
        <SchemaValidation controller={controller} />

        <PredictionControls controller={controller} />

        {/* Ad-hoc override thresholds — applied only to this prediction, not persisted server-side. */}
        <ThresholdOverrides controller={controller} />
    </div>);
}

function InputToolbar({ controller }: {
    controller: Pick<InferenceController,
        | 'csvInputRef' | 'handleCsvChange' | 'sampleSize' | 'setSampleSize' | 'datasetId'
        | 'handleReloadSample' | 'isReloadingSample' | 'handleFormatJson' | 'inputStatus'
        | 'handleClearInput'
    >
}) {
    const {
        csvInputRef, handleCsvChange, sampleSize, setSampleSize, datasetId,
        handleReloadSample, isReloadingSample, handleFormatJson, inputStatus,
        handleClearInput,
    } = controller;
    return (<div className="flex justify-between items-start mb-3 gap-3 flex-wrap">
        <h3 id="inference-input-heading" className="text-lg font-medium text-gray-800 dark:text-gray-100">
            Input Data (JSON)
        </h3>
        <div className="flex items-center gap-1 flex-wrap">
            <input
                ref={csvInputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={handleCsvChange}
            />
            <button
                onClick={() => csvInputRef.current?.click()}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                title="Upload a CSV file (or drag & drop). Target/dropped columns are stripped automatically."
            >
                <Upload className="w-3 h-3" /> CSV
            </button>
            <SampleSizeSegmented
                value={sampleSize}
                onChange={setSampleSize}
                disabled={!datasetId}
            />
            <button
                onClick={() => void handleReloadSample()}
                disabled={!datasetId || isReloadingSample}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Load fresh random rows from the training dataset (target / dropped columns excluded)"
            >
                <RotateCcw
                    className={`w-3 h-3 ${isReloadingSample ? 'animate-spin' : ''}`}
                />
                Sample
            </button>
            <button
                onClick={handleFormatJson}
                disabled={!inputStatus.valid}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Pretty-print JSON"
            >
                <Sparkles className="w-3 h-3" /> Format
            </button>
            <button
                onClick={handleClearInput}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                title="Reset the editor and clear results"
            >
                <Trash2 className="w-3 h-3" /> Clear
            </button>
        </div>
    </div>);
}

function SchemaFeatures({ controller }: { controller: Pick<InferenceController, 'schemaChips'> }) {
    const { schemaChips } = controller;
    return (<>{schemaChips.length > 0 && (
        <div className="mb-3 flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin">
            <span className="text-[10px] uppercase tracking-wider text-gray-400 shrink-0">
                Schema
            </span>
            {schemaChips.map(col => (
                <span
                    key={col.name}
                    className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-gray-100 dark:bg-gray-700/60 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-600"
                    title={`${col.name} (${col.type})`}
                >
                    <span className="font-mono">{col.name}</span>
                    <span className="text-gray-500 dark:text-gray-400">{col.type}</span>
                </span>
            ))}
        </div>
    )}</>);
}

function ExcludedColumns({ controller }: { controller: Pick<InferenceController, 'excludedColumns'> }) {
    const { excludedColumns } = controller;
    return (<>{excludedColumns.size > 0 && (
        <div className="mb-3 flex items-center gap-2 overflow-x-auto pb-1 scrollbar-thin text-[11px]">
            <span className="text-[10px] uppercase tracking-wider text-gray-400 shrink-0">
                Excluded
            </span>
            {[...excludedColumns].map(col => (
                <span
                    key={col}
                    className="shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-rose-50 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800"
                    title="Stripped automatically from CSV uploads and Sample calls"
                >
                    <Trash2 className="w-2.5 h-2.5" />
                    <span className="font-mono">{col}</span>
                </span>
            ))}
        </div>
    )}</>);
}

function SampleNotice({ controller }: { controller: Pick<InferenceController, 'showBanner' | 'autoFilterInfo' | 'setBannerDismissed'> }) {
    const { showBanner, autoFilterInfo, setBannerDismissed } = controller;
    return (<>{showBanner && (
        <div className="mb-3 flex items-start gap-2 text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-900/20 px-3 py-2 rounded border border-amber-200 dark:border-amber-800">
            <span className="flex-1">
                {autoFilterInfo}
                <span className="block text-[10px] text-amber-600/80 dark:text-amber-400/80 mt-0.5">
                    Please verify fields before running.
                </span>
            </span>
            <button
                onClick={() => setBannerDismissed(true)}
                className="text-amber-500 hover:text-amber-700 dark:hover:text-amber-200"
                title="Dismiss"
            >
                <X className="w-3 h-3" />
            </button>
        </div>
    )}</>);
}

function InputEditor({ controller }: {
    controller: Pick<InferenceController,
        | 'editorWrapRef' | 'handleDragOver' | 'handleDragLeave' | 'handleDrop'
        | 'inputStatus' | 'inputData' | 'setInputData' | 'handleTextareaKeyDown'
        | 'isDragging'
    >
}) {
    const {
        editorWrapRef, handleDragOver, handleDragLeave, handleDrop, inputStatus, inputData,
        setInputData, handleTextareaKeyDown, isDragging,
    } = controller;
    return (<div
        ref={editorWrapRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className="relative flex-1 min-h-[16rem] lg:min-h-0"
    >
        <textarea
            id="inference-input-editor"
            aria-labelledby="inference-input-heading"
            aria-describedby="inference-input-status"
            aria-invalid={!inputStatus.valid}
            className="absolute inset-0 w-full h-full p-4 font-mono text-sm bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none resize-none"
            value={inputData}
            onChange={e => setInputData(e.target.value)}
            onKeyDown={handleTextareaKeyDown}
            placeholder='[{"col1": 1, "col2": "A"}]'
            spellCheck={false}
        />
        {isDragging && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-blue-500/10 dark:bg-blue-400/10 border-2 border-dashed border-blue-400 dark:border-blue-500 rounded-lg">
                <div className="flex items-center gap-2 text-blue-600 dark:text-blue-300 font-medium text-sm">
                    <Upload className="w-5 h-5" />
                    Drop CSV to ingest
                </div>
            </div>
        )}
    </div>);
}

function SchemaValidation({ controller }: {
    controller: Pick<InferenceController,
        | 'hasSchemaIssues' | 'schemaCheck' | 'handleFixMissingFields'
        | 'hasBlockingMissingFields' | 'acknowledgeMissingFields'
        | 'setAcknowledgeMissingFields' | 'hasUnknownTypedFields'
    >
}) {
    const {
        hasSchemaIssues, schemaCheck, handleFixMissingFields, hasBlockingMissingFields,
        acknowledgeMissingFields, setAcknowledgeMissingFields, hasUnknownTypedFields,
    } = controller;
    if (!schemaCheck) return null;
    return (<>{hasSchemaIssues && (
        <div className="mt-2 flex flex-col gap-1.5 text-[11px]">
            <div className="flex flex-wrap items-center gap-1.5">
                {schemaCheck.missing.length > 0 && (
                    <>
                        <span
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-50 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800"
                            title={schemaCheck.missing.join(', ')}
                        >
                            <AlertCircle className="w-3 h-3" />
                            {schemaCheck.missing.length} missing
                        </span>
                        <button
                            onClick={() => void handleFixMissingFields()}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
                            title="Review and fill missing fields with value 0"
                        >
                            <Wand2 className="w-3 h-3" /> Fix
                        </button>
                    </>
                )}
                {schemaCheck.extra.length > 0 && (
                    <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800"
                        title={schemaCheck.extra.join(', ')}
                    >
                        <AlertCircle className="w-3 h-3" />
                        {schemaCheck.extra.length} extra
                    </span>
                )}
                <span className="text-[10px] text-gray-400 italic ml-1">
                    Hover badges for field names.
                </span>
            </div>

            {/* Per-row detail: which specific rows are still missing
                                fields, since a field being present "somewhere" in the
                                batch doesn't help the row that's actually missing it. */}
            <MissingSchemaRows schemaCheck={schemaCheck} />

            {hasBlockingMissingFields && (
                <label className="flex items-start gap-1.5 text-gray-600 dark:text-gray-300 cursor-pointer">
                    <input
                        type="checkbox"
                        checked={acknowledgeMissingFields}
                        onChange={e => setAcknowledgeMissingFields(e.target.checked)}
                        className="mt-0.5"
                        data-testid="acknowledge-missing-fields"
                    />
                    <span>
                        I understand some rows are missing schema fields and want to
                        run anyway.
                    </span>
                </label>
            )}

            {hasUnknownTypedFields && (
                <p className="text-[10px] text-gray-400 italic">
                    This model&apos;s schema doesn&apos;t report field types — values
                    are checked for presence only, not type or range, before sending.
                </p>
            )}
        </div>
    )}</>);
}

function PredictionControls({ controller }: {
    controller: Pick<InferenceController,
        | 'activeDeployment' | 'activeRun' | 'inputStatus' | 'canRunPrediction'
        | 'handleCancelRun' | 'handlePredict'
    >
}) {
    const { activeDeployment, activeRun, inputStatus, canRunPrediction, handleCancelRun, handlePredict } = controller;
    const disabled = !activeDeployment || Boolean(activeRun) || !inputStatus.valid || !canRunPrediction;
    return (<div className="mt-3 flex justify-between items-center gap-3 flex-wrap">
        <InputStatusLabel controller={controller} />
        <div className="flex items-center gap-3">
            <kbd className="hidden md:inline-flex items-center gap-1 text-[10px] text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded border border-gray-200 dark:border-gray-600 font-mono">
                Ctrl
                <span className="text-gray-400">+</span>
                Enter
            </kbd>
            {activeRun && (
                <button
                    type="button"
                    onClick={handleCancelRun}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium text-red-600 dark:text-red-400 border border-red-200 dark:border-red-800 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                    title={`Cancel ${activeRun.label}`}
                >
                    <XCircle className="w-4 h-4" /> Cancel
                </button>
            )}
            <button
                onClick={() => void handlePredict()}
                disabled={
                    disabled
                }
                title={
                    predictionButtonTitle(canRunPrediction, activeRun)
                }
                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${disabled
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed dark:bg-gray-800 dark:text-gray-600'
                        : 'action-primary shadow-sm'
                    }`}
            >
                {activeRun ? (
                    <>
                        <Loader2 className="w-4 h-4 animate-spin" /> {activeRun.label} running…
                    </>
                ) : (
                    <>
                        <Play className="w-4 h-4" /> Run Prediction
                    </>
                )}
            </button>
        </div>
    </div>);
}

function InputStatusLabel({ controller }: { controller: Pick<InferenceController, 'inputStatus'> }) {
    const { inputStatus } = controller;
    return (<span
        id="inference-input-status"
        role="status"
        className={`text-xs flex items-center gap-1 ${inputStatus.valid
                ? 'text-gray-500 dark:text-gray-400'
                : 'text-red-600 dark:text-red-400'
            }`}
    >
        {inputStatus.valid ? (
            <>
                <CheckCircle className="w-3 h-3" /> {inputStatus.message}
            </>
        ) : (
            <>
                <AlertCircle className="w-3 h-3" /> {inputStatus.message}
            </>
        )}
    </span>);
}

function predictionButtonTitle(canRunPrediction: boolean, activeRun: InferenceController['activeRun']) {
    return !canRunPrediction
        ? 'Some rows are missing schema fields — fix them or check the acknowledgement above to run anyway'
        : activeRun
            ? `${activeRun.label} is already running`
            : undefined;
}

function MissingSchemaRows({ schemaCheck }: { schemaCheck: SchemaCheck }) {
    return (<>{schemaCheck.rowIssues.length > 0 && (
        <ul
            data-testid="schema-row-issues"
            className="font-mono bg-rose-50/60 dark:bg-rose-900/10 border border-rose-100 dark:border-rose-900/40 rounded px-2 py-1 space-y-0.5 max-h-20 overflow-auto"
        >
            {schemaCheck.rowIssues.slice(0, 5).map(issue => (
                <li key={issue.rowIndex} className="text-rose-700 dark:text-rose-300">
                    Row {issue.rowIndex + 1}: missing {issue.missing.join(', ')}
                </li>
            ))}
            {schemaCheck.rowIssues.length > 5 && (
                <li className="text-rose-500 dark:text-rose-400">
                    …and {schemaCheck.rowIssues.length - 5} more row(s)
                </li>
            )}
        </ul>
    )}</>);
}
