import { useCallback, useEffect, useMemo, useState } from 'react';
import { DatasetService } from '../../../core/api/datasets';
import { deploymentApi, DeploymentInfo } from '../../../core/api/deployment';
import { jobsApi } from '../../../core/api/jobs';
import { toast } from '../../../core/toast';
import { DEFAULT_INPUT, LS_INPUT, LS_SAMPLE_SIZE, projectSampleRow, SAMPLE_OPTIONS } from './inferenceData';

/** Load deployment features and dataset samples without replacing restored input. */
export function useDeploymentInput(setInputData: (input: string) => void) {

    const [activeDeployment, setActiveDeployment] = useState<DeploymentInfo | null>(null);
    const [datasetId, setDatasetId] = useState<string | null>(null);
    /** Columns the model never trained on (target + user-dropped). */
    const [excludedColumns, setExcludedColumns] = useState<Set<string>>(new Set());
    const [isReloadingSample, setIsReloadingSample] = useState(false);
    const [sampleSize, setSampleSize] = useState<number>(() => {
        try {
            const v = Number(localStorage.getItem(LS_SAMPLE_SIZE));
            return SAMPLE_OPTIONS.includes(v) ? v : 1;
        } catch {
            return 1;
        }
    });

    const [autoFilterInfo, setAutoFilterInfo] = useState<string | null>(null);

    const schemaChips = useMemo(
        () => activeDeployment?.input_schema ?? [],
        [activeDeployment],
    );
    useEffect(() => {
        try {
            localStorage.setItem(LS_SAMPLE_SIZE, String(sampleSize));
        } catch {
            /* ignore */
        }
    }, [sampleSize]);

    const loadDeploymentSample = async (
        deployment: DeploymentInfo,
        initialData: Record<string, unknown>,
        usedSchema: boolean,
        excluded: Set<string>,
    ) => {
        try {
            const job = await jobsApi.getJob(deployment.job_id);
            const targetColumn = job.target_column;
            const droppedColumns = job.dropped_columns || [];
            addExcludedColumns(excluded, targetColumn, droppedColumns);

            if (job.dataset_id) {
                setDatasetId(job.dataset_id);
                const sample = await DatasetService.getSample(job.dataset_id, 1);
                if (sample.length > 0) {
                    const sampleRow = sample[0] as Record<string, unknown>;

                    if (usedSchema) {
                        Object.keys(initialData).forEach(key => {
                            if (key in sampleRow) initialData[key] = sampleRow[key];
                        });
                    } else {
                        initialData = projectSampleRow(sampleRow, [], excluded);

                        const filterInfo = describeSampleFilter(targetColumn, droppedColumns);
                        if (filterInfo) setAutoFilterInfo(filterInfo);
                    }
                }
            }
        } catch (err) {
            console.warn('Failed to fetch dataset sample', err);
        }
        return initialData;
    };

    const loadActiveDeployment = useCallback(async () => {
        try {
            const deployment = await deploymentApi.getActive();
            setActiveDeployment(deployment);
            if (!deployment) {
                setDatasetId(null);
                setExcludedColumns(new Set());
                return;
            }

            let initialData: Record<string, unknown> = {};
            let usedSchema = false;
            const excluded = new Set<string>();

            if (deployment.input_schema && deployment.input_schema.length > 0) {
                deployment.input_schema.forEach(col => {
                    initialData[col.name] = 0;
                });
                usedSchema = true;
                setAutoFilterInfo(
                    `Schema loaded from artifacts (${deployment.input_schema.length} features)`,
                );
            }

            if (deployment.job_id) {
                initialData = await loadDeploymentSample(deployment, initialData, usedSchema, excluded);
            }
            setExcludedColumns(excluded);

            seedInitialInput(initialData, setInputData);
        } catch (e) {
            console.error('Failed to load active deployment', e);
            setActiveDeployment(null);
            setDatasetId(null);
            setExcludedColumns(new Set());
        }
    }, [setInputData]);

    useEffect(() => {
        void loadActiveDeployment();
    }, [loadActiveDeployment]);

    const handleReloadSample = useCallback(async () => {
        if (!datasetId || isReloadingSample) return;
        setIsReloadingSample(true);
        try {
            const sample = await DatasetService.getSample(datasetId, sampleSize);
            if (sample.length === 0) {
                toast.error('Dataset returned no rows');
                return;
            }
            const rows = sample.map(r =>
                projectSampleRow(
                    r as Record<string, unknown>,
                    schemaChips,
                    excludedColumns,
                ),
            );
            setInputData(JSON.stringify(rows, null, 2));
            toast.success(
                `Loaded ${rows.length} sample row${rows.length === 1 ? '' : 's'}` +
                (excludedColumns.size > 0 ? ` (excluded ${excludedColumns.size} cols)` : ''),
            );
        } catch (e) {
            console.error('Failed to reload sample', e);
            toast.error('Could not fetch new samples');
        } finally {
            setIsReloadingSample(false);
        }
    }, [datasetId, isReloadingSample, sampleSize, schemaChips, excludedColumns, setInputData]);
    return {
        activeDeployment, setActiveDeployment, datasetId, setDatasetId, excludedColumns,
        setExcludedColumns, isReloadingSample, sampleSize, setSampleSize, autoFilterInfo,
        schemaChips, handleReloadSample,
    };
}

/** Collect the target and dropped columns excluded from model inputs. */
function addExcludedColumns(excluded: Set<string>, targetColumn: unknown, droppedColumns: string[]) {
    if (typeof targetColumn === 'string' && targetColumn) excluded.add(targetColumn);
    droppedColumns.forEach(col => {
        if (typeof col === 'string' && col) excluded.add(col);
    });
}

/** Describe dataset fields omitted when no artifact schema is available. */
function describeSampleFilter(targetColumn: unknown, droppedColumns: string[]) {
    const droppedInfo: string[] = [];
    if (targetColumn) droppedInfo.push(`Target: ${targetColumn}`);
    if (droppedColumns.length > 0) droppedInfo.push(`Dropped: ${droppedColumns.length} cols`);
    return droppedInfo.length > 0 ? `Auto-filtered from dataset: ${droppedInfo.join(', ')}` : null;
}

/** Seed a sample only when browser storage contains no custom input. */
function seedInitialInput(initialData: Record<string, unknown>, setInputData: (input: string) => void) {
    if (Object.keys(initialData).length > 0) {
        let userInputIsDefault = false;
        try {
            const stored = localStorage.getItem(LS_INPUT);
            userInputIsDefault = stored == null || stored === DEFAULT_INPUT;
        } catch {
            userInputIsDefault = true;
        }
        if (userInputIsDefault) {
            setInputData(JSON.stringify([initialData], null, 2));
        }
    }
}
