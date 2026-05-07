import { useCallback, useEffect, useState } from 'react';
import { monitoringApi, DriftJobOption } from '../../../core/api/monitoring';

/**
 * Fetches the list of training jobs that can be used as drift references and
 * exposes a `refresh` action plus a `refreshing` flag for the spinner state.
 * The list is also locally mutable via `updateJobDescription` so the meta-bar
 * can patch a job's description without a full re-fetch.
 */
export function useDriftJobs() {
    const [jobs, setJobs] = useState<DriftJobOption[]>([]);
    const [refreshing, setRefreshing] = useState(false);

    const fetchJobs = useCallback(async () => {
        setRefreshing(true);
        try {
            const data = await monitoringApi.getJobs();
            setJobs(data);
        } catch (err) {
            console.error('Failed to fetch jobs', err);
        } finally {
            setRefreshing(false);
        }
    }, []);

    useEffect(() => {
        void fetchJobs();
    }, [fetchJobs]);

    const updateJobDescription = useCallback(async (jobId: string, description: string) => {
        try {
            await monitoringApi.updateJobDescription(jobId, description);
            setJobs(prev => prev.map(j => (j.job_id === jobId ? { ...j, description } : j)));
        } catch (err) {
            console.error('Failed to save description', err);
        }
    }, []);

    return { jobs, refreshing, refresh: fetchJobs, updateJobDescription };
}
