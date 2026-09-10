import { useEffect, useRef, useState } from 'react';
import { SavedThresholdInfo, thresholdTuningApi } from '../../../core/api/thresholdTuning';


/** Fetch and keep the saved threshold record for the currently active job. */
export const useSavedThresholdInfo = (jobId: string | null): SavedThresholdInfo | null => {
    const [savedThresholds, setSavedThresholds] = useState<SavedThresholdInfo | null>(null);
    const requestSeq = useRef(0);

    useEffect(() => {
        const requestId = ++requestSeq.current;
        if (!jobId) {
            setSavedThresholds(null);
            return;
        }

        let cancelled = false;
        setSavedThresholds(null);
        void thresholdTuningApi.get(jobId)
            .then(saved => {
                if (cancelled || requestSeq.current !== requestId) return;
                setSavedThresholds(saved);
            })
            .catch(err => {
                if (cancelled || requestSeq.current !== requestId) return;
                console.warn('Failed to fetch saved tuned thresholds', err);
                setSavedThresholds(null);
            });

        return () => {
            cancelled = true;
        };
    }, [jobId]);

    return savedThresholds;
};
