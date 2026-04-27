import { create } from 'zustand';
import { jobsApi, JobInfo, RunPipelineRequest } from '../api/jobs';
import { jobEventsSocket } from '../realtime/jobEventsSocket';

interface ActiveParallelRun {
  jobIds: string[];
  startedAt: string;
  completedAt?: string;
}

interface JobState {
  jobs: JobInfo[];
  isLoading: boolean;
  isDrawerOpen: boolean;
  activeTab: 'basic_training' | 'advanced_tuning';
  hasMore: boolean;
  skip: number;
  activeParallelRun: ActiveParallelRun | null;
  
  // Actions
  fetchJobs: () => Promise<void>;
  loadMoreJobs: () => Promise<void>;
  submitJob: (payload: RunPipelineRequest) => Promise<string>;
  cancelJob: (jobId: string) => Promise<void>;
  toggleDrawer: (isOpen?: boolean) => void;
  setTab: (tab: 'basic_training' | 'advanced_tuning') => void;
  setActiveParallelRun: (run: ActiveParallelRun | null) => void;
  promoteJob: (jobId: string) => Promise<void>;
  unpromoteJob: (jobId: string) => Promise<void>;
  
  // Polling
  startPolling: () => void;
  stopPolling: () => void;
}

// Polling cadence. WS-connected: long safety net. WS-disconnected:
// short fallback so UX still feels live if the realtime layer is down.
const POLL_INTERVAL_FAST = 3000;
const POLL_INTERVAL_SLOW = 30_000;
const PAGE_SIZE = 50;

export const useJobStore = create<JobState>((set, get) => {
  let pollingInterval: ReturnType<typeof setInterval> | null = null;
  let unsubscribeWs: (() => void) | null = null;
  let unsubscribeStatus: (() => void) | null = null;
  let wsConnected = false;
  let refreshTimer: ReturnType<typeof setTimeout> | null = null;

  // Coalesce bursty WS events (a Celery task can publish status +
  // progress + status within a few ms). One refresh per 250ms is enough
  // for a UI that humans look at.
  const scheduleRefresh = (run: () => Promise<void>): void => {
    if (refreshTimer) return;
    refreshTimer = setTimeout(() => {
      refreshTimer = null;
      void run();
    }, 250);
  };

  const restartIntervalAtCurrentCadence = (): void => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
    const cadence = wsConnected ? POLL_INTERVAL_SLOW : POLL_INTERVAL_FAST;
    pollingInterval = setInterval(() => { void runPollTick(); }, cadence);
  };

  // The shared poll-tick body: also used as the WS-event refresh.
  const runPollTick = async (): Promise<void> => {
    try {
      const latestJobs = await jobsApi.getJobs(PAGE_SIZE, 0);

      set(state => {
          if (state.jobs.length <= PAGE_SIZE) {
              return { jobs: latestJobs };
          } else {
              return { jobs: [...latestJobs, ...state.jobs.slice(PAGE_SIZE)] };
          }
      });

      // Stop polling if no active jobs and drawer is closed. WS still
      // listens — a fresh `created` event will revive the loop.
      const hasActive = latestJobs.some(j => j.status === 'running' || j.status === 'queued');
      if (!hasActive && !get().isDrawerOpen) {
        get().stopPolling();
      }

      const parallelRun = get().activeParallelRun;
      if (parallelRun) {
        const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
        const allDone = parallelRun.jobIds.every(id => {
          const j = latestJobs.find(job => job.job_id === id);
          return j && TERMINAL.has(j.status);
        });
        if (allDone) {
          if (!parallelRun.completedAt) {
            set({ activeParallelRun: { ...parallelRun, completedAt: new Date().toISOString() } });
          } else {
            const elapsed = Date.now() - new Date(parallelRun.completedAt).getTime();
            if (elapsed >= 5000) set({ activeParallelRun: null });
          }
        }
      }
    } catch (error) {
      console.error('Polling error:', error);
    }
  };

  return {
    jobs: [],
    isLoading: false,
    isDrawerOpen: false,
    activeTab: 'advanced_tuning',
    hasMore: true,
    skip: 0,
    activeParallelRun: null,

    fetchJobs: async () => {
      set({ isLoading: true, skip: 0 });
      try {
        const jobs = await jobsApi.getJobs(PAGE_SIZE, 0);
        set({ jobs, isLoading: false, hasMore: jobs.length === PAGE_SIZE });
      } catch (error) {
        console.error('Failed to fetch jobs:', error);
        set({ isLoading: false });
      }
    },

    loadMoreJobs: async () => {
        const { skip, jobs, isLoading, hasMore } = get();
        if (isLoading || !hasMore) return;
        
        const nextSkip = skip + PAGE_SIZE;
        set({ isLoading: true });
        try {
            const newJobs = await jobsApi.getJobs(PAGE_SIZE, nextSkip);
            set({ 
                jobs: [...jobs, ...newJobs], 
                isLoading: false, 
                skip: nextSkip,
                hasMore: newJobs.length === PAGE_SIZE
            });
        } catch (error) {
            console.error('Failed to load more jobs:', error);
            set({ isLoading: false });
        }
    },

    submitJob: async (payload: RunPipelineRequest) => {
      try {
        const response = await jobsApi.runPipeline(payload);
        // Refresh list immediately
        await get().fetchJobs();
        // Ensure polling is running
        get().startPolling();
        return response.job_id;
      } catch (error) {
        console.error('Failed to submit job:', error);
        throw error;
      }
    },

    cancelJob: async (jobId: string) => {
      try {
        await jobsApi.cancelJob(jobId);
        await get().fetchJobs();
      } catch (error) {
        console.error('Failed to cancel job:', error);
        throw error;
      }
    },

    toggleDrawer: (isOpen) => {
      const currentOpen = get().isDrawerOpen;
      const nextOpen = isOpen !== undefined ? isOpen : !currentOpen;
      
      set({ isDrawerOpen: nextOpen });
      
      // Fetch jobs when opening
      if (nextOpen) {
        get().fetchJobs();
        get().startPolling();
      } else {
        // Stop polling when closing? Maybe keep it running if there are active jobs?
        // For now, let's keep it simple and stop when closed to save resources, 
        // unless we want global notifications.
        // Better: Check if any job is running.
        const hasRunning = get().jobs.some(j => j.status === 'running' || j.status === 'queued');
        if (!hasRunning) {
            get().stopPolling();
        }
      }
    },

    setTab: (tab) => set({ activeTab: tab }),

    setActiveParallelRun: (run) => set({ activeParallelRun: run }),

    promoteJob: async (jobId: string) => {
      await jobsApi.promoteJob(jobId);
      set({ jobs: get().jobs.map(j => j.job_id === jobId ? { ...j, promoted_at: new Date().toISOString() } : j) });
    },

    unpromoteJob: async (jobId: string) => {
      await jobsApi.unpromoteJob(jobId);
      set({ jobs: get().jobs.map(j => j.job_id === jobId ? { ...j, promoted_at: null } : j) });
    },

    startPolling: () => {
      // Subscribe to the realtime channel once; events trigger a
      // (debounced) full refresh so the UI converges without burning
      // a request every 3 seconds.
      if (!unsubscribeWs) {
        unsubscribeWs = jobEventsSocket.subscribe(() => {
          scheduleRefresh(runPollTick);
        });
      }
      if (!unsubscribeStatus) {
        unsubscribeStatus = jobEventsSocket.onStatus((connected) => {
          const changed = connected !== wsConnected;
          wsConnected = connected;
          if (changed && pollingInterval) {
            // Re-arm the safety net at the new cadence.
            restartIntervalAtCurrentCadence();
          }
          if (connected) {
            // Reconnect: pull fresh state in case we missed events.
            scheduleRefresh(runPollTick);
          }
        });
      }
      if (pollingInterval) return;
      restartIntervalAtCurrentCadence();
    },

    stopPolling: () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
      }
      // Keep the WS subscription alive while the tab is open: a new
      // `created` event from elsewhere should still revive the loop.
    }
  };
});
