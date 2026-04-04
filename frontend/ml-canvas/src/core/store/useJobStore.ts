import { create } from 'zustand';
import { jobsApi, JobInfo, RunPipelineRequest } from '../api/jobs';

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
  
  // Polling
  startPolling: () => void;
  stopPolling: () => void;
}

// Polling interval in ms
const POLLING_INTERVAL = 3000;
const PAGE_SIZE = 50;

export const useJobStore = create<JobState>((set, get) => {
  let pollingInterval: ReturnType<typeof setInterval> | null = null;

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

    startPolling: () => {
      if (pollingInterval) return;
      
      pollingInterval = setInterval(() => { void (async () => {
        // Silent fetch (no loading spinner)
        try {
          const latestJobs = await jobsApi.getJobs(PAGE_SIZE, 0);
          
          set(state => {
              if (state.jobs.length <= PAGE_SIZE) {
                  return { jobs: latestJobs };
              } else {
                  // Replace first page, keep the rest
                  return { jobs: [...latestJobs, ...state.jobs.slice(PAGE_SIZE)] };
              }
          });
          
          // Stop polling if no active jobs and drawer is closed
          const hasActive = latestJobs.some(j => j.status === 'running' || j.status === 'queued');
          if (!hasActive && !get().isDrawerOpen) {
            get().stopPolling();
          }

          // Auto-clear parallel run when all tracked jobs are terminal
          const parallelRun = get().activeParallelRun;
          if (parallelRun) {
            const TERMINAL = new Set(['completed', 'failed', 'cancelled']);
            const allDone = parallelRun.jobIds.every(id => {
              const j = latestJobs.find(job => job.job_id === id);
              return j && TERMINAL.has(j.status);
            });
            if (allDone) {
              if (!parallelRun.completedAt) {
                // Mark completion time, keep banner visible
                set({ activeParallelRun: { ...parallelRun, completedAt: new Date().toISOString() } });
              } else {
                // Clear after 5s linger
                const elapsed = Date.now() - new Date(parallelRun.completedAt).getTime();
                if (elapsed >= 5000) set({ activeParallelRun: null });
              }
            }
          }
        } catch (error) {
          console.error('Polling error:', error);
        }
      })(); }, POLLING_INTERVAL);
    },

    stopPolling: () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
      }
    }
  };
});
