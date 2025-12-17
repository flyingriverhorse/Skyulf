import { create } from 'zustand';
import { jobsApi, JobInfo, RunPipelineRequest } from '../api/jobs';

interface JobState {
  jobs: JobInfo[];
  isLoading: boolean;
  isDrawerOpen: boolean;
  activeTab: 'training' | 'tuning';
  hasMore: boolean;
  skip: number;
  
  // Actions
  fetchJobs: () => Promise<void>;
  loadMoreJobs: () => Promise<void>;
  submitJob: (payload: RunPipelineRequest) => Promise<string>;
  cancelJob: (jobId: string) => Promise<void>;
  toggleDrawer: (isOpen?: boolean) => void;
  setTab: (tab: 'training' | 'tuning') => void;
  
  // Polling
  startPolling: () => void;
  stopPolling: () => void;
}

// Polling interval in ms
const POLLING_INTERVAL = 3000;
const PAGE_SIZE = 50;

export const useJobStore = create<JobState>((set, get) => {
  let pollingInterval: NodeJS.Timeout | null = null;

  return {
    jobs: [],
    isLoading: false,
    isDrawerOpen: false,
    activeTab: 'training',
    hasMore: true,
    skip: 0,

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
