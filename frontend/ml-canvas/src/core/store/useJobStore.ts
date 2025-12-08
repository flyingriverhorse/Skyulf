import { create } from 'zustand';
import { jobsApi, JobInfo, RunPipelineRequest } from '../api/jobs';

interface JobState {
  jobs: JobInfo[];
  isLoading: boolean;
  isDrawerOpen: boolean;
  activeTab: 'training' | 'tuning';
  
  // Actions
  fetchJobs: () => Promise<void>;
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

export const useJobStore = create<JobState>((set, get) => {
  let pollingInterval: NodeJS.Timeout | null = null;

  return {
    jobs: [],
    isLoading: false,
    isDrawerOpen: false,
    activeTab: 'training',

    fetchJobs: async () => {
      set({ isLoading: true });
      try {
        const jobs = await jobsApi.listJobs();
        set({ jobs, isLoading: false });
      } catch (error) {
        console.error('Failed to fetch jobs:', error);
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
      
      pollingInterval = setInterval(async () => {
        // Silent fetch (no loading spinner)
        try {
          const jobs = await jobsApi.listJobs();
          set({ jobs });
          
          // Stop polling if no active jobs and drawer is closed
          const hasActive = jobs.some(j => j.status === 'running' || j.status === 'queued');
          if (!hasActive && !get().isDrawerOpen) {
            get().stopPolling();
          }
        } catch (error) {
          console.error('Polling error:', error);
        }
      }, POLLING_INTERVAL);
    },

    stopPolling: () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
      }
    }
  };
});
