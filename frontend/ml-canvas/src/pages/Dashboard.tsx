import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient } from '../core/api/client';
import { jobsApi } from '../core/api/jobs';

interface SystemStats {
  total_jobs: number;
  active_deployments: number;
  data_sources: number;
  training_jobs: number;
  tuning_jobs: number;
}

interface TrainingJobSummary {
  id: string;
  status: string;
  model_type: string;
  created_at: string;
  metrics?: Record<string, any>;
}

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [jobs, setJobs] = useState<TrainingJobSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, recentJobs] = await Promise.all([
          apiClient.get<SystemStats>('/pipeline/stats'),
          jobsApi.getJobs(5, 0, 'training')
        ]);
        setStats(statsRes.data);
        setJobs(
          recentJobs.map(job => ({
            id: job.job_id,
            status: job.status,
            model_type: job.model_type || 'Unknown',
            created_at: job.start_time || job.created_at || new Date().toISOString(),
            metrics: job.metrics || job.result?.metrics
          }))
        );
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Dashboard</h1>
        <Link 
          to="/canvas" 
          className="text-white px-4 py-2 rounded-md shadow-sm transition-all hover:opacity-90"
          style={{ background: 'var(--main-gradient)' }}
        >
          New Experiment
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard 
          title="Total Jobs" 
          value={stats?.total_jobs ?? '-'} 
          loading={loading}
          subtext={`${stats?.training_jobs ?? 0} Training, ${stats?.tuning_jobs ?? 0} Tuning`}
        />
        <StatCard 
          title="Active Deployments" 
          value={stats?.active_deployments ?? '-'} 
          loading={loading}
          color="text-green-600 dark:text-green-400"
        />
        <StatCard 
          title="Data Sources" 
          value={stats?.data_sources ?? '-'} 
          loading={loading}
        />
      </div>

      {/* Recent Activity */}
      <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">Recent Activity</h2>
        {loading ? (
           <div className="text-slate-500 dark:text-slate-400 text-sm">Loading...</div>
        ) : jobs.length === 0 ? (
          <div className="text-slate-500 dark:text-slate-400 text-sm">
            No recent jobs found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700">
              <thead>
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Job ID</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Model</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Created At</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-slate-800 divide-y divide-slate-200 dark:divide-slate-700">
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-slate-100" title={job.id}>
                      {job.id.substring(0, 8)}...
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">{job.model_type}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                        job.status === 'succeeded' ? 'bg-green-100 text-green-800' : 
                        job.status === 'failed' ? 'bg-red-100 text-red-800' : 
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

const StatCard = ({ title, value, loading, subtext, color = "text-slate-900 dark:text-slate-100" }: any) => (
  <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
    <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">{title}</h3>
    <div className={`mt-2 text-3xl font-bold ${color}`}>
      {loading ? '...' : value}
    </div>
    {subtext && <div className="mt-1 text-xs text-slate-400 dark:text-slate-500">{subtext}</div>}
  </div>
);
