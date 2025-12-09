import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClientV2 } from '../core/api/client';

interface SystemStats {
  total_jobs: number;
  active_deployments: number;
  data_sources: number;
  training_jobs: number;
  tuning_jobs: number;
}

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes] = await Promise.all([
          apiClientV2.get<SystemStats>('/pipeline/stats')
        ]);
        setStats(statsRes.data);
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
          className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
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

      {/* Recent Activity Placeholder */}
      <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 shadow-sm">
        <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">Recent Activity</h2>
        <div className="text-slate-500 dark:text-slate-400 text-sm">
          Job history list will appear here...
        </div>
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
