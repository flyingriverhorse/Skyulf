import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts';
import { 
  Activity, Database, Play, Server, 
  Plus, ExternalLink,
  CheckCircle, XCircle, Clock, Settings
} from 'lucide-react';
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
  metrics?: Record<string, unknown>;
}

const COLORS = ['#10B981', '#EF4444', '#F59E0B', '#3B82F6'];

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null;
};

export const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [jobs, setJobs] = useState<TrainingJobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [chartData, setChartData] = useState<{
    statusDist: { name: string; value: number }[];
    dailyActivity: { date: string; count: number }[];
  }>({ statusDist: [], dailyActivity: [] });

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch more jobs for charts (50), but we'll only show 5 in the table
        const [statsRes, recentJobs] = await Promise.all([
          apiClient.get<SystemStats>('/pipeline/stats'),
          jobsApi.getJobs(50, 0) 
        ]);
        
        setStats(statsRes.data);

        const formattedJobs = recentJobs.map(job => ({
          id: job.job_id,
          status: job.status,
          model_type: job.model_type || 'Unknown',
          created_at: job.start_time || job.created_at || new Date().toISOString(),
          metrics: (() => {
            const metricsCandidate = job.metrics ?? (isRecord(job.result) ? (job.result as Record<string, unknown>).metrics : undefined);
            return isRecord(metricsCandidate) ? metricsCandidate : undefined;
          })()
        }));

        setJobs(formattedJobs);

        // Process data for charts
        const statusCounts = formattedJobs.reduce((acc, job) => {
          acc[job.status] = (acc[job.status] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);

        const statusDist = Object.entries(statusCounts).map(([name, value]) => ({ name, value }));

        // Daily activity (last 7 days)
        const last7Days = [...Array(7)].map((_, i) => {
          const d = new Date();
          d.setDate(d.getDate() - i);
          return d.toISOString().split('T')[0];
        }).reverse();

        const dailyCounts = last7Days.map(date => ({
          date: new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          count: formattedJobs.filter(j => j.created_at.startsWith(date)).length
        }));

        setChartData({ statusDist, dailyActivity: dailyCounts });

      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    void fetchData();
  }, []);

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Dashboard</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">Overview of your ML pipeline activities</p>
        </div>
        <div className="flex gap-3">
          <Link 
            to="/canvas" 
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-md shadow-sm transition-all"
          >
            <Plus size={18} />
            New Experiment
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Total Jobs" 
          value={stats?.total_jobs ?? '-'} 
          icon={<Activity className="text-indigo-500" />}
          subtext={`${stats?.training_jobs ?? 0} Training, ${stats?.tuning_jobs ?? 0} Tuning`}
        />
        <StatCard 
          title="Active Deployments" 
          value={stats?.active_deployments ?? '-'} 
          icon={<Server className="text-green-500" />}
          color="text-green-600 dark:text-green-400"
        />
        <StatCard 
          title="Data Sources" 
          value={stats?.data_sources ?? '-'} 
          icon={<Database className="text-blue-500" />}
        />
        <StatCard 
          title="Success Rate" 
          value={jobs.length > 0 ? `${Math.round((jobs.filter(j => j.status === 'succeeded').length / jobs.length) * 100)}%` : '-'} 
          icon={<CheckCircle className="text-emerald-500" />}
          subtext="Last 50 jobs"
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Activity Chart */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
          <h3 className="text-lg font-semibold mb-6 text-slate-900 dark:text-slate-100">Weekly Activity</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData.dailyActivity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis 
                  dataKey="date" 
                  stroke="#64748b" 
                  fontSize={12} 
                  tickLine={false} 
                  axisLine={false} 
                />
                <YAxis 
                  stroke="#64748b" 
                  fontSize={12} 
                  tickLine={false} 
                  axisLine={false} 
                  allowDecimals={false}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                  cursor={{ fill: '#f1f5f9' }}
                />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={30} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Status Distribution */}
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
          <h3 className="text-lg font-semibold mb-6 text-slate-900 dark:text-slate-100">Job Status Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData.statusDist}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {chartData.statusDist.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={
                      entry.name === 'succeeded' ? COLORS[0] :
                      entry.name === 'failed' ? COLORS[1] :
                      entry.name === 'running' ? COLORS[3] : COLORS[2]
                    } />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }} />
                <Legend verticalAlign="bottom" height={36} iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activity Table */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden">
          <div className="p-6 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Recent Jobs</h2>
            <Link to="/jobs" className="text-sm text-indigo-600 hover:text-indigo-700 font-medium flex items-center gap-1">
              View All <ExternalLink size={14} />
            </Link>
          </div>
          
          {loading ? (
             <div className="p-8 text-center text-slate-500">Loading...</div>
          ) : jobs.length === 0 ? (
            <div className="p-8 text-center text-slate-500">No recent jobs found.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200 dark:divide-slate-700">
                <thead className="bg-slate-50 dark:bg-slate-900/50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Job ID</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Model</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">Time</th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-slate-800 divide-y divide-slate-200 dark:divide-slate-700">
                  {jobs.slice(0, 5).map((job) => (
                    <tr key={job.id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <StatusBadge status={job.status} />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-slate-100">
                        <span className="font-mono text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded">
                          {job.id.substring(0, 8)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        {job.model_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500 dark:text-slate-400">
                        {new Date(job.created_at).toLocaleDateString()} <span className="text-xs opacity-70">{new Date(job.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Quick Actions */}
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm p-6">
          <h2 className="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">Quick Actions</h2>
          <div className="space-y-3">
            <QuickActionButton 
              to="/data" 
              icon={<Database size={18} className="text-blue-500" />} 
              title="Manage Data Sources" 
              desc="Upload or connect new data"
            />
            <QuickActionButton 
              to="/canvas" 
              icon={<Play size={18} className="text-indigo-500" />} 
              title="Run Pipeline" 
              desc="Create and execute ML workflows"
            />
            <QuickActionButton 
              to="/deployments" 
              icon={<Server size={18} className="text-green-500" />} 
              title="View Deployments" 
              desc="Monitor active model endpoints"
            />
            <QuickActionButton 
              to="/settings" 
              icon={<Settings size={18} className="text-slate-500" />} 
              title="Settings" 
              desc="Configure system preferences"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ title, value, icon, subtext, color = "text-slate-900 dark:text-slate-100" }: { title: string; value: string | number; icon?: React.ReactNode; subtext?: string; color?: string }) => (
  <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm hover:shadow-md transition-shadow">
    <div className="flex justify-between items-start">
      <div>
        <h3 className="text-sm font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">{title}</h3>
        <div className={`mt-2 text-3xl font-bold ${color}`}>
          {value}
        </div>
      </div>
      {icon && <div className="p-3 bg-slate-50 dark:bg-slate-700/50 rounded-full">{icon}</div>}
    </div>
    {subtext && <div className="mt-2 text-xs text-slate-400 dark:text-slate-500 flex items-center gap-1">
      <Activity size={12} /> {subtext}
    </div>}
  </div>
);

const StatusBadge = ({ status }: { status: string }) => {
  const styles = {
    succeeded: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
    failed: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    running: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
    pending: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
  };
  
  const icons = {
    succeeded: <CheckCircle size={14} />,
    failed: <XCircle size={14} />,
    running: <Activity size={14} className="animate-pulse" />,
    pending: <Clock size={14} />,
  };

  const key = status.toLowerCase() as keyof typeof styles;
  const style = styles[key] || 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-400';
  const icon = icons[key] || <Clock size={14} />;

  return (
    <span className={`px-2.5 py-1 inline-flex items-center gap-1.5 text-xs font-medium rounded-full ${style}`}>
      {icon}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

const QuickActionButton = ({ to, icon, title, desc }: { to: string; icon: React.ReactNode; title: string; desc: string }) => (
  <Link to={to} className="flex items-center gap-4 p-3 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors border border-transparent hover:border-slate-200 dark:hover:border-slate-600 group">
    <div className="bg-white dark:bg-slate-700 p-2 rounded-md shadow-sm group-hover:scale-110 transition-transform">
      {icon}
    </div>
    <div>
      <div className="font-medium text-slate-900 dark:text-slate-100 text-sm">{title}</div>
      <div className="text-xs text-slate-500 dark:text-slate-400">{desc}</div>
    </div>
    <ExternalLink size={14} className="ml-auto text-slate-300 opacity-0 group-hover:opacity-100 transition-opacity" />
  </Link>
);
