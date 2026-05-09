import React, { useState, useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, Database, Rocket, GitBranch, Moon, Sun, Archive, BarChart2, Activity, TrendingUp, Bug, Timer, ScrollText } from 'lucide-react';
import { monitoringApi } from '../core/api/monitoring';

export const Layout: React.FC = () => {
  const location = useLocation();
  const [isDarkMode, setIsDarkMode] = useState<boolean>(() => {
    if (typeof document === 'undefined') return false;
    return document.documentElement.classList.contains('dark');
  });
  const [driftAlert, setDriftAlert] = useState(false);
  const [errorAlert, setErrorAlert] = useState(false);

  useEffect(() => {
    // Inline script in index.html already applied the right class before mount;
    // this effect only keeps state in sync if something else mutated the class.
    setIsDarkMode(document.documentElement.classList.contains('dark'));
  }, []);

  useEffect(() => {
    monitoringApi.getDriftStatus()
      .then(s => setDriftAlert(s.has_drift))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const check = () =>
      monitoringApi.getUnresolvedCount()
        .then(n => setErrorAlert(n > 0))
        .catch(() => {});
    check();
    const id = setInterval(check, 60_000);
    return () => clearInterval(id);
  }, []);

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    try {
      localStorage.setItem('skyulf-theme', newMode ? 'dark' : 'light');
    } catch { /* ignore quota / privacy errors */ }
  };

  const isActive = (path: string) => location.pathname === path;
  const isCollapsed = location.pathname === '/canvas' || location.pathname === '/eda';

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-200">
      {/* Sidebar */}
      <aside className={`${isCollapsed ? 'w-16' : 'w-64'} bg-slate-900 dark:bg-slate-950 text-white flex flex-col shrink-0 transition-all duration-200`}>
        <div className={`${isCollapsed ? 'p-4' : 'p-6'} border-b border-slate-800 dark:border-slate-900 flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center font-bold text-lg shrink-0 shadow-lg shadow-blue-900/20">S</div>
          {!isCollapsed && (
            <h1 className="text-xl font-bold tracking-tight whitespace-nowrap bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              Skyulf ML
            </h1>
          )}
        </div>
        
        <nav className="flex-1 p-2 space-y-2">
          <NavLink to="/" active={isActive('/')} icon={<LayoutDashboard size={20} />} collapsed={isCollapsed}>
            Dashboard
          </NavLink>
          <NavLink to="/jobs" active={isActive('/jobs')} icon={<Activity size={20} />} collapsed={isCollapsed}>
            Jobs
          </NavLink>
          <NavLink to="/eda" active={isActive('/eda')} icon={<BarChart2 size={20} />} collapsed={isCollapsed}>
            EDA
          </NavLink>
          <NavLink to="/drift" active={isActive('/drift')} icon={<TrendingUp size={20} />} collapsed={isCollapsed} badge={driftAlert}>
            Data Drift
          </NavLink>
          <NavLink to="/canvas" active={isActive('/canvas')} icon={<GitBranch size={20} />} collapsed={isCollapsed}>
            ML Canvas
          </NavLink>
          <NavLink to="/data" active={isActive('/data')} icon={<Database size={20} />} collapsed={isCollapsed}>
            Data Sources
          </NavLink>
          <NavLink to="/registry" active={isActive('/registry')} icon={<Archive size={20} />} collapsed={isCollapsed}>
            Model Registry
          </NavLink>
          <NavLink to="/deployments" active={isActive('/deployments')} icon={<Rocket size={20} />} collapsed={isCollapsed}>
            Deployments
          </NavLink>
          <NavLink to="/errors" active={isActive('/errors')} icon={<Bug size={20} />} collapsed={isCollapsed} badge={errorAlert}>
            Error Log
          </NavLink>
          <NavLink to="/slow-nodes" active={isActive('/slow-nodes')} icon={<Timer size={20} />} collapsed={isCollapsed}>
            Slow Nodes
          </NavLink>
          <NavLink to="/audit" active={isActive('/audit')} icon={<ScrollText size={20} />} collapsed={isCollapsed}>
            Audit Log
          </NavLink>
        </nav>

        <div className={`${isCollapsed ? 'p-2' : 'p-4'} border-t border-slate-800 space-y-4`}>
          <button 
            onClick={toggleTheme}
            className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'} w-full ${isCollapsed ? 'px-2' : 'px-4'} py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors`}
            title={isDarkMode ? 'Light Mode' : 'Dark Mode'}
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            {!isCollapsed && (isDarkMode ? 'Light Mode' : 'Dark Mode')}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};

const NavLink = ({ to, children, active, icon, collapsed, badge }: { to: string, children: React.ReactNode, active: boolean, icon?: React.ReactNode, collapsed?: boolean, badge?: boolean }) => (
  <Link
    to={to}
    aria-current={active ? 'page' : undefined}
    className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3'} ${collapsed ? 'px-2' : 'px-4'} py-3 rounded-md text-sm font-medium transition-colors ${
      active 
        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-sm' 
        : 'text-slate-400 hover:text-white hover:bg-slate-800'
    }`}
    title={collapsed ? (children as string) : undefined}
    aria-label={collapsed ? (children as string) : undefined}
  >
    <span className="relative">
      {icon}
      {badge && (
        <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-slate-900 dark:border-slate-950" />
      )}
    </span>
    {!collapsed && children}
  </Link>
);
