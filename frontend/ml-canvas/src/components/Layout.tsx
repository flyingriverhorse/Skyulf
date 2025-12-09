import React, { useState, useEffect } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, Database, Rocket, GitBranch, Moon, Sun } from 'lucide-react';

export const Layout: React.FC = () => {
  const location = useLocation();
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    // Check if dark mode is already active
    if (document.documentElement.classList.contains('dark')) {
      setIsDarkMode(true);
    } else {
      // Check system preference
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (isDark) {
        document.documentElement.classList.add('dark');
        setIsDarkMode(true);
      }
    }
  }, []);

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-200">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 dark:bg-slate-950 text-white flex flex-col shrink-0 transition-colors duration-200">
        <div className="p-6 border-b border-slate-800 dark:border-slate-900 flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-lg">S</div>
          <h1 className="text-xl font-bold tracking-tight">Skyulf ML</h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          <NavLink to="/" active={isActive('/')} icon={<LayoutDashboard size={20} />}>
            Dashboard
          </NavLink>
          <NavLink to="/canvas" active={isActive('/canvas')} icon={<GitBranch size={20} />}>
            ML Canvas
          </NavLink>
          <NavLink to="/data" active={isActive('/data')} icon={<Database size={20} />}>
            Data Sources
          </NavLink>
          <NavLink to="/deployments" active={isActive('/deployments')} icon={<Rocket size={20} />}>
            Deployments
          </NavLink>
        </nav>

        <div className="p-4 border-t border-slate-800 space-y-4">
          <button 
            onClick={toggleTheme}
            className="flex items-center gap-3 w-full px-4 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
          >
            {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            {isDarkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
          <div className="text-xs text-slate-600 text-center">
            v2.0.0-alpha
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};

const NavLink = ({ to, children, active, icon }: { to: string, children: React.ReactNode, active: boolean, icon?: React.ReactNode }) => (
  <Link
    to={to}
    className={`flex items-center gap-3 px-4 py-3 rounded-md text-sm font-medium transition-colors ${
      active 
        ? 'bg-blue-600 text-white shadow-sm' 
        : 'text-slate-400 hover:text-white hover:bg-slate-800'
    }`}
  >
    {icon}
    {children}
  </Link>
);
