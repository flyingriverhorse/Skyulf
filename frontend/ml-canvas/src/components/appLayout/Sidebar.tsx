import React from 'react';
import { Link } from 'react-router-dom';
import {
  LayoutDashboard, Database, Rocket, GitBranch, Moon, Sun, Archive,
  BarChart2, Activity, TrendingUp, Bug, Timer, ScrollText, X,
} from 'lucide-react';
import logoUrl from '../../../../../static/img/logo.png';

interface SidebarProps {
  pathname: string;
  isMobile: boolean;
  isCollapsed: boolean;
  isDrawerOpen: boolean;
  asideRef: React.RefObject<HTMLElement>;
  closeDrawer: () => void;
  driftAlert: boolean;
  errorAlert: boolean;
  isDarkMode: boolean;
  toggleTheme: () => void;
}

/** Render the persistent desktop rail or the mobile navigation drawer. */
export function Sidebar({
  pathname, isMobile, isCollapsed, isDrawerOpen, asideRef, closeDrawer,
  driftAlert, errorAlert, isDarkMode, toggleTheme,
}: SidebarProps) {
  return (
    <aside
      ref={asideRef}
      id="app-sidebar"
      role={isMobile ? 'dialog' : undefined}
      aria-modal={isMobile ? true : undefined}
      aria-label={isMobile ? 'Navigation menu' : undefined}
      className={`${isMobile
          ? `fixed inset-y-0 left-0 z-40 w-64 transform transition-transform duration-200 ${isDrawerOpen ? 'translate-x-0' : '-translate-x-full'}`
          : `${isCollapsed ? 'w-16' : 'w-64'} shrink-0 transition-all duration-200`
        } bg-white dark:bg-slate-900 text-foreground border-r flex flex-col`}
    >
      <SidebarBrand isCollapsed={isCollapsed} isMobile={isMobile} closeDrawer={closeDrawer} />
      <NavigationLinks pathname={pathname} isCollapsed={isCollapsed} driftAlert={driftAlert} errorAlert={errorAlert} />
      <ThemeControl isCollapsed={isCollapsed} isDarkMode={isDarkMode} toggleTheme={toggleTheme} />
    </aside>
  );
}

function SidebarBrand({
  isCollapsed, isMobile, closeDrawer,
}: Pick<SidebarProps, 'isCollapsed' | 'isMobile' | 'closeDrawer'>) {
  return (
    <div className={`h-14 shrink-0 ${isCollapsed ? 'px-4' : 'px-6'} border-b flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'} ${isMobile ? 'justify-between' : ''}`}>
      <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
        <img src={logoUrl} alt="Skyulf logo" width={32} height={32} className="w-8 h-8 object-contain shrink-0" />
        {!isCollapsed && (
          <h1 className="text-xl font-bold tracking-tight whitespace-nowrap text-foreground">
            Skyulf ML
          </h1>
        )}
      </div>
      {isMobile && (
        <button
          onClick={closeDrawer}
          className="p-2.5 text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors focus-ring"
          aria-label="Close navigation menu"
        >
          <X size={20} />
        </button>
      )}
    </div>
  );
}

function NavigationLinks({
  pathname, isCollapsed, driftAlert, errorAlert,
}: Pick<SidebarProps, 'pathname' | 'isCollapsed' | 'driftAlert' | 'errorAlert'>) {
  const isActive = (path: string) => pathname === path;
  return (
    <nav className="flex-1 overflow-y-auto p-2 space-y-2">
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
  );
}

function ThemeControl({
  isCollapsed, isDarkMode, toggleTheme,
}: Pick<SidebarProps, 'isCollapsed' | 'isDarkMode' | 'toggleTheme'>) {
  return (
    <div className={`${isCollapsed ? 'p-2' : 'p-4'} border-t space-y-4`}>
      <button
        onClick={toggleTheme}
        className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'} w-full ${isCollapsed ? 'px-2' : 'px-4'} py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors focus-ring`}
        title={isDarkMode ? 'Light Mode' : 'Dark Mode'}
        aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
        {!isCollapsed && (isDarkMode ? 'Light Mode' : 'Dark Mode')}
      </button>
    </div>
  );
}

const NavLink = ({ to, children, active, icon, collapsed, badge }: {
  to: string;
  children: React.ReactNode;
  active: boolean;
  icon?: React.ReactNode;
  collapsed?: boolean;
  badge?: boolean;
}) => (
  <Link
    to={to}
    aria-current={active ? 'page' : undefined}
    className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3'} ${collapsed ? 'px-2' : 'px-4'} py-3 rounded-md text-sm font-medium transition-colors ${active
        ? 'bg-brand-action text-brand-action-foreground shadow-sm'
        : 'text-muted-foreground hover:text-foreground hover:bg-accent'
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
