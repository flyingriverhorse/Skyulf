import React, { useState, useEffect, useRef } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, Database, Rocket, GitBranch, Moon, Sun, Archive, BarChart2, Activity, TrendingUp, Bug, Timer, ScrollText, Menu, X } from 'lucide-react';
import { monitoringApi } from '../core/api/monitoring';
import { useViewport } from '../core/hooks/useViewport';

export const Layout: React.FC = () => {
  const location = useLocation();
  const [isDarkMode, setIsDarkMode] = useState<boolean>(() => {
    if (typeof document === 'undefined') return false;
    return document.documentElement.classList.contains('dark');
  });
  const [driftAlert, setDriftAlert] = useState(false);
  const [errorAlert, setErrorAlert] = useState(false);
  // Below 768 px the fixed 256/64 px sidebar leaves too little room for
  // page content (FND-001), so it becomes an off-canvas drawer instead of
  // a persistent column at these widths.
  const { isMobile } = useViewport();
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const menuButtonRef = useRef<HTMLButtonElement | null>(null);
  const asideRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    // Inline script in index.html already applied the right class before mount;
    // this effect only keeps state in sync if something else mutated the class.
    setIsDarkMode(document.documentElement.classList.contains('dark'));
  }, []);

  // Close the drawer on route change (a nav click just navigated away, so
  // there's nothing left to show it for) and whenever the viewport grows
  // back to a size where the sidebar is a persistent column again.
  useEffect(() => {
    setIsDrawerOpen(false);
  }, [location.pathname]);
  useEffect(() => {
    if (!isMobile) setIsDrawerOpen(false);
  }, [isMobile]);

  // Escape closes the drawer and returns focus to the button that opened
  // it; focus moves into the drawer's first link on open so keyboard and
  // screen-reader users land somewhere useful. This is a minimal, self-
  // contained version of the containment behavior FND-002 will later make
  // a shared contract across every shell overlay.
  useEffect(() => {
    if (!isDrawerOpen) return;
    const raf = window.requestAnimationFrame(() => {
      asideRef.current?.querySelector<HTMLAnchorElement>('a[href]')?.focus();
    });
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsDrawerOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    const openerButton = menuButtonRef.current;
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener('keydown', onKeyDown);
      openerButton?.focus();
    };
  }, [isDrawerOpen]);

  useEffect(() => {
    monitoringApi.getDriftStatus()
      .then(s => setDriftAlert(s.has_drift))
      .catch(() => {});
  }, []);

  useEffect(() => {
    // Drives the red dot on the "Errors" nav link. We don't need a live
    // counter — the dot just signals "there's something to look at", so
    // a 5-minute poll is plenty. Skip entirely when the user is already
    // on /errors (they're seeing the live list) or when the tab is
    // hidden (saves a request per inactive tab per cycle).
    const check = () =>
      monitoringApi.getUnresolvedCount()
        .then(n => setErrorAlert(n > 0))
        .catch(() => {});
    const tick = () => {
      if (document.hidden) return;
      if (location.pathname === '/errors') return;
      check();
    };
    tick();
    const id = setInterval(tick, 300_000);
    return () => clearInterval(id);
  }, [location.pathname]);

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
  // The icon-only rail only makes sense as a desktop space-saving choice on
  // Canvas/EDA; at mobile widths the sidebar is an off-canvas drawer instead
  // (see isMobile below), so it always renders at full width when opened.
  const isCollapsed = !isMobile && (location.pathname === '/canvas' || location.pathname === '/eda');

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-200">
      {isMobile && isDrawerOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30"
          aria-hidden="true"
          onClick={() => setIsDrawerOpen(false)}
        />
      )}
      {/* Sidebar: a persistent column at 768 px+, an off-canvas drawer below it (FND-001). */}
      <aside
        ref={asideRef}
        id="app-sidebar"
        role={isMobile ? 'dialog' : undefined}
        aria-modal={isMobile ? true : undefined}
        aria-label={isMobile ? 'Navigation menu' : undefined}
        className={`${
          isMobile
            ? `fixed inset-y-0 left-0 z-40 w-64 transform transition-transform duration-200 ${isDrawerOpen ? 'translate-x-0' : '-translate-x-full'}`
            : `${isCollapsed ? 'w-16' : 'w-64'} shrink-0 transition-all duration-200`
        } bg-slate-900 dark:bg-slate-950 text-white flex flex-col`}
      >
        <div className={`${isCollapsed ? 'p-4' : 'p-6'} border-b border-slate-800 dark:border-slate-900 flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'} ${isMobile ? 'justify-between' : ''}`}>
          <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center font-bold text-lg shrink-0 shadow-lg shadow-blue-900/20">S</div>
            {!isCollapsed && (
              <h1 className="text-xl font-bold tracking-tight whitespace-nowrap bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                Skyulf ML
              </h1>
            )}
          </div>
          {isMobile && (
            <button
              onClick={() => setIsDrawerOpen(false)}
              className="p-2.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
              aria-label="Close navigation menu"
            >
              <X size={20} />
            </button>
          )}
        </div>

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
      <main className="flex-1 overflow-auto min-w-0 flex flex-col">
        {isMobile && (
          <div className="h-14 shrink-0 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex items-center px-2 sticky top-0 z-20">
            <button
              ref={menuButtonRef}
              onClick={() => setIsDrawerOpen(true)}
              className="p-3 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-md transition-colors"
              aria-label="Open navigation menu"
              aria-expanded={isDrawerOpen}
              aria-controls="app-sidebar"
            >
              <Menu size={20} />
            </button>
            <span className="ml-2 font-semibold text-slate-900 dark:text-slate-100">Skyulf ML</span>
          </div>
        )}
        {/* min-h-0 lets this flex item shrink below its content's natural
         * height instead of forcing `<main>` to grow past the mobile bar
         * above it -- without it, Canvas's `h-full` root would compute
         * against `<main>`'s full height and overflow past the sticky bar. */}
        <div className="flex-1 min-h-0">
          <Outlet />
        </div>
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
