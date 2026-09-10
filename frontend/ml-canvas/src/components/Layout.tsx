import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { useViewport } from '../core/hooks/useViewport';
import { LayoutHeader } from './appLayout/LayoutHeader';
import { Sidebar } from './appLayout/Sidebar';
import { useLayoutTheme } from './appLayout/useLayoutTheme';
import { useMonitoringAlerts } from './appLayout/useMonitoringAlerts';
import { useNavigationDrawer } from './appLayout/useNavigationDrawer';

export const Layout: React.FC = () => {
  const location = useLocation();
  // Below 768 px the fixed 256/64 px sidebar leaves too little room for
  // page content (FND-001), so it becomes an off-canvas drawer instead of
  // a persistent column at these widths.
  const { isMobile } = useViewport();
  const { isDarkMode, toggleTheme } = useLayoutTheme();
  const { isDrawerOpen, setIsDrawerOpen, menuButtonRef, asideRef } = useNavigationDrawer(location.pathname, isMobile);
  const { driftAlert, errorAlert } = useMonitoringAlerts(location.pathname);

  // The icon-only rail only makes sense as a desktop space-saving choice on
  // Canvas/EDA; at mobile widths the sidebar is an off-canvas drawer instead
  // (see isMobile below), so it always renders at full width when opened.
  const isCollapsed = !isMobile && (location.pathname === '/canvas' || location.pathname === '/eda');

  return (
    <div className="flex h-screen bg-background text-foreground">
      {isMobile && isDrawerOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30"
          aria-hidden="true"
          onClick={() => setIsDrawerOpen(false)}
        />
      )}
      {/* Sidebar: a persistent column at 768 px+, an off-canvas drawer below it (FND-001). */}
      <Sidebar
        pathname={location.pathname}
        isMobile={isMobile}
        isCollapsed={isCollapsed}
        isDrawerOpen={isDrawerOpen}
        asideRef={asideRef}
        closeDrawer={() => setIsDrawerOpen(false)}
        driftAlert={driftAlert}
        errorAlert={errorAlert}
        isDarkMode={isDarkMode}
        toggleTheme={toggleTheme}
      />

      {/* Main Content */}
      <main className="flex-1 overflow-auto min-w-0 flex flex-col">
        <LayoutHeader
          pathname={location.pathname}
          isMobile={isMobile}
          isDrawerOpen={isDrawerOpen}
          menuButtonRef={menuButtonRef}
          openDrawer={() => setIsDrawerOpen(true)}
        />
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
