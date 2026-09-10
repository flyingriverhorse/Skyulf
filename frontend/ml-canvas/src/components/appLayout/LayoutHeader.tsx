import React from 'react';
import { Menu } from 'lucide-react';
import logoUrl from '../../../../../static/img/logo.png';
import { NotificationCenter } from '../layout/NotificationCenter';

interface LayoutHeaderProps {
  pathname: string;
  isMobile: boolean;
  isDrawerOpen: boolean;
  menuButtonRef: React.RefObject<HTMLButtonElement>;
  openDrawer: () => void;
}

/** Keep mobile navigation access and non-canvas notifications above route content. */
export function LayoutHeader({
  pathname, isMobile, isDrawerOpen, menuButtonRef, openDrawer,
}: LayoutHeaderProps) {
  return (
    <>
      {isMobile && (
        <div className="h-14 shrink-0 border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex items-center px-2 sticky top-0 z-20">
          <button
            ref={menuButtonRef}
            onClick={openDrawer}
            className="p-3 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-md transition-colors"
            aria-label="Open navigation menu"
            aria-expanded={isDrawerOpen}
            aria-controls="app-sidebar"
          >
            <Menu size={20} />
          </button>
          <img src={logoUrl} alt="" width={28} height={28} className="ml-2 h-7 w-7 object-contain" />
          <span className="ml-2 font-semibold text-foreground">Skyulf ML</span>
          {pathname !== '/canvas' && <div className="ml-auto"><NotificationCenter /></div>}
        </div>
      )}
      {!isMobile && pathname !== '/canvas' && (
        <div className="h-14 shrink-0 border-b border-border bg-background flex items-center justify-end px-4 sticky top-0 z-20">
          <NotificationCenter />
        </div>
      )}
    </>
  );
}
