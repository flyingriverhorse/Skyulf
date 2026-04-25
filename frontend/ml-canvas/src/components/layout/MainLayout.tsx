import React from 'react';
import { Sidebar } from './Sidebar';
import { FlowCanvas } from '../canvas/FlowCanvas';
import { PropertiesPanel } from './PropertiesPanel';
import { Toolbar } from './Toolbar';
import { ResultsPanel } from './ResultsPanel';
import { Navbar } from './Navbar';
import { RestoreSessionBanner } from './RestoreSessionBanner';
import { ShortcutsOverlay } from './ShortcutsOverlay';
import { JobsDrawer } from '../panels/JobsDrawer';
import { useViewStore } from '../../core/store/useViewStore';
import { useCanvasAutoSave } from '../../core/hooks/useCanvasAutoSave';
import {
  useKeyboardShortcuts,
  SHOW_SHORTCUTS_EVENT,
} from '../../core/hooks/useKeyboardShortcuts';
import { ExperimentsPage } from '../pages/ExperimentsPage';
import { InferencePage } from '../pages/InferencePage';

export const MainLayout: React.FC = () => {
  const { activeView, isPropertiesPanelExpanded } = useViewStore();
  // Throttled localStorage autosave of the canvas graph; restore prompt
  // is rendered below in the canvas branch.
  useCanvasAutoSave();

  // Global keyboard shortcuts (Ctrl+D duplicate, Ctrl+Enter run,
  // ? cheatsheet, Esc close). Undo/redo stays in Toolbar against the
  // zundo temporal store; fit-view F lives inside FlowCanvas where
  // useReactFlow is available.
  const [showShortcuts, setShowShortcuts] = React.useState(false);
  useKeyboardShortcuts({
    onToggleHelp: () => setShowShortcuts((v) => !v),
    onCloseHelp: () => setShowShortcuts(false),
  });

  // Bridge: lets the Toolbar's Keyboard button (or any future UI) open
  // the shortcuts overlay without prop-drilling state through the layout.
  React.useEffect(() => {
    const open = (): void => setShowShortcuts(true);
    window.addEventListener(SHOW_SHORTCUTS_EVENT, open);
    return () => window.removeEventListener(SHOW_SHORTCUTS_EVENT, open);
  }, []);

  return (
    <div className="flex h-full w-full bg-background overflow-hidden flex-col">
      <JobsDrawer />
      <Navbar />
      
      {activeView === 'canvas' ? (
        <div className="flex flex-1 overflow-hidden relative">
          <Sidebar />
          <main className="flex-1 h-full relative flex flex-col transition-all duration-300 ease-in-out">
            {!isPropertiesPanelExpanded && <Toolbar />}
            <div className="flex-1 relative">
              <FlowCanvas />
              <RestoreSessionBanner />
              <ResultsPanel />
            </div>
          </main>
          <PropertiesPanel />
        </div>
      ) : activeView === 'experiments' ? (
        <ExperimentsPage />
      ) : (
        <InferencePage />
      )}
      <ShortcutsOverlay
        open={showShortcuts}
        onClose={() => setShowShortcuts(false)}
      />
    </div>
  );
};
