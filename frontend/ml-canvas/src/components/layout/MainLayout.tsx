import React from 'react';
import { Sidebar } from './Sidebar';
import { FlowCanvas } from '../canvas/FlowCanvas';
import { PropertiesPanel } from './PropertiesPanel';
import { Toolbar } from './Toolbar';
import { ResultsPanel } from './ResultsPanel';
import { Navbar } from './Navbar';
import { RestoreSessionBanner } from './RestoreSessionBanner';
import { JobsDrawer } from '../panels/JobsDrawer';
import { useViewStore } from '../../core/store/useViewStore';
import { useCanvasAutoSave } from '../../core/hooks/useCanvasAutoSave';
import { ExperimentsPage } from '../pages/ExperimentsPage';
import { InferencePage } from '../pages/InferencePage';

export const MainLayout: React.FC = () => {
  const { activeView, isPropertiesPanelExpanded } = useViewStore();
  // Throttled localStorage autosave of the canvas graph; restore prompt
  // is rendered below in the canvas branch.
  useCanvasAutoSave();

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
    </div>
  );
};
