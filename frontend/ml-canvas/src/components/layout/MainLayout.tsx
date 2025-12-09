import React from 'react';
import { Sidebar } from './Sidebar';
import { FlowCanvas } from '../canvas/FlowCanvas';
import { PropertiesPanel } from './PropertiesPanel';
import { Toolbar } from './Toolbar';
import { ResultsPanel } from './ResultsPanel';
import { Navbar } from './Navbar';
import { JobsDrawer } from '../panels/JobsDrawer';
import { useViewStore } from '../../core/store/useViewStore';
import { ExperimentsPage } from '../pages/ExperimentsPage';
import { InferencePage } from '../pages/InferencePage';

export const MainLayout: React.FC = () => {
  const { activeView } = useViewStore();

  return (
    <div className="flex h-full w-full bg-background overflow-hidden flex-col">
      <JobsDrawer />
      <Navbar />
      
      {activeView === 'canvas' ? (
        <div className="flex flex-1 overflow-hidden relative">
          <Sidebar />
          <main className="flex-1 h-full relative flex flex-col transition-all duration-300 ease-in-out">
            <Toolbar />
            <div className="flex-1 relative">
              <FlowCanvas />
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
