import React from 'react';
import { Sidebar } from './Sidebar';
import { FlowCanvas } from '../canvas/FlowCanvas';
import { PropertiesPanel } from './PropertiesPanel';
import { Toolbar } from './Toolbar';
import { ResultsPanel } from './ResultsPanel';

export const MainLayout: React.FC = () => {
  return (
    <div className="flex h-screen w-full bg-background overflow-hidden">
      <Sidebar />
      <main className="flex-1 h-full relative flex flex-col">
        <Toolbar />
        <div className="flex-1 relative">
          <FlowCanvas />
          <ResultsPanel />
        </div>
      </main>
      <PropertiesPanel />
    </div>
  );
};
