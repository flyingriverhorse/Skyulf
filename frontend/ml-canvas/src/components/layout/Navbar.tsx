import React, { useEffect, useState } from 'react';
import { Moon, Sun, Layout, BarChart2, GitBranch, Rocket } from 'lucide-react';
import { useViewStore } from '../../core/store/useViewStore';

export const Navbar: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const { activeView, setView } = useViewStore();

  useEffect(() => {
    // Check initial preference
    if (document.documentElement.classList.contains('dark')) {
      setIsDarkMode(true);
    } else {
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

  return (
    <div className="h-14 border-b bg-card px-4 flex items-center justify-between shrink-0">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-primary/10 rounded-md">
            <Layout className="w-5 h-5 text-primary" />
          </div>
          <h1 className="font-bold text-lg tracking-tight bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
            Skyulf ML Canvas
          </h1>
        </div>

        {/* Navigation */}
        <div className="flex items-center gap-1 bg-secondary/50 p-1 rounded-lg">
          <button
            onClick={() => setView('canvas')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              activeView === 'canvas' 
                ? 'bg-background shadow-sm text-foreground' 
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
            }`}
          >
            <GitBranch className="w-4 h-4" />
            Canvas
          </button>
          <button
            onClick={() => setView('experiments')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              activeView === 'experiments' 
                ? 'bg-background shadow-sm text-foreground' 
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
            }`}
          >
            <BarChart2 className="w-4 h-4" />
            Experiments
          </button>
          <button
            onClick={() => setView('inference')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
              activeView === 'inference' 
                ? 'bg-background shadow-sm text-foreground' 
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50'
            }`}
          >
            <Rocket className="w-4 h-4" />
            Inference
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <a 
          href="/ml-workflow-v1" 
          className="text-sm text-muted-foreground hover:text-primary transition-colors mr-2"
          target="_blank"
          rel="noopener noreferrer"
        >
          Switch to Legacy V1
        </a>
        <button 
          onClick={toggleTheme} 
          className="p-2 hover:bg-accent rounded-md transition-colors text-muted-foreground hover:text-foreground"
          title="Toggle Theme"
        >
          {isDarkMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
        </button>
      </div>
    </div>
  );
};
