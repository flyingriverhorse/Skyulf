import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DataSources } from './pages/DataSources';
import { CanvasPage } from './pages/CanvasPage';
import { JobsPage } from './pages/Jobs';
import { PageSkeleton } from './components/shared';

// Heavy pages are code-split. EDA + DataDrift drag in Plotly (~9.7 MB raw).
// ModelRegistry + Deployments are rarely first-paint routes.
const EDAPage = lazy(() => import('./pages/EDAPage').then(m => ({ default: m.EDAPage })));
const DataDriftPage = lazy(() => import('./pages/DataDriftPage').then(m => ({ default: m.DataDriftPage })));
const ModelRegistry = lazy(() => import('./pages/ModelRegistry').then(m => ({ default: m.ModelRegistry })));
const DeploymentsPage = lazy(() => import('./components/pages/DeploymentsPage').then(m => ({ default: m.DeploymentsPage })));

const RouteFallback = () => <PageSkeleton />;

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Main App Layout (with Sidebar) */}
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="jobs" element={<JobsPage />} />
          <Route path="data" element={<DataSources />} />
          <Route
            path="eda"
            element={
              <Suspense fallback={<RouteFallback />}>
                <EDAPage />
              </Suspense>
            }
          />
          <Route
            path="drift"
            element={
              <Suspense fallback={<RouteFallback />}>
                <DataDriftPage />
              </Suspense>
            }
          />
          <Route path="canvas" element={<CanvasPage />} />
          <Route
            path="registry"
            element={
              <Suspense fallback={<RouteFallback />}>
                <ModelRegistry />
              </Suspense>
            }
          />
          <Route
            path="deployments"
            element={
              <Suspense fallback={<RouteFallback />}>
                <DeploymentsPage />
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
