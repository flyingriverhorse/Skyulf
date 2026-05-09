import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { DataSources } from './pages/DataSources';
import { CanvasPage } from './pages/CanvasPage';
import { JobsPage } from './pages/Jobs';
import { ErrorLogPage } from './pages/ErrorLogPage';
import { PageSkeleton, ErrorBoundary } from './components/shared';

// Heavy pages are code-split. EDA + DataDrift drag in Plotly (~9.7 MB raw).
// ModelRegistry + Deployments are rarely first-paint routes.
const EDAPage = lazy(() => import('./pages/EDAPage').then(m => ({ default: m.EDAPage })));
const DataDriftPage = lazy(() => import('./pages/DataDriftPage').then(m => ({ default: m.DataDriftPage })));
const ModelRegistry = lazy(() => import('./pages/ModelRegistry').then(m => ({ default: m.ModelRegistry })));
const DeploymentsPage = lazy(() => import('./components/pages/DeploymentsPage').then(m => ({ default: m.DeploymentsPage })));
const SlowNodesPage = lazy(() => import('./pages/SlowNodesPage').then(m => ({ default: m.SlowNodesPage })));
const AuditLogPage = lazy(() => import('./pages/AuditLogPage').then(m => ({ default: m.AuditLogPage })));

const RouteFallback = () => <PageSkeleton />;

// Wrap each lazy route in its own boundary so a crash in EDA doesn't
// blank the entire app — the user can navigate elsewhere or hit "Try
// again" without reloading.
const LazyRoute = ({ children }: { children: React.ReactNode }) => (
  <ErrorBoundary>
    <Suspense fallback={<RouteFallback />}>{children}</Suspense>
  </ErrorBoundary>
);

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
              <LazyRoute>
                <EDAPage />
              </LazyRoute>
            }
          />
          <Route
            path="drift"
            element={
              <LazyRoute>
                <DataDriftPage />
              </LazyRoute>
            }
          />
          <Route path="canvas" element={<CanvasPage />} />
          <Route
            path="registry"
            element={
              <LazyRoute>
                <ModelRegistry />
              </LazyRoute>
            }
          />
          <Route
            path="deployments"
            element={
              <LazyRoute>
                <DeploymentsPage />
              </LazyRoute>
            }
          />
          <Route path="errors" element={<ErrorLogPage />} />
          <Route
            path="slow-nodes"
            element={
              <LazyRoute>
                <SlowNodesPage />
              </LazyRoute>
            }
          />
          <Route
            path="audit"
            element={
              <LazyRoute>
                <AuditLogPage />
              </LazyRoute>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
