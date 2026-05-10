import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import App from './App.tsx'
import './index.css'
import { initializeRegistry } from './core/registry/init'
import { ErrorBoundary, ConfirmProvider } from './components/shared'
// Dev-only side-effect import: exposes `window.__skyulfTest` for
// Playwright. Stripped from production by Vite's DCE on the
// `import.meta.env.DEV` guard inside the module.
import './test/devHooks'

// React Query Devtools — dev-only. Vite tree-shakes the import in
// production builds because of the `import.meta.env.DEV` guard, so
// there is zero impact on the prod bundle.
const ReactQueryDevtools = import.meta.env.DEV
  ? React.lazy(() =>
      import('@tanstack/react-query-devtools').then((m) => ({
        default: m.ReactQueryDevtools,
      })),
    )
  : null;

// Initialize node registry before rendering
initializeRegistry();

const queryClient = new QueryClient()

const rootElement = document.getElementById('root');
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <ConfirmProvider>
            <App />
            <Toaster richColors closeButton position="top-right" duration={5000} />
            {ReactQueryDevtools && (
              <React.Suspense fallback={null}>
                <ReactQueryDevtools initialIsOpen={false} />
              </React.Suspense>
            )}
          </ConfirmProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </React.StrictMode>,
  )
}
