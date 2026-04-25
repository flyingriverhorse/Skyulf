import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import App from './App.tsx'
import './index.css'
import { initializeRegistry } from './core/registry/init'
import { ErrorBoundary, ConfirmProvider } from './components/shared'

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
            <Toaster richColors closeButton position="top-right" />
          </ConfirmProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </React.StrictMode>,
  )
}
