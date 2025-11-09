// @ts-nocheck
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './styles.css';

const queryClient = new QueryClient();

const rootNode = document.getElementById('ml-workflow-root');

if (!rootNode) {
  throw new Error('ML workflow root element not found');
}

const root = ReactDOM.createRoot(rootNode);

const sourceId = rootNode.getAttribute('data-source-id') || undefined;

root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App sourceId={sourceId} />
    </QueryClientProvider>
  </React.StrictMode>
);
